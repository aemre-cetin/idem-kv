"""
Schmitt-Trigger Hysteresis Token Stabilizer
==========================================
Eliminates Token Thrashing & Boundary Oscillation in Dynamic KV Pruning.

In multi-step autoregressive generation, heuristic top-K context pruning suffers
from boundary flickering: marginal tokens near the cutoff oscillate between
eviction and retention across consecutive decoding steps, inducing cache thrashing.

The Schmitt-Trigger Hysteresis Stabilizer enforces dual-threshold state attraction:
    - Upgrade to Retained:  Score >= tau_retain
    - Downgrade to Evicted: Score <= tau_evict  (with tau_evict < tau_retain)
    - Deadband Zone:        Maintain current state (zero data relocation)

Protected under U.S. Patent Application Nos. 64/148,668 & 64/152,256.
Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn


class HysteresisTokenStabilizer(nn.Module):
    """
    Schmitt-Trigger Hysteresis Filter for Stable Autoregressive Token Eviction.

    Parameters
    ----------
    tau_retain : float, default=0.55
        Upper score threshold required to promote an evicted token to retained.
    tau_evict : float, default=0.45
        Lower score threshold required to demote a retained token to evicted.
    cooldown_steps : int, default=4
        Minimum generation steps a token must remain in its state before transitioning.
    """

    def __init__(
        self,
        tau_retain: float = 0.55,
        tau_evict: float = 0.45,
        cooldown_steps: int = 4
    ):
        super().__init__()
        assert tau_evict < tau_retain, f"tau_evict ({tau_evict}) must be < tau_retain ({tau_retain})"
        self.tau_retain = float(tau_retain)
        self.tau_evict = float(tau_evict)
        self.cooldown_steps = int(cooldown_steps)

        # Internal state tracking
        self.state: Optional[torch.Tensor] = None        # True = Retained, False = Evicted
        self.cooldown: Optional[torch.Tensor] = None     # Remaining lock steps

    def reset(self):
        """Resets the internal hysteresis filter state."""
        self.state = None
        self.cooldown = None

    def update_states(
        self,
        scores: torch.Tensor,
        protected_prefix_len: int = 4,
        target_capacity: Optional[int] = None
    ) -> torch.Tensor:
        """
        Updates binary retention states under Schmitt-trigger hysteresis.

        Parameters
        ----------
        scores : torch.Tensor of shape (B, H, N) or (B, N)
            Current token importance scores in [0.0, 1.0].
        protected_prefix_len : int, default=4
            Initial prompt tokens always pinned as True (fixed points).
        target_capacity : Optional[int]
            Optional hard budget to constrain active tokens if retention exceeds target.

        Returns
        -------
        is_retained : torch.Tensor of same shape as scores (bool)
        """
        device = scores.device
        shape = scores.shape

        if self.state is None or self.state.shape != shape:
            # Initialize with standard threshold
            self.state = (scores >= ((self.tau_retain + self.tau_evict) / 2.0)).clone()
            self.cooldown = torch.zeros(shape, dtype=torch.int32, device=device)

        # Decrement cooldown
        self.cooldown = torch.clamp(self.cooldown - 1, min=0)

        # Eligible for state change when cooldown == 0
        can_change = (self.cooldown == 0)

        # 1. Low-to-High Promotion: (Evicted & score >= tau_retain) -> Retained
        promote = can_change & (~self.state) & (scores >= self.tau_retain)
        self.state[promote] = True
        self.cooldown[promote] = self.cooldown_steps

        # 2. High-to-Low Demotion: (Retained & score <= tau_evict) -> Evicted
        demote = can_change & self.state & (scores <= self.tau_evict)
        self.state[demote] = False
        self.cooldown[demote] = self.cooldown_steps

        # Always protect attention sink prefix
        if scores.dim() == 3:
            self.state[:, :, :protected_prefix_len] = True
        elif scores.dim() == 2:
            self.state[:, :protected_prefix_len] = True

        # Optional budget cap enforcement
        if target_capacity is not None:
            active_counts = self.state.sum(dim=-1)
            if (active_counts > target_capacity).any():
                # Demote lowest scores among eligible non-prefix tokens
                for idx in torch.nonzero(active_counts > target_capacity):
                    b = idx[0].item()
                    h = idx[1].item() if scores.dim() == 3 else None
                    cur_state = self.state[b, h] if h is not None else self.state[b]
                    cur_scores = scores[b, h] if h is not None else scores[b]
                    
                    active_indices = torch.nonzero(cur_state).squeeze(-1)
                    tail_active = active_indices[active_indices >= protected_prefix_len]
                    
                    excess = len(active_indices) - target_capacity
                    if excess > 0 and len(tail_active) > 0:
                        tail_scores = cur_scores[tail_active]
                        lowest_tail_idx = tail_active[torch.topk(tail_scores, k=min(excess, len(tail_active)), largest=False).indices]
                        if h is not None:
                            self.state[b, h, lowest_tail_idx] = False
                        else:
                            self.state[b, lowest_tail_idx] = False

        return self.state.clone()

