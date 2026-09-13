"""
PermRankKVCompactor: Multi-Head Borda Consensus Rank Compactor for KV-Cache.
Protected under U.S. Patent Application Nos. 64/148,668 & 64/152,256.
Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)

Eliminates cross-head floating-point score calibration in long-context LLMs.
Uses discrete permutation group Borda consensus across attention heads
to determine invariant salient KV-cache tokens with zero score distortion.
"""

from typing import Dict, List, Tuple, Optional, Union
import torch
import numpy as np


def compute_multihead_borda_ranks(head_scores: torch.Tensor) -> torch.Tensor:
    """
    Computes closed-form Borda consensus ranking across H attention heads.
    Args:
        head_scores: Tensor of shape [..., H, N] containing raw head attention scores.
    Returns:
        consensus_ranks: Tensor of shape [..., N] containing unified consensus rank (0 is highest).
    """
    # Convert each head's scores to discrete ordinal ranks in S_N
    # High score = rank 0 (most important)
    sorted_indices = torch.argsort(head_scores, dim=-1, descending=True)
    ranks = torch.argsort(sorted_indices, dim=-1).to(torch.float32)

    # Borda consensus is the mean ordinal rank across all heads
    mean_ranks = torch.mean(ranks, dim=-2)
    # Final consensus ranking
    consensus = torch.argsort(torch.argsort(mean_ranks, dim=-1), dim=-1)
    return consensus


class PermRankKVCompactor:
    """
    Multi-Head Permutation Rank KV-Cache Compactor.
    Works seamlessly on CPU, CUDA, and MPS without Triton requirements.
    """

    def __init__(self, capacity: int = 128):
        self.capacity = capacity

    def build_idempotent_permutation_map(
        self,
        N: int,
        active_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Builds the 2-cycle transposition map f: [0, N-1] -> [0, N-1]
        such that active_indices are mapped to [0, capacity - 1] and f(f(x)) == x.
        """
        target_map = torch.arange(N, dtype=torch.int64, device=device)
        
        K = min(self.capacity, len(active_indices))
        top_k = active_indices[:K]

        active_tail = top_k[top_k >= K]
        num_swaps = active_tail.numel()

        if num_swaps > 0:
            is_active_head = torch.zeros(K, dtype=torch.bool, device=device)
            head_active = top_k[top_k < K]
            is_active_head[head_active] = True
            vacant_head = torch.nonzero(~is_active_head, as_tuple=True)[0][:num_swaps]

            target_map[vacant_head] = active_tail
            target_map[active_tail] = vacant_head

        return target_map

    def compact_kv_tensors(
        self,
        K: torch.Tensor,  # [B, H, N, D]
        V: torch.Tensor,  # [B, H, N, D]
        head_scores: torch.Tensor,  # [B, H, N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compacts Key and Value tensors in-place or via permutation gather.
        Returns (compacted_K, compacted_V, target_map) where length dimension is truncated to capacity.
        """
        B, H, N, D = K.shape
        device = K.device
        
        # Borda consensus rank across heads: shape [B, N]
        consensus_ranks = compute_multihead_borda_ranks(head_scores)

        target_maps = []
        for b in range(B):
            active_b = torch.argsort(consensus_ranks[b])[:self.capacity]
            t_map = self.build_idempotent_permutation_map(N, active_b, device)
            target_maps.append(t_map)

        full_map = torch.stack(target_maps, dim=0)  # [B, N]

        # Apply permutation gather
        gather_idx = full_map.unsqueeze(1).unsqueeze(-1).expand(B, H, N, D)
        K_perm = torch.gather(K, 2, gather_idx)[:, :, :self.capacity, :]
        V_perm = torch.gather(V, 2, gather_idx)[:, :, :self.capacity, :]

        return K_perm, V_perm, full_map

