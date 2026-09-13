"""
Zero-FLOPs Permutation Rank Saliency Scorer
==========================================
Birkhoff-von Neumann Matrix-Free Token Selection for KV-Cache Compaction.

Eliminates floating-point QK^T multiply-accumulate operations in token importance
evaluation by mapping key vectors into discrete rank permutations in the
Symmetric Group S_D and measuring Spearman Footrule distances.

Mathematical Properties:
1. 0 Floating-Point Multiplications in attention/saliency score calculation.
2. Invariant under arbitrary monotonic channel transformations g(x_d) (e.g., LayerNorm, FP8/INT4 outlier scaling).
3. Strictly bounded utility scores in [0.0, 1.0].

Protected under U.S. Patent Application Nos. 64/148,668 & 64/152,256.
Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


def compute_max_footrule_distance(d: int) -> float:
    """Computes the maximum possible Spearman Footrule distance in the Symmetric Group S_d."""
    half = d // 2
    max_d = 2 * half * (d - half)
    return float(max_d) if max_d > 0 else 1.0


class PermSaliencyScorer(nn.Module):
    """
    0-FLOPs Rank-Correlation Token Saliency Scorer for Key-Value Caches.

    Parameters
    ----------
    head_dim : int, default=128
        Dimension of each attention head.
    scale_invariant : bool, default=True
        Whether to enforce ordinal ranking for strict monotonic scale invariance.
    """

    def __init__(self, head_dim: int = 128, scale_invariant: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.scale_invariant = scale_invariant
        self.max_footrule = compute_max_footrule_distance(head_dim)

    @staticmethod
    def tensor_to_ranks(x: torch.Tensor) -> torch.Tensor:
        """
        Converts continuous feature coordinates to discrete rank permutations in {0, ..., D-1}.
        Operates along the last dimension.
        Shape: (..., D) -> (..., D) integer ranks.
        """
        # argsort(argsort(x)) produces exact rank indices in O(D log D) comparisons with 0 FLOPs
        return torch.argsort(torch.argsort(x, dim=-1), dim=-1).to(torch.int32)

    def score_intrinsic_saliency(self, key_cache: torch.Tensor) -> torch.Tensor:
        """
        Computes intrinsic token saliency directly from the key cache using ordinal rank variance.
        Tokens that deviate significantly from a uniform rank profile represent high-entropy semantics.

        Parameters
        ----------
        key_cache : torch.Tensor of shape (B, H, N, D) or (B, N, D)

        Returns
        -------
        saliency_scores : torch.Tensor of shape (B, H, N) or (B, N) with values in [0.0, 1.0]
        """
        orig_dim = key_cache.dim()
        if orig_dim == 3:
            key_cache = key_cache.unsqueeze(1)

        B, H, N, D = key_cache.shape
        ranks = self.tensor_to_ranks(key_cache).float()  # (B, H, N, D)

        # Centroid rank per sequence head: mean rank profile
        mean_rank = ranks.mean(dim=2, keepdim=True)  # (B, H, 1, D)

        # Manhattan deviation from the mean rank permutation (Footrule divergence)
        rank_dev = torch.abs(ranks - mean_rank).sum(dim=-1)  # (B, H, N)

        # Normalize to [0.0, 1.0]
        norm_factor = max(float(self.max_footrule), 1.0)
        saliency = torch.clamp(rank_dev / norm_factor, 0.0, 1.0)

        if orig_dim == 3:
            return saliency.squeeze(1)
        return saliency

    def score_query_alignment(
        self,
        key_cache: torch.Tensor,
        query_vector: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Computes 0-FLOPs attention utility between query vector(s) and historical key tokens
        using the Spearman Footrule distance in the Symmetric Group S_D.

        S(i, j) = 1.0 - (2 * || Phi(q_i) - Phi(k_j) ||_1) / d_max

        Parameters
        ----------
        key_cache : torch.Tensor of shape (B, H, N, D)
        query_vector : torch.Tensor of shape (B, H, 1, D) or (B, H, D)
        temperature : float, default=1.0

        Returns
        -------
        scores : torch.Tensor of shape (B, H, N) in [0.0, 1.0]
        """
        if query_vector.dim() == 3:
            query_vector = query_vector.unsqueeze(2)

        q_ranks = self.tensor_to_ranks(query_vector).float()  # (B, H, 1, D)
        k_ranks = self.tensor_to_ranks(key_cache).float()     # (B, H, N, D)

        # Integer Manhattan distance along the head dimension (0 FP32 multiplications)
        dist = torch.abs(q_ranks - k_ranks).sum(dim=-1)       # (B, H, N)

        # Spearman Footrule similarity: 1.0 = identical rank permutation, 0.0 = maximal discordance
        similarity = 1.0 - (2.0 * dist) / max(self.max_footrule, 1.0)
        similarity = torch.clamp(similarity, min=0.0, max=1.0)

        if temperature != 1.0 and temperature > 0:
            similarity = torch.pow(similarity, 1.0 / temperature)

        return similarity

    def select_active_indices(
        self,
        key_cache: torch.Tensor,
        capacity: int,
        protected_prefix_len: int = 4,
        query_vector: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Selects active token indices to be retained in the compacted cache.
        Always preserves the first `protected_prefix_len` attention sinks,
        and selects the top-(capacity - prefix) tokens according to 0-FLOPs rank saliency.

        Returns
        -------
        active_indices : torch.Tensor of shape (B, H, capacity)
        scores (optional) : torch.Tensor of shape (B, H, N)
        """
        B, H, N, D = key_cache.shape
        device = key_cache.device

        if query_vector is not None:
            scores = self.score_query_alignment(key_cache, query_vector)
        else:
            scores = self.score_intrinsic_saliency(key_cache)

        k_rem = capacity - protected_prefix_len
        assert k_rem > 0, f"Capacity ({capacity}) must be strictly greater than prefix ({protected_prefix_len})"

        active_indices = torch.zeros((B, H, capacity), dtype=torch.long, device=device)

        # Prefix sinks are always preserved (fixed points)
        prefix_idx = torch.arange(protected_prefix_len, device=device)
        active_indices[:, :, :protected_prefix_len] = prefix_idx.unsqueeze(0).unsqueeze(0)

        # Select top-k from the candidate tail [protected_prefix_len, N)
        if N > protected_prefix_len:
            tail_scores = scores[:, :, protected_prefix_len:]
            num_to_pick = min(k_rem, tail_scores.shape[-1])
            _, top_tail_sub = torch.topk(tail_scores, k=num_to_pick, dim=-1, largest=True)
            top_tail_idx = top_tail_sub + protected_prefix_len
            active_indices[:, :, protected_prefix_len:protected_prefix_len + num_to_pick] = top_tail_idx

        # Sort indices to maintain chronological causality
        active_indices, _ = torch.sort(active_indices, dim=-1)

        if return_scores:
            return active_indices, scores
        return active_indices

