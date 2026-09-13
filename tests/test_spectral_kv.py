import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import torch
from idempotent_kv.saliency import (
    ChebyshevSpectralSaliencyScorer,
    HybridPermSpectralScorer,
)


def test_chebyshev_spectral_saliency_scorer():
    scorer = ChebyshevSpectralSaliencyScorer(degree=3)
    B, H, N, D = 2, 4, 32, 64
    key_cache = torch.randn(B, H, N, D)

    # Inject high-frequency non-linear pulse into token 10
    key_cache[:, :, 10, :] += 5.0 * torch.sin(torch.linspace(0, 10, D))

    scores = scorer.score_spectral_energy(key_cache)
    assert scores.shape == (B, H, N)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # Token 10 should have higher harmonic energy than a random flat token
    mean_score_token_10 = scores[:, :, 10].mean().item()
    mean_score_random = scores[:, :, 5].mean().item()
    assert mean_score_token_10 >= mean_score_random


def test_hybrid_perm_spectral_scorer():
    head_dim = 64
    capacity = 16
    prefix_len = 4
    scorer = HybridPermSpectralScorer(head_dim=head_dim, degree=3, alpha=0.5)

    B, H, N, D = 2, 2, 64, head_dim
    key_cache = torch.randn(B, H, N, D)

    # 1. Compute hybrid score without query
    scores = scorer.score(key_cache)
    assert scores.shape == (B, H, N)

    # 2. Compute hybrid score with query alignment
    query = torch.randn(B, H, 1, D)
    q_scores = scorer.score(key_cache, query_vector=query)
    assert q_scores.shape == (B, H, N)

    # 3. Select active indices
    active_idx = scorer.select_active_indices(
        key_cache, capacity=capacity, protected_prefix_len=prefix_len, query_vector=query
    )
    assert active_idx.shape == (B, H, capacity)

    # Check prefix preservation: indices 0, 1, 2, 3 must be present in every head
    for p in range(prefix_len):
        assert (active_idx[:, :, p] == p).all()

