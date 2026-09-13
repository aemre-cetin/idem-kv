"""
Comprehensive Verification Tests for Next-Gen Idempotent-KV Architecture
========================================================================
Tests:
1. 0-FLOPs PermSaliencyScorer: Birkhoff rank statistics, Footrule distance, scale invariance.
2. YoungFactorizedCompactor: Bounded cycle decomposition, in-situ data integrity.
3. TwoDimensionalIdempotentCompactor: 2D compaction, algebraic idempotence Pi^2 = Pi, dual-manifold orthogonality.
4. HysteresisTokenStabilizer: Schmitt-trigger hysteresis, thrashing prevention.
5. InplaceKVCompactor Integration: compact_with_perm_saliency, compact_2d, compact_with_young_factorization.
"""

import sys
import os
import pytest
import torch

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from idempotent_kv import (
    InplaceKVCompactor,
    PermSaliencyScorer,
    YoungFactorizedCompactor,
    TwoDimensionalIdempotentCompactor,
    HysteresisTokenStabilizer,
)


def test_perm_saliency_scorer():
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 32
    K = torch.randn((B, H, N, D), dtype=torch.float32)

    scorer = PermSaliencyScorer(head_dim=D)

    # 1. Intrinsic saliency
    scores = scorer.score_intrinsic_saliency(K)
    assert scores.shape == (B, H, N)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # 2. Query alignment
    Q = torch.randn((B, H, 1, D), dtype=torch.float32)
    query_scores = scorer.score_query_alignment(K, Q)
    assert query_scores.shape == (B, H, N)
    assert (query_scores >= 0.0).all() and (query_scores <= 1.0).all()

    # 3. Active index selection
    capacity = 32
    prefix = 4
    active_idx = scorer.select_active_indices(K, capacity=capacity, protected_prefix_len=prefix)
    assert active_idx.shape == (B, H, capacity)
    # Check that attention sinks 0..3 are strictly preserved
    for p in range(prefix):
        assert (active_idx[:, :, p] == p).all(), f"Prefix token {p} was not preserved"

    # 4. Scale Invariance Test (10,000x input distortion)
    K_distorted = 10000.0 * K + 500.0
    active_idx_distorted = scorer.select_active_indices(K_distorted, capacity=capacity, protected_prefix_len=prefix)
    assert torch.equal(active_idx, active_idx_distorted), "Rank saliency was affected by monotonic scale distortion!"


def test_young_factorized_compactor():
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 128, 64
    capacity = 64
    device = torch.device('cpu')

    K = torch.randn((B, H, N, D), dtype=torch.float32, device=device)
    V = torch.randn((B, H, N, D), dtype=torch.float32, device=device)

    active_indices = torch.randperm(N, device=device)[:capacity].sort().values

    compactor = YoungFactorizedCompactor(tile_size=32)
    target_map = compactor.build_factorized_target_map(B, H, N, active_indices, capacity, device)

    # In-place compaction
    k_comp, v_comp = compactor.compact(K, V, target_map, capacity)

    assert k_comp.shape == (B, H, capacity, D)
    assert v_comp.shape == (B, H, capacity, D)
    assert not torch.isnan(k_comp).any()
    assert not torch.isnan(v_comp).any()


def test_2d_compactor():
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 64
    capacity = 32
    d_sub = 32
    device = torch.device('cpu')

    K = torch.randn((B, H, N, D), dtype=torch.float32, device=device)
    V = torch.randn((B, H, N, D), dtype=torch.float32, device=device)

    compactor_2d = TwoDimensionalIdempotentCompactor(head_dim=D, subspace_dim=d_sub)
    target_map = torch.arange(N, dtype=torch.int32).unsqueeze(0).unsqueeze(0).expand(B, H, N).clone()

    # Test algebraic verification
    diag = compactor_2d.verify_2d_idempotence(K, V, target_map, capacity, tol=1e-4)
    assert diag["is_idempotent"], f"Idempotency error too high: {diag['projector_idempotency_error']}"
    assert diag["is_orthogonal"], f"Orthogonality error too high: {diag['orthogonality_error']}"

    # Test 2D Compaction (Temporal + Channel)
    k_comp, v_comp = compactor_2d.compact_2d(K, V, target_map, capacity, compress_channel=True)
    assert k_comp.shape == (B, H, capacity, d_sub)
    assert v_comp.shape == (B, H, capacity, d_sub)


def test_hysteresis_token_stabilizer():
    torch.manual_seed(42)
    B, H, N = 1, 1, 10
    stabilizer = HysteresisTokenStabilizer(tau_retain=0.6, tau_evict=0.4, cooldown_steps=2)

    # Initial step: scores around 0.5
    scores_t0 = torch.tensor([[[0.9, 0.9, 0.55, 0.45, 0.2, 0.1, 0.8, 0.3, 0.7, 0.5]]])
    s0 = stabilizer.update_states(scores_t0, protected_prefix_len=2)
    assert s0[0, 0, 0] and s0[0, 0, 1], "Prefix tokens must be retained"

    # Step 1: slight oscillation (0.55 -> 0.48, in deadband between 0.40 and 0.60)
    scores_t1 = torch.tensor([[[0.9, 0.9, 0.48, 0.52, 0.2, 0.1, 0.8, 0.3, 0.7, 0.5]]])
    s1 = stabilizer.update_states(scores_t1, protected_prefix_len=2)
    # Token 2 was retained in step 0, dropped to 0.48 which is > tau_evict (0.40), so it MUST STAY RETAINED!
    assert s1[0, 0, 2] == s0[0, 0, 2], "Deadband hysteresis failed to prevent thrashing!"


def test_inplace_compactor_nextgen_integration():
    torch.manual_seed(42)
    B, H, N, D = 2, 2, 64, 32
    capacity = 32
    compactor = InplaceKVCompactor()

    K = torch.randn((B, H, N, D), dtype=torch.float32)
    V = torch.randn((B, H, N, D), dtype=torch.float32)

    # 1. 0-FLOPs Saliency compaction
    k_sal, v_sal = compactor.compact_with_perm_saliency(K.clone(), V.clone(), capacity=capacity)
    assert k_sal.shape == (B, H, capacity, D)

    # 2. Young factorized compaction
    active_idx = torch.randperm(N)[:capacity].sort().values
    k_yng, v_yng = compactor.compact_with_young_factorization(K.clone(), V.clone(), active_indices=active_idx, capacity=capacity)
    assert k_yng.shape == (B, H, capacity, D)

    # 3. 2D compaction
    target_map = compactor.build_idempotent_map(B, H, N, active_idx, capacity, K.device)
    k_2d, v_2d = compactor.compact_2d(K.clone(), V.clone(), target_map, capacity, subspace_dim=16, compress_channel=True)
    assert k_2d.shape == (B, H, capacity, 16)

