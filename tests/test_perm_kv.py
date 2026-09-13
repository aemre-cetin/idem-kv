"""
Unit tests for PermRankKVCompactor in idempotent-kv.
Protected under U.S. Patent Application Nos. 64/148,668 & 64/152,256.
Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from idempotent_kv import PermRankKVCompactor, compute_multihead_borda_ranks


class TestPermRankKV(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.H = 4
        self.N = 64
        self.D = 32
        self.capacity = 16

    def test_multihead_borda_consensus(self):
        # 4 heads with slightly noisy opinions on 10 tokens
        head_scores = torch.randn((self.H, 10))
        # Token 3 is forced to be uniformly top-1 across all heads
        head_scores[:, 3] = 100.0

        consensus = compute_multihead_borda_ranks(head_scores)
        # Token 3 must be rank 0
        self.assertEqual(consensus[3].item(), 0)

    def test_perm_kv_compactor_and_involution(self):
        compactor = PermRankKVCompactor(capacity=self.capacity)

        K = torch.randn((self.B, self.H, self.N, self.D))
        V = torch.randn((self.B, self.H, self.N, self.D))
        head_scores = torch.randn((self.B, self.H, self.N))

        K_comp, V_comp, target_map = compactor.compact_kv_tensors(K, V, head_scores)

        # Dimension checks
        self.assertEqual(K_comp.shape, (self.B, self.H, self.capacity, self.D))
        self.assertEqual(V_comp.shape, (self.B, self.H, self.capacity, self.D))

        # Check involution property on each batch map f(f(i)) == i
        for b in range(self.B):
            m = target_map[b]
            re_m = m[m]
            torch.testing.assert_close(re_m, torch.arange(self.N))


if __name__ == "__main__":
    unittest.main()

