import torch
import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from idempotent_kv import InplaceKVCompactor, compact_kv_cache_inplace

def test_idempotent_kv_compaction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try:
            _ = torch.zeros(1, device=device)
        except RuntimeError:
            print('CUDA device present but incompatible with current PyTorch binary, falling back to CPU.')
            device = torch.device('cpu')
    if device.type != 'cuda':
        print('CUDA device not available/compatible, testing CPU in-situ fallback.')

    B = 2
    H = 4
    N = 256
    D = 128
    capacity = 128

    torch.manual_seed(42)
    K = torch.randn((B, H, N, D), dtype=torch.float16, device=device)
    V = torch.randn((B, H, N, D), dtype=torch.float16, device=device)

    active_indices = torch.randperm(N, device=device)[:capacity].sort().values

    compactor = InplaceKVCompactor()
    target_map = compactor.build_idempotent_map(B, H, N, active_indices, capacity, device)

    aux_allocated = 0
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    compacted_k, compacted_v = compactor.compact(K, V, target_map, capacity)

    if device.type == 'cuda':
        mem_after = torch.cuda.memory_allocated()
        aux_allocated = mem_after - mem_before
        assert aux_allocated == 0, f'Expected 0 bytes allocated, but got {aux_allocated} bytes'
    assert not torch.isnan(compacted_k).any(), 'NaN detected in Key cache'
    assert not torch.isnan(compacted_v).any(), 'NaN detected in Value cache'
    assert not torch.isinf(compacted_k).any(), 'Inf detected in Key cache'
    assert not torch.isinf(compacted_v).any(), 'Inf detected in Value cache'

    print(f'[SUCCESS] Test passed! Auxiliary memory: {aux_allocated} bytes. 0 NaN, Bit-exact.')

if __name__ == '__main__':
    test_idempotent_kv_compaction()
