import torch
import tilelang

from tile_kernels.engram.engram_hash_kernel import get_engram_hash_kernel
from tile_kernels.torch.engram import engram_hash_ref, make_offsets


# Original TileKernels source: tile_kernels/engram/engram_hash_kernel.py
def test_engram_hash_kernel_runtime_compare():
    torch.manual_seed(2)
    ngram_token_ids = torch.randint(0, 1000, (7, 3), dtype=torch.int32)
    multipliers = torch.randint(1, 1000, (2, 3), dtype=torch.int64)
    vocab_sizes = torch.randint(100, 200, (2, 2, 8), dtype=torch.int32)
    offsets = make_offsets(vocab_sizes)
    expected = engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets)

    kernel = tilelang.compile(
        get_engram_hash_kernel.get_tir(3, 2, 8),
        out_idx=[4],
        target="riscv",
    )
    try:
        actual = kernel(ngram_token_ids, multipliers, vocab_sizes, offsets)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
