import torch
import tilelang

from tile_kernels.engram.engram_fused_weight_kernel import get_engram_fused_weight_kernel


# Original TileKernels source: tile_kernels/engram/engram_fused_weight_kernel.py
def test_engram_fused_weight_kernel_runtime_compare():
    torch.manual_seed(0)
    weight_hidden = torch.randn(4, 256, dtype=torch.bfloat16)
    weight_embed = torch.randn(4, 256, dtype=torch.bfloat16)
    expected = weight_hidden.float() * weight_embed.float()

    kernel = tilelang.compile(
        get_engram_fused_weight_kernel.get_tir(256, 4),
        out_idx=[2],
        target="riscv",
    )
    try:
        actual = kernel(weight_hidden, weight_embed)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
