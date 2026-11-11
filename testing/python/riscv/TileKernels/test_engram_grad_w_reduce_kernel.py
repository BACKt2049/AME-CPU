import torch
import tilelang

from tile_kernels.engram.engram_grad_w_reduce_kernel import get_engram_grad_w_reduce_kernel


# Original TileKernels source: tile_kernels/engram/engram_grad_w_reduce_kernel.py
def test_engram_grad_w_reduce_kernel_runtime_compare():
    torch.manual_seed(1)
    grad_w_partial = torch.randn(4, 4, 512, dtype=torch.float32)
    weight_hidden = torch.randn(4, 512, dtype=torch.bfloat16)
    weight_embed = torch.randn(4, 512, dtype=torch.bfloat16)
    grad_weight_hidden = torch.randn(4, 512, dtype=torch.float32)
    grad_weight_embed = torch.randn(4, 512, dtype=torch.float32)

    expected_grad_weight_hidden = grad_weight_hidden.clone() + grad_w_partial.sum(dim=0) * weight_embed.float()
    expected_grad_weight_embed = grad_weight_embed.clone() + grad_w_partial.sum(dim=0) * weight_hidden.float()

    kernel = tilelang.compile(
        get_engram_grad_w_reduce_kernel.get_tir(512, 4, 4),
        out_idx=[],
        target="riscv",
    )
    try:
        kernel(
            grad_w_partial,
            weight_hidden,
            weight_embed,
            grad_weight_hidden,
            grad_weight_embed,
        )
    finally:
        kernel.close()

    torch.testing.assert_close(grad_weight_hidden, expected_grad_weight_hidden)
    torch.testing.assert_close(grad_weight_embed, expected_grad_weight_embed)
