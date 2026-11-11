import torch
import tilelang
import tilelang.language as T

from tile_kernels.moe.expand_to_fused_kernel import get_expand_to_fused_kernel
from tile_kernels.torch.expand_to_fused import expand_to_fused as expand_to_fused_ref


# Original TileKernels source: tile_kernels/moe/expand_to_fused_kernel.py
def test_expand_to_fused_kernel_runtime_compare():
    x = (torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64) / 10).contiguous()
    token_topk_to_pos = torch.tensor([[0], [1], [2]], dtype=torch.int32)
    pos_to_expert = torch.tensor([0, 1, 2], dtype=torch.int32)

    expected = expand_to_fused_ref(x, token_topk_to_pos, pos_to_expert)
    x_sf = torch.empty((x.shape[0], 1), dtype=torch.float32)
    out_sf = torch.empty((pos_to_expert.shape[0], 1), dtype=torch.float32)

    kernel = tilelang.compile(
        get_expand_to_fused_kernel.get_tir(64, 1, None, None, None, T.float32, T.float32),
        out_idx=[2],
        target="riscv",
    )
    try:
        actual = kernel(x, x_sf, out_sf, token_topk_to_pos, pos_to_expert)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
