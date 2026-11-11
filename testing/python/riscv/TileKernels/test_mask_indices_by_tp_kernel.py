import torch
import tilelang
import tilelang.language as T

from tile_kernels.moe.mask_indices_by_tp_kernel import get_mask_indices_by_tp_kernel
from tile_kernels.torch.moe import mask_indices_by_tp as mask_indices_by_tp_ref


# Original TileKernels source: tile_kernels/moe/mask_indices_by_tp_kernel.py
def test_mask_indices_by_tp_kernel_runtime_compare():
    indices = torch.tensor([[0, 5, 8, -1], [3, 7, 12, 15]], dtype=torch.int64)
    per_gpu = 8 // 2
    per_dp = 2 * per_gpu
    expected = mask_indices_by_tp_ref(indices, 8, 2, 1, 2)

    kernel = tilelang.compile(
        get_mask_indices_by_tp_kernel.get_tir(4, T.int64),
        out_idx=[1],
        target="riscv",
    )
    try:
        actual = kernel(indices, per_gpu, per_dp, 2, 1)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
