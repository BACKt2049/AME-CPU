import torch
import tilelang

from tile_kernels.moe.inplace_unique_group_indices_kernel import get_inplace_unique_group_indices_kernel
from tile_kernels.torch.moe import inplace_unique_group_indices as inplace_unique_group_indices_ref


# Original TileKernels source: tile_kernels/moe/inplace_unique_group_indices_kernel.py
def test_inplace_unique_group_indices_kernel_runtime_compare():
    actual = torch.tensor([[1, 2, 1, 3], [4, 4, 2, -1]], dtype=torch.int64)
    expected = actual.clone()
    inplace_unique_group_indices_ref(expected, 8)

    kernel = tilelang.compile(
        get_inplace_unique_group_indices_kernel.get_tir(4, 64, 1),
        out_idx=[],
        target="riscv",
    )
    try:
        kernel(actual)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
