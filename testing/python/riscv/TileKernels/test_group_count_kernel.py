import torch
import tilelang

from tile_kernels.moe.group_count_kernel import get_group_count_kernel
from tile_kernels.torch.moe import group_count as group_count_ref


# Original TileKernels source: tile_kernels/moe/group_count_kernel.py
def test_group_count_kernel_runtime_compare():
    group_idx = torch.tensor(
        [
            [0, 1, -1],
            [2, 2, 1],
            [1, -1, -1],
        ],
        dtype=torch.int64,
    )
    actual = torch.zeros(4, dtype=torch.int32)
    expected = group_count_ref(group_idx, 4)

    kernel = tilelang.compile(
        get_group_count_kernel.get_tir(3, 4, 1),
        out_idx=[],
        target="riscv",
    )
    try:
        kernel(group_idx, actual)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
