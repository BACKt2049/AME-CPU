import torch
import tilelang

from tile_kernels.moe.topk_sum_and_topk_group_idx_kernel import get_topk_sum_and_topk_group_idx_kernel
from tile_kernels.torch.topk import topk_sum_and_topk_group_idx as topk_sum_and_topk_group_idx_ref


# Original TileKernels source: tile_kernels/moe/topk_sum_and_topk_group_idx_kernel.py
def test_topk_sum_and_topk_group_idx_kernel_runtime_compare():
    scores = torch.tensor(
        [
            [
                [9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [8.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [7.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [6.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    expected = topk_sum_and_topk_group_idx_ref(scores, 2, 2)

    kernel = tilelang.compile(
        get_topk_sum_and_topk_group_idx_kernel.get_tir(4, 8, 2, 2),
        out_idx=[1],
        target="riscv",
    )
    try:
        actual = kernel(scores.view(scores.shape[0], -1))
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
