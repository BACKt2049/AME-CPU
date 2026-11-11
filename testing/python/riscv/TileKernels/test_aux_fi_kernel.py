import torch
import tilelang

from tile_kernels.moe.aux_fi_kernel import get_aux_fi_kernel
from tile_kernels.torch.moe import aux_fi as aux_fi_ref


# Original TileKernels source: tile_kernels/moe/aux_fi_kernel.py
def test_aux_fi_kernel_runtime_compare():
    topk_idx = torch.tensor(
        [
            [0, 1, -1],
            [2, 2, 1],
            [1, -1, -1],
        ],
        dtype=torch.int64,
    )
    actual = torch.zeros(4, dtype=torch.float32)
    expected = aux_fi_ref(topk_idx, 4, 2)

    kernel = tilelang.compile(
        get_aux_fi_kernel.get_tir(3, 4, 1),
        out_idx=[],
        target="riscv",
    )
    try:
        kernel(topk_idx, actual, 2)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
