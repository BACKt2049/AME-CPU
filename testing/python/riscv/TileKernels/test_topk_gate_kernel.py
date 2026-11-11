import torch
import tilelang

from tile_kernels.moe.topk_gate_kernel import get_topk_gate_kernel
from tile_kernels.torch.topk import stable_topk


# Original TileKernels source: tile_kernels/moe/topk_gate_kernel.py
def test_topk_gate_kernel_runtime_compare():
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            [0.5, 1.5, 0.4, 1.4, 0.3, 1.3, 0.2, 1.2],
            [1.0, 0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            [2.0, 1.0, 1.9, 1.1, 1.8, 1.2, 1.7, 1.3],
            [3.0, 2.0, 2.9, 2.1, 2.8, 2.2, 2.7, 2.3],
        ],
        dtype=torch.float32,
    )
    expected = stable_topk(scores, 2)

    kernel = tilelang.compile(
        get_topk_gate_kernel.get_tir(8, 2),
        out_idx=[1],
        target="riscv",
    )
    try:
        actual = kernel(scores)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
