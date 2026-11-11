import torch
import tilelang

from tile_kernels.moe.normalize_weight_kernel import get_normalize_weight_kernel
from tile_kernels.torch.moe import normalize_weight as normalize_weight_ref


# Original TileKernels source: tile_kernels/moe/normalize_weight_kernel.py
def test_normalize_weight_kernel_runtime_compare():
    topk_weights = torch.tensor(
        [
            [0.2, 0.3, 0.4, 0.5],
            [0.7, 0.1, 0.2, 0.3],
            [0.8, 0.6, 0.4, 0.2],
            [1.0, 0.5, 0.25, 0.125],
            [0.9, 0.8, 0.7, 0.6],
        ],
        dtype=torch.float32,
    )
    expected_denominator, expected_normalized_weights = normalize_weight_ref(topk_weights)

    kernel = tilelang.compile(
        get_normalize_weight_kernel.get_tir(4),
        out_idx=[1, 2],
        target="riscv",
    )
    try:
        actual_denominator, actual_normalized_weights = kernel(topk_weights)
    finally:
        kernel.close()

    torch.testing.assert_close(actual_denominator, expected_denominator)
    torch.testing.assert_close(actual_normalized_weights, expected_normalized_weights)
