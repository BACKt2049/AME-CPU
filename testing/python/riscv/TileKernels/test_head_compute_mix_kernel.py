import torch
import tilelang

from tile_kernels.mhc.head_compute_mix_kernel import _mhc_head_compute_mix_fwd
from tile_kernels.torch.mhc import mhc_head_compute_mix_ref


# Original TileKernels source: tile_kernels/mhc/head_compute_mix_kernel.py
def test_head_compute_mix_kernel_runtime_compare():
    input_mix = torch.tensor(
        [
            [0.0, 0.5, -1.0, 2.0],
            [1.0, -0.5, 0.25, -2.0],
            [2.0, 0.0, 0.75, -1.5],
        ],
        dtype=torch.float32,
    ).contiguous()
    mhc_scale = torch.tensor([1.25], dtype=torch.float32)
    mhc_base = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32)
    expected = mhc_head_compute_mix_ref(input_mix, mhc_scale, mhc_base, 1e-6)

    kernel = tilelang.compile(
        _mhc_head_compute_mix_fwd.get_tir(4, 1e-6, 2),
        out_idx=[3],
        target="riscv",
    )
    try:
        actual = kernel(input_mix, mhc_scale, mhc_base)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
