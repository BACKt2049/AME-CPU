import torch
import tilelang
import tilelang.language as T

from tile_kernels.mhc.pre_split_mixes_kernel import _mhc_pre_split_mixes_fwd


# Original TileKernels source: tile_kernels/mhc/pre_split_mixes_kernel.py
def test_pre_split_mixes_kernel_runtime_compare():
    input_mixes = torch.arange(2 * 24, dtype=torch.float32).reshape(2, 24).contiguous() / 10
    mhc_scale = torch.tensor([0.5, 1.5, 2.0], dtype=torch.float32)
    mhc_base = torch.tensor([0.1 * i for i in range(24)], dtype=torch.float32)
    mhc_mult = 4
    expected = (
        torch.sigmoid(input_mixes[:, :mhc_mult] * mhc_scale[0] + mhc_base[:mhc_mult]) + 1e-6,
        torch.sigmoid(input_mixes[:, mhc_mult : 2 * mhc_mult] * mhc_scale[1] + mhc_base[mhc_mult : 2 * mhc_mult]) * 2.0,
        (input_mixes[:, 2 * mhc_mult :] * mhc_scale[2] + mhc_base[2 * mhc_mult :]).reshape(2, mhc_mult * mhc_mult),
    )

    kernel = tilelang.compile(
        _mhc_pre_split_mixes_fwd.get_tir(4, 2.0, 1e-6, 2, T.float32),
        out_idx=[3, 4, 5],
        target="riscv",
    )
    try:
        actual = kernel(input_mixes, mhc_scale, mhc_base)
    finally:
        kernel.close()

    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
