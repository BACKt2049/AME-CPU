import torch
import tilelang

from tile_kernels.mhc.pre_apply_mix_kernel import _mhc_pre_apply_mix_fwd


# Original TileKernels source: tile_kernels/mhc/pre_apply_mix_kernel.py
def test_pre_apply_mix_kernel_runtime_compare():
    x = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8).to(torch.bfloat16).contiguous()
    mix = torch.tensor(
        [
            [0.2, 0.3, 0.4, 0.5],
            [0.7, 0.1, 0.2, 0.3],
            [1.0, 0.5, 0.25, 0.125],
        ],
        dtype=torch.float32,
    ).contiguous()
    expected = (x.float() * mix.unsqueeze(-1)).sum(-2).bfloat16()

    kernel = tilelang.compile(
        _mhc_pre_apply_mix_fwd.get_tir(4, 8, 128, 8),
        out_idx=[2],
        target="riscv",
    )
    try:
        actual = kernel(x, mix)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
