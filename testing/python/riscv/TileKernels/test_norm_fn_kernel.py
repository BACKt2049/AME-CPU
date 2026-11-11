import torch
import tilelang
import tilelang.language as T

from tile_kernels.mhc.norm_fn_kernel import _mhc_fn_normw_merge_fwd


# Original TileKernels source: tile_kernels/mhc/norm_fn_kernel.py
def test_norm_fn_kernel_runtime_compare():
    fn = (torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64) / 16).contiguous()
    normw = torch.linspace(0.25, 1.25, 64, dtype=torch.float32).contiguous()
    expected = fn * normw

    kernel = tilelang.compile(
        _mhc_fn_normw_merge_fwd.get_tir(4, 64, T.float32),
        out_idx=[2],
        target="riscv",
    )
    try:
        actual = kernel(fn, normw)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
