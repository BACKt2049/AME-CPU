import torch
import tilelang

from tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_fwd
from tile_kernels.torch.mhc import sinkhorn_normalize_ref


# Original TileKernels source: tile_kernels/mhc/sinkhorn_kernel.py
def test_sinkhorn_kernel_runtime_compare():
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4).contiguous()
    expected = sinkhorn_normalize_ref(x, repeat=3, eps=1e-6)

    kernel = tilelang.compile(
        _mhc_sinkhorn_fwd.get_tir(4, 2, 3, 1e-6),
        out_idx=[1],
        target="riscv",
    )
    try:
        actual = kernel(x)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
