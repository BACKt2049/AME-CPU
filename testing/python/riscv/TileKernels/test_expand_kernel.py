import torch
import tilelang

from tile_kernels.mhc.expand_kernel import expand_to_mhc_fwd_tl
from tile_kernels.torch.mhc import expand_to_mhc_ref


# Original TileKernels source: tile_kernels/mhc/expand_kernel.py
def test_expand_kernel_runtime_compare():
    x = torch.arange(5 * 16, dtype=torch.float32).reshape(5, 16).to(torch.bfloat16).contiguous()
    expected = expand_to_mhc_ref(x, 3)

    kernel = tilelang.compile(
        expand_to_mhc_fwd_tl.get_tir(16, 3),
        out_idx=[1],
        target="riscv",
    )
    try:
        actual = kernel(x)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
