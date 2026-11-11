import pytest
import torch
import tilelang
import tilelang.language as T

from tile_kernels.transpose.batched_transpose_kernel import get_batched_transpose_kernel


# Original TileKernels source: tile_kernels/transpose/batched_transpose_kernel.py
@pytest.mark.parametrize(
    ("torch_dtype", "tilelang_dtype", "rtol", "atol"),
    [
        (torch.float32, T.float32, 1.3e-6, 1e-5),
        (torch.bfloat16, T.bfloat16, 1.3e-6, 1e-5),
    ],
)
def test_batched_transpose_kernel_runtime_compare(torch_dtype, tilelang_dtype, rtol, atol):
    base = (torch.arange(2 * 64 * 64, dtype=torch.float32).reshape(2, 64, 64) / 32).contiguous()
    x = base if torch_dtype == torch.float32 else base.to(torch_dtype)
    expected = x.transpose(1, 2).contiguous()

    kernel = tilelang.compile(
        get_batched_transpose_kernel.get_tir(64, 64, tilelang_dtype),
        out_idx=[1],
        target="riscv",
    )
    try:
        actual = kernel(x)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
