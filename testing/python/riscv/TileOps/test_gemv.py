from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_gemv_float32_runtime_compare():
    n, k = 4, 8
    kernel_cls = get_kernel_class("gemm", "GemvKernel")
    tileops_kernel = kernel_cls(
        "lhs_row",
        1,
        n,
        k,
        torch.float32,
        config={
            "block_n": 4,
            "reduce_threads": 1,
            "num_stages": 1,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    a = torch.linspace(-0.5, 0.5, k, dtype=torch.float32)
    b = torch.linspace(-0.25, 0.25, n * k, dtype=torch.float32).reshape(n, k)

    actual = kernel(a.reshape(1, k), b)
    expected = (b @ a).reshape(1, n)
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)
