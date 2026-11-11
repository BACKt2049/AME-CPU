from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_bmm_float32_runtime_compare():
    batch, m, n, k = 1, 8, 8, 16
    kernel_cls = get_kernel_class("bmm", "BmmKernel")
    tileops_kernel = kernel_cls(
        batch,
        m,
        n,
        k,
        torch.float32,
        config={
            "block_m": 8,
            "block_n": 8,
            "block_k": 16,
            "num_stages": 1,
            "threads": 16,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    a = torch.linspace(-0.5, 0.5, batch * m * k, dtype=torch.float32).reshape(batch, m, k)
    b = torch.linspace(-0.25, 0.25, batch * k * n, dtype=torch.float32).reshape(batch, k, n)

    actual = kernel(a, b)
    expected = torch.bmm(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
