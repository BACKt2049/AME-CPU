from __future__ import annotations

import pytest
import torch

from ._harness import get_elementwise_kernel_class, run_binary_runtime_compare


@pytest.mark.parametrize(
    ("kernel_name", "reference", "rtol", "atol"),
    [
        ("AddFwdKernel", torch.add, 1e-6, 1e-6),
        ("SubFwdKernel", torch.sub, 1e-6, 1e-6),
        ("MulFwdKernel", torch.mul, 1e-6, 1e-6),
        ("DivFwdKernel", torch.div, 1e-5, 1e-5),
        (
            "DivTruncFwdKernel",
            lambda a, b: torch.div(a, b, rounding_mode="trunc"),
            1e-6,
            1e-6,
        ),
        ("FloorDivideFwdKernel", torch.floor_divide, 1e-6, 1e-6),
        ("RemainderFwdKernel", torch.remainder, 1e-6, 1e-6),
        ("PowFwdKernel", torch.pow, 1e-5, 1e-5),
        ("LerpFwdKernel", lambda a, b: torch.lerp(a, b, 0.375), 1e-6, 1e-6),
        ("MaximumFwdKernel", torch.maximum, 1e-6, 1e-6),
        ("MinimumFwdKernel", torch.minimum, 1e-6, 1e-6),
    ],
)
def test_binary_float32_runtime_compare(kernel_name, reference, rtol, atol):
    a = torch.linspace(-4.0, 4.0, 256, dtype=torch.float32)
    b = torch.linspace(0.25, 2.25, 256, dtype=torch.float32)
    if kernel_name == "PowFwdKernel":
        a = torch.linspace(0.25, 2.0, 256, dtype=torch.float32)
        b = torch.linspace(0.5, 2.0, 256, dtype=torch.float32)
    run_binary_runtime_compare(
        get_elementwise_kernel_class(kernel_name),
        a,
        b,
        reference,
        kernel_kwargs={"strategy": "direct", "weight": 0.375}
        if kernel_name == "LerpFwdKernel"
        else {"strategy": "direct"},
        rtol=rtol,
        atol=atol,
    )
