from __future__ import annotations

import pytest
import torch

from ._harness import compile_tileops_kernel, get_elementwise_kernel_class


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, rtol=1e-5, atol=1e-5) -> None:
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("kernel_name", "kernel_kwargs", "reference", "rtol", "atol"),
    [
        (
            "LeakyReluFwdKernel",
            {"negative_slope": 0.125},
            lambda x: torch.where(x > 0, x, 0.125 * x),
            1e-6,
            1e-6,
        ),
        (
            "EluFwdKernel",
            {"alpha": 1.25},
            lambda x: torch.where(x > 0, x, 1.25 * torch.expm1(x)),
            1e-5,
            1e-5,
        ),
        (
            "HardtanhFwdKernel",
            {"min_val": -0.75, "max_val": 1.25},
            lambda x: torch.clamp(x, min=-0.75, max=1.25),
            1e-6,
            1e-6,
        ),
        (
            "SoftplusFwdKernel",
            {"beta": 1.5, "threshold": 20.0},
            lambda x: torch.nn.functional.softplus(x, beta=1.5, threshold=20.0),
            1e-5,
            1e-5,
        ),
        (
            "ClampFwdKernel",
            {"min_val": -1.0, "max_val": 2.0},
            lambda x: torch.clamp(x, min=-1.0, max=2.0),
            1e-6,
            1e-6,
        ),
        (
            "ClampFwdKernel",
            {"min_val": -0.5},
            lambda x: torch.clamp(x, min=-0.5),
            1e-6,
            1e-6,
        ),
        (
            "ClampFwdKernel",
            {"max_val": 0.5},
            lambda x: torch.clamp(x, max=0.5),
            1e-6,
            1e-6,
        ),
        (
            "NanToNumFwdKernel",
            {"nan_val": 0.25, "posinf_val": 8.0, "neginf_val": -8.0},
            lambda x: torch.nan_to_num(x, nan=0.25, posinf=8.0, neginf=-8.0),
            1e-6,
            1e-6,
        ),
    ],
)
def test_parametric_unary_float32_runtime_compare(kernel_name, kernel_kwargs, reference, rtol, atol):
    x = torch.linspace(-4.0, 4.0, 1024, dtype=torch.float32)
    if kernel_name == "NanToNumFwdKernel":
        x[0] = float("nan")
        x[1] = float("inf")
        x[2] = float("-inf")

    tileops_kernel = get_elementwise_kernel_class(kernel_name)(
        N_total=x.numel(),
        dtype=x.dtype,
        **kernel_kwargs,
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    _assert_close(kernel(x.contiguous()), reference(x), rtol=rtol, atol=atol)


def test_prelu_float32_runtime_compare():
    x = torch.linspace(-4.0, 4.0, 1024, dtype=torch.float32)
    weight = torch.tensor([0.125, 0.25, 0.5, 0.75], dtype=torch.float32)

    tileops_kernel = get_elementwise_kernel_class("PreluFwdKernel")(
        N_total=x.numel(),
        C=weight.numel(),
        inner_size=64,
        dtype=x.dtype,
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    expected = torch.nn.functional.prelu(x.reshape(4, 4, 64), weight).reshape(-1)
    _assert_close(kernel(x.contiguous(), weight.contiguous()), expected)


def test_lerp_tensor_float32_runtime_compare():
    x = torch.linspace(-4.0, 4.0, 2048, dtype=torch.float32)
    end = torch.linspace(4.0, -4.0, 2048, dtype=torch.float32)
    weight = torch.linspace(0.0, 1.0, 2048, dtype=torch.float32)

    tileops_kernel = get_elementwise_kernel_class("LerpTensorFwdKernel")(
        N_total=x.numel(),
        dtype=x.dtype,
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    expected = torch.lerp(x, end, weight)
    _assert_close(kernel(x.contiguous(), end.contiguous(), weight.contiguous()), expected)


def test_clamp_tensor_float32_runtime_compare():
    x = torch.linspace(-4.0, 4.0, 2048, dtype=torch.float32)
    lo = torch.full_like(x, -1.5)
    hi = torch.full_like(x, 1.5)

    tileops_kernel = get_elementwise_kernel_class("ClampTensorFwdKernel")(
        N_total=x.numel(),
        dtype=x.dtype,
        has_min=True,
        has_max=True,
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    expected = torch.clamp(x, min=-1.5, max=1.5)
    _assert_close(kernel(x.contiguous(), lo.contiguous(), hi.contiguous()), expected)
