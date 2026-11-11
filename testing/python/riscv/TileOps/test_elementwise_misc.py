from __future__ import annotations

import pytest
import torch

from ._harness import (
    compile_tileops_kernel,
    get_elementwise_kernel_class,
    run_unary_runtime_compare,
)


def _run_binary_bool_compare(
    kernel_name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    reference,
    *,
    kernel_dtype: torch.dtype | None = None,
) -> None:
    a_flat = a.contiguous().reshape(-1)
    b_flat = b.contiguous().reshape(-1)
    kernel_cls = get_elementwise_kernel_class(kernel_name)
    tileops_kernel = kernel_cls(
        (a_flat.numel(),),
        (b_flat.numel(),),
        kernel_dtype or a.dtype,
        config={"strategy": "direct"},
    )
    actual = compile_tileops_kernel(tileops_kernel)(a_flat, b_flat)
    expected = reference(a, b).reshape(-1)
    torch.testing.assert_close(
        actual.reshape(expected.shape).to(torch.bool),
        expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("kernel_name", "reference"),
    [
        ("EqFwdKernel", torch.eq),
        ("NeFwdKernel", torch.ne),
        ("GtFwdKernel", torch.gt),
        ("LtFwdKernel", torch.lt),
        ("GeFwdKernel", torch.ge),
        ("LeFwdKernel", torch.le),
    ],
)
def test_comparison_float32_runtime_compare(kernel_name, reference):
    a = torch.linspace(-4.0, 4.0, 256, dtype=torch.float32)
    b = torch.linspace(0.25, 2.25, 256, dtype=torch.float32)
    _run_binary_bool_compare(kernel_name, a, b, reference)


@pytest.mark.parametrize(
    ("kernel_name", "reference"),
    [
        ("LogicalAndFwdKernel", torch.logical_and),
        ("LogicalOrFwdKernel", torch.logical_or),
    ],
)
def test_logical_float32_runtime_compare(kernel_name, reference):
    a = torch.tensor([0.0, 1.0, -2.0, 0.0], dtype=torch.float32).repeat(64)
    b = torch.tensor([0.0, 0.0, 3.0, 4.0], dtype=torch.float32).repeat(64)
    _run_binary_bool_compare(kernel_name, a, b, reference)


@pytest.mark.parametrize(
    ("kernel_name", "reference"),
    [
        ("EqBoolStorageFwdKernel", torch.eq),
        ("NeBoolStorageFwdKernel", torch.ne),
        ("GtBoolStorageFwdKernel", torch.gt),
        ("LtBoolStorageFwdKernel", torch.lt),
        ("GeBoolStorageFwdKernel", torch.ge),
        ("LeBoolStorageFwdKernel", torch.le),
        ("LogicalAndBoolStorageFwdKernel", torch.logical_and),
        ("LogicalOrBoolStorageFwdKernel", torch.logical_or),
        ("BitwiseAndBoolStorageFwdKernel", torch.bitwise_and),
        ("BitwiseOrBoolStorageFwdKernel", torch.bitwise_or),
        ("BitwiseXorBoolStorageFwdKernel", torch.bitwise_xor),
    ],
)
def test_bool_storage_binary_uint8_runtime_compare(kernel_name, reference):
    a = torch.tensor([0, 1, 1, 0], dtype=torch.int8).repeat(64)
    b = torch.tensor([0, 0, 1, 1], dtype=torch.int8).repeat(64)
    a_bool = a.to(torch.bool)
    b_bool = b.to(torch.bool)
    _run_binary_bool_compare(
        kernel_name,
        a,
        b,
        lambda _a, _b: reference(a_bool, b_bool),
        kernel_dtype=torch.uint8,
    )


@pytest.mark.parametrize(
    ("kernel_name", "reference"),
    [
        ("BitwiseAndFwdKernel", torch.bitwise_and),
        ("BitwiseOrFwdKernel", torch.bitwise_or),
        ("BitwiseXorFwdKernel", torch.bitwise_xor),
    ],
)
def test_bitwise_int32_runtime_compare(kernel_name, reference):
    a = torch.arange(256, dtype=torch.int32)
    b = torch.arange(256, 512, dtype=torch.int32)
    a_flat = a.contiguous().reshape(-1)
    b_flat = b.contiguous().reshape(-1)
    tileops_kernel = get_elementwise_kernel_class(kernel_name)(
        (a_flat.numel(),),
        (b_flat.numel(),),
        a.dtype,
        config={"strategy": "direct"},
    )
    actual = compile_tileops_kernel(tileops_kernel)(a_flat, b_flat)
    torch.testing.assert_close(actual, reference(a, b), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("kernel_name", "reference"),
    [
        ("LogicalNotFwdKernel", lambda x: torch.logical_not(x)),
        ("IsnanFwdKernel", torch.isnan),
        ("IsinfFwdKernel", torch.isinf),
        ("IsfiniteFwdKernel", torch.isfinite),
    ],
)
def test_unary_predicate_float32_runtime_compare(kernel_name, reference):
    x = torch.linspace(-4.0, 4.0, 256, dtype=torch.float32)
    if kernel_name == "IsnanFwdKernel":
        x[17] = float("nan")
    if kernel_name == "IsinfFwdKernel":
        x[29] = float("inf")
    run_unary_runtime_compare(
        get_elementwise_kernel_class(kernel_name),
        x,
        reference,
        kernel_kwargs={"strategy": "direct"},
        rtol=0.0,
        atol=0.0,
    )


def test_bitwise_not_int32_runtime_compare():
    x = torch.arange(-128, 128, dtype=torch.int32)
    run_unary_runtime_compare(
        get_elementwise_kernel_class("BitwiseNotFwdKernel"),
        x,
        torch.bitwise_not,
        kernel_kwargs={"strategy": "direct"},
        rtol=0.0,
        atol=0.0,
    )


def test_logical_not_bool_storage_uint8_runtime_compare():
    x = torch.tensor([0, 1, 1, 0], dtype=torch.int8).repeat(64)
    tileops_kernel = get_elementwise_kernel_class("LogicalNotBoolStorageFwdKernel")(
        N_total=x.numel(),
        dtype=torch.uint8,
        config={"strategy": "direct"},
    )
    actual = compile_tileops_kernel(tileops_kernel)(x.contiguous())
    expected = torch.logical_not(x.to(torch.bool))
    torch.testing.assert_close(
        actual.reshape(expected.shape).to(torch.bool),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_alibi_float32_runtime_compare():
    seq_len, num_heads = 8, 2
    alibi_cls = get_elementwise_kernel_class("AlibiFwdKernel")
    tileops_kernel = alibi_cls(
        seq_len,
        num_heads,
        torch.float32,
        config={"threads": 128, "num_per_thread": 4},
    )
    actual = compile_tileops_kernel(tileops_kernel)()

    row = torch.arange(seq_len, dtype=torch.float32)
    distance = (row[:, None] - row[None, :]).abs()
    expected = torch.stack(
        [
            -torch.pow(
                torch.tensor(2.0),
                -8.0 * (head + 1) / num_heads,
            )
            * distance
            for head in range(num_heads)
        ]
    )
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-5, atol=1e-5)


def test_sinusoidal_float32_runtime_compare():
    seq_len, d_model = 8, 8
    sinusoidal_cls = get_elementwise_kernel_class("SinusoidalFwdKernel")
    tileops_kernel = sinusoidal_cls(
        seq_len,
        d_model,
        torch.float32,
        config={"threads": 128, "num_per_thread": 4},
    )
    actual = compile_tileops_kernel(tileops_kernel)()

    position = torch.arange(seq_len, dtype=torch.float32)[:, None]
    dim = torch.arange(d_model, dtype=torch.float32)[None, :]
    divisor = torch.pow(10000.0, (2.0 * torch.floor(dim / 2.0)) / d_model)
    angle = position / divisor
    expected = torch.where((dim.remainder(2) == 0), torch.sin(angle), torch.cos(angle))
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-5, atol=1e-5)


def test_where_float32_runtime_compare():
    n_total = 2048
    cond = (torch.arange(n_total) % 3 == 0).to(torch.int8)
    x = torch.linspace(-4.0, 4.0, n_total, dtype=torch.float32)
    y = torch.linspace(4.0, -4.0, n_total, dtype=torch.float32)

    tileops_kernel = get_elementwise_kernel_class("WhereFwdKernel")(
        N_total=n_total,
        dtype=x.dtype,
    )
    actual = compile_tileops_kernel(tileops_kernel)(
        cond.contiguous(),
        x.contiguous(),
        y.contiguous(),
    )
    expected = torch.where(cond.to(torch.bool), x, y)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-6, atol=1e-6)


def test_masked_fill_scalar_float32_runtime_compare():
    x = torch.linspace(-4.0, 4.0, 2048, dtype=torch.float32)
    mask = (torch.arange(x.numel()) % 3 == 0).to(torch.int8)
    fill_value = -7.5

    tileops_kernel = get_elementwise_kernel_class("MaskedFillFwdKernel")(
        N_total=x.numel(),
        dtype=x.dtype,
        fill_value=fill_value,
    )
    actual = compile_tileops_kernel(tileops_kernel)(x.contiguous(), mask.contiguous())
    expected = x.masked_fill(mask.to(torch.bool), fill_value)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-6, atol=1e-6)


def test_masked_fill_tensor_value_float32_runtime_compare():
    x = torch.linspace(-4.0, 4.0, 2048, dtype=torch.float32)
    mask = (torch.arange(x.numel()) % 3 == 0).to(torch.int8)
    fill_value = torch.tensor([-2.25], dtype=torch.float32)

    tileops_kernel = get_elementwise_kernel_class("MaskedFillTensorValueFwdKernel")(
        N_total=x.numel(),
        dtype=x.dtype,
    )
    actual = compile_tileops_kernel(tileops_kernel)(
        x.contiguous(),
        mask.contiguous(),
        fill_value.contiguous(),
    )
    expected = x.masked_fill(mask.to(torch.bool), fill_value.item())
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-6, atol=1e-6)
