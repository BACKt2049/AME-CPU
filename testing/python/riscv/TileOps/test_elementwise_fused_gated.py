from __future__ import annotations

import pytest
import torch

from ._harness import get_elementwise_kernel_class, run_fused_gated_runtime_compare


def _gelu_tanh(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * value * (1.0 + torch.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)))


@pytest.mark.parametrize(
    ("kernel_name", "activation", "rtol", "atol"),
    [
        ("SiluAndMulFwdKernel", lambda gate: gate * torch.sigmoid(gate), 1e-5, 1e-5),
        ("GeluAndMulFwdKernel", torch.nn.functional.gelu, 1e-5, 1e-5),
        ("GeluTanhAndMulFwdKernel", _gelu_tanh, 1e-5, 1e-5),
    ],
)
def test_fused_gated_float32_runtime_compare(kernel_name, activation, rtol, atol):
    m, n = 1, 256
    gate = torch.linspace(-4.0, 4.0, m * n, dtype=torch.float32).reshape(m, n)
    value = torch.linspace(0.25, 2.25, m * n, dtype=torch.float32).reshape(m, n)
    x = torch.cat([gate, value], dim=1).contiguous()

    run_fused_gated_runtime_compare(
        get_elementwise_kernel_class(kernel_name),
        x,
        lambda tensor: activation(tensor[:, :n]) * tensor[:, n:],
        m=m,
        n=n,
        kernel_kwargs={"strategy": "direct"},
        rtol=rtol,
        atol=atol,
    )
