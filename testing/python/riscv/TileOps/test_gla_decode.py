from __future__ import annotations

import pytest
import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
):
    scale = q.shape[-1] ** -0.5
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    gk_f = gk.float()
    state_f = state.float()
    new_state = torch.exp(gk_f).unsqueeze(-1) * state_f + k_f.unsqueeze(-1) * v_f.unsqueeze(-2)
    out = scale * torch.einsum("bhk,bhkv->bhv", q_f, new_state)
    return out, new_state


@pytest.mark.parametrize("class_name", ["GLADecodeKernel", "GLADecodeFP32Kernel"])
def test_gla_decode_float32_runtime_compare(class_name: str):
    batch, heads, dim_k, dim_v = 1, 2, 16, 16
    kernel_cls = get_kernel_class("gla_recurrence", class_name)
    tileops_kernel = kernel_cls(
        batch=batch,
        head=heads,
        dim_k=dim_k,
        dim_v=dim_v,
        dtype="float32",
        config={"k_tile": 16, "num_stages": 1, "threads": 128},
    )

    adapter = getattr(tileops_kernel._kernel_fn, "adapter", None)
    assert type(adapter).__name__ == "RiscvKernelAdapter"

    q = torch.linspace(-0.5, 0.5, batch * heads * dim_k, dtype=torch.float32).reshape(
        batch, heads, dim_k
    )
    k = torch.linspace(-0.4, 0.4, batch * heads * dim_k, dtype=torch.float32).reshape(
        batch, heads, dim_k
    )
    v = torch.linspace(-0.3, 0.3, batch * heads * dim_v, dtype=torch.float32).reshape(
        batch, heads, dim_v
    )
    gk = -torch.linspace(0.1, 0.9, batch * heads * dim_k, dtype=torch.float32).reshape(
        batch, heads, dim_k
    )
    state = torch.linspace(
        -0.2,
        0.2,
        batch * heads * dim_k * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, dim_k, dim_v)

    actual = tileops_kernel(q, k, v, gk, state)
    expected = _reference(q, k, v, gk, state)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)
