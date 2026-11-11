from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    a: torch.Tensor,
    dt: torch.Tensor,
    x: torch.Tensor,
    b_in: torch.Tensor,
    c_in: torch.Tensor,
    state: torch.Tensor,
    *,
    n_groups: int,
):
    batch, n_heads, d_head = dt.shape
    heads_per_group = n_heads // n_groups
    new_state = state.float().clone()
    y_out = torch.empty(batch, n_heads, d_head, dtype=torch.float32)
    for b in range(batch):
        for h in range(n_heads):
            group = h // heads_per_group
            for p in range(d_head):
                updated = (
                    torch.exp(dt[b, h, p] * a[h, p]) * state[b, h, p]
                    + dt[b, h, p] * b_in[b, group] * x[b, h, p]
                )
                new_state[b, h, p] = updated
                y_out[b, h, p] = (updated * c_in[b, group]).sum()
    return y_out, new_state


def test_ssd_decode_float32_serial_runtime_compare():
    batch, n_heads, d_head, d_state, n_groups = 1, 2, 2, 4, 1
    kernel_cls = get_kernel_class("mamba.ssd_decode", "SSDDecodeKernel")
    tileops_kernel = kernel_cls(
        batch,
        n_heads,
        d_head,
        d_state,
        n_groups,
        torch.float32,
        config={"block_p": 1, "block_n": 1, "threads": 1},
    )

    a = -torch.linspace(0.1, 0.4, n_heads * d_head * d_state, dtype=torch.float32).reshape(
        n_heads, d_head, d_state
    )
    dt = torch.linspace(0.01, 0.04, batch * n_heads * d_head, dtype=torch.float32).reshape(
        batch, n_heads, d_head
    )
    x = torch.linspace(-0.5, 0.5, batch * n_heads * d_head, dtype=torch.float32).reshape(
        batch, n_heads, d_head
    )
    b_in = torch.linspace(-0.3, 0.3, batch * n_groups * d_state, dtype=torch.float32).reshape(
        batch, n_groups, d_state
    )
    c_in = torch.linspace(-0.2, 0.2, batch * n_groups * d_state, dtype=torch.float32).reshape(
        batch, n_groups, d_state
    )
    state = torch.linspace(
        -0.1,
        0.1,
        batch * n_heads * d_head * d_state,
        dtype=torch.float32,
    ).reshape(batch, n_heads, d_head, d_state)
    state_before = state.clone()

    actual_y = tileops_kernel(a, dt, x, b_in, c_in, state)
    expected_y, expected_state = _reference(
        a,
        dt,
        x,
        b_in,
        c_in,
        state_before,
        n_groups=n_groups,
    )
    torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_y, expected_y, rtol=1e-5, atol=1e-5)
