from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
)


def _load_deltanet_compute_w_u_bwd_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.deltanet",
        types.ModuleType("tileops.kernels.deltanet"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "deltanet")]
    return _load_module(
        "tileops.kernels.deltanet.compute_w_u_bwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "deltanet" / "compute_w_u_bwd.py",
    )


def test_deltanet_compute_w_u_bwd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_deltanet_compute_w_u_bwd_module()
    kernel = module.compute_w_u_bwd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    dw = torch.linspace(-0.5, 0.5, batch * heads * seq_len * dim_k, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_k
    )
    du_partial = torch.linspace(0.4, -0.4, batch * heads * seq_len * dim_v, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_v
    )
    du_corr = torch.linspace(-0.2, 0.2, batch * heads * seq_len * dim_v, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_v
    )
    state = torch.linspace(
        -0.3,
        0.3,
        batch * heads * (seq_len // chunk_size + 1) * dim_k * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v)
    aw = torch.tensor([[[[1.0, 0.0], [-0.125, 1.0]]]], dtype=torch.float32)
    au = torch.tensor([[[[1.0, 0.0], [-0.25, 1.0]]]], dtype=torch.float32)
    k = torch.linspace(-0.4, 0.4, batch * heads * seq_len * dim_k, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_k
    )
    v = torch.linspace(-0.3, 0.3, batch * heads * seq_len * dim_v, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_v
    )
    beta = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)
    dk_partial = torch.linspace(0.05, -0.05, batch * heads * seq_len * dim_k, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_k
    )
    dk_corr = torch.linspace(-0.03, 0.03, batch * heads * seq_len * dim_k, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_k
    )

    actual_dk, actual_dv, actual_dbeta = kernel(
        dw,
        du_partial,
        du_corr,
        state,
        aw,
        au,
        k,
        v,
        beta,
        dk_partial,
        dk_corr,
    )

    s0 = state[0, 0, 0]
    dw_total = dw[0, 0] - du_corr[0, 0] @ s0.T
    du = du_partial[0, 0] + du_corr[0, 0]
    k_beta = k[0, 0] * beta[0, 0].unsqueeze(-1)
    v_beta = v[0, 0] * beta[0, 0].unsqueeze(-1)

    daw = dw_total @ k_beta.T
    d_k_beta = aw[0, 0].T @ dw_total
    dau = du @ v_beta.T
    d_v_beta = au[0, 0].T @ du
    expected_dv = d_v_beta * beta[0, 0].unsqueeze(-1)
    dbeta_direct = (d_k_beta * k[0, 0]).sum(dim=-1) + (d_v_beta * v[0, 0]).sum(dim=-1)

    d_a_inv = daw + dau
    d_a = -(aw[0, 0].T @ (d_a_inv @ aw[0, 0].T))
    d_p = torch.tril(d_a, diagonal=-1)
    dk_a = d_p @ k[0, 0]
    dk_a = dk_a * beta[0, 0].unsqueeze(-1)
    dk_a = dk_a + d_p.T @ k_beta
    expected_dk = dk_partial[0, 0] + dk_corr[0, 0] + d_k_beta * beta[0, 0].unsqueeze(-1) + dk_a
    expected_dbeta = dbeta_direct + (d_p * (k[0, 0] @ k[0, 0].T)).sum(dim=-1)

    torch.testing.assert_close(actual_dk[0, 0], expected_dk, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dv[0, 0], expected_dv, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dbeta[0, 0], expected_dbeta, rtol=1e-5, atol=1e-5)
