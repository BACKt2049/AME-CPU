from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
)


def _load_deltanet_fwd_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.deltanet",
        types.ModuleType("tileops.kernels.deltanet"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "deltanet")]
    _load_module(
        "tileops.kernels.deltanet.fused_prepare_compute_w_u",
        TILEOPS_ROOT / "tileops" / "kernels" / "deltanet" / "fused_prepare_compute_w_u.py",
    )
    return _load_module(
        "tileops.kernels.deltanet.deltanet_fwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "deltanet" / "deltanet_fwd.py",
    )


def test_deltanet_h_recurrence_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_deltanet_fwd_module()
    kernel = module._h_recurrence_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
        block_v=16,
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    k = torch.linspace(
        -0.4,
        0.4,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    w = torch.linspace(
        -0.2,
        0.2,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    u = torch.linspace(
        -0.3,
        0.3,
        batch * heads * seq_len * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_v)
    s0 = torch.linspace(
        -0.1,
        0.1,
        batch * heads * dim_k * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, dim_k, dim_v)

    actual_s, actual_v = kernel(k, w, u, s0)

    state = s0.clone()
    expected_s = torch.empty(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v)
    expected_s[:, :, 0] = state
    expected_v = torch.empty_like(u)
    for chunk in range(seq_len // chunk_size):
        start = chunk * chunk_size
        end = start + chunk_size
        k_c = k[:, :, start:end].float()
        w_c = w[:, :, start:end].float()
        u_c = u[:, :, start:end].float()
        ws = torch.einsum("bhtk,bhkv->bhtv", w_c, state)
        v_c = u_c - ws
        expected_v[:, :, start:end] = v_c
        state = state + torch.einsum("bhtk,bhtv->bhkv", k_c, v_c)
        expected_s[:, :, chunk + 1] = state

    torch.testing.assert_close(actual_s, expected_s, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_v, expected_v, rtol=1e-5, atol=1e-5)
