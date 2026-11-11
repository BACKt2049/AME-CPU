from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
)


def _load_deltanet_prepare_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.deltanet",
        types.ModuleType("tileops.kernels.deltanet"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "deltanet")]
    return _load_module(
        "tileops.kernels.deltanet.fused_prepare_compute_w_u",
        TILEOPS_ROOT / "tileops" / "kernels" / "deltanet" / "fused_prepare_compute_w_u.py",
    )


def test_deltanet_fused_prepare_compute_w_u_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_deltanet_prepare_module()
    kernel = module.fused_prepare_compute_w_u_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    k = torch.linspace(
        -0.4,
        0.4,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    v = torch.linspace(
        -0.3,
        0.3,
        batch * heads * seq_len * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_v)
    beta = torch.tensor([[[0.2, 0.7]]], dtype=torch.float32)

    actual_aw, actual_au, actual_w, actual_u = kernel(k, v, beta)

    gram = k[0, 0] @ k[0, 0].T
    transform = torch.eye(chunk_size, dtype=torch.float32)
    for i in range(chunk_size):
        for j in range(chunk_size):
            if i > j:
                transform[i, j] = -gram[i, j] * beta[0, 0, i]
    k_beta = k[0, 0] * beta[0, 0].unsqueeze(-1)
    v_beta = v[0, 0] * beta[0, 0].unsqueeze(-1)

    torch.testing.assert_close(actual_aw[0, 0], transform, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_au[0, 0], transform, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_w[0, 0], transform @ k_beta, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_u[0, 0], transform @ v_beta, rtol=1e-5, atol=1e-5)
