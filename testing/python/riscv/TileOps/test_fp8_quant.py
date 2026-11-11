from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_fp8_quant_float32_runtime_compare():
    batch, seq_len_kv, kv_group, index_dim = 1, 4, 1, 8
    x = torch.linspace(-2.0, 2.0, batch * seq_len_kv * kv_group * index_dim, dtype=torch.float32).reshape(
        batch, seq_len_kv, kv_group, index_dim
    )

    tileops_kernel = get_kernel_class("fp8_quant", "FP8QuantKernel")(
        batch=batch,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        index_dim=index_dim,
        dtype=torch.float32,
        config={"block_m": 4},
    )
    scale, actual = compile_tileops_kernel(tileops_kernel)(x.reshape(-1, index_dim).contiguous())
    scale = scale.reshape(batch, seq_len_kv, kv_group)
    actual = actual.reshape(batch, seq_len_kv, kv_group, index_dim)

    expected_scale = torch.maximum(x.abs().amax(dim=-1), torch.full((batch, seq_len_kv, kv_group), 1e-4))
    expected_scale = expected_scale / 448.0
    expected = torch.clamp(x / expected_scale.unsqueeze(-1), -448.0, 448.0).to(torch.float8_e4m3fn)

    torch.testing.assert_close(scale, expected_scale, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual, expected)
