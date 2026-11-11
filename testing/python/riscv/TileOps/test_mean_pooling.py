from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_mean_pooling_float32_runtime_compare():
    batch, seq_len, heads, dim = 1, 4, 1, 4
    chunk_size, chunks_per_batch, seq_num = 2, 2, 1
    x = torch.arange(batch * seq_len * heads * dim, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim
    )
    offsets = torch.tensor([0, seq_len], dtype=torch.int32)
    indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)

    kernel_cls = get_kernel_class(
        "attention.deepseek_nsa_mean_pooling_fwd",
        "MeanPoolingFwdKernel",
    )
    tileops_kernel = kernel_cls(
        batch,
        seq_len,
        heads,
        dim,
        chunk_size,
        chunks_per_batch,
        seq_num,
        0,
        torch.float32,
        torch.float32,
        config={"bwidth": 4, "threads": 4},
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(x.reshape(batch, seq_len, heads * dim).contiguous(), offsets, indices)
    actual = actual.reshape(batch, chunks_per_batch, heads, dim)
    expected = x.reshape(batch, chunks_per_batch, chunk_size, heads, dim).mean(dim=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
