from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def _reference(
    x: torch.Tensor,
    cb: torch.Tensor,
    dA_cumsum: torch.Tensor,
    c_mat: torch.Tensor,
    prev_states: torch.Tensor,
    dt: torch.Tensor,
    *,
    n_groups: int,
) -> torch.Tensor:
    batch, seq_len, n_heads, d_head = x.shape
    _, num_chunks, _, chunk_len, _ = cb.shape
    d_state = c_mat.shape[-1]
    heads_per_group = n_heads // n_groups

    x_chunked = x.float().reshape(batch, num_chunks, chunk_len, n_heads, d_head)
    c_chunked = c_mat.float().reshape(batch, num_chunks, chunk_len, n_groups, d_state)
    out = torch.zeros(batch, seq_len, n_heads, d_head, dtype=torch.float32)

    for bi in range(batch):
        for ci in range(num_chunks):
            for hi in range(n_heads):
                group = hi // heads_per_group
                for li in range(chunk_len):
                    decay_l = dA_cumsum[bi, hi, ci, li].float()
                    for pi in range(d_head):
                        hist = torch.dot(
                            c_chunked[bi, ci, li, group],
                            prev_states[bi, ci, hi, pi].float(),
                        ) * torch.exp(decay_l)
                        intra = torch.tensor(0.0)
                        for si in range(li + 1):
                            intra = intra + (
                                cb[bi, ci, group, li, si].float()
                                * torch.exp(decay_l - dA_cumsum[bi, hi, ci, si].float())
                                * dt[bi, hi, ci, si].float()
                                * x_chunked[bi, ci, si, hi, pi].float()
                            )
                        out[bi, ci * chunk_len + li, hi, pi] = hist + intra
    return out


def test_ssd_chunk_scan_float32_runtime_compare():
    batch, num_chunks, chunk_len = 1, 2, 4
    n_heads, d_head, d_state, n_groups = 2, 4, 4, 1
    seq_len = num_chunks * chunk_len
    kernel_cls = get_kernel_class("mamba.ssd_chunk_scan", "SSDChunkScanFwdKernel")
    tileops_kernel = kernel_cls(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        d_head,
        d_state,
        n_groups,
        torch.float32,
        config={
            "block_l": 4,
            "block_p": 4,
            "block_n": 4,
            "block_s": 4,
            "threads": 16,
            "num_stages": 1,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(-0.5, 0.5, batch * seq_len * n_heads * d_head, dtype=torch.float32).reshape(
        batch, seq_len, n_heads, d_head
    )
    cb = torch.linspace(
        -0.2,
        0.2,
        batch * num_chunks * n_groups * chunk_len * chunk_len,
        dtype=torch.float32,
    ).reshape(batch, num_chunks, n_groups, chunk_len, chunk_len)
    causal_mask = torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.bool))
    cb = torch.where(causal_mask, cb, torch.zeros_like(cb))
    dA_cumsum = -torch.linspace(
        0.01,
        0.08,
        batch * n_heads * num_chunks * chunk_len,
        dtype=torch.float32,
    ).reshape(batch, n_heads, num_chunks, chunk_len)
    c_mat = torch.linspace(
        -0.3,
        0.3,
        batch * seq_len * n_groups * d_state,
        dtype=torch.float32,
    ).reshape(batch, seq_len, n_groups, d_state)
    prev_states = torch.linspace(
        -0.4,
        0.4,
        batch * num_chunks * n_heads * d_head * d_state,
        dtype=torch.float32,
    ).reshape(batch, num_chunks, n_heads, d_head, d_state)
    dt = torch.linspace(
        0.01,
        0.08,
        batch * n_heads * num_chunks * chunk_len,
        dtype=torch.float32,
    ).reshape(batch, n_heads, num_chunks, chunk_len)

    actual = kernel(x, cb, dA_cumsum, c_mat, prev_states, dt)
    expected = _reference(x, cb, dA_cumsum, c_mat, prev_states, dt, n_groups=n_groups)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
