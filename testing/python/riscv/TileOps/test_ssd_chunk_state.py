from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def _reference(
    x: torch.Tensor,
    b_mat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    *,
    n_groups: int,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, seq_len, n_heads, d_head = x.shape
    _, _, num_chunks, chunk_len = dt.shape
    d_state = b_mat.shape[-1]
    heads_per_group = n_heads // n_groups

    x_chunked = x.float().reshape(batch, num_chunks, chunk_len, n_heads, d_head)
    b_chunked = b_mat.float().reshape(batch, num_chunks, chunk_len, n_groups, d_state)
    b_heads = b_chunked[:, :, :, torch.arange(n_heads) // heads_per_group, :]

    dA = dA_cumsum.float().permute(0, 2, 1, 3)
    decay = torch.exp(torch.clamp(dA[:, :, :, -1:] - dA, max=0.0))
    weight = decay * dt.float().permute(0, 2, 1, 3)

    if seq_idx is not None:
        seq_chunked = seq_idx.reshape(batch, num_chunks, chunk_len)
        seq_end = seq_chunked[..., -1:]
        same = ((seq_end >= 0) & (seq_chunked == seq_end)).unsqueeze(3)
        weight = weight * same.permute(0, 1, 3, 2)

    weighted = weight.permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)
    out = (weighted * b_heads.unsqueeze(-1) * x_chunked.unsqueeze(-2)).sum(dim=2)
    return out.permute(0, 1, 2, 4, 3)


def _run_case(has_seq_idx: bool) -> torch.Tensor:
    batch, num_chunks, chunk_len = 1, 2, 4
    n_heads, d_head, d_state, n_groups = 2, 4, 4, 1
    seq_len = num_chunks * chunk_len
    kernel_cls = get_kernel_class("mamba.ssd_chunk_state", "SSDChunkStateFwdKernel")
    tileops_kernel = kernel_cls(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        d_head,
        d_state,
        n_groups,
        torch.float32,
        has_seq_idx=has_seq_idx,
        config={"block_n": 4, "block_p": 4, "block_l": 4, "threads": 16},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(-0.5, 0.5, batch * seq_len * n_heads * d_head, dtype=torch.float32).reshape(
        batch, seq_len, n_heads, d_head
    )
    b_mat = torch.linspace(-0.25, 0.25, batch * seq_len * n_groups * d_state, dtype=torch.float32).reshape(
        batch, seq_len, n_groups, d_state
    )
    dt = torch.linspace(0.01, 0.08, batch * n_heads * num_chunks * chunk_len, dtype=torch.float32).reshape(
        batch, n_heads, num_chunks, chunk_len
    )
    dA_cumsum = -torch.linspace(
        0.01, 0.06, batch * n_heads * num_chunks * chunk_len, dtype=torch.float32
    ).reshape(batch, n_heads, num_chunks, chunk_len)
    seq_idx = torch.ones(batch, seq_len, dtype=torch.int32)
    if has_seq_idx:
        seq_idx[:, :chunk_len] = -1

    actual = kernel(x, b_mat, dt, dA_cumsum, seq_idx)
    expected = _reference(
        x,
        b_mat,
        dt,
        dA_cumsum,
        n_groups=n_groups,
        seq_idx=seq_idx if has_seq_idx else None,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    return actual


def test_ssd_chunk_state_float32_runtime_compare():
    _run_case(has_seq_idx=False)


def test_ssd_chunk_state_float32_seq_idx_runtime_compare():
    actual = _run_case(has_seq_idx=True)
    torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]), rtol=0.0, atol=0.0)
