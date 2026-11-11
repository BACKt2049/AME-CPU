from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    offsets: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    groups: int,
    scale: float,
):
    c_seq_len, heads, dim = q.shape
    head_kv = heads // groups
    out = torch.empty_like(q.float())
    for token_idx in range(c_seq_len):
        seq_id = int(token_indices[token_idx, 0])
        local_t = int(token_indices[token_idx, 1])
        bos = int(offsets[seq_id])
        for kv_head in range(head_kv):
            ns = int(block_counts[bos + local_t, kv_head])
            for g in range(groups):
                head = kv_head * groups + g
                scores = []
                values = []
                for n in range(ns):
                    blk = int(block_indices[bos + local_t, kv_head, n])
                    start = blk * block_size
                    if 0 <= start <= local_t:
                        for j in range(block_size):
                            pos = start + j
                            if pos <= local_t:
                                scores.append(
                                    (q[bos + local_t, head].float() * k[bos + pos, kv_head].float()).sum()
                                    * scale
                                )
                                values.append(v[bos + pos, kv_head].float())
                probs = torch.softmax(torch.stack(scores), dim=0)
                out[bos + local_t, head] = torch.stack(values).mul(probs[:, None]).sum(dim=0)
    return out


def test_nsa_fwd_varlen_float32_runtime_compare():
    batch, heads, c_seq_len, dim = 1, 2, 4, 16
    block_size, groups, selected_blocks = 2, 2, 2
    kv_group = heads // groups
    scale = dim**-0.5
    kernel_cls = get_kernel_class("attention.deepseek_nsa_fwd", "NSAFwdVarlenKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        c_seq_len=c_seq_len,
        dim=dim,
        is_causal=True,
        scale=scale,
        block_size=block_size,
        groups=groups,
        selected_blocks=selected_blocks,
        dtype=torch.float32,
        accum_dtype=torch.float32,
        config={"threads": 32},
    )

    q = torch.linspace(-0.5, 0.5, c_seq_len * heads * dim, dtype=torch.float32).reshape(
        c_seq_len, heads, dim
    )
    k = torch.linspace(
        -0.4,
        0.4,
        c_seq_len * kv_group * dim,
        dtype=torch.float32,
    ).reshape(c_seq_len, kv_group, dim)
    v = torch.linspace(
        -0.3,
        0.3,
        c_seq_len * kv_group * dim,
        dtype=torch.float32,
    ).reshape(c_seq_len, kv_group, dim)
    offsets = torch.tensor([0, c_seq_len], dtype=torch.int32)
    token_indices = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.int32)
    block_indices = torch.tensor(
        [[[0, 0]], [[0, 0]], [[0, 1]], [[0, 1]]],
        dtype=torch.int32,
    )
    block_counts = torch.tensor([[1], [1], [2], [2]], dtype=torch.int32)

    actual = tileops_kernel(q, k, v, block_indices, block_counts, offsets, token_indices)
    expected = _reference(
        q,
        k,
        v,
        block_indices,
        block_counts,
        offsets,
        token_indices,
        block_size,
        groups,
        scale,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
