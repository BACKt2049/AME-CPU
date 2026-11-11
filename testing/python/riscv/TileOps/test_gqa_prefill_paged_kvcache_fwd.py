from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _gather_pages(
    pages: torch.Tensor,
    block_table: torch.Tensor,
    length: int,
    page_size: int,
):
    heads_kv, dim = pages.shape[1], pages.shape[2]
    dense = torch.empty(1, length, heads_kv, dim, dtype=pages.dtype)
    for pos in range(length):
        page_idx = pos // page_size
        page_offset = pos % page_size
        physical = block_table[0, page_idx].item() * page_size + page_offset
        dense[0, pos] = pages[physical]
    return dense


def _reference(q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor, heads_kv: int):
    total_q, heads, dim = q.shape
    heads_per_group = heads // heads_kv
    out = torch.empty_like(q)
    scale = dim ** -0.5
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum("qd,kd->qk", q[:, head, :].float(), k_all[0, :, group, :].float())
        probs = torch.softmax(scores * scale, dim=-1)
        out[:, head, :] = torch.einsum("qk,kd->qd", probs, v_all[0, :, group, :].float())
    return out


def test_gqa_prefill_paged_kvcache_fwd_float32_runtime_compare_and_append():
    batch, heads, heads_kv, seq_new, old_len, page_size, max_pages, dim = 1, 4, 2, 3, 2, 4, 2, 16
    total_len = old_len + seq_new
    physical_tokens = max_pages * page_size
    kernel_cls = get_kernel_class("attention.gqa_fwd", "GQAPrefillPagedWithKVCacheFwdKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_pages_per_req=max_pages,
        page_size=page_size,
        dim=dim,
        is_causal=False,
        dtype=torch.float32,
        config={"block_m": 4, "block_n": 4, "num_stages": 1, "threads": 128},
    )

    q = torch.linspace(-0.5, 0.5, seq_new * heads * dim, dtype=torch.float32).reshape(
        seq_new, heads, dim
    )
    k_new = torch.linspace(-0.4, 0.4, seq_new * heads_kv * dim, dtype=torch.float32).reshape(
        seq_new, heads_kv, dim
    )
    v_new = torch.linspace(-0.3, 0.3, seq_new * heads_kv * dim, dtype=torch.float32).reshape(
        seq_new, heads_kv, dim
    )
    k_pages = torch.zeros(physical_tokens, heads_kv, dim, dtype=torch.float32)
    v_pages = torch.zeros_like(k_pages)
    k_pages[:old_len] = torch.linspace(-0.2, 0.2, old_len * heads_kv * dim, dtype=torch.float32).reshape(
        old_len, heads_kv, dim
    )
    v_pages[:old_len] = torch.linspace(-0.1, 0.1, old_len * heads_kv * dim, dtype=torch.float32).reshape(
        old_len, heads_kv, dim
    )
    k_before = k_pages.clone()
    v_before = v_pages.clone()
    cu_seqlens_q = torch.tensor([0, seq_new], dtype=torch.int32)
    cache_seqlens = torch.tensor([old_len], dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    actual = tileops_kernel(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        None,
        None,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
        seq_new,
    )
    expected_k_pages = k_before.clone()
    expected_v_pages = v_before.clone()
    for i in range(seq_new):
        logical = old_len + i
        physical = block_table[0, logical // page_size].item() * page_size + logical % page_size
        expected_k_pages[physical] = k_new[i]
        expected_v_pages[physical] = v_new[i]

    k_all = _gather_pages(expected_k_pages, block_table, total_len, page_size)
    v_all = _gather_pages(expected_v_pages, block_table, total_len, page_size)
    expected = _reference(q, k_all, v_all, heads_kv)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_pages, expected_k_pages, rtol=0, atol=0)
    torch.testing.assert_close(v_pages, expected_v_pages, rtol=0, atol=0)
