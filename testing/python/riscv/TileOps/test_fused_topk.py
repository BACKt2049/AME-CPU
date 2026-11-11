from __future__ import annotations

import torch

from ._harness import compile_tileops_jit, get_kernel_class


def test_fused_topk_softmax_float32_runtime_compare() -> None:
    kernel_cls = get_kernel_class("moe.fused_topk", "FusedTopKKernel")
    del kernel_cls
    module = __import__("tileops.kernels.moe.fused_topk", fromlist=["_fused_topk_kernel"])
    num_tokens, num_experts, top_k = 2, 8, 2
    jit_kernel = module._fused_topk_kernel(
        num_tokens, num_experts, top_k, "softmax", "float32", False
    )
    kernel = compile_tileops_jit(jit_kernel, {"TOKENS_PER_BLOCK": 1}, out_idx=[])

    gating = torch.linspace(-1.0, 1.0, num_tokens * num_experts, dtype=torch.float32).reshape(
        num_tokens, num_experts
    )
    weights = torch.empty((num_tokens, top_k), dtype=torch.float32)
    ids = torch.empty((num_tokens, top_k), dtype=torch.int32)
    kernel(gating.contiguous(), weights, ids)

    scores = torch.softmax(gating, dim=-1)
    expected_weights, expected_ids = torch.topk(scores, top_k, dim=-1, sorted=False)
    order = ids.argsort(dim=-1)
    actual_ids = ids.gather(1, order).to(torch.int64)
    actual_weights = weights.gather(1, order)
    expected_order = expected_ids.argsort(dim=-1)
    expected_ids = expected_ids.gather(1, expected_order)
    expected_weights = expected_weights.gather(1, expected_order)
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-5)


def test_fused_topk_sigmoid_correction_bias_float32_runtime_compare() -> None:
    get_kernel_class("moe.fused_topk", "FusedTopKKernel")
    module = __import__("tileops.kernels.moe.fused_topk", fromlist=["_fused_topk_kernel"])
    num_tokens, num_experts, top_k = 2, 8, 2
    jit_kernel = module._fused_topk_kernel(
        num_tokens, num_experts, top_k, "sigmoid", "float32", True
    )
    kernel = compile_tileops_jit(jit_kernel, {"TOKENS_PER_BLOCK": 1}, out_idx=[])

    gating = torch.linspace(-1.0, 1.0, num_tokens * num_experts, dtype=torch.float32).reshape(
        num_tokens, num_experts
    )
    correction_bias = torch.linspace(0.2, -0.2, num_experts, dtype=torch.float32)
    weights = torch.empty((num_tokens, top_k), dtype=torch.float32)
    ids = torch.empty((num_tokens, top_k), dtype=torch.int32)
    kernel(gating.contiguous(), correction_bias, weights, ids)

    scores = torch.sigmoid(gating)
    expected_ids = (scores + correction_bias).topk(top_k, dim=-1, sorted=False).indices
    expected_weights = scores.gather(1, expected_ids)
    order = ids.argsort(dim=-1)
    actual_ids = ids.gather(1, order).to(torch.int64)
    actual_weights = weights.gather(1, order)
    expected_order = expected_ids.argsort(dim=-1)
    expected_ids = expected_ids.gather(1, expected_order)
    expected_weights = expected_weights.gather(1, expected_order)
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-5)
