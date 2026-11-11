import torch
import tilelang

from tile_kernels.moe.get_fused_mapping_kernel import get_get_fused_mapping_kernel


# Original TileKernels source: tile_kernels/moe/get_fused_mapping_kernel.py
def test_get_fused_mapping_kernel_runtime_compare():
    topk_idx = torch.zeros((1, 1), dtype=torch.int64)
    pos_to_expert = torch.empty((64,), dtype=torch.int32)
    pos_to_token = torch.empty((64,), dtype=torch.int32)
    pos_to_token_topk = torch.empty((64,), dtype=torch.int32)
    token_topk_to_pos = torch.empty((1, 1), dtype=torch.int32)
    expert_start = torch.empty((1,), dtype=torch.int32)
    expert_end = torch.empty((1,), dtype=torch.int32)
    num_tokens_per_expert = torch.empty((1,), dtype=torch.int32)
    num_experts_per_sm = torch.empty((1, 1), dtype=torch.int32)

    kernel = tilelang.compile(
        get_get_fused_mapping_kernel.get_tir(1, 1, 64, 1),
        out_idx=[],
        target="riscv",
    )
    try:
        kernel(
            topk_idx,
            pos_to_expert,
            pos_to_token,
            pos_to_token_topk,
            token_topk_to_pos,
            expert_start,
            expert_end,
            num_tokens_per_expert,
            num_experts_per_sm,
        )
    finally:
        kernel.close()

    assert pos_to_expert[0].item() == 0
    assert pos_to_token[0].item() == 0
    assert pos_to_token_topk[0].item() == 0
    assert token_topk_to_pos[0, 0].item() == 0
