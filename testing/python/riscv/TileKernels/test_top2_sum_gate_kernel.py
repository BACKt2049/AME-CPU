import torch
import tilelang

from tile_kernels.moe.top2_sum_gate_kernel import get_top2_sum_gate_kernel


# Original TileKernels source: tile_kernels/moe/top2_sum_gate_kernel.py
def test_top2_sum_gate_kernel_runtime_compare():
    logits = torch.tensor([[2.0, 0.3, -0.2, 1.5]], dtype=torch.float32)
    bias = torch.zeros(4, dtype=torch.float32)
    dummy_mask = torch.empty((1,), dtype=torch.bool)
    dummy_fix_routing_mask = torch.empty((1,), dtype=torch.bool)
    dummy_to_physical_map = torch.empty((4, 1), dtype=torch.int32)
    dummy_logical_count = torch.empty((4,), dtype=torch.int32)
    topk_idx = torch.empty((1, 1), dtype=torch.int64)
    dummy_unmapped_topk_idx = torch.empty((1, 1), dtype=torch.int64)
    topk_weights = torch.empty((1, 1), dtype=torch.float32)

    kernel = tilelang.compile(
        get_top2_sum_gate_kernel.get_tir(0, 1, 0, 0, 4, False, False, False, False),
        out_idx=[],
        target="riscv",
    )
    try:
        kernel(
            logits,
            bias,
            dummy_mask,
            dummy_fix_routing_mask,
            dummy_to_physical_map,
            dummy_logical_count,
            topk_idx,
            dummy_unmapped_topk_idx,
            topk_weights,
            0,
            1.0,
            0,
            1,
            0,
            1,
        )
    finally:
        kernel.close()

    torch.testing.assert_close(topk_idx, torch.tensor([[0]], dtype=torch.int64))
    torch.testing.assert_close(topk_weights, torch.tensor([[1.0]], dtype=torch.float32))
