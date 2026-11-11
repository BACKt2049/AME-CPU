import torch
import tilelang
import tilelang.language as T

from tile_kernels.moe.reduce_fused_kernel import get_reduce_fused_kernel


def _reference_reduce_fused(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> torch.Tensor:
    num_tokens, num_topk = token_topk_to_pos.shape
    hidden = x.shape[1]
    reduced = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x.device)
    for token in range(num_tokens):
        for k in range(num_topk):
            pos = int(token_topk_to_pos[token, k])
            if pos >= 0:
                reduced[token] += x[pos].float() * topk_weights[token, k]
    return reduced


# Original TileKernels source: tile_kernels/moe/reduce_fused_kernel.py
def test_reduce_fused_kernel_runtime_compare():
    expanded = (torch.arange(3 * 256, dtype=torch.float32).reshape(3, 256) / 32).contiguous()
    token_topk_to_pos = torch.tensor([[0, 2], [-1, 1], [2, -1]], dtype=torch.int32)
    topk_weights = torch.tensor(
        [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    dummy_sf = torch.ones(1, dtype=torch.float32)
    dummy_x_sf = torch.ones(expanded.shape[0], dtype=torch.float32)
    expected = _reference_reduce_fused(expanded, topk_weights, token_topk_to_pos)

    kernel = tilelang.compile(
        get_reduce_fused_kernel.get_tir(256, 2, T.float32, T.float32, False, True, False),
        out_idx=[3],
        target="riscv",
    )
    try:
        actual = kernel(expanded, topk_weights, token_topk_to_pos, dummy_sf, dummy_x_sf)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
