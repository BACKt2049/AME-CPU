import torch
import tilelang

from tile_kernels.mhc.post_kernel import _mhc_post_fwd


# Original TileKernels source: tile_kernels/mhc/post_kernel.py
def test_post_kernel_runtime_compare():
    x = torch.arange(1 * 8, dtype=torch.float32).reshape(1, 8).to(torch.bfloat16).contiguous()
    residual = (torch.arange(1 * 4 * 8, dtype=torch.float32).reshape(1, 4, 8) / 8).to(torch.bfloat16).contiguous()
    post_layer_mix = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32).contiguous()
    comb_res_mix = torch.arange(1 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4).contiguous() / 16
    expected = (post_layer_mix.unsqueeze(-1) * x.float().unsqueeze(1) + torch.einsum("amn,amc->anc", comb_res_mix, residual.float())).bfloat16()

    kernel = tilelang.compile(
        _mhc_post_fwd.get_tir(4, 8, 128, 8),
        out_idx=[4],
        target="riscv",
    )
    try:
        actual = kernel(comb_res_mix, residual, post_layer_mix, x)
    finally:
        kernel.close()

    torch.testing.assert_close(actual, expected)
