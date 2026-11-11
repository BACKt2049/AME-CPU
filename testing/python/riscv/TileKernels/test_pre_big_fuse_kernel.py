import torch
import tilelang

from tile_kernels.mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse
from tile_kernels.torch.mhc import sinkhorn_normalize_ref


# Original TileKernels source: tile_kernels/mhc/pre_big_fuse_kernel.py
def test_pre_big_fuse_kernel_runtime_compare():
    hidden = 8
    mhc = 4
    mhc3 = mhc * (2 + mhc)
    rms_eps = 1e-5
    pre_eps = 1e-6
    sinkhorn_eps = 1e-6
    post_mult = 2.0
    repeat = 2

    gemm_out_mul = (torch.arange(1 * 2 * mhc3, dtype=torch.float32).reshape(1, 2, mhc3) / 16).contiguous()
    gemm_out_sqrsum = torch.tensor([[4.0, 9.0]], dtype=torch.float32)
    mhc_scale = torch.tensor([0.5, 1.5, 2.0], dtype=torch.float32)
    mhc_base = torch.tensor([0.1 * i for i in range(mhc3)], dtype=torch.float32)
    residual = (torch.arange(2 * mhc * hidden, dtype=torch.float32).reshape(2, mhc, hidden) / 8).to(torch.bfloat16).contiguous()

    rms = torch.rsqrt(gemm_out_sqrsum.sum(dim=0) / (mhc * hidden) + rms_eps).unsqueeze(-1)
    mixes = gemm_out_mul.sum(dim=0) * rms
    expected_post = torch.sigmoid(mixes[:, mhc : 2 * mhc] * mhc_scale[1] + mhc_base[mhc : 2 * mhc]) * post_mult
    comb_raw = (mixes[:, 2 * mhc :] * mhc_scale[2] + mhc_base[2 * mhc :]).reshape(2, mhc, mhc)
    expected_comb = sinkhorn_normalize_ref(comb_raw, repeat=repeat, eps=sinkhorn_eps).reshape(2, mhc * mhc)
    pre_mix = torch.sigmoid(mixes[:, :mhc] * mhc_scale[0] + mhc_base[:mhc]) + pre_eps
    expected_layer_input = (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).bfloat16()

    kernel = tilelang.compile(
        _mhc_pre_big_fuse.get_tir(hidden, rms_eps, pre_eps, sinkhorn_eps, post_mult, repeat, 1, mhc),
        out_idx=[5, 6, 7],
        target="riscv",
    )
    try:
        actual_post, actual_comb, actual_layer_input = kernel(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
        )
    finally:
        kernel.close()

    torch.testing.assert_close(actual_post, expected_post)
    torch.testing.assert_close(actual_comb, expected_comb)
    torch.testing.assert_close(actual_layer_input, expected_layer_input)
