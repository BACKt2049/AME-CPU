import torch
import tilelang
import tilelang.language as T
import numpy as np

from tile_kernels.quant.cast_back_kernel import get_cast_back_kernel
from tile_kernels.quant.common import CastInputConfig
from tile_kernels.torch.cast import cast as cast_ref, cast_back as cast_back_ref


# Original TileKernels source: tile_kernels/quant/cast_back_kernel.py
def test_cast_back_kernel_runtime_compare():
    x = (torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128) / 128 - 64).contiguous()
    quant, sf = cast_ref(x, "e4m3", block_size=(1, 128))
    expected = cast_back_ref((quant, sf), "fp32", block_size=(1, 128))

    in_config = CastInputConfig(torch_dtype=torch.float8_e4m3fn, with_sf=True, sf_block=(1, 128))
    kernel = tilelang.compile(
        get_cast_back_kernel.get_tir(128, in_config, T.float32),
        out_idx=[2],
        target="riscv",
    )
    try:
        actual = kernel(quant, sf)
    finally:
        kernel.close()

    actual_np = actual.detach().numpy()
    expected_np = expected.detach().numpy()
    is_finite = bool(np.isfinite(actual_np).all())
    max_abs = float(np.max(np.abs(actual_np - expected_np)))
    assert is_finite, "RISC-V cast_back produced non-finite values"
    assert max_abs <= 1e-3, f"max_abs={max_abs}"
