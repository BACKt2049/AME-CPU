import torch
import tilelang

from tile_kernels.quant.common import CastInputConfig, get_cast_output_config
import tile_kernels.quant.per_block_cast_kernel as per_block_mod
from tile_kernels.quant.per_block_cast_kernel import get_per_block_cast_kernel
from tile_kernels.torch.cast import cast as cast_ref

from ._quant_test_utils import assert_fp4_dequantized_close


# Original TileKernels source: tile_kernels/quant/per_block_cast_kernel.py
def test_per_block_cast_kernel_runtime_compare():
    x = (torch.arange(32 * 256, dtype=torch.float32).reshape(32, 256) / 64 - 3).contiguous()
    expected_out, expected_sf = cast_ref(x, "e2m1", block_size=(32, 32))

    in_config = CastInputConfig(torch_dtype=torch.float32, with_sf=False, sf_block=(1, 1))
    out_config = get_cast_output_config("e2m1", (32, 32))
    old_get_best_vectorize_size = per_block_mod.get_best_vectorize_size
    per_block_mod.get_best_vectorize_size = lambda dtype: 4
    kernel = tilelang.compile(
        get_per_block_cast_kernel.get_tir(256, in_config, out_config),
        out_idx=[1, 2],
        target="riscv",
    )
    try:
        actual_out, actual_sf = kernel(x)
    finally:
        kernel.close()
        per_block_mod.get_best_vectorize_size = old_get_best_vectorize_size

    assert_fp4_dequantized_close(actual_out, actual_sf, expected_out, expected_sf, (32, 32), atol=1e-2, rtol=1e-2)
