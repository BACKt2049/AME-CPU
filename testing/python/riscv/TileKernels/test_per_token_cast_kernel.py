import torch
import tilelang

from tile_kernels.quant.common import CastInputConfig, get_cast_output_config
from tile_kernels.quant.per_token_cast_kernel import get_per_token_cast_kernel
from tile_kernels.torch.cast import cast as cast_ref

from ._quant_test_utils import assert_fp4_dequantized_close


# Original TileKernels source: tile_kernels/quant/per_token_cast_kernel.py
def test_per_token_cast_kernel_runtime_compare():
    x = (torch.arange(32 * 128, dtype=torch.float32).reshape(32, 128) / 64 - 2).contiguous()
    expected_out, expected_sf = cast_ref(x, "e2m1", block_size=(1, 32))
    dummy_x_sf = torch.empty((32, 128), dtype=torch.float32)

    in_config = CastInputConfig(torch_dtype=torch.float32, with_sf=False, sf_block=(1, 1))
    out_config = get_cast_output_config("e2m1", (1, 32))
    kernel = tilelang.compile(
        get_per_token_cast_kernel.get_tir(128, 128, in_config, out_config),
        out_idx=[2, 3],
        target="riscv",
    )
    try:
        actual_out, actual_sf = kernel(x, dummy_x_sf)
    finally:
        kernel.close()

    assert_fp4_dequantized_close(actual_out, actual_sf, expected_out, expected_sf, (1, 32), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_sf, expected_sf)
