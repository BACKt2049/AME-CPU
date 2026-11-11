import torch
import tilelang

from tile_kernels.quant.common import CastInputConfig, get_cast_output_config
from tile_kernels.quant.per_channel_cast_fused_kernel import get_per_channel_cast_fused_kernel
from tile_kernels.torch.per_channel_cast_fused import per_channel_cast_fused as cast_fused_ref

from ._quant_test_utils import assert_dequantized_close


# Original TileKernels source: tile_kernels/quant/per_channel_cast_fused_kernel.py
def test_per_channel_cast_fused_kernel_runtime_compare():
    x = (torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128) / 128 - 3).contiguous()
    expected_out, expected_sf = cast_fused_ref(x, 128, None, False, None)
    x_sf = torch.empty((128, 128), dtype=torch.float32)
    pos_to_token = torch.empty((128,), dtype=torch.int32)

    in_config = CastInputConfig(torch_dtype=torch.float32, with_sf=False, sf_block=(1, 1))
    out_config = get_cast_output_config("e4m3", (128, 1))
    kernel = tilelang.compile(
        get_per_channel_cast_fused_kernel.get_tir(128, False, in_config, out_config),
        out_idx=[1, 2],
        target="riscv",
    )
    try:
        actual_out, actual_sf = kernel(x, x_sf, pos_to_token)
    finally:
        kernel.close()

    assert_dequantized_close(actual_out, actual_sf, expected_out, expected_sf, (128, 1))
