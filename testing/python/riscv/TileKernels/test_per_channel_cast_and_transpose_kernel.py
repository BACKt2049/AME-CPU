import torch
import tilelang
import tilelang.language as T

from tile_kernels.quant.common import get_cast_output_config
from tile_kernels.quant.per_channel_cast_and_transpose_kernel import get_per_channel_cast_and_transpose_kernel
from tile_kernels.torch.cast import cast as cast_ref

from ._quant_test_utils import assert_dequantized_close


# Original TileKernels source: tile_kernels/quant/per_channel_cast_and_transpose_kernel.py
def test_per_channel_cast_and_transpose_kernel_runtime_compare():
    x = (torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64) / 64 - 3).to(torch.bfloat16).contiguous()
    expected_out, expected_sf = cast_ref(x.T.contiguous(), "e4m3", block_size=(1, 128))

    out_config = get_cast_output_config("e4m3", (128, 1))
    kernel = tilelang.compile(
        get_per_channel_cast_and_transpose_kernel.get_tir(64, T.bfloat16, out_config),
        out_idx=[1, 2],
        target="riscv",
    )
    try:
        actual_out, actual_sf = kernel(x)
    finally:
        kernel.close()

    assert_dequantized_close(actual_out, actual_sf.T.contiguous(), expected_out, expected_sf, (1, 128))
