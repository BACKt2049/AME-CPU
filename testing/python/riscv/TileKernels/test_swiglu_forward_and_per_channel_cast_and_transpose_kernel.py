import torch
import tilelang
import tilelang.language as T

from tile_kernels.quant.common import get_cast_output_config
from tile_kernels.quant.swiglu_forward_and_per_channel_cast_and_transpose_kernel import (
    get_swiglu_forward_and_per_channel_cast_and_transpose_kernel,
)
from tile_kernels.torch.cast import cast as cast_ref
from tile_kernels.torch.swiglu import swiglu_forward

from ._quant_test_utils import assert_dequantized_close


# Original TileKernels source: tile_kernels/quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py
def test_swiglu_forward_and_per_channel_cast_and_transpose_kernel_runtime_compare():
    x = torch.linspace(-1.5, 1.5, steps=128 * 256, dtype=torch.float32).reshape(128, 256).to(torch.bfloat16).contiguous()
    expected_out, expected_sf = cast_ref(swiglu_forward(x).T.contiguous(), "e4m3", block_size=(128, 1))
    out_config = get_cast_output_config("e4m3", (128, 1))
    kernel = tilelang.compile(
        get_swiglu_forward_and_per_channel_cast_and_transpose_kernel.get_tir(
            128,
            False,
            False,
            T.bfloat16,
            out_config,
            0.0,
        ),
        out_idx=[1, 2],
        target="riscv",
    )
    try:
        actual_out, actual_sf = kernel(x)
    finally:
        kernel.close()

    assert_dequantized_close(actual_out, actual_sf, expected_out, expected_sf, (128, 1), atol=8e-2, rtol=8e-2)
