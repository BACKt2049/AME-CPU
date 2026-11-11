import torch
import tilelang

from tile_kernels.quant.common import CastInputConfig, get_cast_output_config
from tile_kernels.quant.swiglu_forward_and_per_token_cast_kernel import (
    get_swiglu_forward_and_per_token_cast_kernel,
)
from tile_kernels.torch.cast import cast as cast_ref
from tile_kernels.torch.swiglu import swiglu_forward

from ._quant_test_utils import assert_dequantized_close


# Original TileKernels source: tile_kernels/quant/swiglu_forward_and_per_token_cast_kernel.py
def test_swiglu_forward_and_per_token_cast_kernel_runtime_compare():
    x = torch.linspace(-2.0, 2.0, steps=32 * 256, dtype=torch.float32).reshape(32, 256).contiguous()
    expected_out, expected_sf = cast_ref(swiglu_forward(x), "e4m3", block_size=(1, 128))
    dummy_pos_to_token_topk = torch.empty((32,), dtype=torch.int32)
    dummy_topk_weights = torch.empty((1, 1), dtype=torch.float32)
    dummy_pos_to_expert = torch.empty((32,), dtype=torch.int32)
    dummy_clamped_count = torch.zeros((3,), dtype=torch.int64)

    in_config = CastInputConfig(torch_dtype=torch.float32, with_sf=False, sf_block=(1, 1))
    out_config = get_cast_output_config("e4m3", (1, 128))
    kernel = tilelang.compile(
        get_swiglu_forward_and_per_token_cast_kernel.get_tir(
            128,
            False,
            False,
            False,
            False,
            in_config.dtype,
            out_config,
            None,
        ),
        out_idx=[1, 2],
        target="riscv",
    )
    try:
        actual_out, actual_sf = kernel(
            x,
            dummy_pos_to_token_topk,
            dummy_topk_weights,
            dummy_pos_to_expert,
            dummy_clamped_count,
            0.0,
        )
    finally:
        kernel.close()

    assert_dequantized_close(actual_out, actual_sf, expected_out, expected_sf, (1, 128))
