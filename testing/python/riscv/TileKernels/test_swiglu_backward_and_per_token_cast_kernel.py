import torch
import tilelang

from tile_kernels.quant.common import get_cast_output_config
from tile_kernels.quant.swiglu_backward_and_per_token_cast_kernel import (
    get_swiglu_backward_and_per_token_cast_kernel,
)
from tile_kernels.torch.cast import cast as cast_ref
from tile_kernels.torch import swiglu as swiglu_ref

from ._quant_test_utils import assert_dequantized_close


# Original TileKernels source: tile_kernels/quant/swiglu_backward_and_per_token_cast_kernel.py
def test_swiglu_backward_and_per_token_cast_kernel_runtime_compare():
    swiglu_ref.elementwise_fma = lambda a, b, c: a * b + c
    x_fp32 = torch.linspace(-2.0, 2.0, steps=2 * 256, dtype=torch.float32).reshape(2, 256).contiguous()
    x, x_sf = cast_ref(x_fp32, "e4m3", block_size=(1, 128))
    grad_out = torch.ones((2, 128), dtype=torch.bfloat16)
    weight = torch.tensor([[0.5]], dtype=torch.float32)
    pos_to_token_topk = torch.tensor([0, 0], dtype=torch.int32)
    token_topk_to_pos = torch.tensor([[0]], dtype=torch.int32)
    out_config = get_cast_output_config("e4m3", (1, 128))

    expected_out, expected_x_grad, expected_weight_grad = swiglu_ref.swiglu_backward(
        (x, x_sf),
        grad_out,
        weight,
        pos_to_token_topk,
        token_topk_to_pos,
        128,
    )
    expected_x_grad_fp8, expected_x_grad_fp8_sf = cast_ref(expected_x_grad, "e4m3", block_size=(1, 128))

    kernel = tilelang.compile(
        get_swiglu_backward_and_per_token_cast_kernel.get_tir(
            128,
            out_config,
            False,
        ),
        out_idx=[6, 7, 8, 9, 10],
        target="riscv",
    )
    try:
        actual_out, actual_x_grad_fp8, actual_x_grad_fp8_sf, actual_x_grad, actual_weight_grad = kernel(
            x,
            x_sf,
            grad_out,
            weight,
            pos_to_token_topk,
            token_topk_to_pos,
            0.0,
        )
    finally:
        kernel.close()

    torch.testing.assert_close(actual_out.float(), expected_out.float(), rtol=5e-3, atol=2e-3)
    assert_dequantized_close(actual_x_grad_fp8, actual_x_grad_fp8_sf, expected_x_grad_fp8, expected_x_grad_fp8_sf, (1, 128))
    torch.testing.assert_close(actual_x_grad_fp8_sf, expected_x_grad_fp8_sf)
    torch.testing.assert_close(actual_x_grad.float(), expected_x_grad.float(), rtol=5e-3, atol=2e-3)
    torch.testing.assert_close(actual_weight_grad, expected_weight_grad)
