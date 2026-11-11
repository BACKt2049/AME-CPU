import torch
import tilelang

from tile_kernels.mhc.multilayer_recompute_kernel import _mhc_multilayer_recompute_kernel


# Original TileKernels source: tile_kernels/mhc/multilayer_recompute_kernel.py
def test_multilayer_recompute_kernel_runtime_compare():
    mhc_mult = 2
    hidden = 64
    num_layers = 1
    num_post = 0
    num_tokens = 1

    initial_residual = torch.linspace(-1.0, 1.0, steps=num_tokens * mhc_mult * hidden, dtype=torch.float32).reshape(
        num_tokens, mhc_mult, hidden
    ).to(torch.bfloat16)
    pre_mix = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    layer_input = torch.empty((num_tokens, hidden), dtype=torch.bfloat16)

    pre_mix_ptrs = torch.tensor([pre_mix.data_ptr()], dtype=torch.int64)
    layer_input_ptrs = torch.tensor([layer_input.data_ptr()], dtype=torch.int64)
    empty_ptrs = torch.empty((0,), dtype=torch.int64)

    expected_layer_input = (initial_residual.float() * pre_mix.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)

    kernel = tilelang.compile(
        _mhc_multilayer_recompute_kernel.get_tir(mhc_mult, hidden, num_layers, num_post),
        out_idx=[],
        target="riscv",
    )
    try:
        kernel(
            initial_residual,
            pre_mix_ptrs,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            layer_input_ptrs,
            empty_ptrs,
        )
    finally:
        kernel.close()

    torch.testing.assert_close(layer_input, expected_layer_input)
