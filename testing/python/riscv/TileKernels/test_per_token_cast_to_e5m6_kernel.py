import torch
import tilelang

from tile_kernels.quant.common import CastInputConfig, get_cast_output_config
from tile_kernels.quant.per_token_cast_to_e5m6_kernel import get_per_token_cast_to_e5m6_kernel
from tile_kernels.torch.cast_e5m6 import cast_to_e5m6


# Original TileKernels source: tile_kernels/quant/per_token_cast_to_e5m6_kernel.py
def test_per_token_cast_to_e5m6_kernel_runtime_compare():
    x = torch.zeros((4, 1024), dtype=torch.float32)
    expected_bytes, expected_sf = cast_to_e5m6(x, 1024)

    actual_words = torch.empty((4, 1024 // 8 * 3), dtype=torch.uint32)
    actual_sf = torch.empty((4, 1), dtype=torch.float32)
    in_config = CastInputConfig(torch_dtype=torch.float32, with_sf=False)
    out_config = get_cast_output_config("e5m6", (1, 1024), custom_clamp_min_value=1e-4)

    kernel = tilelang.compile(
        get_per_token_cast_to_e5m6_kernel.get_tir(1024, 1024, in_config, out_config),
        out_idx=[],
        target="riscv",
    )
    try:
        kernel(x, actual_words, actual_sf)
    finally:
        kernel.close()

    torch.testing.assert_close(actual_words.view(torch.uint8), expected_bytes)
    torch.testing.assert_close(actual_sf, expected_sf)
