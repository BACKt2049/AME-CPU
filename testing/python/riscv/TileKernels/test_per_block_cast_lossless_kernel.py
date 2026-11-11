import torch
import tilelang

from tile_kernels.quant.common import CastInputConfig, get_cast_output_config
from tile_kernels.quant.per_block_cast_lossless_kernel import get_per_block_cast_lossless_kernel
from tile_kernels.torch.cast import cast as cast_ref

from ._quant_test_utils import assert_e4m3_matches_e2m1_close, unpack_fp4_e2m1_x2


# Original TileKernels source: tile_kernels/quant/per_block_cast_lossless_kernel.py
def test_per_block_cast_lossless_kernel_runtime_compare():
    x = (torch.arange(32 * 256, dtype=torch.float32).reshape(32, 256) / 64 - 2).contiguous()
    x_fp4, x_sf = cast_ref(x, "e2m1", block_size=(1, 32), round_sf=True)
    x_fp4_unpacked = torch.from_numpy(unpack_fp4_e2m1_x2(x_fp4).astype("int8"))

    in_config = CastInputConfig(torch_dtype=torch.int8, with_sf=True, sf_block=(1, 32))
    out_config = get_cast_output_config("e4m3", (32, 32))
    kernel = tilelang.compile(
        get_per_block_cast_lossless_kernel.get_tir(256, 256, in_config, out_config),
        out_idx=[2, 3],
        target="riscv",
    )
    try:
        actual_out, actual_sf = kernel(x_fp4_unpacked, x_sf)
    finally:
        kernel.close()

    assert_e4m3_matches_e2m1_close(
        actual_out,
        actual_sf,
        x_fp4,
        x_sf,
        actual_block_size=(32, 32),
        expected_block_size=(1, 32),
        atol=1e-2,
        rtol=1e-2,
    )
