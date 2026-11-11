import torch
import tilelang
import tilelang.language as T

from tile_kernels.quant.cast_back_e5m6_kernel import get_cast_back_e5m6_kernel
from tile_kernels.quant.common import CastInputConfig
from tile_kernels.torch.cast_e5m6 import cast_to_e5m6, cast_back_from_e5m6


# Original TileKernels source: tile_kernels/quant/cast_back_e5m6_kernel.py
def test_cast_back_e5m6_kernel_runtime_compare():
    x = torch.zeros((8, 1024), dtype=torch.float32)
    packed, sf = cast_to_e5m6(x, 1024)
    expected = cast_back_from_e5m6((packed, sf), "fp32", (1, 1024))

    in_config = CastInputConfig(torch_dtype=torch.uint8, with_sf=True, sf_block=(1, 1024))
    kernel = tilelang.compile(
        get_cast_back_e5m6_kernel.get_tir(1024, in_config, T.float32),
        out_idx=[2],
        target="riscv",
    )
    try:
        actual = kernel(packed.view(torch.uint32), sf)
    finally:
        kernel.close()

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.max(torch.abs(actual - expected)).item() == 0.0
