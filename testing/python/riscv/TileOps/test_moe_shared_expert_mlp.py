from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
)


def _load_shared_expert_mlp_module():
    _ensure_minimal_tileops_kernel_modules()
    if "tileops.kernels.gemm" not in sys.modules:
        gemm_stub = types.ModuleType("tileops.kernels.gemm")

        class GemmKernel:
            pass

        gemm_stub.GemmKernel = GemmKernel
        sys.modules["tileops.kernels.gemm"] = gemm_stub
    return _load_module(
        "tileops.kernels.moe.shared_expert_mlp",
        TILEOPS_ROOT / "tileops" / "kernels" / "moe" / "shared_expert_mlp.py",
    )


def test_moe_shared_expert_silu_mul_float32_runtime_compare():
    module = _load_shared_expert_mlp_module()
    kernel = module._silu_mul_fused_kernel(2, 4, "float32")(2, 4, 8)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    gate_up = torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32).reshape(2, 8)
    actual = kernel(gate_up)

    gate = gate_up[:, :4]
    up = gate_up[:, 4:]
    expected = gate * torch.sigmoid(gate) * up
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
