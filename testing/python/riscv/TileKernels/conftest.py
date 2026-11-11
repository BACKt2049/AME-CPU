from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
TILEKERNELS_ROOT = REPO_ROOT / "3rdparty" / "TileKernels"

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

for path in (REPO_ROOT, TILEKERNELS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _maybe_set_llvm_root() -> Path | None:
    candidates = [
        os.environ.get("TILELANG_RISCV_LLVM_ROOT"),
        "/root/llvm/build",
        "/home/liquanyi/tool/llvm",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "bin" / "mlir-translate").is_file():
            os.environ.setdefault("TILELANG_RISCV_LLVM_ROOT", str(root))
            os.environ.setdefault("TILELANG_RISCV_MLIR_MODE", "ON")
            return root
    return None


if not TILEKERNELS_ROOT.exists():
    pytest.skip("TileKernels checkout is not available", allow_module_level=True)

if _maybe_set_llvm_root() is None:
    pytest.skip("LLVM/MLIR runtime toolchain not available", allow_module_level=True)

pytest.importorskip("tilelang.tladapter._native")
