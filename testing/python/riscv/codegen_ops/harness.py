from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalizeForRISCV, OptimizeForRISCV


def build_mlir_module(func=None, global_symbol: str = "kernel"):
    if func is None:
        func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0))
    func = func.with_attr("global_symbol", global_symbol)
    mod = tvm.IRModule({global_symbol: func})
    target = tvm.target.Target("riscv")
    return tvm.ffi.get_global_func("target.build.tilelang_riscv")(mod, target)


def build_mlir_from_source(source: str, global_symbol: str):
    func = tvm.script.from_source(source)
    return build_mlir_module(func, global_symbol=global_symbol)


def build_mlir_from_tilelang_prim(func, global_symbol: str):
    func = func.with_attr("global_symbol", global_symbol)
    mod = tvm.IRModule({global_symbol: func})
    target = tvm.target.Target("riscv")
    mod = LowerAndLegalizeForRISCV(mod, target)
    mod = OptimizeForRISCV(mod, target)
    return tvm.ffi.get_global_func("target.build.tilelang_riscv")(mod, target)


def real_mlir_source_or_skip(rt_mod, case_name: str) -> str:
    source = rt_mod.inspect_source()
    if "Placeholder MLIR module" in source:
        pytest.skip("vendored MLIR lowering is disabled in this build")
    dump_mlir(case_name, source)
    return source


def lower_source_to_mlir(source: str, global_symbol: str, case_name: str | None = None) -> str:
    return real_mlir_source_or_skip(
        build_mlir_from_source(source, global_symbol),
        case_name or global_symbol,
    )


def lower_tilelang_prim_to_mlir(func, global_symbol: str, case_name: str | None = None) -> str:
    return real_mlir_source_or_skip(
        build_mlir_from_tilelang_prim(func, global_symbol),
        case_name or global_symbol,
    )


def dump_mlir(case_name: str, source: str) -> None:
    dump_dir = os.environ.get("TILELANG_RISCV_DUMP_MLIR_DIR")
    if not dump_dir:
        return
    out_dir = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name).strip("_")
    if not safe_name:
        safe_name = "kernel"
    (out_dir / f"{safe_name}.mlir").write_text(source, encoding="utf-8")


def assert_contains_all(source: str, expected: Iterable[str]) -> None:
    for item in expected:
        assert item in source


def assert_contains_any(source: str, expected: Iterable[str]) -> None:
    expected = tuple(expected)
    assert any(item in source for item in expected), f"Expected one of {expected!r} in MLIR:\n{source}"


__all__ = [
    "T",
    "assert_contains_all",
    "assert_contains_any",
    "build_mlir_from_source",
    "build_mlir_from_tilelang_prim",
    "build_mlir_module",
    "lower_source_to_mlir",
    "lower_tilelang_prim_to_mlir",
    "real_mlir_source_or_skip",
]
