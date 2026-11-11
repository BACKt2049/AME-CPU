from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness import assert_contains_all, lower_source_to_mlir


@dataclass(frozen=True)
class SourceCase:
    name: str
    symbol: str
    source: str
    expected: tuple[str, ...]
    not_expected: tuple[str, ...] = ()


CASES = (
    SourceCase(
        name="reduce_sum",
        symbol="reduce_sum",
        source="""
# from tvm.script import tir as T
@T.prim_func
def reduce_sum(A: T.Buffer((4, 8), "float32"), B: T.Buffer((4,), "float32")):
    for i, k in T.grid(4, 8):
        with T.block("sum"):
            vi = T.axis.spatial(4, i)
            vk = T.axis.reduce(8, k)
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + A[vi, vk]
""",
        expected=("func.func @reduce_sum", "linalg.fill", "linalg.reduce", "arith.addf"),
    ),
    SourceCase(
        name="reduce_max",
        symbol="reduce_max",
        source="""
# from tvm.script import tir as T
@T.prim_func
def reduce_max(A: T.Buffer((4, 8), "float32"), B: T.Buffer((4,), "float32")):
    for i, k in T.grid(4, 8):
        with T.block("max"):
            vi = T.axis.spatial(4, i)
            vk = T.axis.reduce(8, k)
            with T.init():
                B[vi] = T.float32(-1.0e30)
            B[vi] = T.max(B[vi], A[vi, vk])
""",
        expected=("func.func @reduce_max", "linalg.fill", "linalg.reduce", "arith.select", "arith.cmpf ogt"),
    ),
    SourceCase(
        name="broadcast_normalize",
        symbol="normalize",
        source="""
# from tvm.script import tir as T
@T.prim_func
def normalize(
    A: T.Buffer((4, 8), "float32"),
    RowBias: T.Buffer((4,), "float32"),
    ColScale: T.Buffer((8,), "float32"),
    B: T.Buffer((4, 8), "float32"),
):
    for i, j in T.grid(4, 8):
        with T.block("normalize"):
            vi = T.axis.spatial(4, i)
            vj = T.axis.spatial(8, j)
            B[vi, vj] = (A[vi, vj] - RowBias[vi]) / ColScale[vj]
""",
        expected=(
            "func.func @normalize",
            "linalg.generic",
            "arith.subf",
            "arith.divf",
            "affine_map<(d0, d1) -> (d0)>",
            "affine_map<(d0, d1) -> (d1)>",
        ),
        not_expected=("scf.for",),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_reduction_codegen_ops(case: SourceCase):
    source = lower_source_to_mlir(case.source, case.symbol, f"reduction/{case.name}")
    assert_contains_all(source, case.expected)
    for item in case.not_expected:
        assert item not in source
