from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness import T, assert_contains_all, lower_source_to_mlir, lower_tilelang_prim_to_mlir


@dataclass(frozen=True)
class SourceCase:
    name: str
    symbol: str
    source: str
    expected: tuple[str, ...]


CASES = (
    SourceCase(
        name="elementwise_add",
        symbol="add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def add(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        with T.block("add"):
            vi = T.axis.spatial(4, i)
            C[vi] = A[vi] + B[vi]
""",
        expected=(
            "func.func @add(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)",
            "linalg.generic",
            "arith.addf",
        ),
    ),
    SourceCase(
        name="unary_math",
        symbol="unary_math",
        source="""
# from tvm.script import tir as T
@T.prim_func
def unary_math(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    C: T.Buffer((4,), "float32"),
    D: T.Buffer((4,), "float32"),
    E: T.Buffer((4,), "float32"),
):
    for i in T.serial(4):
        with T.block("sqrt"):
            vi = T.axis.spatial(4, i)
            B[vi] = T.sqrt(A[vi])
    for i in T.serial(4):
        with T.block("rsqrt"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.rsqrt(A[vi])
    for i in T.serial(4):
        with T.block("exp"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.exp(A[vi])
    for i in T.serial(4):
        with T.block("log1p"):
            vi = T.axis.spatial(4, i)
            E[vi] = T.log1p(A[vi])
""",
        expected=("func.func @unary_math", "math.sqrt", "math.rsqrt", "math.exp", "math.log1p"),
    ),
    SourceCase(
        name="extended_unary_math",
        symbol="extended_unary_math",
        source="""
# from tvm.script import tir as T
@T.prim_func
def extended_unary_math(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    C: T.Buffer((4,), "float32"),
    D: T.Buffer((4,), "float32"),
    E: T.Buffer((4,), "float32"),
    F: T.Buffer((4,), "float32"),
):
    for i in T.serial(4):
        with T.block("exp2"):
            vi = T.axis.spatial(4, i)
            B[vi] = T.exp2(A[vi])
    for i in T.serial(4):
        with T.block("log2"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.log2(A[vi])
    for i in T.serial(4):
        with T.block("log"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.log(A[vi] + T.float32(1))
    for i in T.serial(4):
        with T.block("sigmoid"):
            vi = T.axis.spatial(4, i)
            E[vi] = T.sigmoid(A[vi])
    for i in T.serial(4):
        with T.block("abs"):
            vi = T.axis.spatial(4, i)
            F[vi] = T.abs(A[vi])
""",
        expected=(
            "func.func @extended_unary_math",
            "math.exp2",
            "math.log2",
            "math.log",
            "arith.divf",
            "math.absf",
        ),
    ),
    SourceCase(
        name="advanced_math_intrinsics",
        symbol="advanced_math_intrinsics",
        source="""
# from tvm.script import tir as T
@T.prim_func
def advanced_math_intrinsics(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    C: T.Buffer((4,), "float32"),
    D: T.Buffer((4,), "float32"),
    E: T.Buffer((4,), "float32"),
    F: T.Buffer((4,), "float32"),
    G: T.Buffer((4,), "float32"),
    H: T.Buffer((4,), "float32"),
    I: T.Buffer((4,), "float32"),
    J: T.Buffer((4,), "float32"),
    K: T.Buffer((4,), "float32"),
    L: T.Buffer((4,), "float32"),
):
    for i in T.serial(4):
        with T.block("tanh"):
            vi = T.axis.spatial(4, i)
            B[vi] = T.tanh(A[vi])
    for i in T.serial(4):
        with T.block("ceil"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.ceil(A[vi])
    for i in T.serial(4):
        with T.block("floor"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.floor(A[vi])
    for i in T.serial(4):
        with T.block("sin"):
            vi = T.axis.spatial(4, i)
            E[vi] = T.sin(A[vi])
    for i in T.serial(4):
        with T.block("cos"):
            vi = T.axis.spatial(4, i)
            F[vi] = T.cos(A[vi])
    for i in T.serial(4):
        with T.block("erf"):
            vi = T.axis.spatial(4, i)
            G[vi] = T.erf(A[vi])
    for i in T.serial(4):
        with T.block("trunc"):
            vi = T.axis.spatial(4, i)
            H[vi] = T.trunc(A[vi])
    for i in T.serial(4):
        with T.block("nearbyint"):
            vi = T.axis.spatial(4, i)
            I[vi] = T.nearbyint(A[vi])
    for i in T.serial(4):
        with T.block("pow"):
            vi = T.axis.spatial(4, i)
            J[vi] = T.pow(A[vi], B[vi])
    for i in T.serial(4):
        with T.block("round"):
            vi = T.axis.spatial(4, i)
            K[vi] = T.round(A[vi])
    for i in T.serial(4):
        with T.block("fmod"):
            vi = T.axis.spatial(4, i)
            L[vi] = T.fmod(A[vi], B[vi])
""",
        expected=(
            "func.func @advanced_math_intrinsics",
            "math.tanh",
            "math.ceil",
            "math.floor",
            "math.sin",
            "math.cos",
            "math.erf",
            "math.trunc",
            "math.roundeven",
            "math.powf",
            "arith.remf",
        ),
    ),
    SourceCase(
        name="bitwise_intrinsics",
        symbol="bitwise_intrinsics",
        source="""
# from tvm.script import tir as T
@T.prim_func
def bitwise_intrinsics(
    A: T.Buffer((4,), "uint32"),
    B: T.Buffer((4,), "uint32"),
    C: T.Buffer((4,), "uint32"),
    D: T.Buffer((4,), "uint32"),
    E: T.Buffer((4,), "uint32"),
    F: T.Buffer((4,), "uint32"),
    G: T.Buffer((4,), "uint32"),
):
    for i in T.serial(4):
        with T.block("xor"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.bitwise_xor(A[vi], B[vi])
    for i in T.serial(4):
        with T.block("shift"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.shift_left(A[vi] & B[vi], T.uint32(1))
    for i in T.serial(4):
        with T.block("or"):
            vi = T.axis.spatial(4, i)
            E[vi] = A[vi] | B[vi]
    for i in T.serial(4):
        with T.block("shift_right"):
            vi = T.axis.spatial(4, i)
            F[vi] = T.shift_right(A[vi], T.uint32(2))
    for i in T.serial(4):
        with T.block("not"):
            vi = T.axis.spatial(4, i)
            G[vi] = T.bitwise_not(A[vi])
""",
        expected=(
            "func.func @bitwise_intrinsics",
            "arith.xori",
            "arith.andi",
            "arith.shli",
            "arith.ori",
            "arith.shrui",
            "arith.xori",
        ),
    ),
    SourceCase(
        name="scalar_intrinsic_helpers",
        symbol="scalar_intrinsic_helpers",
        source="""
# from tvm.script import tir as T
@T.prim_func
def scalar_intrinsic_helpers(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    C: T.Buffer((4,), "uint32"),
    D: T.Buffer((4,), "uint32"),
    E: T.Buffer((4,), "int32"),
    F: T.Buffer((4,), "float32"),
    G: T.Buffer((4,), "uint32"),
):
    for i in T.serial(4):
        with T.block("isfinite"):
            vi = T.axis.spatial(4, i)
            E[vi] = T.Select(T.isfinite(A[vi]), T.int32(1), T.int32(0))
    for i in T.serial(4):
        with T.block("popcount"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.popcount(C[vi])
    for i in T.serial(4):
        with T.block("copysign"):
            vi = T.axis.spatial(4, i)
            F[vi] = T.copysign(A[vi], B[vi])
    for i in T.serial(4):
        with T.block("reinterpret"):
            vi = T.axis.spatial(4, i)
            G[vi] = T.reinterpret(A[vi], dtype="uint32")
""",
        expected=(
            "func.func @scalar_intrinsic_helpers",
            "math.absf",
            "arith.cmpf one",
            "math.ctpop",
            "math.copysign",
            "arith.bitcast",
        ),
    ),
    SourceCase(
        name="nan_inf_intrinsic_helpers",
        symbol="nan_inf_intrinsic_helpers",
        source="""
# from tvm.script import tir as T
@T.prim_func
def nan_inf_intrinsic_helpers(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    C: T.Buffer((4,), "int32"),
    D: T.Buffer((4,), "int32"),
):
    for i in T.serial(4):
        with T.block("isnan"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.Select(T.isnan(A[vi]), T.int32(1), T.int32(0))
    for i in T.serial(4):
        with T.block("isinf"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.Select(T.isinf(B[vi]), T.int32(1), T.int32(0))
""",
        expected=(
            "func.func @nan_inf_intrinsic_helpers",
            "arith.cmpf uno",
            "math.absf",
            "arith.cmpf oeq",
        ),
    ),
    SourceCase(
        name="integer_div_helpers",
        symbol="integer_div_helpers",
        source="""
# from tvm.script import tir as T
@T.prim_func
def integer_div_helpers(
    A: T.Buffer((4,), "int32"),
    B: T.Buffer((4,), "int32"),
    C: T.Buffer((4,), "int32"),
    D: T.Buffer((4,), "int32"),
):
    for i in T.serial(4):
        with T.block("ceildiv"):
            vi = T.axis.spatial(4, i)
            B[vi] = T.ceildiv(A[vi], 3)
    for i in T.serial(4):
        with T.block("truncdiv"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.truncdiv(A[vi], 3)
    for i in T.serial(4):
        with T.block("truncmod"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.truncmod(A[vi], 3)
""",
        expected=(
            "func.func @integer_div_helpers",
            "arith.floordivsi",
            "arith.divsi",
            "arith.remsi",
        ),
    ),
    SourceCase(
        name="numeric_limits",
        symbol="numeric_limits",
        source="""
# from tvm.script import tir as T
@T.prim_func
def numeric_limits(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    C: T.Buffer((4,), "float32"),
    D: T.Buffer((4,), "int32"),
    E: T.Buffer((4,), "int32"),
):
    for i in T.serial(4):
        with T.block("float_limits"):
            vi = T.axis.spatial(4, i)
            A[vi] = T.infinity("float32")
            B[vi] = T.max_value("float32")
            C[vi] = T.min_value("float32")
    for i in T.serial(4):
        with T.block("int_limits"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.max_value("int32")
            E[vi] = T.min_value("int32")
""",
        expected=(
            "func.func @numeric_limits",
            "arith.constant 0x7F800000 : f32",
            "arith.constant 3.40282347E+38 : f32",
            "arith.constant -3.40282347E+38 : f32",
            "arith.constant 2147483647 : i32",
            "arith.constant -2147483648 : i32",
        ),
    ),
    SourceCase(
        name="if_then_else",
        symbol="if_then_else",
        source="""
# from tvm.script import tir as T
@T.prim_func
def if_then_else(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        with T.block("select"):
            vi = T.axis.spatial(4, i)
            B[vi] = T.if_then_else(vi < 2, A[vi], T.float32(0))
""",
        expected=("func.func @if_then_else", "scf.if", "arith.cmpi slt"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_elementwise_codegen_ops(case: SourceCase):
    source = lower_source_to_mlir(case.source, case.symbol, f"elementwise/{case.name}")
    assert_contains_all(source, case.expected)


def test_tilelang_clamp_codegen_op():
    @T.prim_func
    def tilelang_clamp(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            for i in T.serial(4):
                B[i] = T.clamp(A[i], T.float32(0), T.float32(1))

    source = lower_tilelang_prim_to_mlir(
        tilelang_clamp,
        "tilelang_clamp",
        "elementwise/tilelang_clamp",
    )
    assert_contains_all(
        source,
        (
            "func.func @tilelang_clamp",
            "arith.cmpf ogt",
            "arith.cmpf olt",
            "arith.select",
        ),
    )


def test_tilelang_infinity_call_codegen_op():
    @T.prim_func
    def tilelang_infinity(A: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            A[0] = T.infinity(T.float32)

    source = lower_tilelang_prim_to_mlir(
        tilelang_infinity,
        "tilelang_infinity",
        "elementwise/tilelang_infinity",
    )
    assert_contains_all(
        source,
        (
            "func.func @tilelang_infinity",
            "arith.constant 0x7F800000 : f32",
        ),
    )
