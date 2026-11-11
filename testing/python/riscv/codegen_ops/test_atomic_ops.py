from __future__ import annotations

from dataclasses import dataclass

import pytest

import tilelang.language as T
from harness import (
    assert_contains_all,
    build_mlir_from_source,
    build_mlir_module,
    lower_source_to_mlir,
    lower_tilelang_prim_to_mlir,
)
from tilelang import tvm


@dataclass(frozen=True)
class SourceCase:
    name: str
    symbol: str
    source: str
    expected: tuple[str, ...]


CASES = (
    SourceCase(
        name="float_atomic_add",
        symbol="atomic_add_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_add_f32(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i, 1, 3),
            A[i],
        ))
""",
        expected=("func.func @atomic_add_f32", "memref.load", "arith.addf", "memref.store"),
    ),
    SourceCase(
        name="int_atomic_add",
        symbol="atomic_add_i32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_add_i32(A: T.Buffer((4,), "int32"), B: T.Buffer((4,), "int32")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("int32"), B.data, i, 1, 3),
            A[i],
        ))
""",
        expected=("func.func @atomic_add_i32", "memref.load", "arith.addi", "memref.store"),
    ),
    SourceCase(
        name="atomic_add_return_prev",
        symbol="atomic_add_ret_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_add_ret_f32(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    Old: T.Buffer((4,), "float32"),
):
    for i in T.serial(4):
        Old[i] = T.call_intrin(
            "float32",
            tvm.tir.op.Op.get("tl.atomic_add_ret_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i, 1, 3),
            A[i],
        )
""",
        expected=("func.func @atomic_add_ret_f32", "memref.load", "arith.addf", "memref.store"),
    ),
    SourceCase(
        name="float_atomic_max",
        symbol="atomic_max_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_max_f32(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_max_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i, 1, 3),
            A[i],
        ))
""",
        expected=(
            "func.func @atomic_max_f32",
            "memref.load",
            "arith.cmpf ogt",
            "arith.select",
            "memref.store",
        ),
    ),
    SourceCase(
        name="int_atomic_min",
        symbol="atomic_min_i32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_min_i32(A: T.Buffer((4,), "int32"), B: T.Buffer((4,), "int32")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_min_elem_op"),
            T.tvm_access_ptr(T.type_annotation("int32"), B.data, i, 1, 3),
            A[i],
        ))
""",
        expected=(
            "func.func @atomic_min_i32",
            "memref.load",
            "arith.cmpi slt",
            "arith.select",
            "memref.store",
        ),
    ),
    SourceCase(
        name="uint_atomic_min",
        symbol="atomic_min_u32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_min_u32(A: T.Buffer((4,), "uint32"), B: T.Buffer((4,), "uint32")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_min_elem_op"),
            T.tvm_access_ptr(T.type_annotation("uint32"), B.data, i, 1, 3),
            A[i],
        ))
""",
        expected=(
            "func.func @atomic_min_u32",
            "memref.load",
            "arith.cmpi ult",
            "arith.select",
            "memref.store",
        ),
    ),
    SourceCase(
        name="atomic_max_return_prev",
        symbol="atomic_max_ret_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_max_ret_f32(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    Old: T.Buffer((4,), "float32"),
):
    for i in T.serial(4):
        Old[i] = T.call_intrin(
            "float32",
            tvm.tir.op.Op.get("tl.atomic_max_ret_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i, 1, 3),
            A[i],
        )
""",
        expected=(
            "func.func @atomic_max_ret_f32",
            "memref.load",
            "arith.cmpf ogt",
            "arith.select",
            "memref.store",
        ),
    ),
    SourceCase(
        name="atomic_load_int",
        symbol="atomic_load_i32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_load_i32(A: T.Buffer((4,), "int32"), B: T.Buffer((4,), "int32")):
    for i in T.serial(4):
        B[i] = T.call_intrin(
            "int32",
            tvm.tir.op.Op.get("tl.atomic_load_elem_op"),
            T.tvm_access_ptr(T.type_annotation("int32"), A.data, i, 1, 1),
            5,
        )
""",
        expected=("func.func @atomic_load_i32", "memref.load", "memref.store"),
    ),
    SourceCase(
        name="atomic_store_float",
        symbol="atomic_store_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_store_f32(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_store_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i, 1, 3),
            A[i],
            5,
        ))
""",
        expected=("func.func @atomic_store_f32", "memref.load", "memref.store"),
    ),
    SourceCase(
        name="float16_atomic_addx2",
        symbol="atomic_addx2_f16",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_addx2_f16(A: T.Buffer((8,), "float16"), B: T.Buffer((8,), "float16")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_addx2_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float16"), B.data, i * 2, 2, 3),
            T.tvm_access_ptr(T.type_annotation("float16"), A.data, i * 2, 2, 1),
        ))
""",
        expected=("func.func @atomic_addx2_f16", "memref<8xf16>", "memref.load", "arith.addf", "memref.store"),
    ),
    SourceCase(
        name="float16_atomic_addx2_return_prev",
        symbol="atomic_addx2_ret_f16",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_addx2_ret_f16(
    A: T.Buffer((8,), "float16"),
    B: T.Buffer((8,), "float16"),
    Old: T.Buffer((4,), "float16x2"),
):
    for i in T.serial(4):
        Old[i] = T.call_intrin(
            "float16x2",
            tvm.tir.op.Op.get("tl.atomic_addx2_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float16"), B.data, i * 2, 2, 3),
            T.tvm_access_ptr(T.type_annotation("float16"), A.data, i * 2, 2, 1),
        )
""",
        expected=(
            "func.func @atomic_addx2_ret_f16",
            "memref<4xvector<2xf16>>",
            "vector.from_elements",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float32_atomic_addx4",
        symbol="atomic_addx4_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_addx4_f32(A: T.Buffer((8,), "float32"), B: T.Buffer((8,), "float32")):
    for i in T.serial(2):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_addx4_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * 4, 4, 3),
            T.tvm_access_ptr(T.type_annotation("float32"), A.data, i * 4, 4, 1),
        ))
""",
        expected=("func.func @atomic_addx4_f32", "memref<8xf32>", "memref.load", "arith.addf", "memref.store"),
    ),
    SourceCase(
        name="float32_atomic_addx4_return_prev",
        symbol="atomic_addx4_ret_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_addx4_ret_f32(
    A: T.Buffer((8,), "float32"),
    B: T.Buffer((8,), "float32"),
    Old: T.Buffer((2,), "float32x4"),
):
    for i in T.serial(2):
        Old[i] = T.call_intrin(
            "float32x4",
            tvm.tir.op.Op.get("tl.atomic_addx4_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * 4, 4, 3),
            T.tvm_access_ptr(T.type_annotation("float32"), A.data, i * 4, 4, 1),
        )
""",
        expected=(
            "func.func @atomic_addx4_ret_f32",
            "memref<2xvector<4xf32>>",
            "vector.from_elements",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="offset_float16_atomic_addx2",
        symbol="atomic_addx2_offset_f16",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_addx2_offset_f16(A: T.Buffer((8,), "float16"), b: T.handle):
    B = T.match_buffer(b, (8,), dtype="float16", elem_offset=4, offset_factor=1)
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_addx2_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float16"), B.data, i * 2 + 4, 2, 3),
            T.tvm_access_ptr(T.type_annotation("float16"), A.data, i * 2, 2, 1),
        ))
""",
        expected=(
            "func.func @atomic_addx2_offset_f16",
            "memref<8xf16, strided<[1], offset: 4>>",
            "arith.subi",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="dynamic_offset_float32_atomic_addx4",
        symbol="atomic_addx4_dynamic_offset_f32",
        source="""
# from tvm.script import tir as T
off = T.int32(is_size_var=True)
@T.prim_func
def atomic_addx4_dynamic_offset_f32(A: T.Buffer((8,), "float32"), b: T.handle):
    B = T.match_buffer(b, (8,), dtype="float32", elem_offset=off, offset_factor=1)
    for i in T.serial(2):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_addx4_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * 4 + off, 4, 3),
            T.tvm_access_ptr(T.type_annotation("float32"), A.data, i * 4, 4, 1),
        ))
""",
        expected=(
            "func.func @atomic_addx4_dynamic_offset_f32",
            "memref<8xf32, strided<[1], offset: ?>>",
            "memref.extract_strided_metadata",
            "arith.subi",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="offset_float_atomic_store",
        symbol="atomic_store_offset_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_store_offset_f32(A: T.Buffer((4,), "float32"), b: T.handle):
    B = T.match_buffer(b, (4,), dtype="float32", elem_offset=2, offset_factor=1)
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_store_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i + 2, 1, 3),
            A[i],
            5,
        ))
""",
        expected=(
            "func.func @atomic_store_offset_f32",
            "memref<4xf32, strided<[1], offset: 2>>",
            "arith.subi",
            "memref.load",
            "memref.store",
        ),
    ),
    SourceCase(
        name="strided_float_atomic_add",
        symbol="atomic_add_strided_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_add_strided_f32(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32", strides=(2,)),
):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * 2, 1, 3),
            A[i],
        ))
""",
        expected=(
            "func.func @atomic_add_strided_f32",
            "memref<4xf32, strided<[2]>>",
            "arith.divui",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="dynamic_strided_float_atomic_add",
        symbol="atomic_add_dynamic_strided_f32",
        source="""
# from tvm.script import tir as T
s0 = T.int32(is_size_var=True)
@T.prim_func
def atomic_add_dynamic_strided_f32(A: T.Buffer((4,), "float32"), b: T.handle):
    B = T.match_buffer(b, (4,), dtype="float32", strides=(s0,))
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * s0, 1, 3),
            A[i],
        ))
""",
        expected=(
            "func.func @atomic_add_dynamic_strided_f32",
            "memref<4xf32, strided<[?]>>",
            "memref.extract_strided_metadata",
            "arith.divui",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="offset_float_atomic_add",
        symbol="atomic_add_offset_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_add_offset_f32(A: T.Buffer((4,), "float32"), b: T.handle):
    B = T.match_buffer(b, (4,), dtype="float32", elem_offset=2, offset_factor=1)
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i + 2, 1, 3),
            A[i],
        ))
""",
        expected=(
            "func.func @atomic_add_offset_f32",
            "memref<4xf32, strided<[1], offset: 2>>",
            "arith.subi",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="dynamic_offset_float_atomic_add",
        symbol="atomic_add_dynamic_offset_f32",
        source="""
# from tvm.script import tir as T
off = T.int32(is_size_var=True)
@T.prim_func
def atomic_add_dynamic_offset_f32(A: T.Buffer((4,), "float32"), b: T.handle):
    B = T.match_buffer(b, (4,), dtype="float32", elem_offset=off, offset_factor=1)
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i + off, 1, 3),
            A[i],
        ))
""",
        expected=(
            "func.func @atomic_add_dynamic_offset_f32",
            "memref<4xf32, strided<[1], offset: ?>>",
            "memref.extract_strided_metadata",
            "arith.subi",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="strided_2d_float_atomic_add",
        symbol="atomic_add_strided_2d_f32",
        source="""
# from tvm.script import tir as T
@T.prim_func
def atomic_add_strided_2d_f32(
    A: T.Buffer((4, 4), "float32"),
    B: T.Buffer((4, 4), "float32", strides=(8, 2)),
):
    for i, j in T.grid(4, 4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * 8 + j * 2, 1, 3),
            A[i, j],
        ))
""",
        expected=(
            "func.func @atomic_add_strided_2d_f32",
            "memref<4x4xf32, strided<[8, 2]>>",
            "arith.divui",
            "arith.remui",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="dynamic_strided_2d_float_atomic_add",
        symbol="atomic_add_dynamic_strided_2d_f32",
        source="""
# from tvm.script import tir as T
s0 = T.int32(is_size_var=True)
s1 = T.int32(is_size_var=True)
@T.prim_func
def atomic_add_dynamic_strided_2d_f32(A: T.Buffer((4, 4), "float32"), b: T.handle):
    B = T.match_buffer(b, (4, 4), dtype="float32", strides=(s0, s1))
    for i, j in T.grid(4, 4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * s0 + j * s1, 1, 3),
            A[i, j],
        ))
""",
        expected=(
            "func.func @atomic_add_dynamic_strided_2d_f32",
            "memref<4x4xf32, strided<[?, ?]>>",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_atomic_codegen_ops(case: SourceCase):
    source = lower_source_to_mlir(case.source, case.symbol, f"atomic/{case.name}")
    assert_contains_all(source, case.expected)


def test_vector_atomic_bad_extent_is_rejected():
    source = """
# from tvm.script import tir as T
@T.prim_func
def atomic_addx2_probe(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_addx2_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i, 1, 3),
            T.tvm_access_ptr(T.type_annotation("float32"), A.data, i, 1, 1),
        ))
"""

    with pytest.raises(Exception, match="atomic_addx2 currently expects vector access_ptr extent=2"):
        build_mlir_from_source(source, "atomic_addx2_probe")


def test_tile_atomic_add_region_lowers():
    @T.prim_func
    def tile_atomic_add_region(
        A: T.Buffer((64, 64), "float16"),
        B: T.Buffer((1, 64, 1, 64), "float16"),
    ):
        T.atomic_add(B[0, 0:64, 0, 0:64], A[0:64, 0:64])

    source = lower_tilelang_prim_to_mlir(
        tile_atomic_add_region,
        "tile_atomic_add_region",
        "atomic/tile_atomic_add_region",
    )
    assert_contains_all(
        source,
        (
            "func.func @tile_atomic_add_region",
            "memref<64x64xf16>",
            "memref<1x64x1x64xf16>",
            "memref.load",
            "arith.addf",
            "memref.store",
        ),
    )


def test_bad_multi_dimensional_dynamic_stride_atomic_access_ptr_is_rejected():
    source = """
# from tvm.script import tir as T
s0 = T.int32(is_size_var=True)
s1 = T.int32(is_size_var=True)
@T.prim_func
def atomic_dyn_stride_probe(A: T.Buffer((4, 4), "float32"), b: T.handle):
    B = T.match_buffer(b, (4, 4), dtype="float32", strides=(s0, s1))
    for i, j in T.grid(4, 4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * s0 + j, 1, 3),
            A[i, j],
        ))
"""

    with pytest.raises(Exception, match="could not decode dynamic-strided tvm_access_ptr offset expression"):
        build_mlir_from_source(source, "atomic_dyn_stride_probe")


def test_complex_dynamic_elem_offset_atomic_access_ptr_is_rejected():
    source = """
# from tvm.script import tir as T
off = T.int32(is_size_var=True)
@T.prim_func
def atomic_complex_dyn_elem_offset_probe(A: T.Buffer((4,), "float32"), b: T.handle):
    B = T.match_buffer(b, (4,), dtype="float32", elem_offset=off + 1, offset_factor=1)
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i + off + 1, 1, 3),
            A[i],
        ))
"""

    func = tvm.script.from_source(source, check_well_formed=False)
    with pytest.raises(Exception, match="dynamic tvm_access_ptr elem_offset to be a direct variable"):
        build_mlir_module(func, "atomic_complex_dyn_elem_offset_probe")


def test_non_row_major_static_stride_atomic_access_ptr_is_rejected():
    source = """
# from tvm.script import tir as T
@T.prim_func
def atomic_non_row_major_stride_probe(
    A: T.Buffer((4, 4), "float32"),
    B: T.Buffer((4, 4), "float32", strides=(2, 8)),
):
    for i, j in T.grid(4, 4):
        T.evaluate(T.call_intrin(
            "handle",
            tvm.tir.op.Op.get("tl.atomic_add_elem_op"),
            T.tvm_access_ptr(T.type_annotation("float32"), B.data, i * 2 + j * 8, 1, 3),
            A[i, j],
        ))
"""

    with pytest.raises(Exception, match="row-major-like static strides"):
        build_mlir_from_source(source, "atomic_non_row_major_stride_probe")


__all__ = ["tvm"]
