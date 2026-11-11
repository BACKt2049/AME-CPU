from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness import (
    assert_contains_all,
    build_mlir_module,
    lower_source_to_mlir,
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
        name="static_elem_offset_param",
        symbol="offset_load",
        source="""
# from tvm.script import tir as T
@T.prim_func
def offset_load(a: T.handle, B: T.Buffer((4,), "float32")):
    A = T.match_buffer(a, (4,), dtype="float32", elem_offset=2, offset_factor=1)
    for i in T.serial(4):
        B[i] = A[i]
""",
        expected=(
            "func.func @offset_load",
            "memref<4xf32, strided<[1], offset: 2>>",
            "memref<4xf32>",
            "memref.load",
            "memref.store",
        ),
    ),
    SourceCase(
        name="dynamic_elem_offset_param",
        symbol="dyn_elem_offset_load",
        source="""
# from tvm.script import tir as T
off = T.int32(is_size_var=True)
@T.prim_func
def dyn_elem_offset_load(a: T.handle, B: T.Buffer((4,), "float32")):
    A = T.match_buffer(a, (4,), dtype="float32", elem_offset=off, offset_factor=1)
    for i in T.serial(4):
        B[i] = A[i]
""",
        expected=(
            "func.func @dyn_elem_offset_load",
            "memref<4xf32, strided<[1], offset: ?>>",
            "memref<4xf32>",
            "memref.load",
            "memref.store",
        ),
    ),
    SourceCase(
        name="match_buffer_subview_offset",
        symbol="copy_sub",
        source="""
# from tvm.script import tir as T
@T.prim_func
def copy_sub(A: T.Buffer((8,), "float32"), B: T.Buffer((4,), "float32")):
    with T.block("root"):
        T.reads()
        T.writes()
        A0 = T.match_buffer(A[2:6], (4,), dtype="float32")
        for i in T.serial(4):
            with T.block("copy"):
                vi = T.axis.spatial(4, i)
                B[vi] = A0[vi]
""",
        expected=(
            "func.func @copy_sub",
            "memref.subview",
            "to memref<4xf32, strided<[1], offset: 2>>",
            "linalg.generic",
        ),
    ),
    SourceCase(
        name="match_buffer_dynamic_subview_offset",
        symbol="copy_sub_dyn",
        source="""
# from tvm.script import tir as T
@T.prim_func
def copy_sub_dyn(A: T.Buffer((16,), "float32"), B: T.Buffer((4,), "float32"), off: T.int32):
    with T.block("root"):
        T.reads()
        T.writes()
        A0 = T.match_buffer(A[off:off + 4], (4,), dtype="float32")
        for i in T.serial(4):
            with T.block("copy"):
                vi = T.axis.spatial(4, i)
                B[vi] = A0[vi]
""",
        expected=(
            "func.func @copy_sub_dyn",
            "memref.subview",
            "to memref<4xf32, strided<[1], offset: ?>>",
            "linalg.generic",
        ),
    ),
    SourceCase(
        name="static_strided_param",
        symbol="strided_load",
        source="""
# from tvm.script import tir as T
@T.prim_func
def strided_load(
    A: T.Buffer((4, 4), "float32", strides=(8, 2)),
    B: T.Buffer((4, 4), "float32"),
):
    for i, j in T.grid(4, 4):
        with T.block("copy"):
            vi = T.axis.spatial(4, i)
            vj = T.axis.spatial(4, j)
            B[vi, vj] = A[vi, vj]
""",
        expected=(
            "func.func @strided_load",
            "memref<4x4xf32, strided<[8, 2]>>",
            "memref<4x4xf32>",
            "linalg.generic",
        ),
    ),
    SourceCase(
        name="dynamic_strided_param",
        symbol="dyn_strided_load",
        source="""
# from tvm.script import tir as T
s0 = T.int32(is_size_var=True)
s1 = T.int32(is_size_var=True)
@T.prim_func
def dyn_strided_load(a: T.handle, B: T.Buffer((4, 4), "float32")):
    A = T.match_buffer(a, (4, 4), dtype="float32", strides=(s0, s1))
    for i, j in T.grid(4, 4):
        with T.block("copy"):
            vi = T.axis.spatial(4, i)
            vj = T.axis.spatial(4, j)
            B[vi, vj] = A[vi, vj]
""",
        expected=(
            "func.func @dyn_strided_load",
            "memref<4x4xf32, strided<[?, ?]>>",
            "memref<4x4xf32>",
            "linalg.generic",
        ),
    ),
    SourceCase(
        name="vector_slice_static_elem_offset_param",
        symbol="vec_slice_static_offset",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec_slice_static_offset(a: T.handle, B: T.Buffer((16,), "float32")):
    A = T.match_buffer(a, (16,), dtype="float32", elem_offset=4, offset_factor=1)
    for i in T.serial(4):
        idx = T.Ramp(i * 4, 1, 4)
        B[idx] = A[idx]
""",
        expected=(
            "func.func @vec_slice_static_offset",
            "memref<16xf32, strided<[1], offset: 4>>",
            "memref<16xf32>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="vector_slice_dynamic_elem_offset_param",
        symbol="vec_slice_dynamic_offset",
        source="""
# from tvm.script import tir as T
off = T.int32(is_size_var=True)
@T.prim_func
def vec_slice_dynamic_offset(a: T.handle, B: T.Buffer((16,), "float32")):
    A = T.match_buffer(a, (16,), dtype="float32", elem_offset=off, offset_factor=1)
    for i in T.serial(4):
        idx = T.Ramp(i * 4, 1, 4)
        B[idx] = A[idx]
""",
        expected=(
            "func.func @vec_slice_dynamic_offset",
            "memref<16xf32, strided<[1], offset: ?>>",
            "memref<16xf32>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="vector_slice_static_strided_param",
        symbol="vec_slice_static_strided",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec_slice_static_strided(
    A: T.Buffer((4, 16), "float32", strides=(32, 2)),
    B: T.Buffer((4, 16), "float32", strides=(64, 4)),
):
    for i, j in T.grid(4, 4):
        idx = T.Ramp(j * 4, 1, 4)
        B[i, idx] = A[i, idx]
""",
        expected=(
            "func.func @vec_slice_static_strided",
            "memref<4x16xf32, strided<[32, 2]>>",
            "memref<4x16xf32, strided<[64, 4]>>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="vector_slice_dynamic_strided_param",
        symbol="vec_slice_dynamic_strided",
        source="""
# from tvm.script import tir as T
s0 = T.int32(is_size_var=True)
s1 = T.int32(is_size_var=True)
t0 = T.int32(is_size_var=True)
t1 = T.int32(is_size_var=True)
@T.prim_func
def vec_slice_dynamic_strided(a: T.handle, b: T.handle):
    A = T.match_buffer(a, (4, 16), dtype="float32", strides=(s0, s1))
    B = T.match_buffer(b, (4, 16), dtype="float32", strides=(t0, t1))
    for i, j in T.grid(4, 4):
        idx = T.Ramp(j * 4, 1, 4)
        B[i, idx] = A[i, idx]
""",
        expected=(
            "func.func @vec_slice_dynamic_strided",
            "memref<4x16xf32, strided<[?, ?]>>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="vector_slice_match_buffer_subview_offset",
        symbol="vec_slice_subview",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec_slice_subview(A: T.Buffer((32,), "float32"), B: T.Buffer((16,), "float32")):
    with T.block("root"):
        T.reads()
        T.writes()
        A0 = T.match_buffer(A[8:24], (16,), dtype="float32")
        for i in T.serial(4):
            idx = T.Ramp(i * 4, 1, 4)
            B[idx] = A0[idx]
""",
        expected=(
            "func.func @vec_slice_subview",
            "memref.subview",
            "to memref<16xf32, strided<[1], offset: 8>>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="vector_slice_match_buffer_dynamic_subview_offset",
        symbol="vec_slice_subview_dyn",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec_slice_subview_dyn(A: T.Buffer((64,), "float32"), B: T.Buffer((16,), "float32"), off: T.int32):
    with T.block("root"):
        T.reads()
        T.writes()
        A0 = T.match_buffer(A[off:off + 16], (16,), dtype="float32")
        for i in T.serial(4):
            idx = T.Ramp(i * 4, 1, 4)
            B[idx] = A0[idx]
""",
        expected=(
            "func.func @vec_slice_subview_dyn",
            "memref.subview",
            "to memref<16xf32, strided<[1], offset: ?>>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_layout_codegen_ops(case: SourceCase):
    source = lower_source_to_mlir(case.source, case.symbol, f"layout/{case.name}")
    assert_contains_all(source, case.expected)


def test_strided_local_alloc_lowers():
    source = """
# from tvm.script import tir as T
@T.prim_func
def strided_local_alloc_probe(A: T.Buffer((8,), "float32"), B: T.Buffer((8,), "float32")):
    Tbuf = T.alloc_buffer((4,), "float32", strides=(2,))
    for i in T.serial(1):
        idx = T.Ramp(i * 4, 1, 4)
        Tbuf[idx] = A[idx]
        B[idx] = Tbuf[idx]
"""

    mlir = lower_source_to_mlir(source, "strided_local_alloc_probe", "layout/strided_local_alloc")
    assert_contains_all(
        mlir,
        (
            "func.func @strided_local_alloc_probe",
            "memref.alloca() : memref<4xf32, strided<[2]>>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    )


def test_static_elem_offset_local_alloc_lowers():
    source = """
# from tvm.script import tir as T
@T.prim_func
def offset_local_alloc_probe(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    Tbuf = T.alloc_buffer((4,), "float32", elem_offset=2, offset_factor=1)
    for i in T.serial(4):
        Tbuf[i] = A[i]
        B[i] = Tbuf[i]
"""

    mlir = lower_source_to_mlir(source, "offset_local_alloc_probe", "layout/offset_local_alloc")
    assert_contains_all(
        mlir,
        (
            "func.func @offset_local_alloc_probe",
            "memref.alloca() : memref<4xf32, strided<[1], offset: 2>>",
            "memref.load",
            "memref.store",
        ),
    )


def test_dynamic_strided_local_alloc_lowers():
    source = """
# from tvm.script import tir as T
@T.prim_func
def dynamic_strided_local_alloc_probe(
    s: T.int32,
    A: T.Buffer((8,), "float32"),
    B: T.Buffer((8,), "float32"),
):
    Tbuf = T.alloc_buffer((4,), "float32", strides=(s,))
    for i in T.serial(4):
        Tbuf[i] = A[i]
        B[i] = Tbuf[i]
"""

    mlir = lower_source_to_mlir(
        source,
        "dynamic_strided_local_alloc_probe",
        "layout/dynamic_strided_local_alloc",
    )
    assert_contains_all(
        mlir,
        (
            "func.func @dynamic_strided_local_alloc_probe",
            "memref.alloca()[%",
            "memref<4xf32, strided<[?]>>",
            "memref.load",
            "memref.store",
        ),
    )


@pytest.mark.parametrize("stride", (-1, 0))
def test_non_positive_static_stride_param_is_rejected(stride: int):
    source = f"""
# from tvm.script import tir as T
@T.prim_func
def non_positive_static_stride_param_probe(
    A: T.Buffer((4,), "float32", strides=({stride},)),
    B: T.Buffer((4,), "float32"),
):
    for i in T.serial(4):
        B[i] = A[i]
"""

    func = tvm.script.from_source(source, check_well_formed=False)
    with pytest.raises(Exception, match="Static buffer strides must be positive"):
        build_mlir_module(func, "non_positive_static_stride_param_probe")


def test_negative_elem_offset_param_is_rejected():
    source = """
# from tvm.script import tir as T
@T.prim_func
def negative_elem_offset_param_probe(a: T.handle, B: T.Buffer((4,), "float32")):
    A = T.match_buffer(a, (4,), dtype="float32", elem_offset=-1, offset_factor=1)
    for i in T.serial(4):
        B[i] = A[i]
"""

    func = tvm.script.from_source(source, check_well_formed=False)
    with pytest.raises(Exception, match="Negative elem_offset is not supported"):
        build_mlir_module(func, "negative_elem_offset_param_probe")


def test_negative_elem_offset_local_alloc_is_rejected():
    source = """
# from tvm.script import tir as T
@T.prim_func
def negative_elem_offset_local_alloc_probe(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
):
    Tbuf = T.alloc_buffer((4,), "float32", elem_offset=-1, offset_factor=1)
    for i in T.serial(4):
        Tbuf[i] = A[i]
        B[i] = Tbuf[i]
"""

    func = tvm.script.from_source(source, check_well_formed=False)
    with pytest.raises(Exception, match="Negative elem_offset is not supported"):
        build_mlir_module(func, "negative_elem_offset_local_alloc_probe")


def test_dynamic_elem_offset_local_alloc_lowers():
    source = """
# from tvm.script import tir as T
@T.prim_func
def dynamic_offset_local_alloc_probe(
    off: T.int32,
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
):
    Tbuf = T.alloc_buffer((4,), "float32", elem_offset=off, offset_factor=1)
    for i in T.serial(4):
        Tbuf[i] = A[i]
        B[i] = Tbuf[i]
"""

    mlir = lower_source_to_mlir(
        source,
        "dynamic_offset_local_alloc_probe",
        "layout/dynamic_offset_local_alloc",
    )
    assert_contains_all(
        mlir,
        (
            "func.func @dynamic_offset_local_alloc_probe",
            "memref.alloca()[%",
            "memref<4xf32, strided<[1], offset: ?>>",
            "memref.load",
            "memref.store",
        ),
    )


def test_dynamic_offset_and_strided_local_alloc_vector_slice_lowers():
    source = """
# from tvm.script import tir as T
@T.prim_func
def dynamic_offset_and_strided_local_alloc_vector_slice_probe(
    off: T.int32,
    s: T.int32,
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
):
    Tbuf = T.alloc_buffer((4,), "float32", elem_offset=off, strides=(s,), offset_factor=1)
    idx = T.Ramp(0, 1, 4)
    Tbuf[idx] = A[idx]
    B[idx] = Tbuf[idx]
"""

    mlir = lower_source_to_mlir(
        source,
        "dynamic_offset_and_strided_local_alloc_vector_slice_probe",
        "layout/dynamic_offset_and_strided_local_alloc_vector_slice",
    )
    assert_contains_all(
        mlir,
        (
            "func.func @dynamic_offset_and_strided_local_alloc_vector_slice_probe",
            "memref.alloca()[%",
            "memref<4xf32, strided<[?], offset: ?>>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    )


def test_unbound_dynamic_local_layout_var_is_rejected():
    source = """
# from tvm.script import tir as T
s = T.int32(is_size_var=True)
@T.prim_func
def unbound_dynamic_strided_local_alloc_probe(
    A: T.Buffer((8,), "float32"),
    B: T.Buffer((8,), "float32"),
):
    Tbuf = T.alloc_buffer((4,), "float32", strides=(s,))
    for i in T.serial(4):
        Tbuf[i] = A[i]
        B[i] = Tbuf[i]
"""

    func = tvm.script.from_source(source, check_well_formed=False)
    with pytest.raises(Exception, match="Dynamic local buffer stride.*unbound symbolic var"):
        build_mlir_module(func, "unbound_dynamic_strided_local_alloc_probe")
