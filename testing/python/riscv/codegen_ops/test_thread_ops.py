from __future__ import annotations

from harness import T, assert_contains_all, lower_source_to_mlir, lower_tilelang_prim_to_mlir


def test_thread_binding_codegen_op():
    source = lower_source_to_mlir(
        """
# from tvm.script import tir as T
@T.prim_func
def thread_bound(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for tx in T.thread_binding(4, thread="threadIdx.x"):
        B[tx] = A[tx] + T.float32(1)
""",
        "thread_bound",
        "thread/thread_binding",
    )

    assert_contains_all(
        source,
        ("func.func @thread_bound", "arith.addf", "memref.load", "memref.store"),
    )


def test_block_and_thread_binding_codegen_op():
    source = lower_source_to_mlir(
        """
# from tvm.script import tir as T
@T.prim_func
def block_thread_bound(A: T.Buffer((8,), "float32"), B: T.Buffer((8,), "float32")):
    for bx in T.thread_binding(2, thread="blockIdx.x"):
        for tx in T.thread_binding(4, thread="threadIdx.x"):
            idx = bx * 4 + tx
            B[idx] = A[idx] + T.float32(1)
""",
        "block_thread_bound",
        "thread/block_thread_binding",
    )

    assert "func.func @block_thread_bound" in source
    assert source.count("scf.for") >= 1
    assert_contains_all(source, ("arith.addf", "memref.load", "memref.store"))


def test_dynamic_tilelang_block_launch_codegen_op():
    n = T.dynamic("n")

    @T.prim_func
    def tile_dynamic_block_launch(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, 4), threads=4) as (pid,):
            tid = T.get_thread_binding()
            idx = pid * 4 + tid
            if idx < n:
                B[idx] = A[idx] + T.float32(1)

    source = lower_tilelang_prim_to_mlir(
        tile_dynamic_block_launch,
        "tile_dynamic_block_launch",
        "thread/dynamic_block_launch",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_dynamic_block_launch(%arg0: memref<?xf32>, %arg1: memref<?xf32>)",
            "memref.dim",
            "arith.floordivsi",
            "arith.cmpi slt",
            "arith.addf",
        ),
    )
    assert source.count("scf.for") >= 2


def test_thread_index_helpers_lower_to_serialized_thread_arithmetic():
    @T.prim_func
    def thread_index_helper_probe(B: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=64):
            tid = T.get_thread_binding()
            lane = T.get_lane_idx()
            warp = T.get_warp_idx()
            warp_sync = T.get_warp_idx_sync(32)
            group = T.get_warp_group_idx(32, 2)
            B[tid] = lane + warp * 100 + warp_sync * 1000 + group * 10000

    source = lower_tilelang_prim_to_mlir(
        thread_index_helper_probe,
        "thread_index_helper_probe",
        "thread/thread_index_helpers",
    )

    assert_contains_all(
        source,
        (
            "func.func @thread_index_helper_probe",
            "scf.for",
            "arith.remui",
            "arith.divui",
            "arith.muli",
            "memref.store",
        ),
    )
    assert "tl.get_lane_idx" not in source
    assert "tl.get_warp_idx" not in source
    assert "tl.get_warp_group_idx" not in source


def test_thread_return_guard_codegen_op():
    @T.prim_func
    def tile_thread_return(
        A: T.Tensor((8,), "int32"),
        B: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            if tid >= 4:
                T.thread_return()
            B[tid] = A[tid] + 1

    source = lower_tilelang_prim_to_mlir(tile_thread_return, "tile_thread_return", "thread/thread_return")

    assert_contains_all(source, ("func.func @tile_thread_return", "scf.for", "scf.if", "arith.cmpi", "arith.addi"))


def test_tilelang_loop_break_codegen_op():
    @T.prim_func
    def tile_loop_break_probe(
        A: T.Tensor((8,), "int32"),
        B: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(8):
                if A[i] < 0:
                    T.loop_break()
                B[i] = A[i] + 1

    source = lower_tilelang_prim_to_mlir(
        tile_loop_break_probe,
        "tile_loop_break_probe",
        "thread/loop_break",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_loop_break_probe",
            "scf.for",
            "memref.alloca() : memref<1xi1>",
            "arith.cmpi slt",
            "arith.cmpi eq",
            "arith.addi",
            "memref.store",
        ),
    )
    assert source.count("scf.if") >= 3
    assert "tl.loop_break" not in source


def test_device_assert_codegen_op():
    @T.prim_func
    def device_assert_probe(
        A: T.Tensor((4,), "float32"),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(4):
                T.device_assert(A[i] >= T.float32(0), no_stack_info=True)
                B[i] = A[i] + T.float32(1)

    source = lower_tilelang_prim_to_mlir(device_assert_probe, "device_assert_probe", "thread/device_assert")

    assert_contains_all(
        source,
        ("func.func @device_assert_probe", "scf.for", "arith.addf", "memref.load", "memref.store"),
    )


def test_device_assert_with_msg_codegen_op():
    source = lower_source_to_mlir(
        """
# from tvm.script import tir as T
@T.prim_func
def device_assert_msg_probe(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        T.evaluate(T.call_intrin(
            "void",
            tvm.tir.op.Op.get("tl.device_assert_with_msg"),
            A[i] >= T.float32(0),
            "A must be non-negative",
        ))
        B[i] = A[i] + T.float32(1)
""",
        "device_assert_msg_probe",
        "thread/device_assert_with_msg",
    )

    assert_contains_all(
        source,
        (
            "func.func @device_assert_msg_probe",
            "scf.for",
            "arith.cmpf",
            "arith.addf",
            "memref.load",
            "memref.store",
        ),
    )


def test_assume_codegen_op():
    source = lower_source_to_mlir(
        """
# from tvm.script import tir as T
@T.prim_func
def assume_probe(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        T.assume(i < 4)
        B[i] = A[i] + T.float32(1)
""",
        "assume_probe",
        "thread/assume",
    )

    assert_contains_all(
        source,
        ("func.func @assume_probe", "arith.cmpi slt", "arith.addf", "memref.load", "memref.store"),
    )
    assert "tir.assume" not in source
    assert "tl.assume" not in source


def test_tilelang_assume_codegen_op():
    @T.prim_func
    def tilelang_assume_probe(
        A: T.Tensor((4,), "float32"),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(4):
                T.assume(i < 4)
                B[i] = A[i] + T.float32(1)

    source = lower_tilelang_prim_to_mlir(tilelang_assume_probe, "tilelang_assume_probe", "thread/tilelang_assume")

    assert_contains_all(
        source,
        (
            "func.func @tilelang_assume_probe",
            "arith.addf",
            "memref.load",
            "memref.store",
        ),
    )
    assert "tir.assume" not in source
    assert "tl.assume" not in source
