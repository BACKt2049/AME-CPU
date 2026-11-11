from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness import (
    T,
    build_mlir_from_source,
    build_mlir_from_tilelang_prim,
    lower_source_to_mlir,
    lower_tilelang_prim_to_mlir,
)


@dataclass(frozen=True)
class RejectionCase:
    name: str
    symbol: str
    source: str
    expected: str


CASES = (
    RejectionCase(
        name="handle_reinterpret",
        symbol="handle_reinterpret_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def handle_reinterpret_probe(A: T.Buffer((1,), "int64")):
    T.evaluate(T.reinterpret("handle", A[0]))
""",
        expected="Unsupported tir.reinterpret.*int64.*handle",
    ),
    RejectionCase(
        name="ptx_cp_async",
        symbol="ptx_cp_async_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def ptx_cp_async_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.call_intrin("handle", tvm.tir.op.Op.get("tl.ptx_cp_async")))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.ptx_cp_async",
    ),
    RejectionCase(
        name="tma_load",
        symbol="tma_load_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def tma_load_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.call_intrin("handle", tvm.tir.op.Op.get("tl.tma_load")))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.tma_load",
    ),
    RejectionCase(
        name="create_tma_descriptor",
        symbol="create_tma_descriptor_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def create_tma_descriptor_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.call_intrin("handle", tvm.tir.op.Op.get("tl.create_tma_descriptor")))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.create_tma_descriptor",
    ),
    RejectionCase(
        name="tcgen05_st",
        symbol="tcgen05_st_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def tcgen05_st_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.call_intrin("handle", tvm.tir.op.Op.get("tl.tcgen05_st")))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.tcgen05_st",
    ),
    RejectionCase(
        name="tileop_tcgen05_gemm",
        symbol="tileop_tcgen05_gemm_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def tileop_tcgen05_gemm_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.call_intrin("handle", tvm.tir.op.Op.get("tl.tileop.tcgen05_gemm")))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.tileop.tcgen05_gemm",
    ),
    RejectionCase(
        name="cluster_sync",
        symbol="cluster_sync_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def cluster_sync_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.call_intrin("void", tvm.tir.op.Op.get("tl.cluster_sync")))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.cluster_sync",
    ),
    RejectionCase(
        name="block_rank_in_cluster",
        symbol="block_rank_in_cluster_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def block_rank_in_cluster_probe(B: T.Buffer((1,), "int32")):
    B[0] = T.call_intrin("int32", tvm.tir.op.Op.get("tl.block_rank_in_cluster"))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.block_rank_in_cluster",
    ),
    RejectionCase(
        name="cluster_launch_control",
        symbol="cluster_launch_control_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def cluster_launch_control_probe(B: T.Buffer((1,), "int32")):
    B[0] = T.call_intrin("int32", tvm.tir.op.Op.get("tl.clc_is_canceled"))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.clc_is_canceled",
    ),
    RejectionCase(
        name="warpgroup_wait",
        symbol="warpgroup_wait_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def warpgroup_wait_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.call_intrin("handle", tvm.tir.op.Op.get("tl.warpgroup_wait")))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.warpgroup_wait",
    ),
    RejectionCase(
        name="get_lane_idx",
        symbol="get_lane_idx_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def get_lane_idx_probe(B: T.Buffer((1,), "int32")):
    B[0] = T.call_intrin("int32", tvm.tir.op.Op.get("tl.get_lane_idx"))
""",
        expected="tl.get_lane_idx requires an active threadIdx.x launch",
    ),
    RejectionCase(
        name="get_warp_idx_sync",
        symbol="get_warp_idx_sync_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def get_warp_idx_sync_probe(B: T.Buffer((1,), "int32")):
    B[0] = T.call_intrin("int32", tvm.tir.op.Op.get("tl.get_warp_idx_sync"))
""",
        expected="tl.get_warp_idx_sync requires an active threadIdx.x launch",
    ),
    RejectionCase(
        name="get_warp_idx",
        symbol="get_warp_idx_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def get_warp_idx_probe(B: T.Buffer((1,), "int32")):
    B[0] = T.call_intrin("int32", tvm.tir.op.Op.get("tl.get_warp_idx"))
""",
        expected="tl.get_warp_idx requires an active threadIdx.x launch",
    ),
    RejectionCase(
        name="get_warp_group_idx",
        symbol="get_warp_group_idx_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def get_warp_group_idx_probe(B: T.Buffer((1,), "int32")):
    B[0] = T.call_intrin("int32", tvm.tir.op.Op.get("tl.get_warp_group_idx"))
""",
        expected="tl.get_warp_group_idx requires an active threadIdx.x launch",
    ),
    RejectionCase(
        name="deallocate_tmem",
        symbol="deallocate_tmem_probe",
        source="""
# from tvm.script import tir as T
@T.prim_func
def deallocate_tmem_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.call_intrin("handle", tvm.tir.op.Op.get("tl.deallocate_tmem")))
""",
        expected="Unsupported CUDA/target-specific intrinsic.*tl.deallocate_tmem",
    ),
)


def test_float2half_rz_extern_lowers_to_fp_trunc():
    source = """
# from tvm.script import tir as T
@T.prim_func
def extern_float2half_probe(A: T.Buffer((1,), "float32"), B: T.Buffer((1,), "float16")):
    B[0] = T.call_extern("float16", "__float2half_rz", A[0])
"""
    mlir = lower_source_to_mlir(source, "extern_float2half_probe")
    assert "arith.truncf" in mlir
    assert "Unsupported call_extern" not in mlir


def test_atomic_add_offset_extern_lowers_to_return_old_load_add_store():
    source = """
# from tvm.script import tir as T
@T.prim_func
def atomic_add_offset_probe(B: T.Buffer((8,), "int32"), C: T.Buffer((1,), "int32")):
    C[0] = T.call_extern("int32", "tl_atomic_add_offset", T.address_of(B[0]), T.int32(2), T.int32(3))
"""
    mlir = lower_source_to_mlir(
        source,
        "atomic_add_offset_probe",
        "extern/atomic_add_offset_probe",
    )
    assert "tl_atomic_add_offset" not in mlir
    assert "Unsupported call_extern" not in mlir
    assert "memref.load" in mlir
    assert "arith.addi" in mlir
    assert "memref.store" in mlir


def test_rng_uniform_intrinsics_lower_to_deterministic_scalar_prng():
    source = """
# from tvm.script import tir as T
@T.prim_func
def rng_uniform_probe(B: T.Buffer((4,), "float32")):
    tx = T.launch_thread("threadIdx.x", 4)
    T.evaluate(T.call_intrin("void", tvm.tir.op.Op.get("tl.rng_init"), T.uint64(7), T.Cast("uint64", tx), T.uint64(0), "curandStatePhilox4_32_10_t"))
    B[tx] = T.call_intrin("float32", tvm.tir.op.Op.get("tl.rng_rand_float"), "uniform")
"""
    mlir = lower_source_to_mlir(
        source,
        "rng_uniform_probe",
        "extern/rng_uniform_probe",
    )
    assert "Unsupported TIR expr" not in mlir
    assert "memref.alloca" in mlir
    assert "arith.uitofp" in mlir
    assert "arith.mulf" in mlir


def test_match_sync_externs_lower_to_scalar_constants_for_single_thread():
    source = """
# from tvm.script import tir as T
@T.prim_func
def extern_match_sync_single_thread_probe(
    A: T.Buffer((1,), "uint32"),
    B: T.Buffer((1,), "uint32"),
    C: T.Buffer((1,), "uint32"),
):
    B[0] = T.call_extern("uint32", "__match_any_sync", T.uint32(4294967295), A[0])
    C[0] = T.call_extern("uint32", "__match_all_sync", T.uint32(4294967295), A[0])
"""
    mlir = lower_source_to_mlir(source, "extern_match_sync_single_thread_probe")
    assert "__match_any_sync" not in mlir
    assert "__match_all_sync" not in mlir
    assert "Unsupported call_extern" not in mlir
    assert "arith.constant 1" in mlir


def _match_sync_extern_thread_launch_kernel(extern_name: str):
    @T.prim_func
    def extern_match_sync_thread_launch_probe(
        A: T.Tensor((4,), "uint32"),
        B: T.Tensor((4,), "uint32"),
    ):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.call_extern("uint32", extern_name, T.uint32(4294967295), A[tid])

    return extern_match_sync_thread_launch_probe


def test_match_any_sync_extern_lowers_by_serialized_warp_replay():
    mlir = lower_tilelang_prim_to_mlir(
        _match_sync_extern_thread_launch_kernel("__match_any_sync"),
        "match_any_sync_thread_launch_probe",
        "extern/match_any_sync_thread_launch_probe",
    )
    assert "__match_any_sync" not in mlir
    assert "Unsupported call_extern" not in mlir
    assert "arith.andi" in mlir
    assert "arith.select" in mlir
    assert "scf.if" in mlir


def test_match_any_sync_extern_lowers_guarded_lane_replay_pattern():
    source = """
# from tvm.script import tir as T
@T.prim_func
def extern_match_any_guarded_lane_probe(
    A: T.Buffer((8,), "int64"),
    B: T.Buffer((4,), "uint32"),
):
    tx = T.launch_thread("threadIdx.x", 4)
    lane = T.int32()
    with T.LetStmt(tx % 32, var=lane):
        i: T.int32 = lane + 2
        v: T.int32 = T.Select(i < 8, T.Cast("int32", A[i]), -1)
        B[tx] = T.call_extern("uint32", "__match_any_sync", T.int64(4294967295), v)
"""
    mlir = lower_source_to_mlir(
        source,
        "extern_match_any_guarded_lane_probe",
        "extern/match_any_guarded_lane_probe",
    )
    assert "__match_any_sync" not in mlir
    assert "Unsupported call_extern" not in mlir
    assert "arith.andi" in mlir
    assert "arith.select" in mlir
    assert "scf.if" in mlir


def test_match_all_sync_extern_lowers_by_serialized_warp_replay():
    mlir = lower_tilelang_prim_to_mlir(
        _match_sync_extern_thread_launch_kernel("__match_all_sync"),
        "match_all_sync_thread_launch_probe",
        "extern/match_all_sync_thread_launch_probe",
    )
    assert "__match_all_sync" not in mlir
    assert "Unsupported call_extern" not in mlir
    assert "arith.andi" in mlir
    assert "arith.ori" in mlir
    assert "arith.select" in mlir
    assert "scf.if" in mlir


def test_match_all_sync_extern_lowers_guarded_lane_replay_pattern():
    source = """
# from tvm.script import tir as T
@T.prim_func
def extern_match_all_guarded_lane_probe(
    A: T.Buffer((8,), "int64"),
    B: T.Buffer((4,), "uint32"),
):
    tx = T.launch_thread("threadIdx.x", 4)
    lane = T.int32()
    with T.LetStmt(tx % 32, var=lane):
        i: T.int32 = lane + 2
        v: T.int32 = T.Select(i < 8, T.Cast("int32", A[i]), -1)
        B[tx] = T.call_extern("uint32", "__match_all_sync", T.int64(4294967295), v)
"""
    mlir = lower_source_to_mlir(
        source,
        "extern_match_all_guarded_lane_probe",
        "extern/match_all_guarded_lane_probe",
    )
    assert "__match_all_sync" not in mlir
    assert "Unsupported call_extern" not in mlir
    assert "arith.andi" in mlir
    assert "arith.select" in mlir
    assert "scf.if" in mlir


def test_match_all_sync_extern_rejects_non_linear_index_pattern():
    source = """
# from tvm.script import tir as T
@T.prim_func
def extern_match_all_non_linear_probe(
    A: T.Buffer((8,), "uint32"),
    B: T.Buffer((4,), "uint32"),
):
    tx = T.launch_thread("threadIdx.x", 4)
    B[tx] = T.call_extern("uint32", "__match_all_sync", T.uint32(4294967295), A[tx * 2])
"""
    with pytest.raises(Exception, match="tl.match_all_sync expression is not supported yet"):
        build_mlir_from_source(source, "extern_match_all_non_linear_probe")


def test_pdl_sync_lowers_as_noop_for_single_kernel_mlir():
    source = """
# from tvm.script import tir as T
@T.prim_func
def pdl_sync_probe(A: T.Buffer((1,), "float32"), B: T.Buffer((1,), "float32")):
    B[0] = A[0]
    T.evaluate(T.call_intrin("void", tvm.tir.op.Op.get("tl.pdl_sync")))
    B[0] = B[0] + T.float32(1)
"""
    mlir = lower_source_to_mlir(source, "pdl_sync_probe")
    assert "tl.pdl_sync" not in mlir
    assert "Unsupported CUDA/target-specific intrinsic" not in mlir
    assert "arith.addf" in mlir


def test_ptx_wait_commit_lower_as_noops_for_serialized_backend():
    source = """
# from tvm.script import tir as T
@T.prim_func
def ptx_wait_commit_noop_probe(A: T.Buffer((1,), "float32")):
    T.evaluate(T.ptx_commit_group())
    T.evaluate(T.ptx_wait_group(0))
    A[0] = T.float32(1)
"""
    mlir = lower_source_to_mlir(source, "ptx_wait_commit_noop_probe")
    assert "tir.ptx_commit_group" not in mlir
    assert "tir.ptx_wait_group" not in mlir
    assert "Unsupported CUDA/target-specific intrinsic" not in mlir
    assert "memref.store" in mlir


def test_register_hint_intrinsics_lower_as_noops_for_serialized_backend():
    source = """
# from tvm.script import tir as T
@T.prim_func
def reg_hint_noop_probe(A: T.Buffer((1,), "int32")):
    T.evaluate(T.call_intrin("void", tvm.tir.op.Op.get("tl.set_max_nreg"), T.int32(24), T.int32(0)))
    T.evaluate(T.call_intrin("void", tvm.tir.op.Op.get("tl.no_set_max_nreg")))
    A[0] = T.int32(1)
"""
    mlir = lower_source_to_mlir(source, "reg_hint_noop_probe")
    assert "tl.set_max_nreg" not in mlir
    assert "tl.no_set_max_nreg" not in mlir
    assert "Unsupported CUDA/target-specific intrinsic" not in mlir
    assert "memref.store" in mlir


def test_tma_copy_wgmma_markers_lower_to_sync_copy_and_matmul():
    @T.prim_func
    def tma_wgmma_probe(
        A: T.Tensor((64, 64), "float16"),
        B: T.Tensor((64, 128), "float16"),
        D: T.Tensor((64, 128), "float32"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 64), "float16")
            B_shared = T.alloc_shared((64, 128), "float16")
            C_local = T.alloc_fragment((64, 128), "float32")
            mbar_a = T.alloc_barrier(128)
            mbar_b = T.alloc_barrier(128)

            T.set_max_nreg(24, 0)
            T.tma_copy(A[0:64, 0:64], A_shared, barrier=mbar_a)
            T.barrier_arrive(mbar_a)
            T.tma_copy(B[0:64, 0:128], B_shared, barrier=mbar_b)
            T.barrier_arrive(mbar_b)
            T.mbarrier_wait_parity(mbar_a, 0)
            T.mbarrier_wait_parity(mbar_b, 0)
            T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.wait_wgmma(0)
            T.warpgroup_fence_operand(C_local, num_regs=64)
            T.copy(C_local, D[0:64, 0:128])

    mlir = lower_tilelang_prim_to_mlir(
        tma_wgmma_probe,
        "tma_wgmma_probe",
        "target_intrinsic/tma_wgmma_probe",
    )
    assert "tl.tileop.tma_copy" not in mlir
    assert "tir.ptx_arrive_barrier" not in mlir
    assert "tl.mbarrier_wait_parity" not in mlir
    assert "tl.tileop.wgmma_gemm" not in mlir
    assert "tl.wait_wgmma" not in mlir
    assert "tl.warpgroup_fence_operand" not in mlir
    assert "Unsupported CUDA/target-specific intrinsic" not in mlir
    assert "linalg.matmul" in mlir
    assert "memref.copy" in mlir


def test_cp_async_barrier_noinc_marker_noops_in_serialized_backend():
    @T.prim_func
    def cp_async_barrier_noinc_probe(B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            mbar = T.alloc_barrier(128)
            T.cp_async_barrier_noinc(mbar)
            B[0] = 0

    mlir = lower_tilelang_prim_to_mlir(
        cp_async_barrier_noinc_probe,
        "cp_async_barrier_noinc_probe",
        "target_intrinsic/cp_async_barrier_noinc_probe",
    )
    assert "tl.ptx_cp_async_barrier_noinc" not in mlir
    assert "Unsupported CUDA/target-specific intrinsic" not in mlir
    assert "memref.store" in mlir


def test_mbarrier_expect_tx_marker_noops_in_serialized_backend():
    @T.prim_func
    def mbarrier_expect_tx_probe(B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            mbar = T.alloc_barrier(128)
            T.mbarrier_expect_tx(mbar, 16)
            B[0] = 0

    mlir = lower_tilelang_prim_to_mlir(
        mbarrier_expect_tx_probe,
        "mbarrier_expect_tx_probe",
        "target_intrinsic/mbarrier_expect_tx_probe",
    )
    assert "tl.mbarrier_expect_tx" not in mlir
    assert "Unsupported CUDA/target-specific intrinsic" not in mlir
    assert "memref.store" in mlir


def test_tileop_gemm_dynamic_clear_accum_lowers_to_conditional_zero_fill():
    @T.prim_func
    def dynamic_clear_accum_probe(
        A: T.Tensor((2, 4, 4), "float16"),
        B: T.Tensor((2, 4, 4), "float16"),
        D: T.Tensor((4, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4, 4), "float16")
            B_shared = T.alloc_shared((4, 4), "float16")
            C_local = T.alloc_fragment((4, 4), "float32")
            for k in T.serial(2):
                T.copy(A[k, 0:4, 0:4], A_shared)
                T.copy(B[k, 0:4, 0:4], B_shared)
                T.gemm(A_shared, B_shared, C_local, clear_accum=k == 0)
            T.copy(C_local, D[0:4, 0:4])

    mlir = lower_tilelang_prim_to_mlir(
        dynamic_clear_accum_probe,
        "dynamic_clear_accum_probe",
        "target_intrinsic/dynamic_clear_accum_probe",
    )
    assert "tl.tileop.gemm" not in mlir
    assert "Unsupported" not in mlir
    assert "scf.if" in mlir
    assert "linalg.matmul" in mlir


def test_pointer_table_handle_reinterpret_lowers_for_dynamic_alias_buffer_views():
    source = """
# from tvm.script import tir as T
@T.prim_func
def pointer_table_dynamic_alias_reinterpret_probe(
    ptrs: T.Buffer((1,), "int64"),
    out: T.Buffer((1,), "float32"),
    n: T.int32,
):
    tensor: T.handle("float32", "global") = T.reinterpret("handle", ptrs[0])
    lhs = T.Buffer((n,), "float32", data=tensor)
    rhs = T.Buffer((n,), "float32", data=tensor)
    out[0] = lhs[0] + rhs[n - 1]
"""
    mlir = lower_source_to_mlir(
        source,
        "pointer_table_dynamic_alias_reinterpret_probe",
        "target_intrinsic/pointer_table_dynamic_alias_reinterpret_probe",
    )
    helper_symbol = "@tilelang_riscv_ptr_i64_to_memref_float32_r1_d"
    assert "Unsupported tir.reinterpret" not in mlir
    assert f"func.func private {helper_symbol}(i64, index) -> memref<?xf32>" in mlir
    assert mlir.count(f"call {helper_symbol}") == 1
    assert "arith.addf" in mlir


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_target_specific_intrinsics_are_rejected(case: RejectionCase):
    with pytest.raises(Exception, match=case.expected):
        build_mlir_from_source(case.source, case.symbol)


def test_tileop_reduce_absmax_lowers_static_fragment_slice():
    @T.prim_func
    def tileop_reduce_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            frag = T.alloc_fragment((4,), "float32")
            out = T.alloc_fragment((1,), "float32")
            for i in T.serial(4):
                frag[i] = A[i]
            T.reduce_absmax(frag, out, dim=0)
            B[0] = out[0]

    mlir = lower_tilelang_prim_to_mlir(tileop_reduce_probe, "tileop_reduce_probe")
    assert "Unsupported tile reduction intrinsic" not in mlir
    assert "arith.cmpf" in mlir
    assert "arith.select" in mlir


def test_tileop_reduce_sum_clear_false_lowers_existing_output_accumulator():
    @T.prim_func
    def tileop_reduce_sum_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            frag = T.alloc_fragment((4,), "float32")
            out = T.alloc_fragment((1,), "float32")
            out[0] = B[0]
            for i in T.serial(4):
                frag[i] = A[i]
            T.reduce_sum(frag, out, dim=0, clear=False)
            B[0] = out[0]

    mlir = lower_tilelang_prim_to_mlir(tileop_reduce_sum_probe, "tileop_reduce_sum_probe")
    assert "Unsupported tile reduction intrinsic" not in mlir
    assert "arith.addf" in mlir


@pytest.mark.parametrize(
    ("kind", "expected_op"),
    (
        ("bitand", "arith.andi"),
        ("bitor", "arith.ori"),
        ("bitxor", "arith.xori"),
    ),
)
def test_tileop_reduce_bitwise_lowers_static_fragment_slice(kind: str, expected_op: str):
    if kind == "bitand":

        @T.prim_func
        def tileop_reduce_bitwise_probe(
            A: T.Tensor((4,), "int32"),
            B: T.Tensor((1,), "int32"),
        ):
            with T.Kernel(1, threads=1):
                frag = T.alloc_fragment((4,), "int32")
                out = T.alloc_fragment((1,), "int32")
                for i in T.serial(4):
                    frag[i] = A[i]
                T.reduce_bitand(frag, out, dim=0)
                B[0] = out[0]

    elif kind == "bitor":

        @T.prim_func
        def tileop_reduce_bitwise_probe(
            A: T.Tensor((4,), "int32"),
            B: T.Tensor((1,), "int32"),
        ):
            with T.Kernel(1, threads=1):
                frag = T.alloc_fragment((4,), "int32")
                out = T.alloc_fragment((1,), "int32")
                for i in T.serial(4):
                    frag[i] = A[i]
                T.reduce_bitor(frag, out, dim=0)
                B[0] = out[0]

    else:

        @T.prim_func
        def tileop_reduce_bitwise_probe(
            A: T.Tensor((4,), "int32"),
            B: T.Tensor((1,), "int32"),
        ):
            with T.Kernel(1, threads=1):
                frag = T.alloc_fragment((4,), "int32")
                out = T.alloc_fragment((1,), "int32")
                for i in T.serial(4):
                    frag[i] = A[i]
                T.reduce_bitxor(frag, out, dim=0)
                B[0] = out[0]

    mlir = lower_tilelang_prim_to_mlir(tileop_reduce_bitwise_probe, f"tileop_reduce_{kind}_probe")
    assert "Unsupported tile reduction kind" not in mlir
    assert expected_op in mlir


def test_tileop_reduce_bitwise_rejects_float_dtype():
    @T.prim_func
    def tileop_reduce_bitand_float_probe(
        A: T.Tensor((4,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            frag = T.alloc_fragment((4,), "float32")
            out = T.alloc_fragment((1,), "float32")
            for i in T.serial(4):
                frag[i] = A[i]
            T.reduce_bitand(frag, out, dim=0)
            B[0] = out[0]

    with pytest.raises(Exception, match="Bitwise tile reductions require integer-like"):
        build_mlir_from_tilelang_prim(
            tileop_reduce_bitand_float_probe,
            "tileop_reduce_bitand_float_probe",
        )


def test_tileop_cumsum_lowers_static_single_thread_scan():
    @T.prim_func
    def tileop_cumsum_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            scratch = T.alloc_shared((4,), "int32")
            for i in T.serial(4):
                scratch[i] = A[i]
            T.cumsum(scratch)
            for i in T.serial(4):
                B[i] = scratch[i]

    mlir = lower_tilelang_prim_to_mlir(tileop_cumsum_probe, "tileop_cumsum_probe")
    assert "tl.tileop.cumsum" not in mlir
    assert "Unsupported tile scan intrinsic" not in mlir
    assert "memref.load" in mlir
    assert "arith.addi" in mlir
    assert "memref.store" in mlir


def test_tileop_cumsum_lowers_static_reverse_2d_single_thread_scan():
    @T.prim_func
    def tileop_cumsum_reverse_2d_probe(
        A: T.Tensor((2, 4), "float32"),
        B: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            scratch = T.alloc_shared((2, 4), "float32")
            for i, j in T.grid(2, 4):
                scratch[i, j] = A[i, j]
            T.cumsum(scratch, dim=1, reverse=True)
            for i, j in T.grid(2, 4):
                B[i, j] = scratch[i, j]

    mlir = lower_tilelang_prim_to_mlir(
        tileop_cumsum_reverse_2d_probe,
        "tileop_cumsum_reverse_2d_probe",
    )
    assert "tl.tileop.cumsum" not in mlir
    assert "Unsupported tile scan intrinsic" not in mlir
    assert "arith.addf" in mlir
    assert "memref.store" in mlir


def test_tileop_cumsum_lowers_shared_nonunit_thread_phase_global_scan():
    @T.prim_func
    def tileop_cumsum_shared_thread_probe(
        A: T.Tensor((4,), "int32"),
        B: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(1, threads=4):
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "int32")
            scratch[tx] = A[tx]
            T.sync_threads()
            T.cumsum(scratch)
            T.sync_threads()
            B[tx] = scratch[tx]

    mlir = lower_tilelang_prim_to_mlir(
        tileop_cumsum_shared_thread_probe,
        "tileop_cumsum_shared_thread_probe",
    )
    assert "tl.tileop.cumsum" not in mlir
    assert "Unsupported tile scan intrinsic" not in mlir
    assert "memref.alloca" in mlir
    assert "arith.addi" in mlir
    assert mlir.count("scf.for") >= 2


def test_tileop_cumsum_rejects_nonunit_thread_scan_until_cooperative_model_exists():
    @T.prim_func
    def tileop_cumsum_thread_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tx = T.get_thread_binding()
            if tx < 4:
                B[tx] = A[tx]
            T.cumsum(B)

    with pytest.raises(Exception, match="Unsupported tile scan intrinsic.*tl.tileop.cumsum"):
        build_mlir_from_tilelang_prim(
            tileop_cumsum_thread_probe,
            "tileop_cumsum_thread_probe",
        )


def test_tileop_finalize_reducer_noops_without_nonunit_thread_launch():
    @T.prim_func
    def tileop_finalize_reducer_noop_probe(B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            reducer = T.alloc_reducer((1,), "int32", "min", replication="all")
            T.fill(reducer, T.max_value(T.int32))
            reducer[0] = T.min(reducer[0], T.int32(7))
            T.finalize_reducer(reducer)
            B[0] = reducer[0]

    mlir = lower_tilelang_prim_to_mlir(
        tileop_finalize_reducer_noop_probe,
        "tileop_finalize_reducer_noop_probe",
    )
    assert "tl.tileop.finalize_reducer" not in mlir
    assert "Unsupported reducer finalization intrinsic" not in mlir
    assert "memref.store" in mlir


def test_tileop_finalize_reducer_noops_for_thread_invariant_launch():
    @T.prim_func
    def tileop_finalize_reducer_thread_invariant_probe(
        A: T.Tensor((4,), "int32"),
        B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=4):
            reducer = T.alloc_reducer((1,), "int32", "min", replication="all")
            T.fill(reducer, T.max_value(T.int32))
            for i in T.parallel(4):
                reducer[0] = T.min(reducer[0], A[i])
            T.finalize_reducer(reducer)
            B[0] = reducer[0]

    mlir = lower_tilelang_prim_to_mlir(
        tileop_finalize_reducer_thread_invariant_probe,
        "tileop_finalize_reducer_thread_invariant_probe",
    )
    assert "tl.tileop.finalize_reducer" not in mlir
    assert "Unsupported reducer finalization intrinsic" not in mlir
    assert "scf.for" in mlir
    assert "memref.store" in mlir


def test_tileop_finalize_reducer_rejects_nonunit_thread_launch():
    @T.prim_func
    def tileop_finalize_reducer_thread_probe(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=4):
            tx = T.get_thread_binding()
            reducer = T.alloc_reducer((1,), "float32", "sum", replication="all")
            T.fill(reducer, T.float32(0))
            reducer[0] += T.cast(tx, T.float32)
            T.finalize_reducer(reducer)
            if tx == 0:
                B[0] = reducer[0]

    with pytest.raises(Exception, match="Unsupported reducer finalization intrinsic.*AllReduce"):
        build_mlir_from_tilelang_prim(
            tileop_finalize_reducer_thread_probe,
            "tileop_finalize_reducer_thread_probe",
        )
