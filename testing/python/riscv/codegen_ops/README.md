# RISC-V Codegen Operator Tests

This directory contains focused `riscv` MLIR codegen tests grouped by operator family.

## Layout

- `harness.py`: shared helpers that lower TIR / TileLang prim funcs to RISC-V MLIR and provide small assertion utilities.
- `conftest.py`: adds the optional pytest flag `--riscv-dump-mlir DIR`.
- `test_elementwise_ops.py`: scalar math, bitwise/integer helpers, numeric limits, and simple TileLang scalar ops.
- `test_reduction_ops.py`: reductions and broadcast-style linalg lowering.
- `test_tile_ops.py`: `T.copy`, `T.gemm`, transpose variants, strided operands, and copy-layout diagnostics.
- `test_thread_ops.py`: serialized block/thread lowering and thread-control helpers.
- `test_cooperative_thread_rejections.py`: cooperative thread lowering plus unsupported cooperative diagnostics.
- `test_target_intrinsic_rejections.py`: target-specific intrinsic lowering/no-op/rejection coverage.
- `test_atomic_ops.py`: scalar/vector atomic lowering and unsupported atomic diagnostics.
- `test_vector_ops.py`: vector buffers, vector ops, ramp slices, packed views, and low-precision dtypes.
- `test_layout_ops.py`: elem offsets, strided params, match-buffer subviews, vectorized slices, and invalid layouts.

## Run

From the repo root, the normal path is direct pytest:

```bash
pytest -q testing/python/riscv/codegen_ops
```

To select cases:

```bash
pytest -q testing/python/riscv/codegen_ops -k tile_gemm
```

To dump final generated MLIR:

```bash
pytest -q testing/python/riscv/codegen_ops --riscv-dump-mlir dump/riscv_codegen_ops_mlir
```

`testing/python/riscv/run_codegen_ops.py` is only a convenience wrapper. It sets `PYTHONPATH` and `LD_LIBRARY_PATH`, runs the same pytest suite, and has a shorter `--dump-mlir` option with a default dump directory.

Use the wrapper when your shell environment is not already configured:

```bash
python testing/python/riscv/run_codegen_ops.py --dump-mlir
```
