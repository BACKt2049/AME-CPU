from __future__ import annotations

import pytest

import tilelang.tladapter as tladapter
from tilelang.tladapter.transforms import mlir as mlir_passes


pytest.importorskip("tilelang.tladapter._native")


def test_tladapter_pipeline_runs_canonicalize():
    source = """module {
  func.func @kernel() {
    return
  }
}
"""

    pipeline = tladapter.Pipeline()
    pipeline.add(mlir_passes.canonicalize)
    pipeline.add(mlir_passes.cse)

    lowered = pipeline.run(source)

    assert "func.func @kernel()" in lowered
    assert str(pipeline).startswith("builtin.module(")
