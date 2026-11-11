from __future__ import annotations

import os


def pytest_addoption(parser):
    group = parser.getgroup("riscv-codegen")
    group.addoption(
        "--riscv-dump-mlir",
        action="store",
        default=None,
        metavar="DIR",
        help="Dump final riscv MLIR for each collected codegen_ops case into DIR.",
    )


def pytest_configure(config):
    dump_dir = config.getoption("--riscv-dump-mlir", default=None)
    if dump_dir:
        os.environ["TILELANG_RISCV_DUMP_MLIR_DIR"] = dump_dir
