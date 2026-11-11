import pytest

from tilelang.utils.target import determine_target


def test_determine_target_accepts_riscv():
    assert determine_target("riscv") == "riscv"


def test_determine_target_rejects_legacy_linalg_riscv():
    with pytest.raises(AssertionError) as exc_info:
        determine_target("linalg_riscv")
    assert "`riscv`" in str(exc_info.value)
