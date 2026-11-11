#!/bin/bash
# TileLang-RISC-V 全量验收 · 一次性执行脚本（服务器端）
# 幂等：可反复运行；长任务请放进 tmux： tmux new -s verify 'bash run_all_riscv_tests.sh'
# 产物：~/riscv_results.xml / ~/operator_inventory.json / ~/riscv_results.log
# 之后在本地运行 build_acceptance.py 生成验收表 xlsx。
set -u

REPO="${TILELANG_REPO:-$HOME/tilelang-riscv-clean}"
LLVM_ROOT="${TILELANG_RISCV_LLVM_ROOT:-$HOME/llvm-project/build}"
BUILTINS_DIR="$LLVM_ROOT/lib/clang/24/lib/riscv64-unknown-linux-gnu"
GCC_DIR="/usr/lib/gcc/riscv64-openEuler-linux/12"

# ---------- 1. 环境变量 ----------
export TILELANG_RISCV_LLVM_ROOT="$LLVM_ROOT"
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=$GCC_DIR -L$BUILTINS_DIR -lclang_rt.builtins"
unset TILELANG_DISABLE_CACHE          # 开缓存复用已编译产物，约 30 分钟 → 视机器而定
export CC=/usr/bin/gcc CXX=/usr/bin/g++

echo "=== [1/5] 环境 ==="
echo "REPO      = $REPO"
echo "LLVM_ROOT = $LLVM_ROOT"
python3 -c "import numpy,torch,z3; print('numpy',numpy.__version__,'| torch',torch.__version__,'| z3',z3.get_version_string())" 2>/dev/null \
  || { echo "FATAL: 依赖缺失（numpy<2 / torch / z3）"; exit 1; }
# pip 装 tilelang 会把 numpy 升到 2.x，与 torch 2.4.1 ABI 冲突，强制降回
if python3 -c "import numpy,sys; sys.exit(0 if numpy.__version__.split('.')[0]=='1' else 1)" 2>/dev/null; then
  :
else
  echo "  numpy 版本 >=2，降回 <2 ..."
  python3 -m pip install --user 'numpy<2' || exit 1
fi
for f in mlir-opt mlir-translate clang; do
  [ -x "$LLVM_ROOT/bin/$f" ] || { echo "FATAL: 缺 $LLVM_ROOT/bin/$f"; exit 1; }
done
[ -f "$BUILTINS_DIR/libclang_rt.builtins.a" ] || { echo "FATAL: 缺 compiler-rt builtins"; exit 1; }

# ---------- 2. fp8 测试补丁（幂等） ----------
# PyTorch 2.4.1 CPU 未实现 float8_e4m3fn 的 fill_cpu；测试用 torch.full 直接造 fp8 会崩。
# 改为「先 fp32 造再 .to(fp8)」。这是测试层改动，不改产品源码。
echo "=== [2/5] fp8 测试补丁（幂等）==="
python3 - "$REPO" <<'PY'
import sys, shutil
from pathlib import Path
p = Path(sys.argv[1]) / "testing/python/riscv/TileOps/test_paged_kv_cache_write.py"
src = p.read_text(encoding="utf-8")
if "_FP8_DTYPES" in src:
    print("  已打过补丁，跳过")
    sys.exit(0)
shutil.copy2(p, str(p) + ".orig")
helper = "\n".join([
    "",
    "_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)",
    "",
    "",
    "def _full(shape, value, dtype):",
    "    if dtype in _FP8_DTYPES:",
    "        return torch.full(shape, value, dtype=torch.float32).to(dtype)",
    "    return torch.full(shape, value, dtype=dtype)",
    "",
    "",
    "def _full_like(tensor, value):",
    "    if tensor.dtype in _FP8_DTYPES:",
    "        return torch.full(tensor.shape, value, dtype=torch.float32).to(tensor.dtype)",
    "    return torch.full_like(tensor, value)",
    "",
])
anchor = "from ._harness import compile_tileops_jit, get_kernel_class"
assert anchor in src, "anchor 未找到，补丁中止"
src = src.replace(anchor, anchor + "\n" + helper, 1)
reps = [
    ("k_pages = torch.full((pages, page_size, heads, dim), 0.25, dtype=cache_dtype)",
     "k_pages = _full((pages, page_size, heads, dim), 0.25, cache_dtype)"),
    ("v_pages = torch.full_like(k_pages, -0.5)", "v_pages = _full_like(k_pages, -0.5)"),
    ("k_pages = torch.full((2, 2, 1, 16), 0.25, dtype=torch.float8_e4m3fn)",
     "k_pages = _full((2, 2, 1, 16), 0.25, torch.float8_e4m3fn)"),
]
for old, new in reps:
    assert old in src, "未找到待替换行: " + old[:50]
    src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("  补丁已应用（备份 .orig）")
PY
[ $? -eq 0 ] || { echo "FATAL: fp8 补丁失败"; exit 1; }

# ---------- 3. 全量测试（必须仓库根目录，保证 JUnit classname 契约）----------
echo "=== [3/5] 全量测试 ==="
cd "$REPO" || exit 1
python3 -m pytest -q \
  testing/python/riscv/test_riscv_*.py \
  testing/python/riscv/codegen_ops \
  testing/python/riscv/TileKernels \
  testing/python/riscv/TileOps \
  --junitxml="$HOME/riscv_results.xml" 2>&1 | tee "$HOME/riscv_results.log"
TEST_RC=${PIPESTATUS[0]}
echo "  pytest exit=$TEST_RC"
[ "$TEST_RC" -eq 0 ] || { echo "WARN: 存在失败用例，仍继续生成 inventory"; }

# ---------- 4. 官方清单 ----------
echo "=== [4/5] operator_inventory.py ==="
python3 testing/python/riscv/operator_inventory.py \
  --junit "$HOME/riscv_results.xml" \
  --output "$HOME/operator_inventory.json" --check
echo "  inventory exit=$?"
# 不用 --require-tileops-numerical：按设计必然失败（20 unmapped + 9 component + 21 known_gaps）

# ---------- 5. 汇总 ----------
echo "=== [5/5] 汇总 ==="
grep -E '[0-9]+ (passed|failed)' "$HOME/riscv_results.log" | tail -1
ls -l "$HOME/riscv_results.xml" "$HOME/operator_inventory.json"
echo
echo "服务器端完成。生成验收表需把下列文件下载到本地 workspace/evidence/ 后运行 build_acceptance.py："
echo "  riscv_results.xml / operator_inventory.json / riscv_results.log"
