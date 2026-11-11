# TileLang-RISC-V 上手指南（手动逐条执行版）

> 本手册所有命令都已对着本地仓库源码核对过函数名、路径、参数与开关。
> 你逐条复制执行；遇阻把报错贴回来。
> 远端仓库位置：`~/tilelang-riscv`（`39.102.211.164:6000`，用户 `wangjiayue`）

---

## 心智模型（先记住这条管道）

```
TileLang Python DSL
  → TVM TIR
    → MLIR (Linalg / SCF / MemRef / Vector)     ← src/target/codegen_riscv.cc 生成
      → LLVM dialect MLIR                        ← tilelang/jit/adapter/riscv/libgen.py
        → LLVM IR → 目标文件(.o) → 共享库(.so)    ← wrapper.py 用 clang 链接
          → ctypes 加载执行                       ← adapter.py
```

三层代码入口：
- 构建：`CMakeLists.txt` + `cmake/TileLangRISCVMLIR.cmake`（要求 LLVM/MLIR 24.0）
- 编译核心：`src/target/codegen_riscv.cc`（~12.9k 行）, `src/target/rt_mod_riscv.cc`（注册 `"riscv"` target）
- Python 前端/运行时：`tilelang/engine/phase.py`, `tilelang/tladapter/`, `tilelang/jit/adapter/riscv/`

关键事实：RISC-V 走的是**精简 pass 流水线**，`phase.py` 里 `is_riscv_target()` 分支（第 139、251 行）
**刻意跳过** `LayoutInference` / `InjectSoftwarePipeline` / `VectorizeLoop` 等 GPU 优化。
所以很多为 GPU 写的 kernel 形态在 RISC-V 上并不受支持——这是能力边界，不是 bug。

---

## Step 1 — 登录

```bash
ssh -p 6000 wangjiayue@39.102.211.164
```
密码：`wangjiayue`

```bash
hostname && uname -m && nproc && cd ~/tilelang-riscv && pwd
```
预期：`openeuler-riscv64` / `riscv64` / `64`。后面所有命令都在这里跑。

---

## Step 2 — 勘察现状（搞清楚有什么、缺什么）

### 2.1 仓库与子模块
```bash
cd ~/tilelang-riscv
git log --oneline -5
git status -sb
git submodule status
```
判断：`git log` 顶部应是若干 `[RISC-V] ...` 提交。`git submodule status` **行首无 `-`** 才算已初始化；
若有 `-`，执行：
```bash
git submodule update --init --recursive
```

### 2.2 Python 侧
```bash
python3 --version
python3 -m pip --version
ls -d ~/.venv* 2>/dev/null
python3 -c "import tilelang; print(tilelang.__file__)" 2>&1 | tail -1
```
判断：有没有现成 venv / 是否已装 tilelang。

### 2.3 LLVM/MLIR 工具链（构建硬前提，必须 24.0）
```bash
echo "TILELANG_RISCV_LLVM_ROOT=$TILELANG_RISCV_LLVM_ROOT"
which mlir-opt mlir-translate llc clang ld.lld 2>&1
ls -d ~/buddy-mlir/llvm/build ~/buddy-mlir/llvm/install \
      ~/toolchains/buddy-llvm /root/llvm/build /home/liquanyi/tool/llvm 2>/dev/null
find / -name "MLIRConfig.cmake" -not -path "*/3rdparty/*" 2>/dev/null | head
```
判断：要找的根目录下同时有 `lib/cmake/mlir/MLIRConfig.cmake` 与 `lib/cmake/llvm/LLVMConfig.cmake`。
记下路径 → 它就是 `TILELANG_RISCV_LLVM_ROOT`。
**若整机都没有**：需按官方文档下载 Buddy MLIR `release/v0.1.0` 的 RISC-V 资产或源码构建（见文末排错）。

### 2.4 RISC-V 原生编译器 + sysroot
```bash
which riscv64-unknown-linux-gnu-gcc riscv64-linux-gnu-gcc gcc g++ 2>&1
ls -d /opt/gcc-native /opt/gcc-native/sysroot 2>/dev/null
ls /opt/gcc-native/lib64/lp64d 2>/dev/null | head
```
判断：系统本身是 riscv64 时，系统 `gcc/g++` 即可作 `CC`/`CXX`。

### 2.5 Z3
```bash
python3 -c "import z3; print(z3.__path__[0])" 2>&1 | tail -1
ls -d /usr/include/z3*.h /usr/lib*/libz3.so* 2>/dev/null
```
判断：riscv64 常无 `z3-solver` wheel。若 `import z3` 成功，`Z3_ROOT` 取其包的**上一级**（使下面有 `include/` 和 `lib/`）。

### 2.6 硬件 / RVV
```bash
lscpu | grep -iE "^(Model name|Architecture)|^Flags" | cut -c1-200
```
判断：`Flags` 含 `v` 表示支持 RVV（向量化相关观察的前提）。

> **产出**：LLVM root 路径、CC/CXX、Z3_ROOT、是否需建 venv、子模块是否齐。

---

## Step 3 — 建 Python 环境（仅当 Step 2.2 无可用 venv）

```bash
cd ~/tilelang-riscv
python3 -m venv ~/.venv-tl
source ~/.venv-tl/bin/activate
python -m pip install --upgrade pip
which python    # 应指向 ~/.venv-tl/bin/python
```

---

## Step 4 — 导出环境变量（把 `<...>` 换成 Step 2 的真实路径）

```bash
export TILELANG_RISCV_LLVM_ROOT=/path/to/llvm-mlir     # 内含 lib/cmake/mlir/MLIRConfig.cmake
export Z3_ROOT=/path/to/z3-prefix                       # 内含 include/ 与 lib/libz3.so
export CC=gcc
export CXX=g++
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_MLIR_MODE=ON
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DTILELANG_RISCV_MLIR_MODE=ON"
```
验证：
```bash
echo "$TILELANG_RISCV_LLVM_ROOT"; ls "$TILELANG_RISCV_LLVM_ROOT/lib/cmake/mlir/MLIRConfig.cmake"
```
若工具链是 `lib64/lp64d` 分离布局，再补：
```bash
export LIBRARY_PATH=/opt/gcc-native/lib64/lp64d:/opt/gcc-native/libgcc/lp64d${LIBRARY_PATH:+:$LIBRARY_PATH}
```
> ⚠ 这些 `export` 只在当前 shell 生效。**新开终端要重跑 Step 4。**

---

## Step 5 — 安装依赖并构建

```bash
cd ~/tilelang-riscv
pip install scikit-build-core cython patchelf setuptools_scm cloudpickle pytest
```
```bash
# 先装 vendored tvm-ffi，顺序不能反
pip install ./3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps
```
```bash
# 可编辑安装（RISC-V 官方流程用的就是 -e）
pip install -e . --no-build-isolation -v
```
判断：结尾出现 `Successfully installed tilelang-0.1.8+...`。
- 卡在 `MLIR not found` → 回 Step 4 检查 `TILELANG_RISCV_LLVM_ROOT`
- 报 `-lgcc_s` → 检查 `LIBRARY_PATH` 里的 `lib64/lp64d`

以后改 C++/Python 后的重建都用这一条：
```bash
cd ~/tilelang-riscv && pip install -e . --no-build-isolation -v
```

---

## Step 6 — 验证构建产物

```bash
python -c "import tilelang; print('tilelang:', tilelang.__file__)"
python -c "import tilelang.tladapter._native; print('native adapter: ok')"
python -c "from tilelang import tvm; print(tvm.ffi.get_global_func('target.build.tilelang_riscv'))"
```
判断：三条都成功 = Python 包 / 原生扩展 / RISC-V 后端注册全部到位。第三条能打印后端函数对象，是"riscv target 真接进来了"的直接证据。

---

## Step 7 — 冒烟测试（严格递进，一步过了再进下一步）

> 必须在**仓库根目录**运行：`testing/conftest.py` 会把你仓库内的 `tilelang/` 插到 `sys.path` 最前，
> 保证测的是本地源码而不是别处装的版本。

```bash
cd ~/tilelang-riscv
pytest -q testing/python/riscv/test_riscv_target_parse.py          # target 字符串能被解析
pytest -q testing/python/riscv/test_riscv_toolchain.py            # 能发现 mlir-opt/mlir-translate/llc/clang/ld.lld
pytest -q testing/python/riscv/test_riscv_tladapter_pipeline.py   # MLIR pass pipeline 能跑
pytest -q testing/python/riscv/test_riscv_cache.py               # 产物缓存与重载
pytest -q testing/python/riscv/codegen_ops                       # TIR→MLIR 合成断言（理解产物最佳材料）
pytest -q testing/python/riscv/TileKernels                       # 48 个真实 kernel 端到端（里程碑）
```
- `test_riscv_toolchain` 失败基本都是 Step 4 环境变量没生效。
- 编译器相关验证加新鲜编译开关：
  ```bash
  TILELANG_DISABLE_CACHE=1 pytest -q testing/python/riscv/TileKernels
  ```
- TileKernels 的 `conftest.py` 需要 `TILELANG_RISCV_LLVM_ROOT` 指向真实工具链，否则整目录 skip。
- 扩展项（更深，可选）：
  ```bash
  pytest -q testing/python/riscv/test_riscv_jit_runtime.py
  pytest -q testing/python/riscv/test_riscv_artifact_export.py
  TILELANG_DISABLE_CACHE=1 pytest -q testing/python/riscv/TileOps
  ```

### 批量导出 MLIR 产物
```bash
cd ~/tilelang-riscv
pytest -q testing/python/riscv/codegen_ops --riscv-dump-mlir ~/tl_exp/mlir_dump
ls ~/tl_exp/mlir_dump | head
```
（`--riscv-dump-mlir` 是 `codegen_ops/conftest.py` 注册的选项，内部设置环境变量 `TILELANG_RISCV_DUMP_MLIR_DIR`。）

---

## Step 8 — 最小实验：亲手跑通并看穿降级链

### 8.1 用**已验证可用**的最小 kernel（`alloc_shared` + `T.copy` 形态）

> 注意：RISC-V 后端不跑 layout inference，**不是所有 kernel 形态都支持**。
> 下面这个来自 `testing/python/riscv/test_riscv_jit_runtime.py`，是已知可用的最简形态。
> 不要一上来用 `T.Parallel` elementwise + 复杂的 `T.Kernel`，容易撞到未支持路径。

```bash
mkdir -p ~/tl_exp && cat > ~/tl_exp/exp.py <<'EOF'
import torch, numpy as np
import tilelang, tilelang.language as T

N = 8

@T.prim_func
def tile_copy(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((N,), "float32")
        T.copy(A, A_shared)
        T.copy(A_shared, B)

kernel = tilelang.compile(tile_copy, out_idx=[1], target="riscv")
data = torch.arange(N, dtype=torch.float32)
out = kernel(data)
print("input :", data.tolist())
print("output:", out.tolist())
torch.testing.assert_close(out, data)
print("numeric OK")

# 取回 MLIR 源（get_kernel_source 返回带 linalg 的 MLIR 文本）
mlir = kernel.get_kernel_source()
print("---- MLIR (head) ----")
print("\n".join(mlir.splitlines()[:25]))
EOF
```
运行：
```bash
cd ~/tilelang-riscv && python ~/tl_exp/exp.py
```
判断：打印 `numeric OK` 且能看到含 `func.func @tile_copy`、`linalg.*`、`memref`、`scf.*` 的 MLIR。
数值通过 = 编译→LLVM→原生执行全链路成立。

### 8.2 逐级落盘观察每一跳（`libgen` 的 emit 系列，签名已核实）

`tilelang/jit/adapter/riscv/libgen.py` 提供（传入 MLIR 字符串即可）：
- `emit_mlir(value, path)` → 原 MLIR
- `emit_llvm_ir(value, path)` → LLVM IR
- `emit_asm(value, path, triple=...)` → RISC-V 汇编
- `emit_object(value, path, triple=...)` → 目标文件

默认 triple：`riscv64-unknown-linux-gnu`（可用环境变量 `TILELANG_RISCV_TRIPLE` 覆盖）。

```bash
cat > ~/tl_exp/emit.py <<'EOF'
import tilelang, tilelang.language as T
from tilelang.jit.adapter.riscv import libgen
import os
os.makedirs(os.path.expanduser("~/tl_exp/out"), exist_ok=True)
D = os.path.expanduser("~/tl_exp/out")

N = 8
@T.prim_func
def tile_copy(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((N,), "float32")
        T.copy(A, A_shared)
        T.copy(A_shared, B)

kernel = tilelang.compile(tile_copy, out_idx=[1], target="riscv")
mlir = kernel.get_kernel_source()      # 带 linalg 的 MLIR

libgen.emit_mlir(mlir,       f"{D}/1_kernel.mlir")
libgen.emit_llvm_ir(mlir,    f"{D}/2_kernel.ll")
libgen.emit_asm(mlir,        f"{D}/3_kernel.s")
libgen.emit_object(mlir,     f"{D}/4_kernel.o")
print("wrote 4 stages to", D)
EOF
cd ~/tilelang-riscv && python ~/tl_exp/emit.py
ls -la ~/tl_exp/out
```
逐级观察：
```bash
head -40 ~/tl_exp/out/1_kernel.mlir   # linalg / memref / scf 高层结构化
head -40 ~/tl_exp/out/2_kernel.ll     # LLVM IR（llvm.* / 指针运算）
head -40 ~/tl_exp/out/3_kernel.s      # RISC-V 汇编（关注是否出现向量指令 v* 与 -march 相关）
```
判断：能看到从**结构化 MLIR → LLVM dialect → LLVM IR → RISC-V 汇编**的形态逐级变化。这是"看懂架构"最直观的一步。
若汇编里出现 `vsetvli` / `vle32.v` 等，说明该 kernel 走到了 RVV 向量路径。

### 8.3 对照源码确认"精简流水线"

```bash
cd ~/tilelang-riscv
sed -n '1,15p;135,160p;248,258p' tilelang/engine/phase.py
```
判断：RISC-V 分支里**没有** `LayoutInference` / `InjectSoftwarePipeline` / `VectorizeLoop`。
这解释了为什么 GPU 上的某些优化和 kernel 形态在 riscv 上不生效。

再确认后端注册点：
```bash
grep -n "TVM_REGISTER_TARGET_KIND\|build.tilelang_riscv" src/target/rt_mod_riscv.cc
```

---

## Step 9 — 架构走读顺序（由外到内，配合 Step 8 产物）

1. `tilelang/engine/phase.py:139` — RISC-V 专属 pass 流水线（先看它跳过了什么）
2. `tilelang/engine/lower.py:317` 附近 — riscv 短路分支如何调 `target.build.tilelang_riscv` 并取回 MLIR 源
3. `src/target/rt_mod_riscv.cc` — 注册 target kind `"riscv"` 与 build 函数（~130 行，好读）
4. `src/target/codegen_riscv.cc` — `TIRToMLIRLowerer` 主体（~12.9k 行；先看 `AddFunction`/`Finish` 与类型/intrinsic 辅助）
5. `tilelang/tladapter/toolchain.py` — 工具链自动发现（`resolve_llvm_root` 等）
6. `tilelang/jit/adapter/riscv/libgen.py` → `wrapper.py` → `adapter.py` — MLIR→LLVM→`.so` 链接与 ctypes 执行
7. `tilelang/utils/target.py` — `riscv` target 名称、`normalize_riscv_target`、`TILELANG_RISCV_MLIR_MODE` 自动选择

---

## 排错速查

| 现象 | 处理 |
|---|---|
| CMake 报 LLVM/MLIR 24.0 not found | 检查 `TILELANG_RISCV_LLVM_ROOT` 是否含 `lib/cmake/{llvm,mlir}/*Config.cmake`；否则下载 Buddy MLIR `release/v0.1.0` 的 RISC-V 资产或源码构建 |
| configure 报 `-lgcc_s` | 把工具链 `lib64/lp64d` 加进 `LIBRARY_PATH` |
| `test_riscv_toolchain` 失败 | Step 4 环境变量没生效；确认 `mlir-opt/llc/clang/ld.lld` 可被发现 |
| Z3 lookup error | 设 `Z3_ROOT` 指向含 `include/`+`lib/libz3.so` 的前缀 |
| `tvm_ffi` 导入失败 | 重装 `3rdparty/tvm/3rdparty/tvm-ffi`（Step 5） |
| 首导慢 / 试图编 Torch 扩展 | 保持 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` |
| 改了编译器仍跑旧结果 | `export TILELANG_DISABLE_CACHE=1` |
| 提示 No tests were collected | 多半是整目录 skip（缺 LLVM root 或子模块），检查 `TILELANG_RISCV_LLVM_ROOT` 与 `3rdparty/TileKernels` |
| 想找 RISC-V 示例 | **没有**：`examples/` 全是 GPU 示例；验证载体是 `testing/python/riscv/TileKernels`、`TileOps` |
| 找不到 `run_codegen_ops.py` | 文档 stale，该文件**不存在**；直接用 pytest |

---

## 交付物（你自己产出）

1. Step 2 的环境清单（LLVM root / CC / Z3 / 子模块状态）
2. Step 5 构建成功的完整命令记录
3. Step 7 六档冒烟测试通过情况
4. Step 8 一个最小 kernel 的完整产物链（MLIR → LLVM IR → asm → 执行数值）
5. Step 9 的走读笔记
