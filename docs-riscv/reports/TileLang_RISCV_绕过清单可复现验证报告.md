# TileLang-RISC-V 绕过清单可复现验证报告

> 验证日期：2026-10-21
> 机器：`openeuler-riscv64`（openEuler 24.03，64 核 RISC-V，glibc 2.38，gcc 12/14，Python 3.11.6）
> 方法：**新拉干净代码** `~/tilelang-riscv-clean`（HEAD = `32fce5ed`），把我们之前踩坑总结的修复**系统地重做一遍**，验证绕过清单是否可复现。
> **最终结果：TileKernels 48/48 全部通过（0 失败）** —— 在真 RISC-V 机器上端到端跑通。

---

## 一、核心结论

### 1. bf16 数值错误的定性：**链接问题，不是编译器 bug**

| 证据 | 说明 |
|---|---|
| 修复前 | TileKernels：13 failed, 35 passed（13 个 bf16 全数值错） |
| 修复后 | **TileKernels：48 passed, 0 failed** |
| 单独验证 | expand_kernel / engram_gate / post_kernel 三个之前失败的 bf16 测试全部通过 |

**结论**：bf16 数值错误的根因是 **compiler-rt 的 `__extendhfsf2`/`__truncsfhf2` 没有被链接进 kernel 的 `.so`**。当这些 bf16/fp16 转换函数缺失时，运行时算出垃圾值（nan、1e23、1e38）。**链接修对后，48 个全部恢复正确。**

**这不是 `codegen_riscv.cc` 的编译器逻辑 bug**，是工具链集成问题。

### 2. aa3a6016ae（上游 cibuildwheel 修复）与我们问题**无关**

| 证据 | 说明 |
|---|---|
| 它在 `tile-ai/tilelang`（上游），不在 `RuyiAI-Stack/tilelang-riscv`（我们的 fork） | `git merge-base --is-ancestor aa3a6016ae HEAD` → `Not a valid object name` |
| fork 的 RISC-V CI 是 `ci-buddy-mlir.yml`，**不用 cibuildwheel** | grep 无命中 |
| fork 的 `dist.yml` 用 `cibuildwheel@v3.4`，**没同步到 4.3** | 该修复（4.2→4.3）没吸收进 fork |
| cibuildwheel 的 riscv 修复是"打包 wheel 时回退到系统 python3" | 是发布流程兼容，不是运行时正确性 |
| 我们的错误无一是 cibuildwheel 报的 | scikit_build_core/glibc/R_RISCV_JAL/crtbeginS/__truncsfhf2/numpy |

**结论**：这条上游修复属于"打包 wheel"环节，我们的是"运行时跑不起来"，两个不同环节，且 fork 没吸收它。**它解决不了我们的问题。**

### 3. 绕过清单**可复现**

新拉干净代码后，所有修复**全部有效**，且大部分依赖（torch、z3、compiler-rt、LLVM）在服务器上**保留下来可直接复用**，不用重装。

---

## 二、绕过清单（最终版，已验证可复现）

### 阶段 A：工具链（一次性，之后可复用）

| 项 | 做法 | 本次是否需重做 |
|---|---|---|
| LLVM/MLIR 24 | 源码编 `RuyiAI-Stack/llvm-project` @3845a205；给 `TosaProfileCompliance.cpp` 加 `-Os` CMake 补丁（避 R_RISCV_JAL） | 复用 `~/llvm-project/build` |
| compiler-rt builtins | 补编 `ninja builtins`，产出 `libclang_rt.builtins.a`（含 `__extendhfsf2`/`__truncsfhf2`） | 复用 |
| z3 | 后台 `pip install --user z3-solver`（源码编，~30 分钟） | 复用（已装 4.15.4） |
| torch | `curl` 下 kmtea `torch-2.4.1-cp311-...riscv64.whl` → `unzip` 解压进 site-packages | 复用（已装 2.4.1） |

### 阶段 B：Python 依赖

```bash
python3 -m pip install --user \
  scikit-build-core 'cython==3.1.8' setuptools_scm build packaging wheel \
  cloudpickle ml-dtypes psutil tqdm typing-extensions pytest patchelf requests
python3 -m pip install --user 'numpy<2'    # 必须 <2，否则 torch ABI 崩
python3 -m pip install --user filelock sympy networkx jinja2 fsspec
```

### 阶段 C：构建 tilelang

```bash
# 1. 进 tmux（防断线）
tmux new -s build

# 2. 设环境变量（关键，每次新 shell 都要）
export TILELANG_RISCV_LLVM_ROOT=/home/wangjiayue/llvm-project/build
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 -L/home/wangjiayue/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu -lclang_rt.builtins"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DTILELANG_RISCV_MLIR_MODE=ON"

# 3. 装 vendored tvm-ffi（顺序不能反）
cd ~/tilelang-riscv-clean
python3 -m pip install --user ./3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps

# 4. 构建 tilelang
python3 -m pip install --user -e . --no-build-isolation -v

# 5. ⚠️ 新坑：pip 装 tilelang 会升级 numpy，必须事后降回
python3 -m pip install --user 'numpy<2'
```

### 阶段 D：验证

```bash
python3 -c "import tilelang; print(tilelang.__file__)"
python3 -c "import tilelang.tladapter._native; print('native adapter ok')"
python3 -c "from tilelang import tvm; print(tvm.ffi.get_global_func('target.build.tilelang_riscv'))"
```

### 阶段 E：测试

```bash
cd ~/tilelang-riscv-clean   # 必须在仓库根目录

# 冒烟测试
python3 -m pytest -q testing/python/riscv/test_riscv_target_parse.py \
  testing/python/riscv/test_riscv_toolchain.py \
  testing/python/riscv/test_riscv_tladapter_pipeline.py \
  testing/python/riscv/test_riscv_cache.py

# TileKernels（端到端，48 个）
TILELANG_DISABLE_CACHE=1 python3 -m pytest -q testing/python/riscv/TileKernels

# TileOps（更深，415 个）
TILELANG_DISABLE_CACHE=1 python3 -m pytest -q testing/python/riscv/TileOps
```

---

## 三、本次验证发现的新坑

| 坑 | 现象 | 修法 |
|---|---|---|
| **pip 装 tilelang 会升级 numpy** | `Successfully installed numpy-2.4.6 tilelang-...`，torch 报 `_ARRAY_API not found` | 装完 tilelang 后**立刻** `pip install --user 'numpy<2'` |
| tvm-ffi 是嵌套子模块，浅克隆可能拉不全 | `3rdparty/tvm/3rdparty/tvm-ffi/pyproject.toml` 缺失 | `git submodule update --init --recursive`，或单独 `git submodule update --init --recursive 3rdparty/tvm/3rdparty/tvm-ffi` |

---

## 四、最终环境变量（写进 ~/.bashrc）

```bash
# TileLang-RISC-V 环境（已验证可复现）
export TILELANG_RISCV_LLVM_ROOT=/home/wangjiayue/llvm-project/build
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 -L/home/wangjiayue/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu -lclang_rt.builtins"
```

## 关键依赖版本（已验证可复现）

| 依赖 | 版本 | 备注 |
|---|---|---|
| LLVM/MLIR | 24.0.0git | 自编（RuyiAI-Stack/llvm-project @3845a205），含 TOSA -Os 补丁 |
| compiler-rt | builtins | 补编，含 `__extendhfsf2`/`__truncsfhf2` |
| torch | 2.4.1 | kmtea riscv64 wheel（unzip 解压） |
| numpy | 1.26.4（<2） | 装 tilelang 后必须降回 |
| cython | 3.1.8 | 降级避开 3.3 的 3.9 要求 |
| z3-solver | 4.15.4 | 源码编译 |
| tvm-ffi | vendored | `3rdparty/tvm/3rdparty/tvm-ffi` |

---

## 五、本次验证的意义

1. **绕过清单确实可复现** —— 新拉干净代码，按清单做一遍，就能跑通。
2. **bf16 不是编译器 bug** —— 是 compiler-rt 链接问题，修对后 48/48 全过。
3. **这是官方公开状态里没有的成果** —— README 说官方只在 x86 上跑过 48 个，**SG2044 从未原生复跑**。本次在真 RISC-V 机器上把 48 个全部跑通，等于替项目做了"原生验证"。
4. **aa3a6016ae 无关** —— 不用指望上游那个 cibuildwheel 修复。

---

## 附：测试结果时间线

| 时间 | 测试 | 结果 |
|---|---|---|
| 2026-10-17 | TileKernels（修复前） | 13 failed, 35 passed |
| 2026-10-17 | TileOps（修复前，路径错） | 149 failed, 266 passed |
| 2026-10-21 | TileKernels（修复后，干净代码） | **48 passed, 0 failed** |
