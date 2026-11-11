# TileLang-RISC-V 真机上手踩坑修复记录

> 机器：`openeuler-riscv64`（openEuler 24.03，64 核 RISC-V，glibc 2.38，gcc 12/14，Python 3.11.6）
> 目标：跑通 tilelang-riscv 的构建与测试
> 结果：从"裸系统 + 一堆错"到 TileKernels 35/48 通过，TileOps 266/415 通过
> 记录所有遇到的环境问题、根因、修法，便于复用

---

## 第一阶段：环境与构建（Config → Build）

### 1. 裸系统：什么都没有
**现象**：`python` 不存在，依赖全无。
**根因**：openEuler 默认只有 `python3`（3.11.6），不设 `python` 别名；scikit-build-core、cython、numpy、torch、z3 等全缺。
**修法**：一律用 `python3`；`pip install --user` 逐批装齐。

### 2. glibc 版本不足，预编译包不可用
**现象**：想用 Buddy MLIR 的 riscv64 预编译包，报 glibc 不符。
**根因**：预编译包要 glibc ≥ 2.39，本机 2.38。
**修法**：放弃预编译，**源码编译 LLVM/MLIR 24**（`RuyiAI-Stack/llvm-project` 的 `riscv` 分支，提交 `3845a205`）。

### 3. LLVM 编译失败：R_RISCV_JAL 重定位超范围
**现象**：`relocation truncated to fit: R_RISCV_JAL ... 1070556 is not in [-1048576, 1048575]`
**根因**：`TosaProfileCompliance.cpp` 在 `-O3` 下生成的函数机器码超过 ±1MB，而 RISC-V 的 `JAL` 指令只能跳 ±1MB。**RISC-V 特有**，x86 不会出现。
**修法**：给该文件加 CMake 补丁强制 `-Os`（缩小代码）：
```
set_source_files_properties(TosaProfileCompliance.cpp PROPERTIES COMPILE_OPTIONS "-Os")
```
追加到 `mlir/lib/Dialect/Tosa/Transforms/CMakeLists.txt`。

### 4. 断线杀进程：反复丢进度
**现象**：`client_loop: send disconnect`，LLVM/z3/tvm-ffi 编译进度全没。
**根因**：长任务在**裸 SSH 前台**跑，SSH 一断进程收到 SIGHUP 被杀。
**修法**：**所有长命令进 `tmux`**（`tmux new -s xxx`，断了 `tmux attach -t xxx` 回来）；或 `nohup ... &` + 看日志文件。

### 5. Cython 3.3 vs cp38 冲突
**现象**：`#error "Cython 3.3 requires the Python Limited API version to be 3.9 or greater."`
**根因**：项目 `pyproject.toml` 写死 `wheel.py-api = "cp38"`，但你装的 Cython 3.3 不再支持 <3.9。
**修法**：降级 `cython==3.1.8`（项目要求 `>=3.1.0`）。

### 6. CMake 缓存钉死旧配置
**现象**：改了环境变量/Cython 版本，构建还是老错。
**根因**：`build/CMakeCache.txt` 里钉死了旧的 `SKBUILD_SABI_VERSION=3.8`、空的 `TILELANG_RISCV_LLVM_ROOT`。
**修法**：重跑前**删缓存**：
```bash
rm -rf build/CMakeCache.txt build/CMakeFiles build/tilelang_cython_wrapper.cpp
```

### 7. `TILELANG_RISCV_LLVM_ROOT` 漏设 → 占位后端
**现象**：构建"成功"但跑测试全 skip，`llvm_root: None`。
**根因**：环境变量只在某个 shell export 过，CMake 是在**没 export 的 shell** 配的，导致 `TILELANG_RISCV_LLVM_ROOT:PATH=` 空，MLIR 后端没启用。
**修法**：构建前必须设全变量，并清缓存重配。

### 8. 系统 LLVM 版本不够
**现象**：想用系统 LLVM 省事。
**根因**：系统只有 LLVM 17.0.6，项目要 **24.0**，且没有 MLIR。
**修法**：只能自编 LLVM 24。

### 9. z3 无 riscv64 轮子
**现象**：`pip install z3-solver` 卡几十分钟。
**根因**：官方/kmtea/openEuler 源都没有 riscv64 的 z3 预编译，只能源码编。
**修法**：后台 `nohup pip install --user z3-solver &`，等 20-60 分钟。

### 10. torch 无 riscv64 轮子 + pip 平台标签不匹配
**现象**：`No matching distribution found for torch`，加 `--platform` 又报必须配 `--target`。
**根因**：pip 只认 `linux_riscv64` 标签，不认源里的 `manylinux_2_35_riscv64`。
**修法**：直接 `curl` 下 kmtea 源的 wheel（`torch-2.4.1-cp311-cp311-manylinux_2_35_riscv64.whl`），再 **`unzip` 解压进 site-packages** 绕过 pip 平台检查：
```bash
curl -L -o torch.whl 'https://github.com/KumaTea/pytorch-riscv64/releases/download/2.4/torch-2.4.1-cp311-cp311-manylinux_2_35_riscv64.whl'
unzip -o torch.whl -d ~/.local/lib/python3.11/site-packages/
```

### 11. numpy 2.x vs torch 1.x ABI 冲突
**现象**：`Failed to initialize NumPy: _ARRAY_API not found`，torch 崩。
**根因**：torch 2.4.1 按 numpy 1.x 编译，装了 numpy 2.4.6（2.x 改了 ABI）。
**修法**：降级 `numpy<2`。

### 12. compiler-rt 没编 → bf16/fp16 内置函数缺失
**现象**：`undefined symbol: __extendhfsf2 / __truncsfhf2`
**根因**：自编 LLVM 时 `LLVM_ENABLE_RUNTIMES` 为空，没编 compiler-rt；系统 gcc 12 的 libgcc 也没有这些函数。
**修法**：补编 compiler-rt builtins：
```bash
cmake -DLLVM_ENABLE_RUNTIMES="compiler-rt" -DCOMPILER_RT_BUILD_BUILTINS=ON -DCOMPILER_RT_BUILD_SANITIZERS=OFF .
ninja -j 32 builtins
```

### 13. clang 链接怪癖：`crtbeginS.o` / `-lgcc_s` 找不到
**现象**：链接时报 `/usr/bin/ld: cannot find crtbeginS.o`
**根因**：openEuler + 自编 clang 对 `--gcc-toolchain=/usr` 的解析有问题，传给 ld 只给文件名不给路径。
**修法**：改用 `--gcc-install-dir` 显式指定，经 tilelang 的 `TILELANG_RISCV_CLANG_FLAGS` 注入：
```bash
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12"
```

### 14. compiler-rt 库路径给错
**现象**：`-lclang_rt.builtins` 找不到。
**根因**：库实际在 `lib/clang/24/lib/riscv64-unknown-linux-gnu/`，但 `-L` 给成了 `lib/linux/`（不存在）。
**修法**：用 `find` 找真实路径，**不要猜**：
```bash
find ~/llvm-project/build -name 'libclang_rt.builtins*.a'
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 -L/home/wangjiayue/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu -lclang_rt.builtins"
```

---

## 第二阶段：测试（Testing）

### 15. 缓存把"坏的 .so"记住，重跑不生效
**现象**：改了链接参数/清了缓存，失败数不变。
**根因**：链接出错时链出的坏 .so 被 tilelang 缓存了，重跑直接复用。
**修法**：**改编译器/链接相关配置后，必须 `TILELANG_DISABLE_CACHE=1`** 强制重编。

### 16. bf16 数值全错（13/15 失败的共同特征）
**现象**：`Tensor-likes are not close`，值为 `1e23`/`nan`/`1e38` 天文数字，100% 元素不匹配。
**根因**：bf16 张量的降级链中，`__extendhfsf2`/`__truncsfhf2` 等转换函数的运行时行为与期望不符——**根因是 compiler-rt 没被正确链接进 kernel 的 .so**（见 #12、#14）。
**修法**：确保 compiler-rt 路径对、清缓存重跑。

### 17. fp16 链接失败 vs bf16 数值错（TileOps 的成对规律）
**现象**：TileOps 里同一算子成对失败：
```
test_xxx[dtype0-...] → undefined symbol __truncsfhf2   (fp16 链接缺符号)
test_xxx[dtype1-...] → Tensor-likes are not close!     (bf16 数值错)
```
**根因**：`dtype0=fp16`、`dtype1=bf16`，各自指向 compiler-rt 链接不全的两面。
**修法**：同 #16。

### 18. fp8 特殊值填充不支持（已知限制，非可修）
**现象**：`RuntimeError: "fill_cpu" not implemented for 'Float8_e4m3fn'`
**根因**：torch 在 riscv64 不支持 fp8 张量的 fill 操作。
**修法**：**这是项目官方 CI 也有的限制**，不是环境问题，跳过即可。

### 19. 测试必须在仓库根目录跑
**现象**：`No module named tilelang` 或用了系统装的版本。
**根因**：`testing/conftest.py` 会把仓库内 `tilelang/` 插到 `sys.path` 最前，但只有在仓库根目录跑才生效。
**修法**：`cd ~/tilelang-riscv` 再跑 pytest。

### 20. 无害警告误读为错误
**现象**：`Error in cpuinfo: processor architecture is not supported in cpuinfo`、cpuinfo 杂音、tempfile ResourceWarning。
**根因**：pytorch 在 riscv64 的已知无害警告、Python 的资源清理提示。
**修法**：**不影响测试结果**，忽略。

---

## 最终环境变量（写进 ~/.bashrc）

```bash
# TileLang-RISC-V 环境
export TILELANG_RISCV_LLVM_ROOT=/home/wangjiayue/llvm-project/build
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 -L/home/wangjiayue/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu -lclang_rt.builtins"
```

## 关键依赖版本

| 依赖 | 版本 | 备注 |
|---|---|---|
| LLVM/MLIR | 24.0.0git | 自编（RuyiAI-Stack/llvm-project @3845a205） |
| torch | 2.4.1 | kmtea riscv64 wheel（unzip 解压安装） |
| numpy | 1.26.x（<2） | 降级避免与 torch 1.x ABI 冲突 |
| cython | 3.1.8 | 降级避开 3.3 的 3.9 要求 |
| z3-solver | 4.15.4 | 源码编译 |
| compiler-rt | builtins | 补编（含 __extendhfsf2/__truncsfhf2） |

---

## 经验总结（一句话版）

**在 riscv64 这种非 x86 平台上做开发，最大的敌人不是代码，是"生态不完整"**：

1. **预编译包常缺**（torch/z3/LLVM）→ 备好"源码编"的方案
2. **平台标签/路径别猜** → 一律用 `find` 查真实路径
3. **版本要对齐**（cython/numpy/LLVM）→ 用项目声明的区间，别追最新
4. **长任务防断线** → 一律 tmux
5. **改链接/编译器后** → 必须 `TILELANG_DISABLE_CACHE=1` 清缓存
6. **官方 CI 是 x86** → riscv64 上的链接/内置函数怪癖，官方不会替你踩
