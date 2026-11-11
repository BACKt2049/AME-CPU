# TileLang-RISC-V 环境问题全量汇总

> 机器：`openeuler-riscv64`（RV64GCV：含 `v`/`zfh`/`zvfh`，**无 `zfbfmin`/`zvbfmin`** → bf16 全靠软件 libcall）
> 系统：openEuler 24.03，glibc 2.38，gcc 12.2，Python 3.11.6，64 核
> 代码：`RuyiAI-Stack/tilelang-riscv` @ `32fce5ed`
> 工具链：自编 LLVM/MLIR 24.0.0git（`RuyiAI-Stack/llvm-project` @3845a205）
> 最终结果：**823 用例全通过（0 失败）**，含 TileKernels 48 + TileOps 415

---

## 零、总览

共记录 **22 项**环境问题，按归属分：

| 归属 | 数量 | 说明 |
|---|---|---|
| **项目本身** | 13 | 代码硬编码、契约缺失、CI 覆盖不全、ABI 路径不一致 |
| **上游生态** | 6 | 无 riscv64 wheel、无预编译、PyTorch CPU 限制 |
| **操作方式** | 3 | 断线、环境变量未持久化、命令格式假设 |

按层分：

```
A. 外部生态缺位        6 项  ← 上游
B. 工具链与 ABI        7 项  ← 项目（技术核心，含新发现的路径级 ABI 不一致）
C. 构建系统            4 项  ← 项目
D. 测试基建            3 项  ← 项目
E. 文档与 CI 验证      2 项  ← 项目（真正的根）
```

---

## A. 外部生态缺位（上游问题，项目无 fallback）

### A1. PyTorch 无 riscv64 wheel
- **现象**：`pip install torch` → `No matching distribution found`
- **根因**：PyPI 只为 x86_64/aarch64/ARM64 发 wheel
- **修法**：`curl` 下第三方 wheel（kmtea `torch-2.4.1-cp311-...manylinux_2_35_riscv64.whl`，147MB）→ `unzip` 解压进 site-packages（绕 pip 平台标签）
- **补充坑**：pip 只认 `linux_riscv64` 标签，不认源里的 `manylinux_2_35_riscv64`；加 `--platform` 又强制要求 `--target`
- **状态**：✅ 已解决

### A2. z3-solver 无 riscv64 wheel
- **现象**：`pip install z3-solver` 卡在 `Building wheel` 数十分钟
- **根因**：官方/kmtea/openEuler 源都没有 riscv64 预编译，只能源码编
- **修法**：后台 `nohup pip install --user z3-solver &`，约 20–60 分钟（64 核并行）
- **状态**：✅ 已解决（z3 4.15.4）

### A3. LLVM/MLIR 24 无 riscv64 预编译包
- **现象**：Buddy MLIR `release/v0.1.0` 的预编译包要 glibc ≥2.39，本机 2.38
- **根因**：官方只提供 x86_64 与 riscv64 两种资产，但 riscv64 那个要求 glibc 2.39
- **修法**：源码编 LLVM/MLIR 24（`RuyiAI-Stack/llvm-project` @3845a205），数小时
- **附带问题**：见 B1（R_RISCV_JAL）
- **状态**：✅ 已解决

### A4. 系统 gcc 12 的 libgcc 缺 fp16/bf16 内置函数
- **现象**：`nm libgcc.a | grep truncsfbf2` → 空
- **根因**：gcc 12 的 libgcc 不含 `__truncsfhf2`/`__extendhfsf2`/`__truncsfbf2`/`__extendbfsf2`
- **修法**：补编 compiler-rt builtins（见 B3）
- **状态**：✅ 已解决

### A5. PyTorch CPU 不支持 fp8 的 `fill`
- **现象**：`RuntimeError: "fill_cpu" not implemented for 'Float8_e4m3fn'`
- **根因**：PyTorch 2.4.1 的 CPU 后端未实现 fp8 的 fill（**测试造数据阶段就崩，不是算子问题**）
- **修法**：测试里 `torch.full(dtype=fp8)` → 先 fp32 造再 `.to(fp8)`
- **影响**：TileOps 6 个用例
- **状态**：✅ 已绕过（测试层补丁，备份 `.orig`）

### A6. 系统 LLVM 版本不匹配
- **现象**：系统只有 LLVM 17.0.6，且无 MLIR
- **根因**：`cmake/TileLangRISCVMLIR.cmake:77-78` 硬要求 `find_package(LLVM 24.0 REQUIRED)`
- **修法**：不能用系统 LLVM，必须自编 24
- **状态**：✅ 已解决（与 A3 同一动作）

---

## B. 工具链与 ABI（项目技术核心）

### ★B0【新发现·最重要】两条降级路径的 ABI 不一致

**这是本次调查最有价值的发现，也解释了此前"调用方读 a0 还是 fa0"的矛盾。**

| 路径 | 代码位置 | 调用方式 | 默认 ABI | float 参数 | bf16 返回值 |
|---|---|---|---|---|---|
| **object/asm 路径**（freestanding / qemu） | `libgen.py:363-396` 的 `_emit_llc_artifact` | `llc -mtriple=riscv64-unknown-linux-gnu -filetype=asm\|obj` —— **只有 `-mtriple`，无 `-mattr`、无 `-mabi`** | 无 `+d` 扩展 → **软浮点（lp64）** | **`a0`**（实测汇编：`lw a0, 0(a0)`） | **`a0`**（实测：`sh a0, 0(a1)`） |
| **host .so 路径**（实际跑测试用的） | `wrapper.py:943-967` 的 `build_host_shared_library` | `clang -x ir -target <triple>` | triple 默认 → **lp64d 硬浮点** | **`fa0`** | 视 LLVM 版本 |

**后果**：
1. 同一个 kernel 走两条路径产出 **ABI 不同**的机器码；
2. 若 object 路径的 `.o` 被 lp64d 的 helper / libgcc 链接 → **跨 ABI 错配**（float 参数期望 fa0，实际传在 a0）；
3. `_infer_riscv_abi_flag`（`wrapper.py:748-761`）只在 freestanding 路径用，且从 obj 的 arch tag 反推 `-mabi`，**无法修正已按软浮点编译的 obj**；
4. 这解释了为什么"调用方从 a0 读"（我实测 object 路径）与"调用方从 fa0 读"（你观察 host 路径）都成立——**是两个不同路径**。

**修法方向**：
- `llc` 调用补 `-mattr=+m,+a,+f,+d,+c`（或 `-mabi=lp64d`），使两条路径 ABI 一致；
- 或在 `_emit_llc_artifact` 里统一用 `clang -x ir -c` 而不是裸 `llc`；
- **必须实测确认**：同一 MLIR 经两条路径产出的 `.s` 在 `__truncsfbf2` 调用点是否一致。

**状态**：🔴 已定位，**未修**（需服务器验证）

---

### B1. `llc` 未传 `-mattr`（B0 的根）
- **位置**：`libgen.py:381-388`
- **现象**：汇编里 float 参数走整数寄存器 `a0`
- **根因**：`llc -mtriple=riscv64-unknown-linux-gnu` 不自动启用 `+d`（double float）扩展，退化为软浮点
- **状态**：🔴 未修

---

### B2. compiler-rt builtins 完全没被链接
- **现象**：`undefined symbol: __truncsfhf2` / bf16 数值全错（nan、1e38 等天文数字）
- **位置**：`wrapper.py:943-959` —— 链接命令只有 `-lmlir_c_runner_utils -lmlir_runner_utils -lm`
- **根因**：
  - `_riscv_gcc_runtime_library_dirs`（`:685-718`）**只找 `libgcc_s.so`**，连 `libgcc.a` 都不看，完全不知道 compiler-rt 存在；
  - `_riscv_clang_flags`（`:721-745`）通过 `--gcc-toolchain` 指向 gcc → clang 自动加的 `-lgcc` 只命中不含这些符号的 libgcc；
  - `resolve_tool("clang")` 优先用 `TILELANG_RISCV_LLVM_ROOT/bin/clang`（`tladapter/toolchain.py:93-109`），而这个自编 clang 资源目录里没有 builtins（当初 `LLVM_ENABLE_RUNTIMES` 为空）。
- **修法（我们用的）**：`export TILELANG_RISCV_CLANG_FLAGS="... -L.../lib/clang/24/lib/riscv64-unknown-linux-gnu -lclang_rt.builtins"`
- **验证**：补上后 **一行编译器代码没改**，463/463 → 823/823 全绿
- **状态**：✅ 已解决（环境变量层面绕过）

---

### B3. compiler-rt 本身没编
- **现象**：想链 builtins 却没有库
- **根因**：编 LLVM 时 `LLVM_ENABLE_RUNTIMES=` 空，`LLVM_TOOL_COMPILER_RT_BUILD=FALSE`
- **修法**：增量补编
  ```bash
  cmake -DLLVM_ENABLE_RUNTIMES="compiler-rt" -DCOMPILER_RT_BUILD_BUILTINS=ON \
        -DCOMPILER_RT_BUILD_SANITIZERS=OFF -DCOMPILER_RT_BUILD_XRAY=OFF \
        -DCOMPILER_RT_BUILD_LIBFUZZER=OFF -DCOMPILER_RT_BUILD_PROFILE=OFF .
  ninja -j 32 builtins
  ```
- **产出**：`lib/clang/24/lib/riscv64-unknown-linux-gnu/libclang_rt.builtins.a`（含 `__extendhfsf2`/`__truncsfhf2`/`__extendbfsf2`/`__truncdfbf2`/`__truncsfbf2`）
- **状态**：✅ 已解决

---

### B4. builtins 库路径猜错（`lib/linux` vs `lib/riscv64-unknown-linux-gnu`）
- **现象**：`cannot find -lclang_rt.builtins`
- **根因**：compiler-rt 按 target triple 命名目录，不是固定的 `linux`
- **修法**：`find ~/llvm-project/build -name 'libclang_rt.builtins*.a'` 查真实路径，不猜
- **状态**：✅ 已解决

---

### B5. `-shared` 吞掉未定义符号，失败推迟到运行时
- **位置**：`wrapper.py:945`（`clang -shared`），缺 `-Wl,--no-undefined`
- **现象**：编译"成功"，运行到 `ctypes.CDLL`（`wrapper.py:987`）才报 `OSError: undefined symbol`
- **后果**：坏 `.so` 已写入缓存（见 D1），后续修了环境仍然复现
- **修法**：链接命令加 `-Wl,--no-undefined`
- **状态**：🔴 未修（我们用 `TILELANG_DISABLE_CACHE=1` 绕过）

---

### B6. gcc root 写死 `/opt/gcc-native`
- **位置**：`wrapper.py:679`
- **现象**：非 SG2044 机器上 `gcc_root=None` → 无 `--gcc-toolchain` → `crtbeginS.o` 找不到
- **修法**：设 `TILELANG_RISCV_GCC_ROOT`，或用 `gcc -print-file-name=crtbeginS.o` 探测
- **状态**：✅ 已绕过

---

### B7. `--gcc-toolchain=/usr` 在 openEuler + 自编 clang 下失效
- **现象**：`/usr/bin/ld: cannot find crtbeginS.o` / `-lgcc_s`
- **根因**：clang 传给 ld 时只给文件名不带路径
- **修法**：改用 `--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12`
- **原始命令**：`clang ... crtbeginS.o ...`（无路径）
- **状态**：✅ 已解决

---

## C. 构建系统

### C1. Cython 无版本上限 → 3.3 与 cp38 冲突
- **位置**：`pyproject.toml:66`（`cython>=3.1.0`，无上限）+ `pyproject.toml:76`（`wheel.py-api = "cp38"`）
- **现象**：`#error "Cython 3.3 requires the Python Limited API version to be 3.9 or greater."`
- **修法**：`pip install --user 'cython==3.1.8'`
- **状态**：✅ 已解决

### C2. CMake 缓存钉死旧配置
- **现象**：改了环境变量/Cython 版本，构建仍报旧错；或 `TILELANG_RISCV_LLVM_ROOT:PATH=` 空导致"占位后端"
- **修法**：`rm -rf build/CMakeCache.txt build/CMakeFiles build/tilelang_cython_wrapper.cpp`
- **状态**：✅ 已解决

### C3. `requirements.txt` 与 `pyproject.toml` 的 riscv64 跳过不一致
- **位置**：`requirements.txt:4` 无 `platform_machine != 'riscv64'`；`pyproject.toml:38` 有
- **后果**：`pip install -r requirements.txt` 在 riscv64 上仍会尝试装 `torch-c-dlpack-ext` → 失败
- **状态**：🔴 未修（我们没有用 requirements.txt 装）

### C4. cibuildwheel 不支持 riscv64
- **位置**：`pyproject.toml:250-271`，`case "$(uname -m)"` 里 `*) exit 1`
- **后果**：riscv64 无法产 wheel
- **状态**：🔴 未修（不影响本地使用）

---

## D. 测试基建

### D1. 缓存键不含工具链指纹 → 坏产物永久命中
- **位置**：`tilelang/cache/kernel_cache.py:235-245`（`_generate_key`）
- **现象**：改了 `TILELANG_RISCV_CLANG_FLAGS`、补编了 compiler-rt，**重跑结果完全不变**
- **根因**：`key_data` 只含 `func` 脚本哈希、`args_repr`、`target`、`pass_configs`、`compile_flags`、`_get_base_key()`（version + `platform.machine()` + `libtilelang.so` 哈希）——**不含** `TILELANG_RISCV_CLANG_FLAGS` / `GCC_ROOT` / `_ABI` / `_LLVM_ROOT` / clang 路径 / builtins 库
- **额外问题**：`_get_base_key` 与 `_get_tilelang_lib_stamp` 都有 `@functools.cache`，**进程内改环境变量也不生效**
- **修法**：`export TILELANG_DISABLE_CACHE=1`（我们全程使用）
- **状态**：✅ 已绕过（每次都关缓存，代价是每次都全量重编）

### D2. conftest 写死作者个人路径
- **位置**：`TileKernels/conftest.py:22-24`、`TileOps/conftest.py:26-27` → `/root/llvm/build`、`/home/liquanyi/tool/llvm`
- **后果**：未设 `TILELANG_RISCV_LLVM_ROOT` 时整目录 skip（不是报错，是**静默跳过**，容易误判"测试通过"）
- **状态**：✅ 已绕过（显式 export）

### D3. 测试必须从仓库根目录跑
- **现象**：`operator_inventory.py` 零匹配、静默无告警
- **根因**：它靠 `<testcase classname="testing.python.riscv.TileOps.test_x">` 归属，只在仓库根目录且目录有 `__init__.py` 时才生成这种点分 classname
- **状态**：✅ 已解决（`run_all_riscv_tests.sh` 里固定 `cd $REPO`）

---

## E. 文档与 CI（真正的根）

### E1. CI 只跑 x86 + 编译路径，运行时不回归
- **位置**：`.github/workflows/ci-buddy-mlir.yml:56`（`ubuntu-latest` = x86_64）、`:111-118`（只跑 5 个编译路径测试）
- **不跑的**：`TileKernels/`、`TileOps/`、`test_riscv_jit_runtime.py`、`test_riscv_cache.py`
- **后果**：B0–B7、D1 全部漏检——**B 层所有问题它都发现不了**
- **状态**：🔴 未修（这是复发的根源）

### E2. 文档 "validated" 与 "仍需原生验证" 自相矛盾
- **位置**：`docs/get_started/BuildOnSG2044.md:3`（"validated native bring-up flow"）vs `README.md:165`（"still need SG2044 native validation"）
- **后果**：读者按文档走，以为已验证，实际全是坑
- **状态**：🔴 未修

---

## 汇总表

| # | 问题 | 层 | 归属 | 状态 |
|---|---|---|---|---|
| **B0** | **两条降级路径 ABI 不一致（llc 软浮点 vs clang 硬浮点）** | 工具链/ABI | 项目 | 🔴 **未修** |
| B1 | `llc` 未传 `-mattr` | 工具链 | 项目 | 🔴 未修 |
| B2 | compiler-rt builtins 未链接 | 工具链 | 项目 | ✅ 绕过 |
| B5 | `-shared` 吞未定义符号 | 链接 | 项目 | 🔴 未修 |
| B6 | gcc root 写死 `/opt/gcc-native` | 运行时 | 项目 | ✅ 绕过 |
| B7 | `--gcc-toolchain` 失效 | 运行时 | 项目 | ✅ 绕过 |
| D1 | 缓存键无工具链指纹 | 测试/缓存 | 项目 | ✅ 绕过 |
| D2 | conftest 写死作者路径 | 测试 | 项目 | ✅ 绕过 |
| E1 | CI 只跑 x86 编译路径 | CI | 项目 | 🔴 未修 |
| E2 | 文档 validated 矛盾 | 文档 | 项目 | 🔴 未修 |
| C1 | Cython 无上限 | 构建 | 项目 | ✅ 解决 |
| C2 | CMake 缓存 | 构建 | 项目 | ✅ 解决 |
| C3 | requirements 不一致 | 构建 | 项目 | 🔴 未修 |
| C4 | cibuildwheel 无 riscv64 | 构建 | 项目 | 🔴 未修 |
| D3 | 须从仓库根跑 | 测试 | 项目 | ✅ 解决 |
| A1 | torch 无 riscv64 wheel | 生态 | 上游 | ✅ 绕过 |
| A2 | z3 无 riscv64 wheel | 生态 | 上游 | ✅ 解决 |
| A3 | LLVM 无 riscv64 预编译 | 生态 | 上游 | ✅ 解决 |
| A4 | libgcc 缺 fp16/bf16 内置 | 生态 | 上游 | ✅ 解决 |
| A5 | PyTorch CPU 无 fp8 fill | 生态 | 上游 | ✅ 绕过 |
| A6 | 系统 LLVM 17 不匹配 | 生态 | 上游 | ✅ 解决 |
| B3 | compiler-rt 未编 | 工具链 | 项目 | ✅ 解决 |
| B4 | builtins 路径猜错 | 工具链 | 项目 | ✅ 解决 |

**统计**：22 项中 **17 项已解决/绕过**，5 项未修（B0/B1/B5/C3/C4/E1/E2 中，最关键是 **B0 和 E1**）。

---

## 最关键的三个结论

### 1. B0 是本次最有价值的发现
**项目有两条 ABI 不一致的降级路径**：`llc -mtriple`（软浮点）与 `clang -x ir`（lp64d 硬浮点）。这解释了"调用方读 a0 还是 fa0"的长期矛盾——不是谁看错，是两条路径。

### 2. 真正让坑存在至今的是 E1（CI 覆盖不全）
B 层 7 个问题、D1 缓存问题，**没有一个能被现有 CI 发现**。补一个坑还会长下一个。

### 3. 我们能做到"823 全绿"，靠的是绕过而不是修复
- B2/B6/B7/D1/D2 都是**环境变量 + 关缓存**绕过
- A1/A2/A3/A5 都是**手工装/补丁**绕过
- **这些绕过没有一项写进项目代码**，下一个人还要重踩一遍

---

## 一句话总结

**环境问题的根源是"验证基线缺失"：项目从未在干净的 riscv64 机器上端到端回归，于是 22 项问题（13 项归项目）全部被 x86-only 的 CI 放过。其中 B0（两条路径 ABI 不一致）是新发现的技术核心——它同时解释了 bf16 libcall 的寄存器矛盾，也是把你的"fa0"和我实测的"a0"统一起来的钥匙。**
