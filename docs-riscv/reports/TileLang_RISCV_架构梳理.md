# TileLang-RISC-V 项目架构梳理

> 一份"从上到下"的完整结构说明。配合 `TileLang_RISCV_系统讲解.html` 一起看。

---

## 0. 一句话定位

**TileLang 的一个 fork，给编译器插了一个新后端**：把 TileLang 语言写的算子，自动翻译成能在 RISC-V CPU 上原生执行的动态库。

- 上游基线：`0924dab6`（TileLang 官方）
- 本仓库叠加：**9 个 `[RISC-V]` 提交**
- 版本：`0.1.8`
- 维护者：RuyiAI-Stack

---

## 1. 整体分层（记住这张图）

```
┌─────────────────────────────────────────────────────────────┐
│  用户层      TileLang Python DSL                              │
│              T.Kernel / T.copy / T.gemm / T.alloc_shared ...  │
└──────────────────────────┬──────────────────────────────────┘
                           │  tilelang.jit / tilelang.compile(target="riscv")
┌──────────────────────────▼──────────────────────────────────┐
│  驱动层      tilelang/engine/                                 │
│   lower.py     降级总驱动（riscv 走短路分支）                  │
│   phase.py     编译 pass 流水线（RISC-V 专属精简序列）          │
└──────────────────────────┬──────────────────────────────────┘
                           │  target.build.tilelang_riscv
┌──────────────────────────▼──────────────────────────────────┐
│  编译核心    src/target/                                       │
│   codegen_riscv.cc  ★ TIR → 结构化 MLIR（~12.9k 行）           │
│   rt_mod_riscv.cc     注册 TVM target kind "riscv"（~130 行）  │
└──────────────────────────┬──────────────────────────────────┘
                           │  MLIR 文本
┌──────────────────────────▼──────────────────────────────────┐
│  工具链层    tilelang/tladapter/  +  tilelang/jit/adapter/riscv/│
│   toolchain.py    发现 mlir-opt / mlir-translate / llc / clang │
│   libgen.py       MLIR → LLVM dialect → LLVM IR → .o           │
│   wrapper.py      clang 链接成 .so，处理 ABI/sysroot            │
│   adapter.py      ctypes 加载 .so 并执行                       │
└──────────────────────────┬──────────────────────────────────┘
                           │  原生 .so
┌──────────────────────────▼──────────────────────────────────┐
│  执行层      在 RISC-V 真机（SG2044）上跑，torch CPU 张量入出    │
└─────────────────────────────────────────────────────────────┘

外部依赖（不在仓库里，但要编出来/装好）：
  LLVM/MLIR 24  ← 编译核心链接它，工具链层调用它的命令行
  Z3            ← 求解器，编译期用
  RISC-V GCC    ← 链接原生库
```

---

## 2. 六步降级链（最重要）

一条 kernel 从源码到跑起来，**换六种表示**。前三步在 C++ 里，后三步在 Python 里。

| 步 | 表示 | 说明 | 对应代码 |
|---|---|---|---|
| 1 | **TileLang DSL** | 你写的 `T.copy / T.gemm / T.Kernel` | 用户脚本 / examples |
| 2 | **TVM TIR** | 张量 IR，带 buffer / 线程绑定 | `tilelang/engine/phase.py`、`lower.py` |
| 3 | **结构化 MLIR** ★ | `linalg` / `scf` / `memref` / `vector` dialect | **`src/target/codegen_riscv.cc`** |
| 4 | **LLVM dialect MLIR** | 再降一级，贴近 LLVM | `tilelang/jit/adapter/riscv/libgen.py` |
| 5 | **LLVM IR → 目标文件** | `llc` 出机器码，triple = `riscv64-*` | `libgen.emit_asm / emit_object` |
| 6 | **共享库 .so → 执行** | `clang` 链接 + `ctypes` 加载 | `wrapper.py` / `adapter.py` |

**记忆**：1–3 是"**翻译**"（GPU 思维 → RISC-V 思维），4–6 是"**落地**"（变成可执行文件并跑）。

---

## 3. 目录/模块地图

### 3.1 顶层

| 路径 | 角色 |
|---|---|
| `CMakeLists.txt` | 构建总入口 |
| `cmake/TileLangRISCVMLIR.cmake` | 找 LLVM/MLIR（要 24.0） |
| `pyproject.toml` | Python 打包（scikit-build-core） |
| `src/` | C++ 编译器核心 |
| `tilelang/` | Python 前端 + 运行时 |
| `testing/python/riscv/` | RISC-V 测试（上手主战场） |
| `3rdparty/` | 5 个子模块 |
| `docs/get_started/` | `targets.md`、`BuildOnSG2044.md` |

### 3.2 C++ 核心 `src/`

| 子目录/文件 | 职责 |
|---|---|
| `src/target/codegen_riscv.cc` | ★ **TIR→MLIR 翻译器主体**（~12.9k 行） |
| `src/target/codegen_riscv.h` | 类声明（`CodeGenTileLangRISCV`） |
| `src/target/rt_mod_riscv.cc` | 注册 `target.build.tilelang_riscv`、target kind `"riscv"` |
| `src/op/` | 算子级 intrinsic（copy / gemm / reduce / atomic…） |
| `src/transform/` | ~57 个 TIR pass（layout、pipeline、vectorize…） |
| `src/layout/` | fragment/shared 布局定义 |
| `src/tl_templates/` | 生成的 kernel 里嵌入的 C++ 头（cpu/cuda/hip…） |

### 3.3 Python 前端 `tilelang/`

| 模块 | 职责 |
|---|---|
| `language/` | DSL 原语（`T.*` 命名空间） |
| `engine/lower.py` | **降级总驱动**，riscv 走短路分支 |
| `engine/phase.py` | **pass 流水线**，RISC-V 专属精简序列 |
| `tladapter/` | MLIR 工具链适配（发现工具 + 跑 pass） |
| `jit/adapter/riscv/` | RISC-V 运行时适配（出码 / 链接 / 执行） |
| `jit/kernel.py` | 选适配器：`target=="riscv"` → `RiscvKernelAdapter` |
| `utils/target.py` | `"riscv"` target 名、`TILELANG_RISCV_MLIR_MODE` |
| `transform/` | C++ pass 的 Python 包装 |
| `cache/` | 编译产物缓存 |
| `profiler/` | `do_bench` 性能测试 |

### 3.4 子模块 `3rdparty/`

| 子模块 | 版本 | 作用 |
|---|---|---|
| `tvm` | `882a774` | **编译核心**（TileLang fork） |
| `TileKernels` | `36d9e45d` | TileKernels 测试用的算子库 |
| `TileOPs` | `67250e81` | TileOps 测试用的算子库 |
| `cutlass` | — | CUDA 模板（RISC-V 用不到） |
| `composable_kernel` | — | ROCm（RISC-V 用不到） |

---

## 4. 编译流水线（pass 序列）

`tilelang/engine/phase.py` 里 **RISC-V 有一条专属的精简流水线**：

### 4.1 `LowerAndLegalize`（RISC-V 分支，第 139 行）
```
BindTarget → AddWrapperForSingleBufStore → LegalizeNegativeIndex
  → VerifyParallelLoop → InjectAssumes → Simplify → LowerAccessPtr
  → LegalizeSafeMemoryAccess → Simplify → HoistNonRestrictParams
```

### 4.2 `OptimizeForTarget`（RISC-V 分支，第 251 行）
```
LowerInitBlock → Simplify → RemoveNoOp
```

### 4.3 对比 GPU 分支（**被跳过的东西**）

| Pass | GPU | RISC-V | 影响 |
|---|---|---|---|
| `LayoutInference` | ✅ | ❌ | 依赖布局的复杂写法可能失败 |
| `InjectSoftwarePipeline` | ✅ | ❌ | 没有软件流水优化 |
| `VectorizeLoop` | ✅ | ❌ | 不走 RVV 向量化 |
| `LowerTileOp` | ✅ | ❌ | tile 级算子不降级 |

> **这就是"能力边界"的根源**：RISC-V 先求"功能对"，没做性能优化 pass。

### 4.4 `lower.py` 的 riscv 短路
- `is_cpu_device_backend` 把 `{c, riscv}` 当"设备后端"（第 26 行）
- riscv 分支**跳过 host/device 拆分**，直接调 `target.build.tilelang_riscv`，取回 MLIR 源（约第 317 行）

---

## 5. 运行时执行链

`tilelang/jit/adapter/riscv/` 三个文件，职责清晰：

| 文件 | 职责 | 关键函数 |
|---|---|---|
| `libgen.py` | MLIR → LLVM | `emit_mlir` / `emit_llvm_ir` / `emit_asm` / `emit_object` |
| `wrapper.py` | 链接 + 加载 | `build_host_shared_library` / `HostKernelLibrary` / `load_host_module` / `run_host` |
| `adapter.py` | 适配 TileLang | `RiscvKernelAdapter`（torch CPU 张量入出、动态 shape） |

**默认 triple**：`riscv64-unknown-linux-gnu`（可用 `TILELANG_RISCV_TRIPLE` 覆盖）

**工具链发现**（`tladapter/toolchain.py`）：
- 优先环境变量 `TILELANG_RISCV_LLVM_ROOT`
- 再找 `../buddy-mlir/llvm/{build,install}`、`3rdparty/llvm-project/...`
- 在 `bin/` 下找 `mlir-opt / mlir-translate / llc / clang / ld.lld`

---

## 6. 测试体系（上手主战场）

项目**没有 RISC-V examples**，验证全在 `testing/python/riscv/`：

| 档次 | 目录 | 内容 | 依赖 |
|---|---|---|---|
| ① 核心后端 | `test_riscv_*.py`（12 个） | target 解析/工具链/流水线/缓存/JIT 运行时 | 无 |
| ② 合成 MLIR | `codegen_ops/` | 10 个按算子族分，可 dump MLIR | 无 |
| ③ 真实 kernel | `TileKernels/` | 48 项端到端 | `3rdparty/TileKernels` |
| ④ 大算子库 | `TileOps/` | 415 项 | `3rdparty/TileOPs` |

**关键技巧**：
- 在**仓库根目录**跑（`testing/conftest.py` 会把本地 `tilelang/` 插到 `sys.path` 最前）
- 改了编译器用 `TILELANG_DISABLE_CACHE=1`，避免旧缓存"假通过"
- `codegen_ops` 支持 `--riscv-dump-mlir 目录` 导出生成的 MLIR

---

## 7. 关键环境变量

| 变量 | 作用 |
|---|---|
| `TILELANG_RISCV_LLVM_ROOT` | **指向 LLVM/MLIR 24 安装根**（就是你正在编的） |
| `TILELANG_RISCV_MLIR_MODE` | `ON`/`AUTO`/`OFF`，控制 MLIR 后端启用 |
| `TILELANG_RISCV_TRIPLE` | 覆盖默认 triple |
| `TILELANG_RISCV_RUNNER` | 模拟器（qemu-riscv64 / spike） |
| `TILELANG_DISABLE_CACHE` | 验证编译器改动时设 1 |
| `TVM_FFI_DISABLE_TORCH_C_DLPACK` | 设 1，避免首次导入编 Torch 扩展 |
| `Z3_ROOT` | Z3 前缀（riscv64 上必需） |

---

## 8. 能力边界（已知限制）

| 事项 | 现状 |
|---|---|
| target 拼写 | 只支持 `target="riscv"` |
| 性能优化 | 未做 layout/pipeline/vectorize，先求功能对 |
| 验证环境 | 415 项测试在 x86 通过，**真机 SG2044 尚未原生复跑** |
| LLVM 版本 | 必须 **24.0**（系统 17 用不了） |
| kernel 形态 | 依赖 layout inference 的复杂 GPU 写法可能失败 |
| 示例 | 无 RISC-V examples |
| 文档坑 | 提到的 `run_codegen_ops.py` **不存在**（文档过时） |

---

## 9. 阅读代码的顺序（从外到内）

1. `tilelang/engine/phase.py:139` — RISC-V 精简 pass 序列（先看跳过什么）
2. `tilelang/engine/lower.py:317` — riscv 短路分支
3. `src/target/rt_mod_riscv.cc` — target 注册（~130 行，好读）
4. `src/target/codegen_riscv.cc` — 翻译器主体（先看 `AddFunction`/`Finish`）
5. `tilelang/tladapter/toolchain.py` — 工具链发现
6. `tilelang/jit/adapter/riscv/{libgen,wrapper,adapter}.py` — 出码/链接/执行
7. `tilelang/utils/target.py` — target 名称与自动选择
