# AME-CPU

TileLang 在 RISC-V（AME CPU 前端验证）上的**已跑通版本**与环境问题完整记录。

本仓库源自 [RuyiAI-Stack/tilelang-riscv](https://github.com/RuyiAI-Stack/tilelang-riscv) @ `32fce5ed`，
记录在真实 RISC-V 机器（`openeuler-riscv64`，RV64GCV）上完成的全量算子验证，以及过程中定位的全部环境问题。

---

## 两个分支

| 分支 | 内容 | 用途 |
|---|---|---|
| **`main`** | 完整源码（含 fp8 测试补丁）+ 环境修复脚本 + 证据 + 全部报告 | **直接可跑**：216 个算子全部验证通过的版本 |
| **`env-reports`** | **只保留 Markdown 报告**（环境问题定位 + 修复办法） | 只看结论与修法，不需要代码 |

切换：
```bash
git checkout env-reports   # 只读报告
git checkout main          # 完整代码
```

---

## main 分支的验证结果

**823 个测试用例全部通过，0 失败**（2026-11-06 在真 RISC-V 机器上实测）

| 套件 | 用例数 | 结果 |
|---|---|---|
| TileKernels（37 个算子） | 48 | ✅ 48 passed |
| TileOPs（179 个算子） | 415 | ✅ 415 passed |
| codegen_ops（合成 MLIR） | 252 | ✅ 252 passed |
| 核心后端测试 | 108 | ✅ 108 passed |
| **合计** | **823** | **✅ 823 passed, 0 failed** |

对应 Excel《TileLang算子列表》全部 **216 个算子**（TileKernels 37 + TileOPs 179）。

---

## 快速开始

### 1. 克隆（含子模块）

```bash
git clone --recursive <this-repo-url> AME-CPU
cd AME-CPU
```

子模块未拉全时补：
```bash
git submodule update --init --recursive
```

> ⚠️ 子模块（`3rdparty/tvm` 等）体积约 600MB，首次拉取较慢。
> 本仓库不包含子模块内容，只保留 `.gitmodules` 与 commit 记录。

### 2. 环境准备

**必须阅读**：[`docs-riscv/reports/TileLang_RISCV_环境问题全量汇总.md`](docs-riscv/reports/TileLang_RISCV_环境问题全量汇总.md)

其中记录了 22 项环境问题（13 项属项目本身）。最关键的三件事：

```bash
# ① LLVM/MLIR 24（必须自编，无 riscv64 预编译包）
export TILELANG_RISCV_LLVM_ROOT=/path/to/llvm-project/build

# ② compiler-rt builtins（系统 gcc 12 的 libgcc 缺 __truncsfhf2 等符号）
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 \
  -L$TILELANG_RISCV_LLVM_ROOT/lib/clang/24/lib/riscv64-unknown-linux-gnu \
  -lclang_rt.builtins"

# ③ numpy 必须 <2（torch 2.4.1 按 numpy 1.x 编译）
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

### 3. 构建

```bash
pip install --user ./3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps
pip install --user -e . --no-build-isolation -v
pip install --user 'numpy<2'   # ← pip 装 tilelang 会把 numpy 升到 2.x，必须降回
```

### 4. 验证

```bash
# 三个导入
python3 -c "import tilelang; print(tilelang.__file__)"
python3 -c "import tilelang.tladapter._native; print('native adapter ok')"
python3 -c "from tilelang import tvm; print(tvm.ffi.get_global_func('target.build.tilelang_riscv'))"

# 全量测试（一条命令，幂等）
bash docs-riscv/scripts/run_all_riscv_tests.sh
```

---

## 目录

```
.
├── README.md                      ← 本文件
├── README.upstream.md             ← 上游原 README（保留）
├── docs-riscv/
│   ├── reports/                   ← 全部分析报告（11 篇）
│   │   ├── TileLang_RISCV_环境问题全量汇总.md      ★ 首选阅读（22 项问题 + 修法）
│   │   ├── TileLang_RISCV_环境问题根源摘要.md      五层归因
│   │   ├── TileLang_RISCV_兼容性缺陷深度调查报告.md 寄存器级实证 + 代码定位
│   │   ├── TileLang_RISCV_踩坑修复记录.md          逐条现象→根因→修法
│   │   ├── TileLang_RISCV_绕过清单可复现验证报告.md 可复现验证
│   │   ├── TileLang_RISCV_全量测试Trace.md         完整执行 trace
│   │   ├── TileLang_RISCV_架构梳理.md              六层架构
│   │   ├── TILELANG_RISCV_上手手册.md              手动逐步上手
│   │   ├── wjy分配任务总结.md                      分配的 2 个算子
│   │   ├── 超出分配范围的额外完成总结.md            216 算子全量
│   │   └── trace_engram_kernels.md                 2 个 engram 算子 trace
│   ├── scripts/
│   │   ├── run_all_riscv_tests.sh  一次性执行（幂等）
│   │   ├── build_acceptance.py     生成算子验收表 xlsx
│   │   ├── alias_map.json          Excel→项目名映射
│   │   └── ops_list.json           216 算子清单
│   └── evidence/                   ← 原始证据
│       ├── riscv_results.xml       JUnit（823 testcase）
│       ├── operator_inventory.json 项目官方清单（205+48 条）
│       ├── riscv_results.log       合并测试日志
│       ├── full_final.log          463 全过（关缓存新鲜编译）
│       ├── fp8fix.log              6→10 通过
│       └── phase1.log
├── src/                            ← TileLang C++ 核心（RISC-V codegen 在此）
├── tilelang/                       ← Python 前端与运行时
├── testing/python/riscv/           ← RISC-V 测试（含 fp8 补丁）
├── 3rdparty/                       ← 子模块（gitlink，需 submodule update）
└── ...
```

---

## 本仓库相对上游的关键改动

只改了 **1 处测试文件**（不改产品源码）：

**`testing/python/riscv/TileOps/test_paged_kv_cache_write.py`**
- 4 处 `torch.full(..., dtype=torch.float8_e4m3fn)` → 先 fp32 造再 `.to(fp8)`
- 原因：PyTorch 2.4.1 的 CPU 后端未实现 fp8 的 `fill`，测试造数据阶段就崩（`fill_cpu not implemented`）
- 属**测试层**改动，不影响算法与容差

其余全部是新增的文档与脚本，未修改任何 TileLang 源码。

---

## 子模块版本（本仓库记录）

| 子模块 | commit |
|---|---|
| `3rdparty/tvm` | `882a774844993d103ae6e317ba3c7bbb5952b662` |
| `3rdparty/TileKernels` | `36d9e45d38e204ebb87e6f6e833821eee0482fe5` |
| `3rdparty/TileOPs` | `67250e8172fa4a97c3ab7cb5b6c269169f12299b` |
| `3rdparty/cutlass` | `b2dd65dc864e09688245b316ac46c4a6cd07e15c` |
| `3rdparty/composable_kernel` | `b38bb492a1a55b5abb0c345962143c0f9c482cfb` |

---

## 一句话总结

**上游能编译但在 RISC-V 上跑不对（823 用例里曾经 149+13 失败）；本仓库通过环境修复让 823 用例全绿，并把每一项环境问题的根因、位置、修法都记录成可复现文档。**
