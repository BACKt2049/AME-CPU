# AME-CPU · 环境问题报告（env-reports 分支）

> 本分支**只含 Markdown 报告 + 原始证据**，不含源码。
> 需要可运行代码 → 切到 **`main`** 分支。
>
> 项目：[RuyiAI-Stack/tilelang-riscv](https://github.com/RuyiAI-Stack/tilelang-riscv) @ `32fce5ed`
> 机器：`openeuler-riscv64`（RV64GCV，64 核，glibc 2.38，gcc 12，Python 3.11.6）

---

## 一句话结论

**上游能编译，但在真实 RISC-V 机器上跑不对（823 用例里曾有 149 + 13 个失败）。
根因不是编译器算错，而是"从未建立过在干净 riscv64 机器上端到端跑通的验证基线"，
导致 22 项环境问题被 x86-only 的 CI 全部放过。**

---

## 阅读顺序

| 顺序 | 文档 | 内容 |
|---|---|---|
| **1** | [`reports/TileLang_RISCV_环境问题全量汇总.md`](docs-riscv/reports/TileLang_RISCV_环境问题全量汇总.md) | **★ 首选**：22 项环境问题，按层分组，每项含现象/根因/代码位置/修法/状态 |
| **2** | [`reports/TileLang_RISCV_环境问题根源摘要.md`](docs-riscv/reports/TileLang_RISCV_环境问题根源摘要.md) | 五层归因（生态 → 版本契约 → 工具链 → 硬编码 → 验证基线） |
| **3** | [`reports/TileLang_RISCV_兼容性缺陷深度调查报告.md`](docs-riscv/reports/TileLang_RISCV_兼容性缺陷深度调查报告.md) | 寄存器级反汇编实证 + 五层代码定位 + 修改意见 |
| **4** | [`reports/TileLang_RISCV_踩坑修复记录.md`](docs-riscv/reports/TileLang_RISCV_踩坑修复记录.md) | 逐条"现象 → 根因 → 修法"操作记录 |
| **5** | [`reports/TileLang_RISCV_绕过清单可复现验证报告.md`](docs-riscv/reports/TileLang_RISCV_绕过清单可复现验证报告.md) | 新拉干净代码重做一遍，验证绕过清单可复现 |
| **6** | [`reports/TileLang_RISCV_全量测试Trace.md`](docs-riscv/reports/TileLang_RISCV_全量测试Trace.md) | 完整执行 trace（命令 + 实际输出 + 耗时） |
| 7 | [`reports/TileLang_RISCV_架构梳理.md`](docs-riscv/reports/TileLang_RISCV_架构梳理.md) | 六层架构 + 六阶段执行链 |
| 8 | [`reports/TILELANG_RISCV_上手手册.md`](docs-riscv/reports/TILELANG_RISCV_上手手册.md) | 手动逐步上手命令手册 |
| 9 | [`reports/wjy分配任务总结.md`](docs-riscv/reports/wjy分配任务总结.md) | 分配到的 2 个 engram 算子 |
| 10 | [`reports/超出分配范围的额外完成总结.md`](docs-riscv/reports/超出分配范围的额外完成总结.md) | 216 算子全量验收 |
| 11 | [`reports/trace_engram_kernels.md`](docs-riscv/reports/trace_engram_kernels.md) | 2 个 engram 算子详细 trace |

---

## 22 项环境问题一览

按层分组（详细内容见报告 1）：

| 层 | 数量 | 代表问题 | 归属 |
|---|---|---|---|
| **A. 外部生态缺位** | 6 | torch/z3 无 riscv64 wheel；LLVM 24 无 riscv64 预编译；gcc 12 libgcc 缺 fp16/bf16 内置函数；PyTorch CPU 无 fp8 fill | 上游 |
| **B. 工具链与 ABI** | 7 | **★ 两条降级路径 ABI 不一致**（`llc -mtriple` 软浮点 vs `clang -x ir` 硬浮点）；compiler-rt builtins 未链接；`-shared` 吞未定义符号；gcc root 写死 | 项目 |
| **C. 构建系统** | 4 | Cython 无上限（3.3 与 cp38 冲突）；CMake 缓存钉死；requirements 不一致；cibuildwheel 无 riscv64 | 项目 |
| **D. 测试基建** | 3 | 缓存键无工具链指纹（改环境不生效）；conftest 写死作者个人路径；须从仓库根跑 | 项目 |
| **E. 文档与 CI** | 2 | CI 只跑 x86 + 编译路径；文档 "validated" 与 "仍需原生验证" 自相矛盾 | 项目 |

**归属统计**：13 项项目 / 6 项上游 / 3 项操作方式
**状态统计**：17 项已解决或绕过，**5 项仍未修**

---

## 最关键的三个发现

### 1. B0：两条降级路径 ABI 不一致（未修）

| 路径 | 代码位置 | 调用 | 默认 ABI | float 参数寄存器 |
|---|---|---|---|---|
| object/asm（freestanding/qemu） | `libgen.py:363-396` | `llc -mtriple=...`（**无 `-mattr`/`-mabi`**） | 软浮点 | **`a0`** |
| host .so（实际跑测试） | `wrapper.py:943-967` | `clang -x ir -target` | lp64d 硬浮点 | **`fa0`** |

同一个 kernel 走两条路径产出 **ABI 不同**的机器码。这解释了"bf16 libcall 调用方到底读 a0 还是 fa0"的长期矛盾——**不是谁看错，是两条不同的路径**。

### 2. bf16 失败是链接问题，不是编译器 bug

补上 `-lclang_rt.builtins` 后，**一行编译器代码都没改**：
- TileKernels：13 failed → **48 passed**
- 全量：→ **823 passed / 0 failed**

如果真是代码生成算法错，改链接不可能让它变对。

### 3. 真正让坑存在至今的是 CI 覆盖不全

`.github/workflows/ci-buddy-mlir.yml:111-118` 只跑 5 个编译路径测试，**不跑 TileKernels/TileOps 运行时测试**。B 层 7 个问题、D1 缓存问题，**没有一个能被它发现**。

---

## 修复办法（最小改动包，约 40 行）

### P0：立刻止血

```python
# 1) wrapper.py: 用 clang 自己报告 resource dir，不猜路径
resource_dir = _run_checked([clang, "-print-resource-dir"]).stdout.strip()
for cand in (f"{resource_dir}/lib/linux", f"{resource_dir}/lib/riscv64-unknown-linux-gnu"):
    if Path(cand).is_dir():
        flags += [f"-resource-dir={resource_dir}", f"-L{cand}", "-lclang_rt.builtins"]

# 2) wrapper.py 链接命令加：让失败提前到链接期
"-Wl,--no-undefined"

# 3) kernel_cache.py: 缓存键纳入工具链指纹
"riscv_toolchain": {
    "clang_flags": os.environ.get("TILELANG_RISCV_CLANG_FLAGS", ""),
    "gcc_root": os.environ.get("TILELANG_RISCV_GCC_ROOT", ""),
    "abi": os.environ.get("TILELANG_RISCV_ABI", ""),
    "llvm_root": os.environ.get("TILELANG_RISCV_LLVM_ROOT", ""),
}
```

### P0：修 ABI 路径不一致（B0）

```python
# libgen.py:381-388，llc 补 -mattr，与 clang 路径对齐
[llc, f"-mtriple={triple}", "-mattr=+m,+a,+f,+d,+c", f"-filetype={filetype}", ...]
```

### P1：治根（防复发）

- **建真实 riscv64 CI 回归**（跑 TileKernels + TileOps）← 这一条能预防 80% 的坑再生
- gcc/clang 路径全部改 `-print-*` 探测，删 `/opt/gcc-native` 硬编码
- 文档把 "validated" 改为 "x86-only compile-path validated"

---

## 当前实际生效的绕过方案（环境变量 + 关缓存）

```bash
export TILELANG_RISCV_LLVM_ROOT=/path/to/llvm-project/build
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 \
  -L$TILELANG_RISCV_LLVM_ROOT/lib/clang/24/lib/riscv64-unknown-linux-gnu \
  -lclang_rt.builtins"
export TILELANG_DISABLE_CACHE=1     # 治 D1：缓存键无工具链指纹
```

> ⚠️ 这些绕过**没有一项写进项目代码**，下一个人还要重踩一遍 —— 这正是本分支存在的意义。

---

## 原始证据（docs-riscv/evidence/）

| 文件 | 内容 |
|---|---|
| `riscv_results.xml` | JUnit XML，823 个 testcase |
| `operator_inventory.json` | 项目官方清单（205 + 48 条） |
| `riscv_results.log` | 合并全量测试日志 |
| `full_final.log` | 463 用例全过（关缓存新鲜编译） |
| `fp8fix.log` | 6 个 fp8 失败 → 10 passed |
| `phase1.log` | Phase 1 起止时间 |

生成验收表需要脚本（在 `main` 分支的 `docs-riscv/scripts/`）：
```bash
python build_acceptance.py   # 读 evidence/ 生成 TileLang_算子验收表.xlsx
```

---

## 一句话总结

**22 项环境问题、13 项归项目本身；根因是"验证基线缺失"；5 项至今未修，其中 B0（两条路径 ABI 不一致）是唯一还没解的技术硬骨头。**
