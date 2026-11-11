# TileLang-RISC-V 全量验收 Trace

> 目标：把 Excel 里的 **216 个算子**（TileKernels 37 + TileOPs 179）在真实 riscv64 机器上验收。
> 结论：**823 个测试用例全部通过（0 失败）**，覆盖 108 核心 + 252 codegen_ops + 48 TileKernels + 415 TileOps。
> 本文档记录每一步命令、实际输出、遇到的困难与修法，可复现。

---

## 0. 环境快照

| 项 | 值 |
|---|---|
| 主机 | `openeuler-riscv64`（64 核 RISC-V，glibc 2.38，gcc 12/14） |
| 代码 | `~/tilelang-riscv-clean` @ `32fce5ed`（RuyiAI-Stack/tilelang-riscv main） |
| Python | 3.11.6 |
| 依赖 | numpy 1.26.4 / torch 2.4.1 / z3-solver 4.15.4 / cython 3.1.8 |
| LLVM/MLIR | 24.0.0git（自编，`~/llvm-project/build`） |
| compiler-rt | `~/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu/libclang_rt.builtins.a` |
| 关键环境变量 | 见 `run_all_riscv_tests.sh` 第 1 节 |

**环境变量（每次新 shell 都要重设）**：

```bash
export TILELANG_RISCV_LLVM_ROOT=/home/wangjiayue/llvm-project/build
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 \
  -L/home/wangjiayue/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu \
  -lclang_rt.builtins"
```

---

## 1. 修复时间线（为什么能全绿）

| 阶段 | TileKernels | TileOps |
|---|---|---|
| 最初（未修 compiler-rt 链接） | 13 failed / 35 passed | — |
| 修好 compiler-rt 链接后 | **48/48** | 149 failed / 266 passed |
| 进一步定位后 | — | 6 failed / 409 passed |
| 修掉 6 个 fp8 测试基建问题后 | — | **415/415** |
| 本次合并全量（含 codegen_ops 等） | **823 passed / 0 failed** | |

**决定性根因**：`tilelang/jit/adapter/riscv/wrapper.py:943-959` 的 `build_host_shared_library` 链接命令只带
`-lmlir_c_runner_utils -lmlir_runner_utils -lm`，**没有 `-lclang_rt.builtins`**。而 `codegen_riscv.cc` 生成的
fp16/bf16 转换会调用 `__truncsfhf2` / `__extendhfsf2`，这些符号只在 compiler-rt 里（系统 gcc 12 的 libgcc 没有）。
**这是链接问题，不是编译器生成代码算错** —— 改对链接后 463 用例全过即可证明。

---

## 2. Phase 1 — 全量测试 + JUnit XML

```bash
cd ~/tilelang-riscv-clean       # 必须在仓库根目录
# 设全环境变量；不设 TILELANG_DISABLE_CACHE（开缓存复用编译产物）
python3 -m pytest -q \
  testing/python/riscv/test_riscv_*.py \
  testing/python/riscv/codegen_ops \
  testing/python/riscv/TileKernels \
  testing/python/riscv/TileOps \
  --junitxml=~/riscv_results.xml 2>&1 | tee ~/riscv_results.log
```

**结果**：
```
823 passed, 17 warnings in 1896.06s (0:31:36)
```
产出 `~/riscv_results.xml`（126701 bytes）。

> **坑（classname 契约）**：`operator_inventory.py` 靠 `<testcase classname="...TileOps.test_x">` 归属，
> 而 pytest 只有在**从仓库根目录运行**且目录有 `__init__.py` 时才生成这种点分 classname。
> 换目录或用 `-q` 收集到的扁平 classname 会让它**静默零匹配、不报错**。本次实测：
> `823 testcase` 分布 = core 108 / codegen_ops 252 / TileKernels 48 / TileOps 415，契约满足。

---

## 3. Phase 2 — 项目官方清单

```bash
cd ~/tilelang-riscv-clean
python3 testing/python/riscv/operator_inventory.py \
  --junit ~/riscv_results.xml \
  --output ~/operator_inventory.json --check
```

**结果**（`--check` 通过，exit=0）：
```
TileOps: 205 declarations; {'kernel-test-references': 176, 'unmapped': 20, 'component-test-references': 9}
  JUnit: {'passed': 415}; passed scopes: {'kernel-or-component-numerical': 400, 'rejection': 3, 'public-api-numerical': 12}
TileKernels: 48 declarations; {'kernel-test-references': 48}
  JUnit: {'passed': 48}; passed scopes: {'kernel-or-component-numerical': 48}
```

**不用 `--require-tileops-numerical`**：按设计必然失败（20 unmapped + 9 component-only + 21 known_gaps +
3 rejection 不计数值），且它只给全局退出码，不逐算子判定。

---

## 4. Phase 3 — 别名表 + 验收表

### 为什么需要别名表

Excel 用全称驼峰，项目 manifest（`3rdparty/TileOPs/src/tileops/manifest/spec/*.yaml`，205 条）用缩写：

| Excel | 项目 | 情况 |
|---|---|---|
| `SigmoidFwdOp` | `SigmoidFwdOp` | 精确命中 |
| `GroupedQueryAttentionFwdOp` | `GQADenseFwdOp` / `GQAVarlenFwdOp` / `GQABwdOp`… | 全称↔缩写，且 1:N |
| `MultiHeadLatentAttentionDecodeWithKVCacheFwdOp` | `MLADecodeWithKVCacheFwdOp` | 全称↔缩写 |
| `GemmOp` | `GemmFwdOp` | 不规则差异 |

且 **Excel 216 ≠ 项目枚举 253**（48 JIT factory + 205 manifest），两套枚举粒度不同。
`operator_inventory.py` 只输出正交字段（`upstream_status` / `evidence` / `tests_outcomes` / `tests_passed_scopes`），
**不给单值 verdict**，需要后处理折叠。

### 执行

```bash
# 本地（Windows workspace），读 evidence/ 下 4 份证据
python build_acceptance.py
```

**匹配结果**（`alias_map.json`，216 行）：

| 方式 | 行数 | 含义 |
|---|---|---|
| exact | 108 | 名称精确一致 |
| manual | 69 | 人工确认的别名（全称→缩写，含 1:N 扇出） |
| rule | 33 | 名称归一化 / 词元相似度自动匹配 |
| no-declaration | 6 | 项目 manifest 里**本就没有**该算子的 1:1 声明，未硬塞近似名 |
| unmapped / fuzzy | 0 | 无悬空模糊匹配（低置信项已全部核实或显式声明） |

**Excel 216 行 verdict**：`通过(数值) 197` / `通过(数值, 族级覆盖) 13` / `未映射 6`。

6 个"未映射"（如实标注，非失败）：
`GroupedQueryAttentionSlidingWindowFwdOp`、`GroupedQueryAttentionSlidingWindowVarlenFwdOp`、
`MultiHeadAttentionBwdOp`、`MultiHeadAttentionFwdOp`、`GatedDeltaNetBwdOp`、`GatedDeltaNetDecodeOp`。

### 产出 `TileLang_算子验收表.xlsx`（6 sheet）

| Sheet | 内容 |
|---|---|
| 口径说明 | 计数口径、verdict 定义、不美化原则、匹配统计 |
| 测试套件验收 | 按 142 个测试文件聚合的用例数/通过/失败/数值用例 |
| 算子验收_Excel216 | 216 行，含匹配方式、映射、测试文件、verdict |
| 算子验收_manifest253 | 205+48 条官方 inventory 全字段 + JUnit 结果 |
| 别名表 | 216 行别名映射与置信度 |
| 未映射与已知限制 | 6 个无声明 + 20 unmapped + 9 component + 14 spec-only + 21 known_gaps |

**QA**：openpyxl 读回校验 6 sheet 行列完整（Excel216 表 217 行含表头），0 个公式错误，别名目标 0 个不存在于 inventory。

---

## 5. 遇到的困难清单

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `undefined symbol: __truncsfhf2` / bf16 数值全错 | 链接缺 compiler-rt builtins | `TILELANG_RISCV_CLANG_FLAGS` 加 `-L... -lclang_rt.builtins` |
| 2 | 改了参数结果不变 | 坏的 `.so` 被磁盘缓存记住 | `TILELANG_DISABLE_CACHE=1` 强制重编 |
| 3 | `pip install -e .` 后 torch 崩 | pip 把 numpy 升到 2.x，torch 2.4.1 按 numpy 1.x 编译 | 装完立刻 `pip install --user 'numpy<2'` |
| 4 | `crtbeginS.o` / `-lgcc_s` 找不到 | openEuler + 自编 clang 对 `--gcc-toolchain` 解析有怪癖 | 改用 `--gcc-install-dir=...` |
| 5 | `build_host_shared_library` 找不到 `libclang_rt.builtins` | 路径写成 `lib/linux`，实际是 `lib/riscv64-unknown-linux-gnu` | 用 `find` 查真实路径，不猜 |
| 6 | 6 个 fp8 测试 `RuntimeError: "fill_cpu" not implemented for 'Float8_e4m3fn'` | **PyTorch CPU 不支持 fp8 的 `fill`**，测试准备数据阶段崩，非算子问题 | `torch.full(dtype=fp8)` → 先 fp32 造再 `.to(fp8)`；4 处，备份 `.orig` |
| 7 | `operator_inventory.py` 零匹配 | 不在仓库根跑 → JUnit classname 不含 `.TileOps.` | 从仓库根目录跑 pytest |
| 8 | `--require-tileops-numerical` 失败 | 按设计必然失败（unmapped/known_gaps） | 不用它做闸门 |
| 9 | 长任务断线丢进度 | 裸 SSH 前台跑 | 一律 tmux |
| 10 | LLVM 编译 `R_RISCV_JAL` 超范围 | TOSA 文件 -O3 下函数超 ±1MB（riscv64 特有） | CMake 补丁给该文件强制 `-Os` |

---

## 6. 一次性复现方法

```bash
# 服务器（放 tmux，防断线）
tmux new -s verify 'bash run_all_riscv_tests.sh'

# 把服务器产物下载到本地 workspace/evidence/
#   riscv_results.xml / operator_inventory.json / riscv_results.log / full_final.log / fp8fix.log

# 本地生成验收表
python build_acceptance.py
```

`run_all_riscv_tests.sh` 是**幂等**的：设环境变量 → 检查并应用 fp8 补丁（有 `.orig` 则跳过）→
pytest 带 XML → inventory → 汇总。反复运行不会重复打补丁。

---

## 7. 结论与边界（不美化）

1. **823 个测试用例全通过**，覆盖了 Excel 里 216 个算子中 210 个能对应到项目声明的算子；
   6 个因项目 manifest 无 1:1 声明而如实标「未映射」。
2. **不把"族级测试通过"写成"该族所有算子逐个验证"**：13 行明确标「族级覆盖」，
   inventory 自带声明 "Counts overlap across entries"（一个用例会计给多个算子）。
3. **20 unmapped / 9 component-only / 14 spec-only / 21 known_gaps 均如实列出**，不算通过。
4. 6 个 fp8 失败是 **PyTorch CPU 的限制**（测试基建），不是 tilelang/RISC-V 后端算错；
   补丁只改测试造数据方式，不动产品源码。
5. 项目 CI 只在 x86 上验证编译路径、从未在真 riscv64 上原生回归；本次是在真机上完成的端到端验证。

---

## 附：证据文件

| 文件 | 位置 | 大小 |
|---|---|---|
| `riscv_results.xml` | `evidence/` + 服务器 `~/` | 126701 B |
| `operator_inventory.json` | `evidence/` + 服务器 `~/` | 227808 B |
| `riscv_results.log` | `evidence/` + 服务器 `~/` | 2309 B |
| `full_final.log`（463 全过，关缓存） | `evidence/` + 服务器 `~/` | 58875 B |
| `fp8fix.log`（6→10 通过） | `evidence/` + 服务器 `~/` | 1868 B |
| `phase1.log` | `evidence/` + 服务器 `~/` | 63 B |
