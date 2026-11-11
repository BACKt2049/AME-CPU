# TileLang-RISC-V 环境问题根源摘要

> 机器：`openeuler-riscv64`（RV64GCV，glibc 2.38，gcc 12，Python 3.11.6）
> 结论先行：**环境问题不是零散的配置失误，而是同一个根因长出的五层枝杈——项目从未建立过"在干净 riscv64 机器上端到端跑通"的验证基线，于是所有"只有作者机器才成立"的假设都被固化进了代码和文档。**

---

## 一、一句话根因

**项目造出了"能生成 RISC-V 代码"的编译器，却没造出"这个编译器能在 RISC-V 上跑对"的回归基线。**

所有坑都是这条缺失基线的直接后果。

---

## 二、五层结构（自下而上）

```
第 0 层  外部生态缺位         ← 上游，不可控
第 1 层  依赖版本契约缺失      ← 项目
第 2 层  工具链交付链断裂      ← 项目（核心技术根因）
第 3 层  环境假设硬编码        ← 项目
第 4 层  验证基线缺失          ← 项目（真正的根）
```

### 第 0 层：外部生态缺位（上游问题，项目只能绕过）

| 事实 | 后果 |
|---|---|
| PyPI 无 riscv64 的 `torch` wheel | 必须用第三方 wheel（kmtea）+ `unzip` 绕过 pip 平台标签 |
| PyPI 无 riscv64 的 `z3-solver` wheel | 必须源码编译（20–60 分钟） |
| Buddy MLIR 只有 x86_64 的 LLVM 24 预编译包 | 必须自编 LLVM（数小时） |
| 系统 gcc 12 的 libgcc **不含** `__truncsfhf2/__extendhfsf2/__truncsfbf2` | → 引爆第 2 层 |

**性质**：这不是项目的错，但项目**没有为这些缺位准备任何 fallback**。

### 第 1 层：依赖版本契约缺失

| 位置 | 问题 |
|---|---|
| `pyproject.toml:66` | `cython>=3.1.0` 无上限 → Cython 3.3 与 `cp38` Limited API 冲突，构建 `#error` |
| `requirements.txt:4` vs `pyproject.toml:38` | riscv64 跳过 `torch-c-dlpack-ext` 的标记**只在一边有**，两边不一致 |
| 未声明 | torch 2.4.1 按 numpy 1.x 编译，pip 装 tilelang 会把 numpy 升到 2.x → ABI 崩 |
| `pyproject.toml:250-271` | cibuildwheel 只支持 x86_64/aarch64，riscv64 走 `*) exit 1` |

**性质**：契约只写在"作者当时恰好如此"的区间里，没有上下界，也没有跨架构条件。

### 第 2 层：工具链交付链断裂（核心技术根因）

**这是 bf16/fp16 全部失败的地方。** 一条 libcall 链路上有四处都没交付：

| 环节 | 位置 | 断裂点 |
|---|---|---|
| ① 产生 libcall | `codegen_riscv.cc:6288-6301` | `CastValueToType` 只按位宽比较，**不区分 bf16/fp16**，无条件产 `arith.truncf/extf` |
| ② MLIR 降级 | `libgen.py:306-318` | `convert-arith-to-llvm` **原样放行** → LLVM 选 `__truncsfbf2` |
| ②' 自我制造 | `libgen.py:98`、`:115` | FP8 合法化**自己又插入** bf16/fp16 转换，重新接回 libcall |
| ③ 符号提供 | `wrapper.py:943-959` | 链接命令**没有** `-lclang_rt.builtins`、没有 `-rtlib=compiler-rt` |
| ③' 探测缺失 | `wrapper.py:685-718` | `_riscv_gcc_runtime_library_dirs` 只找 `libgcc_s.so`，**连 `libgcc.a` 都不看**，更不知道 compiler-rt |
| ④ 错误被吞 | `wrapper.py:945` | 缺 `-Wl,--no-undefined`；`-shared` 允许未定义符号 → 失败推迟到 `ctypes.CDLL` |
| ⑤ 缓存固化 | `kernel_cache.py:235-245` | 缓存键**不含** `CLANG_FLAGS`/`GCC_ROOT`/clang 路径 → 坏 `.so` 永久命中 |

**关键实证**：补上 `-lclang_rt.builtins` 后，**一行编译器代码都没改**，463/463 → 823/823 全绿。
→ **说明生成端是对的，断的是"提供符号"这三环。**

### 第 3 层：环境假设硬编码

| 位置 | 写死了什么 | 在哪台机器成立 |
|---|---|---|
| `wrapper.py:679` | gcc 根 `/opt/gcc-native` | SG2044 SDK 机器 |
| `TileKernels/conftest.py:22-24`、`TileOps/conftest.py:26-27` | `/home/liquanyi/tool/llvm` | 作者本人机器 |
| `wrapper.py:721-745` | `--gcc-toolchain=<root>` | x86 / 标准 gcc 布局 |
| `CMakeLists.txt:436-439` | riscv 一律关 lld/mold | SG2044（lld 其实在别处可用） |

**性质**：用硬编码代替探测（`clang -print-resource-dir` / `gcc -print-file-name` 一个都没用）。

### 第 4 层：验证基线缺失（真正的根）

| 证据 | 含义 |
|---|---|
| `ci-buddy-mlir.yml:56` | CI 跑在 `ubuntu-latest`（**x86_64**） |
| `ci-buddy-mlir.yml:111-118` | 只跑 5 个编译路径测试，**不跑 TileKernels/TileOps** |
| `README.md:165` | 自己写 "still need SG2044 native validation" |
| `BuildOnSG2044.md:3` | 却又称 "**validated** native bring-up flow"（自相矛盾） |

**为什么这个坑在作者环境里"物理不存在"**：

| 平台 | `__truncsfhf2/__extendhfsf2` 从哪来 |
|---|---|
| x86（作者 CI/主机） | 工具链**默认自带** |
| riscv64 + gcc 12（你的机器） | libgcc 里**没有**，必须补 compiler-rt |

→ **作者不是漏测，是这个坑在他的环境里根本不存在。** 只有一台干净的 riscv64 机器才能暴露。

---

## 三、症状 → 根因映射表

| 你遇到的报错 | 直接原因 | 归属层 | 定性 |
|---|---|---|---|
| `undefined symbol: __truncsfhf2` | 链接缺 compiler-rt | 第 2 层 | **项目** |
| bf16 数值全错（nan/1e38） | 同上（符号缺失/错版本满足） | 第 2 层 | **项目** |
| 改了参数结果不变 | 缓存键无工具链指纹 | 第 2 层 | **项目** |
| `crtbeginS.o` 找不到 | gcc 根写死 + `--gcc-toolchain` 在 openEuler 失效 | 第 3 层 | **项目** |
| `Cython 3.3 requires ≥3.9` | 版本无上限 | 第 1 层 | **项目** |
| torch/numpy ABI 崩 | 版本契约缺失 | 第 1 层 | 项目（+上游） |
| torch/z3/LLVM 装不上 | 无 riscv64 wheel/预编译 | 第 0 层 | **上游** |
| 测试全 skip（占位后端） | `LLVM_ROOT` 未设 + CMake 缓存 | 第 3 层 | 项目 |
| fp8 `fill_cpu not implemented` | PyTorch CPU 限制 | 第 0 层 | **上游** |
| `R_RISCV_JAL` 超范围 | TOSA 文件 -O3 代码 >±1MB | 第 0 层（riscv 特有） | 上游工具链 |

**统计**：11 类环境问题里，**7 类归项目**、4 类归上游生态。

---

## 四、责任三分

| 类别 | 占比 | 内容 |
|---|---|---|
| **项目本身** | **主导** | 文档夸大 + 代码隐式 x86 假设 + CI 覆盖不全 + 工具链交付链断裂 |
| **上游生态** | 次要 | torch/z3 无 riscv64 wheel、LLVM 24 无 riscv64 预编译、PyTorch CPU 无 fp8 fill |
| **操作方式** | 少量 | 裸 SSH 前台跑长任务（断线丢进度）、环境变量未持久化 |

---

## 五、修复优先级（砍根 vs 砍枝）

| 优先级 | 动作 | 治哪层 | 工作量 |
|---|---|---|---|
| **P0** | `wrapper.py` 用 `clang -print-resource-dir` 自动探测并 `-lclang_rt.builtins` | 第 2 层 | ~10 行 |
| **P0** | 链接命令加 `-Wl,--no-undefined`（失败提前到链接期，不进缓存） | 第 2 层 | 1 行 |
| **P0** | `kernel_cache.py` 缓存键加入工具链指纹 | 第 2 层 | ~10 行 |
| **P1** | 建真实 riscv64 CI 回归（跑 TileKernels + TileOps） | **第 4 层（根）** | 中 |
| **P1** | gcc/clang 路径全部改成 `-print-*` 探测，删 `/opt/gcc-native` 硬编码 | 第 3 层 | ~20 行 |
| **P2** | Cython 补上限、requirements 对齐、cibuildwheel 加 riscv64 分支 | 第 1 层 | ~10 行 |
| **P2** | 文档把 "validated" 改为 "x86-only compile-path validated" | 第 4 层 | 文档 |

**核心判断**：P0 三件事（约 40 行）能把这条链路变成"要么对、要么立刻报错"；但**只有 P1 的 CI 回归能防止复发**——否则补一个坑还会长下一个。

---

## 六、一句话总结

**环境问题的根源是"验证基线缺失"这一个根，长出了五层枝杈：外部生态缺位 → 版本契约缺失 → 工具链交付链断裂 → 环境假设硬编码 → 无 riscv64 回归。**

我们的 823/823 全绿恰好反证了这一点：**编译器主体是对的，缺的从来不是它，是那层从来没人验证过的工具链集成。**
