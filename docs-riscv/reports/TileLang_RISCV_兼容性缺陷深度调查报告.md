# TileLang-RISC-V 兼容性缺陷深度调查报告

> 调查对象：`RuyiAI-Stack/tilelang-riscv` @ `32fce5ed`
> 机器：`openeuler-riscv64`（RV64GCV，`zfh`/`zvfh` 有硬件 fp16，**无 zfbfmin/zvfbfmin**，故 bf16 全靠软件 libcall）
> 工具链：自编 LLVM/MLIR 24.0.0git（`RuyiAI-Stack/llvm-project` @3845a205）+ 系统 gcc 12
> 方法：寄存器级反汇编实证 + 全链路代码定位 + 823 用例实测回归
> **实测基线：823 passed / 0 failed（含 TileKernels 48 + TileOps 415）**

---

## 一、先纠正一个流行但方向错误的判断

你提供的 RCA 认为：「调用方从 `fa0` 读，helper 把 `uint16_t` 结果放到 `a0`，两者 ABI 错配」。

**实测（本机、本工具链）与此相反。** 我用一个真实 TileLang bf16 kernel 反汇编得到：

```asm
# kernel: B[i] = A[i]（f32 → bf16），TileLang 生成
34:  add  a0, s1, s2
36:  lw   a0, 0(a0)          # float 参数放整数寄存器 a0（RV64 soft-float 风格调用）
37:  call __truncsfbf2
38:  add  a1, s0, s3
40:  sh   a0, 0(a1)          # ← 从 a0（整数寄存器）取回 16 位结果，不是 fa0
```

**结论修正**：
- 在 **lp64d** 上，`__truncsfbf2(float) -> i16` 的返回值在 **`a0`**，调用方也从 `a0` 读。**这一侧 ABI 是一致的。**
- 我另外用 clang 编通用 C 函数验证：Clang 对 `_Float16`/`__bf16` **函数返回值**走 `fa0`（`fmv.w.x fa0, a0`）。这与 libcall 路径不同——**两条路径的约定确实不同，但 libcall 自身闭包是自洽的**。
- 所以真正的问题**不是"写 fa0 读 a0"**，而是下面第二、三节的**符号缺失 + 失败被静默固化**。

> 如果别的 MLIR/LLVM 版本、或在 ilp32/lp64（软浮点）目标上，`fa0`/`a0` 分歧可能真实出现。那个方向作为**跨工具链组合**的风险值得防，但不能作为本机失败的根因。

---

## 二、真正的根因：三层责任真空（带代码位置）

### 第 1 层：产生 libcall（`src/target/codegen_riscv.cc`）

| 位置 | 代码 | 问题 |
|---|---|---|
| `codegen_riscv.cc:6288-6301` | `CastValueToType`：浮点→浮点只按 `getWidth()` 比较，f32→bf16 生成 `arith::TruncFOp`（:6298），f16→f32 生成 `ExtFOp`（:6295） | **完全不区分 bf16/fp16/f8**，无条件产生 `arith.truncf/extf`，把"怎么实现"完全交给 LLVM backend |
| `codegen_riscv.cc:6211/6214` | 向量分支的 `ExtFOp`/`TruncFOp` | 产生 `vector<Nxf32> → vector<Nxbf16>`，**下游正则永远匹配不到**（见第 2 层） |
| `codegen_riscv.cc:1740-1745`、`:12657-12663` | `IsFloat2HalfRZCallExtern` 把 `__float2half_rz` 直接转成裸 `arith.truncf` | 一个本可做特判的显式语义锚点被降格，丢了"我要 fp16"的信息 |
| `codegen_riscv.cc:2090-2144` | `LowerScalarType`：f16→`F16Type`，bf16→`BF16Type` | 只做类型映射，不做实现选择 |

**这一层没有 bug，但有结构性缺陷**：它是"源头"，却对"下游能否提供实现"零感知。

### 第 2 层：MLIR 降级（`tilelang/jit/adapter/riscv/libgen.py`）

| 位置 | 内容 | 问题 |
|---|---|---|
| `libgen.py:306-318` | `_POST_LOW_PRECISION_LEGALIZATION_PASSES`，其中 `convert-arith-to-llvm`(:314) | bf16/f16 的 `truncf/extf` **原样交给**它 → LLVM `fptrunc/fpext` → RISC-V backend 选 `__truncsfbf2` |
| `libgen.py:61-152` | `_legalize_low_precision_conversions_for_riscv`（**正则改 MLIR 文本**） | 只处理 **FP8**（e4m3fn/fnuz）与 **F4**，**不处理 bf16/f16** |
| `libgen.py:98` | FP8 解码到 f16/bf16 时插入 `arith.truncf f32 to {dst}` | ← **自己又制造了一个 bf16/f16 libcall**，重新接回依赖 |
| `libgen.py:115` | FP8 编码的 src 是 f16/bf16 时插入 `arith.extf {src} to f32` | ← 同上，制造 `__extendhfsf2/__extendbfsf2` |
| `libgen.py:20-48` | 所有正则（`_F8_EXTF_RE` 等）都是**单行 + 固定类型字符串**匹配 | 向量类型、换行、generic 形式、带属性 → **静默漏匹配**，症状是"偶发 undefined symbol" |
| `libgen.py:147-151` | 依赖 `line.strip() == "module {"` 插入 helper 声明 | module 带属性时声明可能插到模块外 → 非法 MLIR |

**脆弱点**：用正则改文本代替 MLIR pass，任何 printer 格式变化都会静默失效。

### 第 3 层：符号提供与链接（`tilelang/jit/adapter/riscv/wrapper.py`）

| 位置 | 内容 | 问题 |
|---|---|---|
| `wrapper.py:943-959` | 链接命令：`-L{llvm_lib_dir} -lmlir_c_runner_utils -lmlir_runner_utils -lm` | **没有 `-lclang_rt.builtins`、没有 `-rtlib=`、没有 `-resource-dir`** |
| `wrapper.py:661-682` | `_resolve_riscv_gcc_root`：`TILELANG_RISCV_GCC_ROOT`→`GCC_ROOT`→`CC/CXX`推断→`/opt/gcc-native` | **完全不知道 compiler-rt 存在** |
| `wrapper.py:685-718` | `_riscv_gcc_runtime_library_dirs`：只找 `libgcc_s.so`(:706) | **连 `libgcc.a` 都不看**，更不找 `libclang_rt.builtins-*.a` |
| `wrapper.py:721-745` | `_riscv_clang_flags`：`--gcc-toolchain`、`--sysroot`、`-L` | 指向 gcc → clang 自动加的 `-lgcc` 只会命中**不含这些符号的 libgcc** |
| `wrapper.py:945` | `clang -shared` | **缺 `-Wl,--no-undefined`** → 未定义符号不报错，推迟到 `ctypes.CDLL` |
| `wrapper.py:987` | `ctypes.CDLL(str(self.path))` | 到这一步才抛 `OSError: undefined symbol` |
| `wrapper.py:870-919` | `build_qemu_executable`：`-nostdlib -static -ffreestanding` | freestanding 路径**更惨**：没有 helper 注入、没有 `-L` runtime → bf16 直接硬链接错误 |

**这是"符号提供者缺失"的直接原因**：`resolve_tool("clang")` 优先用 `TILELANG_RISCV_LLVM_ROOT/bin/clang`（`tilelang/tladapter/toolchain.py:93-109`），而这个自编 clang 的资源目录里**没有 compiler-rt builtins**（当初编 LLVM 时 `LLVM_ENABLE_RUNTIMES` 为空）。

### 第 4 层：失败被缓存固化（`tilelang/cache/kernel_cache.py`）

| 位置 | 内容 | 问题 |
|---|---|---|
| `kernel_cache.py:235-245` | `key_data`：`func` 脚本哈希、`args_repr`、`target`、`pass_configs`、`compile_flags`、`_get_base_key()` | **不含** `TILELANG_RISCV_CLANG_FLAGS` / `GCC_ROOT` / `_ABI` / `_LLVM_ROOT` / clang 路径 / builtins 库 |
| `kernel_cache.py:104-115` | `_get_base_key`：version + `platform.machine()` + `libtilelang.so` 内容哈希 | 换工具链、改环境变量**都不换 key** |
| `kernel_cache.py:105`、`:63` | 两者都 `@functools.cache` | 进程内改环境变量无效 |
| `kernel_cache.py:449`、`:572-577` | 把 `.so` 二进制写盘 | **坏的 `.so` 也被缓存** |
| `kernel_cache.py:598-599` | 只校验 `.so` + `.pkl` 存在即命中 | 修好环境后仍反复命中坏二进制 |

**这解释了"改了参数结果不变，必须 `TILELANG_DISABLE_CACHE=1`"**——不是玄学，是缓存键不含工具链指纹。

---

## 三、缺陷清单（按层 + 严重度）

| # | 层 | 位置 | 触发条件 | 影响 | 严重度 |
|---|---|---|---|---|---|
| 1 | 运行时/链接 | `wrapper.py:943-959` | 任何含 f32↔bf16/fp16 cast 的 kernel，且环境未注入 compiler-rt | `undefined symbol: __truncsfbf2/__truncsfhf2/__extendhfsf2`，或数值错（若符号被错版本满足） | **高** |
| 2 | 运行时/链接 | `wrapper.py:945` 缺 `-Wl,--no-undefined` | 同上 | 失败被推迟到运行时，且坏 `.so` 进缓存 | **高** |
| 3 | 缓存 | `kernel_cache.py:235-245` | 修好工具链后重跑 | 继续命中坏二进制，修了也白修 | **高** |
| 4 | IR/降级 | `libgen.py:98`、`:115` | FP8 kernel 且输出/输入是 f16/bf16 | FP8 路径**自己制造** bf16/f16 libcall | **高** |
| 5 | 后端 | `codegen_riscv.cc:6211/6214` | 向量化 bf16/f16 cast | 产生 vector cast，正则匹配不到，无逃生通道 | **中高** |
| 6 | 运行时/链接 | `wrapper.py:870-919` freestanding | 用 qemu 路径跑含 bf16 的 kernel | `-nostdlib` 下直接链接失败，无 helper | **中高** |
| 7 | 运行时 | `wrapper.py:679` | 非 SG2044 机器、未设 `TILELANG_RISCV_GCC_ROOT` | gcc root 为 None → 无 `--gcc-toolchain` → `crtbeginS.o` 找不到 | **中高** |
| 8 | 运行时 | `wrapper.py:721-745` | openEuler + 自编 clang | `--gcc-toolchain=/usr` 解析失败，需 `--gcc-install-dir` | **中** |
| 9 | 运行时 | `adapter.py:59`、`:274` vs `:113` | 用 `tilelang.compile(..., compile_flags=[...])` | `compile_flags` 被存但**不传给链接**，唯一通道是环境变量 | **中** |
| 10 | IR/降级 | `libgen.py:20-48` 正则 | MLIR printer 换行/向量类型/generic form | 静默漏匹配 → 偶发 undefined symbol | **中** |
| 11 | IR/降级 | `libgen.py:2109-2135` vs `:20-27` | 用 e5m2 / e3m4 / e4m3b11fnuz / e8m0fnu | codegen 支持但正则与 C helper 都没有 → 掉进 libcall | **中** |
| 12 | 构建 | `pyproject.toml:66` | 装 Cython 3.3 | `#error requires Python Limited API >= 3.9`，构建失败 | **中** |
| 13 | 构建 | `pyproject.toml:250-271` | riscv64 上用 cibuildwheel | `case` 分支 `*) exit 1`，无法产 wheel | **中** |
| 14 | 构建/文档 | `requirements.txt:4` vs `pyproject.toml:38` | riscv64 上 `pip install -r requirements.txt` | `torch-c-dlpack-ext` 未按 riscv64 跳过，构建失败 | **中** |
| 15 | 测试 | `TileKernels/conftest.py:22-24`、`TileOps/conftest.py:26-27` | 未设 `TILELANG_RISCV_LLVM_ROOT` | 落到 `/home/liquanyi/tool/llvm` 等作者路径，整目录 skip | **中** |
| 16 | 测试 | `TileOps/README.md:82` 等 | fp16/bf16 sigmoid | 依赖不支持的 `__fdividef`，数值 gap | **中** |
| 17 | 测试 | TileOps `paged_kv_cache_write` | fp8 造数据 | `torch.full(dtype=fp8)` → CPU `fill_cpu` 未实现（**PyTorch 限制**，非项目 bug） | **低** |
| 18 | 文档 | `BuildOnSG2044.md:3` vs `README.md:165` | 读者按文档走 | "validated" 与"仍需原生验证"自相矛盾 | **中** |
| 19 | CI | `ci-buddy-mlir.yml:56,111-118` | 任何人提交 | 只跑 x86 编译路径，运行时不回归 → 上述坑全部漏检 | **高（根）** |
| 20 | 构建 | `CMakeLists.txt:436-439` | 非 SG2044 的 riscv64 | 强制关 lld/mold（lld 其实可用），编译变慢 | **低** |

**归类**：
- **riscv64 特有**：#1-#9、#11、#13、#20（x86 不会遇到或不会这样失败）
- **跨平台通用**：#10、#12、#14、#15、#18、#19
- **上游依赖**：#17（PyTorch CPU fp8）、#16（`__fdividef` 来自 CUDA 系 intrinsic）

---

## 四、修改意见

### A. 最小改动、最高收益（建议优先做）

**A1. 链接期自动提供 compiler-rt builtins**（治 #1）
`wrapper.py:_riscv_clang_flags` 里增加：
```python
# 用 clang 自己报告 resource dir，而不是猜路径
resource_dir = _run_checked([clang, "-print-resource-dir"]).stdout.strip()
for cand in (f"{resource_dir}/lib/linux", f"{resource_dir}/lib/riscv64-unknown-linux-gnu"):
    if Path(cand).is_dir():
        flags += [f"-resource-dir={resource_dir}", f"-L{cand}", "-lclang_rt.builtins"]
```
更稳的做法是直接 `-rtlib=compiler-rt`，让 clang 自己找；但要保证该 clang 编过 `compiler-rt`。

**A2. 让失败提前到链接期**（治 #2）
`build_host_shared_library` 的 cmd 加 `-Wl,--no-undefined`。
代价：任何真实未定义符号都会立即报错——这正是我们要的。

**A3. 缓存键纳入工具链指纹**（治 #3）
`kernel_cache.py:_generate_key` 的 `key_data` 增加一项：
```python
"riscv_toolchain": {
    "clang_flags": os.environ.get("TILELANG_RISCV_CLANG_FLAGS", ""),
    "gcc_root": os.environ.get("TILELANG_RISCV_GCC_ROOT", ""),
    "abi": os.environ.get("TILELANG_RISCV_ABI", ""),
    "llvm_root": os.environ.get("TILELANG_RISCV_LLVM_ROOT", ""),
    "clang": _tool_fingerprint("clang"),   # realpath + mtime/size
}
```
并去掉 `_get_base_key` 上的 `functools.cache`（或把环境变量也纳入其 cache key）。

### B. 结构性修复（治根，工作量大）

**B1. 在 `CastValueToType` 内联 bf16/f16 转换**（治 #1/#5，绕过 libcall）
`codegen_riscv.cc:6288-6301` 对 `f32→bf16`、`f32→f16` 特判，生成位操作序列（round-to-nearest-even 或 RTZ）：
```
truncf f32->bf16:  bitcast i32 → (rounding) → shift right 16
extf  bf16->f32:   bitcast i16 → zext i32 → shift left 16
```
这样彻底不依赖 `__truncsfbf2`，向量路径也一并解决（#5）。
**注意**：必须与 Torch/参考实现的舍入模式一致，否则数值对拍会差 1 ulp。建议先做 RTZ/RN 两种并加单测。

**B2. 用 MLIR pass 替代文本正则**（治 #10/#11）
把 `_legalize_low_precision_conversions_for_riscv` 改写成真正的 MLIR `ConversionPattern`（在 codegen 或一个 upstream-able pass 里），而不是对文本 `re.match`。可保留正则作为 fallback，但要有"未匹配告警"而不是静默放行。

**B3. 补齐 FP8 全部方言**（治 #11）
`e5m2` / `e3m4` / `e4m3b11fnuz` / `e8m0fnu` 在 `LowerScalarType` 有映射，但 `libgen.py` 正则与 `wrapper.py` 的 C helper 都没有对应实现——要么补齐，要么在 codegen 阶段明确 `LOG(FATAL) << "fp8 fmt X not supported on riscv"`，别让它静默掉进 libcall。

**B4. 两条构建路径对齐**（治 #6）
`build_qemu_executable` 也应注入 `_emit_host_helper_source` 的 helper，或明确文档说"freestanding 路径不支持低精度 cast"。

### C. 工程化（治 #19，最重要但最难）

**C1. 建真实 riscv64 CI 回归**：哪怕每周一次手动触发，跑 `TileKernels` + `TileOps` 运行时测试。这一条能预防 #1-#9 的 80% 再生。
**C2. 把 `wrapper.py` 的环境探测改成显式契约**：gcc root、clang、builtins 全部 `-print-*` 探测，禁止硬编码 `/opt/gcc-native`。
**C3. 文档把 "validated" 改成 "x86-only compile-path validated"**，并给出"干净 riscv64 机器最小环境清单"。
**C4. `pyproject.toml` 补 Cython 上限、`requirements.txt` 补 riscv64 跳过、cibuildwheel 补 riscv64 分支或明确"不支持产 wheel"。**

---

## 五、关于"生成 hidden 局部兼容函数"这个方案的评价

你的方案（在 TileLang 内生成 `visibility("hidden")` 的局部 `__truncsfbf2/__truncdfbf2`，用 float 返回约定交付，覆盖两条构建路径，加缓存版本）**方向是对的**，具体评价：

**优点**
1. **自包含**：不依赖外部 compiler-rt，任何机器都能跑（治 #1 最彻底的方式之一）。
2. **`hidden` 可见性**：避免与将来真链接 compiler-rt 时符号冲突（好细节）。
3. **覆盖 fp32/fp64 两条转换**：比只修 bf16 完整。
4. **覆盖 host + freestanding 两条路径**：治 #6。
5. **缓存版本**：治 #3（但见下）。

**需要注意/可改进**
1. **"用 float 返回约定"这一条要按实测重新确认**：本机 lp64d 的 libcall 约定是 **`i16` 返回在 `a0`**（见第一节实证）。如果生成一个"float 返回"的替代品，就会**把调用方期待 `a0` 变成期待 `fa0`**，反而引入新的不匹配。**必须先确认目标 triple/ABI 下的真实返回寄存器**，或直接生成**和 LLVM 期望签名完全一致**的符号（`i16 __truncsfbf2(float)`），让调用方无感。
2. **不能只改 `-O3` 无关性**：这点对，但要注意 `arith-expand` / `ConvertToLLVMPattern` 在**任何**优化级别都会选 libcall，所以覆盖必须在 IR 层（B1）或链接层（A1），不能靠优化级别规避。
3. **不修改 Expand 算法/不放宽容差**：这是正确的工程自律，赞。
4. **"不是上游工具链全局修复"这一点你已注明**，很好——这个方案是**产品层规避**，不是 ABI 标准统一。若要根本解决，应在 LLVM/compiler-rt 层统一 `__truncsfbf2` 的 ABI（那属于 upstream 议题）。

**建议落点**（与 B1 二选一或并用）：
- 若选"生成局部符号"：在 `wrapper.py` 的 `_emit_host_helper_source` 加一段 C，定义
  `__attribute__((visibility("hidden"))) int16_t __truncsfbf2(float)`（签名严格对齐 LLVM RTLIB 期望），
  并同样覆盖 `__truncdfbf2`、`__extendbfsf2`、`__truncsfhf2`、`__extendhfsf2`。注意 C 里没有原生 bf16 类型时用位操作实现。
- 若选 B1（codegen 内联）：更彻底，且不依赖链接，但改动风险在数值正确性。

---

## 六、一句话结论

**这个项目在 RISC-V 上的根部问题不是"算错了"，而是"生成端、合法化端、链接端、缓存端、CI 端五处都没有把低精度 libcall 当作一条需要端到端交付的契约"**：

- 生成端（`codegen_riscv.cc:6298`）无条件产 libcall；
- 合法化端（`libgen.py:98/115`）只修 FP8 还顺手再造 libcall；
- 链接端（`wrapper.py:943-959`）不提供 symbol，且 `-shared` 吞掉了错误；
- 缓存端（`kernel_cache.py:235-245`）不记工具链指纹，把坏产物固化；
- CI 端（`ci-buddy-mlir.yml:111-118`）只测 x86 编译路径，永远发现不了。

**修任何单点都不够；A1+A2+A3 三件小事（约 40 行）能立刻把这条链路变成"要么对、要么立刻报错"，再配 C1 才能不复发。**
