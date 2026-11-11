# Trace：跑通两个 engram 算子（riscv64）

> 时间：2026-11-05 16:33 CST
> 机器：`openeuler-riscv64`（64 核 RISC-V）
> 代码：`~/tilelang-riscv-clean`（HEAD = `32fce5ed`）
> 目标：跑通 `test_engram_hash_kernel.py` 和 `test_engram_grad_w_reduce_kernel.py`

---

## 结果速览

```
✅ 2 passed in 28.02s（exit=0）

test_engram_hash_kernel_runtime_compare         PASSED [ 50%]
test_engram_grad_w_reduce_kernel_runtime_compare PASSED [100%]
```

**两个算子均跑通。**

---

## 完整 trace

### [T0] 环境快照

```
主机: openeuler-riscv64 | riscv64 | 64 核
代码: 32fce5ed [RISC-V] Build Python extension in Buddy CI

依赖:
  numpy 1.26.4 | torch 2.4.1 | z3 4.15.4

LLVM 工具链:
  mlir-opt      : OK
  mlir-translate: OK
  clang         : OK

compiler-rt:
  /home/wangjiayue/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu/libclang_rt.builtins.a

测试文件:
  testing/python/riscv/TileKernels/test_engram_grad_w_reduce_kernel.py
  testing/python/riscv/TileKernels/test_engram_hash_kernel.py

算子源文件（子模块 pin 36d9e45d）:
  3rdparty/TileKernels/tile_kernels/engram/engram_grad_w_reduce_kernel.py
  3rdparty/TileKernels/tile_kernels/engram/engram_hash_kernel.py
```

**结论**：所有前提就位（之前 TileKernels 48/48 全过已验证）。

### [T1] 设环境变量

```bash
export TILELANG_RISCV_LLVM_ROOT=/home/wangjiayue/llvm-project/build
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 \
  -L/home/wangjiayue/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu \
  -lclang_rt.builtins"
export TILELANG_DISABLE_CACHE=1
```

**关键点**：
- `TILELANG_RISCV_CLANG_FLAGS` 必须带 **compiler-rt builtins 的 -L 和 -l**，否则 fp16/bf16 链接缺 `__truncsfhf2`
- `TILELANG_DISABLE_CACHE=1` 强制重编，避免用之前失败时的缓存

### [T2] 跑测试

```bash
cd ~/tilelang-riscv-clean
python3 -m pytest -v \
  testing/python/riscv/TileKernels/test_engram_hash_kernel.py \
  testing/python/riscv/TileKernels/test_engram_grad_w_reduce_kernel.py
```

**输出**：
```
collected 2 items
testing/python/riscv/TileKernels/test_engram_hash_kernel.py::test_engram_hash_kernel_runtime_compare PASSED [ 50%]
testing/python/riscv/TileKernels/test_engram_grad_w_reduce_kernel.py::test_engram_grad_w_reduce_kernel_runtime_compare PASSED [100%]
============================== 2 passed in 28.02s ==============================
```

**28 秒** = 每个 kernel 编译约 14 秒（TIR→MLIR→LLVM→链接→执行）。

---

## 遇到的困难与观察

### 观察 1：本次**没有新困难**

原因：之前修的 compiler-rt 链接问题**已经覆盖**这两个算子的需求。

### 观察 2：`engram_grad_w_reduce` 是 bf16 敏感算子

- 输入：`grad_w_partial`(fp32) + `weight_hidden`/`weight_embed`(**bf16**) + 累加 buffer(fp32)
- 如果 compiler-rt 没修好，会报 `undefined symbol: __truncsfhf2` 或数值全错
- **本次通过 = 证明 compiler-rt 链接修复对 bf16 有效**

### 观察 3：`engram_hash` 是 int32/int64，无浮点

- 输入全是 int32/int64，输出 int32
- **不涉及 fp16/bf16**，所以它本来就不会有数值坑
- 它通过是"预期内"，不证明 compiler-rt 修复

### 观察 4：`TILELANG_DISABLE_CACHE=1` 是必须的

tilelang 的缓存会把编译产物记住。如果之前链接失败的 `.so` 被缓存，重跑会直接复用旧坏的。**必须清缓存强制重编**。

### 观察 5：`TILELANG_RISCV_CLANG_FLAGS` 必须带 compiler-rt 路径

链接命令默认只有 `-lmlir_c_runner_utils -lmlir_runner_utils -lm`，**没有 `-lclang_rt.builtins`**。fp16/bf16 转换函数 `__truncsfhf2`/`__extendhfsf2` 在 compiler-rt 里，必须显式链接。

### 观察 6：`numpy<2` 必须保持

pip 装 tilelang 时会自动升级 numpy 到 2.x，torch 2.4.1 是按 numpy 1.x 编译的，会崩。**装完 tilelang 后必须降回 `numpy<2`**。

---

## 调用范式（供你自己写脚本）

### engram_hash_kernel（前向，int32 输出）

```python
from tile_kernels.engram.engram_hash_kernel import get_engram_hash_kernel
from tile_kernels.torch.engram import engram_hash_ref, make_offsets
import torch, tilelang

ngram_token_ids = torch.randint(0, 1000, (7, 3), dtype=torch.int32)
multipliers     = torch.randint(1, 1000, (2, 3), dtype=torch.int64)
vocab_sizes     = torch.randint(100, 200, (2, 2, 8), dtype=torch.int32)
offsets         = make_offsets(vocab_sizes)

kernel = tilelang.compile(
    get_engram_hash_kernel.get_tir(3, 2, 8),   # (max_ngram, num_layers, tables_per_ngram)
    out_idx=[4],                                # 第 5 个参数 output 由 runtime 分配
    target="riscv",
)
try:
    actual = kernel(ngram_token_ids, multipliers, vocab_sizes, offsets)
finally:
    kernel.close()
```

### engram_grad_w_reduce_kernel（反向，bf16 权重 + fp32 累加，全 in-place）

```python
from tile_kernels.engram.engram_grad_w_reduce_kernel import get_engram_grad_w_reduce_kernel
import torch, tilelang

grad_w_partial     = torch.randn(4, 4, 512, dtype=torch.float32)   # (blocks, hc, hidden)
weight_hidden      = torch.randn(4, 512,    dtype=torch.bfloat16)  # bf16 权重
weight_embed       = torch.randn(4, 512,    dtype=torch.bfloat16)
grad_weight_hidden = torch.randn(4, 512,    dtype=torch.float32)   # 累加 buffer, in-place
grad_weight_embed  = torch.randn(4, 512,    dtype=torch.float32)

kernel = tilelang.compile(
    get_engram_grad_w_reduce_kernel.get_tir(512, 4, 4),  # (hidden, num_blocks, hc_mult)
    out_idx=[],                                           # 全部 in-place
    target="riscv",
)
try:
    kernel(grad_w_partial, weight_hidden, weight_embed,
           grad_weight_hidden, grad_weight_embed)
finally:
    kernel.close()
```

**区别**：
- `hash`：`out_idx=[4]`（第 5 个参数是新建 output）
- `grad_w_reduce`：`out_idx=[]`（所有张量 in-place 修改）

---

## 前提清单（跑通必须）

| 前提 | 本次状态 |
|---|---|
| `3rdparty/TileKernels` 子模块 | ✅ pin `36d9e45d` |
| tilelang 已构建（含 `tladapter._native`） | ✅ |
| LLVM/MLIR 24 + compiler-rt builtins | ✅ |
| 环境变量（5 个） | ✅ 见 [T1] |
| `numpy<2` | ✅ 1.26.4 |
| `TILELANG_DISABLE_CACHE=1` | ✅ |

---

## 结论

**两个算子在已修好的 riscv64 环境上均跑通（2 passed）。**

- `engram_hash`（int32/无浮点）：预期内通过
- `engram_grad_w_reduce`（bf16 权重）：**证明 compiler-rt 链接修复有效**

**没有遇到新困难**——因为之前修的 compiler-rt 链接问题已经覆盖了这两个算子的需求。如果你的环境还没修 compiler-rt，`engram_grad_w_reduce` 会报 `undefined symbol: __truncsfhf2` 或数值错，`engram_hash` 则能过（因为它不用浮点）。
