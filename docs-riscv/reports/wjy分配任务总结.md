# wjy 分配任务总结：2 个 engram 算子

> 分配来源：`TileLang算子列表.xlsx`（216 个算子，按负责人划分）
> 负责人：**wjy**
> 分配数量：**2 个**（均为 TileKernels 来源）
> 验收环境：`openeuler-riscv64`（RV64GCV，64 核，openEuler 24.03，Python 3.11.6）
> 验收时间：2026-11-05
> **结果：2/2 全部通过（2 passed in 28.02s）**

---

## 一、任务清单

| # | 算子名 | 来源 | 源文件路径 |
|---|---|---|---|
| 1 | `engram_grad_w_reduce_kernel` | TileKernels | `TileKernels/tile_kernels/engram/engram_grad_w_reduce_kernel.py` |
| 2 | `engram_hash_kernel` | TileKernels | `TileKernels/tile_kernels/engram/engram_hash_kernel.py` |

---

## 二、算子 1：`engram_hash_kernel`

### 定位
Engram 模块的**前向路由/寻址算子**（非反向，无梯度）。

### 计算语义
对每个 token 的 n-gram token id 做逐层 xor-multiply 哈希，再对每层各 embedding 表的 vocab size 取模并加偏移，产出 embedding 表的行号索引。

### 接口
- 工厂：`get_engram_hash_kernel(max_ngram_size=3, num_ngram_layers=2, num_embed_table_per_ngram=8)`
- `num_tokens` 是动态维度（`T.dynamic('num_tokens')`）
- 编译参数：`threads=32`，`blk_m=32`；中间量 `hash_local` 用 **int64** 防溢出

### 张量签名
| 张量 | shape | dtype | 角色 |
|---|---|---|---|
| `ngram_token_ids` | (num_tokens, max_ngram_size) | int32 | 输入 |
| `multipliers` | (num_ngram_layers, max_ngram_size) | int64 | 输入（哈希乘子） |
| `vocab_sizes` | (num_ngram_layers, max_ngram_size-1, num_embed_table_per_ngram) | int32 | 输入 |
| `offsets` | (num_ngram_layers, (max_ngram_size-1)*num_embed_table_per_ngram) | int32 | 输入 |
| `output` | (num_ngram_layers, num_tokens, ...) | int32 | **输出** |

### 调用方式
```python
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

### 验证结果
```
testing/python/riscv/TileKernels/test_engram_hash_kernel.py::test_engram_hash_kernel_runtime_compare PASSED
```
- 参考实现：`tile_kernels.torch.engram.engram_hash_ref`（PyTorch）
- **本算子无浮点**（全 int32/int64），因此不受 bf16/fp16 工具链问题影响

---

## 三、算子 2：`engram_grad_w_reduce_kernel`

### 定位
Engram 模块的**反向梯度规约算子**（backward / grad + reduce）。

### 计算语义
把多个 persistent block 上的部分权重梯度 `grad_w_partial` 沿第 0 维规约，与 RMSNorm 权重**交叉相乘**，然后**原地累加**进两个梯度张量：

```
grad_weight_hidden += sum_over_blocks(grad_w_partial) * weight_embed.float()
grad_weight_embed  += sum_over_blocks(grad_w_partial) * weight_hidden.float()
```

### 接口
- 工厂：`get_engram_grad_w_reduce_kernel(hidden_size, num_persistent_blocks, hc_mult=4)`
- 硬约束：`hidden_size % 512 == 0`，`num_persistent_blocks % 4 == 0`
- 编译参数：`threads=128`，`blk_d=512`，`num_batches=4`，规约循环 `T.Pipelined(num_stages=2)`

### 张量签名
| 张量 | shape | dtype | 角色 |
|---|---|---|---|
| `grad_w_partial` | (num_persistent_blocks, hc_mult, hidden_size) | fp32 | 输入 |
| `weight_hidden` | (hc_mult, hidden_size) | **bfloat16** | 输入 |
| `weight_embed` | (hc_mult, hidden_size) | **bfloat16** | 输入 |
| `grad_weight_hidden` | (hc_mult, hidden_size) | fp32 | 输入/输出（原地累加） |
| `grad_weight_embed` | (hc_mult, hidden_size) | fp32 | 输入/输出（原地累加） |

### 调用方式
```python
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

### 验证结果
```
testing/python/riscv/TileKernels/test_engram_grad_w_reduce_kernel.py::test_engram_grad_w_reduce_kernel_runtime_compare PASSED
```
- 参考实现：`grad_weight.clone() + grad_w_partial.sum(dim=0) * weight_other.float()`
- **本算子含 bf16 权重** → 是 bf16 工具链问题（compiler-rt libcall）的敏感点

---

## 四、关键前提（跑通这两个算子的必要条件）

| 前提 | 说明 |
|---|---|
| `3rdparty/TileKernels` 子模块 | pin `36d9e45d` |
| tilelang 已构建 | `pip install -e . --no-build-isolation` |
| LLVM/MLIR 24 | `TILELANG_RISCV_LLVM_ROOT` 指向自编工具链 |
| **compiler-rt builtins** | `TILELANG_RISCV_CLANG_FLAGS` 含 `-L.../riscv64-unknown-linux-gnu -lclang_rt.builtins` |
| `numpy<2` | 与 torch 2.4.1 的 ABI 兼容 |
| `TILELANG_DISABLE_CACHE=1` | 避免命中旧的坏产物 |

完整命令：
```bash
export TILELANG_RISCV_LLVM_ROOT=/home/wangjiayue/llvm-project/build
export TILELANG_RISCV_MLIR_MODE=ON
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12 \
  -L/home/wangjiayue/llvm-project/build/lib/clang/24/lib/riscv64-unknown-linux-gnu \
  -lclang_rt.builtins"

cd ~/tilelang-riscv-clean
TILELANG_DISABLE_CACHE=1 python3 -m pytest -v \
  testing/python/riscv/TileKernels/test_engram_hash_kernel.py \
  testing/python/riscv/TileKernels/test_engram_grad_w_reduce_kernel.py
```

---

## 五、结论

**两个算子均已在真实 riscv64 机器上端到端跑通（编译 → 加载 → 执行 → 数值对拍全部通过）。**

| 算子 | 结果 | 说明 |
|---|---|---|
| `engram_hash_kernel` | ✅ PASSED | int32/int64 路径，不涉及低精度浮点 |
| `engram_grad_w_reduce_kernel` | ✅ PASSED | 含 bf16，验证了 compiler-rt 链接修复有效 |

**遇到的困难：无新问题** —— 因为此前已修好 compiler-rt 链接（`__truncsfhf2`/`__extendhfsf2` 缺失问题），正好覆盖了 `engram_grad_w_reduce_kernel` 的 bf16 需求。

完整 trace 记录见：[trace_engram_kernels.md](./trace_engram_kernels.md)
