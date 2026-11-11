# Understanding Targets

TileLang is built on top of TVM, which relies on **targets** to describe the device you want to compile for.
The target determines which code generator is used (CUDA, HIP, Metal, LLVM, …) and allows you to pass
device-specific options such as GPU architecture flags. This page summarises how to pick and customise a target
when compiling TileLang programs.

## Common target strings

TileLang ships with a small set of common targets; each accepts the full range of TVM options so you can fine-tune
the generated code. The most frequent choices are listed below:

| Base name | Description |
| --------- | ----------- |
| `auto` | Detects CUDA → HIP → Metal in that order. Useful when running the same script across machines. |
| `cuda` | NVIDIA GPUs. Supports options such as `-arch=sm_80`, `-max_num_threads=1024`, etc. |
| `cutedsl` | NVIDIA CUTLASS/CuTe DSL backend. Requires `nvidia-cutlass-dsl`. `cuda` options can also be applied to this target. |
| `hip` | AMD GPUs via ROCm. Options like `-mcpu=gfx90a` can be appended. |
| `metal` | Apple Silicon GPUs (arm64 Macs). |
| `llvm` | CPU execution; accepts the standard TVM LLVM switches. |
| `riscv` | Structured MLIR pipeline for RISC-V execution. Intended for MLIR lowering through Buddy/LLVM and native RVV execution. |
| `webgpu` | Browser / WebGPU runtimes. |
| `c` | Emit plain C source for inspection or custom toolchains. |

To add options, append them after the base name, separated by spaces. For example:

```python
target = "cuda -arch=sm_90"
kernel = tilelang.compile(func, target=target, execution_backend="cython")
# or
@tilelang.jit(target=target)
def compiled_kernel(*args):
    return func(*args)
```

The same convention works for HIP, LLVM, or the RISC-V MLIR backend (e.g. `hip -mcpu=gfx940`,
`llvm -mtriple=x86_64-linux-gnu`, `riscv -mtriple=riscv64-unknown-linux-gnu`).

## RISC-V structured backend

TileLang also exposes a structured MLIR-backed backend for RISC-V bring-up:

```python
kernel = tilelang.compile(func, target="riscv")
```

This path is intentionally different from the CUDA/HIP/Metal backends:

- TileLang lowers TIR into a structured MLIR module.
- The Python runtime adapter then lowers through `mlir-opt`/`mlir-translate`/`llc`.
- Native execution can target a real RISC-V machine by setting up an LLVM/MLIR toolchain and a native RISC-V C/C++ toolchain.

The intended shared path here is `TileLang -> MLIR Linalg -> MLIR vector/RVV lowering`. This document does not assume
shared Triton frontend lowering passes; the commonality starts at the Linalg/downstream MLIR pipeline.

For SG2044-style native bring-up, the practical environment variables are:

```bash
export TILELANG_RISCV_LLVM_ROOT=/path/to/buddy-llvm-build
export Z3_ROOT=/path/to/z3-install-prefix
export CC=/path/to/gcc
export CXX=/path/to/g++
```

On mixed developer hosts you can still use `target="riscv"` for compilation artifact export without executing on the
local machine.

### Scope and limitations

The validated path today is a maintainable native bring-up path, not a claim that every TileLang program is already
supported on RISC-V.

- the verified flow is `TileLang -> MLIR Linalg -> MLIR lowering -> native host adapter -> SG2044 execution`
- x86 is sufficient for source review and compiler-path review, but not for RVV execution validation
- first-time optional Torch DLPack extension builds can be avoided with `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`
- use `target="riscv"` as the only supported target spelling for this backend

### Advanced: Specify Exact Hardware

When you already know the precise GPU model, you can encode it in the target string—either via `-arch=sm_XX` or by
using one of TVM’s pre-defined target tags such as `nvidia/nvidia-h100`.  Supplying this detail is optional for
TileLang in general use, but it becomes valuable when the TVM cost model is enabled (e.g. during autotuning).  The
cost model uses the extra attributes to make better scheduling predictions.  If you skip this step (or do not use the
cost model), generic targets like `cuda` or `auto` are perfectly fine.

All CUDA compute capabilities recognised by TVM’s target registry are listed below.  Pick the one that matches your
GPU and append it to the target string or use the corresponding target tag—for example `nvidia/nvidia-a100`.

| Architecture | GPUs (examples) |
| ------------ | ---------------- |
| `sm_20` | `nvidia/tesla-c2050`, `nvidia/tesla-c2070` |
| `sm_21` | `nvidia/nvs-5400m`, `nvidia/geforce-gt-520` |
| `sm_30` | `nvidia/quadro-k5000`, `nvidia/geforce-gtx-780m` |
| `sm_35` | `nvidia/tesla-k40`, `nvidia/quadro-k6000` |
| `sm_37` | `nvidia/tesla-k80` |
| `sm_50` | `nvidia/quadro-k2200`, `nvidia/geforce-gtx-950m` |
| `sm_52` | `nvidia/tesla-m40`, `nvidia/geforce-gtx-980` |
| `sm_53` | `nvidia/jetson-tx1`, `nvidia/jetson-nano` |
| `sm_60` | `nvidia/tesla-p100`, `nvidia/quadro-gp100` |
| `sm_61` | `nvidia/tesla-p4`, `nvidia/quadro-p6000`, `nvidia/geforce-gtx-1080` |
| `sm_62` | `nvidia/jetson-tx2` |
| `sm_70` | `nvidia/nvidia-v100`, `nvidia/quadro-gv100` |
| `sm_72` | `nvidia/jetson-agx-xavier` |
| `sm_75` | `nvidia/nvidia-t4`, `nvidia/quadro-rtx-8000`, `nvidia/geforce-rtx-2080` |
| `sm_80` | `nvidia/nvidia-a100`, `nvidia/nvidia-a30` |
| `sm_86` | `nvidia/nvidia-a40`, `nvidia/nvidia-a10`, `nvidia/geforce-rtx-3090` |
| `sm_87` | `nvidia/jetson-agx-orin-32gb`, `nvidia/jetson-agx-orin-64gb` |
| `sm_89` | `nvidia/geforce-rtx-4090` |
| `sm_90a` | `nvidia/nvidia-h100` (DPX profile) |
| `sm_100a` | `nvidia/nvidia-b100` |

Refer to NVIDIA’s [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) page or the TVM source
(`3rdparty/tvm/src/target/tag.cc`) for the latest mapping between devices and compute capabilities.

## Creating targets programmatically

If you prefer working with TVM’s `Target` objects, TileLang exposes the helper
`tilelang.utils.target.determine_target` (returns a canonical target string by default, or the `Target`
object when `return_object=True`):

```python
from tilelang.utils.target import determine_target

tvm_target = determine_target("cuda -arch=sm_80", return_object=True)
kernel = tilelang.compile(func, target=tvm_target)
```

You can also build targets directly through TVM:

```python
from tvm.target import Target

target = Target("cuda", host="llvm")
target = target.with_host(Target("llvm -mcpu=skylake"))
```

TileLang accepts either `str` or `Target` inputs; internally they are normalised and cached using the canonical
string representation.  **In user code we strongly recommend passing target strings rather than
`tvm.target.Target` instances—strings keep cache keys compact and deterministic across runs, whereas constructing
fresh `Target` objects may lead to slightly higher hashing overhead or inconsistent identity semantics.**

## Discovering supported targets in code

Looking for a quick reminder of the built-in base names and their descriptions? Use:

```python
from tilelang.utils.target import describe_supported_targets

for name, doc in describe_supported_targets().items():
    print(f"{name:>6}: {doc}")
```

This helper mirrors the table above and is safe to call at runtime (for example when validating CLI arguments).

## Troubleshooting tips

- If you see `Target cuda -arch=sm_80 is not supported`, double-check the spellings and that the option is valid for
  TVM. Any invalid switch will surface as a target-construction error.
- Runtime errors such as “no kernel image is available” usually mean the `-arch` flag does not match the GPU you are
  running on. Try dropping the flag or switching to the correct compute capability.
- When targeting multiple environments, use `auto` for convenience and override with an explicit string only when
  you need architecture-specific tuning.
- If `target="riscv"` fails during configuration with a Z3 lookup error, set `Z3_ROOT` to a local Z3 install prefix;
  the `z3-solver` Python wheel is not available on every architecture.
