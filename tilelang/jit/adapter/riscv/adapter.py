from __future__ import annotations

import shutil
from typing import Any, Callable

import numpy as np
import torch
from tvm import tir
from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.utils.target import determine_target

from .wrapper import HostKernelLibrary, load_host_module, load_host_module_from_binary


class _ExportableSharedLibrary:
    def __init__(self, library_path: str):
        self.library_path = library_path

    def export_library(self, path: str, **_: Any) -> None:
        shutil.copy2(self.library_path, path)


class RiscvKernelAdapter(BaseKernelAdapter):
    host_mod: tvm.IRModule | None = None
    device_mod: tvm.IRModule | None = None
    rt_mod: tvm.runtime.Module | None = None
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        rt_mod: tvm.runtime.Module | None = None,
        host_kernel_source: str | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.target = Target.canon_target(determine_target(target))
        self.host_mod = host_mod
        self.device_mod = device_mod
        self.rt_mod = rt_mod
        self.host_kernel_source = host_kernel_source
        self.device_kernel_source = device_kernel_source
        self.verbose = verbose
        self.pass_configs = pass_configs or {}
        self.compile_flags = compile_flags
        self._host_library: HostKernelLibrary | None = None
        self.libpath: str | None = None
        self.executable: _ExportableSharedLibrary | None = None

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self._materialize_host_library()
        self._post_init()

    @property
    def prim_func(self) -> tir.PrimFunc:
        _, func = next(iter(self.ir_module.functions.items()))
        return func

    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map: dict[tir.Var, tuple[int, int]] = {}
        self._dynamic_symbolic_name_map: dict[str, tuple[int, int]] = {}
        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if not isinstance(shape, tir.Var):
                    continue
                existing = dynamic_symbolic_map.get(shape)
                should_update = existing is None
                if existing is not None:
                    existing_param_idx, _ = existing
                    should_update = existing_param_idx in self.result_idx and i not in self.result_idx
                if should_update:
                    dynamic_symbolic_map[shape] = (i, j)
                    self._dynamic_symbolic_name_map[shape.name] = (i, j)
        return dynamic_symbolic_map

    def _lookup_dynamic_symbolic(self, value: tir.Var) -> tuple[int, int]:
        if value in self.dynamic_symbolic_map:
            return self.dynamic_symbolic_map[value]
        if value.name in self._dynamic_symbolic_name_map:
            return self._dynamic_symbolic_name_map[value.name]
        raise KeyError(f"Dynamic symbolic variable `{value.name}` not found in kernel signature")

    def _get_host_library(self) -> HostKernelLibrary:
        if self._host_library is None:
            source_value = self.rt_mod if self.rt_mod is not None else self.device_kernel_source
            if source_value is None:
                raise RuntimeError("No MLIR source module is available for host execution")
            self._host_library = load_host_module(source_value)
            self.libpath = str(self._host_library.path)
            self.executable = _ExportableSharedLibrary(self.libpath)
        return self._host_library

    def _materialize_host_library(self) -> None:
        self._get_host_library()

    def close(self) -> None:
        if self._host_library is not None:
            self._host_library.close()
            self._host_library = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _resolve_output_shape(self, param: KernelParam, provided_args: dict[int, Any]) -> list[int]:
        analyzer = tvm.arith.Analyzer()

        shape: list[int] = []
        for dim in param.shape:
            expr = dim
            if isinstance(expr, tir.PrimExpr):
                substitution: dict[tir.Var, tir.PrimExpr] = {}

                def collect_var(node):
                    if isinstance(node, tir.Var):
                        ref_tensor_idx, ref_shape_idx = self._lookup_dynamic_symbolic(node)
                        ref_tensor = provided_args[ref_tensor_idx]
                        substitution[node] = tir.IntImm(node.dtype, int(ref_tensor.shape[ref_shape_idx]))

                tvm.tir.stmt_functor.post_order_visit(expr, collect_var)
                if substitution:
                    expr = tvm.tir.stmt_functor.substitute(expr, substitution)
                expr = analyzer.simplify(expr)
            if isinstance(expr, tir.IntImm):
                shape.append(int(expr.value))
            elif isinstance(expr, int):
                shape.append(expr)
            else:
                shape.append(int(expr))
        return shape

    def _normalize_tensor_arg(self, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor input, but got {type(value)}")
        if value.device.type != "cpu":
            raise ValueError("The current riscv host runner only supports CPU tensors")
        if not value.is_contiguous():
            value = value.contiguous()
        return value.detach()

    def _torch_tensor_to_numpy(self, value: torch.Tensor) -> np.ndarray:
        dtype = value.dtype
        if dtype == torch.bfloat16:
            return value.view(torch.uint16).numpy().view(np.dtype("bfloat16"))
        if dtype == getattr(torch, "float4_e2m1fn_x2", None):
            return value.view(torch.uint8).numpy().view(np.int8)
        if dtype == getattr(torch, "float8_e4m3fn", None):
            return value.view(torch.uint8).numpy()
        if dtype == getattr(torch, "float8_e5m2", None):
            return value.view(torch.uint8).numpy()
        if dtype == getattr(torch, "float8_e4m3fnuz", None):
            return value.view(torch.uint8).numpy()
        if dtype == getattr(torch, "float8_e5m2fnuz", None):
            return value.view(torch.uint8).numpy()
        return value.numpy()

    def _host_storage_torch_dtype(self, param: KernelParam) -> torch.dtype:
        if param.is_float4():
            return torch.int8
        if param.torch_dtype() == torch.uint8:
            return torch.int8
        return param.torch_dtype()

    def _restore_result_torch_dtype(self, tensor: torch.Tensor, param: KernelParam) -> torch.Tensor:
        if param.torch_dtype() == torch.uint8 and tensor.dtype == torch.int8:
            return tensor.view(torch.uint8)
        return tensor

    def _host_storage_shape(self, param: KernelParam, provided_args: dict[int, Any]) -> list[int]:
        return self._resolve_output_shape(param, provided_args)

    def _convert_torch_func(self) -> Callable[..., Any]:
        param_dtypes = [self._host_storage_torch_dtype(param) for param in self.params]

        def func(*inputs: Any):
            expected_inputs = len(self.params) - len(self.result_idx)
            if len(inputs) != expected_inputs:
                raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} were provided")

            provided_args: dict[int, Any] = {}
            input_idx = 0
            for i, param in enumerate(self.params):
                if i in self.result_idx:
                    continue
                value = inputs[input_idx]
                input_idx += 1
                if param.is_scalar():
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            raise ValueError("Scalar kernel parameters must be passed as Python scalars or 0-d tensors")
                        value = value.item()
                    provided_args[i] = value
                else:
                    provided_args[i] = self._normalize_tensor_arg(value)

            host_args: list[Any] = []
            for i, param in enumerate(self.params):
                if i in self.result_idx:
                    shape = self._host_storage_shape(param, provided_args)
                    tensor = torch.empty(shape, dtype=param_dtypes[i], device="cpu")
                    host_args.append(tensor)
                    continue

                host_args.append(provided_args[i])

            native_args = [
                self._torch_tensor_to_numpy(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in host_args
            ]
            self._get_host_library()(*native_args)

            if len(self.result_idx) == 1:
                result_idx = self.result_idx[0]
                return self._restore_result_torch_dtype(host_args[result_idx], self.params[result_idx])
            return [
                self._restore_result_torch_dtype(host_args[i], self.params[i])
                for i in self.result_idx
            ]

        return func

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
        result_idx: list[int],
        target: str,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_kernel_source: str,
        device_kernel_source: str,
        kernel_lib_path: str,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.target = Target.canon_target(determine_target(target))
        adapter.host_mod = None
        adapter.device_mod = None
        adapter.rt_mod = None
        adapter.host_kernel_source = host_kernel_source
        adapter.device_kernel_source = device_kernel_source
        adapter.verbose = verbose
        adapter.pass_configs = pass_configs or {}
        adapter.compile_flags = compile_flags
        adapter.libpath = kernel_lib_path
        adapter._host_library = load_host_module_from_binary(device_kernel_source, kernel_lib_path)
        adapter.executable = _ExportableSharedLibrary(kernel_lib_path)

        if isinstance(func_or_mod, tir.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
        adapter._post_init()
        return adapter

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if self.rt_mod is not None and hasattr(self.rt_mod, "inspect_source"):
            return self.rt_mod.inspect_source()
        if isinstance(self.device_kernel_source, str):
            return self.device_kernel_source
        return super().get_kernel_source(kernel_only=kernel_only)

    def get_host_source(self) -> str:
        return self.get_kernel_source(kernel_only=False)
