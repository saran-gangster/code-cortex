"""TensorRT execution using PyTorch CUDA tensors as device buffers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _dependencies():
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - optional GPU dependency
        raise RuntimeError(
            "TensorRT Python bindings are unavailable; install the NVIDIA package matching your CUDA/JetPack"
        ) from exc
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError("PyTorch with CUDA support is required for TensorRT buffer management") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to PyTorch; TensorRT inference cannot run")
    return trt, torch


class TensorRTRunner:
    """A batch-one runner supporting TensorRT named-I/O and legacy binding APIs."""

    def __init__(self, engine_path: str | Path) -> None:
        trt, torch = _dependencies()
        path = Path(engine_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"engine must be a non-empty regular file: {path}")
        self.trt = trt
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.modern = hasattr(self.engine, "num_io_tensors")

    def _torch_dtype(self, trt_dtype):
        trt, torch = self.trt, self.torch
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        if trt_dtype not in mapping:
            raise RuntimeError(f"unsupported TensorRT tensor dtype: {trt_dtype}")
        return mapping[trt_dtype]

    def infer(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.modern:
            return self._infer_modern(inputs)
        return self._infer_legacy(inputs)

    def _infer_modern(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        trt, torch = self.trt, self.torch
        tensors = {}
        output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if name not in inputs:
                    raise ValueError(f"missing TensorRT input: {name}")
                host = np.ascontiguousarray(inputs[name])
                self.context.set_input_shape(name, tuple(host.shape))
                device = torch.as_tensor(host, device="cuda")
                expected = self._torch_dtype(self.engine.get_tensor_dtype(name))
                tensors[name] = device.to(expected)
            else:
                output_names.append(name)
        if hasattr(self.context, "infer_shapes"):
            insufficient = list(self.context.infer_shapes())
            if insufficient:
                raise RuntimeError(
                    "TensorRT cannot infer shapes; insufficient tensors: " + ", ".join(insufficient)
                )
        elif hasattr(self.context, "all_binding_shapes_specified") and not self.context.all_binding_shapes_specified:
            raise RuntimeError("TensorRT input shapes are not fully specified")
        for name in output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"TensorRT left output shape unresolved for {name}: {shape}")
            tensors[name] = torch.empty(
                shape, dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)), device="cuda"
            )
        for name, tensor in tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT execute_async_v3 returned failure")
        torch.cuda.current_stream().synchronize()
        return {name: tensors[name].detach().cpu().numpy() for name in output_names}

    def _infer_legacy(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self.torch
        tensors = {}
        bindings = [0] * self.engine.num_bindings
        output_names = []
        # Set every dynamic input before asking TensorRT for any output shape. Binding
        # order is not guaranteed to put all inputs before all outputs.
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            if not self.engine.binding_is_input(index):
                continue
            if name not in inputs:
                raise ValueError(f"missing TensorRT input: {name}")
            host = np.ascontiguousarray(inputs[name])
            self.context.set_binding_shape(index, tuple(host.shape))
            tensor = torch.as_tensor(host, device="cuda").to(
                self._torch_dtype(self.engine.get_binding_dtype(index))
            )
            tensors[name] = tensor
            bindings[index] = tensor.data_ptr()
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("TensorRT input shapes are not fully specified")
        for index in range(self.engine.num_bindings):
            if self.engine.binding_is_input(index):
                continue
            name = self.engine.get_binding_name(index)
            shape = tuple(self.context.get_binding_shape(index))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"TensorRT left output shape unresolved for {name}: {shape}")
            tensor = torch.empty(
                shape,
                dtype=self._torch_dtype(self.engine.get_binding_dtype(index)),
                device="cuda",
            )
            tensors[name] = tensor
            bindings[index] = tensor.data_ptr()
            output_names.append(name)
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v2(bindings=bindings, stream_handle=stream):
            raise RuntimeError("TensorRT execute_async_v2 returned failure")
        torch.cuda.current_stream().synchronize()
        return {name: tensors[name].detach().cpu().numpy() for name in output_names}


__all__ = ["TensorRTRunner"]
