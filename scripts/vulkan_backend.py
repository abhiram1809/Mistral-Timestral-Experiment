"""Validated FP32 Vulkan offload for large Linear layers; no quantization."""
import ctypes as C
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


class VulkanEngine:
    def __init__(self):
        self.lib = C.CDLL(str(HERE / "libvulkan_linear.so"))
        ptr = C.POINTER(C.c_float)
        self.ptr = ptr
        self.lib.vl_create.argtypes = [C.c_char_p]
        self.lib.vl_create.restype = C.c_void_p
        self.lib.vl_add.argtypes = [C.c_void_p, ptr, ptr, C.c_uint, C.c_uint]
        self.lib.vl_add.restype = C.c_int
        self.lib.vl_run.argtypes = [C.c_void_p, C.c_int, ptr, ptr, C.c_uint]
        self.lib.vl_run.restype = C.c_int
        self.lib.vl_destroy.argtypes = [C.c_void_p]
        self.handle = self.lib.vl_create(str(HERE / "vulkan_linear.spv").encode())
        if not self.handle:
            raise RuntimeError("Vulkan engine initialization failed")

    def add(self, linear):
        w = linear.weight.detach().cpu().contiguous().numpy()
        b = linear.bias.detach().cpu().contiguous().numpy() if linear.bias is not None else None
        index = self.lib.vl_add(self.handle, w.ctypes.data_as(self.ptr),
                                b.ctypes.data_as(self.ptr) if b is not None else None,
                                w.shape[0], w.shape[1])
        if index < 0:
            raise RuntimeError("Vulkan layer upload failed")
        return index

    def run(self, index, x, out_features):
        array = x.detach().contiguous().numpy()
        output = np.empty((*array.shape[:-1], out_features), dtype=np.float32)
        rows = int(array.size // array.shape[-1])
        status = self.lib.vl_run(self.handle, index, array.ctypes.data_as(self.ptr),
                                 output.ctypes.data_as(self.ptr), rows)
        if status:
            raise RuntimeError("Vulkan linear dispatch failed")
        return torch.from_numpy(output)

    def close(self):
        if self.handle:
            self.lib.vl_destroy(self.handle)
            self.handle = None


class VulkanLinear(torch.nn.Module):
    def __init__(self, linear, engine):
        super().__init__()
        self.engine = engine
        self.index = engine.add(linear)
        self.out_features = linear.out_features

    def forward(self, x):
        return self.engine.run(self.index, x, self.out_features)


def offload(model):
    engine = VulkanEngine()
    converted = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and name.startswith("transformer."):
            parent, _, child = name.rpartition(".")
            model.get_submodule(parent).__setattr__(child, VulkanLinear(module, engine))
            converted += 1
    print(f"Offloaded {converted} Linear layers to Vulkan FP32", flush=True)
    if not converted:
        engine.close()
        raise RuntimeError("No layers matched")
    return engine
