"""Untimed binding/operand witnesses; every temporary observer is removed in finally."""
from __future__ import annotations

from collections import Counter
import contextlib
import functools

from dq11_common import adapter_slots


class Observer:
    def __init__(self, model, arm):
        self.model, self.arm = model, arm
        self.calls, self.gradients, self.operations = Counter(), {}, []
        self.route = "backward-or-outside-adapter"
        self.restore, self.handles = [], []
        self.removed = False

    def wrap(self, owner, name, label, *, backward=False):
        original = getattr(owner, name)
        namespace = owner.__dict__
        existed, descriptor = name in namespace, namespace.get(name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            old = self.route
            route = label
            if backward:
                # Reading saved_tensors here would consume a checkpoint unpack before
                # the real backward reads it. The pinned kernels retain packed W in
                # a plain custom tuple; identify the layer without unpacking tensors.
                layers = {self.parameter_layers[t.data_ptr()] for t in args[0].custom_saved_tensors
                          if hasattr(t, "data_ptr") and t.data_ptr() in self.parameter_layers}
                if len(layers) != 1:
                    raise ValueError("ambiguous fused backward layer")
                route += ":" + str(next(iter(layers)))
            self.route = route
            self.calls[route] += 1
            try:
                return original(*args, **kwargs)
            finally:
                self.route = old

        setattr(owner, name, staticmethod(wrapper) if backward else wrapper)
        self.restore.append((owner, name, existed, descriptor))

    @contextlib.contextmanager
    def active(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        observer = self

        class Operands(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                name = str(func)
                tensors = [t for t in args if isinstance(t, torch.Tensor)]
                rank_matmul = ("mm" in name and any(t.ndim == 2 and 16 in t.shape for t in tensors))
                delta_math = (any(op in name for op in ("add.", "mul.", "_to_copy."))
                              and observer.route != "backward-or-outside-adapter")
                if rank_matmul or delta_math:
                    if len(observer.operations) >= 10000:
                        raise ValueError("operand witness exceeded bounded event count")
                    observer.operations.append({"route": observer.route, "operator": name,
                                                "operands": [{"dtype": str(t.dtype), "shape": list(t.shape)} for t in tensors],
                                                "alpha": (kwargs or {}).get("alpha"), "beta": (kwargs or {}).get("beta"),
                                                "result_dtype": str(result.dtype) if isinstance(result, torch.Tensor) else "unknown"})
                return result

        try:
            slots, modules = adapter_slots(self.model)
            self.parameter_layers = {p.data_ptr(): int(key.split(".")[1]) for key, p in slots.items()}
            self.parameter_layers.update({module.get_base_layer().weight.data_ptr(): int(key.split(".")[1])
                                          for key, module in modules.items()})
            for key, parameter in slots.items():
                def gradient(value, key=key):
                    self.gradients[key] = {"dtype": str(value.dtype), "shape": list(value.shape)}
                self.handles.append(parameter.register_hook(gradient))
            if self.arm == "U":
                from unsloth.kernels import fast_lora

                for name, module in self.model.named_modules():
                    if name.endswith(".self_attn"):
                        layer = name.split("layers.", 1)[1].split(".")[0]
                        self.wrap(module, "apply_qkv", "qkv:" + layer)
                        self.wrap(module, "apply_o", "o:" + layer)
                    elif name.endswith(".mlp"):
                        layer = name.split("layers.", 1)[1].split(".")[0]
                        self.wrap(module, "forward", "mlp:" + layer)
                for name in ("LoRA_QKV", "LoRA_W", "LoRA_MLP"):
                    self.wrap(getattr(fast_lora, name), "backward", name, backward=True)
            else:
                for key, module in modules.items():
                    self.wrap(module, "forward", key)
            with Operands():
                yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            for owner, name, existed, descriptor in reversed(self.restore):
                if existed:
                    setattr(owner, name, descriptor)
                else:
                    delattr(owner, name)
            self.restore.clear()
            self.removed = True

    def receipt(self):
        slots, modules = adapter_slots(self.model)
        if set(self.gradients) != set(slots) or not self.operations or not self.removed:
            raise ValueError("incomplete gradient/operand witness or observer left installed")
        expected = ([f"{prefix}:{layer}" for layer in range(32)
                     for prefix in ("qkv", "o", "mlp", "LoRA_QKV", "LoRA_W", "LoRA_MLP")]
                    if self.arm == "U" else list(modules))
        if any(self.calls[key] < 1 for key in expected):
            raise ValueError("partial executed forward/backward binding witness")
        if not any("mm" in row["operator"] and row["route"] != "backward-or-outside-adapter" for row in self.operations):
            raise ValueError("adapter matmul compute was not observed")
        return {"calls": dict(self.calls), "gradients": self.gradients,
                "operations": self.operations, "observer_removed": self.removed}
