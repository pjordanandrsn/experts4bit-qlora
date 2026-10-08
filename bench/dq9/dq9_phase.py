"""DQ9 telemetry: metadata/counters only, no new CUDA sync or tensor copies in readings."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import functools


def inventory(model):
    """Unique tensor storage bytes, including offloaded homes and bnb quant-state tensors; no tensor reads."""
    seen, rows = set(), []

    def add(name, tensor):
        if tensor is None or not hasattr(tensor, "untyped_storage"):
            return
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if not storage.nbytes() or key in seen:
            return
        seen.add(key)
        rows.append({"name": name, "device": str(tensor.device), "dtype": str(tensor.dtype),
                     "shape": list(tensor.shape), "storage_bytes": storage.nbytes()})

    for name, p in model.named_parameters():
        add("parameter/"+name, p)
    for name, b in model.named_buffers():
        add("buffer/"+name, b)
    for name, module in model.named_modules():
        state = getattr(getattr(module, "weight", None), "quant_state", None)
        for level in ("", "state2."):
            obj = getattr(state, "state2", None) if level else state
            for attr in ("absmax", "code", "offset"):
                add("quant/"+name+"/"+level+attr, getattr(obj, attr, None))
        handle = getattr(module, "_dense_offload", None)
        for i, (_module, _attr, _parameter, home) in enumerate(getattr(handle, "slots", ())):
            add(f"home/{name}/{i}", home)
    return rows


def state_inventory(optimizer):
    """Tensor metadata for optimizer states, retaining no state tensors or storage references."""
    seen, rows = set(), []
    for i, state in enumerate(optimizer.state.values()):
        for name, tensor in state.items():
            if not hasattr(tensor, "untyped_storage"):
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if not storage.nbytes() or key in seen:
                continue
            seen.add(key)
            rows.append({"name": f"optimizer/{i}/{name}", "device": str(tensor.device),
                         "dtype": str(tensor.dtype), "shape": list(tensor.shape), "storage_bytes": storage.nbytes()})
    return rows


class PhaseCensus:
    def __init__(self, torch, cache_mode):
        if cache_mode not in ("baseline", "empty"):
            raise ValueError("unregistered cache mode")
        self.torch, self.cache_mode = torch, cache_mode
        self.rows, self.storages, self.step, self.cache_calls = [], [], 0, 0
        self.optimizer_storages = []

    def sample(self, phase):
        # All four APIs read allocator counters on the host. No synchronize, peak reset, or GPU tensor creation.
        cuda = self.torch.cuda
        allocated, reserved = cuda.memory_allocated(), cuda.memory_reserved()
        row = {"phase": phase, "step": self.step, "allocated": allocated, "reserved": reserved,
               "cached": reserved-allocated, "peak_allocated": cuda.max_memory_allocated(),
               "peak_reserved": cuda.max_memory_reserved()}
        self.rows.append(row)
        return row

    @contextmanager
    def installed(self, loader, trainer):
        """Wrap the actual executor's functions, restoring everything after success or failure."""
        torch = self.torch
        with ExitStack() as stack:
            def patch(obj, attr, wrapper):
                original = getattr(obj, attr)
                owned = attr in vars(obj)
                setattr(obj, attr, wrapper(original))
                if owned:
                    stack.callback(setattr, obj, attr, original)
                else:
                    stack.callback(delattr, obj, attr)

            def observe(before, after):
                def decorate(original):
                    @functools.wraps(original)
                    def wrapped(*args, **kwargs):
                        self.sample(before)
                        result = original(*args, **kwargs)
                        self.sample(after)
                        return result
                    return wrapped
                return decorate

            patch(loader, "load_base", observe("loader-start", "loader-done"))
            patch(torch.optim.AdamW, "__init__", observe("optimizer-create-start", "optimizer-created"))
            def optimizer_step(original):
                @functools.wraps(original)
                def wrapped(optimizer, *args, **kwargs):
                    self.sample("optimizer-step-start")
                    result = original(optimizer, *args, **kwargs)
                    self.sample("optimizer-step-done")
                    self.optimizer_storages.append({"step": self.step, "storages": state_inventory(optimizer)})
                    return result
                return wrapped
            patch(torch.optim.AdamW, "step", optimizer_step)
            patch(torch.autograd, "backward", observe("backward-start", "backward-done"))
            patch(torch.nn.utils, "clip_grad_norm_", observe("clip-start", "clip-done"))

            def prepared(original):
                @functools.wraps(original)
                def wrapped(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self.storages = inventory(result.model)
                    self.sample("setup-done")
                    if self.cache_mode == "empty":
                        self.cache_calls += 1
                        torch.cuda.empty_cache()  # The sole registered intervention, before optimizer creation.
                    self.sample("cache-boundary-done")

                    def forward(original_forward):
                        @functools.wraps(original_forward)
                        def call(*args, **kwargs):
                            self.step += 1
                            self.sample("forward-start")
                            output = original_forward(*args, **kwargs)
                            self.sample("forward-loss-done")
                            return output
                        return call
                    patch(result.model, "forward", forward)
                    return result
                return wrapped
            patch(trainer, "prepare", prepared)
            yield self

    def report(self):
        return {"schema": "dq9-phases/1", "cache_mode": self.cache_mode, "cache_calls": self.cache_calls,
                "new_sync_calls": 0, "rows": self.rows, "setup_storages": self.storages, "optimizer_storages": self.optimizer_storages}
