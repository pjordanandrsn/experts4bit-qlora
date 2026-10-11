"""Temporary reporting scopes around unchanged L callables; frameworks loaded lazily."""

from __future__ import annotations

import functools
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from dq11_l_census_reduce import MAX_TRACE_BYTES, PREFIX, STEP_PREFIX


def adapter_class(key):
    if ".mlp." in key:
        return "MLP-adapter"
    return "o-adapter" if key.endswith(".o_proj") else "qkv-adapter"


class Capture:
    def __init__(self, model, slots, modules, trace, *, cuda=True):
        self.model, self.slots, self.modules = model, slots, modules
        self.trace, self.cuda = Path(trace), cuda
        self.active, self.step, self.profiler = False, None, None
        self.handles, self.restores, self.nodes, self.open_scopes = [], [], [], []
        self.gradients, self.calls = defaultdict(dict), defaultdict(lambda: defaultdict(int))
        self.stacks = defaultdict(list)
        self.removed = False

    def enter(self, cls, path):
        if not self.active:
            return None
        from torch.profiler import record_function

        scope = record_function(PREFIX + cls + "|" + path)
        scope.__enter__()
        self.open_scopes.append(scope)
        return scope

    def leave(self, scope):
        if scope is not None:
            if not self.open_scopes or self.open_scopes[-1] is not scope:
                raise ValueError("CENSUS_REFUSAL: reporting scopes are not nested")
            self.open_scopes.pop()
            scope.__exit__(None, None, None)

    @contextmanager
    def region(self, cls, path):
        scope = self.enter(cls, path)
        try:
            yield
        finally:
            self.leave(scope)

    def module_scope(self, module, cls, path):
        def before(owner, args):
            token = (id(owner), threading.get_ident())
            self.stacks[token].append(self.enter(cls, path))
            if self.active and self.step is not None and path in self.modules:
                self.calls[str(self.step)][path] += 1

        def after(owner, args, output):
            token = (id(owner), threading.get_ident())
            if not self.stacks[token]:
                raise ValueError("CENSUS_REFUSAL: unpaired forward reporting hook")
            self.leave(self.stacks[token].pop())

        self.handles.append(module.register_forward_pre_hook(before))
        # Non-reentrant checkpoint early stop can raise inside down_proj.
        self.handles.append(module.register_forward_hook(after, always_call=True))

    def patch(self, owner, name, cls, path):
        namespace = owner.__dict__
        existed, descriptor = name in namespace, namespace.get(name)
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.region(cls, path):
                return original(*args, **kwargs)

        setattr(owner, name, wrapped)
        self.restores.append((owner, name, existed, descriptor))

    @contextmanager
    def installed(self):
        import torch
        import torch.nn.functional as F

        try:
            for key, module in self.modules.items():
                self.module_scope(module, adapter_class(key), key)
                self.module_scope(module.get_base_layer(), "base NF4 GEMM", key + ".base_layer")
            for key, parameter in self.slots.items():
                node = torch.autograd.graph.get_gradient_edge(parameter).node
                self.nodes.append(node)  # retain the same AccumulateGrad object
                stack = []

                def before_node(outputs, key=key, stack=stack):
                    stack.append(self.enter(adapter_class(key.rsplit(".", 1)[0]), key + ".accumulate"))

                def after_node(inputs, outputs, stack=stack):
                    self.leave(stack.pop())

                self.handles.extend((node.register_prehook(before_node), node.register_hook(after_node)))

                def gradient(value, key=key):
                    if self.active and self.step is not None:
                        if key in self.gradients[str(self.step)]:
                            raise ValueError("CENSUS_REFUSAL: repeated parameter gradient")
                        self.gradients[str(self.step)][key] = str(value.dtype)

                self.handles.append(parameter.register_hook(gradient))
            self.patch(F, "scaled_dot_product_attention", "attention", "native-sdpa")
            self.patch(torch.nn.utils, "clip_grad_norm_", "optimizer", "clip-grad-norm")
            self.patch(torch.optim.AdamW, "step", "optimizer", "AdamW.step")
            yield self
        finally:
            while self.open_scopes:
                self.open_scopes.pop().__exit__(None, None, None)
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            self.nodes.clear()
            for owner, name, existed, descriptor in reversed(self.restores):
                if existed:
                    setattr(owner, name, descriptor)
                else:
                    delattr(owner, name)
            self.restores.clear()
            self.active = False
            self.removed = True

    def begin_profile(self):
        import torch
        from torch.profiler import ProfilerActivity, profile, supported_activities

        if self.trace.exists() or self.profiler is not None:
            raise ValueError("CENSUS_REFUSAL: capture artifact already exists or profiler reused")
        activities = [ProfilerActivity.CPU]
        if self.cuda:
            if not torch.cuda.is_available() or ProfilerActivity.CUDA not in supported_activities():
                raise ValueError("CENSUS_REFUSAL: CUDA/CUPTI profiling unavailable")
            activities.append(ProfilerActivity.CUDA)
        self.profiler = profile(
            activities=activities, record_shapes=True, profile_memory=False, with_stack=False, with_flops=False
        )
        self.profiler.start()
        self.active = True

    def end_profile(self):
        if self.profiler is None:
            raise ValueError("CENSUS_REFUSAL: missing reporting profile")
        self.active = False
        self.profiler.stop()
        self.profiler.export_chrome_trace(str(self.trace))
        if self.trace.stat().st_size > MAX_TRACE_BYTES:
            raise ValueError("CENSUS_REFUSAL: raw trace exceeds512MiB cap")

    @contextmanager
    def trainer_clock(self, train_loop):
        """Keep the actual trainer and its code; intercept only its two step clock reads."""
        original = train_loop.__globals__["time"]
        capture = self

        class Clock:
            count = 0
            scope = None
            running = False

            def __getattr__(self, name):
                return getattr(original, name)

            def time(self):
                from torch.profiler import record_function

                self.count += 1
                step = (self.count + 1) // 2
                if self.count % 2:
                    if step == 6:
                        capture.begin_profile()
                        self.running = True
                    if step in (6, 7, 8):
                        capture.step = step
                        self.scope = record_function(STEP_PREFIX + str(step))
                        self.scope.__enter__()
                    return original.time()
                value = original.time()
                if self.scope is not None:
                    self.scope.__exit__(None, None, None)
                    self.scope = None
                if step == 8:
                    capture.end_profile()
                    self.running = False
                    capture.step = None
                return value

        clock = Clock()
        train_loop.__globals__["time"] = clock
        try:
            yield clock
            if clock.count != 80:
                raise ValueError("CENSUS_REFUSAL: trainer clock no longer brackets40updates")
        finally:
            train_loop.__globals__["time"] = original
            if clock.scope is not None:
                clock.scope.__exit__(None, None, None)
            if clock.running:
                capture.end_profile()
