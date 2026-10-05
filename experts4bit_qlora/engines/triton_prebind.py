# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Prebound Triton launches for the training step's hot kernels (ON by default; ``E4B_TRITON_PREBIND=0`` turns them off; read when
the kernel's module is imported). The default follows TC1 amendments 26 and 30 (one RTX 5090 each, triton 3.4): the training step at
0.973x (matched arm) and 0.980x (shipped arm, 60 steps) of the flags off, held-out within 0.0012 (bench/h2h-2026-10-02/tc1/).

A Triton launch, ``kernel[grid](...)``, spends most of its host time before the driver call: it binds the arguments to the
signature, specializes each one (dtype, 16-byte alignment, ``== 1`` and ``% 16`` of integers), formats that specialization into a
string key, looks the compiled kernel up, and builds launch metadata -- 30-90 us a launch under triton 3.4 on an RTX A2000 host,
against a 4-7 us driver call. The fused RMSNorm and rotary kernels launch a few thousand times a Qwen3-30B-A3B training step, for a
handful of distinct keys.

:func:`prebind` wraps a ``@triton.jit`` kernel so that the FIRST launch of each specialization goes through Triton unchanged, and
the compiled kernel Triton returns is remembered under a key built from the same facts Triton specializes on (each tensor's dtype
and pointer alignment, each integer's VALUE, every other argument's type and value, the launch options, the current device and
Triton's debug knobs). Later launches with that key call the kernel's own launcher directly. Same compiled binary, same arguments,
same stream: outputs are bit-identical (``tests/test_triton_prebind.py``).

Anything the shortcut does not cover goes through Triton's normal launch: a Triton version other than the ones this was written
against (:data:`SUPPORTED_TRITON`), launch hooks (profilers), pre-run hooks, a callable grid, a changed global the kernel reads,
an argument of another type, or a malformed call; under triton 3.7 also a registered compiler-stages hook
(``knobs.runtime.add_stages_inspection_hook``), which 3.7 adds to its kernel key. One knob is read less often than Triton reads it:
triton 3.4 re-reads ``TRITON_DEBUG`` from the environment at every launch, the prebound path once per kernel (triton 3.6 and 3.7
themselves read it once, at import). With the flag off, :func:`prebind` returns the kernel itself."""
from __future__ import annotations

import functools
import os

import torch

try:
    import triton
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import JITFunction
except ImportError:                                   # pragma: no cover - Triton is Linux-only
    triton = knobs = driver = JITFunction = None
_COMPILATION = getattr(knobs, "compilation", None)   # 3.6 and 3.7 add an instrumentation mode to every launch's options

__all__ = ["prebind", "prebind_requested", "SUPPORTED_TRITON", "PREBIND_STATS"]

#: Triton releases whose launch protocol (``JITFunction.run`` -> ``CompiledKernel.run``) this module was read against.
SUPPORTED_TRITON = ((3, 4), (3, 6), (3, 7))

#: Launches through the prebound path, and through Triton's own (first launch of a key, or a fallback).
PREBIND_STATS = {"prebound": 0, "triton": 0}

_MISSING = object()
_MAX_KEYS = 4096                                      # integer values are keyed exactly; a sequence-length sweep must not grow it forever
_PLAIN = (int, float, bool, type(None))


def prebind_requested() -> bool:
    return os.environ.get("E4B_TRITON_PREBIND", "1").strip() != "0"


def _triton_version():
    try:
        return tuple(int(v) for v in triton.__version__.split(".")[:2])
    except Exception:
        return None


def prebind(fn, force: bool = False):
    """``fn`` wrapped in :class:`Prebound` unless ``E4B_TRITON_PREBIND=0`` (``force`` wraps regardless), when this Triton is supported; else ``fn``."""
    if not (force or prebind_requested()) or triton is None or _triton_version() not in SUPPORTED_TRITON:
        return fn
    if not isinstance(fn, JITFunction) or not all(hasattr(fn, a) for a in ("params", "used_global_vals", "pre_run_hooks")):
        return fn                                     # e.g. TRITON_INTERPRET=1's InterpretedFunction
    return Prebound(fn)


class Prebound:
    """``kernel[grid](*args, **kwargs)`` with Triton's per-launch binding done once per specialization."""

    def __init__(self, fn):
        self.fn = fn
        self.names = tuple(p.name for p in fn.params)
        self.param_set = frozenset(self.names)
        self.defaults = {p.name: p.default for p in fn.params if p.has_default}
        self.kernels = {}
        self.device = self.stream = None              # Triton's own device / stream getters, bound at the first launch
        # Triton 3.4 re-reads TRITON_DEBUG from the environment at every launch (1.4 us on an RTX A2000 host), 3.6 and 3.7 once at
        # import: here it is read once per kernel under 3.4, and per launch (a plain attribute) under 3.6 and 3.7.
        self.debug = knobs.runtime.debug if _triton_version() < (3, 6) else None
        # Triton 3.7 keys a launch on a registered compiler-stages hook (its custom pass pipeline), and the launch that compiles a key
        # under AsyncCompileMode returns a FutureKernel proxy rather than the CompiledKernel (3.6 resolves it first).
        self.v37 = _triton_version() >= (3, 7)

    def __getitem__(self, grid):
        if callable(grid):
            return self.fn[grid]
        return functools.partial(self.launch, grid)

    def _triton(self, grid, args, kwargs):
        PREBIND_STATS["triton"] += 1
        return self.fn[grid](*args, **kwargs)

    def launch(self, grid, *args, **kwargs):
        fn, rt = self.fn, knobs.runtime
        # A registered launch hook (3.4: a callable; 3.6, 3.7: a non-empty HookChain) needs Triton's launch metadata, and under 3.7 a
        # registered stages hook adds its pipeline's hash to Triton's key: take Triton's path.
        if getattr(rt.launch_enter_hook, "calls", rt.launch_enter_hook) or getattr(rt.launch_exit_hook, "calls", rt.launch_exit_hook) \
                or fn.pre_run_hooks or (self.v37 and rt.add_stages_inspection_hook is not None):
            return self._triton(grid, args, kwargs)
        try:
            # every parameter in signature order (the launcher's argument list), constexprs and defaults included
            full = args + tuple([kwargs[n] if n in kwargs else self.defaults[n] for n in self.names[len(args):]])
            if self.device is None:                   # torch's, on CUDA: bound once rather than through the driver proxy per launch
                self.device, self.stream = driver.active.get_current_device, driver.active.get_current_stream
            dev = self.device()
            # Triton's key, made exact: a tensor's dtype and 16-byte alignment, an integer's value (which fixes its == 1, % 16 and
            # width), any other argument's type and value; plus the launch options, the device and the debug / instrumentation knobs.
            key = (dev, rt.debug if self.debug is None else self.debug, getattr(_COMPILATION, "instrumentation_mode", None),
                   tuple(kwargs.items()),
                   tuple([(a.dtype, a.data_ptr() % 16 == 0) if isinstance(a, torch.Tensor) else a if type(a) is int else (type(a), a)
                          for a in full]))
            hit = self.kernels.get(key)
        except (KeyError, TypeError):                 # a missing argument, an unhashable one: Triton reports it
            return self._triton(grid, args, kwargs)
        if hit is None:
            kernel = self._triton(grid, args, kwargs)
            # Kept only for plain arguments (a tensor passed by keyword would sit in the key, by identity) and a launchable kernel: under
            # 3.7 not a FutureKernel (a later launch of the key, once Triton's cache holds the CompiledKernel, keeps that).
            if (len(full) == len(self.names) and all(isinstance(a, (torch.Tensor,) + _PLAIN) for a in args)
                    and all(type(v) in _PLAIN for v in kwargs.values()) and not (self.v37 and hasattr(kernel, "result"))
                    and all(hasattr(kernel, a) for a in ("run", "function", "packed_metadata"))):
                if len(self.kernels) >= _MAX_KEYS:
                    self.kernels.clear()
                self.kernels[key] = (kernel, kernel.run, kernel.function, kernel.packed_metadata)
            return kernel
        for (name, _), (val, scope) in fn.used_global_vals.items():
            cur = scope.get(name, _MISSING)
            if cur is not val and cur != val:         # Triton refuses a launch after a global it compiled against changed
                return self._triton(grid, args, kwargs)
        kernel, run, function, metadata = hit
        n = len(grid)
        PREBIND_STATS["prebound"] += 1
        # launch metadata and the enter / exit hooks are None: exactly what Triton passes when no hook is registered
        run(grid[0], grid[1] if n > 1 else 1, grid[2] if n > 2 else 1, self.stream(dev), function, metadata, None, None, None, *full)
        return kernel
