# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Import transformers' Gated DeltaNet hybrids on their torch reference path, whatever kernels are installed (#1454).

transformers binds ``causal_conv1d`` and ``fla`` into a hybrid's modeling module when the module is first imported
(``use_kernel_func_from_hub_with_fallback``), and the binding lasts as long as the module stays in ``sys.modules``.
Both packages are CUDA-only: once bound, a CPU forward raises ``Expected x.is_cuda() to be true``, in that test and in
every later test of the session that builds the same family. CI installs neither, so only a machine that has them sees
it.

:func:`reference_modeling` imports the modeling modules with both packages masked, so the first import binds the torch
functions. A CPU test that builds a hybrid gets its modeling module from here, never from a module-scope import: that
runs at collection, before any test can mask it. A module some other path already imported with a kernel bound is
refused, naming the functions, rather than left to fail inside the forward.
"""
import importlib
import sys

#: the CUDA-only packages transformers binds into a Gated DeltaNet modeling module at its first import
KERNELS = ("causal_conv1d", "fla")

#: the modeling modules ``engines/linear_state.py`` drives; its ``_linear_classes`` imports all three, so they are
#: imported together, under one mask, before it can import one unmasked
FAMILIES = ("qwen3_5_moe", "qwen3_5", "qwen3_next")


def _bound(mod):
    """Names in ``mod`` whose fallback closure resolved to a kernel package at import (empty on the torch path)."""
    out = []
    for name, fn in vars(mod).items():
        code, closure = getattr(fn, "__code__", None), getattr(fn, "__closure__", None)
        cells = dict(zip(code.co_freevars, closure)) if code is not None and closure else {}
        if "is_new_implementation" in cells and cells["is_new_implementation"].cell_contents:
            out.append(name)
    return out


def reference_modeling(family="qwen3_5_moe"):
    """``transformers.models.<family>.modeling_<family>``, with every :data:`FAMILIES` module imported under the mask.
    Raises ImportError when the installed transformers lacks ``family``, and RuntimeError when ``family`` was imported
    earlier with a kernel bound."""
    saved = {k: sys.modules[k] for k in KERNELS if k in sys.modules}
    sys.modules.update(dict.fromkeys(KERNELS))
    try:
        mods = {}
        for f in dict.fromkeys((family,) + FAMILIES):
            try:
                mods[f] = importlib.import_module(f"transformers.models.{f}.modeling_{f}")
            except ImportError:
                if f == family:
                    raise
    finally:
        for k in KERNELS:
            if k in saved:
                sys.modules[k] = saved[k]
            else:
                del sys.modules[k]
    if bound := _bound(mods[family]):
        raise RuntimeError(f"{mods[family].__name__} was imported before the kernels were masked; {', '.join(bound)} "
                           "call CUDA-only kernels. Import it through tests/hybrid_reference.py, inside the test.")
    return mods[family]
