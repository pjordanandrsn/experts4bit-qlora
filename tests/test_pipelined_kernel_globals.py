# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Regression: the gather kernel's annotations must resolve in module globals.

Triton evaluates a ``@triton.jit`` function's annotations against
``fn.__globals__`` — not the scope the function was defined in. ``tl`` was
imported *inside* ``_gather_kernel()``, so ``BLOCK: tl.constexpr`` raised
``NameError('tl is not defined')`` at first compile, taking down a judge host
mid-load. It reproduced only where a kernel actually compiled, so every
CUDA-gated suite skipped past it.

These tests need triton but **not** CUDA, which is the point: the sibling
suite (``test_pipelined.py``) is skipped wholesale without a GPU, so the
invariant has to be checked somewhere a CPU-only CI run still executes it.
"""
import pytest

pytest.importorskip("triton")

from experts4bit_qlora import pipelined  # noqa: E402


def test_tl_is_bound_at_module_scope():
    """The binding triton will look for has to exist where it looks."""
    assert "tl" in vars(pipelined), "tl must be a module global, not a function local"
    assert pipelined.tl is not None


def test_gather_kernel_annotations_resolve_against_kernel_globals():
    """Replay triton's own resolution step: eval each annotation in the
    kernel's ``__globals__``. This raises the identical NameError the
    compiler did, without needing a GPU to reach compilation."""
    kernel = pipelined._gather_kernel()
    fn = getattr(kernel, "fn", kernel)
    annotations = getattr(fn, "__annotations__", {}) or {}
    assert annotations, "kernel lost its constexpr annotations — test is not exercising anything"
    for param, annotation in annotations.items():
        if isinstance(annotation, str):  # PEP 563 / triton re-parse path
            eval(annotation, fn.__globals__)  # noqa: S307 - our own source
