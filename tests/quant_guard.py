# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Tell a missing bitsandbytes backend apart from a REAL failure of the code under test.

Two entry points need this, and they need different mechanisms:

* :func:`load_or_skip` -- for tests that go through ``load_moe_4bit_streaming``. The loader
  declines checkpoints by *raising*, with messages we can match, so the fix is a carve-out:
  re-raise a refusal, skip only for an absent backend.
* :func:`require_quantize` -- for tests that quantize an in-memory tensor with ``from_float``.
  There is no refusal vocabulary to match there: ``from_float`` validates with ``ValueError``
  (already outside the caught set), so anything else it raises is a genuine bug -- including
  torch's own ``RuntimeError`` on a shape mismatch. Matching messages cannot work; the fix is
  to ask the environment question ONCE, up front, and then let the call run unguarded.

The second was the bigger hole by an order of magnitude. Mutating ``from_float`` to raise on
entry -- a total break of the primitive -- and running the eleven modules that guarded it:

    guard                     failed   passed   skipped
    the old broad except           1        6       106
    require_quantize              82        6        25

Eighty-one arms had been reporting a dead primitive as green. The 25 that still skip are the
CUDA-only arms, which is exactly the baseline skip set: on a working host the quantize guard
fired *zero* times, so it was pure downside -- inert until it had a bug to hide.

Both directions are pinned, because a guard that never skips is as broken as one that always
does. Forcing the probe to report the backend as absent puts the same modules at
``0 failed / 7 passed / 106 skipped``, each naming the scheme it wanted.

``scripts/validate_expertsnbit.py`` carried the same defect and got the same treatment. Under
that mutant it used to print ``pass=0 fail=0 skip=37`` and **exit 0** -- a validation script
returning success against a library that could not quantize at all. It now exits 1.

Every module that loads a checkpoint through ``load_moe_4bit_streaming`` wraps the call in
``except (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError): pytest.skip(...)``.
That breadth is not laziness — bnb signals a dead or absent 4-bit backend several different ways
depending on how it was built, and a host without one has to skip rather than fail.

The problem is that the loader declines checkpoints through those SAME exception types: a
``model_type`` it does not admit, an expert layout it could not map, an untied head it never found.
Those refusals ARE the regressions the loader test modules exist to catch, so catching them as "no
backend" turns a whole module into an instrument that cannot fail — green, and measuring nothing.

Measured by mutating the loader and re-running, not argued from inspection. Against an unmutated
tree both modules are 65 passed / 0 skipped here (CPU, torch 2.11.0 / transformers 5.13.1 /
bnb 0.50.0), so every arm below really runs and a skip really is a signal:

* **fused-on-disk branch never matches** (``if False and ...`` at ``loader.py:789``) — with the
  guard, ``test_reference_parity``'s gemma4 arm fails and ``test_loader_architectures`` goes
  14 failed / 30 passed across the whole file. Without it, that gemma4 arm reported
  ``SKIPPED ... bitsandbytes 4-bit unavailable: no fused expert stacks found`` — the loader's own
  refusal, printed as a missing quantizer, against a model with zero quantized expert layers.
* **both per-expert readers disabled** (``loader.py:830`` and ``:853``) — with the guard, the olmoe
  and qwen3_moe parity arms fail (2 failed / 19 passed); without it, 19 passed / 2 skipped.
  Between the two mutants that is 3 of 3 architecture arms silently disarmed.
* disabling only ``:830`` is **fully masked** by ``:853`` falling through (21 passed, nothing
  observed) — a per-expert mutation has to disable both readers to test anything.

An earlier mutant on the parent branch, which re-merged the checkpoint and module prefixes, turned
every mixtral arm green as a *skip* the same way.

So the distinction lives here, once, instead of being re-derived (or forgotten) per module.

``LOADER_REFUSALS`` is kept COMPLETE over the refusals ``loader.py`` raises with a type
``QUANTIZE_UNAVAILABLE`` catches — not merely over the ones today's call sites happen to reach.
Reachability shifts every time an arm is added, and this list is the one thing that must not need
updating when it does. ``MoEConventionError`` and the loader's plain ``ValueError``\\s are absent on
purpose: they are not caught in the first place, so they already fail loudly.

This module imports nothing heavier than pytest ON PURPOSE. ``conftest.py`` decides at COLLECTION
time which modules would die on a dependency this machine lacks, and it does so by reading each test
module's own top-level imports. A helper that pulled in torch here would be invisible to that scan,
and would break collection on a torch-less host instead of skipping cleanly.
"""

import pytest

#: How bnb signals a missing/broken 4-bit backend. Build-dependent, hence the breadth.
QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)

#: Substrings marking a refusal raised BY ``experts4bit_qlora.loader`` rather than by bnb. Complete
#: over that module's raises of a ``QUANTIZE_UNAVAILABLE`` type (``FileNotFoundError`` included — it
#: is an ``OSError``, so it is swallowed too).
LOADER_REFUSALS = (
    "Unsupported model_type",                 # model_type not admitted
    "identity ('zero-computation') experts",  # router addresses experts the primitive cannot represent
    "CLAMPED SwiGLU",                         # storage convention shared, epilogue not
    "key rewriter mapped every one of",       # rewriter/checkpoint mismatch
    "nothing this loader can stream",         # neither a shard index nor a single-file checkpoint
    "per-expert biases",                      # arena layout carries biases this path would drop
    "no fused expert stacks",                 # zero expert layers mapped
    "tie_word_embeddings=False",              # untied head never found
    "unmaterialized meta tensors remain",     # something never got materialized
    "quantize_layers excludes",               # excluded layer's experts are packed on disk
)


def is_loader_refusal(exc):
    """True when ``exc`` is the loader declining, not the quantizer being absent."""
    return any(m in str(exc) for m in LOADER_REFUSALS)


def load_or_skip(path, device, dtype, *, what="4-bit", **kw):
    """``load_moe_4bit_streaming``, skipping ONLY for a genuinely absent bnb quantize backend.

    ``what`` names the scheme in the skip reason, so an arm that asked for something other than plain
    4-bit says which. A loader refusal is re-raised and fails the arm — that is the whole point.
    """
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    try:
        return load_moe_4bit_streaming(path, device, dtype, **kw)
    except QUANTIZE_UNAVAILABLE as e:
        if is_loader_refusal(e):
            raise
        pytest.skip(f"bitsandbytes {what} quantize unavailable on {device}: {e}")


# ------------------------------------------------------------------------------------------------
# The quantize-capability probe, for call sites that do NOT go through the loader.
# ------------------------------------------------------------------------------------------------

#: Which bnb primitive each storage scheme goes through, and so what "available" means for it.
#: 16-bit entries are passthrough — the package stores the tensor as-is and never calls bnb, so
#: they cannot be unavailable. Mirrors the package's `_SCHEME_BITS`; `test_quant_guard.py` pins
#: the two together, because a scheme added there and missed here would silently probe the wrong
#: backend.
SCHEME_BACKEND = {
    "nf4": "4bit",
    "fp4": "4bit",
    "int8": "blockwise",
    "fp8": "blockwise",
    "bf16": None,
    "fp16": None,
}

#: (device, quant_type, blocksize) -> None if the backend works, else the reason it does not.
#: Cached because the answer is a property of the host, not of the arm asking, and because the
#: probe would otherwise run once per parametrized case.
_PROBE_CACHE = {}


def _probe(device, quant_type, blocksize):
    """Ask BITSANDBYTES, not this package, whether the backend works here.

    Deliberately NOT routed through ``from_float``. A probe that called the code under test
    could be broken by the very bug it is supposed to let us see: if ``from_float`` raised for
    every input, the probe would raise too, every arm would skip, and the guard would be back to
    hiding a dead primitive. Calling the bnb primitive directly makes the probe answer exactly
    one question — *can this host quantize at all* — and leaves every failure of our own code
    where it belongs, in the arm.

    Built with ``arange`` rather than ``randn``: this runs lazily, at whatever point the first
    arm asks, typically *after* that arm's ``torch.manual_seed(0)``. Drawing from the global RNG
    here would shift every subsequent draw and make seeded results depend on probe ordering.
    """
    import bitsandbytes.functional as bnbF
    import torch

    backend = SCHEME_BACKEND.get(quant_type, "4bit")
    if backend is None:
        return None  # passthrough storage: no codebook, no bnb call, nothing to be unavailable
    n = blocksize  # square and blocksize-aligned, so divisibility is never what fails
    w = ((torch.arange(1, n * n + 1, dtype=torch.float32) * 1e-3).reshape(n, n).to(device)).contiguous()
    try:
        if backend == "4bit":
            bnbF.quantize_4bit(w, blocksize=blocksize, compress_statistics=False, quant_type=quant_type)
        else:
            bnbF.quantize_blockwise(w, blocksize=blocksize)
    except QUANTIZE_UNAVAILABLE as e:
        return f"{type(e).__name__}: {e}"
    return None


def quantize_unavailable_reason(device, quant_type="nf4", blocksize=64):
    """None if bnb can quantize `quant_type`/`blocksize` on `device`, else why it cannot."""
    key = (str(device), quant_type, blocksize)
    if key not in _PROBE_CACHE:
        _PROBE_CACHE[key] = _probe(device, quant_type, blocksize)
    return _PROBE_CACHE[key]


def require_quantize(device, quant_type="nf4", blocksize=64):
    """Skip the calling arm unless bnb can really quantize this scheme on this device.

    Call this INSTEAD OF wrapping the real ``from_float`` in ``except QUANTIZE_UNAVAILABLE``.
    The point of the split is that afterwards the real call runs unguarded, so a bug in the
    code under test fails the arm instead of being reported as a missing backend.
    """
    reason = quantize_unavailable_reason(device, quant_type, blocksize)
    if reason is not None:
        pytest.skip(f"bitsandbytes {quant_type}/bs{blocksize} quantize unavailable on {device}: {reason}")
