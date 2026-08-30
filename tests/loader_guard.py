# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Tell a missing bitsandbytes 4-bit backend apart from a LOADER REFUSAL.

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
