# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""[fast] reaches the kernel through an ExpertsLoRA wrapper — and only when it may.

Regression for a silent no-op: ``enable_fast`` patches ``ExpertsNbit.forward``,
but ``ExpertsLoRA.forward`` re-implements the expert math inline (it injects the
low-rank delta *before* the activation) and never calls ``self.base(...)``. The
patch therefore landed on a method nothing invoked — ``enable_fast`` returned a
non-zero count and the fused kernel ran zero times.

``ExpertsLoRA`` now delegates to the base when the adapter provably contributes
nothing (``B`` is zero-initialised, so an untrained adapter is *identically*
zero). The two safety properties matter as much as the speedup:

  * a NON-zero adapter must never be delegated away (that would silently drop a
    trained adapter and return base-model outputs),
  * and ``enable_fast`` must say so rather than report a count implying work.
"""
import warnings

import pytest
import torch

nf4_grouped = pytest.importorskip("nf4_grouped")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    ExpertsLoRA,
    disable_fast,
    enable_fast,
)


def _wrapped(E=4, H=128, inter=64, r=4, seed=0):
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * inter, H, dtype=torch.float32)
    down = torch.randn(E, H, inter, dtype=torch.float32)
    base = Experts4bit.from_float(
        gate_up, down, has_gate=True, activation=torch.nn.functional.silu,
        quant_type="nf4", compute_dtype=torch.bfloat16,
    ).cuda()
    return ExpertsLoRA(base, r=r, alpha=2 * r, dtype=torch.float32).cuda().eval()


def _route(mod, tokens=8, k=2, seed=1):
    torch.manual_seed(seed)
    H = mod.base._gate_up_shape[1]
    x = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda")
    idx = torch.stack([torch.randperm(mod.base.num_experts, device="cuda")[:k]
                       for _ in range(tokens)])
    w = torch.rand(tokens, k, dtype=torch.bfloat16, device="cuda")
    return x, idx, w


def _count_fused():
    """Wrap the kernel entry point so invocations are counted, not inferred."""
    from experts4bit_qlora import fast as F

    calls = {"n": 0}
    orig = F.fused_experts_forward

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    F.fused_experts_forward = counting
    return calls, (lambda: setattr(F, "fused_experts_forward", orig))


def test_zero_adapter_delegates_and_kernel_actually_runs():
    mod = _wrapped()
    x, idx, w = _route(mod)
    with torch.no_grad():
        ref = mod(x, idx, w).float().cpu()

    calls, restore = _count_fused()
    try:
        n = enable_fast(mod.base)
        assert n == 1, f"expected the base to be patched, got {n}"
        with torch.no_grad():
            got = mod(x, idx, w).float().cpu()
        # The whole point: it must actually execute.
        assert calls["n"] > 0, "fused kernel never invoked through the ExpertsLoRA wrapper"
        # Bit-identical would mean we are still on the reference path: the fused
        # path accumulates in fp32, so it must differ slightly.
        assert not torch.equal(ref, got), "identical output implies the kernel did not run"
        rel = (ref - got).abs().max() / got.abs().max().clamp_min(1e-3)
        assert rel < 5e-2, f"fused output diverged from reference: {rel}"
    finally:
        restore()
        disable_fast(mod.base)


def test_trained_adapter_is_never_delegated_away():
    mod = _wrapped()
    with torch.no_grad():
        mod.gate_up_lora_B.normal_(std=0.02)   # simulate a trained adapter
    mod._delegate_ok = None                    # as train()/load_state_dict would
    # Assert on the DATA question directly: _delegate_to_base() also returns False
    # whenever grad is enabled, so under pytest it would pass for the wrong reason.
    assert mod._adapter_is_zero() is False
    with torch.no_grad():
        assert mod._delegate_to_base() is False

    x, idx, w = _route(mod)
    with torch.no_grad():
        before = mod(x, idx, w).float().cpu()
    enable_fast(mod.base)
    try:
        with torch.no_grad():
            after = mod(x, idx, w).float().cpu()
        # Output must be unchanged: the LoRA path still owns this forward.
        assert torch.equal(before, after), "patching changed a trained-adapter forward"
    finally:
        disable_fast(mod.base)


def test_enable_fast_warns_when_the_patch_is_unreachable():
    mod = _wrapped()
    with torch.no_grad():
        mod.gate_up_lora_B.normal_(std=0.02)
    mod._delegate_ok = None
    assert mod._adapter_is_zero() is False
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        enable_fast(mod)                        # walk the wrapper, find the base
    disable_fast(mod.base)
    assert any("never run" in str(r.message) for r in rec), \
        "enable_fast reported a patch count without warning it is unreachable"


def test_delegation_verdict_is_invalidated_by_loading_an_adapter():
    """A trained adapter arriving after a delegating forward must not inherit the
    cached 'adapter is zero' verdict."""
    mod = _wrapped()
    enable_fast(mod.base)
    try:
        with torch.no_grad():
            assert mod._delegate_to_base() is True      # caches the verdict
        # Not every state_dict value is a tensor: bitsandbytes serializes a
        # Params4bit's quant_state as a nested dict, so a blanket .clone()
        # raises AttributeError on a REAL 4-bit module. This only shows up on
        # CUDA — the CPU CI runner never reaches the quantized path, which is
        # why this test merged having never executed.
        sd = {k: (v.clone() if hasattr(v, "clone") else v)
              for k, v in mod.state_dict().items()}
        sd["gate_up_lora_B"] = torch.randn_like(sd["gate_up_lora_B"]) * 0.02
        mod.load_state_dict(sd)
        assert mod._adapter_is_zero() is False, "stale verdict would drop the adapter"
    finally:
        disable_fast(mod.base)
