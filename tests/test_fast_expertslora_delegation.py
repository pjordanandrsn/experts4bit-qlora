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


def _count_kernel():
    """Count gemm_4bit_grouped, NOT the patched forward.

    Once patched, the wrapper's forward is invoked on every call whether or not it
    fuses — `fused_experts_lora_forward` short-circuits to the reference path under
    `training`. Counting the wrapper therefore reports "it ran" in cases where the
    kernel never executed, which is how this file previously drew the wrong
    conclusion about train mode."""
    import nf4_grouped
    calls = {"n": 0}
    real = nf4_grouped.gemm_4bit_grouped

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    nf4_grouped.gemm_4bit_grouped = counting
    return calls, (lambda: setattr(nf4_grouped, "gemm_4bit_grouped", real))


def test_a_trained_adapter_still_fuses_through_the_wrapper():
    """Regression for a warning that outlived its design.

    `enable_fast` used to patch the BASE, which a trained adapter made unreachable —
    `_delegate_to_base` requires a provably-zero adapter — so it warned "these patches
    will never run". It now patches the WRAPPER, whose fused forward applies the
    low-rank delta on the expert-sorted rows, so a trained adapter fuses like any
    other. The warning was removed; this pins the behaviour that replaced it."""
    mod = _wrapped()
    with torch.no_grad():
        mod.gate_up_lora_B.normal_(std=0.02)
    mod._delegate_ok = None
    assert mod._adapter_is_zero() is False

    x, idx, w = _route(mod)
    with torch.no_grad():
        ref = mod(x, idx, w).float().cpu()

    calls, restore = _count_kernel()
    try:
        assert enable_fast(mod) == 1
        assert hasattr(mod, "_e4b_fast_ref"), "the WRAPPER should be patched"
        assert not hasattr(mod.base, "_e4b_fast_ref"), "the base should be skipped"
        with torch.no_grad():
            got = mod(x, idx, w).float().cpu()
        assert calls["n"] > 0, "fused kernel never ran with a trained adapter"
        rel = (ref - got).abs().max() / got.abs().max().clamp_min(1e-3)
        assert rel < 5e-2, f"fused output diverged from the reference: {rel}"
    finally:
        restore()
        disable_fast(mod)


def test_train_mode_warns_and_the_kernel_really_does_not_run():
    """The surviving warning, asserted on the KERNEL rather than on the wrapper.

    `fused_experts_lora_forward` returns the reference path while `training` is set,
    to preserve the summation order a reentrant-checkpoint recompute reproduces. So
    the fused kernel genuinely does not execute, and the warning is earned."""
    mod = _wrapped().train()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        enable_fast(mod)
    assert any("TRAINING mode" in str(r.message) for r in rec), \
        "no warning that train mode bypasses the fused path"

    calls, restore = _count_kernel()
    try:
        x, idx, w = _route(mod)
        with torch.no_grad():
            mod(x, idx, w)
        assert calls["n"] == 0, \
            f"the warning says the kernel is bypassed in train mode, but it ran ({calls['n']}x)"
    finally:
        restore()
        disable_fast(mod)


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
