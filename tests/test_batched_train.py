"""Behaviour of the kernel-free batched training path.

Numerical parity lives in ``test_fused_train_parity.py``, which puts this path under
the same contract as the fused one (forward, dL/dx, and every LoRA gradient against
the reference ``ExpertsLoRA.forward``). What is asserted HERE is everything around
the fast path: that it declines what it cannot reproduce, that each fallback lands on
the reference rather than on a wrong answer, and that disabling restores.

A fast path is only as trustworthy as its refusals. Every branch below returns the
reference result, so each test asserts equality with the reference — a fallback that
silently computed something else would still "work".
"""

import pytest

from quant_guard import require_quantize

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    ExpertsLoRA,
    disable_batched_train,
    enable_batched_train,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E, HID, INTER, TOP_K = 8, 128, 192, 2


def _build(seed=0, quant_type="nf4"):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    require_quantize(DEVICE, quant_type)
    base = Experts4bit.from_float(gate_up, down, quant_type=quant_type,
                                  compute_dtype=torch.bfloat16)
    mod = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        for p in (mod.gate_up_lora_B, mod.down_lora_B):
            p.normal_(0, 0.02)
    return mod.train()


def _inputs(n_tok=32, seed=1, hot_expert=False):
    torch.manual_seed(seed)
    hs = torch.randn(n_tok, HID, dtype=torch.bfloat16, device=DEVICE)
    if hot_expert:
        # One expert takes nearly everything: max(sizes) ~ n_tok while the other
        # groups hold one row each, which is what blows the padded block up.
        idx = torch.zeros(n_tok, TOP_K, dtype=torch.long, device=DEVICE)
        idx[:, 1] = torch.arange(n_tok, device=DEVICE) % E
        idx[0, 1] = 1
    else:
        idx = torch.randint(0, E, (n_tok, TOP_K), device=DEVICE)
    wts = torch.rand(n_tok, TOP_K, dtype=torch.bfloat16, device=DEVICE)
    return hs, idx, wts


def _both(mod, args):
    """(reference, batched) for the same module and inputs."""
    ref = mod(*args)
    assert enable_batched_train(mod) == 1
    got = mod(*args)
    return ref, got


def test_patches_and_restores():
    mod = _build()
    before = mod.forward
    assert enable_batched_train(mod) == 1
    assert enable_batched_train(mod) == 0, "second call must not double-patch"
    assert disable_batched_train(mod) == 1
    assert mod.forward == before
    assert disable_batched_train(mod) == 0


def test_matches_the_reference_forward():
    """Parity is contracted elsewhere; this is the smoke check that the patch is live
    and computing the same function on ordinary routing."""
    mod = _build()
    args = _inputs()
    ref, got = _both(mod, args)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 1.5e-2, rel


def test_router_skew_falls_back_to_the_reference():
    """Group sizes come from the router. One hot expert makes the padded block
    G*max(sizes) rows wide however few rows are real, so past the guard the reference
    is used — bad routing must not cost MORE than not having this path at all."""
    from experts4bit_qlora.engines.batched import _PAD_WASTE_LIMIT

    mod = _build()
    args = _inputs(n_tok=64, hot_expert=True)
    flat = args[1].reshape(-1)
    counts = torch.bincount(flat, minlength=E)
    sizes = counts[counts > 0]
    assert len(sizes) * int(sizes.max()) > _PAD_WASTE_LIMIT * int(sizes.sum()), \
        "fixture is not skewed past the guard"
    ref, got = _both(mod, args)
    # Fallback means the REFERENCE result, bit for bit — not a close approximation.
    assert torch.equal(got, ref)


def test_evicted_expert_storage_falls_back():
    """Under offload the packed buffers are 0-element placeholders between forwards.
    Entering the whole-stack dequant with one yields a shaped-but-empty stack rather
    than an error, so the check has to happen before the dequant."""
    mod = _build()
    args = _inputs()
    # `.data` reassignment is how offload.py actually evicts — the module keeps its
    # Parameter/buffer, the storage under it becomes a 0-element placeholder.
    real = mod.base.gate_up_proj.data
    placeholder = torch.empty(0, dtype=real.dtype, device=real.device)

    # What the UNPATCHED module does with an evicted buffer is the specification:
    # whatever it is, the patched module must do the same thing rather than feed the
    # placeholder to the dequant and get a shaped-but-empty stack back.
    mod.base.gate_up_proj.data = placeholder
    with pytest.raises(Exception) as unpatched:
        mod(*args)
    mod.base.gate_up_proj.data = real

    assert enable_batched_train(mod) == 1
    mod.base.gate_up_proj.data = placeholder
    try:
        with pytest.raises(type(unpatched.value)):
            mod(*args)
    finally:
        mod.base.gate_up_proj.data = real

    # And it still works once the layer is staged back in.
    disable_batched_train(mod)
    ref = mod(*args)
    assert enable_batched_train(mod) == 1
    rel = ((mod(*args).float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 1.5e-2, rel


def test_declines_a_base_whose_forward_it_cannot_reproduce():
    """gpt-oss adds per-expert biases that nothing in this path applies. A base with a
    custom forward and no `_apply_gate` hook is skipped, not silently accelerated."""
    mod = _build()

    class _CustomForward(type(mod.base)):
        def forward(self, *a, **kw):            # noqa: D401 - stand-in for gpt-oss
            raise AssertionError("should never be called")

    mod.base.__class__ = _CustomForward
    assert not hasattr(mod.base, "_apply_gate")
    assert enable_batched_train(mod) == 0


def test_declines_non_4bit_storage():
    """The whole-stack dequant is a 4-bit trick — the flattened absmax only tiles
    because each expert is quantized independently at 4 bits."""
    mod = _build()
    mod.base.bits = 8
    assert enable_batched_train(mod) == 0


def test_does_not_stack_on_the_fused_training_patch():
    """Both patch ExpertsLoRA.forward. Wrapping the fused one would make this path's
    fallbacks land on the fused forward rather than the reference."""
    pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm >= 0.2.4")
    if not torch.cuda.is_available():
        pytest.skip("fused training path is CUDA-only")
    from experts4bit_qlora import disable_fast_train, enable_fast_train

    mod = _build()
    if enable_fast_train(mod) != 1:
        pytest.skip("enable_fast_train declined this module")
    assert enable_batched_train(mod) == 0, "batched path wrapped the fused patch"
    assert disable_fast_train(mod) == 1
    assert enable_batched_train(mod) == 1, "should patch once the lane is free"


def test_does_not_stack_in_the_reverse_order_either():
    """The mirror of the above, which the first version of this guard missed.

    batched-then-fused left the module unrecoverable: the fused patch captured the
    BATCHED forward as its reference, `disable_batched_train` restored from underneath
    it, and `disable_fast_train` then reinstated the batched forward — with no
    `_e4b_batched_ref` left to undo it. Both directions must refuse.
    """
    pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm >= 0.2.4")
    if not torch.cuda.is_available():
        pytest.skip("fused training path is CUDA-only")
    from experts4bit_qlora import disable_fast_train, enable_fast_train

    mod = _build()
    reference = mod.forward
    assert enable_batched_train(mod) == 1
    assert enable_fast_train(mod) == 0, "fused path patched over the batched one"
    assert disable_batched_train(mod) == 1
    assert mod.forward == reference, "did not get the true reference forward back"
    assert not hasattr(mod, "_e4b_train_ref")
    assert disable_fast_train(mod) == 0


def test_refuses_to_unpatch_from_under_a_stacked_patch():
    """Belt to the braces above: even if something patches over this path, unwinding
    out of order must not leave the module on a forward nothing can remove."""
    mod = _build()
    reference = mod.forward
    assert enable_batched_train(mod) == 1
    mod._e4b_train_ref = mod.forward          # simulate an outer patch
    mod.forward = lambda *a, **kw: None
    with pytest.warns(RuntimeWarning, match="disable_fast_train first"):
        assert disable_batched_train(mod) == 0
    # Unwind properly and the reference is still recoverable.
    mod.forward = mod._e4b_train_ref
    del mod._e4b_train_ref
    assert disable_batched_train(mod) == 1
    assert mod.forward == reference


def test_empty_batch_matches_the_reference():
    """No token routed anywhere: `sizes.max()` raises on the empty tensor where the
    reference returns its zero accumulator. Route it to the reference instead."""
    mod = _build()
    hs = torch.zeros(0, HID, dtype=torch.bfloat16, device=DEVICE)
    idx = torch.zeros(0, TOP_K, dtype=torch.long, device=DEVICE)
    wts = torch.zeros(0, TOP_K, dtype=torch.bfloat16, device=DEVICE)
    ref = mod(hs, idx, wts)
    assert enable_batched_train(mod) == 1
    got = mod(hs, idx, wts)
    assert got.shape == ref.shape == (0, HID)
    assert torch.equal(got, ref)


def test_batched_path_keeps_the_v4_clamps():
    """DeepSeek-V4's clamped SwiGLU through the BATCHED path.

    test_fast_v4.py pins the clamps for both fused paths; nothing pinned them for
    batched_experts_train_forward. It reaches the epilogue via the same `_epilogue`
    hook, so it SHOULD be right — but "should via shared hook" is how the fused
    training path shipped with no eligibility gate at all, and an unclamped SwiGLU
    here trains a function the frozen base does not compute.
    """
    if not torch.cuda.is_available():
        pytest.skip("V4 fixture is CUDA-only")
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_fast_v4 import LIMIT, TOL, _b_rel, _v4_lora

    mod, x, idx, w = _v4_lora(seed=7, limit=LIMIT)
    unclamped, *_ = _v4_lora(seed=7, limit=0.0)
    mod.train()
    unclamped.train()
    ref = mod(x, idx, w)
    ref_noclamp = unclamped(x, idx, w)
    assert _b_rel(ref, ref_noclamp) > 10 * TOL, "clamp barely binds — fixture invalid"

    assert enable_batched_train(mod) == 1, "batched path declined the V4 module"
    got = mod(x, idx, w)
    assert got.requires_grad, "batched training path must stay differentiable"
    assert _b_rel(got, ref) < TOL, f"diverged from the clamped reference: {_b_rel(got, ref)}"
    assert _b_rel(got, ref_noclamp) > 10 * TOL, "batched path served UNCLAMPED SwiGLU"
    got.float().sum().backward()
    assert mod.gate_up_lora_A.grad is not None
    disable_batched_train(mod)
