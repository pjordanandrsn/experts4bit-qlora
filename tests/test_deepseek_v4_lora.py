"""QLoRA over DeepSeek-V4 experts: the adapter must compute the BASE's epilogue.

`ExpertsLoRA` never calls `base.forward` -- it re-implements the expert math so the
low-rank delta lands before the nonlinearity -- so it also owns the choice of
nonlinearity. That choice was hardcoded to `act_fn(gate) * up`, which is wrong for every
architecture whose experts clamp. The failure is silent: the model trains, the loss
falls, and it is optimising a function the frozen base does not compute.

The decisive check is `test_zero_adapter_reproduces_the_bare_base`: at init `B = 0`, so
the delta is *identically* zero and a faithful adapter must reproduce the bare module to
rounding. It is not BIT-identical, and that is expected rather than a defect -- V4's own
forward applies the router weight before `w2` (following its reference) while the adapter
applies it after, which is algebraically equal with no bias but reorders two fp32
operations. The tolerance is pinned against the wrong-epilogue case so it cannot silently
widen into vacuity.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from experts4bit_qlora.deepseek_v4 import (  # noqa: E402
    DEFAULT_SWIGLU_LIMIT,
    _DeepseekV4ForwardMixin,
)
from experts4bit_qlora.lora import ExpertsLoRA, _epilogue  # noqa: E402

E, H, INTER, K, TOKENS = 4, 128, 64, 2, 6


def _stacks(seed=0, scale=1.2):
    g = torch.Generator().manual_seed(seed)
    bf = lambda t: t.bfloat16().float()
    return (bf(torch.randn(E, 2 * INTER, H, generator=g) * scale),
            bf(torch.randn(E, H, INTER, generator=g) * scale))


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(TOKENS, H, generator=g).bfloat16().float()
    logits = torch.randn(TOKENS, E, generator=g)
    val, idx = torch.topk(logits, K, dim=-1)
    return x, idx, torch.softmax(val, dim=-1)


def _base(limit=DEFAULT_SWIGLU_LIMIT):
    gu, dn = _stacks()
    return _DeepseekV4ForwardMixin.from_deepseek_v4(
        gu, dn, limit=limit, quant_type="bf16", compute_dtype=torch.float32)


def test_base_exposes_the_epilogue_hook():
    assert callable(getattr(_base(), "_apply_gate", None))


def test_epilogue_helper_prefers_the_hook_over_plain_swiglu():
    base = _base(limit=0.5)
    proj = torch.randn(3, 2 * INTER) * 8            # large enough that clamping bites
    gate, up = proj.chunk(2, dim=-1)
    plain = F.silu(gate) * up
    got = _epilogue(base, proj)
    assert got.shape == plain.shape
    assert not torch.allclose(got, plain), "hook was ignored; the clamp did not apply"
    assert got.dtype == proj.dtype, "hook result must come back in the adapter's dtype"


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def test_zero_adapter_reproduces_the_bare_base():
    """B is zero-initialised, so the delta is identically zero -- not merely small.
    What remains is fp32 reordering from the router-weight placement, ~1e-7 relative."""
    base = _base()
    x, idx, w = _inputs()
    bare = base(x, idx, w)
    wrapped = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32)
    wrapped.eval()
    with torch.no_grad():
        got = wrapped(x, idx, w)
    assert _rel(got, bare) < 1e-5, _rel(got, bare)


def test_a_plain_swiglu_adapter_would_have_differed():
    """Proves the previous test is load-bearing rather than vacuously true: strip the
    hook and the same zero adapter no longer reproduces the base."""
    base = _base(limit=0.5)
    x, idx, w = _inputs(seed=3)
    bare = base(x, idx, w)
    wrapped = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32)
    wrapped.eval()
    # `_apply_gate` is inherited from the mixin, so it cannot be deleted from the concrete
    # class; shadow it on the INSTANCE, which is what `_epilogue`'s getattr resolves.
    base._apply_gate = None
    try:
        with torch.no_grad():
            without = wrapped(x, idx, w)
    finally:
        del base._apply_gate

    with torch.no_grad():
        withhook = wrapped(x, idx, w)
    faithful, broken = _rel(withhook, bare), _rel(without, bare)
    # the tolerance in the test above is only meaningful if the wrong epilogue lands far
    # outside it -- assert the separation rather than trusting a hand-picked bound
    assert faithful < 1e-5, faithful
    assert broken > 1e-2, broken
    assert broken / max(faithful, 1e-12) > 1000


def test_adapter_parameters_are_trainable_and_the_base_is_not():
    base = _base()
    wrapped = ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32)
    trainable = {n for n, p in wrapped.named_parameters() if p.requires_grad}
    assert trainable, "no trainable adapter parameters"
    assert all("lora" in n for n in trainable), sorted(trainable)


def test_gradients_flow_to_the_adapter_through_the_clamped_glu():
    base = _base()
    x, idx, w = _inputs(seed=5)
    wrapped = ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32)
    wrapped.train()
    # break the zero-init so the delta path is actually exercised
    with torch.no_grad():
        for n, p in wrapped.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.02)
    out = wrapped(x, idx, w)
    out.float().pow(2).mean().backward()
    grads = {n: p.grad for n, p in wrapped.named_parameters() if p.requires_grad}
    assert grads, "no adapter parameters"
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in grads.values()), "no finite non-zero gradient reached the adapter"
