# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""`dispatched_modules` — hook what is CALLED, not what is patched.

`target_modules` returns the frozen `ExpertsNbit` bases, which is correct for
installing an engine's forward and wrong for observing one. A wrapped base is only
called once an engine is attached and `ExpertsLoRA._delegate_to_base` hands the
forward down, so a pre-hook registered on it before that never fires.

That matters because the reason to hook these modules is almost always to build a
routing histogram for an INFORMED hot set, and the calibration pass runs before the
engine exists by construction. The failure is silent and self-consistent: zero counts
-> `topk` of zeros returns `0..K-1` -> "informed" becomes the by-index set -> the A/B
reports the two as indistinguishable, which is exactly what a real null looks like.
Observed for real on granite 2026-08-04 before this helper existed.

No CUDA and no quantization here: the modules are built on `meta`, so this runs in CI
where the residency suites skip. The contract is module-tree bookkeeping, not kernels.
"""
import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora import Experts4bit, ExpertsLoRA  # noqa: E402
from experts4bit_qlora.engines.hot_residency import (  # noqa: E402
    dispatched_modules,
    target_modules,
    wrapped_bases,
)

E, H, INTER = 4, 128, 64   # hidden/intermediate must be multiples of the 64 blocksize


def _bare():
    """An unallocated expert stack — `meta` keeps this CPU- and CI-safe."""
    return Experts4bit(num_experts=E, hidden_dim=H, intermediate_dim=INTER,
                       has_gate=True, activation=torch.nn.functional.silu,
                       quant_type="nf4", compute_dtype=torch.bfloat16, device="meta")


def _wrapped():
    return ExpertsLoRA(_bare(), r=4, alpha=8, dtype=torch.bfloat16)


class _Net(torch.nn.Module):
    """Interleaved on purpose: bare, wrapped, bare, wrapped."""

    def __init__(self):
        super().__init__()
        self.a = _bare()
        self.b = _wrapped()
        self.c = _bare()
        self.d = _wrapped()


def test_returns_wrapper_where_one_exists_and_bare_module_otherwise():
    net = _Net()
    got = dispatched_modules(net)
    assert got == [net.a, net.b, net.c, net.d], "wrong module or wrong order"
    # the distinction is the whole point: two of these are NOT what target_modules gave
    tgt = target_modules(net)
    assert tgt == [net.a, net.b.base, net.c, net.d.base]
    differing = [i for i, (h, t) in enumerate(zip(got, tgt)) if h is not t]
    assert differing == [1, 3], differing


def test_index_alignment_with_target_modules_is_exact():
    """`hot_sets[i]` has to mean the same layer in both lists, or a calibration
    histogram gets applied to the wrong layer — which is worse than no histogram,
    because an informed hot set on the wrong layer is a uniform random draw."""
    net = _Net()
    tgt, disp = target_modules(net), dispatched_modules(net)
    assert len(tgt) == len(disp)
    bases = wrapped_bases(net)
    for t, d in zip(tgt, disp):
        # each pair is either the same bare module, or (base, its own wrapper)
        assert d is t or (id(t) in bases and getattr(d, "base", None) is t)


def test_hooks_on_target_modules_are_the_silent_trap():
    """The regression this helper exists for, asserted as behaviour rather than
    described in a comment: with no engine attached, a pre-hook on a wrapped BASE
    never fires, while the same hook on the dispatched module does."""
    net = _Net()
    fired = {"base": 0, "dispatched": 0}
    wrapper = net.b
    base = net.b.base
    h1 = base.register_forward_pre_hook(lambda *a, **k: fired.__setitem__("base", fired["base"] + 1))
    h2 = wrapper.register_forward_pre_hook(
        lambda *a, **k: fired.__setitem__("dispatched", fired["dispatched"] + 1))
    try:
        # Call the module the model would call. `meta` storage means no real math
        # happens, but the hook fires before dispatch, which is what is under test.
        x = torch.zeros(1, H, dtype=torch.bfloat16, device="meta")
        idx = torch.zeros(1, 2, dtype=torch.long, device="meta")
        w = torch.zeros(1, 2, dtype=torch.bfloat16, device="meta")
        try:
            wrapper(x, idx, w)
        except Exception:
            pass  # the forward may not complete on meta; the pre-hook already ran
    finally:
        h1.remove()
        h2.remove()
    assert fired["dispatched"] == 1, "hook on the dispatched module did not fire"
    assert fired["base"] == 0, (
        "a pre-hook on the wrapped BASE fired without an engine attached — if this "
        "ever becomes true the trap is gone, but so is this helper's rationale")


def test_bare_only_model_is_unchanged():
    """With no wrappers the two lists are identical, so adopting `dispatched_modules`
    is safe for callers that never touch the loader path."""
    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _bare()
            self.b = _bare()

    net = Bare()
    assert dispatched_modules(net) == target_modules(net) == [net.a, net.b]


def test_exported_from_the_package_root():
    """It pairs with `hot_sets_from_profile`, which is a root export — someone
    building an informed hot set should find this without knowing the module."""
    import experts4bit_qlora as e4b
    assert e4b.dispatched_modules is dispatched_modules
    assert "dispatched_modules" in e4b.__all__
