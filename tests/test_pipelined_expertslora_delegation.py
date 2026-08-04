# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Pipelined residency reaches the engine through an ExpertsLoRA wrapper.

``enable_pipelined_residency`` used to raise NotImplementedError the moment every
``ExpertsNbit`` under the model was an ``ExpertsLoRA.base`` — which is every model
``load_moe_4bit_streaming`` returns. The engine was therefore unreachable on the
loader path, and the "VRAM-resident vs host-streamed" fidelity row could never run.

The wrapper does delegate: ``ExpertsLoRA._delegate_to_base`` hands the whole forward
to the base when an engine is attached and the adapter provably contributes nothing
(``B`` is zero-initialised, so an untrained adapter is *identically* zero).

What these tests hold down, in order of how badly each failure would mislead:

  * the engine must ACTUALLY EXECUTE — asserted from the engine's own device-side
    fetch counters, not inferred from the output. This is the load-bearing one: a
    residency split that never runs reproduces the unsplit reference exactly, so a
    dead patch scores a perfect zero divergence and reads as a PASS in precisely
    the confirmatory benchmark this unblocks;
  * a NON-zero adapter must never be delegated away (that would silently drop a
    trained adapter and serve base-model outputs), and the engine must stay cold;
  * ``enable_pipelined_residency`` must warn rather than return a count implying
    work it cannot do;
  * the deprecated v0 engine, which ``_delegate_to_base`` does NOT know about, must
    still refuse wrapped bases instead of patching a forward nobody calls.
"""
import warnings

import pytest
import torch

pytest.importorskip("nf4_grouped")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _no_triton_interpreter():
    """The address-gather is compiled-only — raw device/UVA pointers segfault the
    host-side Triton interpreter (same guard as tests/test_pipelined.py)."""
    import os
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("Triton interpreter mode active (raw-pointer gather is compiled-only)")


from experts4bit_qlora import Experts4bit, ExpertsLoRA  # noqa: E402
from experts4bit_qlora.hot_residency import target_modules  # noqa: E402
from experts4bit_qlora.pipelined import (  # noqa: E402
    disable_pipelined_residency,
    enable_pipelined_residency,
)

E, H, INTER, K = 8, 128, 64, 2
HOT = [0, 1, 2, 3]          # half resident, half streamed — a real split
# One hot expert and one cold one, so both traffic counters must move. The hot id is
# deliberately NOT 0: `_PipelinedResidency._prime()` fills every slot with expert 0's
# row at construction, so routing to expert 0 is a legitimate have-skip and moves zero
# bytes — which looks exactly like an engine that never ran.
ROUTE = [2, 5]


def _wrapped(r=4, seed=0):
    torch.manual_seed(seed)
    gate_up = torch.randn(E, 2 * INTER, H, dtype=torch.float32)
    down = torch.randn(E, H, INTER, dtype=torch.float32)
    base = Experts4bit.from_float(
        gate_up.cuda(), down.cuda(), has_gate=True,
        quant_type="nf4", compute_dtype=torch.bfloat16,
    )
    return ExpertsLoRA(base, r=r, alpha=2 * r, dtype=torch.float32).cuda().eval()


def _route(seed=1):
    torch.manual_seed(seed)
    hs = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    ti = torch.tensor([ROUTE], dtype=torch.long, device="cuda")
    tw = torch.tensor([[0.6, 0.4]], dtype=torch.bfloat16, device="cuda")
    return hs, ti, tw


def _enable(mod):
    return enable_pipelined_residency(
        mod, [torch.tensor(HOT, dtype=torch.long)], device="cuda", k_slots=K)


def test_wrapped_base_is_a_residency_target():
    """The enumeration change itself: a wrapped base is targetable and index-bearing.
    While it was excluded, `hot_sets` was length 0 for every loader model and the
    engine refused before it ever looked at a weight."""
    mod = _wrapped()
    assert target_modules(mod) == [mod.base]
    assert _enable(mod) == 1
    assert disable_pipelined_residency(mod) == 1


def test_zero_adapter_delegates_and_the_engine_actually_runs():
    mod = _wrapped()
    hs, ti, tw = _route()
    with torch.no_grad():
        ref = mod(hs, ti, tw).float().cpu()

    assert _enable(mod) == 1
    try:
        st = mod.base._pipelined
        # The engine's own counters, before any forward through the wrapper. `_prime()`
        # populates the slots without touching them, so this is a true zero.
        assert st.traffic() == {"hot_d2d_bytes": 0, "cold_pcie_bytes": 0}
        with torch.no_grad():
            got = mod(hs, ti, tw).float().cpu()

        # The whole point: proof of execution from inside the engine. An output
        # comparison cannot establish this — a patch that never runs returns the
        # reference values, which is exactly what "correct" looks like here.
        moved = st.traffic()
        assert moved["cold_pcie_bytes"] > 0, (
            "no cold-tier traffic: the pipelined engine never ran through the "
            f"ExpertsLoRA wrapper ({moved})")
        assert moved["hot_d2d_bytes"] > 0, (
            f"no hot-stack traffic: the resident tier was never read ({moved})")

        rel = (ref - got).abs().max() / got.abs().max().clamp_min(1e-3)
        assert rel < 1.5e-2, f"residency changed the arithmetic: rel={rel}"
    finally:
        disable_pipelined_residency(mod)


def test_trained_adapter_is_never_delegated_away():
    mod = _wrapped()
    with torch.no_grad():
        mod.gate_up_lora_B.normal_(std=0.02)   # simulate a trained adapter
    mod._delegate_ok = None                    # as train()/load_state_dict would
    # Assert the DATA question directly: _delegate_to_base() also returns False
    # whenever grad is enabled, so it would pass for the wrong reason otherwise.
    assert mod._adapter_is_zero() is False

    hs, ti, tw = _route()
    with torch.no_grad():
        before = mod(hs, ti, tw).float().cpu()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _enable(mod)
    try:
        with torch.no_grad():
            after = mod(hs, ti, tw).float().cpu()
        # The LoRA path still owns this forward: the delta lands pre-activation.
        assert torch.equal(before, after), "patching changed a trained-adapter forward"
        assert mod.base._pipelined.traffic() == {"hot_d2d_bytes": 0, "cold_pcie_bytes": 0}, \
            "the engine ran despite a non-zero adapter — the delta was dropped"
    finally:
        disable_pipelined_residency(mod)


def test_warns_when_a_non_zero_adapter_makes_the_patch_unreachable():
    mod = _wrapped()
    with torch.no_grad():
        mod.gate_up_lora_B.normal_(std=0.02)
    mod._delegate_ok = None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        n = _enable(mod)
    disable_pipelined_residency(mod)
    assert n == 1
    assert any("never run" in str(r.message) for r in rec), \
        "reported a patch count without warning that it is unreachable"


def test_warns_in_train_mode():
    """train mode is the silent one: the adapter is zero, so nothing about the DATA
    is wrong — `_delegate_to_base` just requires `not self.training`, and a loader
    hands back a model in nn.Module's default train mode."""
    mod = _wrapped().train()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _enable(mod)
    disable_pipelined_residency(mod)
    assert any("TRAINING mode" in str(r.message) for r in rec), \
        "a train-mode model silently bypasses every patch"


def test_v0_hot_residency_still_refuses_wrapped_bases():
    """`target_modules` now includes wrapped bases, but `_delegate_to_base` keys off
    `_e4b_fast_ref`/`_e4b_pipe_ref` and never looks for `_e4b_hot_ref`. The v0 engine
    must keep refusing rather than inherit a reachability it does not have."""
    from experts4bit_qlora.hot_residency import enable_hot_residency
    mod = _wrapped()
    with pytest.raises(NotImplementedError, match="enable_pipelined_residency"):
        enable_hot_residency(mod, [torch.tensor(HOT, dtype=torch.long)], device="cuda")
