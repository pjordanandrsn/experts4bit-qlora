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
from experts4bit_qlora.hot_residency import target_modules, wrapped_bases  # noqa: E402
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
        # The resident tier is no longer witnessed by BYTES: since the in-place hot
        # path a hot expert is read from its row in the shared store and nothing is
        # copied for it, so hot_d2d is 0 whether it was read or never touched. The
        # row dispatch is the direct witness instead, and a sharper one — it names
        # which row the GEMM actually read for each routed lane.
        assert moved["hot_d2d_bytes"] == 0, (
            f"a hot lane was copied into a slot instead of read in place ({moved})")
        # ROUTE = [2, 5] against HOT = [0,1,2,3]: lane 0 is hot, lane 1 is cold.
        assert int(st.row_idx[0]) < st.n_hot, (
            f"hot expert {ROUTE[0]} was not served from the resident rows ({st.row_idx})")
        assert int(st.row_idx[1]) >= st.n_hot, (
            f"cold expert {ROUTE[1]} was not served from a gathered slot ({st.row_idx})")

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


# ---------------------------------------------------------------------------
# The loader path itself.
#
# Everything above builds `ExpertsLoRA(Experts4bit(...))` by hand. That is the
# right unit for the delegation rules, but it is not the shape the failure was
# reported on: `load_moe_4bit_streaming` is what produces wrapped bases in the
# field, and it is the only way the fidelity bench builds a model. A hand-built
# wrapper can keep passing while the loader grows a detail that breaks the
# composition -- an extra nn.Module layer between the block and the base, an
# expert module built on `meta`, a compute_dtype the engine falls back on -- and
# nothing here would notice.
# ---------------------------------------------------------------------------

def _loader_model(tmp_path):
    """A real `load_moe_4bit_streaming` model from the tiny 2-layer OLMoE fixture.

    `_olmoe`/`_write_ckpt` come from the loader suite rather than being copied:
    a second checkpoint writer would drift from the real one exactly where it
    matters (`_write_ckpt`'s dtype and shared-storage handling are both
    load-bearing and both were bug fixes), and a fixture that no longer matches
    the loader's expected on-disk layout fails as "the engine broke".

    pytest's default (prepend) import mode puts `tests/` on `sys.path` -- there
    is no `tests/__init__.py` -- so the sibling module imports by bare name.
    Imported inside the function so an import-mode change costs these two tests
    rather than the whole module at collection time.
    """
    from test_loader_architectures import _olmoe, _write_ckpt

    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    _write_ckpt(_olmoe(), str(tmp_path), per_expert=False)
    model, cfg = load_moe_4bit_streaming(str(tmp_path), "cuda", torch.bfloat16, r=4, alpha=8)
    model.config.use_cache = False
    return model.eval(), cfg


def _decode(model, cfg):
    """One DECODE-shaped forward (B=1, T=1).

    Load-bearing: the patched forward hands back to the reference whenever
    `hidden.shape[0] != 1`, so a prefill-shaped forward -- which is what the
    other loader tests run -- exercises the engine not at all and would assert
    zero traffic no matter what.
    """
    with torch.no_grad():
        model(input_ids=torch.randint(0, cfg.vocab_size, (1, 1), device="cuda"))


def test_streaming_loader_model_can_enable_residency(tmp_path):
    """The reported failure, end to end.

    `enable_pipelined_residency` raised NotImplementedError ("every ExpertsNbit
    here is an ExpertsLoRA.base") for every model this loader returns, so the
    engine was unreachable from the only path that builds one at model scale.
    """
    model, cfg = _loader_model(tmp_path)
    mods = target_modules(model)
    assert mods, "the loader model exposes no residency targets at all"
    # Every target is a wrapped base: this model is precisely the all-wrapped
    # case the old guard singled out, not a mixture that would dodge it.
    assert {id(m) for m in mods} == wrapped_bases(model)

    hot = [torch.arange(m.num_experts, dtype=torch.long) for m in mods]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        n = enable_pipelined_residency(model, hot, device="cuda",
                                       k_slots=cfg.num_experts_per_tok)
    try:
        assert n == len(mods)
        # eval + an untrained (identically zero) adapter is the delegating case,
        # so neither unreachability warning belongs here. If one fires, the
        # loader is handing back something the engine cannot actually serve.
        assert not [r for r in rec if "[pipelined]" in str(r.message)], \
            [str(r.message) for r in rec]
    finally:
        assert disable_pipelined_residency(model) == n


def test_loader_hot_sets_land_on_the_layer_target_modules_named(tmp_path):
    """`hot_sets[i]` must mean the i-th module of `target_modules(model)`.

    This is the reason `enable_pipelined_residency` shares that enumeration
    instead of re-deriving one: the two lists agreeing is what keeps per-layer
    residency state off the wrong layer. Nothing checks it on a real model --
    the hand-built fixtures above have exactly one MoE module, where every
    ordering is the same ordering.
    """
    model, cfg = _loader_model(tmp_path)
    mods = target_modules(model)
    assert len(mods) >= 2, "fixture needs two MoE layers or the two sets cannot differ"

    # Opposite extremes, so the check cannot depend on which experts the router
    # happens to pick: a fully-streamed layer can only ever move COLD bytes and a
    # fully-resident one can only ever move HOT bytes. Swap the two entries and
    # both assertions below fail. Giving both layers the same set would prove
    # nothing -- a mis-indexed stamp would be indistinguishable from a correct one.
    streamed, resident = mods[0], mods[-1]
    hot = [torch.empty(0, dtype=torch.long) for _ in mods]
    hot[-1] = torch.arange(resident.num_experts, dtype=torch.long)

    n = enable_pipelined_residency(model, hot, device="cuda",
                                   k_slots=cfg.num_experts_per_tok)
    try:
        assert n == len(mods)
        # The stamp itself, before any forward: cheapest and most direct.
        assert streamed._pipelined.hot_ids.numel() == 0
        assert resident._pipelined.hot_ids.tolist() == list(range(resident.num_experts))

        _decode(model, cfg)

        # Traffic > 0 is guaranteed, not routing-dependent: `_prime()` seeds every
        # slot with expert 0's row, and top-k picks k DISTINCT experts, so at most
        # one routed id can be a have-skip and the rest must miss.
        s, r = streamed._pipelined.traffic(), resident._pipelined.traffic()
        assert s["cold_pcie_bytes"] > 0, \
            f"engine never ran on the streamed layer through the loader's wrapper ({s})"
        assert s["hot_d2d_bytes"] == 0, \
            f"a layer given an EMPTY hot set moved hot bytes — hot_sets is misindexed ({s})"
        assert r["cold_pcie_bytes"] == 0, \
            f"a FULLY-resident layer streamed from the cold arena — hot_sets is misindexed ({r})"
        # The fully-resident layer moves NO bytes of either kind since the in-place
        # hot path, so traffic can no longer witness that it ran — zero is also what
        # a dead engine reports. The row dispatch can: every lane on the resident
        # layer must name a row in the hot segment, and every lane on the streamed
        # layer must name a gathered slot. That is a strictly sharper statement of
        # the property this test exists for (hot_sets landing on the named layer).
        rs, rr = streamed._pipelined, resident._pipelined
        assert bool((rr.row_idx < rr.n_hot).all()), \
            f"resident layer served a lane from a slot — engine idle or misindexed ({rr.row_idx})"
        assert bool((rs.row_idx >= rs.n_hot).all()), \
            f"streamed layer served a lane from the hot segment ({rs.row_idx})"
    finally:
        disable_pipelined_residency(model)
