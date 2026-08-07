"""serve's residency wiring: the refusals, the ordering, and the silent-fallback guard.

The engine itself is tested in test_pipelined.py. What is new here is serve DECIDING to
attach it, and every way that decision can go quietly wrong:

* a hot set chosen by index instead of by frequency — the dial appears to work and buys
  ~nothing (a 16-wide index set on a 256-expert top-6 layer catches ~6% of routed slots),
* a profile from a different model — the engine takes one set per module IN ORDER, so a
  length mismatch shifts every set onto the wrong layer,
* an engine that patches 0 modules — "residency on" in the logs, streaming in reality,
* a trained adapter — the wrapper stops delegating and residency silently stops running.

Every one of those is a green-looking failure, which is the class this file exists for.
No CUDA and no model load: these exercise the decision logic directly.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.serve import ServeConfig  # noqa: E402


def _engine(**kw):
    """A ServeEngine shell with just the fields _enable_residency touches. Constructing the
    real one would spin worker threads; the decision logic under test needs none of that."""
    from experts4bit_qlora.serve import Engine

    eng = Engine.__new__(Engine)
    eng.cfg = ServeConfig(**kw)
    eng.residency_n = 0
    eng.residency_coverage = None
    return eng


class _Cfg:
    num_experts_per_tok = 4


def _profile(path, layers=2, experts=8):
    """A minimal E4B_EXPERT_PROFILE JSONL: layer rows + skewed expert counts."""
    with open(path, "w") as f:
        for lid in range(layers):
            f.write(json.dumps({"row": "layer", "layer_id": lid, "num_experts": experts}) + "\n")
            for eid in range(experts):
                f.write(json.dumps({"row": "expert", "layer_id": lid, "expert_id": eid,
                                    "tokens_routed": (experts - eid) * 10}) + "\n")


def test_off_by_default_is_a_no_op():
    eng = _engine()
    eng._enable_residency(object(), _Cfg())          # must not touch the model at all
    assert eng.residency_n == 0


def test_unknown_mode_is_rejected():
    eng = _engine(residency="cold-engine")
    with pytest.raises(ValueError, match="not supported"):
        eng._enable_residency(object(), _Cfg())


def test_refuses_a_by_index_hot_set(tmp_path):
    """The load-bearing refusal. Without a profile the only thing left is by-index, which
    is the behaviour hot_sets_from_profile exists to replace — so this raises rather than
    serving a dial that buys ~6% and looks enabled."""
    eng = _engine(residency="pipelined", hot_per_layer=8)
    with pytest.raises(ValueError, match="frequency-ranked"):
        eng._enable_residency(object(), _Cfg())

    prof = tmp_path / "p.jsonl"
    _profile(prof)
    eng = _engine(residency="pipelined", hot_profile=str(prof))   # profile but no K
    with pytest.raises(ValueError, match="E4B_HOT_PER_LAYER"):
        eng._enable_residency(object(), _Cfg())


def test_missing_profile_file_is_named(tmp_path):
    eng = _engine(residency="pipelined", hot_profile=str(tmp_path / "nope.jsonl"),
                  hot_per_layer=4)
    with pytest.raises(FileNotFoundError, match="E4B_HOT_PROFILE"):
        eng._enable_residency(object(), _Cfg())


def test_unknown_top_k_is_refused(tmp_path):
    """k_slots sizes the slot store; a forward with a different k falls back to the
    reference path, so guessing it would be a silent de-acceleration."""
    prof = tmp_path / "p.jsonl"
    _profile(prof)
    eng = _engine(residency="pipelined", hot_profile=str(prof), hot_per_layer=4)

    class NoK:
        pass

    with pytest.raises(ValueError, match="E4B_K_SLOTS"):
        eng._enable_residency(object(), NoK())


def test_profile_model_mismatch_is_refused(tmp_path, monkeypatch):
    """One set per targeted module, in module order. A 2-layer profile against a 3-module
    model would shift every set onto the wrong layer — and still 'work'."""
    import experts4bit_qlora.hot_residency as hr

    prof = tmp_path / "p.jsonl"
    _profile(prof, layers=2)
    monkeypatch.setattr(hr, "target_modules", lambda m: [object()] * 3)
    eng = _engine(residency="pipelined", hot_profile=str(prof), hot_per_layer=4)
    with pytest.raises(ValueError, match="profile/model mismatch"):
        eng._enable_residency(object(), _Cfg())


def test_zero_patched_modules_is_refused(tmp_path, monkeypatch):
    """'Residency requested and not running' must fail loudly: the whole point of the
    deployment knob is that its effect is invisible in anything but tok/s."""
    import experts4bit_qlora.hot_residency as hr
    import experts4bit_qlora.pipelined as pl

    prof = tmp_path / "p.jsonl"
    _profile(prof, layers=2)
    monkeypatch.setattr(hr, "target_modules", lambda m: [object()] * 2)
    monkeypatch.setattr(pl, "enable_pipelined_residency", lambda *a, **k: 0)
    eng = _engine(residency="pipelined", hot_profile=str(prof), hot_per_layer=4)
    with pytest.raises(RuntimeError, match="patched 0 modules"):
        eng._enable_residency(object(), _Cfg())


def test_happy_path_records_coverage_and_k(tmp_path, monkeypatch):
    import experts4bit_qlora.hot_residency as hr
    import experts4bit_qlora.pipelined as pl

    prof = tmp_path / "p.jsonl"
    _profile(prof, layers=2, experts=8)
    seen = {}

    def fake_enable(model, hot_sets, device="cuda", k_slots=None, **kw):
        seen["hot_sets"] = hot_sets
        seen["k_slots"] = k_slots
        return len(hot_sets)

    monkeypatch.setattr(hr, "target_modules", lambda m: [object()] * 2)
    monkeypatch.setattr(pl, "enable_pipelined_residency", fake_enable)
    eng = _engine(residency="pipelined", hot_profile=str(prof), hot_per_layer=4)
    eng._enable_residency(object(), _Cfg())

    assert eng.residency_n == 2
    assert seen["k_slots"] == 4, "top-k must come from the model config when unset"
    # frequency-ranked, most-routed first — NOT 0..3
    assert seen["hot_sets"][0] == [0, 1, 2, 3] or seen["hot_sets"][0][0] == 0
    assert eng.residency_coverage is not None and 0.0 < eng.residency_coverage <= 1.0


def test_explicit_k_slots_overrides_the_config(tmp_path, monkeypatch):
    import experts4bit_qlora.hot_residency as hr
    import experts4bit_qlora.pipelined as pl

    prof = tmp_path / "p.jsonl"
    _profile(prof, layers=2)
    seen = {}
    monkeypatch.setattr(hr, "target_modules", lambda m: [object()] * 2)
    monkeypatch.setattr(pl, "enable_pipelined_residency",
                        lambda m, h, device="cuda", k_slots=None, **kw: seen.setdefault("k", k_slots) or len(h))
    eng = _engine(residency="pipelined", hot_profile=str(prof), hot_per_layer=4, k_slots=6)
    eng._enable_residency(object(), _Cfg())
    assert seen["k"] == 6


def test_env_surface_round_trips(monkeypatch):
    for k, v in (("E4B_RESIDENCY", "pipelined"), ("E4B_HOT_PROFILE", "/tmp/p.jsonl"),
                 ("E4B_HOT_PER_LAYER", "12"), ("E4B_K_SLOTS", "8")):
        monkeypatch.setenv(k, v)
    cfg = ServeConfig.from_env()
    assert (cfg.residency, cfg.hot_profile, cfg.hot_per_layer, cfg.k_slots) == \
        ("pipelined", "/tmp/p.jsonl", 12, 8)


def test_residency_defaults_are_off():
    cfg = ServeConfig()
    assert cfg.residency == "" and cfg.hot_per_layer == 0 and cfg.k_slots == 0


def test_adapter_swap_invalidates_the_delegation_cache():
    """Bugbot's finding on the PR that added the swap guard, reproduced then fixed.

    ``copy_`` onto the live LoRA params fires neither of ExpertsLoRA's invalidations
    (train()/load_state_dict), so ``_delegate_ok`` kept the PREVIOUS adapter's verdict.
    With residency attached, a stale True keeps delegating to the patched base — BASE
    outputs served under the new adapter's name — and the swap warning read the same
    stale cache, so it stayed silent about exactly the state it existed to report.
    """
    from experts4bit_qlora import ExpertsLoRA, ExpertsNbit

    gu = torch.randn(4, 2 * 32, 16)
    dn = torch.randn(4, 16, 32)
    base = ExpertsNbit.from_float(gu, dn, quant_type="bf16", compute_dtype=torch.float32)
    mod = ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32).eval()
    assert mod._adapter_is_zero()                      # caches True (B is zero-init)

    # serve's swap pattern: raw copy_ over the live params — no train(), no load_state_dict
    with torch.no_grad():
        mod.gate_up_lora_B.data.copy_(torch.randn_like(mod.gate_up_lora_B))
    assert mod._adapter_is_zero(), "precondition: the cache IS stale (bug reproduces)"

    # what _swap_adapter now does before consulting it
    mod._delegate_ok = None
    assert not mod._adapter_is_zero(), "invalidation must surface the non-zero adapter"

    # and swapping back to a zero adapter re-decides fresh in the other direction
    with torch.no_grad():
        mod.gate_up_lora_B.data.zero_()
    mod._delegate_ok = None
    assert mod._adapter_is_zero()
