"""DQ11 MLP-adapter BUILD Phase 2's runner (bench/dq11/dq11_mlp_phase2_run.py): the reduce's rungs (Amendment 5
section 7), the drift report, T_arm and the static fit, on synthetic records. CPU only; nothing here loads a model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "dq11"))
import dq11_mlp_phase2_run as r2  # noqa: E402

BASE_S = 1.0


def _b(**kw):
    out = {"fail": [], "calibration_indeterminate": [], "unjudged_at_k": [], "passed": ["0:dx"], "rank1_unresolved": [],
           "accuracy_over_reorder": [], "k_min_seen": 12}
    out.update(kw)
    return out


def _arm(name, t, *, fallback=0, enabled=96, witness_mlp=None, imported=None, quality=None):
    fused = name.startswith("F1")
    rec = {"part": name, "training": {"step_seconds": [9.0] * r2.WARMUP + [t] * 35, "losses": [1.0] * 40},
           "witness": {"mlp_lora_forwards": (0 if fused else 192) if witness_mlp is None else witness_mlp,
                       "attention_lora_forwards": 256},
           "engine_imported": fused if imported is None else imported}
    if fused:
        rec.update(enabled_adapters=enabled, refused={}, engine_stats={"fused": 32, "adapters": 96, "calls": 64,
                                                                       "fallback": fallback})
    if name in r2.QUALITY_ARMS:
        q = quality or {"alpaca-heldout": 5.0, "wikitext-test": 7.0}
        rec["quality_initial"] = {k: {"ppl": v} for k, v in q.items()}
        rec["quality_final"] = {k: {"ppl": v - 0.5} for k, v in q.items()}
    return rec


def _write(d, *, target="5090", f1=0.9, b=None, void=None, engagement_ok=True, arms=None, quality_f1=None):
    d.mkdir(parents=True, exist_ok=True)
    recs = {"stage-check": {"target": target},
            "b-read": {"read": {"engagement": {"ok": engagement_ok}, "outcome": b if b is not None else _b(),
                                **({"void": void} if void else {})}}}
    for a in r2.TIMED:
        recs[a] = _arm(a, BASE_S * (f1 if a.startswith("F1") else 1.0),
                       quality=quality_f1 if a == "F1a" else None)
    recs.update(arms or {})
    for k, v in recs.items():
        (d / f"{k}.json").write_text(json.dumps(v))
    return d


def test_the_timed_order_and_pairs_are_the_preregs():
    assert r2.TIMED == ("F0a", "F1a", "F1b", "F0b", "F0c", "F1c", "F1d", "F0d")       # ABBA twice
    assert r2.PAIRS == (("F0a", "F1a"), ("F0b", "F1b"), ("F0c", "F1c"), ("F0d", "F1d"))
    assert r2.SELF_PAIRS == (("F0a", "F0b"), ("F1a", "F1b"), ("F0c", "F0d"), ("F1c", "F1d"))
    assert r2.NOISY_BAND == (0.985, 1.015) and r2.H_BAR == 0.05 and r2.QUALITY_BAND == 0.05


def test_t_arm_is_the_median_of_updates_six_on():
    rec = {"training": {"step_seconds": [100.0] * 5 + [1.0, 2.0, 3.0]}}
    assert r2.t_arm(rec) == 2.0


def test_a_clean_read_with_h_over_the_bar_is_a_gain(tmp_path):
    out = r2.reduce(_write(tmp_path, f1=0.9))
    assert out["outcome"] == "GAIN" and out["h_min"] == pytest.approx(0.1) and out["quality"]["ok"]
    assert out["shakedown"] is False and out["science_eligible"] is True


def test_h_under_the_bar_is_no_gain_with_the_number(tmp_path):
    out = r2.reduce(_write(tmp_path, f1=0.97))
    assert out["outcome"].startswith("NO_GAIN: H 0.03") and "QUALITY_FAIL" not in out["outcome"]


def test_quality_out_of_its_band_is_no_gain(tmp_path):
    out = r2.reduce(_write(tmp_path, f1=0.9, quality_f1={"alpaca-heldout": 5.06, "wikitext-test": 7.0}))
    assert out["outcome"] == "NO_GAIN: QUALITY_FAIL" and not out["quality"]["ok"]


def test_a_noisy_self_pair_voids_and_still_names_an_accuracy_fail(tmp_path):
    noisy = {"F0b": _arm("F0b", 1.02)}
    out = r2.reduce(_write(tmp_path / "a", arms=noisy))
    assert out["outcome"] == "VOID (NOISY, a strike): F0b/F0a"
    out = r2.reduce(_write(tmp_path / "b", arms=noisy, b=_b(fail=["15:dx"])))
    assert out["outcome"] == "VOID (NOISY, a strike): F0b/F0a; ACCURACY_FAIL: 15:dx" and out["b"]["fail"] == ["15:dx"]


def test_engagement_is_the_first_rung(tmp_path):
    bad = {"F1c": _arm("F1c", 0.9, fallback=1), "F0b": _arm("F0b", 1.02)}               # also NOISY: engagement decides
    out = r2.reduce(_write(tmp_path / "a", arms=bad, b=_b(fail=["0:dx"])))
    assert out["outcome"].startswith("VOID (harness): F1c: enabled 96, refused {}, fallback 1")
    out = r2.reduce(_write(tmp_path / "b", engagement_ok=False))
    assert out["outcome"].startswith("VOID (harness): b-read:")
    out = r2.reduce(_write(tmp_path / "c", void="REF_MISMATCH: 0:R4:dx"))
    assert out["outcome"] == "VOID (harness): REF_MISMATCH: 0:R4:dx"
    out = r2.reduce(_write(tmp_path / "d", arms={"F1a": _arm("F1a", 0.9, witness_mlp=3)}))
    assert out["outcome"].startswith("VOID (harness): F1a:") and "PEFT MLP LoRA forwards 3" in out["outcome"]
    out = r2.reduce(_write(tmp_path / "e", arms={"F0c": _arm("F0c", 1.0, imported=True)}))
    assert out["outcome"].startswith("VOID (harness): F0c: the unfused arm imported the engine")


def test_the_accuracy_rungs_in_order_and_unjudged_caps_a_gain(tmp_path):
    every = _b(fail=["0:dx"], calibration_indeterminate=["15:dh"], unjudged_at_k=["0:dh"])
    assert r2.reduce(_write(tmp_path / "a", b=every))["outcome"] == "ACCURACY_FAIL: 0:dx"
    two = _b(calibration_indeterminate=["15:dh"], unjudged_at_k=["0:dh"])
    assert r2.reduce(_write(tmp_path / "b", b=two))["outcome"] == "NO_GAIN (accuracy indeterminate: 15:dh)"
    one = _b(unjudged_at_k=["0:dh"])
    out = r2.reduce(_write(tmp_path / "c", b=one, f1=0.5))                               # H = 0.5 cannot buy a GAIN
    assert out["outcome"] == "NO_GAIN (accuracy unjudged: 0:dh)"
    assert out["named"]["unjudged_at_k"] == ["0:dh"]


def test_rank1_and_reorder_are_named_beside_a_gain(tmp_path):
    out = r2.reduce(_write(tmp_path, b=_b(rank1_unresolved=["0:out_down"], accuracy_over_reorder=["0:out_gate"])))
    assert out["outcome"] == "GAIN"
    assert out["named"] == {"rank1_unresolved": ["0:out_down"], "accuracy_over_reorder": ["0:out_gate"]}


def test_the_a2000_shakedown_is_labelled_and_never_science(tmp_path):
    out = r2.reduce(_write(tmp_path, target="a2000"))
    assert out["outcome"] == "SHAKEDOWN (never Phase 2 data): GAIN" and out["science_eligible"] is False


def test_drift_is_reported_and_never_changes_the_outcome(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    d = _write(tmp_path, f1=0.9)
    assert r2.reduce(d)["drift"]["label"] == "drift: not read"                            # no drift records
    for a, scale in (("D0", 1.0), ("D1", 1.01)):
        save_file({"step1.A": torch.ones(4) * scale, "step1.B": torch.ones(4), "step2.A": torch.ones(4) * scale,
                   "step2.B": torch.ones(4)}, str(d / f"drift-{a}.safetensors"))
        (d / f"{a}.json").write_text(json.dumps({"snapshots": {"file": f"drift-{a}.safetensors"},
                                                 "training": {"losses": [1.0, 0.9 + (scale - 1)]}}))
    out = r2.reduce(d)
    assert out["outcome"] == "GAIN" and out["drift"]["label"].startswith("under C")
    assert out["drift"]["relative_distance"]["step1.A"] == pytest.approx(0.01)
    assert out["drift"]["relative_distance"]["step2.B"] == 0.0
    (d / "D1.json").write_text("{not json")
    out = r2.reduce(d)
    assert out["outcome"] == "GAIN" and out["drift"]["label"] == "drift: not read" and "JSONDecodeError" in out["drift"]["reason"]


def test_the_static_fit_registers_two_hours():
    est = [75 * 60]                                                                       # Amendment 5's upper estimate
    assert r2.GUARD_S == 7200 and r2.BOOT_S == 300 and r2.FIT_FACTOR == 1.25
    assert r2.static_fit(est)["fits"] and r2.static_fit(est)["need_s"] == 6000.0
    assert not r2.static_fit(est, guard_s=5400)["fits"]                                   # 1.5 h fails its own fit
    assert r2.static_fit([5460])["fits"] and not r2.static_fit([5461])["fits"]           # (300 + 5460) x 1.25 = 7200


def test_reduce_refuses_records_of_another_target(tmp_path):
    d = _write(tmp_path, target="a2000")
    with pytest.raises(SystemExit, match="target a2000, not 5090"):
        r2.main(["--target", "5090", "--part", "reduce", "--directory", str(d), "--cache", str(d), "--out", str(d)])
    assert r2.main(["--target", "a2000", "--part", "reduce", "--directory", str(d), "--cache", str(d), "--out", str(d)]) == 0
    assert json.loads((d / "phase2.json").read_text())["outcome"].startswith("SHAKEDOWN")
