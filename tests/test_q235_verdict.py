# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's milestone verdict (``bench/q235/instrument/q235_verdict.py``) on fabricated box records: one PASS, a FAIL for each rule,
and a VOID for each cause (Amendment 3's instrument checks among them), each with its exit code."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
VERDICT = ROOT / "bench" / "q235" / "instrument" / "q235_verdict.py"
RULES = ("step0_all_finite", "step0_every_B_nonzero", "step0_every_A_zero", "step0_every_B_changed", "step1_all_finite",
         "step1_every_A_nonzero", "step1_every_B_nonzero", "step1_every_A_changed", "step1_every_B_changed")
PRO6000 = "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 575.64.05, 5, 5, 16, 16, 600.00 W"


def _sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _save_run(d, layers, topk, loss, rec):
    d.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(layers):
        torch.save(t, d / f"layer{i:03d}.pt")
    torch.save({"router_topk": topk}, d / "nf4.pt")
    json.dump({"loss": loss}, open(d / "nf4.json", "w"))
    json.dump({**rec, "layer_sha256": [_sha(t) for t in layers]}, open(d / "box.json", "w"))


def _box_run(root, *, code="0", card=PRO6000, mode="science", rules=None, losses=(2.0, 1.9), repeat_loss=2.0, scale=1.01,
             peak=24.0 * (1 << 30), null_scale=1.0, plant_kick=3.0, plant_from=2, plant_at=2, repeat=True, pre_scale=1.0):
    """A fetched science run (q235_run.sh's layout) and a reference beside it; 4 layers of [2, 6, 8], top-2 of 8 experts. The box
    is the reference's per-layer updates times ``scale`` (u = scale - 1); the repeat's null is the box times ``null_scale``; each plant
    is the null with ``plant_kick`` x a fixed vector added from layer ``plant_from`` on (u at that layer about plant_kick x 10), and its
    earlier layers the null's times ``pre_scale``."""
    g = torch.Generator().manual_seed(0)
    w, ref = root / "run", root / "ref"
    (w / "receipts").mkdir(parents=True)
    mask = torch.ones(2, 6, dtype=torch.long)
    torch.save({"input_ids": torch.zeros(2, 6, dtype=torch.long), "attention_mask": mask, "labels": torch.zeros(2, 6, dtype=torch.long)},
               w / "receipts" / "tokens.pt")
    h0 = torch.randn(2, 6, 8, generator=g) * 10
    ups = [torch.randn(2, 6, 8, generator=g) * 0.1 for _ in range(4)]
    ref_l = [h0 + sum(ups[:i + 1]) for i in range(4)]
    box_l = [h0 + scale * sum(ups[:i + 1]) for i in range(4)]
    topk = [torch.randint(0, 8, (12, 2), generator=g) for _ in range(4)]
    ref.mkdir()
    for i, t in enumerate(ref_l):
        torch.save(t, ref / f"layer{i:03d}.pt")
    torch.save({"router_topk": topk}, ref / "reference.pt")
    json.dump({"loss": 2.01}, open(ref / "reference.json", "w"))
    step = {"loss": losses[0], "elapsed_s": 60.0, "h2d_bytes_offload": 220 * (1 << 30), "max_memory_allocated": int(peak)}
    box = {"rules": rules if rules is not None else {k: True for k in RULES}, "frozen_sha256_after_both_steps_equal": True,
           "in_process": {"loss_0": 2.0, "loss_1": 2.0}, "step": step, "step1": {**step, "loss": losses[1]}, "plant": {"kind": None}}
    _save_run(w / "receipts" / "box", box_l, topk, 2.0, box)
    if repeat:
        null = [t * null_scale if null_scale != 1.0 else t.clone() for t in box_l]
        kick = torch.randn(2, 6, 8, generator=g)
        plants = {}
        for kind in ("routing", "epilogue"):
            pl = [t + plant_kick * kick if i >= plant_from else (t * pre_scale if pre_scale != 1.0 else t.clone())
                  for i, t in enumerate(null)]
            _save_run(w / "receipts" / "repeat" / f"plant_{kind}", pl, topk, 2.3,
                      {"plant": {"kind": kind, "layer": plant_at}, "in_process": True})
            plants[kind] = {"layer": plant_at}
        _save_run(w / "receipts" / "repeat", null, topk, repeat_loss, {"repeat": {"loss": repeat_loss}, "plants": plants})
    json.dump({"schema": "q235-run/1", "mode": mode, "science_eligible": mode == "science"}, open(w / "receipts" / "run.json", "w"))
    (w / "receipts" / "card.txt").write_text(card + "\n")
    (w / "TC1_EXIT_CODE.n0nce").write_text(code + "\n")
    (w / "summary.txt").write_text("")
    return w, ref


def _verdict(w, ref):
    r = subprocess.run([sys.executable, str(VERDICT), str(w), "--ref", str(ref), "--out", str(w / "verdict.json")],
                       capture_output=True, text=True, cwd=ROOT)
    return r.returncode, json.load(open(w / "verdict.json"))


def test_a_clean_box_is_a_milestone_pass(tmp_path):
    rc, v = _verdict(*_box_run(tmp_path))
    assert (rc, v["verdict"]) == (0, "PASS"), v["why"]
    assert all(x["ok"] for x in v["rules"].values()) and v["reported"]["estimator_issue"] is False


@pytest.mark.parametrize("kw,rule", [
    ({"card": "NVIDIA GeForce RTX 5090, 32607, 575.64.05, 5, 5, 16, 16, 575.00 W"}, "1_completed_on_class"),
    ({"losses": (float("nan"), 1.9)}, "2_loss_finite"),
    ({"code": "12", "rules": {**{k: True for k in RULES}, "step1_every_A_changed": False}, "repeat": False}, "3_5_strict"),
    ({"rules": {}}, "3_5_strict"),                                             # fail-closed: no rules recorded
    ({"repeat_loss": 2.0 * (1 + 3e-4)}, "6_repeat"),
    ({"scale": 1.9}, "7_band"),                                                # u = 0.9 > 0.775 on every layer
])
def test_each_rule_fails_on_its_own(tmp_path, kw, rule):
    rc, v = _verdict(*_box_run(tmp_path, **kw))
    assert (rc, v["verdict"]) == (1, "FAIL") and rule in v["why"], v["why"]
    assert [k for k, x in v["rules"].items() if x["ok"] is False] == [rule], v["rules"]   # None: not read (exit 12)


@pytest.mark.parametrize("setup,cause", [
    (lambda w: [(w / "TC1_EXIT_CODE.n0nce").write_text("18\n"), (w / "REFUSAL").write_text("host rate floor\n")], "refusal: host rate floor"),
    (lambda w: [(w / "TC1_EXIT_CODE.n0nce").write_text("14\n"), (w / "REFUSAL").write_text("pull rate floor\n")], "refusal: pull rate floor"),
    (lambda w: (w / "host-fault.json").write_text('{"kind": "reboot"}'), "host fault"),
    (lambda w: [(w / "TC1_EXIT_CODE.n0nce").write_text("11\n"), (w / "summary.txt").write_text("q235: VOID: deadline reserve reached\n")],
     "time-left guard"),
    (lambda w: (w / "TC1_EXIT_CODE.n0nce").write_text("9\n"), "harness error (exit 9)"),
    (lambda w: (w / "receipts" / "repeat" / "box.json").unlink(), "harness error: FileNotFoundError"),
])
def test_each_void_cause_gives_no_reading(tmp_path, setup, cause):
    w, ref = _box_run(tmp_path)
    setup(w)
    rc, v = _verdict(w, ref)
    assert (rc, v["verdict"]) == (2, "VOID") and v["why"].startswith(cause), v["why"]


def test_the_band_is_the_contribution_not_the_state(tmp_path):
    """u = 0.4 on every layer: inside the band. The full state is reported, never gated."""
    rc, v = _verdict(*_box_run(tmp_path, scale=1.4))
    assert (rc, v["verdict"]) == (0, "PASS"), v["why"]
    assert abs(v["rules"]["7_band"]["u_interior_max"] - 0.4) < 1e-4 and v["rules"]["7_band"]["interior"] == [1, 3]
    assert v["reported"]["u_profile"][0] is None and "median_upper_middle" in v["reported"]["full_state"]
    ins = v["instrument"]
    assert ins["null_bitwise"] and ins["null_vs_box_max"] == 0
    assert all(p["fails_band"] and p["pre_plant_bitwise"] and p["pre_plant_within_floor"] and p["planted_layer_differs"]
               for p in ins["plants"].values())


def test_a_plant_s_earlier_layers_are_read_with_the_tolerance_not_bit_for_bit(tmp_path):
    """Last-bit drift before the planted layer (about 1e-6 relative) reads; bit-for-bit is only reported."""
    rc, v = _verdict(*_box_run(tmp_path, pre_scale=1 + 1e-6))
    assert (rc, v["verdict"]) == (0, "PASS"), v["why"]
    p = v["instrument"]["plants"]["routing"]
    assert p["pre_plant_bitwise"] is False and p["pre_plant_within_floor"] and 0 < p["pre_plant_max_rel_diff"] < 1e-5


def test_a_strict_failure_stops_before_the_repeat_and_is_a_fail(tmp_path):
    rc, v = _verdict(*_box_run(tmp_path, code="12", rules={**{k: True for k in RULES}, "step0_every_B_changed": False}, repeat=False))
    assert (rc, v["verdict"]) == (1, "FAIL") and v["rules"]["6_repeat"]["ok"] is None and v["rules"]["7_band"]["ok"] is None


@pytest.mark.parametrize("kw,cause", [
    ({"null_scale": 1.01}, "instrument: the null is not the box"),
    ({"plant_from": 1}, "instrument: the routing plant touched the layers before it"),
    ({"pre_scale": 1.01}, "instrument: the routing plant touched the layers before it (max 0.01"),
    ({"plant_kick": 0.0}, "instrument: the routing plant did not engage"),
    ({"plant_kick": 0.05}, "instrument: the routing plant passes the band"),
    ({"plant_at": 3}, "instrument: the routing plant is not at layer 2"),
    ({"null_scale": 1.01, "scale": 1.9}, "instrument: the null is not the box"),           # VOID before the band's FAIL
])
def test_each_instrument_check_voids_before_any_band(tmp_path, kw, cause):
    rc, v = _verdict(*_box_run(tmp_path, **kw))
    assert (rc, v["verdict"]) == (2, "VOID") and v["why"].startswith(cause), v["why"]


def test_a_rehearsal_is_never_a_milestone(tmp_path):
    rc, v = _verdict(*_box_run(tmp_path, mode="a2000"))
    assert (rc, v["verdict"]) == (2, "VOID") and "not a science run" in v["why"]


def test_a_device_peak_far_from_the_estimate_is_reported_not_gated(tmp_path):
    rc, v = _verdict(*_box_run(tmp_path, peak=30.0 * (1 << 30)))
    assert (rc, v["verdict"]) == (0, "PASS") and v["reported"]["estimator_issue"] is True
