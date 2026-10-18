# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's milestone verdict (``bench/q235/instrument/q235_verdict.py``) on fabricated box records: one PASS, a FAIL for each rule,
and a VOID for each cause (Amendment 3's instrument checks among them), each with its exit code."""
from __future__ import annotations

import hashlib
import json
import math
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
             peak=24.0 * (1 << 30), null_scale=1.0, plant_kick=3.0, plant_from=2, plant_at=2, repeat=True, pre_scale=1.0,
             shape_b=None, a8=None):
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
    if shape_b is not None:                                 # Amendment 6: {"ga": (step 0, step 1), "plants": (mean, overwrite), "zero": [...]}
        def ga(x, zero=()):
            return {"compared": 48, "max_rel": x, "max_rel_tensor": "model.layers.3.mlp.experts.lora_B", "zero_mismatch": list(zero),
                    "cosine": 1 - x * x, "norm_ratio": 1.0, "per_layer_max_rel": {"3": x}}
        box["shape_b"] = {"micro_batch": 2, "accumulation": 4, "valid_per_micro_batch": [125, 252, 480, 149], "valid_total": 1006}
        for k, x, z in (("step", shape_b["ga"][0], shape_b.get("zero", ())), ("step1", shape_b["ga"][1], ())):
            box[k]["rule8"] = {"algorithms": shape_b.get("ran", "deterministic"), "restored": "default",
                               "check": ga(x, z), "floor": ga(shape_b.get("floor", 0.0))}
            box[k]["algorithms"] = "default"
            box[k]["ga_mb_check"] = ga(x)                     # the registered step's own accumulation: reported only
        box["step"]["ga_check"] = box["step1"]["ga_check"] = {**ga(0.3), "rel_p50": 0.2}       # shape A vs B: reported only
        box["ga_plants"] = {k: {"loss": 2.1, **ga(0.4), "vs_micro_batches": ga(v)} for k, v in zip(("mean", "overwrite"), shape_b["plants"])}
        box["route_flips"] = {"flipped": 12, "token_layers": 4000, "flipped_fraction": 0.003, "per_layer": [], "layers_with_flips": 3}
    if a8 is not None:                                      # Amendment 8: {"dx": (step 0 norms, step 1 norms), "heldout": (b, s0, a) or None}
        def dx(norms):
            bad = [[0, i] for i, v in enumerate(norms) if not (isinstance(v, float) and math.isfinite(v) and v > 0)]
            vals = [v for v in norms if isinstance(v, float)]
            return {"layers": len(norms), "passes": 1, "norms": [list(norms)], "finite_nonzero": not bad, "failing": bad,
                    "failing_count": len(bad), "min": min(vals) if vals else None, "max": max(vals) if vals else None}
        box["step"]["dx"], box["step1"]["dx"] = dx(a8["dx"][0]), dx(a8["dx"][1])
        box["rules"] = {**box["rules"], "step0_dx_finite_nonzero": box["step"]["dx"]["finite_nonzero"],
                        "step1_dx_finite_nonzero": box["step1"]["dx"]["finite_nonzero"]}
        held = a8.get("heldout")
        box["amendment8"] = {"dx": True, "heldout": held is not None}
        if held is not None:
            b, s0, a = held
            fin = all(math.isfinite(v) for v in held)
            box["heldout"] = {"tokens_sha256": {"input_ids": "0" * 64}, "rows": 2, "real_tokens": 12, "loss_before": b,
                              "loss_after_step0": s0, "loss_after": a, "rel_change_step0": (s0 - b) / b, "rel_change": (a - b) / b,
                              "all_finite": fin, "train": {"loss_before": 1.87, "loss_after_step0": 1.30, "loss_after": 0.91}}
            box["rules"]["heldout_finite"] = fin
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


def _verdict_with(w, ref, bar):
    """The verdict in this process, with GA_BAR set (the registered value is the calibration's; these tests fix one)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("q235_verdict_ga", VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.GA_BAR = bar
    return mod.judge(str(w), str(ref))


def test_shape_b_passes_rule_8_within_the_bar_and_fails_beyond_it(tmp_path):
    """Amendment 6: an accumulation within the bar at both steps passes as rules 1-8; beyond it at either step it is a FAIL on rule 8,
    and a tensor zero in shape A but not in shape B is a FAIL too. The views ride along."""
    ok = _verdict_with(*_box_run(tmp_path / "a", shape_b={"ga": (0.01, 0.02), "plants": (0.6, 0.9)}), 0.05)
    assert ok["verdict"] == "PASS" and ok["why"] == "rules 1-8 hold", ok["why"]
    r8 = ok["rules"]["8_accumulation"]
    assert r8["ok"] and r8["bar"] == 0.05 and r8["max_rel"] == {"step": 0.01, "step1": 0.02} and set(r8["cosine"]) == {"step", "step1"}
    assert r8["reported_a_vs_b"]["step"]["max_rel"] == 0.3 and r8["reported_route_flips"]["flipped"] == 12   # reported, not gating
    bad = _verdict_with(*_box_run(tmp_path / "b", shape_b={"ga": (0.01, 0.2), "plants": (0.6, 0.9)}), 0.05)
    assert bad["verdict"] == "FAIL" and bad["why"] == "rules failing: 8_accumulation" and bad["rules"]["8_accumulation"]["over"] == {"step1": 0.2}
    zero = _verdict_with(*_box_run(tmp_path / "c", shape_b={"ga": (0.01, 0.02), "plants": (0.6, 0.9),
                                                             "zero": ["model.layers.0.self_attn.q_proj.lora_A"]}), 0.05)
    assert zero["verdict"] == "FAIL" and zero["rules"]["8_accumulation"]["zero_mismatch"]


@pytest.mark.parametrize("plants, kind", [((0.04, 0.9), "mean"), ((0.6, 0.05), "overwrite")])
def test_an_accumulation_plant_inside_the_bar_voids_the_box(tmp_path, plants, kind):
    """Amendment 6's instrument check: a plant at or under the bar means the bar cannot see that bug at this depth -- VOID."""
    v = _verdict_with(*_box_run(tmp_path, shape_b={"ga": (0.01, 0.02), "plants": plants}), 0.05)
    assert v["verdict"] == "VOID" and v["why"].startswith(f"instrument: the {kind} accumulation plant passes the bar"), v["why"]


def test_shape_b_without_a_registered_bar_is_void_and_shape_a_never_reads_rule_8(tmp_path):
    v = _verdict_with(*_box_run(tmp_path / "b", shape_b={"ga": (0.01, 0.02), "plants": (0.6, 0.9)}), None)
    assert v["verdict"] == "VOID" and v["why"] == "not registered: Amendment 6's accumulation bar", v["why"]
    a = _verdict_with(*_box_run(tmp_path / "a"), 0.05)
    assert a["verdict"] == "PASS" and a["why"] == "rules 1-7 hold" and "8_accumulation" not in a["rules"]


@pytest.mark.parametrize("extra, cause", [({"floor": 0.02}, "floor at step is 0.02"), ({"ran": "default"}, "ran 'default'")])
def test_rule_8_cannot_be_read_without_its_deterministic_check(tmp_path, extra, cause):
    """Amendment 6: the check passes must have run deterministic (default restored for the step), and their floor -- two deterministic
    accumulations against each other -- must sit at most BAR / 3, or rule 8 is VOID, never a FAIL from noise."""
    v = _verdict_with(*_box_run(tmp_path, shape_b={"ga": (0.01, 0.02), "plants": (0.6, 0.9), **extra}), 0.05)
    assert v["verdict"] == "VOID" and cause in v["why"], v["why"]


def test_rule_8_s_bar_is_the_third_calibration_s():
    """Amendment 6: BAR = 3 x max(s94, the 30B's s(48)) from bench/q235/am6/calibration-3-deterministic.json, rounded to 3 figures."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("q235_verdict_pin", VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cal = json.loads((ROOT / "bench" / "q235" / "am6" / "calibration-3-deterministic.json").read_text())
    assert not cal["soft_any"] and all(cal["plants_registered"].values())
    assert mod.GA_BAR == round(cal["bar"], 4) == 0.0231 and set(mod.GA_PLANTS) == {"mean", "overwrite"}


def _verdict_a8(w, ref, band, bar=0.0231):
    """The verdict in this process with HELDOUT_BAND set (Amendment 8's registered value comes from its calibration)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("q235_verdict_a8", VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HELDOUT_BAND, mod.GA_BAR = band, bar
    return mod.judge(str(w), str(ref))


DX = ([0.5, 0.4, 0.3, 0.2], [0.6, 0.5, 0.4, 0.3])


def test_amendment_8_passes_rules_9_and_10_and_names_them(tmp_path):
    v = _verdict_a8(*_box_run(tmp_path / "a", a8={"dx": DX, "heldout": (2.0, 1.99, 1.98)}), 0.05)
    assert v["verdict"] == "PASS" and v["why"] == "rules 1-7, 9-10 hold", v["why"]
    assert v["rules"]["9_dx"]["ok"] and v["rules"]["9_dx"]["min"] == {"step": 0.2, "step1": 0.3}
    r10 = v["rules"]["10_heldout"]
    assert r10["ok"] and r10["band"] == 0.05 and abs(r10["rel_change"] + 0.01) < 1e-12
    b = _verdict_a8(*_box_run(tmp_path / "b", shape_b={"ga": (0.01, 0.02), "plants": (0.6, 0.9)},
                              a8={"dx": DX, "heldout": (2.0, 1.99, 1.98)}), 0.05, bar=0.05)
    assert b["verdict"] == "PASS" and b["why"] == "rules 1-10 hold", b["why"]


@pytest.mark.parametrize("dx", [([0.5, 0.0, 0.3, 0.2], DX[1]), (DX[0], [0.6, float("nan"), 0.4, 0.3]), (DX[0], [0.6, 0.5, None, 0.3])])
def test_a_layer_with_a_zero_nonfinite_or_missing_dx_fails_rule_9_and_the_strict_rules(tmp_path, dx):
    v = _verdict_a8(*_box_run(tmp_path, a8={"dx": dx, "heldout": (2.0, 1.99, 1.98)}), 0.05)
    assert v["verdict"] == "FAIL" and not v["rules"]["9_dx"]["ok"] and not v["rules"]["3_5_strict"]["ok"], v["why"]


@pytest.mark.parametrize("held", [(2.0, 2.05, 2.3), (2.0, 2.22, 2.0), (2.0, float("inf"), 1.98), (2.0, float("nan"), 1.9)])
def test_a_held_out_increase_beyond_the_cap_at_either_read_or_a_nonfinite_read_fails_rule_10(tmp_path, held):
    v = _verdict_a8(*_box_run(tmp_path, a8={"dx": DX, "heldout": held}), 0.05)
    assert v["verdict"] == "FAIL" and v["rules"]["10_heldout"]["ok"] is False, v["why"]


def test_rule_10_is_void_until_its_band_is_registered_and_a_missing_read_is_a_harness_error(tmp_path):
    v = _verdict_a8(*_box_run(tmp_path / "a", a8={"dx": DX, "heldout": (2.0, 1.99, 1.98)}), None)
    assert v["verdict"] == "VOID" and v["why"] == "not registered: Amendment 8's held-out band", v["why"]
    m = _verdict_a8(*_box_run(tmp_path / "b", a8={"dx": DX, "heldout": None}), 0.05)
    assert m["verdict"] == "VOID" and "without its registered held-out read" in m["why"], m["why"]


def test_records_from_before_amendment_8_read_as_before(tmp_path):
    """Runs 2 and 3 carry no amendment8: rules 9 and 10 do not apply, and their verdicts are unchanged (rules 1-7 / 1-8)."""
    a = _verdict_a8(*_box_run(tmp_path / "a"), None)
    assert a["verdict"] == "PASS" and a["why"] == "rules 1-7 hold" and "9_dx" not in a["rules"] and "10_heldout" not in a["rules"]
    b = _verdict_a8(*_box_run(tmp_path / "b", shape_b={"ga": (0.01, 0.02), "plants": (0.6, 0.9)}), None, bar=0.05)
    assert b["verdict"] == "PASS" and b["why"] == "rules 1-8 hold"


def test_rule_10_s_band_is_the_ruled_plain_cap():
    """Amendment 8, as ruled: a plain absolute cap of 0.10 on |rel_change| (a blow-up bound), not derived from the 30B leg."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("q235_verdict_band", VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.HELDOUT_BAND == 0.10


def test_rule_10_is_one_sided_a_drop_passes_and_only_an_increase_beyond_the_cap_fails(tmp_path):
    """As re-ruled: a blow-up bound only. -39 % (the 30B rehearsal read -38.6 %: two steps teach the shared template) and +9.5 %
    pass; +10.5 % fails; NaN fails."""
    for i, held in enumerate(((2.0, 1.23, 1.22), (2.0, 2.1, 2.19), (2.0, 1.9, 1.81))):
        v = _verdict_a8(*_box_run(tmp_path / f"in{i}", a8={"dx": DX, "heldout": held}), 0.10)
        assert v["verdict"] == "PASS" and v["rules"]["10_heldout"]["one_sided"], v["why"]
    assert v["rules"]["10_heldout"]["reported_train_rows"]["loss_after"] == 0.91          # the step's rows ride along, reported
    for i, held in enumerate(((2.0, 2.1, 2.21), (2.0, float("nan"), 1.9))):
        v = _verdict_a8(*_box_run(tmp_path / f"out{i}", a8={"dx": DX, "heldout": held}), 0.10)
        assert v["verdict"] == "FAIL" and v["rules"]["10_heldout"]["ok"] is False, v["why"]
