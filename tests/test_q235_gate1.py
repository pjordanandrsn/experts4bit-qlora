# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235 gate 1's judge (``bench/q235/instrument/q235_gate1.py``) applies the registered rules mechanically: on fabricated rehearsal
records it passes a box within the bands whose repeat is within tolerance and whose two planted errors fail, voids the bands when a
plant passes them, fails a box out of band, a repeat outside tolerance or a box record without its STRICT rules, and calls a
missing record a harness error."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "bench" / "q235" / "instrument" / "q235_gate1.py"
RULES = ("step0_all_finite", "step0_every_B_nonzero", "step0_every_A_zero", "step0_every_B_changed", "step1_all_finite",
         "step1_every_A_nonzero", "step1_every_B_nonzero", "step1_every_A_changed", "step1_every_B_changed")


def _judge_module():
    spec = importlib.util.spec_from_file_location("q235_gate1", JUDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rehearsal(root, *, box_scale=1.01, plant_routing=None, plant_epilogue=None, repeat_loss=2.0, box_rules=True):
    """ref + box + repeat + two plants in rehearse.sh's layout; 4 layers of [2, 6, 8] outputs, top-2 of 8 experts."""
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[1, 4:] = 0
    torch.save({"input_ids": torch.zeros(2, 6, dtype=torch.long), "attention_mask": mask, "labels": torch.zeros(2, 6, dtype=torch.long)},
               root / "tokens.pt")
    ref = [torch.randn(2, 6, 8, generator=g) for _ in range(4)]
    topk = [torch.randint(0, 8, (12, 2), generator=g) for _ in range(4)]

    def write(name, layers, tk, loss, record=None):
        d = root / name
        d.mkdir()
        for i, t in enumerate(layers):
            torch.save(t, d / f"layer{i:03d}.pt")
        if name == "ref":
            torch.save({"router_topk": tk}, d / "reference.pt")
            json.dump({"loss": loss}, open(d / "reference.json", "w"))
        else:
            torch.save({"router_topk": tk}, d / "nf4.pt")
            json.dump({"loss": loss}, open(d / "nf4.json", "w"))
            json.dump(record, open(d / "box.json", "w"))

    clean = {"rules": {k: True for k in RULES} if box_rules else {}, "frozen_sha256_after_both_steps_equal": True,
             "in_process": {"loss_0": 2.0, "loss_1": 2.0, "loss_equal": True}}
    write("ref", ref, topk, 2.01)
    write("box", [t * box_scale for t in ref], topk, 2.0, clean)
    write("repeat", [t * box_scale for t in ref], topk, repeat_loss, {"repeat": {"loss": repeat_loss}})
    broken = [ref[0] * 1.01, ref[1] * 1.01, -ref[2], -ref[3]]                       # order-1 error from the planted layer on
    shifted = [tk if i < 2 else (tk + 1) % 8 for i, tk in enumerate(topk)]
    write("plant_routing", plant_routing or broken, shifted, 2.5, {**clean, "plant": {"kind": "routing", "layer": 2}})
    write("plant_epilogue", plant_epilogue or broken, topk, 2.7, {**clean, "plant": {"kind": "epilogue", "layer": 2}})
    return ref


def _run(root):
    r = subprocess.run([sys.executable, str(JUDGE), str(root), "--tokens", str(root / "tokens.pt"), "--out", str(root / "gate1.json")],
                       capture_output=True, text=True, cwd=ROOT)
    return r.returncode, json.load(open(root / "gate1.json"))


def test_a_clean_rehearsal_passes(tmp_path):
    _rehearsal(tmp_path)
    rc, g = _run(tmp_path)
    assert (rc, g["verdict"]) == (0, "PASS"), g["why"]
    assert all(p["fails_as_registered"] for p in g["plants"].values()) and g["repeat"]["within_tolerance"]


def test_a_plant_that_passes_the_bands_voids_them(tmp_path):
    """An inert plant (the clean box's own records under the epilogue plant's name) passes the bands: the bands are void."""
    import shutil
    _rehearsal(tmp_path)
    shutil.rmtree(tmp_path / "plant_epilogue")
    shutil.copytree(tmp_path / "box", tmp_path / "plant_epilogue")
    rc, g = _run(tmp_path)
    assert (rc, g["verdict"]) == (2, "BANDS_VOID") and "epilogue" in g["why"], g["why"]
    assert g["plants"]["routing"]["fails_as_registered"] and not g["plants"]["epilogue"]["fails_as_registered"]


def test_the_epilogue_plant_must_fail_by_its_registered_bound(tmp_path):
    """Its 'must fail' line is the maximum relative error: an overlap-only failure does not count for it."""
    j = _judge_module()
    overlap_only = {"rel_err_median": 0.1, "rel_err_max": 0.2, "router_overlap_min": 0.5}
    assert j.band_failures(overlap_only) and not j.plant_fails_as_registered("epilogue", overlap_only)
    assert j.plant_fails_as_registered("routing", overlap_only)


def test_a_box_out_of_band_fails(tmp_path):
    _rehearsal(tmp_path, box_scale=1.4)                                         # 0.40 relative: median above 0.25
    rc, g = _run(tmp_path)
    assert (rc, g["verdict"]) == (1, "FAIL") and "rel_err_median" in g["why"], g["why"]


def test_a_repeat_outside_the_floor_fails(tmp_path):
    _rehearsal(tmp_path, repeat_loss=2.0 * (1 + 3e-4))                           # 3e-4 relative > the 1e-4 floor, no spread
    rc, g = _run(tmp_path)
    assert (rc, g["verdict"]) == (1, "FAIL") and "repeat gap" in g["why"], g["why"]


def test_a_box_record_without_its_rules_fails_closed(tmp_path):
    _rehearsal(tmp_path, box_rules=False)
    rc, g = _run(tmp_path)
    assert (rc, g["verdict"]) == (1, "FAIL") and "step0_all_finite" in g["why"], g["why"]


def test_a_missing_record_is_a_harness_error(tmp_path):
    _rehearsal(tmp_path)
    (tmp_path / "repeat" / "box.json").unlink()
    rc, g = _run(tmp_path)
    assert (rc, g["verdict"]) == (3, "HARNESS_ERROR"), g
