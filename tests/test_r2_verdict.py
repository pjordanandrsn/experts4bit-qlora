# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2-Mixtral's milestone verdict (``bench/fam235/r2_verdict.py``): Amendment 1's depth-local band on Q235's verdict, on Q235's
fabricated box records (``tests/test_q235_verdict.py``'s 4-layer run: interior 1..3, plants at 2, mapped to 8x7B layers 8, 16 and 24).

- The self-test replays Amendment 1's numbers and reproduces the recorded VALIDATED.
- Unregistered, the tool VOIDs every run: no placeholder class or estimate is ever read.
- The band is per layer: u = 0.5 everywhere FAILs at layer 2 only (band 0.474) and passes layers 1 and 3 (0.528, 0.692), where
  Q235's single 0.775 passes the same box.
- One threshold both ways: a plant under band_2 is VOID (instrument), and a plant over it reads, though it is under Q235's 0.775."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "bench" / "fam235"
TOOL = ROOT / "bench" / "fam235" / "r2_verdict.py"
sys.path.insert(0, str(ROOT / "tests"))
from test_q235_verdict import RULES, _box_run, _verdict  # noqa: E402

PRO6000_RE = r"^NVIDIA RTX PRO 6000"          # the fabricated run's card; a stand-in registration for these tests only


@pytest.fixture
def r2(monkeypatch):
    spec = importlib.util.spec_from_file_location("r2_verdict", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REGISTERED", {"card_re": PRO6000_RE, "estimate_device_gib": 24.76, "estimate_link_gib": 237.94})
    return mod


def test_the_self_test_reproduces_amendment_1():
    r = subprocess.run([sys.executable, str(TOOL), "--self-test"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "ok   validation: VALIDATED, as recorded" in r.stdout, r.stdout + r.stderr


RTX5090 = "NVIDIA GeForce RTX 5090, 32607, 575.64.05, 5, 5, 16, 16, 575.00 W"


def test_the_registration_is_the_runner_s_science_class(tmp_path):
    """R2 Amendment 3: the verdict's class is the one r2_run.sh seals; an RTX 5090 run reads, any other class FAILs rule 1."""
    import re
    spec = importlib.util.spec_from_file_location("r2_verdict_registered", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sealed = re.search(r"^  science\) CARD_RE='([^']*)'", (F / "r2_run.sh").read_text(), re.M).group(1)
    assert mod.REGISTERED["card_re"] == sealed == "^NVIDIA GeForce RTX 5090"

    def verdict(card, sub):
        w, ref = _box_run(tmp_path / sub, card=card)
        r = subprocess.run([sys.executable, str(TOOL), str(w), "--ref", str(ref), "--out", str(w / "v.json")],
                           capture_output=True, text=True, cwd=ROOT)
        return r.returncode, json.load(open(w / "v.json"))
    rc, v = verdict(RTX5090, "rtx5090")
    assert (rc, v["verdict"]) == (0, "PASS") and v["rules"]["1_completed_on_class"]["ok"], v["why"]
    rc, v = verdict("NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 575.64.05, 5, 5, 16, 16, 600.00 W", "pro6000")
    assert (rc, v["verdict"]) == (1, "FAIL") and "1_completed_on_class" in v["why"], v["why"]


def test_a_clean_box_passes_with_its_margins_reported(tmp_path, r2):
    v = r2.judge(*_box_run(tmp_path))
    assert v["verdict"] == "PASS", v["why"]
    b = v["rules"]["7_band"]
    assert b["interior"] == [1, 3] and b["maps_to"] == {1: 8, 2: 16, 3: 24} and b["shallow_band"] == [1, 1]
    assert abs(b["margin_min"] - b["band"][2] / 0.01) < 1e-2 and b["margin_min_layer"] == 2
    assert all(p["fails_band"] and p["band_at_layer"] == b["band"][2] for p in v["instrument"]["plants"].values())


def test_the_band_is_per_layer_where_q235_s_single_band_passes(tmp_path, r2):
    w, ref = _box_run(tmp_path, scale=1.5)                                      # u = 0.5 on every layer
    v = r2.judge(w, ref)
    assert v["verdict"] == "FAIL" and v["rules"]["7_band"]["layers_over"] == [2], v["rules"]["7_band"]
    assert [k for k, x in v["rules"].items() if x["ok"] is False] == ["7_band"]
    assert _verdict(w, ref)[1]["verdict"] == "PASS"                             # Q235's 0.775 on the same box


def test_one_threshold_both_ways_on_the_plants(tmp_path, r2):
    under = r2.judge(*_box_run(tmp_path / "under", plant_kick=0.03))
    p = under["instrument"]["plants"]["routing"]
    assert p["u_at_layer"] < p["band_at_layer"]
    assert under["verdict"] == "VOID" and under["why"].startswith("instrument: the routing plant passes the band"), under["why"]
    w, ref = _box_run(tmp_path / "over", plant_kick=0.05)
    over = r2.judge(w, ref)
    p = over["instrument"]["plants"]["routing"]
    assert p["band_at_layer"] < p["u_at_layer"] < 0.775 and over["verdict"] == "PASS", over["why"]
    assert _verdict(w, ref)[1]["verdict"] == "VOID"                             # under Q235's band the same plant is blind


@pytest.mark.parametrize("kw,cause", [
    ({"null_scale": 1.01}, "instrument: the null is not the box"),
    ({"plant_kick": 0.0}, "instrument: the routing plant did not engage"),
    ({"plant_at": 3}, "instrument: the routing plant is not at layer 2"),
    ({"null_scale": 1.01, "scale": 1.5}, "instrument: the null is not the box"),      # VOID before the band's FAIL
])
def test_the_instrument_checks_void_before_any_band(tmp_path, r2, kw, cause):
    v = r2.judge(*_box_run(tmp_path, **kw))
    assert v["verdict"] == "VOID" and v["why"].startswith(cause), v["why"]


def test_a_strict_failure_is_a_fail_with_the_band_not_read(tmp_path, r2):
    v = r2.judge(*_box_run(tmp_path, code="12", rules={**{k: True for k in RULES}, "step0_every_B_changed": False}, repeat=False))
    assert v["verdict"] == "FAIL" and v["rules"]["7_band"]["ok"] is None and v["rules"]["3_5_strict"]["ok"] is False


def test_a_refusal_is_still_q235_s_void(tmp_path, r2):
    w, ref = _box_run(tmp_path)
    (w / "TC1_EXIT_CODE.n0nce").write_text("18\n")
    (w / "REFUSAL").write_text("host rate floor\n")
    v = r2.judge(w, ref)
    assert v["verdict"] == "VOID" and v["why"].startswith("refusal: host rate floor"), v["why"]


def test_a_changed_numbers_file_is_refused(r2, monkeypatch):
    monkeypatch.setattr(r2, "NUMBERS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="not the file Amendment 1 registered"):
        r2.band_profile(56)
