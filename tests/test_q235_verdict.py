# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's milestone verdict (``bench/q235/instrument/q235_verdict.py``) on fabricated box records: one PASS, a FAIL for each rule,
and a VOID for each cause, each with its exit code."""
from __future__ import annotations

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


def _box_run(root, *, code="0", card=PRO6000, mode="science", rules=None, losses=(2.0, 1.9), repeat_loss=2.0, scale=1.01,
             peak=24.0 * (1 << 30)):
    """A fetched science run (q235_run.sh's layout) and a reference beside it; 4 layers of [2, 6, 8], top-2 of 8 experts."""
    g = torch.Generator().manual_seed(0)
    w, ref = root / "run", root / "ref"
    for d in (w / "receipts" / "box", w / "receipts" / "repeat", ref):
        d.mkdir(parents=True)
    mask = torch.ones(2, 6, dtype=torch.long)
    torch.save({"input_ids": torch.zeros(2, 6, dtype=torch.long), "attention_mask": mask, "labels": torch.zeros(2, 6, dtype=torch.long)},
               w / "receipts" / "tokens.pt")
    layers = [torch.randn(2, 6, 8, generator=g) for _ in range(4)]
    topk = [torch.randint(0, 8, (12, 2), generator=g) for _ in range(4)]
    for i, t in enumerate(layers):
        torch.save(t, ref / f"layer{i:03d}.pt")
        torch.save(t * scale, w / "receipts" / "box" / f"layer{i:03d}.pt")
    torch.save({"router_topk": topk}, ref / "reference.pt")
    json.dump({"loss": 2.01}, open(ref / "reference.json", "w"))
    torch.save({"router_topk": topk}, w / "receipts" / "box" / "nf4.pt")
    json.dump({"loss": 2.0}, open(w / "receipts" / "box" / "nf4.json", "w"))
    step = {"loss": losses[0], "elapsed_s": 60.0, "h2d_bytes_offload": 220 * (1 << 30), "max_memory_allocated": int(peak)}
    box = {"rules": rules if rules is not None else {k: True for k in RULES}, "frozen_sha256_after_both_steps_equal": True,
           "in_process": {"loss_0": 2.0, "loss_1": 2.0}, "step": step, "step1": {**step, "loss": losses[1]}}
    json.dump(box, open(w / "receipts" / "box" / "box.json", "w"))
    json.dump({"repeat": {"loss": repeat_loss}}, open(w / "receipts" / "repeat" / "box.json", "w"))
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
    ({"code": "12", "rules": {**{k: True for k in RULES}, "step1_every_A_changed": False}}, "3_5_strict"),
    ({"rules": {}}, "3_5_strict"),                                             # fail-closed: no rules recorded
    ({"repeat_loss": 2.0 * (1 + 3e-4)}, "6_repeat"),
    ({"scale": 1.4}, "7_bands"),
])
def test_each_rule_fails_on_its_own(tmp_path, kw, rule):
    rc, v = _verdict(*_box_run(tmp_path, **kw))
    assert (rc, v["verdict"]) == (1, "FAIL") and rule in v["why"], v["why"]
    assert [k for k, x in v["rules"].items() if not x["ok"]] == [rule], v["rules"]


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


def test_a_rehearsal_is_never_a_milestone(tmp_path):
    rc, v = _verdict(*_box_run(tmp_path, mode="a2000"))
    assert (rc, v["verdict"]) == (2, "VOID") and "not a science run" in v["why"]


def test_a_device_peak_far_from_the_estimate_is_reported_not_gated(tmp_path):
    rc, v = _verdict(*_box_run(tmp_path, peak=30.0 * (1 << 30)))
    assert (rc, v["verdict"]) == (0, "PASS") and v["reported"]["estimator_issue"] is True
