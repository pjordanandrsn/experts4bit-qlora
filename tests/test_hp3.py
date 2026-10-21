"""bench/hp3 (HP3-PREREG.md): G1' on HPN read 8's real L8 records, the per-run VOIDs and the three-way read."""

import copy
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "hp3"))
import hp3_reduce as R  # noqa: E402

READ8 = REPO / "bench" / "hpn" / "receipts" / "hpn-a2000-read8"
PROJ = READ8 / "proj"


def _base(eval_name, arm_name, tag):
    """A real L8 base (HPN read 8's public projection) as hp1_eval writes it, with its per-row gaps against the real in-loop step 0."""
    b = json.loads((PROJ / "hpn" / eval_name).read_text())["bases"][0]
    i0 = [c for c in json.loads((PROJ / arm_name).read_text())["eval_curve"] if c["step"] == 0][0]["row_losses"]
    return dict(copy.deepcopy(b), zero_adapter_row_gaps={tag: [round(e - i, 6) for e, i in zip(b["zero_adapter"]["row_losses"], i0)]})


def _cal():
    """e4b and Unsloth as read at L8 on one build; the axolotl slot reuses e4b's second in-loop arm on e4b's base (its stand-in)."""
    return {"bases": [_base("eval_e4b_L8_1.json", "qwen3hpnL8_e4b_hpn_e4b_a.json", R.CAL_TAGS["e4b"]),
                      _base("eval_uns_L8.json", "qwen3hpnL8_unsloth_hpn_uns.json", R.CAL_TAGS["unsloth"]),
                      _base("eval_e4b_L8_2.json", "qwen3hpnL8_e4b_hpn_e4b_b.json", R.CAL_TAGS["axolotl"])]}


def test_g1_passes_on_real_identical_bases_and_reports_hp2s_delta0_as_the_in_loop_term():
    """HPN read 8 at L8: e4b's and Unsloth's bases are byte-identical, so G1' passes. HP2's delta0 is still reported, and on these
    records it is HPN's 0.0052: the in-loop term, the number HP2's G1 would have STOPped on."""
    rec = R.g1(_cal())
    assert rec["verdict"] == "PASS" and rec["readable_pairs"] == ["e4b-unsloth", "e4b-axolotl", "unsloth-axolotl"]
    assert all(rec["frameworks"][f]["g1"] == "PASS" for f in R.FRAMEWORKS)
    assert rec["frameworks"]["unsloth"]["max_row_diff"] == 0 and rec["frameworks"]["unsloth"]["same_digest"]
    hpn = json.loads((READ8 / "hpn" / "result.json").read_text())["base_term"]["8"]
    d0 = rec["delta0_reported"]["e4b-unsloth"]
    assert abs(d0["delta0"]) == hpn["abs_delta0_E"] and d0["se"] == hpn["se_delta0_E"] and d0["term"].startswith("in-loop")
    assert rec["home_field"]["closest"] in R.FRAMEWORKS and rec["scope"] == R.SCOPE and rec["labels"] == R.LABELS


def test_one_row_off_by_more_than_the_n0_band_closes_that_framework_only():
    ev = _cal()
    ev["bases"][2]["zero_adapter"]["row_losses"][17] += 2e-5
    rec = R.g1(ev)
    assert rec["frameworks"]["axolotl"]["g1"] == "CLOSE" and "bases differ: outside this registration" in rec["frameworks"]["axolotl"]["reason"]
    assert rec["verdict"] == "PARTIAL" and rec["readable_pairs"] == ["e4b-unsloth"]
    ev = _cal()
    ev["bases"][2]["zero_adapter"]["row_losses"][17] += 9e-6                       # inside 1e-5, but the row mean moves
    ev["bases"][2]["zero_adapter"]["row_mean"] += 1e-6
    assert R.g1(ev)["frameworks"]["axolotl"]["g1"] == "CLOSE"


def test_a_different_digest_closes_and_both_closing_ends_the_box_with_the_record_written(tmp_path):
    ev = _cal()
    for b in ev["bases"][1:]:
        b["digest"] = "0" * 64
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(ev))
    out = tmp_path / "g1.json"
    r = subprocess.run([sys.executable, str(REPO / "bench" / "hp3" / "hp3_reduce.py"), "g1", str(p), "--out", str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == R.EXIT_G1_CLOSE and "HP3_G1=CLOSE" in r.stdout
    rec = json.loads(out.read_text())
    assert rec["verdict"] == "CLOSE" and rec["readable_pairs"] == []
    assert "the dequantized digests differ" in rec["frameworks"]["unsloth"]["reason"]


def test_a_base_that_does_not_match_its_own_training_run_is_unusable(tmp_path):
    ev = _cal()
    ev["bases"][1]["digest_matches_training_run"] = False
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(ev))
    r = subprocess.run([sys.executable, str(REPO / "bench" / "hp3" / "hp3_reduce.py"), "g1", str(p), "--out", str(tmp_path / "g.json")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == R.EXIT_UNUSABLE and "UNUSABLE" in r.stdout


def test_an_absent_framework_reads_absent_and_its_pairs_not_read():
    ev = _cal()
    del ev["bases"][2]
    rec = R.g1(ev)
    assert rec["frameworks"]["axolotl"]["g1"] == "ABSENT" and rec["verdict"] == "PARTIAL"
    out, code = R.verdict([_conf()], rec, _receipts())
    assert out["pairs"]["e4b-axolotl"]["verdict"] == "NOT_READ" and out["pairs"]["e4b-unsloth"]["verdict"] != "NOT_READ"
    assert out["three_way"]["verdict"] == "NOT_READ" and code == R.EXIT_INCOMPLETE


# ---- the verdict: synthetic confirmation runs in hp1_eval's arm format

Y = {"e4b": (0.7701, 0.7712, 0.7695), "unsloth": (0.7703, 0.7709, 0.7698), "axolotl": (0.7699, 0.7714, 0.7696)}


def _conf(ys=Y, rt=0.0004):
    arms = []
    for fw, vals in ys.items():
        for s, y in zip(R.SEEDS, vals):
            arms.append({"tag": f"{R.RUN_PREFIX[fw]}{s}", "status": "SCORED", "evaluator": {"row_mean": y}, "adapter_roundtrip": rt,
                         "evaluator_minus_inloop_final": 0.001, "order": {"order_seed": s, "order_sha": f"sha{s}"}, "own_base_only": True})
    return {"arms": arms}


def _off():
    return {"kernels": "off", "config_experts_implementation": "eager", "plugins": [],
            "cfg_lora_kernels": {k: False for k in R.AX.LORA_KERNELS}}


def _receipts(ax=None):
    return {f"{R.RUN_PREFIX['axolotl']}{s}": {"tag": f"{R.RUN_PREFIX['axolotl']}{s}", "axolotl": ax or _off()} for s in R.SEEDS}


def test_three_close_frameworks_read_three_way_parity_with_the_scope_and_labels_on_every_pair():
    out, code = R.verdict([_conf()], R.g1(_cal()), _receipts())
    assert code == R.EXIT_OK and out["three_way"]["verdict"] == "THREE_WAY_PARITY" and out["three_way"]["pairs_read"] == 3
    for p in out["pairs"].values():
        assert p["verdict"] == "PARITY" and p["scope"] == R.SCOPE and p["deltaN_reported"]["term"].startswith("in-loop")
    assert out["labels"] == R.LABELS


def test_axolotl_on_its_kernel_path_or_without_a_receipt_voids_its_runs():
    on = dict(_off(), kernels="on", config_experts_implementation="scattermoe", plugins=[R.AX.KERNELS_PLUGIN])
    for receipts in (_receipts(on), {}):
        out, code = R.verdict([_conf()], R.g1(_cal()), receipts)
        assert out["pairs"]["e4b-axolotl"]["verdict"] == "NOT_READ" and out["pairs"]["unsloth-axolotl"]["verdict"] == "NOT_READ"
        assert out["pairs"]["e4b-unsloth"]["verdict"] == "PARITY" and code == R.EXIT_INCOMPLETE
        assert all(k.startswith("axolotl") for k in out["void"])
    out, _ = R.verdict([_conf()], R.g1(_cal()), _receipts(on))
    assert "engagement" in out["void"]["axolotl seed 3410"]


def test_a_round_trip_outside_the_match_band_voids_the_run():
    conf = _conf()
    conf["arms"][4]["adapter_roundtrip"] = 0.0101                     # unsloth seed 3411
    out, _ = R.verdict([conf], R.g1(_cal()), _receipts())
    assert out["void"] == {"unsloth seed 3411": "6(b): |adapter_roundtrip| 0.010100 > 0.01"}
    assert out["pairs"]["e4b-unsloth"]["verdict"] == "NOT_READ" and out["pairs"]["e4b-axolotl"]["verdict"] == "PARITY"


def test_a_closed_framework_reads_close_on_its_pairs_and_runs_nothing():
    ev = _cal()
    ev["bases"][1]["digest"] = "1" * 64
    out, _ = R.verdict([_conf()], R.g1(ev), _receipts())
    assert out["pairs"]["e4b-unsloth"]["verdict"] == "CLOSE" and out["pairs"]["unsloth-axolotl"]["verdict"] == "CLOSE"
    assert out["pairs"]["e4b-unsloth"]["wording"] == "bases differ: outside this registration"
    assert "unsloth" not in out["y"]


def test_an_ordered_three_way_and_the_order_check():
    ys = {"e4b": (0.700, 0.701, 0.702), "unsloth": (0.760, 0.762, 0.761), "axolotl": (0.820, 0.823, 0.821)}
    out, code = R.verdict([_conf(ys)], R.g1(_cal()), _receipts())
    assert [out["pairs"][k]["verdict"] for k in ("e4b-unsloth", "e4b-axolotl", "unsloth-axolotl")] == ["BEATS", "BEATS", "BEATS"]
    assert out["three_way"]["verdict"] == "ORDERED" and code == R.EXIT_OK
    assert R._order([("e4b", "unsloth"), ("unsloth", "axolotl"), ("e4b", "axolotl")])
    assert not R._order([("e4b", "unsloth"), ("unsloth", "axolotl"), ("axolotl", "e4b")])


def test_mixed_verdicts_read_mixed():
    ys = {"e4b": (0.700, 0.701, 0.702), "unsloth": (0.760, 0.762, 0.761), "axolotl": (0.7601, 0.7622, 0.7608)}
    out, _ = R.verdict([_conf(ys)], R.g1(_cal()), _receipts())
    assert out["pairs"]["unsloth-axolotl"]["verdict"] in ("PARITY", "INCONCLUSIVE") and out["three_way"]["verdict"] == "MIXED"
