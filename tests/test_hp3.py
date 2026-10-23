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

Y = {"e4b": (0.7701, 0.7712, 0.7695, 0.7704, 0.7708), "unsloth": (0.7703, 0.7709, 0.7698, 0.7702, 0.7710),
     "axolotl": (0.7699, 0.7714, 0.7696, 0.7706, 0.7705)}


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
    conf["arms"][6]["adapter_roundtrip"] = 0.0101                     # unsloth seed 3411 (five e4b runs come first)
    out, _ = R.verdict([conf], R.g1(_cal()), _receipts())
    assert out["void"] == {"unsloth seed 3411": "6(b): |adapter_roundtrip| 0.010100 > 0.01"}
    assert out["pairs"]["e4b-unsloth"]["verdict"] == "NOT_READ" and out["pairs"]["e4b-axolotl"]["verdict"] == "PARITY"


def test_a_closed_framework_reads_close_on_its_pairs_and_runs_nothing():
    ev = _cal()
    ev["bases"][1]["digest"] = "1" * 64
    out, _ = R.verdict([_conf()], R.g1(ev), _receipts())
    assert out["pairs"]["e4b-unsloth"]["verdict"] == "CLOSE" and out["pairs"]["unsloth-axolotl"]["verdict"] == "CLOSE"
    assert out["pairs"]["e4b-unsloth"]["wording"] == "bases differ: outside this registration"
    assert "unsloth" not in out["y"] and out["three_way"]["verdict"] == "NOT_READ"
    assert out["three_way"]["closed_pairs"]["e4b-unsloth"]["digests"]["unsloth"] == "1" * 64      # the record names the pair and its digests


def test_a_run_without_its_in_loop_step_n_number_voids_with_its_reason():
    conf = _conf()
    del conf["arms"][0]["evaluator_minus_inloop_final"]              # e4b seed 3410
    out, _ = R.verdict([conf], R.g1(_cal()), _receipts())
    assert out["void"] == {"e4b seed 3410": "no evaluator_minus_inloop_final recorded (the in-loop step-N number is missing)"}
    assert out["pairs"]["e4b-unsloth"]["verdict"] == "NOT_READ" and out["pairs"]["unsloth-axolotl"]["verdict"] == "PARITY"
    assert out["three_way"]["unread_pairs"]["e4b-axolotl"].startswith("not read: VOID runs: e4b seed 3410")


def test_an_ordered_three_way_and_the_order_check():
    ys = {"e4b": (0.700, 0.701, 0.702, 0.700, 0.701), "unsloth": (0.760, 0.762, 0.761, 0.760, 0.763),
          "axolotl": (0.820, 0.823, 0.821, 0.822, 0.820)}
    out, code = R.verdict([_conf(ys)], R.g1(_cal()), _receipts())
    assert [out["pairs"][k]["verdict"] for k in ("e4b-unsloth", "e4b-axolotl", "unsloth-axolotl")] == ["BEATS", "BEATS", "BEATS"]
    assert out["three_way"]["verdict"] == "ORDERED" and code == R.EXIT_OK
    assert R._order([("e4b", "unsloth"), ("unsloth", "axolotl"), ("e4b", "axolotl")])
    assert not R._order([("e4b", "unsloth"), ("unsloth", "axolotl"), ("axolotl", "e4b")])


def test_mixed_verdicts_read_mixed():
    ys = {"e4b": (0.700, 0.701, 0.702, 0.700, 0.701), "unsloth": (0.760, 0.762, 0.761, 0.760, 0.763),
          "axolotl": (0.7601, 0.7622, 0.7608, 0.7603, 0.7627)}
    out, _ = R.verdict([_conf(ys)], R.g1(_cal()), _receipts())
    assert out["pairs"]["unsloth-axolotl"]["verdict"] in ("PARITY", "INCONCLUSIVE") and out["three_way"]["verdict"] == "MIXED"


def test_hp3s_paired_reading_is_hp1s_rule_at_hp1s_three_seeds():
    """Amendment 1 gives HP3 its own seed list; the verdict rule must be HP1's, unchanged. At HP1's three seeds every field equals
    hp1_reduce.paired_reading's, for a BEATS, a PARITY and an INCONCLUSIVE case."""
    h1 = R.H1.CONFIRMATION_SEEDS
    recs = [{"tag": "t", "own_base_only": False}]
    for a, b, sig, m in (((0.70, 0.701, 0.702), (0.76, 0.762, 0.761), 0.001, 0.005),
                         ((0.7701, 0.7712, 0.7695), (0.7703, 0.7709, 0.7698), 0.0007, 0.005),
                         ((0.770, 0.790, 0.760), (0.771, 0.770, 0.772), 0.01, 0.01),
                         ((0.7670, 0.7669, 0.7671), (0.7700, 0.7700, 0.7700), 0.001, 0.005)):   # BEATS and inside +-m: order decides
        ya, yb = dict(zip(h1, a)), dict(zip(h1, b))
        assert R.paired_reading("x", ya, recs, yb, recs, sig, m, seeds=h1) == R.H1.paired_reading("x", ya, recs, yb, recs, sig, m)
    ya, yb = dict(zip(h1, (0.7670, 0.7669, 0.7671))), dict(zip(h1, (0.77, 0.77, 0.77)))
    assert R.paired_reading("x", ya, recs, yb, recs, 0.001, 0.005, seeds=h1)["verdict"] == "BEATS"     # BEATS is tested before PARITY
    assert R.SEEDS == (3410, 3411, 3412, 3413, 3414)


def test_the_verdict_reports_the_seed_rho_and_reads_all_five_seeds():
    out, _ = R.verdict([_conf()], R.g1(_cal()), _receipts())
    p = out["pairs"]["e4b-unsloth"]
    assert p["headline"]["seeds"] == list(R.SEEDS) and p["headline"]["df"] == 4
    assert isinstance(p["rho_seed_reported"], float) and -1 <= p["rho_seed_reported"] <= 1
    assert R.seed_rho({1: 1.0, 2: 1.0}, {1: 0.5, 2: 0.7}, seeds=(1, 2)) is None          # no spread on one side: not reported
    assert R.seed_rho({1: 1.0, 2: 2.0, 3: 3.0}, {1: 2.0, 2: 4.0, 3: 6.0}, seeds=(1, 2, 3)) == 1.0


def test_the_power_table_reproduces_the_amendments_quoted_cells():
    """Amendment 1 quotes the evaluator's registered-order table; bench/hp3/hp3_power.py reproduces it at the evaluator's RNG seed."""
    import hp3_power as P
    t = P.table(reps=4000, seed=20261021)
    assert [round(t[(0.002, 0.0)][n]["PARITY"], 2) for n in (3, 5, 8)] == [0.40, 0.81, 0.94]
    assert [round(t[(0.001, 0.0)][n]["PARITY"], 2) for n in (3, 5, 8)] == [0.86, 0.92, 0.94]
    t4 = P.table(reps=4000, seed=20261021, counts=(4,))              # part 7's 4-seed column, its own stream
    assert [round(t4[c][4]["PARITY"], 2) for c in ((0.002, 0.0), (0.003, 0.5), (0.005, 0.82))] == [0.66, 0.64, 0.73]


def test_a_seed_plan_reads_its_prefix_and_names_the_reduced_power():
    """Amendment 1 part 7: a plan of 4 seeds reads 3410-3413 (df 3) and says it is a reduced-power read."""
    plan = {"n": 4, "seeds": [3410, 3411, 3412, 3413], "basis": "timing", "reason": "fit: 4 seeds at seed 3410's wall time 2400 s"}
    out, code = R.verdict([_conf()], R.g1(_cal()), _receipts(), plan)
    p = out["pairs"]["e4b-unsloth"]
    assert code == R.EXIT_OK and out["seed_count"] == 4 and out["seeds_read"] == [3410, 3411, 3412, 3413]
    assert p["headline"]["seeds"] == [3410, 3411, 3412, 3413] and p["headline"]["df"] == 3
    assert "reduced-power read on 4 seeds" in out["reduced_power"] and out["seed_plan_reason"].startswith("fit: 4 seeds")
    full, _ = R.verdict([_conf()], R.g1(_cal()), _receipts())
    assert full["seed_count"] == 5 and full["reduced_power"] is None


def test_a_host_limited_plan_reads_nothing_and_every_pair_is_not_read():
    plan = {"n": 0, "seeds": [], "basis": "timing", "reason": "host-limited: fewer than 3 seeds fit at seed 3410's wall time 9000 s"}
    out, code = R.verdict([_conf()], R.g1(_cal()), _receipts(), plan)
    assert code == R.EXIT_INCOMPLETE and out["three_way"]["verdict"] == "NOT_READ"
    assert all(p["verdict"] == "NOT_READ" and "host-limited" in p["wording"] for p in out["pairs"].values())


def test_a_seed_plan_outside_the_registered_counts_or_not_by_timing_is_refused():
    for bad in ({"n": 2, "seeds": [3410, 3411], "basis": "timing"}, {"n": 6, "seeds": list(range(3410, 3416)), "basis": "timing"},
                {"n": 4, "seeds": [3410, 3411, 3412, 3414], "basis": "timing"}, {"n": 4, "seeds": [3410, 3411, 3412, 3413], "basis": "loss"},
                {"n": "4", "seeds": [3410, 3411, 3412, 3413], "basis": "timing"}, {"seeds": [], "basis": "timing"}):
        try:
            R.plan_seeds(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def _bound(tmp_path, plan_obj, bind=True, mtimes=None):
    """eval_P.json carrying (or not) plan_sha256 of the plan file, as hp1_eval --seed-plan writes it; optional mtimes per file."""
    import hashlib
    import os
    ev, g1, plan = tmp_path / "eval_P.json", tmp_path / "g1.json", tmp_path / "seed_plan.json"
    g1.write_text(json.dumps(R.g1(_cal())))
    plan.write_text(json.dumps(plan_obj))
    conf = _conf()
    if bind:
        conf["plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    ev.write_text(json.dumps(conf))
    for f, t in (mtimes or {}).items():
        os.utime(tmp_path / f, (t, t))
    return ev, g1, plan


PLAN4 = {"n": 4, "seeds": [3410, 3411, 3412, 3413], "basis": "timing", "reason": "fit: 4 seeds at seed 3410's wall time 2400 s"}


def test_the_seed_plan_is_bound_by_content_so_a_rederivation_with_any_mtimes_still_verdicts(tmp_path):
    """Amendment 1 part 7 (the maintainer's content binding): the store keeps content, not mtimes, so a re-derivation whose
    checkout wrote the plan AFTER the evaluator output still reads; only the hash decides."""
    ev, g1, plan = _bound(tmp_path, PLAN4, mtimes={"seed_plan.json": 3000, "eval_P.json": 1000})
    out = tmp_path / "v.json"
    rc = R.main(["verdict", str(ev), "--g1", str(g1), "--seed-plan", str(plan), "--out", str(out)])
    assert rc in (R.EXIT_OK, R.EXIT_INCOMPLETE) and json.loads(out.read_text())["seed_count"] == 4   # no axolotl receipts: its pairs NOT_READ


def test_a_plan_edited_after_the_evaluators_pass_or_an_unbound_output_refuses(tmp_path):
    (tmp_path / "edit").mkdir()
    ev, g1, plan = _bound(tmp_path / "edit", PLAN4)
    plan.write_text(json.dumps(dict(PLAN4, n=5, seeds=[3410, 3411, 3412, 3413, 3414])))     # edited after the evaluator hashed it
    args = ["verdict", str(ev), "--g1", str(g1), "--seed-plan", str(plan), "--out", str(tmp_path / "v.json")]
    assert R.main(args) == R.EXIT_UNUSABLE
    (tmp_path / "unbound").mkdir()
    ev, g1, plan = _bound(tmp_path / "unbound", PLAN4, bind=False)                         # the evaluator never saw the plan
    assert R.main(["verdict", str(ev), "--g1", str(g1), "--seed-plan", str(plan), "--out", str(tmp_path / "v2.json")]) == R.EXIT_UNUSABLE
    d = json.loads(ev.read_text())
    d["plan_sha256"] = None                                                                 # null is a refusal too (rule 7)
    ev.write_text(json.dumps(d))
    assert R.main(["verdict", str(ev), "--g1", str(g1), "--seed-plan", str(plan), "--out", str(tmp_path / "v3.json")]) == R.EXIT_UNUSABLE


def test_a_bound_output_without_the_plan_refuses(tmp_path):
    """The inverted gap: the runner made a plan (the evaluator hashed it) but the reducer was not given it."""
    ev, g1, _ = _bound(tmp_path, PLAN4)
    assert R.main(["verdict", str(ev), "--g1", str(g1), "--out", str(tmp_path / "v.json")]) == R.EXIT_UNUSABLE
