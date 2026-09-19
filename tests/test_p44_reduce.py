"""The P44 reducer applies the registered rules and nothing else: the two-text K8 rule through k8_gate.verdict,
the KL reading rule (every stratum <= 1.10x NF4 AND pooled delta <= 0.005), the P4 control, the P3 tail statistic.
A missing row is NOT_READ, never a pass."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p44_reduce", os.path.join(HERE, "..", "bench", "p44", "p44_reduce.py"))
p44 = importlib.util.module_from_spec(_spec)
sys.modules["p44_reduce"] = p44
_spec.loader.exec_module(p44)


def _k8(ppl, src, sha="a" * 64, steps=2048):
    return {"k8": "ppl", "ppl": ppl, "text_sha": sha, "steps": steps, "ppl_source": src}


def _write(d, name, obj):
    with open(os.path.join(d, name), "w") as f:
        json.dump(obj, f)


def test_k8_two_text_rule_int4all_two_sided_and_calibexp_one_sided(tmp_path):
    d = str(tmp_path)
    _write(d, "olmoe_ppl_nf4_wikitext.json", _k8(10.000, "wikitext"))
    _write(d, "olmoe_ppl_nf4_c4val1.json", _k8(20.000, "c4val1", sha="b" * 64))
    _write(d, "olmoe_ppl_int4all_wikitext.json", _k8(10.030, "wikitext"))
    _write(d, "olmoe_ppl_int4all_c4val1.json", _k8(20.063, "c4val1", sha="b" * 64))       # over budget on c4val1
    _write(d, "olmoe_ppl_calibexp_all_wikitext.json", _k8(9.990, "wikitext"))               # same-domain improvement
    _write(d, "olmoe_ppl_calibexp_all_c4val1.json", _k8(20.040, "c4val1", sha="b" * 64))
    v = p44.k8_verdicts(d)
    assert v["int4all"]["verdict"] == "FAIL" and v["int4all"]["deltas"]["c4val1"] == pytest.approx(0.063, abs=1e-6)
    # calibrated: an improvement on ONE text (the calibration domain) is not corroborated -> FAIL by the clause
    assert v["calibexp_all"]["verdict"] == "FAIL"
    assert any("improvement needs the same sign" in ln for ln in v["calibexp_all"]["lines"])
    # both texts within +0.05 and no improvement -> PASS one-sided
    _write(d, "olmoe_ppl_calibexp_all_wikitext.json", _k8(10.020, "wikitext"))
    v = p44.k8_verdicts(d)
    assert v["calibexp_all"]["verdict"] == "PASS" and v["calibexp_all"]["licenses"]


def test_k8_missing_text_is_not_read(tmp_path):
    d = str(tmp_path)
    _write(d, "olmoe_ppl_nf4_wikitext.json", _k8(10.0, "wikitext"))
    _write(d, "olmoe_ppl_int4all_wikitext.json", _k8(10.01, "wikitext"))
    v = p44.k8_verdicts(d)
    assert v["int4all"]["verdict"] == "NOT_READ" and "c4val1" in v["int4all"]["reason"]


def _kl_row(arm, pooled, strata):
    return {"arm": arm, "kl_mean": pooled,
            "per_stratum": {s: {"kl_mean": k, "n_tokens_scored": 100} for s, k in strata.items()}}


def test_kl_reading_rule_every_stratum_and_pooled(tmp_path):
    ctrl = _kl_row("nf4", 0.0100, {"general": 0.0080, "technical": 0.0090, "code": 0.0120, "longctx": 0.0100})
    good = _kl_row("int4_r1epi", 0.0104, {"general": 0.0085, "technical": 0.0095, "code": 0.0130, "longctx": 0.0105})
    r = p44.reading_rule(ctrl, good)
    assert r["licenses"] and r["licensed_by"] == "kl-vs-bf16"
    # one stratum over 1.10x refuses even when pooled is fine
    bad = _kl_row("calattn_r1epi", 0.0104, {"general": 0.0085, "technical": 0.0095, "code": 0.0133, "longctx": 0.0105})
    r = p44.reading_rule(ctrl, bad)
    assert not r["licenses"] and not r["per_stratum"]["code"]["within_band"] and r["licensed_by"] is None
    # pooled delta over 0.005 refuses even when every stratum is within band (control large)
    ctrl2 = _kl_row("nf4", 0.100, {"general": 0.1, "technical": 0.1, "code": 0.1, "longctx": 0.1})
    cand2 = _kl_row("x", 0.106, {"general": 0.106, "technical": 0.106, "code": 0.106, "longctx": 0.106})
    assert not p44.reading_rule(ctrl2, cand2)["licenses"]
    # P4: r1epi must equal nf4 to < 1e-4 on every stratum
    r1 = _kl_row("r1epi", 0.0100, {"general": 0.00800001, "technical": 0.0090, "code": 0.0120, "longctx": 0.0100})
    assert p44.p4_control(ctrl, r1)["holds"]
    r1b = _kl_row("r1epi", 0.0100, {"general": 0.0082, "technical": 0.0090, "code": 0.0120, "longctx": 0.0100})
    assert not p44.p4_control(ctrl, r1b)["holds"]


def test_kl_verdicts_read_files_and_note_holes(tmp_path):
    d = str(tmp_path)
    ctrl = _kl_row("nf4", 0.01, {"general": 0.01, "code": 0.01})
    _write(d, "gemma4_kl.json", {"scorer": {"name": "decode"}, "rows": [ctrl, _kl_row("int4_r1epi", 0.0102, {"general": 0.0102, "code": 0.0101})],
                                 "not_measured": {"calattn_r1epi": {"error": "RuntimeError: boom"}}})
    v = p44.kl_verdicts(d)
    assert v["gemma4"]["int4_r1epi"]["licenses"]
    assert v["gemma4"]["calattn_r1epi"] == {"verdict": "NOT_READ"}
    assert v["gemma4"]["P4_r1epi_equals_nf4"] == {"verdict": "NOT_READ"}
    assert "calattn_r1epi" in v["gemma4"]["not_measured"]
    assert "gptoss" not in v


def _census_rows(n_layers, n_experts, heavy):
    rows = []
    for layer in range(n_layers):
        for e in range(n_experts):
            for role in ("gu", "dn"):
                err = 100.0 if (layer, e) in heavy else 1.0
                rows.append({"layer": layer, "expert": e, "role": role, "rows": 64, "recipe_method": "gptq",
                             "rtn": {"sq_err_act": err * 2, "rel_act": 0.1, "rel_frob": 0.1},
                             "gptq": {"sq_err_act": err, "rel_act": 0.05, "rel_frob": 0.1}})
    # one never-routed expert: no Hessian, no activation error
    rows.append({"layer": 0, "expert": n_experts, "role": "gu", "rows": 0, "recipe_method": "rtn",
                 "rtn": {"sq_err_act": None, "rel_act": None, "rel_frob": 0.1}, "gptq": None})
    return rows


def test_census_tail_statistic_recipe_vs_rtn(tmp_path):
    d = str(tmp_path)
    heavy = {(0, 0), (1, 0), (2, 0), (3, 0)}                     # 4 of 80 experts carry ~ 4*100 of (4*100 + 76*1)
    _write(d, "census_granite.json", {"rows": _census_rows(8, 10, heavy), "layers_censused": list(range(8))})
    v = p44.census_verdict(d, "granite")
    assert v["method"] == "recipe" and v["n_experts"] == 80 and v["roles_without_hessian"] == 1
    assert v["experts_for_share"] == 3 and v["verdict"] == "P3_HOLDS"          # 3 heavy experts = 300 >= 0.5 * 476
    # flat errors: half the experts needed -> refuted
    _write(d, "census_mixtral.json", {"rows": _census_rows(4, 8, set()), "layers_censused": list(range(4))})
    v = p44.census_verdict(d, "mixtral")
    assert v["method"] == "rtn" and v["fraction_for_share"] == pytest.approx(0.5) and v["verdict"] == "P3_REFUTED"
    assert p44.census_verdict(d, "olmoe" if False else "granite")["verdict"] == "P3_HOLDS"
    assert p44.census_verdict(str(tmp_path / "nowhere"), "granite")["verdict"] == "NOT_READ"


def test_render_and_reduce_run_on_an_empty_dir(tmp_path):
    v = p44.reduce(str(tmp_path))
    md = p44.render_md(v)
    assert "NOT_READ" in md and "P44-b" in md
