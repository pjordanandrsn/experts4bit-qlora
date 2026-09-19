"""P49 (bench/p49): the reducer applies the registered rules and nothing else."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p49_reduce", os.path.join(HERE, "..", "bench", "p49", "p49_reduce.py"))
p49 = importlib.util.module_from_spec(_spec)
sys.modules["p49_reduce"] = p49
_spec.loader.exec_module(p49)


def _row(arm, kl, qt="nf4", bs=64):
    return {"arm": arm, "kl_mean": kl, "top1_agreement": 0.8, "builder": f"loader_only_0:{qt}", "per_stratum": {},
            "engagement": {"quant_type": qt, "blocksize": bs, "quantized_types": [qt], "n_quantized": 1}}


def _kl(rows, scorer="prefill", self_kl=0.279):
    return {"scorer": {"name": "auto", "used": scorer}, "rows": rows, "not_measured": {},
            "reference_pass": {"self_consistency": {"kl_mean": self_kl, "passes": self_kl < 1e-2}}}


def _act(depths):
    arms = [{"arm": f"L{i:02d}_nf4", "layer": i, "quant_type": "nf4", "blocksize": 64, "rel_err_raw": raw, "rel_err_norm2": n2,
             "amplification_norm2_over_raw": n2 / raw, "cos_raw_mean": 0.99} for i, raw, n2 in depths]
    ref = {str(i): {"contrib_norm2_over_resid": 0.3, "rms_raw": 0.01 * (i + 1), "norm_gain_norm2_over_raw": 10.0 / (i + 1), "dense_over_moe": 1.0} for i, _, _ in depths}
    return {"arms": arms, "reference": {"layers": ref}, "not_measured": {}}


def test_int8_survives_and_amplification_story(tmp_path):
    rows = [_row("L00_nf4", 0.89), _row("L00_int8", 0.012, "int8"), _row("L00_fp8", 0.03, "fp8"), _row("L00_fp4", 0.95, "fp4"),
            _row("L00_nf4b32", 0.80, "nf4", 32), _row("L07_nf4", 0.44), _row("L14_nf4", 0.10), _row("L21_nf4", 0.014), _row("L27_nf4", 0.0056)]
    json.dump(_kl(rows), open(tmp_path / "gemma4fmt_kl.json", "w"))
    json.dump(_act([(0, 0.10, 0.60), (7, 0.11, 0.30), (14, 0.10, 0.12), (21, 0.09, 0.05), (27, 0.10, 0.03)]), open(tmp_path / "gemma4fmt_act.json", "w"))
    v = p49.reduce(str(tmp_path))
    assert v["P0_anchor"]["verdict"] == "HOLDS" and v["P1_int8"]["verdict"] == "HOLDS" and "SURVIVE" in v["P1_int8"]["reads"]
    assert v["P2_fp8"]["verdict"] == "HOLDS" and v["P3_fp4"]["verdict"] == "HOLDS" and v["P4_block32"]["verdict"] == "HOLDS"
    p5 = v["P5_amplification"]
    assert p5["verdict"] == "HOLDS" and p5["spearman_kl_vs_rel_norm2"] == 1.0 and p5["norm2_ratio_first_over_last"] == 20.0
    assert "P5_amplification: HOLDS" in p49.render_md(v)


def test_int8_fails_and_raw_cancellation_alternative(tmp_path):
    rows = [_row("L00_nf4", 0.89), _row("L00_int8", 0.35, "int8"), _row("L00_nf4b32", 0.30, "nf4", 32), _row("L07_nf4", 0.44), _row("L27_nf4", 0.0056), _row("L14_nf4", 0.10)]
    json.dump(_kl(rows), open(tmp_path / "gemma4fmt_kl.json", "w"))
    json.dump(_act([(0, 0.60, 0.61), (7, 0.30, 0.31), (14, 0.12, 0.12), (27, 0.03, 0.03)]), open(tmp_path / "gemma4fmt_act.json", "w"))
    v = p49.reduce(str(tmp_path))
    assert v["P1_int8"]["verdict"] == "REFUTED" and "high-precision" in v["P1_int8"]["reads"]
    assert v["P4_block32"]["verdict"] == "REFUTED" and v["P2_fp8"]["verdict"] == "NOT_READ" and v["P3_fp4"]["verdict"] == "NOT_READ"
    assert v["P5_amplification"]["verdict"].startswith("ALTERNATIVE")


def test_inconclusive_int8_missing_files_and_refused_scorer(tmp_path):
    json.dump(_kl([_row("L00_nf4", 0.89), _row("L00_int8", 0.1, "int8")]), open(tmp_path / "gemma4fmt_kl.json", "w"))
    v = p49.reduce(str(tmp_path))
    assert v["P1_int8"]["verdict"] == "INCONCLUSIVE" and v["P5_amplification"]["verdict"] == "NOT_READ"
    json.dump(_kl([_row("L00_nf4", 0.89)], scorer="decode", self_kl=0.279), open(tmp_path / "gemma4fmt_kl.json", "w"))
    v = p49.reduce(str(tmp_path))
    assert v["kl_verdict"].startswith("NOT_READ") and v["P0_anchor"]["verdict"] == "NOT_READ"
    assert p49.reduce(str(tmp_path / "nowhere"))["P0_anchor"]["verdict"] == "NOT_READ"
