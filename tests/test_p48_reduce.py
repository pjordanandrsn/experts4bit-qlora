"""P48 (bench/p48): the per-layer reducer applies the registered rules and nothing else."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p48_reduce", os.path.join(HERE, "..", "bench", "p48", "p48_reduce.py"))
p48 = importlib.util.module_from_spec(_spec)
sys.modules["p48_reduce"] = p48
_spec.loader.exec_module(p48)


def _row(i, kl):
    return {"arm": f"L{i:02d}", "kl_mean": kl, "top1_agreement": 0.8, "builder": f"loader_only_{i}", "per_stratum": {},
            "engagement": {"builder": f"loader_only_{i}", "n_quantized": 1, "n_unquantized": 29, "expected_quantized": 1, "expected_unquantized": 29}}


def _receipt(rows, scorer="prefill", self_kl=0.279):
    return {"scorer": {"name": "auto", "used": scorer}, "rows": rows, "not_measured": {},
            "reference_pass": {"self_consistency": {"kl_mean": self_kl, "passes": self_kl < 1e-2}}}


def _probe(tmp_path, errs):
    p = tmp_path / "probe.jsonl"
    with open(p, "w") as f:
        for i, e in enumerate(errs):
            for proj in ("gate_up_proj", "down_proj"):
                f.write(json.dumps({"layer": i, "proj": proj, "nf4_rel_err_med": e, "expert_amax_over_rms_max": 10 + i, "rms": 0.03}) + "\n")
    return str(p)


def test_concentrated_first_half_profile(tmp_path):
    kls = [0.30, 0.25, 0.20, 0.10, 0.05] + [0.02] * 10 + [0.005] * 15      # sum 1.175
    json.dump(_receipt([_row(i, k) for i, k in enumerate(kls)]), open(tmp_path / "gemma4layer_kl.json", "w"))
    v = p48.reduce(str(tmp_path), _probe(tmp_path, [0.093] * 30))
    assert v["complete"] and v["P1_additivity"]["verdict"] == "HOLDS"
    assert v["P2_concentration"]["verdict"] == "HOLDS" and v["P2_concentration"]["P2b_first_half"] == "HOLDS"
    assert v["P2_concentration"]["top5"][0]["layer"] == 0
    assert v["P3_model_bound"]["verdict"] == "HOLDS" and v["P3_model_bound"]["min_kl"] == 0.005
    # the probe is flat -> Spearman is undefined/0 -> weights do not predict it
    assert v["P4_probe"]["verdict"] in ("HOLDS", "NOT_READ")
    md = p48.render_md(v)
    assert "P2_concentration: HOLDS" in md and "| 0 | 0.3000" in md


def test_spread_profile_and_probe_that_predicts(tmp_path):
    kls = [0.036] * 30                                                        # sum 1.08, spread
    json.dump(_receipt([_row(i, k) for i, k in enumerate(kls)]), open(tmp_path / "gemma4layer_kl.json", "w"))
    v = p48.reduce(str(tmp_path), None)
    assert v["P2_concentration"]["verdict"] == "REFUTED" and v["P2_concentration"]["P2b_first_half"] == "REFUTED"
    assert v["P3_model_bound"]["verdict"] == "REFUTED"
    assert v["P4_probe"]["verdict"] == "NOT_READ"
    kls = [0.3 - 0.01 * i for i in range(30)]                                 # monotone in layer, and so is the probe
    json.dump(_receipt([_row(i, k) for i, k in enumerate(kls)]), open(tmp_path / "gemma4layer_kl.json", "w"))
    v = p48.reduce(str(tmp_path), _probe(tmp_path, [0.2 - 0.005 * i for i in range(30)]))
    assert v["P4_probe"]["verdict"] == "REFUTED" and v["P4_probe"]["spearman_kl_vs_nf4_rel_err"] > 0.99


def test_partial_and_refused(tmp_path):
    json.dump(_receipt([_row(i, 0.05) for i in range(10)]), open(tmp_path / "gemma4layer_kl.json", "w"))
    v = p48.reduce(str(tmp_path))
    assert not v["complete"] and v["P1_additivity"]["verdict"] == "NOT_READ" and v["P2_concentration"]["verdict"] == "NOT_READ"
    assert v["P3_model_bound"]["verdict"].startswith("NOT_READ")
    json.dump(_receipt([_row(0, 0.05)], scorer="decode", self_kl=0.279), open(tmp_path / "gemma4layer_kl.json", "w"))
    assert p48.reduce(str(tmp_path))["verdict"].startswith("NOT_READ")
    assert p48.reduce(str(tmp_path / "nowhere")) == {"verdict": "NOT_READ: no gemma4layer_kl.json"}
    assert p48.spearman([1, 2, 3], [3, 2, 1]) == -1.0 and p48.spearman([1, 2], [1, 2]) is None
