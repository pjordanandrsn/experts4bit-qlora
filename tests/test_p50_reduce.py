"""P50 (bench/p50): the keep-k curve reducer applies the registered rules and nothing else."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "bench", *rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p50 = _load("p50_reduce", ("p50", "p50_reduce.py"))
ss = _load("serve_stack_p50", ("p44", "serve_stack.py"))

GB = 2 ** 30


def _row(k, kl, top1, per_nf4_gb=0.43, per_bf16_gb=1.52):
    nq, nu = 30 - k, k
    eb = {"quantized": int(nq * per_nf4_gb * GB), "bf16": int(nu * per_bf16_gb * GB)}
    return {"arm": f"K{k:02d}", "kl_mean": kl, "top1_agreement": top1, "per_stratum": {},
            "engagement": {"expert_bytes": eb, "expert_bytes_total_gb": round((eb["quantized"] + eb["bf16"]) / GB, 3),
                           "n_quantized": nq, "n_unquantized": nu}}


def _rec(rows, scorer="prefill", self_kl=0.279):
    return {"scorer": {"name": "auto", "used": scorer}, "rows": rows, "not_measured": {},
            "reference_pass": {"self_consistency": {"kl_mean": self_kl, "passes": self_kl < 1e-2}}}


def test_expensive_remedy_curve(tmp_path):
    rows = [_row(5, 0.62, 0.74), _row(10, 0.30, 0.82), _row(15, 0.133, 0.89), _row(20, 0.041, 0.958), _row(24, 0.017, 0.968)]
    json.dump(_rec(rows), open(tmp_path / "gemma4keep_kl.json", "w"))
    v = p50.reduce(str(tmp_path))
    assert v["P1_anchor"]["verdict"] == "HOLDS"
    assert v["P2_curve"]["verdict"] == "HOLDS" and v["P2_curve"]["monotone"]
    assert v["P3_k20"]["verdict"] == "HOLDS" and v["passing_k"] == 20
    assert v["P4_cost"]["verdict"] == "HOLDS" and v["P4_cost"]["ratio"] > 2.5 and "4-bit advantage" in v["P4_cost"]["reads"]
    assert v["P5_top1"]["verdict"] == "HOLDS"
    assert "P3_k20: HOLDS" in p50.render_md(v)


def test_only_k24_reaches_the_floor_and_cheap_remedy(tmp_path):
    rows = [_row(5, 0.62, 0.74), _row(15, 0.133, 0.89), _row(20, 0.070, 0.94), _row(24, 0.030, 0.96, per_bf16_gb=0.5)]
    json.dump(_rec(rows), open(tmp_path / "gemma4keep_kl.json", "w"))
    v = p50.reduce(str(tmp_path))
    assert v["P3_k20"]["verdict"].startswith("ALTERNATIVE") and v["passing_k"] == 24
    assert v["P4_cost"]["verdict"] == "REFUTED" and "cheap" in v["P4_cost"]["reads"]
    assert v["P5_top1"]["verdict"] == "HOLDS" and v["P5_top1"]["k"] == 24   # read at the PASSING k, not the smallest arm


def test_no_k_reaches_the_floor_and_refused_scorer(tmp_path):
    rows = [_row(15, 0.133, 0.89), _row(20, 0.09, 0.93), _row(24, 0.06, 0.94)]
    json.dump(_rec(rows), open(tmp_path / "gemma4keep_kl.json", "w"))
    v = p50.reduce(str(tmp_path))
    assert v["P3_k20"]["verdict"].startswith("REFUTED") and v["passing_k"] is None
    assert v["P4_cost"]["verdict"] == "NOT_READ" and v["P5_top1"]["verdict"] == "NOT_READ"
    json.dump(_rec([_row(15, 0.133, 0.89)], scorer="decode"), open(tmp_path / "gemma4keep_kl.json", "w"))
    assert p50.reduce(str(tmp_path))["verdict"].startswith("NOT_READ")
    assert p50.reduce(str(tmp_path / "nowhere")) == {"verdict": "NOT_READ: no gemma4keep_kl.json"}


def test_keep_builder_layer_sets():
    assert ss.builder_for("gemma4keep", "K20") == "loader_keep_20"
    assert ss.quantize_layer_set("loader_keep_20", 30) == set(range(20, 30))
    assert ss.quantize_layer_set("loader_keep_15", 30) == ss.quantize_layer_set("loader_hi", 30)
    assert ss.quantize_layer_set("loader_keep_0", 30) == set(range(30))
    assert ss.needs_arena("gemma4keep") is False
    for arm in ss.ARMS["gemma4keep"]:
        assert ss.arm_env("gemma4keep", arm)["E4B_SERVE_EXP_INT4"] == "0"
