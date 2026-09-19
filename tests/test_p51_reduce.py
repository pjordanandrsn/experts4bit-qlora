"""P51 (bench/p51): the graded-store-map reducer and the tiers grammar."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "bench", *rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p51 = _load("p51_reduce", ("p51", "p51_reduce.py"))
ss = _load("serve_stack_p51", ("p44", "serve_stack.py"))
GB = 2 ** 30


def _row(arm, kl, gb, census, top1=0.92):
    return {"arm": arm, "kl_mean": kl, "top1_agreement": top1, "builder": ss.BUILDERS["gemma4mix"][arm],
            "per_stratum": {}, "engagement": {"expert_bytes_total_gb": gb, "stacks_by_store": census}}


def _rec(rows, scorer="prefill", self_kl=0.279):
    return {"scorer": {"name": "auto", "used": scorer}, "rows": rows, "not_measured": {},
            "reference_pass": {"self_consistency": {"kl_mean": self_kl, "passes": self_kl < 1e-2}}}


ANCHOR_CENSUS = {"bf16(base)": 20, "nf4/b64": 10}
GRADED_CENSUS = {"bf16(base)": 10, "int8/b64": 10, "nf4/b64": 10}
CRUSH_CENSUS = {"bf16(base)": 10, "int8/b64": 10, "nf4/b256": 10}


def test_graded_map_ships(tmp_path):
    rows = [_row("bf16_20", 0.0469, 32.35, ANCHOR_CENSUS), _row("int8_20", 0.71, 19.1, {"int8/b64": 20, "nf4/b64": 10}),
            _row("graded_10_10", 0.082, 25.7, GRADED_CENSUS), _row("graded_5_15", 0.24, 22.2, {"bf16(base)": 5, "int8/b64": 15, "nf4/b64": 10}),
            _row("graded_10_10_crush", 0.090, 25.3, CRUSH_CENSUS)]
    json.dump(_rec(rows), open(tmp_path / "gemma4mix_kl.json", "w"))
    v = p51.reduce(str(tmp_path))
    assert v["M1_anchor"]["verdict"] == "HOLDS"
    assert v["M2_uniform_int8"]["verdict"] == "HOLDS" and "as P49" in v["M2_uniform_int8"]["reads"]
    assert v["M3_graded"]["verdict"] == "HOLDS" and round(v["M3_graded"]["vs_anchor"], 1) == 1.7
    assert v["M4_bytes"]["verdict"] == "HOLDS" and v["M4_bytes"]["saved_gb"] == 6.65
    assert v["M5_crush"]["verdict"] == "HOLDS" and "cheap half" in v["M5_crush"]["reads"]
    assert v["decision"].startswith("SHIP the graded map")
    assert "M3_graded: HOLDS" in p51.render_md(v)


def test_grading_does_not_pay(tmp_path):
    rows = [_row("bf16_20", 0.0469, 32.35, ANCHOR_CENSUS), _row("int8_20", 0.71, 19.1, {"int8/b64": 20, "nf4/b64": 10}),
            _row("graded_10_10", 0.35, 25.7, GRADED_CENSUS), _row("graded_10_10_crush", 0.60, 21.0, CRUSH_CENSUS)]
    json.dump(_rec(rows), open(tmp_path / "gemma4mix_kl.json", "w"))
    v = p51.reduce(str(tmp_path))
    assert v["M3_graded"]["verdict"] == "REFUTED"
    assert v["M5_crush"]["verdict"] == "REFUTED" and "pays" in v["M5_crush"]["reads"]
    assert v["decision"].startswith("SHIP P50's uniform bf16 head")


def test_refuted_anchor_reads_nothing_else(tmp_path):
    rows = [_row("bf16_20", 0.21, 32.35, ANCHOR_CENSUS), _row("graded_10_10", 0.05, 25.7, GRADED_CENSUS)]
    json.dump(_rec(rows), open(tmp_path / "gemma4mix_kl.json", "w"))
    v = p51.reduce(str(tmp_path))
    assert v["M1_anchor"]["verdict"] == "REFUTED" and "M3_graded" not in v
    assert v["verdict"].startswith("NOT_READ: the anchor is refused")
    assert "M1_anchor: REFUTED" in p51.render_md(v)


def test_inconclusive_and_missing_and_refused_scorer(tmp_path):
    rows = [_row("bf16_20", 0.0469, 32.35, ANCHOR_CENSUS), _row("int8_20", 0.40, 19.1, {"int8/b64": 20, "nf4/b64": 10}),
            _row("graded_10_10", 0.20, 25.7, GRADED_CENSUS)]
    json.dump(_rec(rows), open(tmp_path / "gemma4mix_kl.json", "w"))
    v = p51.reduce(str(tmp_path))
    assert v["M2_uniform_int8"]["verdict"] == "INCONCLUSIVE" and v["M3_graded"]["verdict"] == "INCONCLUSIVE"
    assert v["M5_crush"]["verdict"] == "NOT_READ" and v["decision"].startswith("REPORT only")
    json.dump(_rec(rows, scorer="decode"), open(tmp_path / "gemma4mix_kl.json", "w"))
    assert p51.reduce(str(tmp_path))["verdict"].startswith("NOT_READ")
    assert p51.reduce(str(tmp_path / "nowhere")) == {"verdict": "NOT_READ: no gemma4mix_kl.json"}


def test_tiers_grammar_and_census():
    assert ss.store_map("loader_tiers_bf16:10_int8:10_nf4:10", 30)[0] is None
    assert ss.store_map("loader_tiers_bf16:10_int8:10_nf4:10", 30)[10] == ("int8", 64)
    assert ss.store_map("loader_tiers_bf16:10_int8:10_nf4b256:10", 30)[29] == ("nf4", 256)
    assert ss.tier_census("loader_tiers_bf16:10_int8:10_nf4:10", 30) == GRADED_CENSUS
    assert ss.tier_census("loader_tiers_bf16:10_int8:10_nf4b256:10", 30) == CRUSH_CENSUS
    # the anchor is byte-for-byte P50's keep-20
    m = ss.store_map("loader_tiers_bf16:20_nf4:10", 30)
    assert {k for k, v in m.items() if v is not None} == ss.quantize_layer_set("loader_keep_20", 30)
    assert all(m[i] is None for i in range(20))
    with pytest.raises(ValueError, match="sum to"):
        ss.store_map("loader_tiers_bf16:20_nf4:5", 30)
    assert ss.parse_tiers("loader_keep_20") is None
    assert ss.needs_arena("gemma4mix") is False and ss.control_arm("gemma4mix") == "bf16_20"
    for arm in ss.ARMS["gemma4mix"]:
        assert ss.arm_env("gemma4mix", arm)["E4B_SERVE_EXP_INT4"] == "0"


def test_builder_check_refuses_a_map_that_did_not_apply():
    kl = _load("kl_serve_p51", ("p44", "kl_serve.py"))
    ok = {"builder": "loader_tiers_bf16:10_int8:10_nf4:10", "n_quantized": 20, "n_unquantized": 10,
          "expected_quantized": 20, "expected_unquantized": 10, "moe_layers": 30,
          "tier_census_expected": GRADED_CENSUS, "stacks_by_store": dict(GRADED_CENSUS)}
    kl._builder_check(ok)
    with pytest.raises(RuntimeError, match="tiers ask for"):
        kl._builder_check(dict(ok, stacks_by_store={"bf16(base)": 10, "nf4/b64": 20}))
