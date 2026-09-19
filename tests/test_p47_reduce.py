"""P47 (bench/p47): the reducer applies the registered rules and nothing else; the serve_stack builders and the
kl_serve builder check refuse a layer set that did not apply."""
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


p47 = _load("p47_reduce", ("p47", "p47_reduce.py"))
ss = _load("serve_stack", ("p44", "serve_stack.py"))


def _row(arm, kl, builder, nq, nu, eq, eu, top1=0.7):
    return {"arm": arm, "kl_mean": kl, "top1_agreement": top1, "builder": builder,
            "per_stratum": {"general": {"kl_mean": kl}},
            "engagement": {"builder": builder, "n_quantized": nq, "n_unquantized": nu, "expected_quantized": eq, "expected_unquantized": eu, "int4_expert_layers": 0}}


def _receipt(rows, scorer="prefill", self_kl=0.279):
    return {"scorer": {"name": "auto", "used": scorer}, "rows": rows, "not_measured": {},
            "reference_pass": {"self_consistency": {"kl_mean": self_kl, "passes": self_kl < 1e-2}}}


def test_model_not_stack_reading(tmp_path):
    rows = [_row("served_nf4", 1.077, "served", 30, 0, 30, 0), _row("loader_nf4", 1.06, "loader", 30, 0, 30, 0),
            _row("loader_bf16experts", 0.012, "loader_unquant", 0, 30, 0, 30),
            _row("loader_nf4_lo", 0.20, "loader_lo", 15, 15, 15, 15), _row("loader_nf4_hi", 0.85, "loader_hi", 15, 15, 15, 15)]
    json.dump(_receipt(rows), open(tmp_path / "gemma4diag_kl.json", "w"))
    v = p47.reduce(str(tmp_path))
    assert v["P1_anchor"]["verdict"] == "HOLDS"
    assert v["P2_stack_vs_model"]["verdict"] == "HOLDS"
    assert v["P3_modelling"]["verdict"] == "HOLDS"
    assert v["P4_depth"]["verdict"] == "HOLDS" and v["P4_depth"]["adds"] is True and "second" in v["P4_depth"]["reads"]
    assert v["P5_reference_self_kl"]["verdict"] == "HOLDS"
    md = p47.render_md(v)
    assert "P2_stack_vs_model: HOLDS" in md


def test_stack_reading_and_modelling_defect(tmp_path):
    rows = [_row("served_nf4", 1.10, "served", 30, 0, 30, 0), _row("loader_nf4", 0.05, "loader", 30, 0, 30, 0),
            _row("loader_bf16experts", 0.9, "loader_unquant", 0, 30, 0, 30),
            _row("loader_nf4_lo", 0.02, "loader_lo", 15, 15, 15, 15), _row("loader_nf4_hi", 0.03, "loader_hi", 15, 15, 15, 15)]
    json.dump(_receipt(rows), open(tmp_path / "gemma4diag_kl.json", "w"))
    v = p47.reduce(str(tmp_path))
    assert v["P2_stack_vs_model"]["verdict"] == "REFUTED"
    assert v["P3_modelling"]["verdict"] == "REFUTED"
    assert v["P4_depth"]["verdict"] == "REFUTED"


def test_missing_rows_are_not_read_and_refused_scorer_reads_nothing(tmp_path):
    json.dump(_receipt([_row("served_nf4", 1.077, "served", 30, 0, 30, 0)]), open(tmp_path / "gemma4diag_kl.json", "w"))
    v = p47.reduce(str(tmp_path))
    assert v["P1_anchor"]["verdict"] == "HOLDS"
    for k in ("P2_stack_vs_model", "P3_modelling", "P4_depth"):
        assert v[k]["verdict"] == "NOT_READ"
    json.dump(_receipt([_row("served_nf4", 1.077, "served", 30, 0, 30, 0)], scorer="decode", self_kl=0.279), open(tmp_path / "gemma4diag_kl.json", "w"))
    v = p47.reduce(str(tmp_path))
    assert v["verdict"].startswith("NOT_READ") and "P1_anchor" not in v
    assert p47.reduce(str(tmp_path / "nowhere")) == {"verdict": "NOT_READ: no gemma4diag_kl.json"}


def test_serve_stack_builders_and_layer_sets():
    assert list(ss.ARMS["gemma4diag"]) == ["served_nf4", "loader_nf4", "loader_bf16experts", "loader_nf4_lo", "loader_nf4_hi"]
    assert ss.control_arm("gemma4diag") == "served_nf4" and ss.control_arm("gemma4") == "nf4"
    assert ss.builder_for("gemma4diag", "served_nf4") == "served" and ss.builder_for("gemma4", "nf4") == "served"
    assert ss.builder_for("gemma4diag", "loader_bf16experts") == "loader_unquant"
    assert ss.quantize_layer_set("loader", 30) is None
    assert ss.quantize_layer_set("loader_unquant", 30) == set()
    assert ss.quantize_layer_set("loader_lo", 30) == set(range(15)) and ss.quantize_layer_set("loader_hi", 30) == set(range(15, 30))
    assert ss.MODELS["gemma4diag"] == ss.MODELS["gemma4"]
    # every P47 arm carries the nf4 lever set: no int4 store, no fusion
    for arm in ss.ARMS["gemma4diag"]:
        env = ss.arm_env("gemma4diag", arm)
        assert env["E4B_SERVE_EXP_INT4"] == "0" and env["E4B_SERVE_ATTN_INT4_CALIB"] == "0"
        assert env["E4B_FUSE_T1_GLUE"] == "0" and env["E4B_FUSE_ROUTER_EPI"] == "0"
    with pytest.raises(ValueError):
        ss.build_loader_model("x/y", "served")
    # P48: one NF4 layer at a time
    assert len(ss.ARMS["gemma4layer"]) == 30 and list(ss.ARMS["gemma4layer"])[:2] == ["L00", "L01"]
    assert ss.builder_for("gemma4layer", "L07") == "loader_only_7"
    assert ss.quantize_layer_set("loader_only_7", 30) == {7}
    with pytest.raises(ValueError, match="outside"):
        ss.quantize_layer_set("loader_only_30", 30)
    assert ss.needs_arena("gemma4diag") is True and ss.needs_arena("gemma4layer") is False and ss.needs_arena("gemma4") is True
    assert ss.control_arm("gemma4layer") == "L00"
    assert ss.main(["needs_arena", "gemma4layer"]) == 0


def test_builder_check_refuses_a_layer_set_that_did_not_apply():
    kl = _load("kl_serve", ("p44", "kl_serve.py"))
    kl._builder_check({"builder": "served"})  # served rows carry no expectation
    kl._builder_check({"builder": "loader_lo", "n_quantized": 15, "n_unquantized": 15, "expected_quantized": 15, "expected_unquantized": 15})
    with pytest.raises(RuntimeError, match="did not apply"):
        kl._builder_check({"builder": "loader_unquant", "n_quantized": 30, "n_unquantized": 0, "expected_quantized": 0, "expected_unquantized": 30})
