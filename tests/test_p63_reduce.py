"""Lane P63's reducer applies the pre-registration literally: every verdict branch fires on synthetic receipts built to
trigger it, the instrument gate voids a stack, and the prediction tables name only sub-arms the probe registers
(bench/p63/p63_reduce.py; bench/p63/P63-PREREG.md)."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "bench" / "p63" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("p63_reduce")
PROBE = _load("p63_probe")


def _mode(exact=0, positions=68, flips=1, kl=1e-3, layer=0):
    return {"positions": positions, "exact_positions": exact, "argmax_flips": flips, "kl_mean": kl, "kl_max": 10 * kl,
            "min_first_diff_layer": layer, "first_diff_site_hist": {"attn_core": positions - exact}}


def _replay(exact_by_n, kind="experts"):
    name = {"experts": "L0.experts", "attn": "L0.attn.qkv_proj", "lm_head": "lm_head",
            "router": "L0.router", "input_layernorm": "L0.input_layernorm"}[kind]
    by_n = {"1": {"rows": 1, "rows_bit_equal": 1}}
    for n, ex in exact_by_n.items():
        by_n[str(n)] = {"rows": n, "rows_bit_equal": n if ex else n - 1}
    return {name: {"calls": 160, "by_n": by_n, "replay_faithful_at_1": True}}


def _arm(stack, subarms, census_pairs=None, dec="nf4.dotpad", combine=None, cross=None):
    arm = {"device": "SYNTH", "census": {}, "refused": {}, "subarms": subarms, "cross_controls": cross or {}}
    if census_pairs is not None:
        recs = [{"path": dec, "T": 1}] if stack == "nf4" else []
        arm["kernel_census"] = {"records": recs, "pairs": census_pairs}
    if combine is not None:
        arm["combine_census"] = {"records": combine}
    return arm


def _write(tmp_path, arms):
    for stack, arm in arms.items():
        d = tmp_path / stack
        d.mkdir()
        (d / "p63_arm.json").write_text(json.dumps(arm))
    return R.reduce(str(tmp_path))


def _sub(replay=None, modes=None, det=None):
    return {"replay": replay or {}, "modes": modes or {"verify17": _mode()}, "determinism": det or {}}


def test_a_nondeterministic_control_voids_the_stack(tmp_path):
    rep = _write(tmp_path, {"int4": _arm("int4", {"hf.default": _sub(det={"control_repeat_bit_identical": False})})})
    assert not rep["gates"]["int4"]["ok"]
    assert rep["P6"] == [] and rep["P2"] and all(r["verdict"] == "UNREAD" for r in rep["P2"] if r["stack"] == "int4")


def test_an_unfaithful_replay_voids_the_stack(tmp_path):
    rp = _replay({16: False})
    rp["L0.experts"]["replay_faithful_at_1"] = False
    rep = _write(tmp_path, {"int4": _arm("int4", {"hf.default": _sub(replay=rp)})})
    assert not rep["gates"]["int4"]["ok"]


def test_p1_held_refuted_defect_and_unread(tmp_path):
    pairs = [
        {"a": "int4.gemv@T=1", "b": f"int4.gemv@T={T}", "class": "EXACT"} for T in (16, 17, 160)] + [
        {"a": "int4.gemv@T=1", "b": f"int4.deq_bf16@T={T}", "class": "PRECISION"} for T in (16, 17, 160)] + [
        {"a": "int4.gemv@T=1", "b": "int4.grouped_gemm@T=16", "class": "EXACT"},          # predicted REORDER
        {"a": "int4.gemv@T=1", "b": "int4.grouped_gemm@T=160", "class": "DEFECT?"}]
    rep = _write(tmp_path, {"int4": _arm("int4", {"hf.default": _sub()}, census_pairs=pairs)})
    v = {r["pair"]: r["verdict"] for r in rep["P1"] if r["stack"] == "int4"}
    assert v["int4.gemv -> int4.gemv"] == "HELD"
    assert v["int4.gemv -> int4.deq_bf16"] == "HELD"
    assert v["int4.gemv -> int4.grouped_gemm"] == "DEFECT?"
    assert not [r for r in rep["P1"] if r["stack"] == "nf4"]          # no nf4 receipt: nothing, not a fabricated row


def test_p1_resolves_the_nf4_decode_path_the_box_ran(tmp_path):
    pairs = [{"a": "nf4.gemv_scalar@T=1", "b": "nf4.gemv_scalar@T=16", "class": "REORDER"}]
    rep = _write(tmp_path, {"nf4": _arm("nf4", {"hf.default": _sub()}, census_pairs=pairs, dec="nf4.gemv_scalar")})
    row = [r for r in rep["P1"] if r["pair"] == "nf4.gemv_scalar -> nf4.gemv_scalar"][0]
    assert row["want"] == "EXACT" and row["verdict"] == "REFUTED"


def test_p2_exact_needs_every_module_and_not_needs_one(tmp_path):
    subs = {"hf.device": _sub(replay=_replay({16: True, 17: True, 160: False})),
            "hf.singleton": _sub(replay=_replay({16: True, 17: False, 160: True})),
            "hf.default": _sub(replay={**_replay({16: False, 17: False, 160: False}),
                                       **_replay({16: False, 17: False, 160: False}, "attn")})}
    rep = _write(tmp_path, {"int4": _arm("int4", subs)})
    v = {(r["subarm"], r["kind"]): r["verdict"] for r in rep["P2"] if r["stack"] == "int4"}
    assert v[("hf.device", "experts")] == "HELD"
    assert v[("hf.singleton", "experts")] == "REFUTED"
    assert v[("hf.default", "experts")] == "HELD" and v[("hf.default", "attn")] == "HELD"
    assert v[("hf.combine0", "experts")] == "UNREAD"


def test_p3_combine_rows_exact_the_chain_informational_and_a_defect(tmp_path):
    comb = [{"path": "combine_rows", "T": T, "rows": T, "rows_bit_equal": T, "bound_ratio": 0.99,
             "bit_equal_to_chain": False} for T in (1, 16)] + [
            {"path": "chain", "T": T, "rows": T, "rows_bit_equal": T - 1, "bound_ratio": 1.5} for T in (1, 16)]
    rep = _write(tmp_path, {"int4": _arm("int4", {"hf.default": _sub()}, combine=comb)})
    v = {r["path"]: r for r in rep["P3"]}
    assert v["combine_rows"]["verdict"] == "HELD" and v["combine_rows"]["vs_chain"] == [False]
    assert v["chain"]["verdict"] == "DEFECT?"


@pytest.mark.parametrize("kl,flips,want", [(0.005, 2, "HELD"), (0.02, 2, "BETWEEN"), (0.05, 2, "REFUTED"),
                                           (0.005, 20, "BETWEEN")])
def test_p5_bands(tmp_path, kl, flips, want):
    cross = {"hf.default vs hf.combine0 (T=1 controls)": {"logits_bit_equal_positions": 0, "positions": 160,
                                                          "argmax_flips": flips, "kl_max": kl * 5, "kl_mean": kl,
                                                          "first_layer_not_bit_equal": 0},
             "hf.default vs hf.device (T=1 controls)": {"logits_bit_equal_positions": 160, "positions": 160,
                                                        "argmax_flips": 0, "kl_max": 0.0, "kl_mean": 0.0,
                                                        "first_layer_not_bit_equal": None}}
    rep = _write(tmp_path, {"nf4": _arm("nf4", {"hf.default": _sub()}, cross=cross)})
    assert rep["P5"][0]["verdict"] == want
    p4 = {r["pair"].split(" vs ")[1].split(" (")[0]: r["verdict"] for r in rep["P4"]}
    assert p4 == {"hf.combine0": "HELD", "hf.device": "HELD"}


def test_p6_sizes_and_an_exact_mode_refutes(tmp_path):
    modes = {"verify17": _mode(kl=0.001, flips=1), "verify16": _mode(kl=0.05, flips=1),
             "prefill": _mode(exact=160, positions=160, flips=0, kl=0.0, layer=None)}
    rep = _write(tmp_path, {"int4": _arm("int4", {"hf.default": _sub(modes=modes),
                                                  "hf.device": {"error": "RuntimeError('x')"}})})
    v = {(r["subarm"], r["mode"]): r for r in rep["P6"]}
    assert v[("hf.default", "verify17")]["size"] == "IN-BAND" and v[("hf.default", "verify17")]["verdict"] == "HELD"
    assert v[("hf.default", "verify16")]["size"] == "BETWEEN"
    assert v[("hf.default", "prefill")]["verdict"].startswith("EXACT")
    assert v[("hf.device", "*")]["verdict"] == "UNREAD"


def test_p7_router_dtype_and_the_defect_scan(tmp_path):
    rp = {"L0.router": {"calls": 160, "replay_faithful_at_1": True, "by_n": {
        "1": {"rows": 1, "rows_bit_equal": 1}, "16": {"rows": 16, "rows_bit_equal": 0},
        "17": {"rows": 17, "rows_bit_equal": 0}, "160": {"rows": 160, "rows_bit_equal": 0, "dtype_differs": True}}},
          "L0.attn.q_proj": {"calls": 160, "replay_faithful_at_1": True, "by_n": {
        "1": {"rows": 1, "rows_bit_equal": 1, "path": "int4.gemv", "bound_ratio": 0.1},
        "160": {"rows": 160, "rows_bit_equal": 0, "path": "int4.cublas_cache", "bound_ratio": 1.2,
                "bound_ratio_fp32_reduction": 0.7}}}}
    census = [{"a": "int4.gemv@T=1", "b": "int4.gemv@T=16", "class": "EXACT"}]
    arm = _arm("int4", {"hf.default": _sub(replay=rp)}, census_pairs=census)
    arm["kernel_census"]["records"] = [{"path": "int4.grouped_gemm", "T": 160, "bound_ratio": 1.3}]
    rep = _write(tmp_path, {"int4": arm})
    p7 = {r["stack"]: r["verdict"] for r in rep["P7"]}
    assert p7["int4"] == "HELD" and p7["nf4"] == "UNREAD"
    d = rep["D"]
    assert d["verdict"] == "DEFECT?" and [x["path"] for x in d["defects"]] == ["int4.grouped_gemm"]
    assert [x["path"] for x in d["reduced_precision_reduction_only"]] == ["int4.cublas_cache"]
    assert "## D " in R.to_md(rep) and "## P7" in R.to_md(rep)


def test_the_prediction_tables_name_only_registered_subarms():
    names = {s: {a["name"] for a in PROBE.SUBARMS[s]} for s in PROBE.SUBARMS}
    for (stack, sub, _kind) in R.P2_REPLAY:
        assert sub in names[stack], (stack, sub)
    every = set().union(*names.values())
    assert set(R.P4_CROSS) <= every
    for stack, subs in PROBE.SUBARMS.items():
        assert subs[0]["name"] == "hf.default", "the first sub-arm is the full-replay one and the cross-control base"


def test_markdown_renders_every_section(tmp_path):
    rep = _write(tmp_path, {"int4": _arm("int4", {"hf.default": _sub()})})
    md = R.to_md(rep)
    for h in ("G0", "P1", "P2", "P3", "P4 / P5", "P6"):
        assert f"## {h}" in md
