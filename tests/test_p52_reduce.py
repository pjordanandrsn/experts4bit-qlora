"""P52 (bench/p52): the Gemma-4 gate reducer. The bar is fixed; G2 refuted reads nothing."""
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


p52 = _load("p52_reduce", ("p52", "p52_reduce.py"))
SET = {"name": "kl_prompts_heldout", "sha256": "2030ae10e0fe537f", "n_scored": 100,
       "strata": {"general": 36, "technical": 28, "code": 28, "longctx": 8}}


def _row(arm, kl, top1, gb=None):
    return {"arm": arm, "kl_mean": kl, "top1_agreement": top1, "per_stratum": {"general": {"kl_mean": kl}},
            "engagement": {"expert_bytes_total_gb": gb}}


def _rec(rows, scorer="prefill", self_kl=0.279, pset=None):
    return {"scorer": {"name": "auto", "used": scorer}, "rows": rows, "not_measured": {},
            "prompt_set": pset if pset is not None else SET,
            "reference_pass": {"self_consistency": {"kl_mean": self_kl, "passes": self_kl < 1e-2}}}


def _write(d, gemma_rows, gptoss_rows=None, gset=None):
    json.dump(_rec(gemma_rows, pset=gset), open(os.path.join(d, "gemma4mix_kl.json"), "w"))
    if gptoss_rows is not None:
        json.dump(_rec(gptoss_rows, scorer="decode", self_kl=0.0005), open(os.path.join(d, "gptoss_kl.json"), "w"))


def test_gate_fails_and_that_is_a_real_reading(tmp_path):
    _write(str(tmp_path), [_row("bf16_20", 0.0472, 0.915, 32.35), _row("graded_10_10", 0.174, 0.851, 25.70),
                           _row("bf16_13", 0.251, 0.824, 25.21)], [_row("nf4_r12", 0.0231, 0.935)])
    v = p52.reduce(str(tmp_path))
    assert v["G2_bar_provenance"]["verdict"] == "HOLDS" and v["G1_set_comparable"]["verdict"] == "HOLDS"
    assert v["G3_gate"]["verdict"] == "FAILS" and "RUN and not passed" in v["G3_gate"]["reads"]
    assert v["G4_matched_bytes"]["verdict"] == "HOLDS"
    assert v["decision"].startswith("DOCUMENTED OPTION")
    assert "G3_gate: FAILS" in p52.render_md(v)


def test_gate_passes(tmp_path):
    _write(str(tmp_path), [_row("bf16_20", 0.0472, 0.915, 32.35), _row("graded_10_10", 0.081, 0.941, 25.70),
                           _row("bf16_13", 0.12, 0.90, 25.21)], [_row("nf4_r12", 0.0231, 0.935)])
    v = p52.reduce(str(tmp_path))
    assert v["G3_gate"]["verdict"] == "PASSES" and v["decision"].startswith("SHIP the graded map")


def test_g2_refuted_reads_nothing(tmp_path):
    _write(str(tmp_path), [_row("bf16_20", 0.0472, 0.915), _row("graded_10_10", 0.05, 0.95)],
           [_row("nf4_r12", 0.060, 0.90)])                 # the bar's provenance does not replicate
    v = p52.reduce(str(tmp_path))
    assert v["G2_bar_provenance"]["verdict"] == "REFUTED" and "G3_gate" not in v
    assert v["verdict"].startswith("NOT_READ: G2 refuted")


def test_wrong_prompt_set_and_missing_files_read_nothing(tmp_path):
    _write(str(tmp_path), [_row("graded_10_10", 0.05, 0.95)], [_row("nf4_r12", 0.0231, 0.935)],
           gset={"name": "kl_prompts", "sha256": "eaa7792260b3f10d", "n_scored": 200, "strata": {}})
    v = p52.reduce(str(tmp_path))
    assert v["verdict"].startswith("NOT_READ: this lane scores the held-out set")
    os.remove(os.path.join(str(tmp_path), "gptoss_kl.json"))
    _write(str(tmp_path), [_row("graded_10_10", 0.05, 0.95)])
    v = p52.reduce(str(tmp_path))
    assert v["verdict"].startswith("NOT_READ: the bar's provenance point was not measured")
    assert p52.reduce(str(tmp_path / "nowhere"))["verdict"].startswith("NOT_READ: no gemma4mix_kl.json")


def test_g1_refuted_is_disclosed_but_the_gate_still_reads(tmp_path):
    _write(str(tmp_path), [_row("bf16_20", 0.090, 0.88), _row("graded_10_10", 0.174, 0.851)],
           [_row("nf4_r12", 0.0231, 0.935)])
    v = p52.reduce(str(tmp_path))
    assert v["G1_set_comparable"]["verdict"] == "REFUTED" and "not interchangeable" in v["G1_set_comparable"]["reads"]
    assert v["G3_gate"]["verdict"] == "FAILS"


def test_heldout_prompts_are_disjoint_and_proportional():
    sys.path.insert(0, os.path.join(HERE, "..", "bench"))
    import kl_prompts_heldout as h
    assert h.assert_disjoint_from_committed() == 100
    assert h.strata_counts() == {"general": 36, "technical": 28, "code": 28, "longctx": 8}
    assert len({p["id"] for p in h.PROMPTS}) == 100
    assert all(p["id"].startswith(p["stratum"] + "-h") for p in h.PROMPTS)
