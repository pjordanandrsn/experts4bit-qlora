# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane FAM's instrument (``bench/fam/fam_box.py``) and rule (``bench/fam/fam_reduce.py``) on CPU, before any GPU is
rented (bench/fam/PREREG-fam.md; e4b#1362).

1. **Self-tests.** The box's and the reducer's own.
2. **One family end to end, on a tiny gpt-oss in bf16** (P108's stand-in attention, P115's ``int4_b32`` stand-in
   kernels, the reference unpadded): the OFF process scores every cell -- R, rep, chunk, half (shape 12), split1, the
   scale mutant and the graded mutants -- and each single-knob ON process scores ON against the saved references. The
   reducer's integrity checks then run on these real records with the tiny model's constants (layers, census, per-step
   calls), so the box and the rule are proven to agree on the record format: no VOID.
3. **The census and per-step calls the registration predicts**, per knob alone and all at ``auto``, on tiny gpt-oss
   (``2L + 1`` / ``L`` / ``L``; ``L + 1, L, L``), GraniteMoe and a Qwen3.5-MoE hybrid.

What CPU cannot show: the real decode attention kernel (the stand-in ignores ``n_split``, so ``split1`` repeats R bit for
bit here and is not a floor draw), the bucket padding, and grouped-nf4-gemm's expert routes.
"""
import copy
import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("fam", "p115", "p110", "p108", "p97", "p109")]


def _load(name, lane="fam"):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / lane / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub():
    sys.path.insert(0, str(ROOT / "tests"))
    import test_p115_quality_box as t
    return t._kernel_stub()


def _gpt_oss(layers=2):
    pytest.importorskip("transformers.models.gpt_oss", reason="needs transformers with gpt-oss")
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    cfg = GptOssConfig(vocab_size=256, hidden_size=128, intermediate_size=64, num_hidden_layers=layers, head_dim=32,
                       num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
                       max_position_embeddings=512, sliding_window=16)
    torch.manual_seed(0)
    return GptOssForCausalLM(cfg).to(torch.bfloat16).eval()


def test_self_tests():
    assert _load("fam_box").self_test() == 0
    assert _load("fam_reduce").self_test() == 0


P, C, CHUNK, FLOOR_CHUNK = 136, 6, 136, 68          # every prefill forward above the folds' 64-row decode bound
UNPADDED = {"device_grouping": True}
L = 2


def _apply(model, config, box):
    from experts4bit_qlora.serve_paged import FUSION_KNOBS, PagedServeConfig, _apply_fusions
    modes = {k: box.CONFIGS[config][k] for k in FUSION_KNOBS}
    return _apply_fusions(model, PagedServeConfig(fusion_modes=modes))


@pytest.fixture(scope="module")
def gptoss_records(tmp_path_factory):
    box = _load("fam_box")
    q = _load("p115_quality", "p115")
    mp = pytest.MonkeyPatch()
    ref_root = tmp_path_factory.mktemp("ref")
    g = torch.Generator().manual_seed(3)
    windows = {t: [torch.randint(0, 256, (P + C,), generator=g).tolist() for _ in range(36)] for t in box.TEXTS}
    recs = {}
    try:
        for config in ("OFF", "ON_glue", "ON_r2", "ON_epi", "ON_auto"):
            mp.setitem(sys.modules, "int4_b32", _stub())
            counters = q.KernelCounters().install()
            model = _gpt_oss(L)
            census = _apply(model, config, box)
            fwd = q.ForwardCounter(model)
            cells = box.run_cells(model, windows, config=config, texts=box.TEXTS, shapes=box.SHAPES,
                                  sets=tuple(box.SETS), prompt=P, cont=C, chunk=CHUNK, floor_chunk=FLOOR_CHUNK,
                                  device="cpu", ref_root=str(ref_root), ref_kw=UNPADDED, stand_in=True,
                                  counters=counters, fwd=fwd)
            red = _load("fam_reduce")
            recs[config] = {"config": config, "path": "default", "model": red.MODELS["gptoss"][0],
                            "revision": red.MODELS["gptoss"][1], "e4b_sha": "E", "gnf4_sha": None,
                            "census": {k: census[k] for k in box.CENSUS_KEYS},
                            "store": {"int4_expert_layers": 0, "int4_store_kinds": {}}, "max_mem_gb": 0.0,
                            "cells": cells, "status": "ok"}
    finally:
        mp.undo()
    return box, recs


def _tiny_reducer(red):
    """The reducer with the tiny model's registered constants."""
    red.CONT = C
    red.PREFILL = (P, CHUNK, FLOOR_CHUNK)
    red.ATTN_LAYERS = {**red.ATTN_LAYERS, "gptoss": L}
    red.CENSUS = {**red.CENSUS, ("gptoss", "OFF"): [0, 0, [0, 0], 0], ("gptoss", "ON_glue"): [0, 2 * L + 1, [0, 0], 0],
                  ("gptoss", "ON_r2"): [0, 0, [L, 0], 0], ("gptoss", "ON_epi"): [0, 0, [0, 0], L],
                  ("gptoss", "ON_auto"): [0, 2 * L + 1, [L, 0], L]}
    red.PER_STEP = {**red.PER_STEP, ("gptoss", "ON_glue"): {"rmsnorm_rows": 2 * L + 1},
                    ("gptoss", "ON_r2"): {"rmsnorm_resid_rows": L}, ("gptoss", "ON_epi"): {"router_epilogue": L},
                    ("gptoss", "ON_auto"): {"rmsnorm_rows": L + 1, "rmsnorm_resid_rows": L, "router_epilogue": L}}
    return red


def test_every_cell_and_arm_is_scored(gptoss_records):
    box, recs = gptoss_records
    off = recs["OFF"]
    assert len(off["cells"]) == len(box.TEXTS) * len(box.SHAPES) * len(box.SETS) == 12
    for cell, c in off["cells"].items():
        text, shape, set_ = cell.split("|")
        assert c["base"]["arms"] == list(box.off_arms(int(shape)))
        assert c["base"]["group"] == int(shape)
        assert set(c["extra"]) == set(box.extra_arms(set_))
        rows = c["base"]["per_window"][text]
        assert [len(rows[a]) for a in box.off_arms(int(shape))] == [12, int(shape), 12] + ([12] if int(shape) > 1 else []) + [12]
        for arm, x in c["extra"].items():
            eng = x["record"]["engagement"][text][arm]
            assert x["wrapped_calls"] == sum(e["decode_calls"] for e in eng) > 0, (cell, arm)
    for config in ("ON_glue", "ON_r2", "ON_epi", "ON_auto"):
        assert set(recs[config]["cells"]) == set(off["cells"])


def test_the_wrappers_change_what_they_name(gptoss_records):
    box, recs = gptoss_records
    c = recs["OFF"]["cells"]["wikitext|1|A"]
    R = c["base"]["per_window"]["wikitext"]["R"]
    split = c["extra"]["split1"]["record"]["per_window"]["wikitext"]["split1"]
    assert [x["nll"] for x in split] == [x["nll"] for x in R]       # the stand-in ignores n_split: no draw on CPU
    for arm in ("mut098", "mut095", "mut090"):
        m = c["extra"][arm]["record"]["per_window"]["wikitext"][arm]
        assert [x["nll"] for x in m] != [x["nll"] for x in R], arm
    big = c["base"]["per_window"]["wikitext"]["mutant_scale"]
    d = lambda rows: sum(abs(a["nll"] - b["nll"]) for a, b in zip(rows, R)) / len(R)  # noqa: E731
    assert d(big) > d(c["extra"]["mut090"]["record"]["per_window"]["wikitext"]["mut090"]) > \
        d(c["extra"]["mut098"]["record"]["per_window"]["wikitext"]["mut098"]) > 0


def test_the_census_and_per_step_calls_per_knob(gptoss_records):
    box, recs = gptoss_records
    red = _tiny_reducer(_load("fam_reduce"))
    for config, rec in recs.items():
        assert red._census(rec) == red.CENSUS[("gptoss", config)], (config, rec["census"])


def test_the_reducer_reads_the_records_without_an_integrity_fault(gptoss_records):
    box, recs = gptoss_records
    red = _tiny_reducer(_load("fam_reduce"))
    cells = list(recs["OFF"]["cells"])
    why = []
    red.family_checks("gptoss", recs, "E", cells, why, cont=C)
    assert why == [], why[:5]
    v = red.reduce_family("gptoss", recs, "E")
    assert set(v["verdict"]) == {"ON_glue", "ON_r2", "ON_epi", "ON_auto"}
    assert all(x != "VOID" or "no draw" in " ".join(v["why"]) for x in v["verdict"].values()), v["why"][:3]


def test_a_broken_record_is_void(gptoss_records):
    box, recs = gptoss_records
    red = _tiny_reducer(_load("fam_reduce"))
    bad = copy.deepcopy(recs)
    e = bad["ON_epi"]["cells"]["c4val1|12|B"]["base"]["engagement"]["c4val1"]["ON"][0]
    e["kernels"]["router_epilogue"] -= 1
    why = []
    red.family_checks("gptoss", bad, "E", list(bad["OFF"]["cells"]), why, cont=C)
    assert any("glue-kernel calls" in w for w in why), why


@pytest.mark.parametrize("family", ["granite", "qwen3_5"])
def test_the_census_per_knob_on_other_families(monkeypatch, family):
    box = _load("fam_box")
    q = _load("p115_quality", "p115")
    sys.path.insert(0, str(ROOT / "tests"))
    import test_p115_quality_box as t
    if family == "granite":
        pytest.importorskip("transformers.models.granitemoe", reason="needs transformers with GraniteMoe")
        build, layers = (lambda: t._tiny_granite(2)), 2
        want = {"ON_glue": ([0, 5, [0, 0], 0], {"rmsnorm_rows": 5}),
                "ON_r2": ([0, 0, [2, 2], 0], {"rmsnorm_resid_rows": 2, "scaled_resid_add_rows": 2, "rope_heads": 4}),
                "ON_epi": ([0, 0, [0, 0], 2], {"router_epilogue": 2}),
                "ON_auto": ([0, 5, [2, 2], 2], {"rmsnorm_rows": 3, "rmsnorm_resid_rows": 2, "scaled_resid_add_rows": 2,
                                               "rope_heads": 4, "router_epilogue": 2})}
    else:
        pytest.importorskip("transformers.models.qwen3_5_moe", reason="needs transformers with Qwen3.5-MoE")
        monkeypatch.setitem(sys.modules, "causal_conv1d", None)
        monkeypatch.setitem(sys.modules, "fla", None)
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

        def build():
            cfg = Qwen3_5MoeTextConfig(vocab_size=256, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                                       num_key_value_heads=2, head_dim=32, num_experts=4, num_experts_per_tok=2,
                                       moe_intermediate_size=64, shared_expert_intermediate_size=64,
                                       max_position_embeddings=512)
            torch.manual_seed(0)
            return m.Qwen3_5MoeForCausalLM(cfg).to(torch.bfloat16).eval()
        layers = 4
        want = {"ON_glue": ([0, 0, [0, 0], 0], {}), "ON_r2": ([0, 0, [0, 0], 0], {}),
                "ON_epi": ([0, 0, [0, 0], layers], {"router_epilogue": layers}),
                "ON_auto": ([0, 0, [0, 0], layers], {"router_epilogue": layers})}
    for config, (census_want, step_want) in want.items():
        monkeypatch.setitem(sys.modules, "int4_b32", _stub())
        counters = q.KernelCounters().install()
        model = build()
        census = _apply(model, config, box)
        assert [census[k] for k in box.CENSUS_KEYS] == census_want, (config, census)
        with torch.no_grad():
            before = counters.snapshot()
            model(input_ids=torch.randint(0, 256, (4, 1)), position_ids=torch.full((4, 1), 32), use_cache=False)
        got = {k: v - before[k] for k, v in counters.snapshot().items() if v - before[k]}
        assert got == step_want, (config, got)
    assert math.isfinite(layers)


def test_mixtral_at_full_depth_is_the_registered_census_per_step_and_fp32_router_path(monkeypatch):
    """Amendment 5: a tiny Mixtral at the real depth (32 layers) reads exactly the registered census and per-step glue
    calls for each config, and wherever the router epilogue engages, every patched router keeps fp32 weights
    (``fusion_report``'s ``fp32_upstream``, which the reducer checks on the card)."""
    pytest.importorskip("transformers.models.mixtral", reason="needs transformers with Mixtral")
    from transformers import MixtralConfig, MixtralForCausalLM

    from experts4bit_qlora.serve_paged import FUSION_KNOBS, PagedServeConfig, _apply_fusions
    box, red = _load("fam_box"), _load("fam_reduce")
    q = _load("p115_quality", "p115")
    for config in red.FAMILY_CONFIGS["mixtral"]:
        monkeypatch.setitem(sys.modules, "int4_b32", _stub())
        counters = q.KernelCounters().install()
        cfg = MixtralConfig(vocab_size=256, hidden_size=128, intermediate_size=64, num_hidden_layers=32,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_local_experts=4,
                            num_experts_per_tok=2, max_position_embeddings=512)
        torch.manual_seed(0)
        model = MixtralForCausalLM(cfg).to(torch.bfloat16).eval()
        report = {}
        census = _apply_fusions(model, PagedServeConfig(fusion_modes={k: box.CONFIGS[config][k] for k in FUSION_KNOBS}),
                                report=report)
        assert [census[k] for k in box.CENSUS_KEYS] == red.CENSUS[("mixtral", config)], (config, census)
        with torch.no_grad():
            before = counters.snapshot()
            model(input_ids=torch.randint(0, 256, (4, 1)), position_ids=torch.full((4, 1), 32), use_cache=False)
        got = {k: v - before[k] for k, v in counters.snapshot().items() if v - before[k]}
        assert got == red.PER_STEP.get(("mixtral", config), {}), (config, got)
        epi = report["folds"]["E4B_FUSE_ROUTER_EPI"]
        if census["fuse_router_epilogue_n"]:
            assert epi["patched"] == epi["fp32_upstream"] == red.FP32_ROUTERS["mixtral"], (config, epi)
