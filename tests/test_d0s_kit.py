# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""D0s's kit on CPU (bench/d0s/PREREG-d0s.md): every phase that needs no GPU.

- the reducer's rule, on fixed numbers, at each gate's boundary, and its refusals;
- the REF tool's walk on a tiny synthetic OLMoE checkpoint (D0's own pinned forward), its reproduction check, and
  D0's tools' pin;
- the attention-only adapter derivation;
- the staged pins, and the driver's refusals and dry run.
"""
import csv
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
D0S = ROOT / "bench" / "d0s"
D0 = ROOT / "bench" / "d0" / "receipts" / "d0-p224s3"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, D0S / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("d0s_reduce")


# ---- the reducer ----------------------------------------------------------------------------------------------------


def test_the_margins_are_the_registrations_constants_from_d0s_record():
    for fx in ("nodq", "dq"):
        d0 = R.d0_record(str(D0), fx)
        assert 0.05 * d0["improvement"] == R.MARGINS[fx]
        held = json.loads((D0 / fx / "heldout.json").read_text())["rows"]
        assert [r["REF"][1] for r in d0["rows"]] == [len(r["ids"]) - 1 for r in held] and len(held) == 64


# a D0-shaped record on round numbers: 4 rows of 50 tokens, Q 200 and REF 180 nats a row (D0's improvement field is
# its real one, so the registered-margin check holds); an arm's row NLL is given whole
ROWS, N_TOK, Q_NLL, REF_NLL = 4, 50, 200.0, 180.0
COST = {"cost": {"W1": {"ttft_s": 0.1, "itl_s": 0.01, "tok_s": 100.0},
                 "W16": {"ttft_s": 0.5, "itl_s": 0.02, "tok_s": 800.0}}, "peak_vram_bytes": 10 ** 9}
IDS = {"attention_only": "a" * 64, "adapter": "f" * 64}
PREP = {"fixtures": {fx: {"attention_only": {"adapter_id": IDS["attention_only"], "source_adapter_id": IDS["adapter"]},
                          "cells": {}} for fx in ("nodq", "dq")}}


def _d0(fx="nodq"):
    return {"rows": [{"Q": (Q_NLL, N_TOK), "REF": (REF_NLL, N_TOK)} for _ in range(ROWS)],
            "improvement": R.d0_record(str(D0), fx)["improvement"]}


def _served(arm):
    if arm.startswith("BASE-ATTN"):
        return {"adapter_id": IDS["attention_only"], "merged_arena": None, "fingerprints": {"experts": {"status": "verified"}},
                "placement": "vram", "graph_status": {"1": "captured"}}
    return {"adapter_id": IDS["adapter"], "merged_arena": {"source": R.SOURCES[arm], "adapter_id": IDS["adapter"]},
            "fingerprints": {"experts": {"status": R.MERGED_VERIFIED}}, "placement": "vram", "graph_status": {"1": "captured"}}


def _arm(row_nll, arm="MERGED-Q", argmax=0, cost=None):
    return {"rows": [{"nll": row_nll, "n": N_TOK, "argmax": [argmax] * N_TOK} for _ in range(ROWS)],
            **json.loads(json.dumps(cost or COST)), "served": _served(arm), "device": "NVIDIA GeForce RTX 4090"}


def _ref():
    return {"status": "OK", "rows": [{"n": N_TOK, "ref_argmax": [0] * N_TOK, "q_argmax": [0] * N_TOK}
                                     for _ in range(ROWS)]}


def _arms(**over):
    arms = {"BASE-ATTN-1": _arm(Q_NLL, "BASE-ATTN-1"), "BASE-ATTN-2": _arm(Q_NLL, "BASE-ATTN-2"),
            "MERGED-Q": _arm(REF_NLL, "MERGED-Q"), "MERGED-W": _arm(REF_NLL, "MERGED-W")}
    arms.update({k.replace("_", "-"): v for k, v in over.items()})
    return arms


def _fixture(fx="nodq", arms=None, ref=None, prepare=None):
    return R.reduce_fixture(fx, arms or _arms(), _d0(fx), ref or _ref(), PREP if prepare is None else prepare)


def _cell(fx, x):
    return _fixture(fx, _arms(MERGED_Q=x))["cells"]["MERGED-Q"]


def test_full_retention_passes_and_none_fails_below_ref():
    v = _fixture()
    assert v["status"] == "READ" and [c["status"] for c in v["cells"].values()] == ["PASS", "PASS"]
    c = v["cells"]["MERGED-Q"]
    assert c["R"] == 1.0 and c["gap"] == 0.0 and c["gap_side"] is None and c["unexplained_speedup"] == []
    c = _cell("nodq", _arm(Q_NLL))
    assert c["status"] == "FAIL" and c["R"] == 0.0 and not c["gates"]["retention"] and c["gap_side"] == R.BELOW_REF


@pytest.mark.parametrize("row_nll, passes", [(181.0, True), (181.002, False)])   # R = 19/20 = 0.95 exactly, and less
def test_the_retention_floor_is_inclusive(row_nll, passes):
    c = _cell("nodq", _arm(row_nll))
    assert c["gates"]["retention"] is passes and c["R_ci90"][0] == c["R"]   # identical rows: the CI is the point


@pytest.mark.parametrize("fx", ["nodq", "dq"])
def test_the_gap_margin_is_two_sided_and_names_the_side(fx):
    m = R.MARGINS[fx]
    for k in (0.9999, -0.9999):
        assert _cell(fx, _arm(REF_NLL + N_TOK * m * k))["gates"]["gap"]
    low = _cell(fx, _arm(REF_NLL + N_TOK * m * 1.0001))
    assert low["status"] == "FAIL" and low["gap_side"] == R.BELOW_REF and low["gap"] == pytest.approx(1.0001 * m)
    high = _cell(fx, _arm(REF_NLL - N_TOK * m * 1.0001))       # R just above 1 + M / 0.4 (this record's 0.4-nat D)
    assert high["gates"]["retention"] and high["R_ci90"][0] > 1.04       # the CI lower bound does not rescue it
    assert high["status"] == "FAIL" and high["gap_side"] == R.BEYOND_REF and "artifact to explain" in high["gap_side_text"]


def test_the_canary_is_pooled_exact_counts_with_one_point_of_slack():
    # N = 200 and BASE-ATTN-1 agrees with Q everywhere (a_B = 200): 100 a_X >= 100 x 200 - 200 iff a_X >= 198
    for agree, passes in ((198, True), (197, False)):
        x = _arm(REF_NLL)
        for k in range(ROWS * N_TOK - agree):
            x["rows"][k // N_TOK]["argmax"][k % N_TOK] = 1
        c = _cell("nodq", x)
        assert c["canary_agree"] == agree and c["gates"]["canary"] is passes


def _costed(width, metric, factor, vram=10 ** 9, b2=None):
    cost = json.loads(json.dumps(COST))
    cost["cost"][width][metric] *= factor
    cost["peak_vram_bytes"] = vram
    arms = _arms(MERGED_Q=_arm(REF_NLL, cost=cost))
    if b2 is not None:
        c2 = json.loads(json.dumps(COST))
        c2["cost"][width][metric] *= b2
        arms["BASE-ATTN-2"] = _arm(Q_NLL, "BASE-ATTN-2", cost=c2)
    return _fixture(arms=arms)["cells"]["MERGED-Q"]


def test_cost_is_slower_only_within_the_self_pair_band_floored_at_three_percent():
    assert _costed("W16", "tok_s", 0.975)["gates"]["cost"]               # 2.5 % slower: inside the 3 % floor
    assert not _costed("W16", "tok_s", 0.96)["gates"]["cost"]            # 4 % slower
    assert not _costed("W1", "itl_s", 1.04)["gates"]["cost"]             # a larger ITL is slower
    assert _costed("W16", "tok_s", 0.955, b2=1.02)["gates"]["cost"]     # the self-pair's 2 % widens the band to 5 %
    fast = _costed("W16", "tok_s", 1.05)                                 # faster passes, the speedup named
    assert fast["gates"]["cost"] and fast["unexplained_speedup"] == ["W16.tok_s"]
    fast = _costed("W1", "itl_s", 0.96)
    assert fast["gates"]["cost"] and fast["unexplained_speedup"] == ["W1.itl_s"]
    assert _costed("W1", "itl_s", 1.0, vram=int(0.991e9))["gates"]["cost"]
    assert not _costed("W1", "itl_s", 1.0, vram=int(1.011e9))["gates"]["cost"]
    assert not _costed("W1", "itl_s", 1.0, vram=int(0.989e9))["gates"]["cost"]   # VRAM within 1 % either way


def test_writer_refused_and_not_read_are_named():
    arms = _arms()
    del arms["MERGED-W"]
    prep = json.loads(json.dumps(PREP))
    prep["fixtures"]["nodq"]["cells"]["MERGED-Q"] = {"status": "WRITER_REFUSED", "why": "self-check"}
    v = _fixture(arms=arms, prepare=prep)
    assert v["cells"]["MERGED-Q"]["status"] == "WRITER_REFUSED" and v["cells"]["MERGED-W"]["status"] == "NOT_READ"
    assert _fixture(arms={"BASE-ATTN-1": arms["BASE-ATTN-1"]})["status"] == "NOT_READ"
    with pytest.raises(SystemExit, match="not the registered"):
        R.reduce_fixture("dq", _arms(), _d0("nodq"), _ref(), PREP)          # nodq's improvement under dq's margin


# gate 5, completeness: every way a record can be short, each one a VOID (removing any one check fails its case)
def _trunc_argmax(r):
    r["rows"][0]["argmax"] = r["rows"][0]["argmax"][:1]


def _set_row(key, value):
    return lambda r: r["rows"][1].__setitem__(key, value)


def _drop_cost(width, metric=None):
    return lambda r: (r["cost"].pop(width) if metric is None else r["cost"][width].pop(metric))


INCOMPLETE = {
    "argmax truncated": _trunc_argmax,
    "argmax not ints": lambda r: r["rows"][2].__setitem__("argmax", [0.0] * N_TOK),
    "nll None": _set_row("nll", None),
    "nll NaN": _set_row("nll", float("nan")),
    "nll bool": _set_row("nll", True),
    "n changed": _set_row("n", N_TOK + 1),
    "a row dropped": lambda r: r["rows"].pop(),
    "cost empty": lambda r: r.__setitem__("cost", {}),
    "cost W16 missing": _drop_cost("W16"),
    "cost W1 tok_s missing": _drop_cost("W1", "tok_s"),
    "cost zero": lambda r: r["cost"]["W16"].__setitem__("itl_s", 0.0),
    "peak VRAM missing": lambda r: r.pop("peak_vram_bytes"),
}


@pytest.mark.parametrize("why", sorted(INCOMPLETE))
@pytest.mark.parametrize("arm", ["BASE-ATTN-1", "BASE-ATTN-2", "MERGED-Q"])
def test_an_incomplete_record_is_void_never_read(arm, why):
    arms = _arms()
    INCOMPLETE[why](arms[arm])
    v = _fixture(arms=arms)
    if arm == "MERGED-Q":                                       # the cell is VOID, the rest of the fixture reads
        assert v["status"] == "READ" and v["cells"]["MERGED-Q"]["status"] == "VOID"
        assert v["cells"]["MERGED-W"]["status"] == "PASS"
    else:
        assert v["status"] == "VOID" and v["why"].startswith(arm) and not v["cells"]


def test_the_reviewers_truncated_base_canary_is_void_not_a_lower_floor():
    arms = _arms()
    for r in arms["BASE-ATTN-1"]["rows"]:
        r["argmax"] = r["argmax"][:1]                          # 1 of 50 positions a row: a_B would read 4, not 200
    assert _fixture(arms=arms)["status"] == "VOID"


@pytest.mark.parametrize("key", ["ref_argmax", "q_argmax"])
def test_a_short_ref_argmax_file_voids_the_fixture(key):
    ref = _ref()
    ref["rows"][3][key] = ref["rows"][3][key][:-1]
    v = _fixture(ref=ref)
    assert v["status"] == "VOID" and "REF argmax row 3" in v["why"]


# gate 5, engagement: an arm served on another path is VOID
def _served_set(key, value):
    return lambda r: r["served"].__setitem__(key, value)


WRONG_PATH = {
    ("BASE-ATTN-2", "served a merged arena"): _served_set("merged_arena", {"source": "arena"}),
    ("BASE-ATTN-1", "not the attention-only adapter"): _served_set("adapter_id", IDS["adapter"]),
    ("MERGED-Q", "the other writer source"): _served_set("merged_arena", {"source": "checkpoint", "adapter_id": IDS["adapter"]}),
    ("MERGED-W", "no merged arena"): _served_set("merged_arena", None),
    ("MERGED-Q", "a block for another adapter"): _served_set("merged_arena", {"source": "arena", "adapter_id": "b" * 64}),
    ("MERGED-Q", "another served adapter"): _served_set("adapter_id", IDS["attention_only"]),
    ("MERGED-W", "the expert base not verified (merged)"): _served_set("fingerprints", {"experts": {"status": "verified"}}),
    ("MERGED-Q", "another placement"): _served_set("placement", "solver"),
    ("MERGED-Q", "another graph status"): _served_set("graph_status", None),
    ("MERGED-Q", "no placement recorded"): lambda r: r["served"].pop("placement"),
    ("MERGED-W", "another device"): lambda r: r.__setitem__("device", "NVIDIA GeForce RTX 5090"),
    ("BASE-ATTN-2", "another device"): lambda r: r.__setitem__("device", "NVIDIA GeForce RTX 5090"),
}


@pytest.mark.parametrize("arm, why", sorted(WRONG_PATH))
def test_an_arm_served_on_another_path_is_void(arm, why):
    arms = _arms()
    WRONG_PATH[(arm, why)](arms[arm])
    v = _fixture(arms=arms)
    if arm.startswith("MERGED"):
        assert v["status"] == "READ" and v["cells"][arm]["status"] == "VOID"
        other = "MERGED-W" if arm == "MERGED-Q" else "MERGED-Q"
        assert v["cells"][other]["status"] == "PASS"
    else:
        assert v["status"] == "VOID" and v["why"].startswith(arm)


def test_without_the_derived_adapters_ids_nothing_reads():
    assert _fixture(prepare={"fixtures": {}})["status"] == "VOID"


def test_the_cli_reads_a_run_names_the_side_and_voids_an_unreadable_record(tmp_path, capsys):
    run, refd = tmp_path / "run", tmp_path / "ref"
    (run / "results").mkdir(parents=True)
    refd.mkdir()
    real = {fx: R.d0_record(str(D0), fx) for fx in ("nodq", "dq")}
    prep = {"fixtures": {fx: PREP["fixtures"]["nodq"] for fx in real}}
    (run / "prepare.json").write_text(json.dumps(prep))
    for fx, d0 in real.items():
        n = [r["REF"][1] for r in d0["rows"]]
        (refd / f"ref-argmax-{fx}.json").write_text(json.dumps(
            {"status": "OK", "rows": [{"n": k, "ref_argmax": [0] * k, "q_argmax": [0] * k} for k in n]}))
        for arm, src in (("BASE-ATTN-1", "Q"), ("BASE-ATTN-2", "Q"), ("MERGED-Q", "REF"), ("MERGED-W", "REF")):
            nll = [r[src][0] - (0.07 * k if (fx, arm) == ("nodq", "MERGED-Q") else 0) for r, k in zip(d0["rows"], n)]
            rec = {"rows": [{"nll": v, "n": k, "argmax": [0] * k} for v, k in zip(nll, n)],
                   **COST, "served": _served(arm), "device": "NVIDIA GeForce RTX 4090"}
            (run / "results" / f"{fx}-{arm}.json").write_text(json.dumps(rec))
    (run / "results" / "dq-MERGED-W.json").write_text("{not json")
    out = tmp_path / "verdict.json"
    assert R.main(["--run", str(run), "--d0", str(D0), "--ref-argmax", str(refd), "--out", str(out)]) == 0
    v = json.loads(out.read_text())["fixtures"]
    assert v["nodq"]["cells"]["MERGED-Q"]["gap_side"] == R.BEYOND_REF          # 0.07 nats a token beyond REF
    assert v["nodq"]["cells"]["MERGED-W"]["status"] == "PASS" and v["dq"]["cells"]["MERGED-Q"]["status"] == "PASS"
    assert v["dq"]["cells"]["MERGED-W"]["status"] == "VOID"
    printed = capsys.readouterr().out
    assert "BEYOND_REF" in printed and "D0S dq: READ" in printed


# ---- the REF tool -----------------------------------------------------------------------------------------------------


def _torch():
    for need in ("bitsandbytes", "safetensors", "transformers"):
        pytest.importorskip(need)
    return pytest.importorskip("torch")


L, E, H, INTER, NH, V, RANK = 1, 4, 128, 64, 4, 64, 2


def _tiny_olmoe(tmp, live_expert_b: bool):
    """A one-layer OLMoE-shaped checkpoint under D0's tensor names, its fixture-base (CPU NF4 like make_synth.py), an
    adapter (attention and expert LoRA) and 6 held-out rows."""
    t = _torch()
    from safetensors.torch import save_file
    from transformers import OlmoeConfig
    import bitsandbytes.functional as F
    g = t.Generator().manual_seed(0)
    rnd = lambda *s: (t.randn(*s, generator=g) * 0.05).to(t.bfloat16)  # noqa: E731
    ck = tmp / "ckpt"
    ck.mkdir()
    OlmoeConfig(hidden_size=H, intermediate_size=INTER, num_experts=E, num_experts_per_tok=2, num_attention_heads=NH,
                num_key_value_heads=NH, num_hidden_layers=L, vocab_size=V, norm_topk_prob=False).to_json_file(
        str(ck / "config.json"))
    W = {"model.embed_tokens.weight": rnd(V, H), "model.norm.weight": t.ones(H, dtype=t.bfloat16), "lm_head.weight": rnd(V, H)}
    for li in range(L):
        p = f"model.layers.{li}."
        W.update({p + "input_layernorm.weight": t.ones(H, dtype=t.bfloat16),
                  p + "post_attention_layernorm.weight": t.ones(H, dtype=t.bfloat16),
                  p + "self_attn.q_norm.weight": t.ones(H, dtype=t.bfloat16),
                  p + "self_attn.k_norm.weight": t.ones(H, dtype=t.bfloat16), p + "mlp.gate.weight": rnd(E, H)})
        W.update({p + f"self_attn.{n}_proj.weight": rnd(H, H) for n in "qkvo"})
        for e in range(E):
            W.update({p + f"mlp.experts.{e}.gate_proj.weight": rnd(INTER, H), p + f"mlp.experts.{e}.up_proj.weight": rnd(INTER, H),
                      p + f"mlp.experts.{e}.down_proj.weight": rnd(H, INTER)})
    save_file(W, str(ck / "model.safetensors"))
    (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in W}}))
    fb = tmp / "fixture-base"
    fb.mkdir()
    stacks = {}
    for li in range(L):
        for which, (n, k) in (("gate_up", (2 * INTER, H)), ("down", (H, INTER))):
            packed, absmax = [], []
            for e in range(E):
                p = f"model.layers.{li}.mlp.experts.{e}."
                w = (t.cat([W[p + "gate_proj.weight"], W[p + "up_proj.weight"]]) if which == "gate_up"
                     else W[p + "down_proj.weight"]).float()
                q, st = F.quantize_4bit(w.contiguous(), blocksize=64, quant_type="nf4", compress_statistics=False)
                packed.append(q.reshape(-1))
                absmax.append(st.absmax.reshape(-1))
            stacks[f"{li}.{which}_proj"], stacks[f"{li}.{which}_absmax"] = t.stack(packed), t.stack(absmax)
    save_file(stacks, str(fb / "stacks.safetensors"))
    ad = tmp / "adapter"
    ad.mkdir()
    A = {}
    for li in range(L):
        for n in "qkvo":
            A[f"base_model.model.model.layers.{li}.self_attn.{n}_proj.lora_A"] = t.randn(RANK, H, generator=g) * 0.05
            A[f"base_model.model.model.layers.{li}.self_attn.{n}_proj.lora_B"] = t.randn(H, RANK, generator=g) * 0.05
        for which, (n, k) in (("gate_up", (2 * INTER, H)), ("down", (H, INTER))):
            A[f"model.layers.{li}.mlp.experts.{which}_lora_A"] = t.randn(E, RANK, k, generator=g)
            A[f"model.layers.{li}.mlp.experts.{which}_lora_B"] = (t.randn(E, n, RANK, generator=g) * 0.2 if live_expert_b
                                                                else t.zeros(E, n, RANK))
    save_file(A, str(ad / "adapter.safetensors"))
    (ad / "adapter_manifest.json").write_text(json.dumps({"setup": {"r": RANK, "alpha": 2 * RANK}}))
    held = tmp / "heldout.json"
    held.write_text(json.dumps({"rows": [{"ids": t.randint(0, V, (int(n),), generator=g).tolist()}
                                         for n in t.randint(6, 14, (6,), generator=g)]}))
    return ck, fb, ad, held


def _ref_tool():
    sys.path.insert(0, str(D0 / "tools"))
    return _load("d0s_ref")


@pytest.fixture(scope="module")
def ref_tool():
    _torch()
    return _ref_tool()


def test_d0s_tools_match_their_pin_and_a_changed_one_refuses(ref_tool, tmp_path):
    ref_tool.check_tools(str(D0 / "tools"))
    bad = tmp_path / "tools"
    shutil.copytree(D0 / "tools", bad)
    with open(bad / "d0_forward.py", "a") as f:
        f.write("\n# changed\n")
    with pytest.raises(SystemExit, match="differ from their SHA256SUMS"):
        ref_tool.check_tools(str(bad))


@pytest.mark.parametrize("live", [False, True])
def test_the_walk_scores_every_position_and_ref_equals_q_without_an_expert_delta(ref_tool, tmp_path, live):
    ck, fb, ad, held = _tiny_olmoe(tmp_path, live)
    got = ref_tool.walk(str(ck), str(fb), False, str(ad), str(held))
    rows = json.load(open(held))["rows"]
    for arm in ("Q", "REF"):
        assert [n for _nll, n, _am in got[arm]] == [len(r["ids"]) - 1 for r in rows]
        assert all(len(am) == n and all(0 <= a < V for a in am) for _nll, n, am in got[arm])
    same = all(a[0] == b[0] and a[2] == b[2] for a, b in zip(got["Q"], got["REF"]))
    assert same is (not live)


def test_the_cli_reproduces_a_record_or_says_void(ref_tool, tmp_path):
    ck, fb, ad, held = _tiny_olmoe(tmp_path, True)
    got = ref_tool.walk(str(ck), str(fb), False, str(ad), str(held))

    def arms_csv(path, bump=0.0):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "row", "nll", "n", "dmax", "dmean"])
            for arm in ("Q", "REF"):
                for r, (nll, n, _am) in enumerate(got[arm]):
                    w.writerow([arm, r, nll * (1 + (bump if arm == "REF" and r == 0 else 0)), n, 0, 0])
    for bump, rc, status in ((0.0, 0, "OK"), (1e-4, 3, "VOID")):
        csvp = tmp_path / f"arms-{bump}.csv"
        arms_csv(csvp, bump)
        out = tmp_path / f"ref-{bump}.json"
        assert ref_tool.main(["--ckpt", str(ck), "--fixture-base", str(fb), "--adapter", str(ad), "--heldout", str(held),
                              "--arms-csv", str(csvp), "--out", str(out)]) == rc
        rec = json.loads(out.read_text())
        assert rec["status"] == status and len(rec["rows"]) == 6
        assert [r["ref_argmax"] for r in rec["rows"]] == [am for _nll, _n, am in got["REF"]]


# ---- the attention-only adapter ----------------------------------------------------------------------------------------


def test_the_attention_only_adapter_zeroes_every_expert_b_and_keeps_the_rest(tmp_path):
    t = _torch()
    from safetensors.torch import save_file

    from experts4bit_qlora.adapter_dir import servable_refusals, verify_adapter_dir
    box = _load("d0s_box")
    src = tmp_path / "src"
    src.mkdir()
    g = t.Generator().manual_seed(1)
    tensors = {"model.layers.0.self_attn.q_proj.lora_A": t.randn(2, 128, generator=g),
               "model.layers.0.self_attn.q_proj.lora_B": t.randn(128, 2, generator=g),
               "model.layers.0.mlp.experts.gate_up_lora_A": t.randn(3, 2, 128, generator=g),
               "model.layers.0.mlp.experts.gate_up_lora_B": t.randn(3, 128, 2, generator=g),
               "model.layers.0.mlp.experts.down_lora_A": t.randn(3, 2, 64, generator=g),
               "model.layers.0.mlp.experts.down_lora_B": t.randn(3, 128, 2, generator=g)}
    save_file(tensors, str(src / "adapter.safetensors"))
    body = (src / "adapter.safetensors").read_bytes()
    fp = {"scheme_id": "qfp-v1", "sha256": "a" * 64, "modules": 2, "seconds": 0.0}
    man = {"schema": "backend-adapter/2", "backend": "experts4bit", "weights": "adapter.safetensors",
           "files": {"adapter.safetensors": {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}},
           "resolved_model_revision": "c" * 40, "setup": {"quant_type": "nf4", "blocksize": 64, "r": 2, "alpha": 4},
           "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
           "expert_absmax": {"form": "fp32"}, "train_fused_qkv": 0, "attn_quant": {"form": "bf16"}, "versions": {},
           "lora_scale_applied": {"attention": 2.0, "experts": 2.0},
           "fingerprints": {"experts": fp, "attention": dict(fp, sha256="b" * 64)}, "run_id": "r",
           "created_at": "2026-10-24T00:00:00Z"}
    (src / "adapter_manifest.json").write_text(json.dumps(man))
    rec = box.attention_only(str(src), str(tmp_path / "attn"))
    a, b = verify_adapter_dir(src), verify_adapter_dir(tmp_path / "attn")
    assert rec["source_adapter_id"] == a.adapter_id != b.adapter_id == rec["adapter_id"] and rec["zeroed"] == 2
    assert b.manifest["fingerprints"] == a.manifest["fingerprints"]
    assert all(not bool(t.any(b.tensors[n] != 0)) for n in b.expert_tensors if n.endswith("_lora_B"))
    assert t.equal(b.tensors["model.layers.0.self_attn.q_proj.lora_B"], a.tensors["model.layers.0.self_attn.q_proj.lora_B"])
    live = [c for c, _ in servable_refusals(a, served_revision=None, requested_revision=None, arena_index=None,
                                            levers={}, check_served_revision=False)]
    attn = [c for c, _ in servable_refusals(b, served_revision=None, requested_revision=None, arena_index=None,
                                            levers={}, check_served_revision=False)]
    assert 1 in live and 1 not in attn                            # attention only: no live expert LoRA on the base arena
    with pytest.raises(SystemExit, match="exists"):
        box.attention_only(str(src), str(tmp_path / "attn"))


# ---- the pins and the driver -------------------------------------------------------------------------------------------


def test_every_staged_file_equals_its_pin():
    pins = [ln.split("  ", 1) for ln in (D0S / "staged.sha256").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    for want, name in pins:
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == want, name
    names = [name for _want, name in pins]
    for must in ("bench/d0s/d0s_box.py", "bench/d0s/d0s_run.sh", "tests/test_merged_arena_gpu.py",
                 "bench/smoke/gpu_serve_smoke.py", "bench/d0/receipts/d0-p224s3/nodq/heldout.json",
                 "bench/d0/receipts/d0-p224s3/dq/heldout.json"):
        assert must in names, must


def _drive(tmp_path, **env):
    full = {k: v for k, v in os.environ.items() if not k.startswith(("E4B_", "D0S_"))}
    full.update({"E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q",
                 "E4B_RENT_RUN_DIR": str(tmp_path / "run"), "E4B_RENT_RUN_ID": "d0s-test-1",
                 "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1", "D0S_DRIVE_DRYRUN": "1",
                 "E4B_SHA": "b" * 40, **env})
    r = subprocess.run(["bash", str(D0S / "d0s_drive.sh")], cwd=ROOT, env=full, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


def _bash_has_python3() -> bool:
    return shutil.which("bash") is not None and subprocess.run(["bash", "-c", "command -v python3"],
                                                               capture_output=True).returncode == 0


def _need_driver(ready: bool, ci) -> None:
    """The driver's tests need bash with python3 (its nonce). A machine without it skips them, except under CI=true,
    where it FAILS: CI never passes the driver's tests by skipping them."""
    if ready:
        return
    why = "the driver's tests need bash with python3 (the driver's nonce)"
    if str(ci or "").strip().lower() not in ("", "0", "false"):
        pytest.fail(why)
    pytest.skip(why)


@pytest.mark.parametrize("ci", ["true", "1", "yes"])
def test_a_driver_test_without_python3_fails_under_ci(ci):
    # pytest.raises(pytest.fail.Exception) would let a skip escape (Skipped is its own outcome) and report this test
    # SKIPPED, so a regression to skipping is caught here explicitly as a failure
    try:
        _need_driver(False, ci)
    except pytest.skip.Exception:
        pytest.fail(f"skipped under CI={ci} instead of failing")
    except pytest.fail.Exception as e:
        assert "python3" in str(e)
    else:
        pytest.fail(f"neither failed nor skipped under CI={ci}")
    _need_driver(True, ci)


@pytest.mark.parametrize("ci", [None, "", "0", "false"])
def test_a_driver_test_without_python3_skips_outside_ci(ci):
    try:
        _need_driver(False, ci)
    except pytest.skip.Exception as e:
        assert "python3" in str(e)
    else:
        raise AssertionError(f"did not skip outside CI (CI={ci!r})")


def test_the_driver_refuses_a_reading_without_adapters_and_dry_runs_the_shake(tmp_path):
    _need_driver(_bash_has_python3(), os.environ.get("CI"))
    rc, out = _drive(tmp_path)
    assert rc == 78 and "D0S_ADAPTERS" in out, out
    rc, out = _drive(tmp_path, D0S_SHAKE="1")
    assert rc == 0 and "DRYRUN stage" in out and "D0S_SHAKE=1" in out, out
