#!/usr/bin/env python3
"""p115_reduce.py -- lane P115's registered rule (bench/p115/PREREG-p115.md; e4b#1313), from the run directory.

Receipts: the speed phase's ``arm_F0a.json``, ``arm_F1a.json``, ``arm_F1b.json``, ``arm_F0b.json`` (run in that order,
ABBA, each from ``p115_box.py``) and the quality phase's ``quality_off.json`` and ``quality_on.json``
(``p115_quality.py``). F0 is the default server with the four fusion knobs unset; F1 is the registered B=1 fused stack
(``E4B_PAGED_FUSE_QKV=1`` + the three folds; the three folds only under the proving run).

The verdict is the first of these that applies:
  VOID          an arm or a quality record is missing or not ok; a record names another e4b / grouped-nf4-gemm commit or
                model revision; the arms read different prompts or ran different lengths; a slope is void; or
                engagement is wrong:
                - speed: every arm reports "graph" for every bucket; F0's fusion census is all zero and ``fuse_qkv``
                  false; F1's census is ``census_for(family, layers, fused)`` and ``fuse_qkv`` as registered;
                - quality: a text short of its registered windows (``rep``: one group), the two phases scored
                  different windows, a pass with the wrong decode attention calls, grouping flag or bucket statistics
                  (every decode step an eager padded step of the group's bucket, no replays), an OFF pass that called
                  a glue kernel, an ON pass whose kernel calls are not ``per_step(family, layers)`` x the decode steps,
                  an ON pass whose ``qkv_proj`` calls are not layers x forwards (fused), the OFF / ON census wrong; or
                  ``mutant_scale`` passes the bar on either text (the gate cannot fail).
  NOISY         a self-pair (F0b/F0a or F1b/F1a, decode tok/s, either workload) falls outside [0.96, 1.04].
  FUNCTION_FAIL F0b's tokens differ from F0a's or F1b's from F1a's on any row of either workload at either length, or an
                arm's timed reps do not all digest the same. (F1 differing from F0 is expected and reported.)
  QUALITY_FAIL  on either text, ON fails P110's bar against the floor (mean d_ON <= B_floor + 0.01 nats and
                mean |d_ON| <= 2 x max(S_floor, 0.005)); or on wikitext the K8 perplexity moves by more than 0.05
                (two-sided, as ``experts4bit_qlora.k8_gate`` gates an uncalibrated arm). c4val1's K8 move is reported.
  SLOWER        g1 = min(F1a/F0a, F1b/F0b) on W1 < 1.10, or g16 = the same on W16 < 1.00.
  DEFAULT_AUTO  otherwise.

The floor is the quality phase's ``half`` and ``chunk`` under R's arithmetic, plus ``rep`` if R did not repeat bit for
bit: ``B_floor`` = the largest |mean d_f|, ``S_floor`` = the largest mean |d_f| (P110's definitions).

Reported beside the verdict: the pair ratios, their geometric means, ms per step for every arm, F1-vs-F0 token agreement,
every quality arm's bias / spread / SE / max |d| / KL / argmax agreement per text, both texts' K8 perplexities.

    python p115_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA]
    python p115_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

TAGS = ("F0a", "F1a", "F1b", "F0b")
BUCKETS = (1, 2, 4, 8, 16)
SELF_LO, SELF_HI = 0.96, 1.04
GAIN_MIN_W1, GAIN_MIN_W16 = 1.10, 1.00
TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005
K8_BUDGET = 0.05
K8_GATED = ("wikitext",)
TEXTS = ("wikitext", "c4val1")
GNF4_SHA = "b4f93f1c62d1e3436ed45bec8ccd608c90433737"
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}
FAMILY = {"Qwen/Qwen3-30B-A3B": ("qwen3", 48), "ibm-granite/granite-3.1-3b-a800m-instruct": ("granite", 32)}
WINDOWS = {"Qwen/Qwen3-30B-A3B": 48, "ibm-granite/granite-3.1-3b-a800m-instruct": 12}
OFF_ARMS = ("R", "rep", "half", "chunk", "mutant_scale")
FLOORS = ("half", "chunk")
KERNELS = ("rmsnorm_rows", "rmsnorm_resid_rows", "scaled_resid_add_rows", "rope_norm_heads", "rope_heads",
           "router_epilogue")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")


def census_for(family: str, layers: int, fused: bool) -> dict:
    """The fusion census ``serve_paged._apply_fusions`` reports with the fused stack on (SC1's 95 receipts: Qwen3-30B
    ``48 / 193 / [48, 48] / 48``; Granite-3.1-3b ``0 / 65 / [32, 32] / 32``). Glue round 1 patches every plain RMSNorm:
    Qwen3's input, post-attention, q and k norms per layer plus the final norm; Granite's attention has no q/k norms."""
    L = int(layers)
    if family == "qwen3":
        return {"fuse_qkv_n": L if fused else 0, "fuse_t1_glue_n": 4 * L + 1, "fuse_t1_glue_r2_n": [L, L],
                "fuse_router_epilogue_n": L}
    if family == "granite":
        return {"fuse_qkv_n": 0, "fuse_t1_glue_n": 2 * L + 1, "fuse_t1_glue_r2_n": [L, L], "fuse_router_epilogue_n": L}
    raise ValueError(family)


def per_step(family: str, layers: int) -> dict:
    """Glue-kernel calls per decode step with the fused stack on (the nonzero ones). Qwen3: the input norms and the final
    norm (round 1), the residual + post-attention norm fold, q/k norm + rotary per attention, one router epilogue per MoE
    layer. Granite: its scaled residual fold and scaled residual add, rotary without a norm."""
    L = int(layers)
    if family == "qwen3":
        return {"rmsnorm_rows": L + 1, "rmsnorm_resid_rows": L, "rope_norm_heads": 2 * L, "router_epilogue": L}
    if family == "granite":
        return {"rmsnorm_rows": L + 1, "rmsnorm_resid_rows": L, "scaled_resid_add_rows": L, "rope_heads": 2 * L,
                "router_epilogue": L}
    raise ValueError(family)


def _zero(v) -> bool:
    return all(x == 0 for x in v) if isinstance(v, (list, tuple)) else v == 0


def _bucket(n: int) -> int:
    return min(b for b in BUCKETS if b >= n)


def _groups(n: int, group: int) -> list:
    return [min(group, n - g0) for g0 in range(0, n, group)]


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def _agree(a, b):
    """Rows identical, and the first diverging position of every row (None where identical)."""
    first = []
    for x, y in zip(a, b):
        i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
        first.append(i if i is not None or len(x) == len(y) else min(len(x), len(y)))
    return {"rows_identical": sum(f is None for f in first), "rows": len(first), "first_divergence": first}


def _stats(per, arm, ref):
    d = [x["nll"] - ref[x["window"]] for x in per.get(arm, [])]
    n = len(d)
    if not n:
        return {"n": 0}
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    kls = [x["kl"] for x in per[arm] if "kl" in x]
    return {"n": n, "bias": mean, "spread": sum(abs(v) for v in d) / n, "se": sd / math.sqrt(n),
            "max_abs": max(abs(v) for v in d), "mean_kl": sum(kls) / len(kls) if kls else None,
            "argmax_agree": sum(x["argmax_agree"] for x in per[arm]) / n}


def _passes(s, fl):
    return s["bias"] <= fl["B_floor"] + TOL and s["spread"] <= SPREAD_X * max(fl["S_floor"], SPREAD_MIN)


def _ppl(per_r):
    return math.exp(sum(x["nll"] for x in per_r) / len(per_r))


def speed_faults(arms: dict, e4b_sha: str) -> list:
    void = []
    for t, r in arms.items():
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"{t}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{t}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(r.get("model")) != r.get("revision"):
            void.append(f"{t}: model {r.get('model')}@{r.get('revision')} is not a registered revision")
            continue
        g = r.get("graph_status") or {}
        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        fam, L = FAMILY[r["model"]]
        fused = not r.get("prove")
        f = r.get("fusions") or {}
        if t.startswith("F0"):
            if r.get("fuse_qkv") or not all(_zero(f.get(k, 1)) for k in CENSUS_KEYS):
                void.append(f"{t}: the default arm reports fusions {f} (fuse_qkv={r.get('fuse_qkv')})")
        else:
            want = census_for(fam, L, fused)
            if {k: f.get(k) for k in CENSUS_KEYS} != want or bool(r.get("fuse_qkv")) != fused:
                void.append(f"{t}: fusions {f} (fuse_qkv={r.get('fuse_qkv')}), registered {want} (fuse_qkv={fused})")
    if len({r.get("prompts_sha256") for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r.get("short"), r.get("long"), r.get("reps")) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    if len({bool(r.get("prove")) for r in arms.values()}) != 1:
        void.append("the arms disagree on the proving run")
    if any(_rate(arms[t], w) is None for t in TAGS for w in ("W16", "W1")):
        void.append("a decode slope is void")
    return void


def quality_faults(off: dict, on: dict, e4b_sha: str) -> list:
    void = []
    for name, rec in (("quality_off", off), ("quality_on", on)):
        if rec.get("e4b_sha") != e4b_sha:
            void.append(f"{name}: e4b {rec.get('e4b_sha')} != {e4b_sha}")
        if rec.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{name}: grouped-nf4-gemm {rec.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(rec.get("model")) != rec.get("revision"):
            void.append(f"{name}: model {rec.get('model')}@{rec.get('revision')} is not a registered revision")
    if void:
        return void
    if off.get("model") != on.get("model") or off.get("prove") != on.get("prove"):
        return [f"the phases ran different subjects: {off.get('model')} / {on.get('model')}"]
    fam, L = FAMILY[off["model"]]
    fused = not off.get("prove")
    want_n = WINDOWS[off["model"]]
    C, group = off["cont"], off["group"]
    for k in ("cont", "group", "prompt", "chunk", "floor_chunk", "layers", "windows_sha256"):
        if off.get(k) != on.get(k):
            void.append(f"the phases differ in {k}: {off.get(k)} / {on.get(k)}")
    if off.get("layers") != L:
        void.append(f"{off.get('layers')} layers, the family registers {L}")
    for name, rec, arms in (("quality_off", off, OFF_ARMS), ("quality_on", on, ("ON",))):
        cen = rec.get("census") or {}
        if name == "quality_off":
            if rec.get("fuse_qkv") or not all(_zero(cen.get(k, 1)) for k in CENSUS_KEYS):
                void.append(f"quality_off: census {cen}")
        elif {k: cen.get(k) for k in CENSUS_KEYS} != census_for(fam, L, fused) or bool(rec.get("fuse_qkv")) != fused:
            void.append(f"quality_on: census {cen}, registered {census_for(fam, L, fused)}")
        for t in TEXTS:
            per = (rec.get("per_window") or {}).get(t, {})
            for arm in arms:
                n = len(per.get(arm, []))
                want = group if arm == "rep" else want_n
                if n != want:
                    void.append(f"{name} {t}: arm {arm} has {n} windows, expected {want}")
            sizes = _groups(want_n, group)
            for arm in arms:
                passes = (rec.get("engagement") or {}).get(t, {}).get(arm, [])
                arm_sizes = sizes[:1] if arm == "rep" else sizes
                if len(passes) != len(arm_sizes):
                    void.append(f"{name} {t} {arm}: {len(passes)} passes, expected {len(arm_sizes)}")
                    continue
                for k, (e, n) in enumerate(zip(passes, arm_sizes)):
                    void += [f"{name} {t} {arm} pass {k}: {m}" for m in _pass_faults(e, arm, n, C, L, fam, fused)]
    return void


def _pass_faults(e, arm, n, C, L, fam, fused) -> list:
    out = []
    halves = arm == "half"
    want_calls = (C - 1) * L * (2 if halves else 1)
    if e.get("decode_calls") != want_calls:
        out.append(f"{e.get('decode_calls')} decode attention calls, expected {want_calls}")
    if not (e.get("grouping_flags_in_pass") or {}).get("device_grouping"):
        out.append("device grouping off")
    st = e.get("graph_status") or {}
    if [st.get(str(b)) for b in BUCKETS] != ["eager: capture=False"] * len(BUCKETS):
        out.append(f"graph_status {st}")
    parts = [n // 2, n - n // 2] if halves else [n]
    want = {}
    for p in parts:
        b = _bucket(p)
        w = want.setdefault(str(b), {"eager_steps": 0, "pad_rows": 0})
        w["eager_steps"] += C - 1
        w["pad_rows"] += (b - p) * (C - 1)
    gs = e.get("graph_stats") or {}
    for b, w in want.items():
        got = gs.get(b, {})
        if (got.get("eager_steps"), got.get("pad_rows")) != (w["eager_steps"], w["pad_rows"]):
            out.append(f"bucket {b} stats {got}, expected {w}")
    if any(v.get("replays", 0) for v in gs.values()):
        out.append(f"a replay in a capture=False pass: {gs}")
    k = e.get("kernels") or {}
    if arm == "ON":
        steps = (C - 1) * (2 if halves else 1)
        want_k = {name: per_step(fam, L).get(name, 0) * steps for name in KERNELS}
        if {name: k.get(name, 0) for name in KERNELS} != want_k:
            out.append(f"kernel calls {k}, expected {want_k}")
        want_q = L * e.get("forwards", -1) if fused else 0
        if e.get("qkv_calls") != want_q:
            out.append(f"qkv_proj calls {e.get('qkv_calls')}, expected {want_q} ({e.get('forwards')} forwards)")
    elif any(k.get(name, 0) for name in KERNELS) or e.get("qkv_calls"):
        out.append(f"the unfused arm called a glue kernel: {k}, qkv_proj {e.get('qkv_calls')}")
    return out


def quality_read(off: dict, on: dict) -> dict:
    out = {}
    for t in TEXTS:
        ref = {x["window"]: x["nll"] for x in off["per_window"][t]["R"]}
        st = {a: _stats(off["per_window"][t], a, ref) for a in OFF_ARMS if a != "R"}
        st["ON"] = _stats(on["per_window"][t], "ON", ref)
        draws = list(FLOORS) + ([] if (off.get("rep_identical") or {}).get(t) else ["rep"])
        fl = {"draws": draws, "B_floor": max(abs(st[f]["bias"]) for f in draws),
              "S_floor": max(st[f]["spread"] for f in draws)}
        ppl_r, ppl_on = _ppl(off["per_window"][t]["R"]), _ppl(on["per_window"][t]["ON"])
        out[t] = {"stats": st, "floor": fl, "bias_bar": fl["B_floor"] + TOL,
                  "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN), "mutant_passes": _passes(st["mutant_scale"], fl),
                  "on_passes": _passes(st["ON"], fl), "k8": {"ppl_R": ppl_r, "ppl_ON": ppl_on, "delta": ppl_on - ppl_r,
                                                             "gated": t in K8_GATED,
                                                             "passes": abs(ppl_on - ppl_r) <= K8_BUDGET}}
    return out


def reduce(arms: dict, off: dict | None, on: dict | None, e4b_sha: str) -> dict:
    out = {"lane": "P115", "verdict": None, "reasons": [],
           "bars": {"self_pair": [SELF_LO, SELF_HI], "gain_min_w1": GAIN_MIN_W1, "gain_min_w16": GAIN_MIN_W16,
                    "tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN, "k8_budget": K8_BUDGET,
                    "k8_gated": list(K8_GATED)}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    missing += [n for n, r in (("quality_off", off), ("quality_on", on)) if not r]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = speed_faults(arms, e4b_sha) + quality_faults(off, on, e4b_sha)
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    q = quality_read(off, on)
    out["quality"] = q
    void = [f"{t}: mutant_scale passes the bar (the gate cannot fail)" for t in TEXTS if q[t]["mutant_passes"]]
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    F0a, F1a, F1b, F0b = (arms[t] for t in TAGS)
    short, long_ = F0a["short"], F0a["long"]
    rates = {t: {w: _rate(arms[t], w) for w in ("W16", "W1")} for t in TAGS}
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in ("W16", "W1")} for t in TAGS}
    selfp = {f"{a}/{b} {w}": rates[a][w] / rates[b][w] for a, b in (("F0b", "F0a"), ("F1b", "F1a")) for w in ("W16", "W1")}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    pairs = {w: [rates["F1a"][w] / rates["F0a"][w], rates["F1b"][w] / rates["F0b"][w]] for w in ("W16", "W1")}
    out["pair_ratios"] = {w: [round(x, 4) for x in v] for w, v in pairs.items()}
    out["g16"], out["g1"] = round(min(pairs["W16"]), 4), round(min(pairs["W1"]), 4)
    out["geomean"] = {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()}
    out["f1_vs_f0_tokens"] = {f"{w}@{n}": _agree(_tok(F1a, w, n), _tok(F0a, w, n)) for w in ("W16", "W1") for n in (short, long_)}
    fn = []
    for a, b in (("F0b", "F0a"), ("F1b", "F1a")):
        for w in ("W16", "W1"):
            for n in (short, long_):
                if _tok(arms[a], w, n) != _tok(arms[b], w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[a], w, n), _tok(arms[b], w, n))) if x != y]
                    fn.append(f"{a} != {b} on {w} at {n} tokens, rows {rows}")
    for t in TAGS:
        for w in ("W16", "W1"):
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    noisy = [f"{k} = {v:.4f}" for k, v in selfp.items() if not SELF_LO <= v <= SELF_HI]
    qfail = []
    for t in TEXTS:
        s = q[t]["stats"]["ON"]
        if not q[t]["on_passes"]:
            qfail.append(f"{t}: ON bias {s['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), spread {s['spread']:.5f} "
                         f"(bar {q[t]['spread_bar']:.5f})")
        if q[t]["k8"]["gated"] and not q[t]["k8"]["passes"]:
            qfail.append(f"{t}: K8 perplexity {q[t]['k8']['ppl_R']:.5f} -> {q[t]['k8']['ppl_ON']:.5f} "
                         f"(|delta| > {K8_BUDGET})")
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif qfail:
        out.update(verdict="QUALITY_FAIL", reasons=qfail)
    elif out["g1"] < GAIN_MIN_W1 or out["g16"] < GAIN_MIN_W16:
        out.update(verdict="SLOWER", reasons=[f"g1 = {out['g1']} (bar {GAIN_MIN_W1}), g16 = {out['g16']} "
                                              f"(bar {GAIN_MIN_W16})"])
    else:
        out.update(verdict="DEFAULT_AUTO", reasons=[
            f"g1 = {out['g1']}, g16 = {out['g16']} (geomean {out['geomean']['W1']} / {out['geomean']['W16']}); "
            + "; ".join(f"{t} ON bias {q[t]['stats']['ON']['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), K8 "
                        f"{q[t]['k8']['delta']:+.4f}" for t in TEXTS)])
    return out


# ------------------------------------------------------------------ self-test --

E = "a" * 40
QWEN = "Qwen/Qwen3-30B-A3B"
GRAN = "ibm-granite/granite-3.1-3b-a800m-instruct"


def _fake_arm(tag, rate16, rate1, toks, *, model=QWEN, prove=False, e4b=E, graphs=None, fusions=None, fuse_qkv=None,
              digests=None, short=4, long_=8):
    fam, L = FAMILY[model]
    f1 = tag.startswith("F1")
    if fusions is None:
        fusions = census_for(fam, L, not prove) if f1 else {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0,
                                                             "fuse_t1_glue_r2_n": [0, 0], "fuse_router_epilogue_n": 0}
    graphs = graphs or {str(b): "graph" for b in BUCKETS}

    def wl(rate):
        return {"decode_tok_s": rate, "decode_ms_per_step": round(1000 / rate, 3),
                "tokens": {str(short): [t[:short] for t in toks], str(long_): toks},
                "rep_digests": digests or {str(short): ["x"] * 3, str(long_): ["y"] * 3}}
    return {"tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": model, "revision": REVS[model],
            "prove": prove, "graph_status": graphs, "fusions": fusions,
            "fuse_qkv": (f1 and not prove) if fuse_qkv is None else fuse_qkv,
            "prompts_sha256": "p", "short": short, "long": long_, "reps": 3, "workloads": {"W16": wl(rate16), "W1": wl(rate1)}}


def _fake_pass(arm, n, C, L, fam, fused):
    halves = arm == "half"
    parts = [n // 2, n - n // 2] if halves else [n]
    gs = {str(b): {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in BUCKETS}
    for p in parts:
        b = _bucket(p)
        gs[str(b)]["eager_steps"] += C - 1
        gs[str(b)]["pad_rows"] += (b - p) * (C - 1)
    steps = (C - 1) * (2 if halves else 1)
    forwards = n + steps
    on = arm == "ON"
    return {"decode_calls": (C - 1) * L * (2 if halves else 1), "grouping_flags_in_pass": {"device_grouping": True},
            "graph_status": {str(b): "eager: capture=False" for b in BUCKETS}, "graph_stats": gs,
            "kernels": {k: (per_step(fam, L).get(k, 0) * steps if on else 0) for k in KERNELS},
            "forwards": forwards, "qkv_calls": (L * forwards if (on and fused) else 0)}


def _fake_quality(phase, *, model=QWEN, prove=False, e4b=E, bias=None, spread_noise=0.001, mutant=0.4, rep_identical=True):
    fam, L = FAMILY[model]
    fused = not prove
    n, group, C = WINDOWS[model], 12, 128 if model == QWEN else 32
    bias = bias or {}
    arms = OFF_ARMS if phase == "off" else ("ON",)
    per, eng = {}, {}
    for t in TEXTS:
        base = {w: 2.0 + 0.01 * w + (0.5 if t == "c4val1" else 0.0) for w in range(n)}
        per[t], eng[t] = {}, {}
        for arm in arms:
            off_by = {"R": 0.0, "rep": 0.0, "half": 0.0005, "chunk": -0.0004, "mutant_scale": mutant}.get(arm, bias.get(t, 0.0))
            ws = range(group) if arm == "rep" else range(n)
            per[t][arm] = [{"window": w, "nll": base[w] + off_by + (spread_noise if w % 2 else -spread_noise) * (arm != "R"),
                            "argmax_agree": 1.0, "kl": 0.001} for w in ws]
            sizes = _groups(n, group)[:1] if arm == "rep" else _groups(n, group)
            eng[t][arm] = [_fake_pass(arm, s, C, L, fam, fused) for s in sizes]
    on = phase == "on"
    census = census_for(fam, L, fused) if on else {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0],
                                                   "fuse_router_epilogue_n": 0}
    return {"model": model, "revision": REVS[model], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "prove": prove,
            "fuse_qkv": on and fused, "census": census, "phase": phase, "arms": list(arms), "group": group, "prompt": 512,
            "cont": C, "chunk": 512, "floor_chunk": 256, "layers": L, "windows_sha256": {t: t for t in TEXTS},
            "rep_identical": {t: rep_identical for t in TEXTS}, "per_window": per, "engagement": eng}


def self_test() -> int:
    base = [[r * 100 + i for i in range(8)] for r in range(16)]
    fused_toks = [list(r) for r in base]
    fused_toks[3][5] += 1                                             # F1 differs from F0: expected, reported

    def arms(**over):
        a = {"F0a": _fake_arm("F0a", 740.0, 99.4, base), "F1a": _fake_arm("F1a", 800.0, 140.0, fused_toks),
             "F1b": _fake_arm("F1b", 802.0, 140.4, fused_toks), "F0b": _fake_arm("F0b", 742.0, 99.5, base)}
        a.update(over)
        return a

    def run(a=None, off=None, on=None):
        return reduce(a or arms(), off if off is not None else _fake_quality("off"),
                      on if on is not None else _fake_quality("on"), E)

    cases = []
    r = run()
    cases.append(("default auto", r["verdict"] == "DEFAULT_AUTO"))
    cases.append(("report", r["g1"] == round(min(140 / 99.4, 140.4 / 99.5), 4)
                  and r["f1_vs_f0_tokens"]["W16@8"]["rows_identical"] == 15))
    cases.append(("missing arm", run(a={k: v for k, v in arms().items() if k != "F0b"})["verdict"] == "VOID"))
    cases.append(("missing quality", reduce(arms(), _fake_quality("off"), None, E)["verdict"] == "VOID"))
    cases.append(("wrong e4b", run(a=arms(F1a=_fake_arm("F1a", 800.0, 140.0, fused_toks, e4b="b" * 40)))["verdict"] == "VOID"))
    cases.append(("a bucket eager", run(a=arms(F1b=_fake_arm("F1b", 802.0, 140.4, fused_toks,
                                                              graphs={**{str(b): "graph" for b in BUCKETS}, "16": "eager: x"})))["verdict"] == "VOID"))
    cases.append(("F1 census short", run(a=arms(F1a=_fake_arm("F1a", 800.0, 140.0, fused_toks,
                                                               fusions={**census_for("qwen3", 48, True), "fuse_qkv_n": 0})))["verdict"] == "VOID"))
    cases.append(("F0 fused", run(a=arms(F0a=_fake_arm("F0a", 740.0, 99.4, base, fuse_qkv=True)))["verdict"] == "VOID"))
    on = _fake_quality("on")
    on["engagement"]["wikitext"]["ON"][1]["kernels"]["router_epilogue"] -= 1
    cases.append(("ON kernel count short", run(on=on)["verdict"] == "VOID"))
    off = _fake_quality("off")
    off["engagement"]["c4val1"]["half"][0]["kernels"]["rmsnorm_rows"] = 3
    cases.append(("OFF called a kernel", run(off=off)["verdict"] == "VOID"))
    on = _fake_quality("on")
    on["engagement"]["c4val1"]["ON"][2]["qkv_calls"] -= 48
    cases.append(("qkv calls short", run(on=on)["verdict"] == "VOID"))
    off = _fake_quality("off")
    off["engagement"]["wikitext"]["R"][0]["graph_stats"]["16"]["pad_rows"] = 0
    cases.append(("R not padded", run(off=off)["verdict"] == "VOID"))
    off = _fake_quality("off")
    off["per_window"]["c4val1"]["R"].pop()
    cases.append(("a text short", run(off=off)["verdict"] == "VOID"))
    on = _fake_quality("on")
    on["windows_sha256"] = {"wikitext": "x", "c4val1": "c4val1"}
    cases.append(("phases scored different windows", run(on=on)["verdict"] == "VOID"))
    cases.append(("mutant passes", run(off=_fake_quality("off", mutant=0.004))["verdict"] == "VOID"))
    cases.append(("noisy", run(a=arms(F0b=_fake_arm("F0b", 700.0, 99.5, base)))["verdict"] == "NOISY"))
    moved = [list(x) for x in fused_toks]
    moved[9][7] += 1
    cases.append(("F1 not deterministic", run(a=arms(F1b=_fake_arm("F1b", 802.0, 140.4, moved)))["verdict"] == "FUNCTION_FAIL"))
    cases.append(("reps differ", run(a=arms(F0a=_fake_arm("F0a", 740.0, 99.4, base,
                                                           digests={"4": ["x"] * 3, "8": ["y", "z", "y"]})))["verdict"] == "FUNCTION_FAIL"))
    cases.append(("bias on c4val1", run(on=_fake_quality("on", bias={"c4val1": 0.02}))["verdict"] == "QUALITY_FAIL"))
    cases.append(("K8 on wikitext", run(on=_fake_quality("on", bias={"wikitext": 0.006}))["verdict"] == "QUALITY_FAIL"))
    r = run(on=_fake_quality("on", bias={"c4val1": 0.0055}))         # c4val1's K8 moves > 0.05 inside the floor bar
    cases.append(("c4val1 K8 reported, not gated", r["verdict"] == "DEFAULT_AUTO" and abs(r["quality"]["c4val1"]["k8"]["delta"]) > K8_BUDGET))
    cases.append(("slower at 1", run(a=arms(F1a=_fake_arm("F1a", 800.0, 105.0, fused_toks),
                                            F1b=_fake_arm("F1b", 802.0, 105.2, fused_toks)))["verdict"] == "SLOWER"))
    cases.append(("slower at 16", run(a=arms(F1a=_fake_arm("F1a", 735.0, 140.0, fused_toks),
                                             F1b=_fake_arm("F1b", 736.0, 140.4, fused_toks)))["verdict"] == "SLOWER"))
    cases.append(("rep joins the floor", run(off=_fake_quality("off", rep_identical=False))["quality"]["wikitext"]["floor"]["draws"]
                  == ["half", "chunk", "rep"]))
    g = {t: _fake_arm(t, 300.0 if t.startswith("F0") else 330.0, 250.0 if t.startswith("F0") else 290.0,
                      base if t.startswith("F0") else fused_toks, model=GRAN, prove=True) for t in TAGS}
    cases.append(("the Granite proof", reduce(g, _fake_quality("off", model=GRAN, prove=True),
                                              _fake_quality("on", model=GRAN, prove=True), E)["verdict"] == "DEFAULT_AUTO"))
    cases.append(("the tables", census_for("qwen3", 48, True) == {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193,
                                                                   "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48}
                  and census_for("granite", 32, False)["fuse_t1_glue_n"] == 65
                  and per_step("qwen3", 48) == {"rmsnorm_rows": 49, "rmsnorm_resid_rows": 48, "rope_norm_heads": 96,
                                                "router_epilogue": 48}))
    deep = copy.deepcopy(_fake_quality("on"))
    deep["census"]["fuse_t1_glue_n"] = 192
    cases.append(("ON census wrong", run(on=deep)["verdict"] == "VOID"))
    bad = [n for n, ok in cases if not ok]
    print(f"p115_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    arms = {}
    for t in TAGS:
        f = os.path.join(a.dir, f"arm_{t}.json")
        if os.path.exists(f):
            arms[t] = json.load(open(f))
    q = {}
    for ph in ("off", "on"):
        f = os.path.join(a.dir, f"quality_{ph}.json")
        q[ph] = json.load(open(f)) if os.path.exists(f) else None
    v = reduce(arms, q["off"], q["on"], a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1)
    print(f"P115_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("g1", "g16", "geomean", "pair_ratios", "self_pairs", "decode_tok_s", "ms_per_step", "f1_vs_f0_tokens"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    for t, s in (v.get("quality") or {}).items():
        print(f"  quality {t}: floor {json.dumps(s['floor'])} ON {json.dumps(s['stats']['ON'])} K8 {json.dumps(s['k8'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
