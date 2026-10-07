#!/usr/bin/env python3
"""p116_reduce.py -- lane P116's registered rule (bench/p116/PREREG-p116.md; e4b#1313).

Records (``p116_box.py``): ``arm_{B0a,B1a,B1b,B0b}.json`` (speed, W16 and W1) and ``quality_off.json`` /
``quality_on.json`` (P110's teacher-forced instrument at one window per pass, T == 1). B0 = the shipped default
(``GNF4_GEMV_BW`` unset); B1 = ``GNF4_GEMV_BW=1`` at K33's selected plans.

The rule, first rung that applies:
  VOID           a record is missing or not ok; another e4b / grouped-nf4-gemm commit or model revision; a bucket not
                 captured; the arms read different prompts or lengths or fusion censuses; a void slope; ENGAGEMENT off:
                 B1 must dispatch every single-row decode GEMV to ``bw_prmt32`` (no dot-pad, no scalar, no tree, no
                 split-K) and B0 none to ``bw_*`` and some to the family's incumbent (dot-pad on Qwen3, the scalar GEMV
                 on Granite); the quality records fail Phase B's integrity checks at one window per pass (window counts,
                 decode attention calls, bucket-1 eager steps, no replays, glue-kernel and q/k/v calls equal between R
                 and ON, the same dispatch rule); or the scale mutant passes the quality bar (the gate cannot fail).
  NOISY          a self-pair (B0b/B0a, B1b/B1a) outside [0.96, 1.04] at W16, or outside [0.985, 1.015] at W1 (half the
                 W1 gain bar, so instrument drift cannot pass as the 1.03 gain; added in review).
  FUNCTION_FAIL  B0b != B0a or B1b != B1a on any row, workload or length; timed reps of one arm digest differently.
                 (B1 != B0 is expected -- another reduction order -- and reported.)
  QUALITY_FAIL   on either text bias_ON > B_floor + 0.01 or spread_ON > 2 * max(S_floor, 0.005), the floor drawn from
                 ``chunk`` (and ``rep`` when R did not repeat bit for bit); on wikitext |K8 delta ppl| > 0.05.
  SLOWER         g1 = min(B1a/B0a, B1b/B0b) at W1 < 1.03, or g16 (the same at W16) < 0.99.
  DEFAULT_ON     otherwise.

    python p116_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA] [--proof]
    python p116_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p115_reduce as pb  # noqa: E402  (Phase B's statistics, staged at its registered bytes)

TAGS = ("B0a", "B1a", "B1b", "B0b")
BUCKETS = (1, 2, 4, 8, 16)
SELF_LO, SELF_HI = 0.96, 1.04
SELF_W1_LO, SELF_W1_HI = 0.985, 1.015   # W1's self-pairs: half the 1.03 gain bar (P111 read 0.997-1.002); added in review
GAIN_MIN_W1, GAIN_MIN_W16 = 1.03, 0.99
TOL, SPREAD_X, SPREAD_MIN, K8_BUDGET, K8_GATED = pb.TOL, pb.SPREAD_X, pb.SPREAD_MIN, pb.K8_BUDGET, pb.K8_GATED
TEXTS = pb.TEXTS
GNF4_SHA = "5a60c37dbd0756040052b603c9b0ee680f06d444"     # grouped-nf4-gemm with K33 (#500 + #501): K33's measured cut
QWEN, GRAN = "Qwen/Qwen3-30B-A3B", "ibm-granite/granite-3.1-3b-a800m-instruct"
REVS = {QWEN: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", GRAN: "a02780686e08a03fe0d2679a293b5c74a90efa89"}
LAYERS = {QWEN: 48, GRAN: 32}
WINDOWS = {QWEN: 48, GRAN: 12}
INCUMBENT = {QWEN: "dotpad", GRAN: "scalar"}             # today's single-row route at each family's shapes (>= 160 SMs)
OFF_ARMS = ("R", "rep", "chunk", "mutant_scale")
FLOORS = ("chunk",)
GEMV = ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk")
BW = ("bw_tree", "bw_prmt32", "bw_splitk")


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def engagement_faults(name, arm, model, d) -> list:
    """The dispatch rule on a tally delta ``d``: B1 all ``bw_prmt32``; B0 none to ``bw_*``, some to the incumbent."""
    d = d or {}
    inc = INCUMBENT.get(model, "dotpad")
    other = [k for k in GEMV if not k.startswith(inc)]
    if arm == "B1":
        if d.get("bw_prmt32", 0) <= 0 or any(d.get(k, 0) for k in ("bw_tree", "bw_splitk") + GEMV):
            return [f"{name}: B1 dispatched {d}, registered bw_prmt32 only"]
    elif any(d.get(k, 0) for k in BW) or d.get(inc, 0) + d.get(inc + "_splitk", 0) <= 0 \
            or any(d.get(k, 0) for k in other):
        return [f"{name}: B0 dispatched {d}, registered {inc} only"]
    return []


def speed_faults(arms: dict, e4b_sha: str, proof: bool) -> list:
    void = []
    want_model = GRAN if proof else QWEN
    for t, r in arms.items():
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"{t}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{t}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if r.get("model") != want_model or REVS.get(r.get("model")) != r.get("revision"):
            void.append(f"{t}: model {r.get('model')}@{r.get('revision')} is not {want_model}@{REVS[want_model][:8]}")
            continue
        if r.get("arm") != t[:2]:
            void.append(f"{t}: the record says arm {r.get('arm')}")
        g = r.get("graph_status") or {}
        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        void += engagement_faults(t, t[:2], r["model"], r.get("dispatch_total"))
    if len({json.dumps(r.get("fusions"), sort_keys=True) for r in arms.values()}) != 1:
        void.append("the arms report different fusion censuses (only the decode GEMV switch may differ)")
    if len({r.get("prompts_sha256") for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r.get("short"), r.get("long"), r.get("reps")) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    if any(_rate(arms[t], w) is None for t in TAGS for w in ("W16", "W1") if t in arms):
        void.append("a decode slope is void")
    return void


def _t1_pass_faults(e, C, L) -> list:
    """One window, one pass: bucket 1 eager (capture=False), C - 1 decode steps, no padding, no replay."""
    out = []
    if e.get("decode_calls") != (C - 1) * L:
        out.append(f"{e.get('decode_calls')} decode attention calls, expected {(C - 1) * L}")
    if not (e.get("grouping_flags_in_pass") or {}).get("device_grouping"):
        out.append("device grouping off")
    st = e.get("graph_status") or {}
    if [st.get(str(b)) for b in BUCKETS] != ["eager: capture=False"] * len(BUCKETS):
        out.append(f"graph_status {st}")
    gs = e.get("graph_stats") or {}
    got = gs.get("1", {})
    if (got.get("eager_steps"), got.get("pad_rows")) != (C - 1, 0):
        out.append(f"bucket 1 stats {got}, expected {C - 1} eager steps and no padding")
    if any(v.get("replays", 0) for v in gs.values()) or any(v.get("eager_steps", 0) for b, v in gs.items() if b != "1"):
        out.append(f"another bucket or a replay in a one-window pass: {gs}")
    return out


def quality_faults(off: dict, on: dict, e4b_sha: str, proof: bool) -> list:
    void = []
    want_model = GRAN if proof else QWEN
    for name, rec in (("quality_off", off), ("quality_on", on)):
        if rec.get("status") != "ok":
            void.append(f"{name}: status {rec.get('status')}")
        if rec.get("e4b_sha") != e4b_sha:
            void.append(f"{name}: e4b {rec.get('e4b_sha')} != {e4b_sha}")
        if rec.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{name}: grouped-nf4-gemm {rec.get('gnf4_sha')} != {GNF4_SHA}")
        if rec.get("model") != want_model or REVS.get(rec.get("model")) != rec.get("revision"):
            void.append(f"{name}: model {rec.get('model')}@{rec.get('revision')} is not {want_model}")
        if rec.get("group") != 1:
            void.append(f"{name}: group {rec.get('group')}, registered 1 (T == 1)")
    if void:
        return void
    L, want_n, C = LAYERS[want_model], WINDOWS[want_model], off["cont"]
    for k in ("cont", "prompt", "chunk", "floor_chunk", "layers", "windows_sha256", "fusions"):
        if off.get(k) != on.get(k):
            void.append(f"the phases differ in {k}: {off.get(k)} / {on.get(k)}")
    if off.get("layers") != L:
        void.append(f"{off.get('layers')} layers, the family registers {L}")
    void += engagement_faults("quality_off", "B0", want_model, off.get("dispatch_measure"))
    void += engagement_faults("quality_on", "B1", want_model, on.get("dispatch_measure"))
    for t in TEXTS:
        for name, rec, arms in (("quality_off", off, OFF_ARMS), ("quality_on", on, ("ON",))):
            per = (rec.get("per_window") or {}).get(t, {})
            eng = (rec.get("engagement") or {}).get(t, {})
            for arm in arms:
                want = 1 if arm == "rep" else want_n
                if len(per.get(arm, [])) != want:
                    void.append(f"{name} {t}: arm {arm} has {len(per.get(arm, []))} windows, expected {want}")
                passes = eng.get(arm, [])
                if len(passes) != want:
                    void.append(f"{name} {t} {arm}: {len(passes)} passes, expected {want}")
                    continue
                for k, e in enumerate(passes):
                    void += [f"{name} {t} {arm} pass {k}: {m}" for m in _t1_pass_faults(e, C, L)]
        r_passes = (off.get("engagement") or {}).get(t, {}).get("R", [])
        on_passes = (on.get("engagement") or {}).get(t, {}).get("ON", [])
        for k, (a, b) in enumerate(zip(r_passes, on_passes)):
            if (a.get("kernels"), a.get("qkv_calls")) != (b.get("kernels"), b.get("qkv_calls")):
                void.append(f"{t} window {k}: glue kernels / q/k/v calls differ between R ({a.get('kernels')}, "
                            f"{a.get('qkv_calls')}) and ON ({b.get('kernels')}, {b.get('qkv_calls')})")
    return void


def quality_read(off: dict, on: dict) -> dict:
    out = {}
    for t in TEXTS:
        ref = {x["window"]: x["nll"] for x in off["per_window"][t]["R"]}
        st = {a: pb._stats(off["per_window"][t], a, ref) for a in OFF_ARMS if a != "R"}
        st["ON"] = pb._stats(on["per_window"][t], "ON", ref)
        draws = list(FLOORS) + ([] if (off.get("rep_identical") or {}).get(t) else ["rep"])
        fl = {"draws": draws, "B_floor": max(abs(st[f]["bias"]) for f in draws),
              "S_floor": max(st[f]["spread"] for f in draws)}
        ppl_r, ppl_on = pb._ppl(off["per_window"][t]["R"]), pb._ppl(on["per_window"][t]["ON"])
        out[t] = {"stats": st, "floor": fl, "bias_bar": fl["B_floor"] + TOL,
                  "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN),
                  "mutant_passes": pb._passes(st["mutant_scale"], fl), "on_passes": pb._passes(st["ON"], fl),
                  "k8": {"ppl_R": ppl_r, "ppl_ON": ppl_on, "delta": ppl_on - ppl_r, "gated": t in K8_GATED,
                         "passes": abs(ppl_on - ppl_r) <= K8_BUDGET}}
    return out


def reduce(arms: dict, off: dict | None, on: dict | None, e4b_sha: str, *, proof=False) -> dict:
    out = {"lane": "P116", "proof": proof, "verdict": None, "reasons": [],
           "bars": {"self_pair": [SELF_LO, SELF_HI], "self_pair_w1": [SELF_W1_LO, SELF_W1_HI], "gain_min_w1": GAIN_MIN_W1, "gain_min_w16": GAIN_MIN_W16,
                    "tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN, "k8_budget": K8_BUDGET,
                    "k8_gated": list(K8_GATED), "floors": list(FLOORS)}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    missing += [n for n, r in (("quality_off", off), ("quality_on", on)) if not r]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = speed_faults(arms, e4b_sha, proof) + quality_faults(off, on, e4b_sha, proof)
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    q = quality_read(off, on)
    out["quality"] = q
    void = [f"{t}: mutant_scale passes the bar (the gate cannot fail)" for t in TEXTS if q[t]["mutant_passes"]]
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    B0a, B1a, B1b, B0b = (arms[t] for t in TAGS)
    short, long_ = B0a["short"], B0a["long"]
    rates = {t: {w: _rate(arms[t], w) for w in ("W16", "W1")} for t in TAGS}
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in ("W16", "W1")} for t in TAGS}
    selfp = {f"{a}/{b} {w}": rates[a][w] / rates[b][w] for a, b in (("B0b", "B0a"), ("B1b", "B1a")) for w in ("W16", "W1")}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    pairs = {w: [rates["B1a"][w] / rates["B0a"][w], rates["B1b"][w] / rates["B0b"][w]] for w in ("W16", "W1")}
    out["pair_ratios"] = {w: [round(x, 4) for x in v] for w, v in pairs.items()}
    out["g16"], out["g1"] = round(min(pairs["W16"]), 4), round(min(pairs["W1"]), 4)
    out["geomean"] = {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()}
    out["b1_vs_b0_tokens"] = {f"{w}@{n}": pb._agree(_tok(B1a, w, n), _tok(B0a, w, n)) for w in ("W16", "W1")
                              for n in (short, long_)}
    out["dispatch"] = {t: arms[t].get("dispatch_total") for t in TAGS}
    fn = []
    for a, b in (("B0b", "B0a"), ("B1b", "B1a")):
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
    band = lambda k: (SELF_W1_LO, SELF_W1_HI) if k.endswith(" W1") else (SELF_LO, SELF_HI)  # noqa: E731
    noisy = [f"{k} = {v:.4f} (band {band(k)})" for k, v in selfp.items() if not band(k)[0] <= v <= band(k)[1]]
    qfail = []
    for t in TEXTS:
        s = q[t]["stats"]["ON"]
        if not q[t]["on_passes"]:
            qfail.append(f"{t}: ON bias {s['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), spread {s['spread']:.5f} "
                         f"(bar {q[t]['spread_bar']:.5f})")
        if q[t]["k8"]["gated"] and not q[t]["k8"]["passes"]:
            qfail.append(f"{t}: K8 perplexity {q[t]['k8']['ppl_R']:.5f} -> {q[t]['k8']['ppl_ON']:.5f} (|delta| > {K8_BUDGET})")
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif qfail:
        out.update(verdict="QUALITY_FAIL", reasons=qfail)
    elif out["g1"] < GAIN_MIN_W1 or out["g16"] < GAIN_MIN_W16:
        out.update(verdict="SLOWER", reasons=[f"g1 = {out['g1']} (bar {GAIN_MIN_W1}), g16 = {out['g16']} (bar {GAIN_MIN_W16})"])
    else:
        out.update(verdict="DEFAULT_ON", reasons=[
            f"g1 = {out['g1']}, g16 = {out['g16']} (geomean {out['geomean']['W1']} / {out['geomean']['W16']}); "
            + "; ".join(f"{t} ON bias {q[t]['stats']['ON']['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), K8 "
                        f"{q[t]['k8']['delta']:+.4f}" for t in TEXTS)])
    return out


# ------------------------------------------------------------------ self-test --
E = "a" * 40


def _fake_arm(tag, rate16, rate1, toks, *, model=QWEN, e4b=E, graphs=None, dispatch=None, rep_ok=True):
    arm = tag[:2]
    inc = INCUMBENT[model]
    L = LAYERS[model]
    d = dispatch or ({**{k: 0 for k in GEMV + BW}, "bw_prmt32": 2 * L} if arm == "B1"
                     else {**{k: 0 for k in GEMV + BW}, inc: 2 * L})
    w = {}
    for name, rate in (("W16", rate16), ("W1", rate1)):
        b = 16 if name == "W16" else 1
        w[name] = {"batch": b, "decode_tok_s": rate, "decode_ms_per_step": 1000.0 / rate * b,
                   "tokens": {"32": toks[:b], "160": toks[:b]},
                   "rep_digests": {"32": ["x", "x", "x" if rep_ok else "y"], "160": ["z"] * 3}}
    return {"arm": arm, "tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": model,
            "revision": REVS[model], "graph_status": graphs or {str(b): "graph" for b in BUCKETS},
            "fusions": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0], "fuse_router_epilogue_n": 0},
            "prompts_sha256": "p", "short": 32, "long": 160, "reps": 3, "workloads": w, "dispatch_total": d}


def _fake_pass(C, L):
    gs = {str(b): {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in BUCKETS}
    gs["1"]["eager_steps"] = C - 1
    return {"decode_calls": (C - 1) * L, "grouping_flags_in_pass": {"device_grouping": True},
            "graph_status": {str(b): "eager: capture=False" for b in BUCKETS}, "graph_stats": gs,
            "kernels": {k: 0 for k in pb.KERNELS}, "forwards": C, "qkv_calls": 0}


def _fake_quality(phase, *, model=QWEN, bias=None, mutant=0.4, chunk=-0.0004, rep_identical=True, dispatch=None,
                  n=None, group=1):
    L, C = LAYERS[model], (128 if model == QWEN else 32)
    n = n if n is not None else WINDOWS[model]
    bias = bias or {}
    arms = OFF_ARMS if phase == "off" else ("ON",)
    inc = INCUMBENT[model]
    d = dispatch or ({**{k: 0 for k in GEMV + BW}, inc: 2 * L * n} if phase == "off"
                     else {**{k: 0 for k in GEMV + BW}, "bw_prmt32": 2 * L * n})
    per, eng = {}, {}
    for t in TEXTS:
        base = {w: 2.0 + 0.01 * w + (0.5 if t == "c4val1" else 0.0) for w in range(n)}
        per[t], eng[t] = {}, {}
        for arm in arms:
            off_by = {"R": 0.0, "rep": 0.0, "chunk": chunk, "mutant_scale": mutant}.get(arm, bias.get(t, 0.0))
            ws = range(1) if arm == "rep" else range(n)
            per[t][arm] = [{"window": w, "nll": base[w] + off_by + (0.001 if w % 2 else -0.001) * (arm != "R"),
                            "argmax_agree": 1.0, "kl": 0.001} for w in ws]
            eng[t][arm] = [_fake_pass(C, L) for _ in ws]
    return {"mode": "quality", "arm": "B0" if phase == "off" else "B1", "status": "ok", "model": model,
            "revision": REVS[model], "e4b_sha": E, "gnf4_sha": GNF4_SHA, "group": group, "prompt": 512, "cont": C,
            "chunk": 512, "floor_chunk": 256, "layers": L, "windows_sha256": {t: t for t in TEXTS},
            "fusions": {"fuse_qkv_n": 0}, "rep_identical": {t: rep_identical for t in TEXTS}, "per_window": per,
            "engagement": eng, "dispatch_measure": d}


def self_test() -> int:
    toks = [[i, i + 1] for i in range(16)]
    other = [[i, i + 2] for i in range(16)]

    def arms(**over):
        base = {"B0a": (800.0, 110.0), "B1a": (801.0, 125.0), "B1b": (800.5, 124.8), "B0b": (799.0, 110.2)}
        out = {}
        for t, (r16, r1) in base.items():
            kw = over.get(t, {})
            out[t] = _fake_arm(t, kw.pop("r16", r16), kw.pop("r1", r1), kw.pop("toks", toks), **kw)
        return out
    off, on = _fake_quality("off"), _fake_quality("on", bias={"wikitext": 0.001, "c4val1": 0.001})
    v = lambda a, o=off, n=on, **kw: reduce(a, o, n, E, **kw)["verdict"]  # noqa: E731
    cases = [
        ("default on", v(arms()) == "DEFAULT_ON"),
        ("slower at W1", v(arms(B1a={"r1": 111.0}, B1b={"r1": 111.0})) == "SLOWER"),
        ("W16 regression", v(arms(B1a={"r16": 780.0}, B1b={"r16": 780.0})) == "SLOWER"),
        ("noisy", v(arms(B0b={"r1": 100.0})) == "NOISY"),
        ("W1 drift inside the old band is NOISY", v(arms(B0b={"r1": 112.5})) == "NOISY"),
        ("W16 drift inside its band is not", v(arms(B0b={"r16": 820.0}, B1b={"r16": 821.0})) == "DEFAULT_ON"),
        ("function fail", v(arms(B1b={"toks": other})) == "FUNCTION_FAIL"),
        ("reps differ", v(arms(B0a={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("B1 != B0 is reported, not failed", v(arms(B1a={"toks": other}, B1b={"toks": other})) == "DEFAULT_ON"),
        ("B1 reached dot-pad", v(arms(B1a={"dispatch": {"bw_prmt32": 96, "dotpad": 2}})) == "VOID"),
        ("B1 took the tree", v(arms(B1a={"dispatch": {"bw_prmt32": 0, "bw_tree": 96}})) == "VOID"),
        ("B0 reached bw", v(arms(B0a={"dispatch": {"dotpad": 96, "bw_prmt32": 2}})) == "VOID"),
        ("B0 no incumbent", v(arms(B0b={"dispatch": {"scalar": 96}})) == "VOID"),
        ("a bucket not captured", v(arms(B1a={"graphs": {"1": "eager"}})) == "VOID"),
        ("wrong commit", v(arms(B0a={"e4b": "b" * 40})) == "VOID"),
        ("missing arm", reduce({k: x for k, x in arms().items() if k != "B1b"}, off, on, E)["verdict"] == "VOID"),
        ("quality bias", v(arms(), n=_fake_quality("on", bias={"c4val1": 0.02})) == "QUALITY_FAIL"),
        ("quality K8 wikitext", v(arms(), n=_fake_quality("on", bias={"wikitext": 0.009})) == "QUALITY_FAIL"),
        ("c4val1 K8 reported only", v(arms(), n=_fake_quality("on", bias={"c4val1": 0.009})) == "DEFAULT_ON"),
        ("mutant passes", v(arms(), o=_fake_quality("off", mutant=0.0001)) == "VOID"),
        ("quality group 12", v(arms(), o=_fake_quality("off", group=12)) == "VOID"),
        ("quality short", v(arms(), n=_fake_quality("on", n=47)) == "VOID"),
        ("quality on reached dot-pad", v(arms(), n=_fake_quality("on", dispatch={"bw_prmt32": 9, "dotpad": 1})) == "VOID"),
        ("rep not identical joins the floor", reduce(arms(), _fake_quality("off", rep_identical=False), on, E)
         ["quality"]["wikitext"]["floor"]["draws"] == ["chunk", "rep"]),
        ("proof", reduce({t: _fake_arm(t, 300.0, 50.0 if t[:2] == "B0" else 60.0, toks, model=GRAN) for t in TAGS},
                         _fake_quality("off", model=GRAN), _fake_quality("on", model=GRAN), E, proof=True)["verdict"]
         == "DEFAULT_ON"),
        ("proof on qwen is void", reduce(arms(), off, on, E, proof=True)["verdict"] == "VOID"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p116_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--proof", action="store_true")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()

    def load(name):
        f = os.path.join(a.dir, name)
        return json.load(open(f)) if os.path.exists(f) else None
    arms = {t: r for t in TAGS if (r := load(f"arm_{t}.json"))}
    v = reduce(arms, load("quality_off.json"), load("quality_on.json"), a.e4b_sha, proof=a.proof)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P116_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
