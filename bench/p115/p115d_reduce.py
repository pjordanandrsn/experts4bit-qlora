#!/usr/bin/env python3
"""p115d_reduce.py -- lane P115 Phase D's registered rule (bench/p115/PREREG-p115.md, Amendment 3; e4b#1313).

Records (``p115d_box.py``): ``arm_{D0a,D1a,D1b,D0b}.json`` (speed, W16 and W1) and ``sane_off.json`` / ``sane_on.json``
(Phase B's instrument at one window per pass, T == 1). D0 = the four fusion knobs at ``0``; D1 = all four ``auto``. The
decode GEMV is grouped-nf4-gemm 0.43.0's default in both: the bandwidth route at Qwen3-30B-A3B's shapes (P116).

The rule, first rung that applies:
  VOID           a record is missing or not ok; another e4b / grouped-nf4-gemm commit or model revision; a speed arm with
                 a bucket not captured; the arms read different prompts or lengths; a void slope; the fusion census off
                 its registration (D0 all zero; D1 the family's registered census) or the modes not the arm's; the
                 bandwidth GEMV not where the registration puts it (reading: every W1 speed workload and both SANE
                 phases dispatch ``bw_prmt32`` and never dot-pad; the proof's Granite, whose shapes the route does not
                 cover, dispatches its scalar incumbent and no ``bw_*``); a SANE phase not at one window per pass or
                 short of its windows.
  FUNCTION_FAIL  a self-pair (D0b/D0a, D1b/D1a) decodes different tokens on any row, workload or length, or one arm's timed
                 reps digest differently (D1 != D0 is expected and reported).
  COMBINED_FAIL  Phase C's SANE gate fails on the combination: |mean NLL difference ON - R| > 0.02 nats or argmax
                 agreement < 0.95 over the 12 windows.
  COMBINED_SANE  otherwise.

Reported, not ruled: g1 = min(D1a/D0a, D1b/D0b) at W1 and g16 at W16 (the stack's gain on top of the bandwidth GEMV),
the self-pairs, the SANE KL, and the token identity of D1 against D0.

    python p115d_reduce.py --dir RUN_DIR --out verdict_d.json [--e4b-sha SHA] [--proof]
    python p115d_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

TAGS = ("D0a", "D1a", "D1b", "D0b")
WORKLOADS = ("W16", "W1")
BUCKETS = (1, 2, 4, 8, 16)
GNF4_SHA = "6ee2e10408161a9d3c874975c9191a7f2957e6f4"     # grouped-nf4-gemm v0.43.0: P116's GEMV is its default
QWEN, GRAN = "Qwen/Qwen3-30B-A3B", "ibm-granite/granite-3.1-3b-a800m-instruct"
REVS = {QWEN: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", GRAN: "a02780686e08a03fe0d2679a293b5c74a90efa89"}
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
PREDICTED = {QWEN: {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48},
             GRAN: {"fuse_qkv_n": 0, "fuse_t1_glue_n": 65, "fuse_t1_glue_r2_n": [32, 32], "fuse_router_epilogue_n": 32}}
ZERO = {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0], "fuse_router_epilogue_n": 0}
MODES = {"D0": "0", "D1": "auto"}
SANE_WINDOWS, SANE_BIAS, SANE_ARGMAX = 12, 0.02, 0.95       # Phase C's gate (Amendment 2), unchanged
BW = ("bw_tree", "bw_prmt32", "bw_splitk")


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def gemv_faults(name, model, d) -> list:
    """Where P116's route must and must not run: Qwen3 -> ``bw_prmt32`` and no dot-pad; Granite -> scalar, no ``bw_*``."""
    d = d or {}
    if model == QWEN:
        if d.get("bw_prmt32", 0) <= 0 or d.get("dotpad", 0) or d.get("dotpad_splitk", 0) or d.get("bw_tree", 0):
            return [f"{name}: dispatched {d}; registered bw_prmt32 and no dot-pad (grouped-nf4-gemm 0.43.0's default)"]
    elif any(d.get(k, 0) for k in BW) or d.get("scalar", 0) + d.get("scalar_splitk", 0) <= 0:
        return [f"{name}: dispatched {d}; registered the scalar GEMV and no bw_* (Granite's shapes are not in _BW_SHAPES)"]
    return []


def record_faults(name, r, e4b_sha, want_model, arm) -> list:
    out = []
    if r.get("e4b_sha") != e4b_sha:
        out.append(f"{name}: e4b {r.get('e4b_sha')} != {e4b_sha}")
    if r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{name}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
    if r.get("model") != want_model or REVS.get(want_model) != r.get("revision"):
        out.append(f"{name}: model {r.get('model')}@{r.get('revision')} is not {want_model}")
        return out
    if r.get("arm") != arm:
        out.append(f"{name}: the record says arm {r.get('arm')}")
    cen = {k: (r.get("fusions") or {}).get(k) for k in CENSUS_KEYS}
    want = ZERO if arm == "D0" else PREDICTED[want_model]
    if cen != want:
        out.append(f"{name}: census {cen}, registered {want}")
    modes = r.get("fusion_modes") or {}
    if any(str(modes.get(k)) != MODES[arm] for k in ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2",
                                                     "E4B_FUSE_ROUTER_EPI")):
        out.append(f"{name}: fusion modes {modes}, registered all {MODES[arm]}")
    return out


def reduce(arms: dict, off: dict | None, on: dict | None, e4b_sha: str, *, proof=False) -> dict:
    model = GRAN if proof else QWEN
    out = {"lane": "P115 Phase D", "proof": proof, "verdict": None, "reasons": [],
           "bars": {"sane_bias": SANE_BIAS, "sane_argmax": SANE_ARGMAX, "sane_windows": SANE_WINDOWS}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    missing += [n for n, r in (("sane_off", off), ("sane_on", on)) if not r or r.get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = []
    for t, r in arms.items():
        void += record_faults(t, r, e4b_sha, model, t[:2])
        g = r.get("graph_status") or {}
        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        void += gemv_faults(f"{t} W1", model, r["workloads"]["W1"].get("dispatch"))
        if any(_rate(r, w) is None for w in WORKLOADS):
            void.append(f"{t}: a decode slope is void")
    if len({r.get("prompts_sha256") for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r.get("short"), r.get("long"), r.get("reps")) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    for name, rec, arm, key in (("sane_off", off, "D0", "R"), ("sane_on", on, "D1", "ON")):
        void += record_faults(name, rec, e4b_sha, model, arm)
        void += gemv_faults(name, model, rec.get("dispatch_measure"))
        if rec.get("group") != 1:
            void.append(f"{name}: group {rec.get('group')}, registered 1 (T == 1, where the GEMV and the folds meet)")
        n = len(((rec.get("per_window") or {}).get("wikitext") or {}).get(key, []))
        if n != SANE_WINDOWS:
            void.append(f"{name}: {n} windows, registered {SANE_WINDOWS}")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    # the speed half, reported
    D0a, D1a, D1b, D0b = (arms[t] for t in TAGS)
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    pairs = {w: [rates["D1a"][w] / rates["D0a"][w], rates["D1b"][w] / rates["D0b"][w]] for w in WORKLOADS}
    out["speed"] = {"decode_tok_s": rates,
                    "g1": round(min(pairs["W1"]), 4), "g16": round(min(pairs["W16"]), 4),
                    "geomean": {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()},
                    "self_pairs": {f"{a}/{b} {w}": round(rates[a][w] / rates[b][w], 4)
                                   for a, b in (("D0b", "D0a"), ("D1b", "D1a")) for w in WORKLOADS},
                    "d1_vs_d0_rows_identical": {f"{w}@{n}": sum(x == y for x, y in zip(_tok(D1a, w, n), _tok(D0a, w, n)))
                                                for w in WORKLOADS for n in (D0a["short"], D0a["long"])}}
    fn = []
    for a, b in (("D0b", "D0a"), ("D1b", "D1a")):
        for w in WORKLOADS:
            for n in (D0a["short"], D0a["long"]):
                if _tok(arms[a], w, n) != _tok(arms[b], w, n):
                    fn.append(f"{a} != {b} on {w} at {n} tokens")
    for t in TAGS:
        for w in WORKLOADS:
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    ref = {x["window"]: x["nll"] for x in off["per_window"]["wikitext"]["R"]}
    onw = on["per_window"]["wikitext"]["ON"]
    d = [x["nll"] - ref[x["window"]] for x in onw]
    sane = {"bias": sum(d) / len(d), "argmax_agree": sum(x["argmax_agree"] for x in onw) / len(onw),
            "mean_kl": sum(x.get("kl", 0.0) for x in onw) / len(onw), "n": len(onw)}
    out["sane"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sane.items()}
    if fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif abs(sane["bias"]) > SANE_BIAS or sane["argmax_agree"] < SANE_ARGMAX:
        out.update(verdict="COMBINED_FAIL", reasons=[f"SANE {out['sane']} (gate |bias| <= {SANE_BIAS}, argmax >= {SANE_ARGMAX})"])
    else:
        out.update(verdict="COMBINED_SANE", reasons=[
            f"SANE bias {sane['bias']:+.5f} nats, argmax {sane['argmax_agree']:.4f}, KL {sane['mean_kl']:.5f}; reported: "
            f"g1 {out['speed']['g1']}, g16 {out['speed']['g16']}"])
    return out


# ------------------------------------------------------------------ self-test --
E = "a" * 40


def _fake_arm(tag, r16, r1, toks, *, model=QWEN, e4b=E, census=None, modes=None, graphs=None, w1_dispatch=None,
              rep_ok=True):
    arm = tag[:2]
    dw1 = w1_dispatch or ({"bw_prmt32": 96, "dotpad": 0} if model == QWEN else {"scalar": 64, "bw_prmt32": 0})
    w = {}
    for name, rate in (("W16", r16), ("W1", r1)):
        b = 16 if name == "W16" else 1
        w[name] = {"batch": b, "decode_tok_s": rate, "tokens": {"32": toks[:b], "160": toks[:b]},
                   "dispatch": dw1 if name == "W1" else {},
                   "rep_digests": {"32": ["x", "x", "x" if rep_ok else "y"], "160": ["z"] * 3}}
    return {"arm": arm, "tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": model,
            "revision": REVS[model], "graph_status": graphs or {str(b): "graph" for b in BUCKETS},
            "fusions": census or (ZERO if arm == "D0" else PREDICTED[model]),
            "fusion_modes": modes or {k: MODES[arm] for k in ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE",
                                                              "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")},
            "prompts_sha256": "p", "short": 32, "long": 160, "reps": 3, "workloads": w}


def _fake_sane(arm, *, model=QWEN, bias=0.001, agree=0.97, n=SANE_WINDOWS, group=1, dispatch=None):
    key = "R" if arm == "D0" else "ON"
    per = [{"window": i, "nll": 2.0 + 0.01 * i + (bias if arm == "D1" else 0.0), "argmax_agree": agree, "kl": 0.01}
           for i in range(n)]
    d = dispatch or ({"bw_prmt32": 9000, "dotpad": 0} if model == QWEN else {"scalar": 9000})
    return {"mode": "sane", "arm": arm, "status": "ok", "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": model,
            "revision": REVS[model], "fusions": ZERO if arm == "D0" else PREDICTED[model],
            "fusion_modes": {k: MODES[arm] for k in ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2",
                                                     "E4B_FUSE_ROUTER_EPI")},
            "group": group, "per_window": {"wikitext": {key: per}}, "dispatch_measure": d}


def self_test() -> int:
    toks = [[i, i + 1] for i in range(16)]
    other = [[i, i + 2] for i in range(16)]

    def arms(model=QWEN, **over):
        base = {"D0a": (800.0, 120.0), "D1a": (900.0, 160.0), "D1b": (901.0, 159.0), "D0b": (799.0, 121.0)}
        out = {}
        for t, (r16, r1) in base.items():
            kw = dict(over.get(t, {}))
            out[t] = _fake_arm(t, kw.pop("r16", r16), kw.pop("r1", r1), kw.pop("toks", toks), model=model, **kw)
        return out
    off, on = _fake_sane("D0"), _fake_sane("D1")
    v = lambda a, o=off, n=on, **kw: reduce(a, o, n, E, **kw)["verdict"]  # noqa: E731
    cases = [
        ("combined sane", v(arms()) == "COMBINED_SANE"),
        ("bias over the gate", v(arms(), n=_fake_sane("D1", bias=0.03)) == "COMBINED_FAIL"),
        ("argmax under the gate", v(arms(), n=_fake_sane("D1", agree=0.93)) == "COMBINED_FAIL"),
        ("speed is reported, not ruled", v(arms(D1a={"r1": 100.0}, D1b={"r1": 100.0})) == "COMBINED_SANE"),
        ("D1 != D0 is reported", v(arms(D1a={"toks": other}, D1b={"toks": other})) == "COMBINED_SANE"),
        ("D1b != D1a", v(arms(D1b={"toks": other})) == "FUNCTION_FAIL"),
        ("reps differ", v(arms(D0a={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("the bandwidth GEMV did not run at W1", v(arms(D1a={"w1_dispatch": {"dotpad": 96}})) == "VOID"),
        ("dot-pad beside it", v(arms(D0b={"w1_dispatch": {"bw_prmt32": 90, "dotpad": 6}})) == "VOID"),
        ("SANE without the GEMV", v(arms(), n=_fake_sane("D1", dispatch={"dotpad": 9000})) == "VOID"),
        ("SANE at T > 1", v(arms(), o=_fake_sane("D0", group=12)) == "VOID"),
        ("SANE short", v(arms(), n=_fake_sane("D1", n=11)) == "VOID"),
        ("D1 census short", v(arms(D1a={"census": {**PREDICTED[QWEN], "fuse_qkv_n": 0}})) == "VOID"),
        ("D0 not unfused", v(arms(D0a={"census": PREDICTED[QWEN]})) == "VOID"),
        ("modes not the arm's", v(arms(D1b={"modes": {"E4B_PAGED_FUSE_QKV": "1"}})) == "VOID"),
        ("a bucket not captured", v(arms(D1a={"graphs": {"1": "eager"}})) == "VOID"),
        ("wrong commit", v(arms(D0a={"e4b": "b" * 40})) == "VOID"),
        ("missing arm", reduce({k: x for k, x in arms().items() if k != "D1b"}, off, on, E)["verdict"] == "VOID"),
        ("missing SANE", reduce(arms(), off, None, E)["verdict"] == "VOID"),
        ("g1 is reported", reduce(arms(), off, on, E)["speed"]["g1"] == round(min(160 / 120, 159 / 121), 4)),
        ("proof on Granite (scalar, no bw)", reduce(arms(model=GRAN), _fake_sane("D0", model=GRAN),
                                                    _fake_sane("D1", model=GRAN), E, proof=True)["verdict"] == "COMBINED_SANE"),
        ("proof with bw on Granite is VOID", reduce(arms(model=GRAN, D1a={"w1_dispatch": {"bw_prmt32": 64}}),
                                                     _fake_sane("D0", model=GRAN), _fake_sane("D1", model=GRAN), E,
                                                     proof=True)["verdict"] == "VOID"),
        ("the reading's records on a proof", reduce(arms(), off, on, E, proof=True)["verdict"] == "VOID"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p115d_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    v = reduce(arms, load("sane_off.json"), load("sane_on.json"), a.e4b_sha, proof=a.proof)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P115D_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
