#!/usr/bin/env python3
"""p118_reduce.py -- lane P118's registered rule (bench/p118/PREREG-p118.md; e4b#1313).

Records (``p118_box.py``): ``arm_{L0a,L1a,L1b,L0b}.json`` (W16 and W1, plus a traced pass per workload). L0 = the
shipped default (``E4B_PAGED_DECODE_LOOKAHEAD`` unset); L1 = ``E4B_PAGED_DECODE_LOOKAHEAD=1``.

The rule, first rung that applies:
  VOID           a record is missing or not ok; another e4b / grouped-nf4-gemm commit or model revision; a bucket not
                 captured; the arms read different prompts, lengths or fusion censuses; a void slope; ENGAGEMENT off: L1
                 must run the lookahead scheduler, collect every step it issued, have a newer step queued behind at least
                 OVERLAP_MIN of its collects at each workload, and discard nothing (no stop ids); L0 must run the
                 synchronous scheduler and never call the lookahead entry points.
  NOISY          a self-pair (L0b/L0a, L1b/L1a) outside [0.96, 1.04] at W16, or outside [0.99, 1.01] at W1 (half the W1
                 gain bar, so instrument drift cannot pass as the gain).
  FUNCTION_FAIL  any two arms decode different tokens on any row, workload or length (the lookahead feeds each step the
                 synchronous step's inputs, so L1 == L0 is the contract, not a hope); timed reps of one arm digest
                 differently; the arms' bucket statistics differ (the same rows in the same buckets, step for step).
  SLOWER         g1 = min(L1a/L0a, L1b/L0b) at W1 < 1.02, or g16 (the same at W16) < 0.99.
  DEFAULT_ON     otherwise.

Reported, not ruled: each arm's traced decode steps (the step period, and in L0 the device time and the host gap), and the
L0 host gap against the step time the lookahead saved.

    python p118_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA] [--proof]
    python p118_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

TAGS = ("L0a", "L1a", "L1b", "L0b")
WORKLOADS = ("W16", "W1")
BUCKETS = (1, 2, 4, 8, 16)
SELF_LO, SELF_HI = 0.96, 1.04
SELF_W1_LO, SELF_W1_HI = 0.99, 1.01     # W1's self-pairs: half the 1.02 gain bar (P111's read 0.997-1.002)
GAIN_MIN_W1, GAIN_MIN_W16 = 1.02, 0.99
OVERLAP_MIN = 0.9                        # L1: share of its collects at each workload with a newer step queued behind
GNF4_SHA = "b4f93f1c62d1e3436ed45bec8ccd608c90433737"     # grouped-nf4-gemm 0.42.0: e4b CI's pin at registration
QWEN, GRAN = "Qwen/Qwen3-30B-A3B", "ibm-granite/granite-3.1-3b-a800m-instruct"
REVS = {QWEN: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", GRAN: "a02780686e08a03fe0d2679a293b5c74a90efa89"}


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def engagement_faults(tag, r) -> list:
    arm = tag[:2]
    if arm == "L1":
        out = [] if r.get("lookahead") is True else [f"{tag}: the scheduler ran without the lookahead"]
        for w in WORKLOADS:
            c = (r["workloads"][w].get("lookahead_calls") or {})
            i, k, o = c.get("issues", 0), c.get("collects", 0), c.get("overlapped", 0)
            if i <= 0 or k != i or o < OVERLAP_MIN * k:
                out.append(f"{tag} {w}: {i} issues, {k} collects, {o} overlapped (registered: every issue collected, "
                           f">= {OVERLAP_MIN:.0%} overlapped)")
        if r.get("lookahead_discarded", 0):
            out.append(f"{tag}: {r['lookahead_discarded']} tokens discarded with no stop ids")
        return out
    c = r.get("lookahead_calls") or {}
    if r.get("lookahead") is not False or any(c.get(k, 0) for k in ("issues", "collects", "overlapped")):
        return [f"{tag}: L0 lookahead={r.get('lookahead')} calls={c}"]
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
        void += engagement_faults(t, r)
    if len({json.dumps(r.get("fusions"), sort_keys=True) for r in arms.values()}) != 1:
        void.append("the arms report different fusion censuses (only the lookahead switch may differ)")
    if len({r.get("prompts_sha256") for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r.get("short"), r.get("long"), r.get("reps")) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    if any(_rate(arms[t], w) is None for t in TAGS for w in WORKLOADS if t in arms):
        void.append("a decode slope is void")
    return void


def mechanism(arms: dict) -> dict:
    """The traced passes, reported: per arm and workload the decode step period (and in L0 its device time and host gap),
    and per workload the L0 host gap beside the period the lookahead saved (L0 - L1, means of the a/b arms)."""
    tr = {t: arms[t].get("trace") or {} for t in TAGS}
    out = {"per_arm": tr, "saved_vs_gap": {}}
    for w in WORKLOADS:
        l0 = [tr[t].get(w, {}) for t in ("L0a", "L0b")]
        l1 = [tr[t].get(w, {}) for t in ("L1a", "L1b")]
        try:
            gap = sum(x["host_gap_ms_p50"] for x in l0) / 2
            saved = sum(x["step_ms_p50"] for x in l0) / 2 - sum(x["step_ms_p50"] for x in l1) / 2
            dev = sum(x["device_ms_p50"] for x in l0) / 2
        except (KeyError, TypeError):
            out["saved_vs_gap"][w] = "trace incomplete"
            continue
        out["saved_vs_gap"][w] = {"l0_host_gap_ms": round(gap, 4), "l0_device_ms": round(dev, 4),
                                  "period_saved_ms": round(saved, 4),
                                  "saved_over_gap": round(saved / gap, 3) if gap > 0 else None}
    return out


def reduce(arms: dict, e4b_sha: str, *, proof=False) -> dict:
    out = {"lane": "P118", "proof": proof, "verdict": None, "reasons": [],
           "bars": {"self_pair": [SELF_LO, SELF_HI], "self_pair_w1": [SELF_W1_LO, SELF_W1_HI],
                    "gain_min_w1": GAIN_MIN_W1, "gain_min_w16": GAIN_MIN_W16, "overlap_min": OVERLAP_MIN}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = speed_faults(arms, e4b_sha, proof)
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    L0a, L1a, L1b, L0b = (arms[t] for t in TAGS)
    short, long_ = L0a["short"], L0a["long"]
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in WORKLOADS} for t in TAGS}
    selfp = {f"{a}/{b} {w}": rates[a][w] / rates[b][w] for a, b in (("L0b", "L0a"), ("L1b", "L1a")) for w in WORKLOADS}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    pairs = {w: [rates["L1a"][w] / rates["L0a"][w], rates["L1b"][w] / rates["L0b"][w]] for w in WORKLOADS}
    out["pair_ratios"] = {w: [round(x, 4) for x in v] for w, v in pairs.items()}
    out["g16"], out["g1"] = round(min(pairs["W16"]), 4), round(min(pairs["W1"]), 4)
    out["geomean"] = {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()}
    out["lookahead_calls"] = {t: {w: arms[t]["workloads"][w].get("lookahead_calls") for w in WORKLOADS} for t in TAGS}
    out["mechanism"] = mechanism(arms)
    fn = []
    for a, b in (("L0b", "L0a"), ("L1b", "L1a"), ("L1a", "L0a"), ("L1b", "L0b")):
        for w in WORKLOADS:
            for n in (short, long_):
                if _tok(arms[a], w, n) != _tok(arms[b], w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[a], w, n), _tok(arms[b], w, n))) if x != y]
                    fn.append(f"{a} != {b} on {w} at {n} tokens, rows {rows}")
    for t in TAGS:
        for w in WORKLOADS:
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    if len({json.dumps(arms[t].get("graph_stats"), sort_keys=True) for t in TAGS}) != 1:
        fn.append("the arms' bucket statistics differ: " + "; ".join(f"{t} {arms[t].get('graph_stats')}" for t in TAGS))
    band = lambda k: (SELF_W1_LO, SELF_W1_HI) if k.endswith(" W1") else (SELF_LO, SELF_HI)  # noqa: E731
    noisy = [f"{k} = {v:.4f} (band {band(k)})" for k, v in selfp.items() if not band(k)[0] <= v <= band(k)[1]]
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif out["g1"] < GAIN_MIN_W1 or out["g16"] < GAIN_MIN_W16:
        out.update(verdict="SLOWER", reasons=[f"g1 = {out['g1']} (bar {GAIN_MIN_W1}), g16 = {out['g16']} (bar {GAIN_MIN_W16})"])
    else:
        out.update(verdict="DEFAULT_ON", reasons=[
            f"g1 = {out['g1']}, g16 = {out['g16']} (geomean {out['geomean']['W1']} / {out['geomean']['W16']}); "
            "tokens identical in every arm, workload and length"])
    return out


# ------------------------------------------------------------------ self-test --
E = "a" * 40


def _fake_arm(tag, rate16, rate1, toks, *, model=QWEN, e4b=E, graphs=None, rep_ok=True, calls=None, look=None,
              discarded=0, stats=None, trace=None):
    arm = tag[:2]
    on = arm == "L1"
    w = {}
    for name, rate in (("W16", rate16), ("W1", rate1)):
        b = 16 if name == "W16" else 1
        c = calls.get(name) if calls else ({"issues": 1000, "collects": 1000, "overlapped": 992} if on
                                           else {"issues": 0, "collects": 0, "overlapped": 0})
        w[name] = {"batch": b, "decode_tok_s": rate, "decode_ms_per_step": 1000.0 / rate * b,
                   "tokens": {"32": toks[:b], "160": toks[:b]}, "lookahead_calls": c,
                   "rep_digests": {"32": ["x", "x", "x" if rep_ok else "y"], "160": ["z"] * 3}}
    tot = {k: sum(w[n]["lookahead_calls"][k] for n in w) for k in ("issues", "collects", "overlapped")}
    return {"arm": arm, "tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": model,
            "revision": REVS[model], "graph_status": graphs or {str(b): "graph" for b in BUCKETS},
            "fusions": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0], "fuse_router_epilogue_n": 0},
            "prompts_sha256": "p", "short": 32, "long": 160, "reps": 3, "workloads": w,
            "lookahead": on if look is None else look, "lookahead_calls": tot, "lookahead_discarded": discarded,
            "graph_stats": stats or {"1": {"replays": 1270, "eager_steps": 0, "rows": 1270, "pad_rows": 0}},
            "trace": trace if trace is not None else {
                n: ({"step_ms_p50": 8.0, "device_ms_p50": 8.0, "host_gap_ms_p50": 0.6} if on
                    else {"step_ms_p50": 8.6, "device_ms_p50": 8.0, "host_gap_ms_p50": 0.6}) for n in WORKLOADS}}


def self_test() -> int:
    toks = [[i, i + 1] for i in range(16)]
    other = [[i, i + 2] for i in range(16)]

    def arms(**over):
        base = {"L0a": (800.0, 110.0), "L1a": (816.0, 117.0), "L1b": (816.5, 116.8), "L0b": (799.0, 110.2)}
        out = {}
        for t, (r16, r1) in base.items():
            kw = dict(over.get(t, {}))
            out[t] = _fake_arm(t, kw.pop("r16", r16), kw.pop("r1", r1), kw.pop("toks", toks), **kw)
        return out
    v = lambda a, **kw: reduce(a, E, **kw)["verdict"]  # noqa: E731
    low = {w: {"issues": 1000, "collects": 1000, "overlapped": 500} for w in WORKLOADS}
    unbalanced = {w: {"issues": 1000, "collects": 999, "overlapped": 990} for w in WORKLOADS}
    cases = [
        ("default on", v(arms()) == "DEFAULT_ON"),
        ("slower at W1", v(arms(L1a={"r1": 111.0}, L1b={"r1": 111.0})) == "SLOWER"),
        ("a 1.5 % gain is under the bar", v(arms(L1a={"r1": 111.6}, L1b={"r1": 111.8})) == "SLOWER"),
        ("W16 regression", v(arms(L1a={"r16": 780.0}, L1b={"r16": 780.0})) == "SLOWER"),
        ("noisy at W1 (1.5 % drift)", v(arms(L0b={"r1": 111.7})) == "NOISY"),
        ("W16 drift inside its band is not", v(arms(L0b={"r16": 820.0}, L1b={"r16": 836.0})) == "DEFAULT_ON"),
        ("L1 != L0 tokens", v(arms(L1a={"toks": other}, L1b={"toks": other})) == "FUNCTION_FAIL"),
        ("L1b != L1a tokens", v(arms(L1b={"toks": other})) == "FUNCTION_FAIL"),
        ("reps differ", v(arms(L0a={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("bucket statistics differ", v(arms(L1a={"stats": {"1": {"replays": 1269}}})) == "FUNCTION_FAIL"),
        ("L1 without the lookahead", v(arms(L1a={"look": False})) == "VOID"),
        ("L1 barely overlapped", v(arms(L1b={"calls": low})) == "VOID"),
        ("L1 left a step uncollected", v(arms(L1a={"calls": unbalanced})) == "VOID"),
        ("L1 discarded with no stop ids", v(arms(L1a={"discarded": 1})) == "VOID"),
        ("L0 called the lookahead", v(arms(L0a={"calls": {w: {"issues": 1, "collects": 1, "overlapped": 0}
                                                          for w in WORKLOADS}})) == "VOID"),
        ("L0 ran the lookahead scheduler", v(arms(L0b={"look": True})) == "VOID"),
        ("a bucket not captured", v(arms(L1a={"graphs": {"1": "eager"}})) == "VOID"),
        ("wrong commit", v(arms(L0a={"e4b": "b" * 40})) == "VOID"),
        ("missing arm", reduce({k: x for k, x in arms().items() if k != "L1b"}, E)["verdict"] == "VOID"),
        ("the mechanism is reported", reduce(arms(), E)["mechanism"]["saved_vs_gap"]["W1"]
         == {"l0_host_gap_ms": 0.6, "l0_device_ms": 8.0, "period_saved_ms": 0.6, "saved_over_gap": 1.0}),
        ("a missing trace is reported, not ruled", (lambda r: r["verdict"] == "DEFAULT_ON"
                                                   and r["mechanism"]["saved_vs_gap"]["W1"] == "trace incomplete")(
            reduce(arms(L0a={"trace": {}}), E))),
        ("proof", reduce({t: _fake_arm(t, 300.0, 50.0 if t[:2] == "L0" else 52.0, toks, model=GRAN) for t in TAGS},
                         E, proof=True)["verdict"] == "DEFAULT_ON"),
        ("proof on qwen is void", reduce(arms(), E, proof=True)["verdict"] == "VOID"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p118_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    v = reduce(arms, a.e4b_sha, proof=a.proof)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P118_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
