#!/usr/bin/env python3
"""p111_reduce.py -- lane P111's registered rule (bench/p111/PREREG-p111.md), from the four arm receipts.

Receipts: arm_S0a.json, arm_S1a.json, arm_S1b.json, arm_S0b.json (run in that order, ABBA), each from p111_box.py. S0 is
the per-layer KV selection (``E4B_KV_STEP_SELECT`` off, the shipped default), S1 one selection per step (on). Both are the
default ``serve_paged`` server, which captures bucketed decode graphs.

The verdict is the first of these that applies:
  VOID          an arm is missing or not ok; a receipt names another e4b / grouped-nf4-gemm commit or model revision; the
                arms read different prompts or ran different lengths; a slope is void; or engagement is wrong: every arm
                must report "graph" for every bucket in (1, 2, 4, 8, 16), and its ``step_select`` must match its arm.
  NOISY         a self-pair (S0b/S0a or S1b/S1a, decode tok/s, either workload) falls outside [0.96, 1.04].
  FUNCTION_FAIL S1a or S1b emits a token different from S0a's on any row of either workload at either length, S0b differs
                from S0a, or an arm's timed reps do not all digest the same.
  SLOWER        g16 = min(S1a/S0a, S1b/S0b) on W16 < 1.00, or g1 = the same on W1 < 1.00.
  DEFAULT_ON    otherwise.

Reported beside the verdict: the pair ratios, their geometric means, and ms per step for every arm.
"""
import argparse
import json
import math
import os
import sys

TAGS = ("S0a", "S1a", "S1b", "S0b")
BUCKETS = ("1", "2", "4", "8", "16")
SELF_LO, SELF_HI = 0.96, 1.04
GAIN_MIN = 1.00
GNF4_SHA = "51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00"
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def reduce(arms: dict, e4b_sha: str) -> dict:
    out = {"verdict": None, "reasons": [], "bars": {"self_pair": [SELF_LO, SELF_HI], "gain_min": GAIN_MIN}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"arm(s) missing or not ok: {missing}"])
        return out
    void = []
    for t, r in arms.items():
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"{t}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{t}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(r.get("model")) != r.get("revision"):
            void.append(f"{t}: model {r.get('model')}@{r.get('revision')} is not a registered revision")
        g = r.get("graph_status") or {}
        if [g.get(b) for b in BUCKETS] != ["graph"] * 5:
            void.append(f"{t}: graph_status {g}")
        if bool(r.get("step_select")) != t.startswith("S1"):
            void.append(f"{t}: step_select={r.get('step_select')}")
    if len({r["prompts_sha256"] for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r["short"], r["long"], r["reps"]) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    rates = {t: {w: _rate(arms[t], w) for w in ("W16", "W1")} for t in TAGS}
    if any(v is None for d in rates.values() for v in d.values()):
        void.append("a decode slope is void")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    S0a, S1a, S1b, S0b = (arms[t] for t in TAGS)
    short, long_ = S0a["short"], S0a["long"]
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in ("W16", "W1")} for t in TAGS}
    selfp = {f"{a}/{b} {w}": rates[a][w] / rates[b][w] for a, b in (("S0b", "S0a"), ("S1b", "S1a")) for w in ("W16", "W1")}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    pairs = {w: [rates["S1a"][w] / rates["S0a"][w], rates["S1b"][w] / rates["S0b"][w]] for w in ("W16", "W1")}
    out["pair_ratios"] = {w: [round(x, 4) for x in v] for w, v in pairs.items()}
    out["g16"], out["g1"] = round(min(pairs["W16"]), 4), round(min(pairs["W1"]), 4)
    out["geomean"] = {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()}
    fn = []
    for w in ("W16", "W1"):
        for n in (short, long_):
            for t in ("S1a", "S1b", "S0b"):
                if _tok(arms[t], w, n) != _tok(S0a, w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[t], w, n), _tok(S0a, w, n))) if x != y]
                    fn.append(f"{t} != S0a on {w} at {n} tokens, rows {rows}")
    for t in TAGS:
        for w in ("W16", "W1"):
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    noisy = [f"{k} = {v:.4f}" for k, v in selfp.items() if not SELF_LO <= v <= SELF_HI]
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif out["g16"] < GAIN_MIN or out["g1"] < GAIN_MIN:
        out.update(verdict="SLOWER", reasons=[f"g16 = {out['g16']}, g1 = {out['g1']} (bar {GAIN_MIN})"])
    else:
        out.update(verdict="DEFAULT_ON", reasons=[f"tokens identical on every row; g16 = {out['g16']}, g1 = {out['g1']} "
                                                  f"(geomean {out['geomean']['W16']} / {out['geomean']['W1']})"])
    return out


# ------------------------------------------------------------------ self-test --

def _fake(tag, rate16, rate1, toks, *, e4b="a" * 40, graphs=None, step=None, digests=None, short=4, long_=8):
    if graphs is None:
        graphs = {b: "graph" for b in BUCKETS}

    def wl(rate):
        return {"decode_tok_s": rate, "decode_ms_per_step": round(1000 / rate, 3),
                "tokens": {str(short): [t[:short] for t in toks], str(long_): toks},
                "rep_digests": digests or {str(short): ["x"] * 3, str(long_): ["y"] * 3}}
    return {"tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": "Qwen/Qwen3-30B-A3B",
            "revision": REVS["Qwen/Qwen3-30B-A3B"], "graph_status": graphs,
            "step_select": tag.startswith("S1") if step is None else step, "prompts_sha256": "p", "short": short,
            "long": long_, "reps": 3, "workloads": {"W16": wl(rate16), "W1": wl(rate1)}}


def self_test() -> int:
    base = [[r * 100 + i for i in range(8)] for r in range(16)]

    def arms(**over):
        a = {"S0a": _fake("S0a", 740.0, 99.4, base), "S1a": _fake("S1a", 768.0, 104.0, base),
             "S1b": _fake("S1b", 770.0, 104.2, base), "S0b": _fake("S0b", 742.0, 99.5, base)}
        a.update(over)
        return a
    E = "a" * 40
    cases = []
    cases.append(("default on", reduce(arms(), E)["verdict"] == "DEFAULT_ON"))
    cases.append(("missing", reduce({k: v for k, v in arms().items() if k != "S0b"}, E)["verdict"] == "VOID"))
    cases.append(("wrong e4b", reduce(arms(S1a=_fake("S1a", 768.0, 104.0, base, e4b="b" * 40)), E)["verdict"] == "VOID"))
    cases.append(("switch not in force", reduce(arms(S1b=_fake("S1b", 770.0, 104.2, base, step=False)), E)["verdict"] == "VOID"))
    cases.append(("a bucket eager", reduce(arms(S0a=_fake("S0a", 740.0, 99.4, base,
                                                            graphs={**{b: "graph" for b in BUCKETS}, "16": "eager: x"})), E)["verdict"] == "VOID"))
    cases.append(("noisy", reduce(arms(S0b=_fake("S0b", 700.0, 99.5, base)), E)["verdict"] == "NOISY"))
    off = [list(r) for r in base]
    off[5][6] += 1
    cases.append(("a token moves", reduce(arms(S1b=_fake("S1b", 770.0, 104.2, off)), E)["verdict"] == "FUNCTION_FAIL"))
    cases.append(("reps differ", reduce(arms(S1a=_fake("S1a", 768.0, 104.0, base, digests={"4": ["x"] * 3, "8": ["y", "z", "y"]})), E)["verdict"] == "FUNCTION_FAIL"))
    cases.append(("slower at 16", reduce(arms(S1a=_fake("S1a", 735.0, 104.0, base), S1b=_fake("S1b", 736.0, 104.2, base)), E)["verdict"] == "SLOWER"))
    cases.append(("slower at 1", reduce(arms(S1a=_fake("S1a", 768.0, 99.0, base), S1b=_fake("S1b", 770.0, 99.1, base)), E)["verdict"] == "SLOWER"))
    r = reduce(arms(), E)
    cases.append(("report", r["g16"] == round(min(768 / 740, 770 / 742), 4) and r["geomean"]["W1"] > 1.0))
    bad = [n for n, ok in cases if not ok]
    print(f"p111_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    v = reduce(arms, a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1)
    print(f"P111_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("g16", "g1", "geomean", "pair_ratios", "self_pairs", "decode_tok_s", "ms_per_step"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
