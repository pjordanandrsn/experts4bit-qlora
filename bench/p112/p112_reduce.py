#!/usr/bin/env python3
"""p112_reduce.py -- lane P112's registered rule (bench/p112/PREREG-p112.md), from the four arm receipts.

Receipts: arm_P0a.json, arm_P1a.json, arm_P1b.json, arm_P0b.json (run in that order, ABBA), each from p112_box.py. P0 is
``GNF4_PDL`` off (the shipped default), P1 on. All four are the default ``serve_paged`` server (graphs on) with SC1's
int4 configuration.

The verdict is the first of these that applies:
  VOID          an arm is missing or not ok; a receipt names another e4b / grouped-nf4-gemm commit or model revision; the
                arms read different prompts or ran different lengths; a slope is void; or engagement is wrong:
                - every arm must report "graph" for every bucket in (1, 2, 4, 8, 16);
                - SC1's int4 configuration must be in force in every arm (int4 stores on every MoE layer, int4 attention,
                  the fused qkv, T1 glue and router epilogue), with the same counts in all four;
                - ``pdl_active`` must match the arm;
                - the build must launch the same number of switched gnf4 kernels in every arm, more than zero. In P1
                  every compiled variant of a launched switched kernel must carry launch_pdl, every launch with a handle
                  must carry PDL, and the compile launches (the hook sees no handle on them) may not outnumber the
                  variants; in P0 no variant and no launch carries it (Amendment 1).
  NOISY         a self-pair (P0b/P0a or P1b/P1a, decode tok/s, either workload) falls outside [0.96, 1.04].
  FUNCTION_FAIL P1a or P1b emits a token different from P0a's on any row of either workload at either length, P0b differs
                from P0a, or an arm's timed reps do not all digest the same.
  SLOWER        g16 = min(P1a/P0a, P1b/P0b) on W16 < 1.00, or g1 = the same on W1 < 1.00.
  DEFAULT_ON    otherwise.

Reported beside the verdict: the pair ratios, their geometric means, ms per step for every arm, and each arm's
switched-kernel launch accounting.
"""
import argparse
import json
import math
import os
import sys

TAGS = ("P0a", "P1a", "P1b", "P0b")
BUCKETS = ("1", "2", "4", "8", "16")
SELF_LO, SELF_HI = 0.96, 1.04
GAIN_MIN = 1.00
GNF4_SHA = "951a97fbb489b806d238922559b55724b55db8fd"
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}
INT4_ON = ("int4_attn_projections", "fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n")


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _launches(r):
    acct = r.get("pdl_launches_build") or {}
    return sum(v[0] for v in acct.values()), sum(v[1] for v in acct.values())


def _accounting_errors(t, r):
    """Amendment 1: rows are [launches, with launch_pdl, compile launches, variants, variants with launch_pdl]."""
    on, out = t.startswith("P1"), []
    for name, row in sorted((r.get("pdl_launches_build") or {}).items()):
        if len(row) != 5:
            out.append(f"{t}: {name} accounting row {row} is not the amended form")
            continue
        n, with_pdl, compiles, variants, variants_pdl = row
        if compiles > variants:
            out.append(f"{t}: {name} has {compiles} compile launches but {variants} compiled variants")
        if on and (variants_pdl != variants or variants == 0 or with_pdl != n - compiles):
            out.append(f"{t}: {name} {with_pdl} of {n - compiles} handled launches and {variants_pdl} of {variants} variants carried PDL")
        if not on and (variants_pdl or with_pdl):
            out.append(f"{t}: {name} carried PDL in an off arm ({with_pdl} launches, {variants_pdl} variants)")
    return out


def _engagement(t, r):
    out = []
    g = r.get("graph_status") or {}
    if [g.get(b) for b in BUCKETS] != ["graph"] * 5:
        out.append(f"{t}: graph_status {g}")
    i4 = r.get("int4") or {}
    if not i4.get("moe_layers") or i4.get("int4_expert_layers") != i4.get("moe_layers"):
        out.append(f"{t}: int4 expert stores on {i4.get('int4_expert_layers')}/{i4.get('moe_layers')} MoE layers")
    for k in INT4_ON:
        if not i4.get(k):
            out.append(f"{t}: {k}={i4.get(k)!r}, SC1's int4 configuration is not in force")
    r2 = i4.get("fuse_t1_glue_r2_n")
    if not (isinstance(r2, list) and r2 and all(r2)):
        out.append(f"{t}: fuse_t1_glue_r2_n={r2!r}")
    on = t.startswith("P1")
    if bool(r.get("pdl_active")) != on:
        out.append(f"{t}: pdl_active={r.get('pdl_active')}")
    n, _n_pdl = _launches(r)
    if n == 0:
        out.append(f"{t}: the build launched no switched gnf4 kernel")
    out += _accounting_errors(t, r)
    return out


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
        void += _engagement(t, r)
    if len({json.dumps(r.get("int4"), sort_keys=True) for r in arms.values()}) != 1:
        void.append("the arms ran different int4 configurations")
    if len({_launches(r)[0] for r in arms.values()}) != 1:
        void.append(f"the builds launched different numbers of switched kernels: {[_launches(arms[t])[0] for t in TAGS]}")
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
    P0a = arms["P0a"]
    short, long_ = P0a["short"], P0a["long"]
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in ("W16", "W1")} for t in TAGS}
    out["switched_launches_build"] = {t: list(_launches(arms[t])) for t in TAGS}
    selfp = {f"{a}/{b} {w}": rates[a][w] / rates[b][w] for a, b in (("P0b", "P0a"), ("P1b", "P1a")) for w in ("W16", "W1")}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    pairs = {w: [rates["P1a"][w] / rates["P0a"][w], rates["P1b"][w] / rates["P0b"][w]] for w in ("W16", "W1")}
    out["pair_ratios"] = {w: [round(x, 4) for x in v] for w, v in pairs.items()}
    out["g16"], out["g1"] = round(min(pairs["W16"]), 4), round(min(pairs["W1"]), 4)
    out["geomean"] = {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()}
    fn = []
    for w in ("W16", "W1"):
        for n in (short, long_):
            for t in ("P1a", "P1b", "P0b"):
                if _tok(arms[t], w, n) != _tok(P0a, w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[t], w, n), _tok(P0a, w, n))) if x != y]
                    fn.append(f"{t} != P0a on {w} at {n} tokens, rows {rows}")
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

INT4_OK = {"int4_expert_layers": 48, "moe_layers": 48, "int4_attn_projections": 96, "fuse_qkv_n": 48,
           "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48}


def _fake(tag, rate16, rate1, toks, *, e4b="a" * 40, graphs=None, pdl=None, n_launch=4000, n_pdl=None, int4=None,
          digests=None, short=4, long_=8, compiles=4, variants=4, variants_pdl=None):
    if graphs is None:
        graphs = {b: "graph" for b in BUCKETS}
    on = tag.startswith("P1")
    if n_pdl is None:
        n_pdl = n_launch - compiles if on else 0
    if variants_pdl is None:
        variants_pdl = variants if on else 0

    def wl(rate):
        return {"decode_tok_s": rate, "decode_ms_per_step": round(1000 / rate, 3),
                "tokens": {str(short): [t[:short] for t in toks], str(long_): toks},
                "rep_digests": digests or {str(short): ["x"] * 3, str(long_): ["y"] * 3}}
    return {"tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": "Qwen/Qwen3-30B-A3B",
            "revision": REVS["Qwen/Qwen3-30B-A3B"], "graph_status": graphs,
            "pdl_active": on if pdl is None else pdl,
            "pdl_launches_build": {"_gemv_int4_b32": [n_launch, n_pdl, compiles, variants, variants_pdl]},
            "int4": dict(INT4_OK) if int4 is None else int4, "prompts_sha256": "p", "short": short,
            "long": long_, "reps": 3, "workloads": {"W16": wl(rate16), "W1": wl(rate1)}}


def self_test() -> int:
    base = [[r * 100 + i for i in range(8)] for r in range(16)]

    def arms(**over):
        a = {"P0a": _fake("P0a", 1600.0, 220.0, base), "P1a": _fake("P1a", 1630.0, 232.0, base),
             "P1b": _fake("P1b", 1632.0, 233.0, base), "P0b": _fake("P0b", 1602.0, 221.0, base)}
        a.update(over)
        return a
    E = "a" * 40
    cases = []
    cases.append(("default on", reduce(arms(), E)["verdict"] == "DEFAULT_ON"))
    cases.append(("missing", reduce({k: v for k, v in arms().items() if k != "P0b"}, E)["verdict"] == "VOID"))
    cases.append(("wrong e4b", reduce(arms(P1a=_fake("P1a", 1630.0, 232.0, base, e4b="b" * 40)), E)["verdict"] == "VOID"))
    cases.append(("switch not read", reduce(arms(P1b=_fake("P1b", 1632.0, 233.0, base, pdl=False)), E)["verdict"] == "VOID"))
    cases.append(("partial pdl", reduce(arms(P1a=_fake("P1a", 1630.0, 232.0, base, n_pdl=3990)), E)["verdict"] == "VOID"))
    cases.append(("a variant without pdl", reduce(arms(P1b=_fake("P1b", 1632.0, 233.0, base, variants_pdl=3)), E)["verdict"] == "VOID"))
    cases.append(("compiles outnumber variants", reduce(arms(P1a=_fake("P1a", 1630.0, 232.0, base, compiles=5, n_pdl=3995)), E)["verdict"] == "VOID"))
    cases.append(("the pre-amendment row form", reduce(arms(P0a={**_fake("P0a", 1600.0, 220.0, base),
                                                                 "pdl_launches_build": {"_gemv_int4_b32": [4000, 0]}}), E)["verdict"] == "VOID"))
    cases.append(("pdl in an off arm", reduce(arms(P0b=_fake("P0b", 1602.0, 221.0, base, n_pdl=1)), E)["verdict"] == "VOID"))
    cases.append(("a pdl variant in an off arm", reduce(arms(P0a=_fake("P0a", 1600.0, 220.0, base, variants_pdl=1)), E)["verdict"] == "VOID"))
    cases.append(("no switched kernel", reduce(arms(**{t: _fake(t, 1600.0, 220.0, base, n_launch=0) for t in TAGS}), E)["verdict"] == "VOID"))
    cases.append(("launch counts differ", reduce(arms(P1b=_fake("P1b", 1632.0, 233.0, base, n_launch=3990)), E)["verdict"] == "VOID"))
    cases.append(("int4 not in force", reduce(arms(P0a=_fake("P0a", 1600.0, 220.0, base, int4={**INT4_OK, "fuse_t1_glue_n": 0})), E)["verdict"] == "VOID"))
    cases.append(("partial int4 stack", reduce(arms(P1a=_fake("P1a", 1630.0, 232.0, base, int4={**INT4_OK, "int4_expert_layers": 40})), E)["verdict"] == "VOID"))
    cases.append(("a bucket eager", reduce(arms(P0a=_fake("P0a", 1600.0, 220.0, base,
                                                            graphs={**{b: "graph" for b in BUCKETS}, "16": "eager: x"})), E)["verdict"] == "VOID"))
    cases.append(("noisy", reduce(arms(P0b=_fake("P0b", 1520.0, 221.0, base)), E)["verdict"] == "NOISY"))
    off = [list(r) for r in base]
    off[5][6] += 1
    cases.append(("a token moves", reduce(arms(P1b=_fake("P1b", 1632.0, 233.0, off)), E)["verdict"] == "FUNCTION_FAIL"))
    cases.append(("reps differ", reduce(arms(P1a=_fake("P1a", 1630.0, 232.0, base, digests={"4": ["x"] * 3, "8": ["y", "z", "y"]})), E)["verdict"] == "FUNCTION_FAIL"))
    cases.append(("slower at 16", reduce(arms(P1a=_fake("P1a", 1590.0, 232.0, base), P1b=_fake("P1b", 1591.0, 233.0, base)), E)["verdict"] == "SLOWER"))
    cases.append(("slower at 1", reduce(arms(P1a=_fake("P1a", 1630.0, 219.0, base), P1b=_fake("P1b", 1632.0, 219.5, base)), E)["verdict"] == "SLOWER"))
    r = reduce(arms(), E)
    cases.append(("report", r["g1"] == round(min(232 / 220, 233 / 221), 4) and r["geomean"]["W16"] > 1.0
                  and r["switched_launches_build"]["P1a"] == [4000, 3996]))
    bad = [n for n, ok in cases if not ok]
    print(f"p112_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    print(f"P112_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("g16", "g1", "geomean", "pair_ratios", "self_pairs", "decode_tok_s", "ms_per_step", "switched_launches_build"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
