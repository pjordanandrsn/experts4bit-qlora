#!/usr/bin/env python3
"""p113_reduce.py -- lane P113's registered rule (bench/p113/PREREG-p113.md), from the six arm receipts.

Receipts: arm_OFF1, arm_ALL1, arm_CAP1, arm_CAP2, arm_ALL2, arm_OFF2 (.json; run in that order, a palindrome), each from
p113_box.py. OFF is ``GNF4_PDL=0`` (the shipped default), ALL is ``GNF4_PDL=1``, CAP is ``GNF4_PDL=1
GNF4_PDL_MAX_ROWS=8``. All six are P112's subject: the default serve_paged server (graphs on) with SC1's int4
configuration. Every rate is the decode-only slope (each pass's wall minus its largest ttft).

The verdict is the first of these that applies:
  VOID          an arm is missing or not ok; a receipt names another e4b / grouped-nf4-gemm commit or model revision; the
                arms read different prompts or ran different lengths; a decode-only slope is void; or engagement is wrong:
                - every arm reports "graph" for every bucket in (1, 2, 4, 8, 16);
                - SC1's int4 configuration is in force in every arm, with the same counts in all six;
                - ``pdl_active`` is False in OFF and True in ALL and CAP; ``pdl_cap`` is 8 in CAP and 0 elsewhere;
                - every build launches the same number of switched kernels, more than zero, in P112 Amendment 1's
                  accounting form, with no more compile launches than variants. In OFF nothing carries PDL. In ALL every
                  variant and every handled launch does. In CAP the cap splits the build: at least one handled launch
                  carries PDL and at least one does not.
  NOISY         a self-pair (OFF2/OFF1, ALL2/ALL1 or CAP2/CAP1, decode tok/s, either workload) falls outside
                [0.96, 1.04].
  FUNCTION_FAIL an arm emits a token different from OFF1's on any row of either workload at either length, or an arm's
                timed reps do not all digest the same.
  ALL_DEFAULT   gALL1 = min(ALL1/OFF1, ALL2/OFF2) on W1 >= 1.02 and gALL16, the same on W16, >= 1.00.
  CAP_DEFAULT   otherwise, gCAP1 >= 1.02 and gCAP16 >= 0.99.
  NONE          otherwise.

Reported beside the verdict: every pair ratio and geometric mean, ms per step per arm, the whole-pass slopes, and the
launch accounting.
"""
import argparse
import json
import math
import os
import sys

TAGS = ("OFF1", "ALL1", "CAP1", "CAP2", "ALL2", "OFF2")
BUCKETS = ("1", "2", "4", "8", "16")
SELF_LO, SELF_HI = 0.96, 1.04
GAIN_B1 = 1.02          # a B=1 gain worth a default
ALL_B16 = 1.00          # PDL everywhere may not cost B=16
CAP_B16 = 0.99          # the cap launches nothing at B=16 with PDL: the noise allowance around 1.00
CAP_ROWS = 8
GNF4_SHA = "bc2214ce46fe3d1967ccce3ca36aca73f23198e9"
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}
INT4_ON = ("int4_attn_projections", "fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n")


def _setting(t):
    return t[:3]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _launches(r):
    acct = r.get("pdl_launches_build") or {}
    return sum(v[0] for v in acct.values())


def _engagement(t, r):
    out, s = [], _setting(t)
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
    if bool(r.get("pdl_active")) != (s != "OFF"):
        out.append(f"{t}: pdl_active={r.get('pdl_active')}")
    if r.get("pdl_cap") != (CAP_ROWS if s == "CAP" else 0):
        out.append(f"{t}: pdl_cap={r.get('pdl_cap')}")
    acct = r.get("pdl_launches_build") or {}
    if not acct or _launches(r) == 0:
        out.append(f"{t}: the build launched no switched gnf4 kernel")
    handled = with_pdl = 0
    for name, row in sorted(acct.items()):
        if len(row) != 5:
            out.append(f"{t}: {name} accounting row {row} is not P112 Amendment 1's form")
            continue
        n, w, compiles, variants, variants_pdl = row
        handled += n - compiles
        with_pdl += w
        if compiles > variants:
            out.append(f"{t}: {name} has {compiles} compile launches but {variants} compiled variants")
        if s == "OFF" and (w or variants_pdl):
            out.append(f"{t}: {name} carried PDL with the switch off ({w} launches, {variants_pdl} variants)")
        if s == "ALL" and (variants_pdl != variants or variants == 0 or w != n - compiles):
            out.append(f"{t}: {name} {w} of {n - compiles} handled launches and {variants_pdl} of {variants} variants carried PDL")
    if s == "CAP" and not (0 < with_pdl < handled):
        out.append(f"{t}: the cap did not split the build ({with_pdl} of {handled} handled launches carried PDL)")
    return out


def reduce(arms: dict, e4b_sha: str) -> dict:
    out = {"verdict": None, "reasons": [], "bars": {"self_pair": [SELF_LO, SELF_HI], "gain_b1": GAIN_B1,
                                                    "all_b16": ALL_B16, "cap_b16": CAP_B16}}
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
    if len({_launches(r) for r in arms.values()}) != 1:
        void.append(f"the builds launched different numbers of switched kernels: {[_launches(arms[t]) for t in TAGS]}")
    if len({r["prompts_sha256"] for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r["short"], r["long"], r["reps"]) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    rates = {t: {w: _rate(arms[t], w) for w in ("W16", "W1")} for t in TAGS}
    if any(v is None for d in rates.values() for v in d.values()):
        void.append("a decode-only slope is void")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    OFF1 = arms["OFF1"]
    short, long_ = OFF1["short"], OFF1["long"]
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in ("W16", "W1")} for t in TAGS}
    out["whole_pass_tok_s"] = {t: {w: (arms[t]["workloads"][w].get("whole_pass") or {}).get("decode_tok_s")
                                   for w in ("W16", "W1")} for t in TAGS}
    out["switched_launches_build"] = {t: [_launches(arms[t]),
                                          sum(v[1] for v in arms[t]["pdl_launches_build"].values())] for t in TAGS}
    selfp = {f"{s}2/{s}1 {w}": rates[f"{s}2"][w] / rates[f"{s}1"][w] for s in ("OFF", "ALL", "CAP") for w in ("W16", "W1")}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    pairs = {s: {w: [rates[f"{s}1"][w] / rates["OFF1"][w], rates[f"{s}2"][w] / rates["OFF2"][w]] for w in ("W16", "W1")}
             for s in ("ALL", "CAP")}
    out["pair_ratios"] = {s: {w: [round(x, 4) for x in v] for w, v in d.items()} for s, d in pairs.items()}
    out["g"] = {s: {w: round(min(v), 4) for w, v in d.items()} for s, d in pairs.items()}
    out["geomean"] = {s: {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in d.items()} for s, d in pairs.items()}
    fn = []
    for w in ("W16", "W1"):
        for n in (short, long_):
            for t in TAGS[1:]:
                if _tok(arms[t], w, n) != _tok(OFF1, w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[t], w, n), _tok(OFF1, w, n))) if x != y]
                    fn.append(f"{t} != OFF1 on {w} at {n} tokens, rows {rows}")
    for t in TAGS:
        for w in ("W16", "W1"):
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    noisy = [f"{k} = {v:.4f}" for k, v in selfp.items() if not SELF_LO <= v <= SELF_HI]
    g = out["g"]
    txt = (f"ALL g1 {g['ALL']['W1']} g16 {g['ALL']['W16']}; CAP g1 {g['CAP']['W1']} g16 {g['CAP']['W16']} "
           f"(geomeans ALL {out['geomean']['ALL']}, CAP {out['geomean']['CAP']})")
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif g["ALL"]["W1"] >= GAIN_B1 and g["ALL"]["W16"] >= ALL_B16:
        out.update(verdict="ALL_DEFAULT", reasons=[f"tokens identical on every row; {txt}"])
    elif g["CAP"]["W1"] >= GAIN_B1 and g["CAP"]["W16"] >= CAP_B16:
        out.update(verdict="CAP_DEFAULT", reasons=[f"tokens identical on every row; {txt}"])
    else:
        out.update(verdict="NONE", reasons=[f"tokens identical on every row; {txt}"])
    return out


# ------------------------------------------------------------------ self-test --

INT4_OK = {"int4_expert_layers": 48, "moe_layers": 48, "int4_attn_projections": 96, "fuse_qkv_n": 48,
           "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48}


def _fake(tag, rate16, rate1, toks, *, e4b="a" * 40, graphs=None, pdl=None, cap=None, acct=None, int4=None,
          digests=None, short=4, long_=8):
    s = _setting(tag)
    if graphs is None:
        graphs = {b: "graph" for b in BUCKETS}
    if acct is None:
        acct = {"OFF": {"_gemv_int4_b32": [576, 0, 4, 4, 0], "_rmsnorm_rows": [879, 0, 4, 4, 0]},
                "ALL": {"_gemv_int4_b32": [576, 572, 4, 4, 4], "_rmsnorm_rows": [879, 875, 4, 4, 4]},
                "CAP": {"_gemv_int4_b32": [576, 300, 4, 4, 3], "_rmsnorm_rows": [879, 500, 4, 4, 3]}}[s]

    def wl(rate):
        return {"decode_tok_s": rate, "decode_ms_per_step": round(1000 / rate, 3) if rate else None,
                "tokens": {str(short): [t[:short] for t in toks], str(long_): toks},
                "rep_digests": digests or {str(short): ["x"] * 3, str(long_): ["y"] * 3},
                "whole_pass": {"decode_tok_s": rate * 0.98 if rate else None}}
    return {"tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": "Qwen/Qwen3-30B-A3B",
            "revision": REVS["Qwen/Qwen3-30B-A3B"], "graph_status": graphs,
            "pdl_active": (s != "OFF") if pdl is None else pdl,
            "pdl_cap": (CAP_ROWS if s == "CAP" else 0) if cap is None else cap, "pdl_launches_build": acct,
            "int4": dict(INT4_OK) if int4 is None else int4, "prompts_sha256": "p", "short": short,
            "long": long_, "reps": 3, "workloads": {"W16": wl(rate16), "W1": wl(rate1)}}


def self_test() -> int:
    base = [[r * 100 + i for i in range(8)] for r in range(16)]
    RATES = {"OFF": (1760.0, 225.0), "ALL": (1735.0, 233.5), "CAP": (1758.0, 233.0)}

    def arms(rates=None, **over):
        rt = dict(RATES, **(rates or {}))
        a = {t: _fake(t, *rt[_setting(t)], base) for t in TAGS}
        a.update(over)
        return a

    def v(a):
        return reduce(a, "a" * 40)["verdict"]
    cases = [
        ("cap default", v(arms()) == "CAP_DEFAULT"),
        ("all default", v(arms(rates={"ALL": (1765.0, 233.5)})) == "ALL_DEFAULT"),
        ("all default wins over cap", v(arms(rates={"ALL": (1762.0, 233.0), "CAP": (1761.0, 234.0)})) == "ALL_DEFAULT"),
        ("cap at the 0.99 boundary", v(arms(rates={"CAP": (1760.0 * 0.99, 233.0)})) == "CAP_DEFAULT"),
        ("cap costs B=16", v(arms(rates={"CAP": (1735.0, 233.0)})) == "NONE"),
        ("no B=1 gain", v(arms(rates={"ALL": (1735.0, 226.0), "CAP": (1758.0, 226.0)})) == "NONE"),
        ("missing", v({k: x for k, x in arms().items() if k != "OFF2"}) == "VOID"),
        ("wrong e4b", v(arms(ALL1=_fake("ALL1", *RATES["ALL"], base, e4b="b" * 40))) == "VOID"),
        ("switch not read", v(arms(CAP2=_fake("CAP2", *RATES["CAP"], base, pdl=False))) == "VOID"),
        ("cap not read", v(arms(CAP1=_fake("CAP1", *RATES["CAP"], base, cap=0))) == "VOID"),
        ("pdl with the switch off", v(arms(OFF2=_fake("OFF2", *RATES["OFF"], base,
                                                       acct={"_gemv_int4_b32": [576, 1, 4, 4, 0], "_rmsnorm_rows": [879, 0, 4, 4, 0]}))) == "VOID"),
        ("all missed a launch", v(arms(ALL2=_fake("ALL2", *RATES["ALL"], base,
                                                   acct={"_gemv_int4_b32": [576, 571, 4, 4, 4], "_rmsnorm_rows": [879, 875, 4, 4, 4]}))) == "VOID"),
        ("the cap did not split", v(arms(CAP1=_fake("CAP1", *RATES["CAP"], base,
                                                     acct={"_gemv_int4_b32": [576, 572, 4, 4, 4], "_rmsnorm_rows": [879, 875, 4, 4, 4]}))) == "VOID"),
        ("old accounting form", v(arms(OFF1=_fake("OFF1", *RATES["OFF"], base, acct={"_gemv_int4_b32": [576, 0]}))) == "VOID"),
        ("launch counts differ", v(arms(ALL1=_fake("ALL1", *RATES["ALL"], base,
                                                    acct={"_gemv_int4_b32": [570, 566, 4, 4, 4], "_rmsnorm_rows": [879, 875, 4, 4, 4]}))) == "VOID"),
        ("int4 not in force", v(arms(OFF1=_fake("OFF1", *RATES["OFF"], base, int4={**INT4_OK, "fuse_qkv_n": 0}))) == "VOID"),
        ("a bucket eager", v(arms(CAP2=_fake("CAP2", *RATES["CAP"], base,
                                             graphs={**{b: "graph" for b in BUCKETS}, "16": "eager: x"}))) == "VOID"),
        ("void slope", v(arms(ALL1=_fake("ALL1", None, 233.5, base))) == "VOID"),
        ("noisy", v(arms(OFF2=_fake("OFF2", 1650.0, 225.0, base))) == "NOISY"),
    ]
    off = [list(r) for r in base]
    off[3][5] += 1
    cases.append(("a token moves", v(arms(CAP2=_fake("CAP2", *RATES["CAP"], off))) == "FUNCTION_FAIL"))
    cases.append(("reps differ", v(arms(ALL1=_fake("ALL1", *RATES["ALL"], base,
                                                   digests={"4": ["x"] * 3, "8": ["y", "z", "y"]}))) == "FUNCTION_FAIL"))
    r = reduce(arms(), "a" * 40)
    cases.append(("report", r["g"]["CAP"]["W1"] == round(233.0 / 225.0, 4) and r["switched_launches_build"]["ALL1"] == [1455, 1447]
                  and r["whole_pass_tok_s"]["OFF1"]["W1"] == round(225.0 * 0.98, 4)))
    bad = [n for n, ok in cases if not ok]
    print(f"p113_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    print(f"P113_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("g", "geomean", "pair_ratios", "self_pairs", "decode_tok_s", "ms_per_step", "whole_pass_tok_s",
              "switched_launches_build"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
