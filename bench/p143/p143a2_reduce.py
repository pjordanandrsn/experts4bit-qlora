#!/usr/bin/env python3
"""p143a2_reduce.py -- lane P143 Amendment 2's registered rule (bench/p143/PREREG-p143.md, "Amendment 2"; e4b#1313).

Records (``p143a2_box.py`` / ``p143a2_kernel.py``): ``kernel.json`` (the tolerance contract and the B 2/4 timing),
``arm_{S0a,S1a,S1b,S0b}.json`` (speed, W16 and W1, on the default graph server) and ``sane_off.json`` / ``sane_on.json``
(Phase B's instrument at one window per pass, T == 1, on the default server built eager). S0 = ``E4B_ATTN_SPLIT_CAP=0``
(the library's plan, named); S1 = ``E4B_ATTN_SPLIT_CAP=16``. Everything else is the shipped default: the B=1 fused stack
(auto on qwen3_moe) and the bandwidth decode GEMV.

The rule, first rung that applies:
  VOID           a record missing or not ok; another e4b / grouped-nf4-gemm commit or model revision; a bucket not
                 captured; the arms read different prompts or lengths; a void slope; the fused stack's census or modes off
                 the default's; the bandwidth GEMV not where the default puts it (Phase D's checks); the cap's engagement
                 off its registration -- S1's build records cap 16 (explicit) with the captured buckets 1, 2, 4 at 16 splits
                 and 8, 16 at the library's plan, S0's records no cap and no plan, S1's SANE phase capped B = 1 and S0's
                 did not; a SANE phase not at one window per pass or short of its windows; the kernel record short of its
                 grid.
  KERNEL_FAIL    (a) at any B in {1, 2, 4} and context in {256, 640, 1024, 2048, 4096}: the cap-16 output's RMS error
                 against the fp64 reference > 1.05 x the library plan's; or (b) at any B in {2, 4} and context in {640, 1024,
                 2048}: the cap-16 time minus the plan's > the larger of the two arms' |pass 1 - pass 2|.
  FUNCTION_FAIL  a self-pair (S0b/S0a, S1b/S1a) decodes different tokens on any row, workload or length; one arm's timed reps
                 digest differently; or W16 decodes different tokens in S1 than in S0 (the cap cannot bind at B = 16, which
                 plans 5 splits: W16 is the no-change check).
  QUALITY_FAIL   the SANE gate fails: |mean NLL difference ON - R| > 0.02 nats or argmax agreement < 0.95 over 12 windows.
  FLIP           g1 = min(S1a/S0a, S1b/S0b) at W1 >= 1.03: qwen3_moe enters ATTN_SPLIT_CAP_DEFAULT_FAMILIES at 16.
  NO_FLIP        otherwise: the cap stays opt-in, with the numbers.

Reported, not ruled: g16, the self-pairs, the SANE KL, every kernel cell, the W1 token identity of S1 against S0.

    python p143a2_reduce.py --dir RUN_DIR --out verdict_p143a2.json --e4b-sha SHA
    python p143a2_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

TAGS = ("S0a", "S1a", "S1b", "S0b")
WORKLOADS = ("W16", "W1")
BUCKETS = (1, 2, 4, 8, 16)
GNF4_SHA = "38b308887688ea5bda7dd449bd3cc3dc1426b995"     # grouped-nf4-gemm v0.47.1: decode_split_plan + split_cap
MODEL, REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
CENSUS = {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48}
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
CAP = {"S0": None, "S1": 16}
# what the 5090's library plan gives each captured bucket at Qwen3's 4 KV heads and 4096-token slots, and the cap's
CAPTURED_PLANS = {1: (32, 16), 2: (32, 16), 4: (23, 16), 8: (11, 11), 16: (5, 5)}
SANE_WINDOWS, SANE_BIAS, SANE_ARGMAX = 12, 0.02, 0.95       # Phase C/D's gate, unchanged
CONTRACT_B, CONTRACT_CTX, CONTRACT_RATIO = (1, 2, 4), (256, 640, 1024, 2048, 4096), 1.05
TIMING_B, TIMING_CTX = (2, 4), (640, 1024, 2048)
FLIP_G1 = 1.03
BW = ("bw_tree", "bw_prmt32", "bw_splitk")
SMS = 170                                                   # the RTX 5090: the plans above are its
SERVED_TOKENS, SERVED_BLOCKS = 4096, 256                    # the server's KV per sequence (16-token blocks)
REF_SANITY = 0.25                                           # the plan's RMS error / the reference's RMS: the reference
                                                            # must describe the kernel (a wrong head map reads ~1.4)


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def gemv_faults(name, d) -> list:
    d = d or {}
    if d.get("bw_prmt32", 0) <= 0 or d.get("dotpad", 0) or d.get("dotpad_splitk", 0) or d.get("bw_tree", 0):
        return [f"{name}: dispatched {d}; the default is bw_prmt32 and no dot-pad at Qwen3's shapes"]
    return []


def w1_faults(t, r) -> list:
    """Phase D's (Amendment 4): W1 replays the bucket-1 graph captured at the build, so the engagement is the capture
    tally, bound to W1 by bucket 1's replays; an eager W1 step must pass on its own tally."""
    gs1 = (r.get("graph_stats") or {}).get("1") or {}
    out = gemv_faults(f"{t} W1 capture", r.get("dispatch_build"))
    if not gs1.get("replays"):
        out.append(f"{t} W1: bucket 1 never replayed ({gs1}); the capture tally is not what W1 ran")
    if gs1.get("eager_steps") or any((r["workloads"]["W1"].get("dispatch") or {}).values()):
        out += gemv_faults(f"{t} W1 eager", r["workloads"]["W1"].get("dispatch"))
    return out


def record_faults(name, r, e4b_sha, arm) -> list:
    out = []
    if r.get("e4b_sha") != e4b_sha:
        out.append(f"{name}: e4b {r.get('e4b_sha')} != {e4b_sha}")
    if r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{name}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
    if (r.get("model"), r.get("revision")) != (MODEL, REV):
        out.append(f"{name}: model {r.get('model')}@{r.get('revision')} is not {MODEL}@{REV}")
        return out
    if r.get("arm") != arm:
        out.append(f"{name}: the record says arm {r.get('arm')}")
    if r.get("sm_count") != SMS:
        out.append(f"{name}: {r.get('sm_count')} SMs, registered the RTX 5090's {SMS}")
    cen = {k: (r.get("fusions") or {}).get(k) for k in CENSUS}
    if cen != CENSUS:
        out.append(f"{name}: census {cen}, the default's {CENSUS}")
    modes, sources = r.get("fusion_modes") or {}, r.get("fusion_sources") or {}
    if any(str(modes.get(k)) != "auto" or sources.get(k) != "default-allowlisted" for k in FUSION_KNOBS):
        out.append(f"{name}: fusion modes {modes} / sources {sources}, the default is auto (default-allowlisted)")
    return out


def cap_faults(name, r, arm, *, captured) -> list:
    """The cap's engagement. ``captured``: the speed build's record (every bucket at capture); otherwise the SANE phase's
    (the eager decode, B = 1 only)."""
    sc = r.get("attn_split_cap") or {}
    plans = {int(b): p for b, p in (sc.get("plans") or {}).items()}
    if arm == "S0":
        if sc.get("cap") is not None or plans:
            return [f"{name}: S0 must run the library's plan; it records {sc}"]
        return []
    out = []
    if sc.get("cap") != 16 or sc.get("source") != "explicit":
        out.append(f"{name}: cap {sc.get('cap')} from {sc.get('source')}, registered 16 (explicit)")
    want = CAPTURED_PLANS if captured else {1: CAPTURED_PLANS[1]}
    for b, (plan, n) in want.items():
        p = plans.get(b) or {}
        if (p.get("plan"), p.get("n_split")) != (plan, n) or not p.get("calls"):
            out.append(f"{name}: B = {b} recorded {p}, registered plan {plan} -> {n} splits")
    if set(plans) - set(want):
        out.append(f"{name}: the cap met row counts {sorted(set(plans) - set(want))} outside {sorted(want)}")
    return out


def kernel_faults(k) -> tuple:
    """(void, fail, cells): the kernel record's grid complete (void otherwise), and its two registered checks."""
    void, fail, cells = [], [], {}
    k = k or {}
    con, tim = k.get("contract") or {}, k.get("timing") or {}
    if k.get("gnf4_sha") != GNF4_SHA:
        void.append(f"kernel: grouped-nf4-gemm {k.get('gnf4_sha')} != {GNF4_SHA}")
    if (k.get("device") or {}).get("sm_count") != SMS:
        void.append(f"kernel: {(k.get('device') or {}).get('sm_count')} SMs, registered {SMS}")
    if (k.get("geometry") or {}).get("block_table_cols") != SERVED_BLOCKS:
        void.append(f"kernel: block table {(k.get('geometry') or {}).get('block_table_cols')} columns, the server's {SERVED_BLOCKS}")
    for key, bs in (("contract_plans", CONTRACT_B), ("timing_plans", (1,) + TIMING_B)):
        got = {int(b): ((p or {}).get("plan"), (p or {}).get("n_split")) for b, p in (k.get(key) or {}).items()}
        want = {b: CAPTURED_PLANS[b] for b in bs}
        if got != want:
            void.append(f"kernel {key}: {got}, registered {want}")
    for b in CONTRACT_B:
        for c in CONTRACT_CTX:
            x = con.get(f"{b}/{c}")
            if not x or not all(isinstance(x.get(f), (int, float)) and x.get(f) > 0 for f in ("err_plan", "err_cap")):
                void.append(f"kernel contract B={b} ctx={c} missing")
                continue
            ratio = x["err_cap"] / x["err_plan"]
            cells[f"contract {b}/{c}"] = round(ratio, 4)
            if not x.get("ref_rms") or x["err_plan"] / x["ref_rms"] > REF_SANITY:
                void.append(f"kernel contract B={b} ctx={c}: the plan's error {x['err_plan']:.3e} against a reference RMS "
                            f"{x.get('ref_rms')}: the fp64 reference does not describe the kernel")
            if ratio > CONTRACT_RATIO:
                fail.append(f"contract B={b} ctx={c}: err {x['err_cap']:.3e} > {CONTRACT_RATIO} x the plan's {x['err_plan']:.3e}")
    for b in TIMING_B:
        for c in TIMING_CTX:
            try:
                p = [tim[f"{b}/{c}/plan"][s] for s in ("p1", "p2")]
                q = [tim[f"{b}/{c}/cap"][s] for s in ("p1", "p2")]
            except (KeyError, TypeError):
                void.append(f"kernel timing B={b} ctx={c} missing")
                continue
            loss = sum(q) / 2 - sum(p) / 2
            noise = max(abs(p[0] - p[1]), abs(q[0] - q[1]))
            cells[f"timing {b}/{c}"] = {"plan_ms": round(sum(p) / 2, 6), "cap_ms": round(sum(q) / 2, 6),
                                       "loss_ms": round(loss, 6), "noise_ms": round(noise, 6)}
            if loss > noise:
                fail.append(f"timing B={b} ctx={c}: the cap loses {loss:.4f} ms per 48 launches (noise {noise:.4f})")
    return void, fail, cells


def reduce(kernel: dict | None, arms: dict, off: dict | None, on: dict | None, e4b_sha: str) -> dict:
    out = {"lane": "P143 Amendment 2", "verdict": None, "reasons": [],
           "bars": {"contract_ratio": CONTRACT_RATIO, "sane_bias": SANE_BIAS, "sane_argmax": SANE_ARGMAX,
                    "sane_windows": SANE_WINDOWS, "flip_g1": FLIP_G1}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    missing += [n for n, r in (("sane_off", off), ("sane_on", on), ("kernel", kernel)) if not r or r.get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = []
    for t, r in arms.items():
        void += record_faults(t, r, e4b_sha, t[:2])
        void += cap_faults(t, r, t[:2], captured=True)
        g = r.get("graph_status") or {}
        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        void += w1_faults(t, r)
        if any(_rate(r, w) is None for w in WORKLOADS):
            void.append(f"{t}: a decode slope is void")
    if len({r.get("prompts_sha256") for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r.get("short"), r.get("long"), r.get("reps")) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    for name, rec, arm, key in (("sane_off", off, "S0", "R"), ("sane_on", on, "S1", "ON")):
        void += record_faults(name, rec, e4b_sha, arm)
        void += cap_faults(name, rec, arm, captured=False)
        void += gemv_faults(name, rec.get("dispatch_measure"))
        if rec.get("group") != 1:
            void.append(f"{name}: group {rec.get('group')}, registered 1 (T == 1)")
        sk = rec.get("sane_kv") or {}
        served = ((sk.get("max_tokens_per_seq"), sk.get("blocks_per_seq"), sk.get("plan_b1"))
                  == ([SERVED_TOKENS], [SERVED_BLOCKS], CAPTURED_PLANS[1][0]))
        if not served or (sk.get("passes") or 0) < SANE_WINDOWS or sk.get("plan_clash") or sk.get("split_cap") != CAP[arm]:
            void.append(f"{name}: the pass's KV {sk}: registered {SERVED_TOKENS} tokens ({SERVED_BLOCKS} blocks, plan "
                        f"{CAPTURED_PLANS[1][0]} at B = 1), cap {CAP[arm]}, one KV per window")
        n = len(((rec.get("per_window") or {}).get("wikitext") or {}).get(key, []))
        if n != SANE_WINDOWS:
            void.append(f"{name}: {n} windows, registered {SANE_WINDOWS}")
    kvoid, kfail, cells = kernel_faults(kernel)
    void += kvoid
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    out["kernel"] = cells
    S0a, S1a, S1b, S0b = (arms[t] for t in TAGS)
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    pairs = {w: [rates["S1a"][w] / rates["S0a"][w], rates["S1b"][w] / rates["S0b"][w]] for w in WORKLOADS}
    out["speed"] = {"decode_tok_s": rates, "g1": round(min(pairs["W1"]), 4), "g16": round(min(pairs["W16"]), 4),
                    "geomean": {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()},
                    "self_pairs": {f"{a}/{b} {w}": round(rates[a][w] / rates[b][w], 4)
                                   for a, b in (("S0b", "S0a"), ("S1b", "S1a")) for w in WORKLOADS},
                    "w1_s1_vs_s0_identical": {str(n): _tok(S1a, "W1", n) == _tok(S0a, "W1", n)
                                              for n in (S0a["short"], S0a["long"])}}
    fn = []
    for a, b in (("S0b", "S0a"), ("S1b", "S1a")):
        for w in WORKLOADS:
            for n in (S0a["short"], S0a["long"]):
                if _tok(arms[a], w, n) != _tok(arms[b], w, n):
                    fn.append(f"{a} != {b} on {w} at {n} tokens")
    for n in (S0a["short"], S0a["long"]):
        if _tok(S1a, "W16", n) != _tok(S0a, "W16", n):
            fn.append(f"W16 decodes differently with the cap at {n} tokens (it plans 5 splits: nothing should change)")
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
    g1 = out["speed"]["g1"]
    if kfail:
        out.update(verdict="KERNEL_FAIL", reasons=kfail)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif abs(sane["bias"]) > SANE_BIAS or sane["argmax_agree"] < SANE_ARGMAX:
        out.update(verdict="QUALITY_FAIL", reasons=[f"SANE {out['sane']} (gate |bias| <= {SANE_BIAS}, argmax >= {SANE_ARGMAX})"])
    elif g1 >= FLIP_G1:
        out.update(verdict="FLIP", reasons=[f"g1 {g1} >= {FLIP_G1}; SANE bias {sane['bias']:+.5f}, argmax "
                                            f"{sane['argmax_agree']:.4f}; g16 {out['speed']['g16']} (reported)"])
    else:
        out.update(verdict="NO_FLIP", reasons=[f"g1 {g1} < {FLIP_G1}; the cap stays opt-in"])
    return out


# ------------------------------------------------------------------ self-test --
E = "a" * 40


def _cap_record(arm, *, captured, plans=None):
    if arm == "S0":
        return {"cap": None, "source": "explicit", "plans": {}}
    want = CAPTURED_PLANS if captured else {1: CAPTURED_PLANS[1]}
    return {"cap": 16, "source": "explicit",
            "plans": plans if plans is not None else {str(b): {"calls": 48, "plan": p, "n_split": n} for b, (p, n) in want.items()}}


def _fake_arm(tag, r16, r1, toks, *, e4b=E, census=None, graphs=None, tally=None, replays=68, rep_ok=True, cap=None,
              toks16=None, sms=SMS):
    arm = tag[:2]
    d = tally or {"bw_prmt32": 192, "dotpad": 0}
    zero = {k: 0 for k in ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk") + BW}
    w = {}
    for name, rate in (("W16", r16), ("W1", r1)):
        b = 16 if name == "W16" else 1
        tk = (toks16 or toks)[:b] if name == "W16" else toks[:b]
        w[name] = {"batch": b, "decode_tok_s": rate, "tokens": {"32": tk, "160": tk}, "dispatch": dict(zero),
                   "rep_digests": {"32": ["x", "x", "x" if rep_ok else "y"], "160": ["z"] * 3}}
    stats = {str(b): {"replays": replays if b == 1 else 44, "eager_steps": 0} for b in BUCKETS}
    return {"arm": arm, "tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": MODEL, "revision": REV,
            "sm_count": sms, "graph_status": graphs or {str(b): "graph" for b in BUCKETS}, "fusions": census or dict(CENSUS),
            "fusion_modes": {k: "auto" for k in FUSION_KNOBS}, "fusion_sources": {k: "default-allowlisted" for k in FUSION_KNOBS},
            "attn_split_cap": cap if cap is not None else _cap_record(arm, captured=True),
            "prompts_sha256": "p", "short": 32, "long": 160, "reps": 3, "workloads": w,
            "graph_stats": stats, "dispatch_build": dict(d)}


def _fake_sane(arm, *, bias=0.001, agree=0.97, n=SANE_WINDOWS, group=1, cap=None, tokens=SERVED_TOKENS):
    key = "R" if arm == "S0" else "ON"
    per = [{"window": i, "nll": 2.0 + 0.01 * i + (bias if arm == "S1" else 0.0), "argmax_agree": agree, "kl": 0.01}
           for i in range(n)]
    blocks = -(-tokens // 16)
    sk = {"passes": n, "asked": [656], "max_tokens_per_seq": [tokens], "blocks_per_seq": [blocks], "split_cap": CAP[arm],
          "plan_clash": [], "plan_b1": 32 if blocks >= 128 else 11}
    return {"mode": "sane", "arm": arm, "status": "ok", "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": MODEL, "revision": REV,
            "sm_count": SMS, "sane_kv": sk, "fusions": dict(CENSUS), "fusion_modes": {k: "auto" for k in FUSION_KNOBS},
            "fusion_sources": {k: "default-allowlisted" for k in FUSION_KNOBS},
            "attn_split_cap": cap if cap is not None else _cap_record(arm, captured=False),
            "group": group, "per_window": {"wikitext": {key: per}}, "dispatch_measure": {"bw_prmt32": 9000, "dotpad": 0}}


def _fake_kernel(*, err_cap=1.0e-3, err_plan=1.0e-3, plan_ms=0.40, cap_ms=0.36, noise=0.001, drop=None, ref_rms=0.05,
                 sms=SMS, contract_plans=None):
    con = {f"{b}/{c}": {"err_plan": err_plan, "err_cap": err_cap} for b in CONTRACT_B for c in CONTRACT_CTX}
    tim = {}
    for b in TIMING_B:
        for c in TIMING_CTX:
            tim[f"{b}/{c}/plan"] = {"p1": plan_ms, "p2": plan_ms + noise}
            tim[f"{b}/{c}/cap"] = {"p1": cap_ms, "p2": cap_ms + noise}
    for x in con.values():
        x["ref_rms"] = ref_rms
    if drop:
        con.pop(drop, None)
    plans = {str(b): {"calls": 96, "plan": CAPTURED_PLANS[b][0], "n_split": CAPTURED_PLANS[b][1]} for b in CONTRACT_B}
    tplans = {str(b): {"plan": CAPTURED_PLANS[b][0], "n_split": CAPTURED_PLANS[b][1]} for b in (1,) + TIMING_B}
    return {"status": "ok", "gnf4_sha": GNF4_SHA, "device": {"name": "NVIDIA GeForce RTX 5090", "sm_count": sms},
            "geometry": {"block_table_cols": SERVED_BLOCKS}, "contract": con, "timing": tim,
            "contract_plans": plans if contract_plans is None else contract_plans, "timing_plans": tplans}


def self_test() -> int:
    toks = [[i, i + 1] for i in range(16)]
    other = [[i, i + 2] for i in range(16)]

    def arms(r1=(120.0, 130.0, 129.5, 121.0), **over):
        base = {"S0a": (800.0, r1[0]), "S1a": (800.5, r1[1]), "S1b": (800.2, r1[2]), "S0b": (799.0, r1[3])}
        out = {}
        for t, (r16, r1_) in base.items():
            kw = dict(over.get(t, {}))
            out[t] = _fake_arm(t, kw.pop("r16", r16), kw.pop("r1", r1_), kw.pop("toks", toks), **kw)
        return out
    off, on, ker = _fake_sane("S0"), _fake_sane("S1"), _fake_kernel()
    v = lambda a, k=ker, o=off, n=on: reduce(k, a, o, n, E)["verdict"]  # noqa: E731
    s1_no_cap = _cap_record("S1", captured=True, plans={})
    cases = [
        ("g1 1.07 flips", v(arms()) == "FLIP"),
        ("g1 1.02 does not flip", v(arms(r1=(120.0, 122.4, 122.5, 120.0))) == "NO_FLIP"),
        ("g1 is the WORSE pair", reduce(ker, arms(r1=(120.0, 130.0, 122.0, 120.0)), off, on, E)["speed"]["g1"] == round(122 / 120, 4)),
        ("exactly 1.03 flips", v(arms(r1=(100.0, 103.0, 103.0, 100.0))) == "FLIP"),
        ("SANE bias over the gate", v(arms(), n=_fake_sane("S1", bias=0.03)) == "QUALITY_FAIL"),
        ("SANE argmax under the gate", v(arms(), n=_fake_sane("S1", agree=0.94)) == "QUALITY_FAIL"),
        ("the contract fails at 1.06x", v(arms(), k=_fake_kernel(err_cap=1.06e-3)) == "KERNEL_FAIL"),
        ("the contract holds at 1.05x", v(arms(), k=_fake_kernel(err_cap=1.05e-3)) == "FLIP"),
        ("a B 2/4 timing loss beyond noise", v(arms(), k=_fake_kernel(cap_ms=0.42)) == "KERNEL_FAIL"),
        ("a loss inside noise is not a loss", v(arms(), k=_fake_kernel(cap_ms=0.4005, noise=0.001)) == "FLIP"),
        ("a contract cell missing is VOID", v(arms(), k=_fake_kernel(drop="4/4096")) == "VOID"),
        ("W16 decodes differently with the cap", v(arms(S1a={"toks16": other}, S1b={"toks16": other})) == "FUNCTION_FAIL"),
        ("W1 may differ (reported)", v(arms(S1a={"toks": other, "toks16": toks}, S1b={"toks": other, "toks16": toks})) == "FLIP"),
        ("a self-pair differs", v(arms(S1b={"toks": other})) == "FUNCTION_FAIL"),
        ("timed reps differ", v(arms(S0a={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("S1 never engaged the cap", v(arms(S1a={"cap": s1_no_cap})) == "VOID"),
        ("S1's B = 8 capped below its plan", v(arms(S1b={"cap": _cap_record("S1", captured=True, plans={
            **{str(b): {"calls": 1, "plan": p, "n_split": n} for b, (p, n) in CAPTURED_PLANS.items()},
            "8": {"calls": 1, "plan": 11, "n_split": 10}})})) == "VOID"),
        ("S0 engaged a cap", v(arms(S0a={"cap": {"cap": 16, "source": "explicit", "plans": {}}})) == "VOID"),
        ("S1's SANE uncapped", v(arms(), n=_fake_sane("S1", cap={"cap": 16, "source": "explicit", "plans": {}})) == "VOID"),
        ("the fused stack off its default", v(arms(S0a={"census": {**CENSUS, "fuse_qkv_n": 0}})) == "VOID"),
        ("the GEMV on dot-pad", v(arms(S1a={"tally": {"dotpad": 96}})) == "VOID"),
        ("a bucket not captured", v(arms(S1a={"graphs": {"1": "eager"}})) == "VOID"),
        ("bucket 1 never replayed", v(arms(S0b={"replays": 0})) == "VOID"),
        ("SANE at T > 1", v(arms(), o=_fake_sane("S0", group=12)) == "VOID"),
        ("SANE short", v(arms(), n=_fake_sane("S1", n=11)) == "VOID"),
        ("wrong commit", v(arms(S0a={"e4b": "b" * 40})) == "VOID"),
        ("missing arm", reduce(ker, {k: x for k, x in arms().items() if k != "S1b"}, off, on, E)["verdict"] == "VOID"),
        ("missing kernel record", reduce(None, arms(), off, on, E)["verdict"] == "VOID"),
        ("SANE at P115's 656 tokens (the cap cannot bind)", v(arms(), n=_fake_sane("S1", tokens=656)) == "VOID"),
        ("SANE's S0 phase at P115's 656 tokens", v(arms(), o=_fake_sane("S0", tokens=656)) == "VOID"),
        ("a card that is not the 5090", v(arms(S0b={"sms": 188})) == "VOID"),
        ("the kernel record off the 5090", v(arms(), k=_fake_kernel(sms=188)) == "VOID"),
        ("the contract never capped", v(arms(), k=_fake_kernel(contract_plans={})) == "VOID"),
        ("a reference that does not describe the kernel", v(arms(), k=_fake_kernel(err_plan=0.02, err_cap=0.02, ref_rms=0.05)) == "VOID"),
        ("the kernel fails before quality is read", v(arms(), k=_fake_kernel(err_cap=2e-3), n=_fake_sane("S1", bias=0.5)) == "KERNEL_FAIL"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p143a2_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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

    def load(name):
        f = os.path.join(a.dir, name)
        return json.load(open(f)) if os.path.exists(f) else None
    arms = {t: r for t in TAGS if (r := load(f"arm_{t}.json"))}
    v = reduce(load("kernel.json"), arms, load("sane_off.json"), load("sane_on.json"), a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P143A2_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
