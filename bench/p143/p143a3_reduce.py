#!/usr/bin/env python3
"""p143a3_reduce.py -- lane P143 Amendment 3's registered rule (bench/p143/PREREG-p143.md, "Amendment 3"; e4b#1313):
the confirmatory served read of E4B_ATTN_SPLIT_CAP=16 on Qwen3-30B-A3B, on terms stricter than Amendment 2's, binding
for this lever (no Amendment 4 on the served flip).

Records (``p143a2_box.py``, A2's box at its registered bytes): ``arm_{S0a,S1a,S1b,S0b,S0c,S1c,S1d,S0d}.json`` (ABBA
twice, each through P109's W16 and W1 at five timed reps, on the default graph server) and ``sane_off.json`` /
``sane_on.json`` (A2's SANE: Phase B's instrument at one window per pass, T == 1, the pass KV at the server's 4096
tokens). S0 = ``E4B_ATTN_SPLIT_CAP=0`` (the library's plan, named); S1 = ``E4B_ATTN_SPLIT_CAP=16``. No kernel step: A2's
tolerance contract and timing (``p143a2-5090-1``, the same kernel bytes) are on the record.

The rule, first rung that applies:
  VOID           a record missing or not ok; A2's checks on every record (pins, card, census and modes, the GEMV, the
                 captured buckets, bucket 1 replayed, the cap's engagement at capture and in SANE, the SANE pass KV at
                 4096 tokens, one window per pass, 12 windows); the arms on different prompts or lengths, or not at the
                 registered five reps; a void slope.
  NOISY          (a VOID, and a strike) any W1 self-pair -- S0b/S0a, S1b/S1a, S0d/S0c, S1d/S1c -- outside [0.985, 1.015].
  FUNCTION_FAIL  an arm decodes different tokens from its own arm's first record (every S0 against S0a, every S1 against
                 S1a) on any row, workload or length; one arm's timed reps digest differently.
  QUALITY_FAIL   SANE: |mean NLL difference ON - R| > 0.02 nats or argmax agreement < 0.95 over 12 windows.
  FLIP           g1 = min over the FOUR S1/S0 pairs (S1a/S0a, S1b/S0b, S1c/S0c, S1d/S0d) at W1 >= 1.03 AND g16 (the same
                 at W16) >= 0.99: qwen3_moe enters ATTN_SPLIT_CAP_DEFAULT_FAMILIES at 16.
  NO_FLIP        otherwise (each failed bar named): the cap stays opt-in, permanently for this lever.

Reported, never ruled: every pair and self-pair ratio (the cross-block self-pairs too), the geomean over the four pairs,
the SANE KL, W1 token identity of S1 against S0, and W16's (rows identical out of 16, first divergent step per row).

    python p143a3_reduce.py --dir RUN_DIR --out verdict_p143a3.json --e4b-sha SHA
    python p143a3_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

TAGS = ("S0a", "S1a", "S1b", "S0b", "S0c", "S1c", "S1d", "S0d")      # ABBA twice
PAIRS = (("S1a", "S0a"), ("S1b", "S0b"), ("S1c", "S0c"), ("S1d", "S0d"))
SELF_PAIRS = (("S0b", "S0a"), ("S1b", "S1a"), ("S0d", "S0c"), ("S1d", "S1c"))
NOISY_BAND = (0.985, 1.015)
REPS = 5
WORKLOADS = ("W16", "W1")
BUCKETS = (1, 2, 4, 8, 16)
GNF4_SHA = "38b308887688ea5bda7dd449bd3cc3dc1426b995"     # grouped-nf4-gemm v0.47.1: decode_split_plan + split_cap
MODEL, REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
CENSUS = {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48}
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
CAP = {"S0": None, "S1": 16}
# what the 5090's library plan gives each captured bucket at Qwen3's 4 KV heads and 4096-token slots, and the cap's
CAPTURED_PLANS = {1: (32, 16), 2: (32, 16), 4: (23, 16), 8: (11, 11), 16: (5, 5)}
SANE_WINDOWS, SANE_BIAS, SANE_ARGMAX = 12, 0.02, 0.95       # Phase C/D's gate, unchanged (A2's)
FLIP_G1, NOREG_G16 = 1.03, 0.99
BW = ("bw_tree", "bw_prmt32", "bw_splitk")
SMS = 170                                                   # the RTX 5090: the plans above are its
SERVED_TOKENS, SERVED_BLOCKS = 4096, 256                    # the server's KV per sequence (16-token blocks)


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def divergence(a: list, b: list) -> dict:
    """Rows identical out of the rows, and each row's first divergent step (None when identical): reported, never ruled."""
    first = []
    for x, y in zip(a, b):
        i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
        first.append(i if i is not None or len(x) == len(y) else min(len(x), len(y)))
    return {"rows_identical": sum(f is None for f in first), "rows": len(first), "first_divergent_step": first}


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


def reduce(arms: dict, off: dict | None, on: dict | None, e4b_sha: str) -> dict:
    out = {"lane": "P143 Amendment 3", "verdict": None, "reasons": [],
           "bars": {"sane_bias": SANE_BIAS, "sane_argmax": SANE_ARGMAX, "sane_windows": SANE_WINDOWS, "flip_g1": FLIP_G1,
                    "noreg_g16": NOREG_G16, "noisy_band": list(NOISY_BAND), "reps": REPS}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    missing += [n for n, r in (("sane_off", off), ("sane_on", on)) if not r or r.get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = []
    for t in TAGS:
        r = arms[t]
        void += record_faults(t, r, e4b_sha, t[:2])
        void += cap_faults(t, r, t[:2], captured=True)
        g = r.get("graph_status") or {}
        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        void += w1_faults(t, r)
        if any(_rate(r, w) is None for w in WORKLOADS):
            void.append(f"{t}: a decode slope is void")
        if r.get("reps") != REPS:
            void.append(f"{t}: {r.get('reps')} timed reps, registered {REPS}")
    if len({arms[t].get("prompts_sha256") for t in TAGS}) != 1:
        void.append("the arms read different prompts")
    if len({(arms[t].get("short"), arms[t].get("long"), arms[t].get("reps")) for t in TAGS}) != 1:
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
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    S0a, S1a = arms["S0a"], arms["S1a"]
    lens = (S0a["short"], S0a["long"])
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    pairs = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in PAIRS} for w in WORKLOADS}
    selfp = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in SELF_PAIRS} for w in WORKLOADS}
    cross = {w: {f"{a}/{b}": round(rates[a][w] / rates[b][w], 4) for a, b in (("S0c", "S0a"), ("S1c", "S1a"))}
             for w in WORKLOADS}
    g1_min, g16_min = min(pairs["W1"].values()), min(pairs["W16"].values())   # the rule reads these, unrounded
    out["speed"] = {"decode_tok_s": rates, "g1": round(g1_min, 4), "g16": round(g16_min, 4),     # rounded: the report's
                    "g1_unrounded": g1_min, "g16_unrounded": g16_min,
                    "pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in pairs.items()},
                    "geomean": {w: round(math.prod(p.values()) ** (1 / len(p)), 4) for w, p in pairs.items()},
                    "self_pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in selfp.items()},
                    "cross_block_self_pairs": cross,
                    "w1_s1_vs_s0_identical": {str(n): _tok(S1a, "W1", n) == _tok(S0a, "W1", n) for n in lens},
                    "w16_s1_vs_s0": {f"{a}/{b} {n}": divergence(_tok(arms[a], "W16", n), _tok(arms[b], "W16", n))
                                     for a, b in PAIRS for n in lens}}
    noisy = [f"W1 self-pair {k} = {v:.4f} outside [{NOISY_BAND[0]}, {NOISY_BAND[1]}]"
             for k, v in selfp["W1"].items() if not NOISY_BAND[0] <= v <= NOISY_BAND[1]]
    if noisy:
        out.update(verdict="NOISY", reasons=noisy + ["NOISY is a VOID and a strike: no flip is read"])
        return out
    fn = []
    for t in TAGS:
        first = "S0a" if t.startswith("S0") else "S1a"
        if t == first:
            continue
        for w in WORKLOADS:
            for n in lens:
                if _tok(arms[t], w, n) != _tok(arms[first], w, n):
                    fn.append(f"{t} != {first} on {w} at {n} tokens")
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
        out.update(verdict="QUALITY_FAIL", reasons=[f"SANE {out['sane']} (gate |bias| <= {SANE_BIAS}, argmax >= {SANE_ARGMAX})"])
    elif g1_min >= FLIP_G1 and g16_min >= NOREG_G16:
        out.update(verdict="FLIP", reasons=[f"g1 {g1_min:.6f} (the min of four pairs) >= {FLIP_G1} and g16 {g16_min:.6f} >= "
                                            f"{NOREG_G16}; SANE bias {sane['bias']:+.5f}, argmax {sane['argmax_agree']:.4f}"])
    else:
        why = [f"g1 {g1_min:.6f} (the min of four pairs) < {FLIP_G1}"] if g1_min < FLIP_G1 else []
        why += [f"g16 {g16_min:.6f} < {NOREG_G16} (W16's no-regression line)"] if g16_min < NOREG_G16 else []
        out.update(verdict="NO_FLIP", reasons=why + ["the cap stays opt-in, permanently for this lever (A3 is binding)"])
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
              toks16=None, sms=SMS, reps=REPS):
    arm = tag[:2]
    d = tally or {"bw_prmt32": 192, "dotpad": 0}
    zero = {k: 0 for k in ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk") + BW}
    w = {}
    for name, rate in (("W16", r16), ("W1", r1)):
        b = 16 if name == "W16" else 1
        tk = (toks16 or toks)[:b] if name == "W16" else toks[:b]
        w[name] = {"batch": b, "decode_tok_s": rate, "tokens": {"32": tk, "160": tk}, "dispatch": dict(zero),
                   "rep_digests": {"32": ["x"] * (reps - 1) + ["x" if rep_ok else "y"], "160": ["z"] * reps}}
    stats = {str(b): {"replays": replays if b == 1 else 44, "eager_steps": 0} for b in BUCKETS}
    return {"arm": arm, "tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": MODEL, "revision": REV,
            "sm_count": sms, "graph_status": graphs or {str(b): "graph" for b in BUCKETS}, "fusions": census or dict(CENSUS),
            "fusion_modes": {k: "auto" for k in FUSION_KNOBS}, "fusion_sources": {k: "default-allowlisted" for k in FUSION_KNOBS},
            "attn_split_cap": cap if cap is not None else _cap_record(arm, captured=True),
            "prompts_sha256": "p", "short": 32, "long": 160, "reps": reps, "workloads": w,
            "graph_stats": stats, "dispatch_build": dict(d)}


def _fake_sane(arm, *, bias=0.0, agree=1.0, n=SANE_WINDOWS, group=1, cap=None, tokens=SERVED_TOKENS):
    key = "R" if arm == "S0" else "ON"
    per = [{"window": i, "nll": 2.0 + 0.01 * i + (bias if arm == "S1" else 0.0), "argmax_agree": agree, "kl": 0.0}
           for i in range(n)]
    blocks = -(-tokens // 16)
    sk = {"passes": n, "asked": [656], "max_tokens_per_seq": [tokens], "blocks_per_seq": [blocks], "split_cap": CAP[arm],
          "plan_clash": [], "plan_b1": 32 if blocks >= 128 else 11}
    return {"mode": "sane", "arm": arm, "status": "ok", "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": MODEL, "revision": REV,
            "sm_count": SMS, "sane_kv": sk, "fusions": dict(CENSUS), "fusion_modes": {k: "auto" for k in FUSION_KNOBS},
            "fusion_sources": {k: "default-allowlisted" for k in FUSION_KNOBS},
            "attn_split_cap": cap if cap is not None else _cap_record(arm, captured=False),
            "group": group, "per_window": {"wikitext": {key: per}}, "dispatch_measure": {"bw_prmt32": 9000, "dotpad": 0}}


# A2's scale (S0 ~250, S1 ~260 tok/s at W1, ~933 / ~962 at W16), exact: every pair 1.04, every self-pair 1.000
BASE = {t: ((933.0, 250.0) if t.startswith("S0") else (962.0, 260.0)) for t in TAGS}


def _cases() -> list:
    toks = [[i, i + 1] for i in range(16)]
    other = [[i, i + 2] for i in range(16)]
    ramp = [[i, i + 7] if i < 4 else [i, i + 1] for i in range(16)]       # rows 0-3 move at step 1 (the 1..4-row ramp)

    def arms(**over):
        out = {}
        for t, (r16, r1) in BASE.items():
            kw = dict(over.get(t, {}))
            out[t] = _fake_arm(t, kw.pop("r16", r16), kw.pop("r1", r1), kw.pop("toks", toks), **kw)
        return out
    off, on = _fake_sane("S0"), _fake_sane("S1")
    v = lambda a, o=off, n=on: reduce(a, o, n, E)["verdict"]  # noqa: E731
    r = lambda a, o=off, n=on: reduce(a, o, n, E)  # noqa: E731
    cases = [
        ("every pair ~1.04 flips", v(arms()) == "FLIP"),
        ("g1 is the min of FOUR pairs: the second block at 1.02 does not flip",
         (lambda x: x["verdict"] == "NO_FLIP" and x["speed"]["g1"] == 1.02)(r(arms(S1c={"r1": 255.0}, S1d={"r1": 255.0})))),
        ("the first block passing is not enough", v(arms(S1c={"r1": 256.0}, S1d={"r1": 256.0})) == "NO_FLIP"),
        ("exactly 1.03 on every pair flips", v(arms(**{t: {"r1": 257.5} for t in TAGS if t.startswith("S1")})) == "FLIP"),
        ("1.0299 on one pair does not", v(arms(S1a={"r1": 257.475}, S1b={"r1": 257.475})) == "NO_FLIP"),
        ("the bars read the UNROUNDED min: a pair of 1.02996 (rounds to 1.03) is NO_FLIP",
         (lambda x: x["verdict"] == "NO_FLIP" and x["speed"]["g1"] == 1.03 and x["speed"]["g1_unrounded"] < 1.03)(
             r(arms(**{t: {"r1": 250.0 * 1.02996} for t in TAGS if t.startswith("S1")})))),
        ("a g16 of 0.98996 (rounds to 0.99) is NO_FLIP",
         (lambda x: x["verdict"] == "NO_FLIP" and x["speed"]["g16"] == 0.99 and any("g16" in y for y in x["reasons"]))(
             r(arms(**{t: {"r16": 933.0 * 0.98996} for t in TAGS if t.startswith("S1")})))),
        ("g16 0.98 does not flip", (lambda x: x["verdict"] == "NO_FLIP" and any("g16" in y for y in x["reasons"]))(
            r(arms(S1c={"r16": 921.2})))),
        ("a W1 self-pair at 0.9845 (A2's S1b/S1a) is NOISY",
         (lambda x: x["verdict"] == "NOISY" and any("S1b/S1a" in y for y in x["reasons"]))(r(arms(S1b={"r1": 255.97})))),
        ("A2's real first block (p143a2-5090-1: S0a 250.49, S1a 260.64, S1b 256.6, S0b 249.2) is NOISY",
         (lambda x: x["verdict"] == "NOISY" and x["speed"]["self_pairs"]["W1"]["S1b/S1a"] == 0.9845)(
             r(arms(S0a={"r1": 250.49}, S1a={"r1": 260.64}, S1b={"r1": 256.6}, S0b={"r1": 249.2})))),
        ("a W1 self-pair at 1.016 is NOISY", v(arms(S0d={"r1": 254.0})) == "NOISY"),
        ("the second block's self-pair counts", v(arms(S1d={"r1": 265.0})) == "NOISY"),
        ("NOISY outranks a passing g1", v(arms(S0b={"r1": 246.0}, S1b={"r1": 260.0})) == "NOISY"),
        ("a W16 self-pair off the band is not NOISY (W1 only)", v(arms(S0b={"r16": 960.0})) == "FLIP"),
        ("the band's edges pass (0.985 and 1.015 exactly)",
         v(arms(S0a={"r1": 200.0}, S0b={"r1": 203.0}, S1a={"r1": 400.0}, S1b={"r1": 394.0})) == "FLIP"),
        ("a ramp-step divergence at W16 stays REPORTING",
         (lambda x: x["verdict"] == "FLIP" and x["speed"]["w16_s1_vs_s0"]["S1c/S0c 32"]["rows_identical"] == 12)(
             r(arms(**{t: {"toks16": ramp} for t in TAGS if t.startswith("S1")})))),
        ("W1 may differ across arms (reported)",
         v(arms(**{t: {"toks": other, "toks16": toks} for t in TAGS if t.startswith("S1")})) == "FLIP"),
        ("an S1 arm differs from S1a: FUNCTION_FAIL", v(arms(S1d={"toks": other})) == "FUNCTION_FAIL"),
        ("an S0 arm differs from S0a: FUNCTION_FAIL", v(arms(S0c={"toks": other})) == "FUNCTION_FAIL"),
        ("timed reps differ", v(arms(S0d={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("SANE bias over the gate", v(arms(), n=_fake_sane("S1", bias=0.03)) == "QUALITY_FAIL"),
        ("SANE argmax under the gate", v(arms(), n=_fake_sane("S1", agree=0.94)) == "QUALITY_FAIL"),
        ("every arm at three reps, not five, is VOID", v(arms(**{t: {"reps": 3} for t in TAGS})) == "VOID"),
        ("one arm at three reps is VOID", v(arms(S0a={"reps": 3})) == "VOID"),
        ("a missing second-block arm is VOID", reduce({k: x for k, x in arms().items() if k != "S0d"}, off, on, E)["verdict"] == "VOID"),
        ("S1 never engaged the cap", v(arms(S1c={"cap": _cap_record("S1", captured=True, plans={})})) == "VOID"),
        ("S0 engaged a cap", v(arms(S0d={"cap": {"cap": 16, "source": "explicit", "plans": {}}})) == "VOID"),
        ("SANE at P115's 656 tokens", v(arms(), n=_fake_sane("S1", tokens=656)) == "VOID"),
        ("S1's SANE uncapped", v(arms(), n=_fake_sane("S1", cap={"cap": 16, "source": "explicit", "plans": {}})) == "VOID"),
        ("a card that is not the 5090", v(arms(S1d={"sms": 188})) == "VOID"),
        ("the GEMV on dot-pad", v(arms(S0c={"tally": {"dotpad": 96}})) == "VOID"),
        ("wrong commit", v(arms(S1c={"e4b": "b" * 40})) == "VOID"),
        ("the geomean is over four pairs", round(r(arms())["speed"]["geomean"]["W1"], 2) == 1.04),
    ]
    return cases


def self_test() -> int:
    try:
        cases = _cases()
    except Exception as e:  # noqa: BLE001 -- a rule that crashes on a registered case fails its self-test
        print(f"p143a3_reduce self-test FAILED [{type(e).__name__}: {e}]")
        return 1
    bad = [n for n, ok in cases if not ok]
    print(f"p143a3_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    v = reduce(arms, load("sane_off.json"), load("sane_on.json"), a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P143A3_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
