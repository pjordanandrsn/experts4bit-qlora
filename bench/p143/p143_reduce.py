#!/usr/bin/env python3
"""p143_reduce.py -- lane P143 step 1's rule, fixed before data (bench/p143/PREREG-p143.md; e4b#1313).

Times are the bench's per-replay medians for 48 launches (one per Qwen3 layer), averaged over the two passes. BAR is 3 %
of P142's measured node period P = 4.380166 ms.

VOID          a missing arm; the self-pair of the library's plan outside [0.97, 1.03] at any context (pass 2 / pass 1);
              the library's plan more than 3 % from the 32-split arm at any context (the rule assumes it IS 32 at B=1);
              the library's plan at 640 more than 1.5x or less than 0.5x P142's served launch (0.5558 ms / 48).
GO_FIXED      some fixed split s in {4, 8, 16} saves >= BAR at 640 against the library's plan AND loses no more than the
              noise at 4096 (its time minus the plan's, against the larger of the two arms' |pass 1 - pass 2|).
GO_ADAPTIVE   otherwise, the best split PER CONTEXT saves >= BAR at 640: an upper bound for a fixed-grid plan that picks
              its live split count on the device; step 2 must build it and read it.
PARK          otherwise.

    python p143_reduce.py --bench p143.json --out verdict_p143.json --e4b-sha <sha>
    python p143_reduce.py --self-test
"""
import argparse
import json
import sys

CTX = (256, 640, 1024, 2048, 4096)
SPLITS = (None, 4, 8, 16, 32)
FIXED = (4, 8, 16)
P142_P_MS = 4.380166
BAR = round(0.03 * P142_P_MS, 6)                     # 0.131405 ms per step
P142_ATTN_MS = 0.555808                              # _fp8_paged_decode_split_f8dot, 48 launches, P142 removable
SELF_PAIR = (0.97, 1.03)
ANCHOR = (0.5, 1.5)
PLAN_IS_32 = 0.03


def _key(c, s):
    return f"{c}/{s}"


def judge(rec) -> dict:
    faults = []
    passes = (rec or {}).get("passes") or {}
    t, noise = {}, {}
    for c in CTX:
        for s in SPLITS:
            k = _key(c, s)
            try:
                a, b = passes["1"][k]["median_ms"], passes["2"][k]["median_ms"]
            except (KeyError, TypeError):
                faults.append(f"arm {k} missing")
                continue
            t[(c, s)], noise[(c, s)] = (a + b) / 2, abs(a - b)
            if s is None and not SELF_PAIR[0] <= b / a <= SELF_PAIR[1]:
                faults.append(f"the library plan's self-pair at {c} is {b / a:.4f}, outside {SELF_PAIR}")
    if not faults:
        for c in CTX:
            r = t[(c, None)] / t[(c, 32)]
            if abs(r - 1) > PLAN_IS_32:
                faults.append(f"the library plan at {c} is {r:.4f}x the 32-split arm: not the plan this rule assumes")
        anchor = t[(640, None)] / P142_ATTN_MS
        if not ANCHOR[0] <= anchor <= ANCHOR[1]:
            faults.append(f"the library plan at 640 is {anchor:.3f}x P142's served launches, outside {ANCHOR}")
    out = {"verdict": "VOID" if faults else None, "why": faults, "bar_ms": BAR}
    if faults:
        return out
    fixed = []
    for s in FIXED:
        save = t[(640, None)] - t[(640, s)]
        loss = t[(4096, s)] - t[(4096, None)]
        tol = max(noise[(4096, None)], noise[(4096, s)])
        fixed.append({"n_split": s, "save_640_ms": round(save, 6), "loss_4096_ms": round(loss, 6), "noise_4096_ms": round(tol, 6),
                      "clears": save >= BAR and loss <= tol})
    best_per_ctx = {c: min(SPLITS, key=lambda s: t[(c, s)] if s is not None else t[(c, None)] + 1e-12) for c in CTX}
    oracle_save = t[(640, None)] - t[(640, best_per_ctx[640])]
    passing = [f for f in fixed if f["clears"]]
    if passing:
        top = max(passing, key=lambda f: f["save_640_ms"])
        out["verdict"], out["plan"] = "GO_FIXED", top
    elif oracle_save >= BAR:
        out["verdict"], out["plan"] = "GO_ADAPTIVE", {"best_per_ctx": {str(c): best_per_ctx[c] for c in CTX},
                                                      "save_640_ms": round(oracle_save, 6)}
    else:
        out["verdict"] = "PARK"
    out["fixed"] = fixed
    out["oracle"] = {"best_per_ctx": {str(c): best_per_ctx[c] for c in CTX}, "save_640_ms": round(oracle_save, 6),
                     "save_640_share_of_P142": round(oracle_save / P142_P_MS, 5)}
    out["grid_ms_per_step"] = {_key(c, s): round(t[(c, s)], 6) for c in CTX for s in SPLITS}
    out["us_per_launch"] = {_key(c, s): round(t[(c, s)] * 1000 / 48, 3) for c in CTX for s in SPLITS}
    out["anchor_640_vs_P142"] = round(t[(640, None)] / P142_ATTN_MS, 4)
    return out


# ---------------------------------------------------------------------------------------------------------- self-test --
def _fake(f, noise=0.0, plan_scale=1.0):
    """f(ctx, split) -> ms per 48 launches; pass 2 = pass 1 + noise. The library plan is the 32 arm x plan_scale."""
    p1, p2 = {}, {}
    for c in CTX:
        for s in SPLITS:
            v = f(c, 32) * plan_scale if s is None else f(c, s)
            p1[_key(c, s)] = {"median_ms": v}
            p2[_key(c, s)] = {"median_ms": v + noise}
    return {"passes": {"1": p1, "2": p2}}


def _base(c, s):
    """A 32-split launch near P142's 11.6 us at 640; fewer splits faster at short context, slower at long."""
    work = c / s * 0.004            # ms per 48 launches: serial work per split
    return 0.50 + work + s * 0.001


def self_test() -> int:
    cases = []

    def case(name, cond):
        cases.append((name, bool(cond)))
    r = judge(_fake(_base))
    case("a fixed split that loses at 4096 is not GO_FIXED", r["verdict"] != "GO_FIXED")
    flat = judge(_fake(lambda c, s: 0.55 if s == 32 else 0.40 if c <= 1024 else 0.55))
    case("a split saving 0.15 ms at 640 and equal at 4096 is GO_FIXED", flat["verdict"] == "GO_FIXED"
         and flat["plan"]["save_640_ms"] >= BAR)
    tiny = judge(_fake(lambda c, s: 0.55 if s == 32 else 0.50 if c <= 1024 else 0.60))
    case("a 0.05 ms saving is under the bar: PARK", tiny["verdict"] == "PARK")
    adapt = judge(_fake(lambda c, s: 0.55 if s == 32 else (0.38 if c <= 1024 else 0.70)))
    case("only per-context choice clears: GO_ADAPTIVE", adapt["verdict"] == "GO_ADAPTIVE"
         and adapt["oracle"]["best_per_ctx"]["4096"] in (None, 32))
    noisy = judge(_fake(lambda c, s: 0.55, noise=0.03))
    case("a self-pair outside [0.97, 1.03] is VOID", noisy["verdict"] == "VOID")
    off = judge(_fake(lambda c, s: 0.55, plan_scale=1.10))
    case("a library plan that is not the 32-split arm is VOID", off["verdict"] == "VOID")
    far = judge(_fake(lambda c, s: 1.20))
    case("a launch far from P142's served one is VOID (the bench does not transfer)", far["verdict"] == "VOID")
    edge = judge(_fake(lambda c, s: 0.55 if s == 32 else 0.55 - BAR if c == 640 else 0.55))
    case("a saving exactly at the bar clears", edge["verdict"] == "GO_FIXED")
    case("a missing arm is VOID", judge({"passes": {"1": {}, "2": {}}})["verdict"] == "VOID")
    bad = [n for n, ok in cases if not ok]
    print(f"p143_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    try:
        rec = json.load(open(a.bench))
    except (OSError, ValueError):
        rec = None
    v = judge(rec)
    v.update({"lane": "p143", "e4b_sha": a.e4b_sha, "gnf4_sha": "70bc03cb7a61b03c4e801e7423c58c103a1b6d7d"})
    json.dump(v, open(a.out, "w"), indent=1)
    print("P143_VERDICT " + json.dumps({k: v.get(k) for k in ("verdict", "why", "plan")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
