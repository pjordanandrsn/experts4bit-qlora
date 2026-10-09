# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane FAM, Amendment 4's rule: the router epilogue's speed on Qwen3.6 at one row (bench/fam/PREREG-fam.md; e4b#1362).

Reads ``fam_speed.py``'s record. The verdict is the first that applies:

1. **VOID.** Any of:
   - the record not ok;
   - another model, revision or e4b commit;
   - knobs other than the box's;
   - a census other than ``0 / 0 / [0, 0] / 40``, or a setting that touched other than 40 routers;
   - a block missing, out of its order, on other slots, or with the wrong step counts;
   - a runner that did not capture every bucket, replayed bucket 1 other than ``WARM + steps + BUSY`` times, replayed
     any other bucket, or stepped eagerly;
   - a setting that emitted different tokens in its two blocks (determinism).
2. **NOISY.** Block a's ON/OFF ratio and block b's differ by more than ``NOISE`` (1.5 %).
3. **FASTER** if both ratios are at most ``FAST`` (0.98); **SLOWER** if both are at least ``SLOW`` (1.02); else
   **NO_GAIN**.

``ratio_x`` is median(ON) / median(OFF) of the timed steps within block x. Reported, never gated: the medians, each
block's median per-pair ratio, ON-against-OFF token agreement and the first step they part (the epilogue changes
arithmetic, and FAM's quality read licensed that), the GPU clock log's range, and peak memory.

Usage: ``python fam_speed_reduce.py --rec speed_qw36.json --out speed_verdict.json --e4b-sha SHA [--proof]``.
"""
import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fam_reduce  # noqa: E402  (the model pins)
import fam_speed  # noqa: E402  (the box's constants)

FAST, SLOW, NOISE = 0.98, 1.02, 0.015
CENSUS = [0, 0, [0, 0], 40]
ROUTERS = 40


def median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def integrity(rec, e4b_sha, steps):
    """VOID reasons, [] when the record is whole."""
    why = []
    if rec.get("status") != "ok":
        return [f"record status {rec.get('status')!r}"]
    model, rev = fam_reduce.MODELS["qw36"]
    if (rec.get("model"), rec.get("revision")) != (model, rev):
        why.append(f"model {rec.get('model')}@{rec.get('revision')}, registered {model}@{rev}")
    if rec.get("e4b_sha") != e4b_sha:
        why.append(f"e4b {rec.get('e4b_sha')} != {e4b_sha}")
    if rec.get("knobs") != fam_speed.KNOBS:
        why.append(f"knobs {rec.get('knobs')} != {fam_speed.KNOBS}")
    census = [rec.get("census", {}).get(k) for k in fam_speed.CENSUS_KEYS]
    if census != CENSUS:
        why.append(f"census {census} != {CENSUS}")
    if rec.get("fused_routers") != ROUTERS:
        why.append(f"{rec.get('fused_routers')} fused routers, registered {ROUTERS}")
    buckets = [str(b) for b in rec.get("buckets") or []]
    if "1" not in buckets:
        why.append(f"buckets {buckets} lack bucket 1")
    blocks = rec.get("blocks") or {}
    if set(blocks) != set(fam_speed.BLOCKS):
        return why + [f"blocks {sorted(blocks)} != {sorted(fam_speed.BLOCKS)}"]
    n = fam_speed.WARM + steps + fam_speed.BUSY
    for b, blk in blocks.items():
        if tuple(blk.get("order") or ()) != fam_speed.BLOCKS[b]:
            why.append(f"block {b}: order {blk.get('order')} != {list(fam_speed.BLOCKS[b])}")
        if (blk.get("steps"), blk.get("warm"), blk.get("busy")) != (steps, fam_speed.WARM, fam_speed.BUSY):
            why.append(f"block {b}: steps/warm/busy {(blk.get('steps'), blk.get('warm'), blk.get('busy'))}")
        if blk.get("slots") != fam_speed.SLOT:
            why.append(f"block {b}: slots {blk.get('slots')} != {fam_speed.SLOT}")
        if blk.get("graphs") is not True:
            why.append(f"block {b}: decoded without graphs")
        arms = blk.get("arms") or {}
        if set(arms) != set(fam_speed.SETTINGS):
            why.append(f"block {b}: arms {sorted(arms)}")
            continue
        for st, arm in arms.items():
            tag = f"block {b} {st}"
            if arm.get("routers") != ROUTERS:
                why.append(f"{tag}: the setting touched {arm.get('routers')} routers, registered {ROUTERS}")
            gs = arm.get("graph_status") or {}
            if sorted(gs) != sorted(buckets) or any(v != "graph" for v in gs.values()):
                why.append(f"{tag}: graph status {gs}")
            stats = arm.get("graph_stats") or {}
            eager = sum(int(v.get("eager_steps", 0)) for v in stats.values())
            replays = {k: int(v.get("replays", 0)) for k, v in stats.items()}
            if eager:
                why.append(f"{tag}: {eager} eager steps")
            if replays.get("1") != n or any(c for k, c in replays.items() if k != "1"):
                why.append(f"{tag}: replays {replays}, registered bucket 1 x {n}")
            if len(arm.get("step_ms") or []) != steps or any(x <= 0 for x in arm.get("step_ms") or []):
                why.append(f"{tag}: {len(arm.get('step_ms') or [])} timed steps, registered {steps}")
            if len(arm.get("tokens") or []) != n:
                why.append(f"{tag}: {len(arm.get('tokens') or [])} tokens, registered {n}")
    for st in fam_speed.SETTINGS:
        ta = blocks["a"].get("arms", {}).get(st, {}).get("tokens")
        tb = blocks["b"].get("arms", {}).get(st, {}).get("tokens")
        if ta is not None and tb is not None and ta != tb:
            why.append(f"{st}: blocks a and b emit different tokens from step {_first_diff(ta, tb)} (nondeterministic)")
    return why


def reduce(rec, e4b_sha, proof=False):
    steps = fam_speed.PROOF_STEPS if proof else fam_speed.STEPS
    out = {"proof": bool(proof), "constants": {"FAST": FAST, "SLOW": SLOW, "NOISE": NOISE, "steps": steps}}
    why = integrity(rec, e4b_sha, steps)
    if why:
        return {**out, "verdict": "VOID", "reasons": why}
    blocks = rec["blocks"]
    m = {(b, st): median(blocks[b]["arms"][st]["step_ms"]) for b in fam_speed.BLOCKS for st in fam_speed.SETTINGS}
    ratio = {b: m[(b, "ON")] / m[(b, "OFF")] for b in fam_speed.BLOCKS}
    pair = {b: median([on / off for off, on in zip(blocks[b]["arms"]["OFF"]["step_ms"], blocks[b]["arms"]["ON"]["step_ms"])])
            for b in fam_speed.BLOCKS}
    agree = {}
    for b in fam_speed.BLOCKS:
        t_off, t_on = blocks[b]["arms"]["OFF"]["tokens"], blocks[b]["arms"]["ON"]["tokens"]
        agree[b] = {"same": sum(x == y for x, y in zip(t_off, t_on)), "of": len(t_off), "first_divergence": _first_diff(t_off, t_on)}
    sm = [float(r[1]) for blk in blocks.values() for r in blk.get("clock") or [] if len(r) > 1 and r[1] not in ("", "[N/A]")]
    report = {"medians": {f"{st}_{b}": round(m[(b, st)], 4) for b in fam_speed.BLOCKS for st in fam_speed.SETTINGS},
              "ratio_a": round(ratio["a"], 4), "ratio_b": round(ratio["b"], 4),
              "block_disagreement": round(abs(ratio["a"] / ratio["b"] - 1), 4),
              "pair_ratio_median": {b: round(pair[b], 4) for b in fam_speed.BLOCKS},
              "on_off_tokens": agree, "sm_clock_mhz": [min(sm), max(sm)] if sm else None,
              "memory": {b: blocks[b].get("memory") for b in fam_speed.BLOCKS}}
    if abs(ratio["a"] / ratio["b"] - 1) > NOISE:
        verdict = "NOISY"
    elif ratio["a"] <= FAST and ratio["b"] <= FAST:
        verdict = "FASTER"
    elif ratio["a"] >= SLOW and ratio["b"] >= SLOW:
        verdict = "SLOWER"
    else:
        verdict = "NO_GAIN"
    return {**out, "verdict": verdict, "reasons": [], "report": report}


# ------------------------------------------------------------------------------------------------- self-test --

def _synthetic(ms=(10.0, 9.7, 9.7, 10.0), steps=fam_speed.STEPS, e4b="E", same_tokens=True):
    """A whole record; ``ms`` is the timed step of (OFF_a, ON_a, ON_b, OFF_b)."""
    model, rev = fam_reduce.MODELS["qw36"]
    n = fam_speed.WARM + steps + fam_speed.BUSY
    toks = list(range(n))
    on_toks = toks if same_tokens else toks[:100] + [t + 1 for t in toks[100:]]
    buckets = [1, 2, 4, 8, 16]
    val = {("a", "OFF"): ms[0], ("a", "ON"): ms[1], ("b", "ON"): ms[2], ("b", "OFF"): ms[3]}

    def arm(b, st):
        return {"routers": ROUTERS, "graph_status": {str(k): "graph" for k in buckets},
                "graph_stats": {str(k): {"replays": n if k == 1 else 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}
                                for k in buckets},
                "step_ms": [val[(b, st)] + 0.001 * (i % 3) for i in range(steps)],
                "tokens": list(on_toks if st == "ON" else toks)}
    blocks = {b: {"block": b, "order": list(fam_speed.BLOCKS[b]), "steps": steps, "warm": fam_speed.WARM,
                  "busy": fam_speed.BUSY, "graphs": True, "slots": dict(fam_speed.SLOT),
                  "arms": {st: arm(b, st) for st in fam_speed.SETTINGS}, "clock": [[0.0, "2900", "14001", "300", "50", "P2"]],
                  "memory": {"max_allocated_mib": 25000.0}} for b in fam_speed.BLOCKS}
    return {"status": "ok", "model": model, "revision": rev, "e4b_sha": e4b, "knobs": dict(fam_speed.KNOBS),
            "census": dict(zip(fam_speed.CENSUS_KEYS, copy.deepcopy(CENSUS))), "fused_routers": ROUTERS,
            "buckets": buckets, "blocks": blocks}


def self_test() -> int:
    cases = []

    def case(name, rec, want, proof=False):
        got = reduce(rec, "E", proof)
        cases.append((name, got["verdict"] == want, got.get("verdict"), got.get("reasons")))
        return got

    g = case("faster", _synthetic(), "FASTER")
    cases.append(("ratios from medians", g["report"]["ratio_a"] == 0.97 and g["report"]["ratio_b"] == 0.97, g["report"]))
    case("slower", _synthetic((10.0, 10.3, 10.3, 10.0)), "SLOWER")
    case("no gain", _synthetic((10.0, 9.95, 9.95, 10.0)), "NO_GAIN")
    case("one block past the bar is no gain", _synthetic((10.0, 9.75, 9.85, 10.0)), "NO_GAIN")
    case("noisy: the blocks' ratios disagree", _synthetic((10.0, 9.7, 10.0, 10.0)), "NOISY")
    case("a level shift shared by both settings of a block is not noisy", _synthetic((10.0, 9.7, 10.67, 11.0)), "FASTER")
    g = case("ON's tokens parting from OFF's is reported, not VOID", _synthetic(same_tokens=False), "FASTER")
    cases.append(("the first divergence is reported", g["report"]["on_off_tokens"]["a"]["first_divergence"] == 100, None))
    case("the proof's step count", _synthetic(steps=fam_speed.PROOF_STEPS), "FASTER", proof=True)
    case("a reading at the proof's step count VOIDs", _synthetic(steps=fam_speed.PROOF_STEPS), "VOID")

    def broken(name, fn):
        r = _synthetic()
        fn(r)
        case(name, r, "VOID")
    broken("not ok", lambda r: r.update(status="failed"))
    broken("another revision", lambda r: r.update(revision="x"))
    broken("another e4b commit", lambda r: r.update(e4b_sha="F"))
    broken("other knobs", lambda r: r["knobs"].update(E4B_FUSE_T1_GLUE="auto"))
    broken("the census", lambda r: r["census"].update(fuse_router_epilogue_n=39))
    broken("fewer fused routers", lambda r: r.update(fused_routers=39))
    broken("a setting touched fewer routers", lambda r: r["blocks"]["a"]["arms"]["OFF"].update(routers=0))
    broken("a block missing", lambda r: r["blocks"].pop("b"))
    broken("a block out of its order", lambda r: r["blocks"]["b"].update(order=["OFF", "ON"]))
    broken("both settings on one slot", lambda r: r["blocks"]["a"].update(slots={"OFF": 0, "ON": 0}))
    broken("eager decode", lambda r: r["blocks"]["a"].update(graphs=False))
    broken("a bucket not captured", lambda r: r["blocks"]["a"]["arms"]["ON"]["graph_status"].update({"16": "eager: x"}))
    broken("an eager step", lambda r: r["blocks"]["b"]["arms"]["OFF"]["graph_stats"]["1"].update(eager_steps=1))
    broken("another bucket replayed", lambda r: r["blocks"]["b"]["arms"]["ON"]["graph_stats"]["2"].update(replays=1))
    broken("bucket 1 short", lambda r: r["blocks"]["a"]["arms"]["OFF"]["graph_stats"]["1"].update(replays=10))
    broken("a timed step short", lambda r: r["blocks"]["a"]["arms"]["ON"]["step_ms"].pop())
    broken("a token short", lambda r: r["blocks"]["b"]["arms"]["OFF"]["tokens"].pop())
    broken("nondeterministic: OFF's blocks part", lambda r: r["blocks"]["b"]["arms"]["OFF"]["tokens"].__setitem__(7, -1))
    cases.append(("median of an even count", median([1, 4, 2, 3]) == 2.5 and median([3, 1, 2]) == 2, None))
    bad = [c for c in cases if not c[1]]
    for c in bad:
        print(f"  FAILED case: {c[0]}: {c[2]} {c[3] if len(c) > 3 else ''}")
    print(f"fam_speed_reduce self-test {'OK' if not bad else 'FAILED'} ({len(cases) - len(bad)}/{len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--rec")
    p.add_argument("--out")
    p.add_argument("--e4b-sha")
    p.add_argument("--proof", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    rec = json.load(open(a.rec)) if a.rec and os.path.exists(a.rec) else {"status": "missing"}
    v = reduce(rec, a.e4b_sha, a.proof)
    with open(a.out, "w") as f:
        json.dump(v, f, indent=1)
    print(f"FAM_SPEED_VERDICT {v['verdict']} {json.dumps(v.get('report', {}).get('medians'))} "
          f"ratios {v.get('report', {}).get('ratio_a')} {v.get('report', {}).get('ratio_b')} {v['reasons'][:3]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
