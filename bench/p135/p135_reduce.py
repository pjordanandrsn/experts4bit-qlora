# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P135's rule (Track B1's R-b): what Mixtral's ON_auto folds buy at one row (bench/p135/PREREG-p135.md).

FAM Amendment 4's rule (``bench/fam/fam_speed_reduce.py``) with Mixtral's constants. Reads ``p135_speed.py``'s record.
The verdict is the first that applies:

1. **VOID.** Any of:
   - the record not ok;
   - another model, revision or e4b commit;
   - knobs other than the box's (FAM's ON_auto), or a prefill fold knob set;
   - a census other than ``0 / 65 / [32, 32] / 32``; folds other than ``FOLDS`` per fold module; a setting that touched
     other than all of them;
   - the router epilogue not on its fp32 path (``fusion_report``: 32 patched, 32 ``fp32_upstream``; FAM Amendment 5);
   - a block missing, out of its order, on other slots, or with the wrong step counts;
   - a runner that did not capture bucket 1, replayed it other than ``WARM + steps + BUSY`` times, or stepped eagerly;
   - a setting that emitted different tokens in its two blocks (determinism).
2. **NOISY.** Block a's ON/OFF ratio and block b's differ by more than ``NOISE`` (1.5 %).
3. **FASTER** if both ratios are at most ``FAST`` (0.98); **SLOWER** if both are at least ``SLOW`` (1.02); else
   **NO_GAIN**.

``ratio_x`` is median(ON) / median(OFF) of the timed steps within block x. Reported, never gated: the medians, each
block's median per-pair ratio, ON-against-OFF token agreement and the first step they part (the folds change
arithmetic, and FAM's quality read licensed that), the GPU clock log's range, and peak memory.

Usage: ``python p135_reduce.py --rec speed.json --out verdict.json --e4b-sha SHA``.
"""
import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import p135_speed as box  # noqa: E402  (the box's constants)

FAST, SLOW, NOISE = 0.98, 1.02, 0.015
CENSUS = [0, 65, [32, 32], 32]
#: The patched forwards per fold module on Mixtral at ON_auto: every RMSNorm (32 x 2 + the final one), round 2's decoder
#: layer and attention in each of 32 layers, and 32 routers (tests/test_p135_speed.py, a tiny Mixtral at the real depth)
FOLDS = {"glue_fuse": 65, "glue_r2": 64, "router_epilogue": 32}
N_FOLDS = sum(FOLDS.values())
FP32_ROUTERS = 32


def median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def integrity(rec, e4b_sha, steps=box.STEPS):
    """VOID reasons, [] when the record is whole."""
    why = []
    if rec.get("status") != "ok":
        return [f"record status {rec.get('status')!r}"]
    if (rec.get("model"), rec.get("revision")) != (box.MODEL, box.REVISION):
        why.append(f"model {rec.get('model')}@{rec.get('revision')}, registered {box.MODEL}@{box.REVISION}")
    if rec.get("e4b_sha") != e4b_sha:
        why.append(f"e4b {rec.get('e4b_sha')} != {e4b_sha}")
    if rec.get("knobs") != box.KNOBS:
        why.append(f"knobs {rec.get('knobs')} != {box.KNOBS}")
    if any((v or "").strip() not in ("", "0") for v in (rec.get("off_knobs") or {}).values()) \
            or set(rec.get("off_knobs") or {}) != set(box.OFF_KNOBS):
        why.append(f"prefill fold knobs {rec.get('off_knobs')}: the decode-only folds must be read")
    census = [rec.get("census", {}).get(k) for k in box.CENSUS_KEYS]
    if census != CENSUS:
        why.append(f"census {census} != {CENSUS}")
    if rec.get("fused_folds") != N_FOLDS or rec.get("folds_by_module") != FOLDS:
        why.append(f"folds {rec.get('folds_by_module')} ({rec.get('fused_folds')}), registered {FOLDS}")
    epi = (rec.get("fusion_report") or {}).get("E4B_FUSE_ROUTER_EPI") or {}
    if epi.get("patched") != FP32_ROUTERS or epi.get("fp32_upstream") != FP32_ROUTERS:
        why.append(f"router epilogue {epi}: registered {FP32_ROUTERS} patched, all on the fp32 path")
    if [str(b) for b in rec.get("buckets") or []] != ["1"]:
        why.append(f"buckets {rec.get('buckets')}, registered [1]")
    blocks = rec.get("blocks") or {}
    if set(blocks) != set(box.BLOCKS):
        return why + [f"blocks {sorted(blocks)} != {sorted(box.BLOCKS)}"]
    n = box.WARM + steps + box.BUSY
    for b, blk in blocks.items():
        if tuple(blk.get("order") or ()) != box.BLOCKS[b]:
            why.append(f"block {b}: order {blk.get('order')} != {list(box.BLOCKS[b])}")
        if (blk.get("steps"), blk.get("warm"), blk.get("busy")) != (steps, box.WARM, box.BUSY):
            why.append(f"block {b}: steps/warm/busy {(blk.get('steps'), blk.get('warm'), blk.get('busy'))}")
        if blk.get("slots") != box.SLOT:
            why.append(f"block {b}: slots {blk.get('slots')} != {box.SLOT}")
        if blk.get("graphs") is not True:
            why.append(f"block {b}: decoded without graphs")
        arms = blk.get("arms") or {}
        if set(arms) != set(box.SETTINGS):
            why.append(f"block {b}: arms {sorted(arms)}")
            continue
        for st, arm in arms.items():
            tag = f"block {b} {st}"
            if arm.get("folds") != N_FOLDS:
                why.append(f"{tag}: the setting touched {arm.get('folds')} folds, registered {N_FOLDS}")
            gs = arm.get("graph_status") or {}
            if gs != {"1": "graph"}:
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
    for st in box.SETTINGS:
        ta = blocks["a"].get("arms", {}).get(st, {}).get("tokens")
        tb = blocks["b"].get("arms", {}).get(st, {}).get("tokens")
        if ta is not None and tb is not None and ta != tb:
            why.append(f"{st}: blocks a and b emit different tokens from step {_first_diff(ta, tb)} (nondeterministic)")
    return why


def reduce(rec, e4b_sha):
    out = {"constants": {"FAST": FAST, "SLOW": SLOW, "NOISE": NOISE, "steps": box.STEPS}}
    why = integrity(rec, e4b_sha)
    if why:
        return {**out, "verdict": "VOID", "reasons": why}
    blocks = rec["blocks"]
    m = {(b, st): median(blocks[b]["arms"][st]["step_ms"]) for b in box.BLOCKS for st in box.SETTINGS}
    ratio = {b: m[(b, "ON")] / m[(b, "OFF")] for b in box.BLOCKS}
    pair = {b: median([on / off for off, on in zip(blocks[b]["arms"]["OFF"]["step_ms"], blocks[b]["arms"]["ON"]["step_ms"])])
            for b in box.BLOCKS}
    agree = {}
    for b in box.BLOCKS:
        t_off, t_on = blocks[b]["arms"]["OFF"]["tokens"], blocks[b]["arms"]["ON"]["tokens"]
        agree[b] = {"same": sum(x == y for x, y in zip(t_off, t_on)), "of": len(t_off), "first_divergence": _first_diff(t_off, t_on)}
    sm = [float(r[1]) for blk in blocks.values() for r in blk.get("clock") or [] if len(r) > 1 and r[1] not in ("", "[N/A]")]
    report = {"medians": {f"{st}_{b}": round(m[(b, st)], 4) for b in box.BLOCKS for st in box.SETTINGS},
              "ratio_a": round(ratio["a"], 4), "ratio_b": round(ratio["b"], 4),
              "block_disagreement": round(abs(ratio["a"] / ratio["b"] - 1), 4),
              "pair_ratio_median": {b: round(pair[b], 4) for b in box.BLOCKS},
              "saved_ms": {b: round(m[(b, "OFF")] - m[(b, "ON")], 4) for b in box.BLOCKS},
              "on_off_tokens": agree, "sm_clock_mhz": [min(sm), max(sm)] if sm else None,
              "memory": {b: blocks[b].get("memory") for b in box.BLOCKS}}
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

def _synthetic(ms=(20.0, 19.0, 19.0, 20.0), e4b="E", same_tokens=True):
    """A whole record; ``ms`` is the timed step of (OFF_a, ON_a, ON_b, OFF_b)."""
    steps = box.STEPS
    n = box.WARM + steps + box.BUSY
    toks = list(range(n))
    on_toks = toks if same_tokens else toks[:100] + [t + 1 for t in toks[100:]]
    val = {("a", "OFF"): ms[0], ("a", "ON"): ms[1], ("b", "ON"): ms[2], ("b", "OFF"): ms[3]}

    def arm(b, st):
        return {"folds": N_FOLDS, "graph_status": {"1": "graph"},
                "graph_stats": {"1": {"replays": n, "eager_steps": 0, "rows": 0, "pad_rows": 0}},
                "step_ms": [val[(b, st)] + 0.001 * (i % 3) for i in range(steps)],
                "tokens": list(on_toks if st == "ON" else toks)}
    blocks = {b: {"block": b, "order": list(box.BLOCKS[b]), "steps": steps, "warm": box.WARM, "busy": box.BUSY,
                  "graphs": True, "slots": dict(box.SLOT), "arms": {st: arm(b, st) for st in box.SETTINGS},
                  "clock": [[0.0, "2900", "14001", "300", "50", "P2"]], "memory": {"max_allocated_mib": 30000.0}}
              for b in box.BLOCKS}
    return {"status": "ok", "model": box.MODEL, "revision": box.REVISION, "e4b_sha": e4b, "knobs": dict(box.KNOBS),
            "off_knobs": dict.fromkeys(box.OFF_KNOBS), "census": dict(zip(box.CENSUS_KEYS, copy.deepcopy(CENSUS))),
            "fused_folds": N_FOLDS, "folds_by_module": dict(FOLDS),
            "fusion_report": {"E4B_FUSE_ROUTER_EPI": {"patched": 32, "fp32_upstream": 32}},
            "buckets": [1], "blocks": blocks}


def self_test() -> int:
    cases = []

    def case(name, rec, want):
        got = reduce(rec, "E")
        cases.append((name, got["verdict"] == want, got.get("verdict"), got.get("reasons")))
        return got

    g = case("faster", _synthetic(), "FASTER")
    cases.append(("ratios from medians", g["report"]["ratio_a"] == 0.95 and g["report"]["ratio_b"] == 0.95, g["report"]))
    cases.append(("the saving is reported", g["report"]["saved_ms"] == {"a": 1.0, "b": 1.0}, g["report"]["saved_ms"]))
    case("slower", _synthetic((20.0, 20.6, 20.6, 20.0)), "SLOWER")
    case("no gain", _synthetic((20.0, 19.9, 19.9, 20.0)), "NO_GAIN")
    case("one block past the bar is no gain", _synthetic((20.0, 19.5, 19.7, 20.0)), "NO_GAIN")
    case("noisy: the blocks' ratios disagree", _synthetic((20.0, 19.0, 20.0, 20.0)), "NOISY")
    case("a level shift shared by both settings of a block is not noisy", _synthetic((20.0, 19.0, 20.9, 22.0)), "FASTER")
    g = case("ON's tokens parting from OFF's is reported, not VOID", _synthetic(same_tokens=False), "FASTER")
    cases.append(("the first divergence is reported", g["report"]["on_off_tokens"]["a"]["first_divergence"] == 100, None))
    cases.append(("the fold total", N_FOLDS == 161, N_FOLDS))

    def broken(name, fn):
        r = _synthetic()
        fn(r)
        case(name, r, "VOID")
    broken("not ok", lambda r: r.update(status="failed"))
    broken("another revision", lambda r: r.update(revision="x"))
    broken("another model", lambda r: r.update(model="mistralai/Mixtral-8x22B-Instruct-v0.1"))
    broken("another e4b commit", lambda r: r.update(e4b_sha="F"))
    broken("other knobs", lambda r: r["knobs"].update(E4B_FUSE_T1_GLUE="1"))
    broken("the prefill glue on", lambda r: r["off_knobs"].update(E4B_FUSE_PREFILL_GLUE="1"))
    broken("the prefill knobs unrecorded", lambda r: r.update(off_knobs={}))
    broken("the census", lambda r: r["census"].update(fuse_t1_glue_n=64))
    broken("round 2's census", lambda r: r["census"].update(fuse_t1_glue_r2_n=[32, 31]))
    broken("fewer folds", lambda r: r.update(fused_folds=160))
    broken("another fold split", lambda r: r.update(folds_by_module={"glue_fuse": 65, "glue_r2": 63, "router_epilogue": 33}))
    broken("a router off the fp32 path", lambda r: r["fusion_report"]["E4B_FUSE_ROUTER_EPI"].update(fp32_upstream=31))
    broken("no fusion report", lambda r: r.update(fusion_report=None))
    broken("other buckets", lambda r: r.update(buckets=[1, 2]))
    broken("a setting touched fewer folds", lambda r: r["blocks"]["a"]["arms"]["OFF"].update(folds=0))
    broken("a block missing", lambda r: r["blocks"].pop("b"))
    broken("a block out of its order", lambda r: r["blocks"]["b"].update(order=["OFF", "ON"]))
    broken("both settings on one slot", lambda r: r["blocks"]["a"].update(slots={"OFF": 0, "ON": 0}))
    broken("eager decode", lambda r: r["blocks"]["a"].update(graphs=False))
    broken("bucket 1 not captured", lambda r: r["blocks"]["a"]["arms"]["ON"].update(graph_status={"1": "eager: x"}))
    broken("an eager step", lambda r: r["blocks"]["b"]["arms"]["OFF"]["graph_stats"]["1"].update(eager_steps=1))
    broken("another bucket replayed",
           lambda r: r["blocks"]["b"]["arms"]["ON"]["graph_stats"].update({"2": {"replays": 1, "eager_steps": 0}}))
    broken("bucket 1 short", lambda r: r["blocks"]["a"]["arms"]["OFF"]["graph_stats"]["1"].update(replays=10))
    broken("a timed step short", lambda r: r["blocks"]["a"]["arms"]["ON"]["step_ms"].pop())
    broken("a token short", lambda r: r["blocks"]["b"]["arms"]["OFF"]["tokens"].pop())
    broken("nondeterministic: OFF's blocks part", lambda r: r["blocks"]["b"]["arms"]["OFF"]["tokens"].__setitem__(7, -1))
    cases.append(("median of an even count", median([1, 4, 2, 3]) == 2.5 and median([3, 1, 2]) == 2, None))
    bad = [c for c in cases if not c[1]]
    for c in bad:
        print(f"  FAILED case: {c[0]}: {c[2]} {c[3] if len(c) > 3 else ''}")
    print(f"p135_reduce self-test {'OK' if not bad else 'FAILED'} ({len(cases) - len(bad)}/{len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--rec")
    p.add_argument("--out")
    p.add_argument("--e4b-sha")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    rec = json.load(open(a.rec)) if a.rec and os.path.exists(a.rec) else {"status": "missing"}
    v = reduce(rec, a.e4b_sha)
    with open(a.out, "w") as f:
        json.dump(v, f, indent=1)
    print(f"P135_VERDICT {v['verdict']} {json.dumps(v.get('report', {}).get('medians'))} "
          f"ratios {v.get('report', {}).get('ratio_a')} {v.get('report', {}).get('ratio_b')} {v['reasons'][:3]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
