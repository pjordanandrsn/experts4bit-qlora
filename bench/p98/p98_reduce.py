#!/usr/bin/env python3
"""Lane P98's reducer (bench/p98/PREREG-p98.md; e4b#564): the registered rule over the three arm records written by
``p98_box.py`` (``arm_g.json`` graphs, ``arm_e.json`` the padded-eager oracle, ``arm_p.json`` plain eager).

VOID when any of these holds:
- a record is missing, or ran a rehearsal knob (stand-in attention; placement other than all-vram);
- a record is not the registered shape (16 requests, 256-token prompts, 48 + 4i new tokens, W1 96, warm 3, buckets
  1,2,4,8,16) or loaded another model or commit, or a request produced other than its token budget;
- a layout is not Qwen3.6's on a compact pool (10 pool layers, 10 attention, 30 linear, the linear state allocated;
  frozen in the graph arms);
- the oracle is not one: arm e captured a bucket or replayed a graph; arm p ran graphs;
- the W16 trace never stepped a bucket (every bucket must be exercised).

NOT_SUPPORTED when, in arm g, a bucket did not capture, a step ran eagerly, or the tokens (W16 or W1) differ from arm
e's. SUPPORTED otherwise. Decode throughput (W16 and W1, every arm), the graph / plain-eager ratios and arm g's token
agreement with plain eager are reported, not gated.

    python p98_reduce.py --dir RUN_DIR --out verdict.json
    python p98_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

SUBJECT = {"model": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
           "pool": 10, "attn": 10, "linear": 30}
SHAPE = {"n": 16, "prompt": 256, "new_base": 48, "new_step": 4, "b1_new": 96, "warm_steps": 3,
         "buckets": [1, 2, 4, 8, 16]}


def _bucket_for(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(n)


def _max_new(shape=SHAPE):
    return [shape["new_base"] + shape["new_step"] * i for i in range(shape["n"])]


def _check(rec, arm, shape=SHAPE):
    why = []
    rh = rec.get("rehearsal", {})
    if rh.get("stand_in_attention") is not False or rh.get("placement") != "all-vram":
        why.append(f"arm {arm}: rehearsal knobs {rh}")
    for k in ("n", "prompt", "b1_new", "warm_steps", "buckets"):
        if rec.get(k) != shape[k]:
            why.append(f"arm {arm}: {k} {rec.get(k)!r} != registered {shape[k]!r}")
    if rec.get("max_new") != _max_new(shape):
        why.append(f"arm {arm}: max_new is not 48 + 4i")
    if rec.get("model") != SUBJECT["model"] or rec.get("revision") != SUBJECT["revision"] \
            or rec.get("loaded_commit") != SUBJECT["revision"]:
        why.append(f"arm {arm}: {rec.get('model')}@{rec.get('revision')} loaded {rec.get('loaded_commit')}")
    lay = rec.get("layout", {})
    if (lay.get("kv_pool_layers"), len(lay.get("attn_layers", [])), len(lay.get("linear_layers", []))) != \
            (SUBJECT["pool"], SUBJECT["attn"], SUBJECT["linear"]) or not lay.get("linear_state_allocated"):
        why.append(f"arm {arm}: layout {lay}")
    if arm in ("g", "e") and not lay.get("linear_state_frozen"):
        why.append(f"arm {arm}: the linear-state pool was not frozen for capture")
    w16 = rec.get("w16", {}).get("tokens") or []
    w1 = rec.get("w1", {}).get("tokens") or []
    if [len(t) for t in w16] != _max_new(shape) or [len(t) for t in w1] != [shape["b1_new"]]:
        why.append(f"arm {arm}: token counts {[len(t) for t in w16]} / {[len(t) for t in w1]}")
    return why


def reduce(recs, shape=SHAPE):
    why = [f"arm {a} record missing" for a in ("g", "e", "p") if recs.get(a) is None]
    if why:
        return {"verdict": "VOID", "why": why}
    g, e, p = recs["g"], recs["e"], recs["p"]
    for arm, rec in (("g", g), ("e", e), ("p", p)):
        why += _check(rec, arm, shape)
    est = e.get("engine", {}).get("graph_status") or {}
    if sorted(est) != sorted(str(b) for b in shape["buckets"]) or any(v != "eager: capture=False" for v in est.values()) \
            or any(s.get("replays", 0) for s in (e.get("graph_stats") or {}).values()):
        why.append(f"arm e is not the padded-eager oracle: status {est}")
    if p.get("graph_stats") is not None or p.get("engine", {}).get("graph_status") is not None:
        why.append("arm p ran decode graphs")
    hit = {_bucket_for(r, shape["buckets"]) for r in g.get("w16", {}).get("rows_per_call", [])}
    missing = [b for b in shape["buckets"] if b not in hit]
    if missing:
        why.append(f"the W16 trace never stepped bucket(s) {missing}")
    fail = []
    gst = g.get("engine", {}).get("graph_status") or {}
    for b in shape["buckets"]:
        s = gst.get(str(b))
        if s != "graph":
            fail.append(f"bucket {b} did not capture: {s}")
    stats = g.get("graph_stats") or {}
    if sum(s.get("eager_steps", 0) for s in stats.values()):
        fail.append(f"arm g ran {sum(s.get('eager_steps', 0) for s in stats.values())} eager steps")
    if any(stats.get(str(b), {}).get("replays", 0) == 0 for b in shape["buckets"]) and not missing and not fail:
        fail.append(f"a captured bucket never replayed: {stats}")
    for w in ("w16", "w1"):
        if g.get(w, {}).get("tokens") != e.get(w, {}).get("tokens"):
            fail.append(f"{w}: graph tokens differ from the padded eager step")

    def tps(rec, w):
        return (rec.get(w, {}).get("decode") or {}).get("tok_per_s")

    def agree(a, b):
        pairs = [(x, y) for ta, tb in zip(a, b) for x, y in zip(ta, tb)]
        return sum(x == y for x, y in pairs) / len(pairs) if pairs else None

    out = {"why": why, "not_supported": fail,
           "decode_tok_per_s": {w: {arm: tps(r, w) for arm, r in (("g", g), ("e", e), ("p", p))} for w in ("w16", "w1")},
           "graph_over_plain": {w: (tps(g, w) / tps(p, w) if tps(g, w) and tps(p, w) else None) for w in ("w16", "w1")},
           "graph_vs_plain_token_agreement": {w: agree(g.get(w, {}).get("tokens") or [], p.get(w, {}).get("tokens") or [])
                                              for w in ("w16", "w1")},
           "graph_status": gst, "graph_stats": stats,
           "ms_per_step_by_rows": {arm: (r.get("w16", {}).get("decode") or {}).get("ms_per_step_by_rows")
                                   for arm, r in (("g", g), ("p", p))}}
    out["verdict"] = "VOID" if why else ("NOT_SUPPORTED" if fail else "SUPPORTED")
    return out


def _fixture():
    mn = _max_new()
    rows = list(range(16, 0, -1))
    w16 = [[7] * m for m in mn]
    w1 = [[3] * SHAPE["b1_new"]]

    def rec(arm, status, stats):
        return {"arm": arm, "model": SUBJECT["model"], "revision": SUBJECT["revision"],
                "loaded_commit": SUBJECT["revision"], "n": 16, "prompt": 256, "max_new": mn, "b1_new": 96,
                "warm_steps": 3, "buckets": [1, 2, 4, 8, 16],
                "rehearsal": {"stand_in_attention": False, "placement": "all-vram"},
                "engine": {"graph_status": status}, "graph_stats": stats,
                "layout": {"kv_pool_layers": 10, "attn_layers": list(range(3, 40, 4)),
                           "linear_layers": [i for i in range(40) if i % 4 != 3], "linear_state": True,
                           "linear_state_frozen": arm != "p", "linear_state_allocated": True},
                "w16": {"tokens": copy.deepcopy(w16), "rows_per_call": rows, "decode": {"tok_per_s": 100.0}},
                "w1": {"tokens": copy.deepcopy(w1), "decode": {"tok_per_s": 20.0}}}
    bs = [1, 2, 4, 8, 16]
    g = rec("g", {str(b): "graph" for b in bs}, {str(b): {"replays": 5, "eager_steps": 0} for b in bs})
    e = rec("e", {str(b): "eager: capture=False" for b in bs}, {str(b): {"replays": 0, "eager_steps": 5} for b in bs})
    p = rec("p", None, None)
    p["w16"]["decode"]["tok_per_s"], p["w1"]["decode"]["tok_per_s"] = 40.0, 10.0
    return {"g": g, "e": e, "p": p}


def self_test():
    base = _fixture()
    cases = []

    def case(name, want, edit=None):
        recs = copy.deepcopy(base)
        if edit:
            edit(recs)
        got = reduce(recs)
        cases.append((name, want, got["verdict"], got.get("why"), got.get("not_supported")))

    case("registered fixture", "SUPPORTED")
    case("plain eager disagrees with the graph (reported only)", "SUPPORTED",
         lambda r: r["p"]["w16"]["tokens"][0].__setitem__(0, 9))
    case("a bucket did not capture", "NOT_SUPPORTED",
         lambda r: r["g"]["engine"]["graph_status"].__setitem__("8", "eager: RuntimeError: x"))
    case("arm g ran an eager step", "NOT_SUPPORTED", lambda r: r["g"]["graph_stats"]["4"].__setitem__("eager_steps", 1))
    case("W16 graph tokens differ", "NOT_SUPPORTED", lambda r: r["g"]["w16"]["tokens"][3].__setitem__(5, 8))
    case("W1 graph tokens differ", "NOT_SUPPORTED", lambda r: r["g"]["w1"]["tokens"][0].__setitem__(0, 8))
    case("a captured bucket never replayed", "NOT_SUPPORTED",
         lambda r: r["g"]["graph_stats"]["2"].__setitem__("replays", 0))
    case("arm g missing", "VOID", lambda r: r.__setitem__("g", None))
    case("stand-in attention", "VOID", lambda r: r["p"]["rehearsal"].__setitem__("stand_in_attention", True))
    case("solver placement", "VOID", lambda r: r["g"]["rehearsal"].__setitem__("placement", "solver"))
    case("other commit loaded", "VOID", lambda r: r["e"].__setitem__("loaded_commit", "0" * 40))
    case("short prompts", "VOID", lambda r: r["g"].__setitem__("prompt", 64))
    case("a request short of its budget", "VOID", lambda r: r["p"]["w16"]["tokens"][0].pop())
    case("full-size pool", "VOID", lambda r: r["g"]["layout"].__setitem__("kv_pool_layers", 40))
    case("pool not frozen in the graph arm", "VOID", lambda r: r["g"]["layout"].__setitem__("linear_state_frozen", False))
    case("oracle captured a bucket", "VOID", lambda r: r["e"]["engine"]["graph_status"].__setitem__("1", "graph"))
    case("oracle replayed", "VOID", lambda r: r["e"]["graph_stats"]["1"].__setitem__("replays", 2))
    case("plain arm ran graphs", "VOID", lambda r: r["p"].__setitem__("graph_stats", {"1": {"replays": 1}}))
    case("trace never stepped bucket 2", "VOID",
         lambda r: r["g"]["w16"].__setitem__("rows_per_call", [16, 15, 9, 8, 5, 4, 3, 1]))
    bad = [c for c in cases if c[1] != c[2]]
    for n, w, g_, why, ns in bad:
        print(f"SELF-TEST FAIL {n}: want {w}, got {g_} ({why}; {ns})")
    if bad:
        return 1
    print(f"self-test OK ({len(cases)} cases)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    d = Path(a.dir)
    recs = {arm: (json.loads((d / f"arm_{arm}.json").read_text()) if (d / f"arm_{arm}.json").is_file() else None)
            for arm in ("g", "e", "p")}
    v = reduce(recs)
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P98 VERDICT {v['verdict']}")
    for w in v.get("why", []):
        print(f"  void: {w}")
    for w in v.get("not_supported", []):
        print(f"  not supported: {w}")
    if "decode_tok_per_s" in v:
        print(f"  decode tok/s {v['decode_tok_per_s']} | graph/plain {v['graph_over_plain']} | graph-vs-plain token "
              f"agreement {v['graph_vs_plain_token_agreement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
