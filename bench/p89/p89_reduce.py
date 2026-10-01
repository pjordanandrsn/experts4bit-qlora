#!/usr/bin/env python3
"""Lane P89's reducer (bench/p89/PREREG-p89.md; e4b#564). Is K23's lean glue (E4B_INT4_LEAN_GLUE=1) faster on an RTX
5090 at B=16 with the same output bits?

Reads, from the run directory:
- premise: premise.json, the on-box run of tests/test_k19_row_exact_gpu.py + tests/test_k23_lean_glue_gpu.py (K19's
           rows row-count invariant; the lean route bit-equal to the default at B=16, eager, captured, from token rows)
- speed:   e4b_b16_{off,on}{,_r2}.json (`step_ms_clean`, the graph-replay window, and `tokens`, each row's prefill +
           warm + timed greedy tokens), and the first draws' replay censuses logs/census_b16_{off,on}.txt (P42's table,
           parsed by bench/p42/p42_reduce.py's `parse_census`)

  VOID           the premise did not hold, did not run, or skipped a test; a timed draw is missing or failed; a census
                 is missing; two draws of one arm differ by more than 3 %; K19 is not at its 96 calls per step in both
                 arms or the tile builder not at its 48; ON did not engage (the expansion's index_select is not gone --
                 at least 48 fewer index_select calls per step -- or ON does not launch at least 192 fewer kernels per
                 step, 4 per layer, than OFF)
  IDENTITY_FAIL  any row's tokens differ between OFF and ON in either draw pair: the route moved an output, so it stays
                 opt-in whatever its speed (it is bit-identical by construction; this is the in-model check of that)
  LICENSED       identical tokens and B=16 is FASTER (median ON / median OFF <= 0.97)
  NOT_FASTER     identical tokens and B=16 is not faster
Reported beside the verdict: kernel launches per step in each arm, and the ON-minus-OFF change per kernel name.

    python p89_reduce.py --dir <dir> --out verdict.json
    python p89_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

B16_FASTER = 0.97
DRAW_SPREAD = 0.03
EXPERT_CALLS = 96                   # per decode step: 48 layers x (gate_up + down)
LAYERS = 48
K19_KERNEL = "_gemm_int4_b32_grouped_smallm"
TILE_KERNEL = "_tile_table_r1"
SELECT = "indexSelect"              # P88's census names it 'void at::native::(anonymous namespace)::indexSelectS...'
MIN_SELECTS_GONE = LAYERS           # the (token, slot) expansion: one index_select per layer
MIN_LAUNCHES_GONE = 4 * LAYERS      # the expansion + at least three of the glue launches, per layer
PREMISE_PASSED = "6 passed"         # 3 + 3 tests in the two staged (sha-pinned) files


def _p42():
    here = Path(__file__).resolve().parent
    for cand in (here / "p42_reduce.py", here.parent / "p42" / "p42_reduce.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("p42_reduce", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("p42_reduce.py not found beside this reducer or in bench/p42")


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _draw(d: Path, name: str):
    r = _json(d / name)
    try:
        return {"ms": float(r["step_ms_clean"]), "tokens": r["tokens"]}
    except (KeyError, TypeError, ValueError):
        return None


def _census(text: str, p42) -> dict:
    replays, rows = p42.parse_census(text)
    rep = max(replays or 1, 1)
    by = {}
    for r in rows:
        n = r["name"].strip()
        by[n] = by.get(n, 0) + r["calls"]
    return {"replays": replays, "per_step": {n: c / rep for n, c in by.items()}}


def load(d: Path) -> dict:
    p42 = _p42()
    data = {"premise": _json(d / "premise.json"), "speed": {}, "census": {}}
    for tag in ("off", "on"):
        data["speed"][tag] = [_draw(d, f"e4b_b16_{tag}.json"), _draw(d, f"e4b_b16_{tag}_r2.json")]
        c = d / "logs" / f"census_b16_{tag}.txt"
        if c.is_file():
            data["census"][tag] = _census(c.read_text(), p42)
    return data


def _void(v, why):
    v["verdict"] = "VOID"
    v["reasons"].append(why)
    return v


def _calls(c, pred):
    return sum(n for name, n in c["per_step"].items() if pred(name))


def reduce(data: dict) -> dict:
    v: dict = {"lane": "P89", "verdict": None, "reasons": []}
    pr = data.get("premise")
    last = " ".join(pr.get("last") or []) if isinstance(pr, dict) else ""
    if not isinstance(pr, dict) or pr.get("rc") != 0 or PREMISE_PASSED not in last or "skipped" in last:
        return _void(v, f"the on-box premise (K19 row-exact + K23 bit-equal, {PREMISE_PASSED}) did not hold, did not "
                        f"run, or skipped a test: {pr}")
    for tag in ("off", "on"):
        draws = data["speed"].get(tag)
        if not draws or None in draws:
            return _void(v, f"B=16 {tag}: a timed draw is missing or failed")
        ms = [x["ms"] for x in draws]
        if (max(ms) - min(ms)) / min(ms) > DRAW_SPREAD:
            return _void(v, f"B=16 {tag}: draws {ms} differ by more than {DRAW_SPREAD:.0%}")
        c = data["census"].get(tag)
        if not c or not c.get("replays"):
            return _void(v, f"B=16 {tag}: the first draw's census is missing")
    off, on = data["census"]["off"], data["census"]["on"]
    for tag, c in (("off", off), ("on", on)):
        k19 = _calls(c, lambda n: n == K19_KERNEL)
        tiles = _calls(c, lambda n: n == TILE_KERNEL)
        if abs(k19 - EXPERT_CALLS) > 0.5 or abs(tiles - LAYERS) > 0.5:
            return _void(v, f"{tag}: K19 ran {k19:g} and the tile builder {tiles:g} times per step, not "
                            f"{EXPERT_CALLS} and {LAYERS} (K19's licensed route is not the one measured)")
    sel_gone = _calls(off, lambda n: SELECT in n) - _calls(on, lambda n: SELECT in n)
    launches = {"off": sum(off["per_step"].values()), "on": sum(on["per_step"].values())}
    v["launches_per_step"] = launches
    v["index_select_gone_per_step"] = sel_gone
    names = set(off["per_step"]) | set(on["per_step"])
    v["per_kernel_delta"] = {n: on["per_step"].get(n, 0) - off["per_step"].get(n, 0) for n in sorted(names)
                             if abs(on["per_step"].get(n, 0) - off["per_step"].get(n, 0)) > 0.5}
    if sel_gone < MIN_SELECTS_GONE - 0.5:
        return _void(v, f"ON removed {sel_gone:g} index_select launches per step, not >= {MIN_SELECTS_GONE} (the "
                        f"token-row expansion was still made: the lean route did not engage)")
    if launches["off"] - launches["on"] < MIN_LAUNCHES_GONE - 0.5:
        return _void(v, f"ON launched {launches['on']:g} kernels per step vs OFF {launches['off']:g}: fewer than "
                        f"{MIN_LAUNCHES_GONE} gone (the glue was not folded)")
    for i in range(2):
        a, b = data["speed"]["off"][i]["tokens"], data["speed"]["on"][i]["tokens"]
        if not isinstance(a, dict) or not a or a != b:
            rows = sorted(set(a or {}) | set(b or {}))
            bad = [r for r in rows if (a or {}).get(r) != (b or {}).get(r)]
            v["verdict"] = "IDENTITY_FAIL"
            v["reasons"].append(f"draw {i + 1}: tokens differ between OFF and ON in rows {bad[:8]} -- the lean route "
                                f"moved an output; it stays opt-in")
            return v
    m_off = statistics.median(x["ms"] for x in data["speed"]["off"])
    m_on = statistics.median(x["ms"] for x in data["speed"]["on"])
    v["speed"] = {"off_ms": m_off, "on_ms": m_on, "ratio": m_on / m_off}
    r = m_on / m_off
    if r <= B16_FASTER:
        v["verdict"] = "LICENSED"
        v["reasons"].append(f"B=16 {m_on:.3f} vs {m_off:.3f} ms (x{r:.3f}); tokens identical in both draw pairs; "
                            f"{launches['off'] - launches['on']:g} fewer launches per step")
    else:
        v["verdict"] = "NOT_FASTER"
        v["reasons"].append(f"B=16 ratio {r:.3f} > {B16_FASTER}; tokens identical in both draw pairs")
    return v


def _synthetic(off=10.14, on=9.45, premise=0, last="6 passed in 61.0s", k19=(96.0, 96.0), tiles=(48.0, 48.0),
               sel_gone=48.0, launches_gone=288.0):
    toks = {str(i): [1, 2, 3, i] for i in range(16)}
    base = {K19_KERNEL: k19[0], TILE_KERNEL: tiles[0], "void at::native::(anonymous namespace)::indexSelectS...": 96.0,
            "void at::native::vectorized_elementwise_kernel<4, at...": 247.0, "other": 1265.375}
    lean = dict(base)
    lean[K19_KERNEL], lean[TILE_KERNEL] = k19[1], tiles[1]
    lean["void at::native::(anonymous namespace)::indexSelectS..."] -= sel_gone
    lean["other"] -= launches_gone - sel_gone
    return {"premise": {"rc": premise, "last": [last]},
            "speed": {"off": [{"ms": off, "tokens": toks}, {"ms": off, "tokens": toks}],
                      "on": [{"ms": on, "tokens": dict(toks)}, {"ms": on, "tokens": dict(toks)}]},
            "census": {"off": {"replays": 8, "per_step": base}, "on": {"replays": 8, "per_step": lean}}}


def self_test() -> None:
    got = reduce(_synthetic())
    assert got["verdict"] == "LICENSED" and abs(got["launches_per_step"]["off"] - got["launches_per_step"]["on"] - 288) < 1e-9, got
    assert reduce(_synthetic(on=9.9))["verdict"] == "NOT_FASTER"                             # 0.976 > 0.97
    assert reduce(_synthetic(premise=1))["verdict"] == "VOID"
    assert reduce(_synthetic(last="3 passed, 3 skipped in 9.0s"))["verdict"] == "VOID"      # K23's file skipped
    s = _synthetic()
    s["premise"] = None
    assert reduce(s)["verdict"] == "VOID"
    assert reduce(_synthetic(k19=(96.0, 0.0)))["verdict"] == "VOID"                          # ON left K19
    assert reduce(_synthetic(tiles=(0.0, 48.0)))["verdict"] == "VOID"                        # OFF is not K19's route
    assert reduce(_synthetic(sel_gone=0.0, launches_gone=288.0))["verdict"] == "VOID"        # expansion still made
    assert reduce(_synthetic(launches_gone=96.0))["verdict"] == "VOID"                       # glue not folded
    s = _synthetic()
    s["speed"]["on"][1]["tokens"] = dict(s["speed"]["on"][1]["tokens"], **{"7": [1, 2, 4, 7]})
    got = reduce(s)
    assert got["verdict"] == "IDENTITY_FAIL" and "['7']" in got["reasons"][0], got           # one row, second draw
    s = _synthetic(on=9.0)
    s["speed"]["on"][0]["tokens"] = {}
    assert reduce(s)["verdict"] == "IDENTITY_FAIL"                                           # whatever the speed
    s = _synthetic()
    s["speed"]["on"][1]["ms"] = 9.45 * 1.05
    assert reduce(s)["verdict"] == "VOID"                                                    # draws 5 % apart
    s = _synthetic()
    s["speed"]["off"][1] = None
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    del s["census"]["on"]
    assert reduce(s)["verdict"] == "VOID"
    print("p89_reduce self-test OK (14 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1, default=str))
    print(f"P89_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    if "speed" in v:
        s = v["speed"]
        print(f"  B=16: off {s['off_ms']:.3f} ms, on {s['on_ms']:.3f} ms, ratio {s['ratio']:.3f}")
    if "launches_per_step" in v:
        print(f"  launches/step: off {v['launches_per_step']['off']:.1f}, on {v['launches_per_step']['on']:.1f}; "
              f"index_select gone {v['index_select_gone_per_step']:.1f}")
        for n, dlt in v["per_kernel_delta"].items():
            print(f"    {dlt:+8.1f}  {n[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
