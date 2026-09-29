#!/usr/bin/env python3
"""Lane B771b's reducer (bench/b771b/PREREG-b771b.md): does the fixed bucketed-graph path (e4b#777, grouped-nf4-gemm#413)
decode exactly as the eager runner, and what is P80's ratio on it?

Inputs, all written on the box:
- ``armT.xml`` (the GPU tests, junit), ``armM.xml`` (the invariant test under the registered mutation, junit);
- ``b771b_{A1,B1,B2,A2,P,Ad}.json`` (the decode arms: P80's five plus Ad, the eager runner with the device grouping).

    python b771b_reduce.py --dir <dir> --out verdict.json
    python b771b_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BAND = 1.03
ARMS = ("A1", "B1", "B2", "A2", "P", "Ad")
MODES = {"A1": "eager", "A2": "eager", "B1": "graph", "B2": "graph", "P": "padded", "Ad": "eager"}
GROUPING = {"A1": False, "A2": False, "B1": True, "B2": True, "P": True, "Ad": True}
INVARIANT = "test_every_bucket_step_advances_its_rows_own_kv_length"
REPLAY = "test_a_bucket_replay_decodes_exactly_as_the_padded_eager_step"
ROUTING = "test_a_bound_bucket_of_one_row_appends_through_the_bucket_slot"
BYTES = ("test_bitwise_against_eager_path", "test_bt1_bitwise_against_t1_loop")


def junit_outcomes(xml_text: str) -> dict:
    """testcase name (with its parameter id) -> 'passed' | 'failed' | 'skipped' | 'error'."""
    out = {}
    for tc in ET.fromstring(xml_text).iter("testcase"):
        name = tc.get("name", "")
        kinds = {child.tag for child in tc}
        out[name] = ("failed" if "failure" in kinds else "error" if "error" in kinds
                     else "skipped" if "skipped" in kinds else "passed")
    return out


def _named(outcomes: dict, stem: str) -> dict:
    return {k: v for k, v in outcomes.items() if k == stem or k.startswith(stem + "[")}


def _first_divergence(x: dict, y: dict):
    for row in sorted(x, key=int):
        a, b = x[row], y.get(row, [])
        for i, (u, v) in enumerate(zip(a, b)):
            if u != v:
                return {"row": int(row), "token_index": i}
        if len(a) != len(b):
            return {"row": int(row), "token_index": min(len(a), len(b)), "length_differs": True}
    return None


def reduce(tests: dict, mutation: dict, recs: dict) -> dict:
    v: dict = {"lane": "B771b", "band": BAND, "reasons": [], "verdict": None}
    # ---- the tests: the invariant, the replay, the routing and gnf4's byte gates must PASS on sm_89+ (none skipped)
    if not tests or not mutation:
        v["verdict"] = "VOID"
        v["reasons"].append("a test arm wrote no junit")
        return v
    need = {**_named(tests, INVARIANT), **_named(tests, REPLAY), **_named(tests, ROUTING)}
    for b in BYTES:
        need.update(_named(tests, b))
    v["tests"] = need
    if len(_named(tests, INVARIANT)) != 2 or not _named(tests, REPLAY) or not _named(tests, ROUTING) \
            or any(not _named(tests, b) for b in BYTES):
        v["verdict"] = "VOID"
        v["reasons"].append("a registered test did not run (missing from the junit)")
        return v
    if any(s == "skipped" for s in need.values()):
        v["verdict"] = "VOID"
        v["reasons"].append(f"a registered test SKIPPED on an sm_89+ box: {[k for k, s in need.items() if s == 'skipped']}")
        return v
    failed = [k for k, s in need.items() if s != "passed"]
    if failed:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"registered tests failed on the fixed path: {failed}")
        return v
    mut = _named(mutation, INVARIANT)
    v["mutation"] = mut
    if not mut or any(s != "failed" for s in mut.values()):
        v["verdict"] = "VOID"
        v["reasons"].append(f"the invariant test did not fail under the mutation (inert for the bug it names): {mut}")
        return v
    # ---- the decode arms
    missing = [k for k in ARMS if not isinstance(recs.get(k), dict) or "timed_agg_tok_s" not in recs[k]]
    if missing:
        v["verdict"] = "VOID"
        v["reasons"].append(f"decode arm(s) without a receipt: {missing}")
        return v
    for k in ARMS:
        r = recs[k]
        if r.get("mode") != MODES[k] or bool(r.get("device_grouping")) is not GROUPING[k]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k} ran mode {r.get('mode')!r} / device_grouping {r.get('device_grouping')!r}")
            return v
    trace = recs["A1"]["trace"]
    if any(recs[k]["trace"] != trace for k in ARMS):
        v["verdict"] = "VOID"
        v["reasons"].append("arms disagree on the trace")
        return v
    for k in ("B1", "B2"):
        st = recs[k].get("graph_status") or {}
        eager = sum(int(s.get("eager_steps", 0)) for s in (recs[k].get("graph_stats") or {}).values())
        if not st or any(s != "graph" for s in st.values()) or eager:
            v["verdict"] = "REFUTED"
            v["reasons"].append(f"{k}: a bucket fell back to eager ({st}, eager_steps {eager})")
            return v
    tok = {k: recs[k]["passes"][1]["tokens"] for k in ARMS}
    idn = {"Ad_vs_P": _first_divergence(tok["Ad"], tok["P"]), "B1_vs_P": _first_divergence(tok["B1"], tok["P"]),
           "B2_vs_P": _first_divergence(tok["B2"], tok["P"]), "A1_vs_A2": _first_divergence(tok["A1"], tok["A2"]),
           # reported, not decisive: A1 runs the default T > 1 grouping, a different kernel set
           "A1_vs_Ad": _first_divergence(tok["A1"], tok["Ad"])}
    v["identity"] = idn
    if idn["A1_vs_A2"] is not None:
        v["verdict"] = "VOID"
        v["reasons"].append(f"the control does not repeat itself: {idn['A1_vs_A2']}")
        return v
    bad = {k: idn[k] for k in ("Ad_vs_P", "B1_vs_P", "B2_vs_P") if idn[k] is not None}
    if bad:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"the graph path does not decode as the eager runner with the same grouping: {bad}")
        return v
    agg = {k: float(recs[k]["timed_agg_tok_s"]) for k in ARMS}
    r = {"A2/A1": agg["A2"] / agg["A1"], "B2/B1": agg["B2"] / agg["B1"], "B1/A1": agg["B1"] / agg["A1"],
         "B2/A2": agg["B2"] / agg["A2"], "P/A1": agg["P"] / agg["A1"], "B1/P": agg["B1"] / agg["P"],
         "Ad/A1": agg["Ad"] / agg["A1"]}
    v["agg_tok_s"], v["ratios"] = agg, r
    v["step_ms_by_active"] = {k: recs[k]["timed_mean_step_ms_by_active"] for k in ARMS}
    for sp in ("A2/A1", "B2/B1"):
        if not (1 / BAND <= r[sp] <= BAND):
            v["verdict"] = "VOID"
            v["reasons"].append(f"self-pair {sp} = {r[sp]:.4f} outside [{1 / BAND:.4f}, {BAND}]")
            return v
    if r["B1/A1"] > BAND and r["B2/A2"] > BAND:
        v["verdict"] = "CONFIRMED"
        v["reasons"].append(f"tests hold, the mutation is caught, Ad == P == B1 == B2 bitwise, and both pairings exceed "
                            f"{BAND}: B1/A1 {r['B1/A1']:.4f}, B2/A2 {r['B2/A2']:.4f}")
    else:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"a pairing at or below {BAND}: B1/A1 {r['B1/A1']:.4f}, B2/A2 {r['B2/A2']:.4f}")
    return v


def _load(d: Path):
    def xml(name):
        f = d / name
        return junit_outcomes(f.read_text()) if f.is_file() and f.stat().st_size else {}
    recs = {}
    for k in ARMS:
        f = d / f"b771b_{k}.json"
        if f.is_file():
            try:
                recs[k] = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                recs[k] = {"status": f"unreadable: {e}"}
    return xml("armT.xml"), xml("armM.xml"), recs


def _junit(cases: dict) -> str:
    body = []
    for name, kind in cases.items():
        inner = {"failed": "<failure message='x'/>", "skipped": "<skipped message='x'/>", "error": "<error message='x'/>"}.get(kind, "")
        body.append(f"<testcase classname='t' name='{name}'>{inner}</testcase>")
    return "<testsuites><testsuite>" + "".join(body) + "</testsuite></testsuites>"


def _synthetic(*, tests=None, mut=None, agg=None, tok_b=None, tok_ad=None, grouping_a=False):
    t = {f"{INVARIANT}[graph]": "passed", f"{INVARIANT}[padded-eager]": "passed", REPLAY: "passed", ROUTING: "passed",
         f"{BYTES[0]}[0-1]": "passed", f"{BYTES[1]}[1]": "passed"}
    t.update(tests or {})
    m = {f"{INVARIANT}[graph]": "failed", f"{INVARIANT}[padded-eager]": "failed"}
    m.update(mut or {})
    agg = agg or {"A1": 100.0, "A2": 101.0, "B1": 220.0, "B2": 221.0, "P": 140.0, "Ad": 120.0}
    base = {"0": [1, 2, 3], "1": [4, 5]}
    trace = [2, 2, 1]
    recs = {}
    for k in ARMS:
        toks = tok_b if (k.startswith("B") and tok_b) else tok_ad if (k == "Ad" and tok_ad) else base
        st = {b: "graph" for b in ("1", "2")} if MODES[k] == "graph" else None
        stats = {b: {"eager_steps": 0} for b in ("1", "2")} if MODES[k] == "graph" else None
        grp = GROUPING[k] if not (k in ("A1", "A2") and grouping_a) else True
        recs[k] = {"mode": MODES[k], "device_grouping": grp, "trace": trace, "graph_status": st, "graph_stats": stats,
                   "timed_agg_tok_s": agg[k], "timed_mean_step_ms_by_active": {"2": 1.0},
                   "passes": [{"tokens": toks}, {"tokens": toks}]}
    return junit_outcomes(_junit(t)), junit_outcomes(_junit(m)), recs


def self_test() -> None:
    assert reduce(*_synthetic())["verdict"] == "CONFIRMED"
    # the invariant fails on the fixed path: refuted
    assert reduce(*_synthetic(tests={f"{INVARIANT}[graph]": "failed"}))["verdict"] == "REFUTED"
    # a registered test skipped on the box: void (the card was supposed to run it)
    assert reduce(*_synthetic(tests={f"{BYTES[0]}[0-1]": "skipped"}))["verdict"] == "VOID"
    # the mutation is not caught: the test is inert, void
    assert reduce(*_synthetic(mut={f"{INVARIANT}[graph]": "passed"}))["verdict"] == "VOID"
    # the graph path still decodes differently from the eager runner with the same grouping: refuted
    got = reduce(*_synthetic(tok_b={"0": [1, 2, 9], "1": [4, 5]}))
    assert got["verdict"] == "REFUTED" and got["identity"]["B1_vs_P"] == {"row": 0, "token_index": 2}, got
    assert reduce(*_synthetic(tok_ad={"0": [1, 7, 3], "1": [4, 5]}))["verdict"] == "REFUTED"
    # slow graphs: refuted; drift: void; the wrong grouping on the control: void
    assert reduce(*_synthetic(agg={"A1": 100.0, "A2": 101.0, "B1": 102.0, "B2": 103.0, "P": 100.0, "Ad": 100.0}))["verdict"] == "REFUTED"
    assert reduce(*_synthetic(agg={"A1": 100.0, "A2": 110.0, "B1": 220.0, "B2": 221.0, "P": 140.0, "Ad": 120.0}))["verdict"] == "VOID"
    assert reduce(*_synthetic(grouping_a=True))["verdict"] == "VOID"
    # a missing arm or junit: void
    t, m, r = _synthetic()
    del r["Ad"]
    assert reduce(t, m, r)["verdict"] == "VOID"
    assert reduce({}, m, _synthetic()[2])["verdict"] == "VOID"
    print("b771b_reduce self-test OK (11 cases)")


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
    v = reduce(*_load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"B771B_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    for k, x in (v.get("ratios") or {}).items():
        print(f"  {k} = {x:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
