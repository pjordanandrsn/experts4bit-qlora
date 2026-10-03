#!/usr/bin/env python3
"""Lane P110's reducer (bench/p110/PREREG-p110.md; e4b#770): the graph server's arithmetic (P: device grouping and bucket
padding) against the eager default's (R), judged against the eager default's own arithmetically neutral perturbations.

Input: ``box.json`` (``p110_box.py``) and ``summary.txt`` (the premise line) in the run directory.

Per window w and arm X, ``d_X(w)`` = the mean NLL of the continuation under X minus under R (positive: X is worse).
- **The floor:** ``half``, ``chunk`` and ``rev``, plus ``rep`` if R did not repeat bit for bit.
  - ``B_floor`` = the largest |mean over windows of d_f|;
  - ``S_floor`` = the largest mean over windows of |d_f|.
- **passes(X):** ``mean d_X <= B_floor + TOL`` and ``mean |d_X| <= SPREAD_X * max(S_floor, SPREAD_MIN)``.

Verdict:
- **NO_READING:** no premise line saying it held, or no box record.
- **VOID**, any of:
  - the record names another e4b / grouped-nf4-gemm commit, or the model is not at a registered revision;
  - an arm has fewer windows than registered (``rep``: one group);
  - engagement is wrong:
    - every pass must make ``(cont - 1) * layers`` decode attention calls (``half``: twice that);
    - R, rep and the floor must run with device grouping off, and D, P and the mutant with it on;
    - P and the mutant must run every decode step as an eager padded step of the group's bucket
      (``graph_status`` all ``eager: capture=False``, no replays, ``pad_rows`` = (bucket - group) * (cont - 1) per pass);
  - ``mutant_scale`` passes (the gate cannot fail).
- **AT_PARITY:** P passes.
- **COST:** otherwise.

Reported, never gated: every arm's bias, spread, standard error, max |d|, mean KL against R and argmax agreement; D's
pass or fail; R's bit-identity on repeat.

    python p110_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA]
    python p110_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

GNF4_SHA = "51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00"
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}
WINDOWS = {"Qwen/Qwen3-30B-A3B": 48, "ibm-granite/granite-3.1-3b-a800m-instruct": 12}
TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005
BUCKETS = (1, 2, 4, 8, 16)
FLOORS = ("half", "chunk", "rev")
ARMS = ("R", "rep") + FLOORS + ("D", "P", "mutant_scale")
DEVICE_ARMS = ("D", "P", "mutant_scale")


def stats(per, arm):
    r = {x["window"]: x["nll"] for x in per["R"]}
    d = [x["nll"] - r[x["window"]] for x in per[arm]]
    n = len(d)
    if not n:
        return {"n": 0}
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    kls = [x["kl"] for x in per[arm] if "kl" in x]
    return {"n": n, "bias": mean, "spread": sum(abs(v) for v in d) / n, "se": sd / math.sqrt(n),
            "max_abs": max(abs(v) for v in d), "mean_kl": sum(kls) / len(kls) if kls else None,
            "argmax_agree": sum(x["argmax_agree"] for x in per[arm]) / n}


def floor_of(st, rep_identical):
    draws = list(FLOORS) + ([] if rep_identical else ["rep"])
    return {"draws": draws, "B_floor": max(abs(st[f]["bias"]) for f in draws), "S_floor": max(st[f]["spread"] for f in draws)}


def passes(s, fl):
    return s["bias"] <= fl["B_floor"] + TOL and s["spread"] <= SPREAD_X * max(fl["S_floor"], SPREAD_MIN)


def engagement_faults(box) -> list:
    out = []
    C, L, g = box["cont"], box["layers"], box["group"]
    bucket = min(b for b in BUCKETS if b >= g)
    for arm in ARMS:
        for k, e in enumerate(box["engagement"].get(arm, [])):
            want_calls = (C - 1) * L * (2 if arm == "half" else 1)
            if e.get("decode_calls") != want_calls:
                out.append(f"{arm} pass {k}: {e.get('decode_calls')} decode attention calls, expected {want_calls}")
            dg = (e.get("grouping_flags_in_pass") or {}).get("device_grouping")
            if bool(dg) != (arm in DEVICE_ARMS):
                out.append(f"{arm} pass {k}: device_grouping={dg}")
            if arm in ("P", "mutant_scale"):
                st = e.get("graph_status") or {}
                if [st.get(str(b)) for b in BUCKETS] != ["eager: capture=False"] * len(BUCKETS):
                    out.append(f"{arm} pass {k}: graph_status {st}")
                gs = (e.get("graph_stats") or {}).get(str(bucket), {})
                if (gs.get("eager_steps"), gs.get("pad_rows"), gs.get("replays", 0)) != (C - 1, (bucket - g) * (C - 1), 0):
                    out.append(f"{arm} pass {k}: bucket {bucket} stats {gs}, expected {C - 1} eager steps, "
                               f"{(bucket - g) * (C - 1)} pad rows, no replays")
    return out


def reduce(run: Path, e4b_sha: str) -> dict:
    summ = (run / "summary.txt").read_text() if (run / "summary.txt").is_file() else ""
    out = {"lane": "P110"}
    if "premise ok" not in summ or not (run / "box.json").is_file():
        out.update(verdict="NO_READING", reasons=["premise did not hold" if "premise ok" not in summ else "no box record"])
        return out
    box = json.loads((run / "box.json").read_text())
    per = box["per_window"]
    void = []
    if box.get("e4b_sha") != e4b_sha:
        void.append(f"e4b {box.get('e4b_sha')} != {e4b_sha}")
    if box.get("gnf4_sha") != GNF4_SHA:
        void.append(f"grouped-nf4-gemm {box.get('gnf4_sha')} != {GNF4_SHA}")
    if REVS.get(box.get("model")) != box.get("revision"):
        void.append(f"model {box.get('model')}@{box.get('revision')} is not a registered revision")
    want = WINDOWS.get(box.get("model"), 0)
    for arm in ARMS:
        n = len(per.get(arm, []))
        if n != (box.get("group") if arm == "rep" else want):
            void.append(f"arm {arm} has {n} windows, expected {box.get('group') if arm == 'rep' else want}")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    void += engagement_faults(box)
    st = {arm: stats(per, arm) for arm in ARMS if arm != "R"}
    fl = floor_of(st, box.get("rep_identical"))
    out.update(stats=st, floor=fl, rep_identical=box.get("rep_identical"), D_passes=passes(st["D"], fl),
               bar={"tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN,
                    "bias_bar": fl["B_floor"] + TOL, "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN)})
    if passes(st["mutant_scale"], fl):
        void.append("mutant_scale passes the bar: the gate cannot fail")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    s = st["P"]
    if passes(s, fl):
        out.update(verdict="AT_PARITY", reasons=[f"P bias {s['bias']:+.5f} <= {fl['B_floor'] + TOL:.5f}, spread {s['spread']:.5f} "
                                                 f"<= {SPREAD_X * max(fl['S_floor'], SPREAD_MIN):.5f}"])
    else:
        out.update(verdict="COST", reasons=[f"P bias {s['bias']:+.5f} (bar {fl['B_floor'] + TOL:.5f}), spread {s['spread']:.5f} "
                                            f"(bar {SPREAD_X * max(fl['S_floor'], SPREAD_MIN):.5f})"])
    return out


# ------------------------------------------------------------------------------------------------ self-test --

def _box(model="Qwen/Qwen3-30B-A3B", group=12, cont=8, layers=4, d=None, e4b="a" * 40):
    """A synthetic box record: per-arm NLL offsets ``d[arm]`` (default small and neutral), every engagement exact."""
    d = d or {}
    n = WINDOWS[model]
    per, eng = {}, {}
    bucket = min(b for b in BUCKETS if b >= group)
    for arm in ARMS:
        k = group if arm == "rep" else n
        off = d.get(arm, 0.0 if arm in ("R", "rep") else 0.001)
        per[arm] = [{"window": w, "nll": 2.0 + 0.01 * (w % 5) + (off if isinstance(off, float) else off[w]),
                     "argmax_agree": 1.0, **({} if arm == "R" else {"kl": 1e-4})} for w in range(k)]
        passes_n = 1 if arm == "rep" else n // group
        e = {"decode_calls": (cont - 1) * layers * (2 if arm == "half" else 1),
             "grouping_flags_in_pass": {"device_grouping": arm in DEVICE_ARMS}}
        if arm in ("P", "mutant_scale"):
            e["graph_status"] = {str(b): "eager: capture=False" for b in BUCKETS}
            e["graph_stats"] = {str(bucket): {"eager_steps": cont - 1, "pad_rows": (bucket - group) * (cont - 1), "replays": 0}}
        eng[arm] = [copy.deepcopy(e) for _ in range(passes_n)]
    return {"model": model, "revision": REVS[model], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "group": group, "cont": cont,
            "layers": layers, "rep_identical": True, "per_window": per, "engagement": eng}


def self_test() -> int:
    import tempfile
    E = "a" * 40
    cases = []

    def run(box, premise=True):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            (p / "summary.txt").write_text("premise ok\n" if premise else "premise failed\n")
            if box is not None:
                (p / "box.json").write_text(json.dumps(box))
            return reduce(p, E)

    big = {"mutant_scale": 0.8}
    cases.append(("parity", run(_box(d=big))["verdict"] == "AT_PARITY"))
    cases.append(("no premise", run(_box(d=big), premise=False)["verdict"] == "NO_READING"))
    cases.append(("no box", run(None)["verdict"] == "NO_READING"))
    cases.append(("cost", run(_box(d={**big, "P": 0.05}))["verdict"] == "COST"))
    cases.append(("a negative bias passes the bias bar", run(_box(d={**big, "P": -0.008}))["verdict"] == "AT_PARITY"))
    cases.append(("a large change either way fails the spread bar", run(_box(d={**big, "P": -0.05}))["verdict"] == "COST"))
    cases.append(("mutant passes", run(_box(d={"mutant_scale": 0.002}))["verdict"] == "VOID"))
    cases.append(("wrong e4b", run(_box(d=big, e4b="b" * 40))["verdict"] == "VOID"))
    b = _box(d=big)
    b["revision"] = "main"
    cases.append(("wrong revision", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["per_window"]["D"] = b["per_window"]["D"][:-1]
    cases.append(("short arm", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["half"][0]["decode_calls"] -= 1
    cases.append(("calls", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["R"][0]["grouping_flags_in_pass"]["device_grouping"] = True
    cases.append(("R device-grouped", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["P"][0]["graph_stats"]["16"]["replays"] = 3
    cases.append(("P replayed", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["P"][0]["graph_stats"]["16"]["pad_rows"] = 0
    cases.append(("P unpadded", run(b)["verdict"] == "VOID"))
    floor_wide = {**big, "half": 0.02, "P": 0.025}               # a wider floor admits a larger subject bias
    cases.append(("floor widens the bar", run(_box(d=floor_wide))["verdict"] == "AT_PARITY"))
    r = run(_box(d={**big, "D": 0.05}))
    cases.append(("D reported only", r["verdict"] == "AT_PARITY" and r["D_passes"] is False))
    b = _box(d=big)
    b["rep_identical"] = False
    b["per_window"]["rep"] = [dict(x, nll=x["nll"] + 0.004) for x in b["per_window"]["rep"]]
    r = run(b)
    cases.append(("rep joins the floor", "rep" in r["floor"]["draws"] and r["verdict"] == "AT_PARITY"))
    cases.append(("granite proof", run(_box(model="ibm-granite/granite-3.1-3b-a800m-instruct", d=big))["verdict"] == "AT_PARITY"))
    bad = [n for n, ok in cases if not ok]
    print(f"p110_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    v = reduce(Path(a.dir), a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1)
    print(f"P110_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("floor", "bar", "D_passes", "rep_identical"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    for arm, s in (v.get("stats") or {}).items():
        print(f"  {arm}: bias {s['bias']:+.5f} spread {s['spread']:.5f} se {s['se']:.5f} max {s['max_abs']:.5f} "
              f"kl {s['mean_kl']} agree {s['argmax_agree']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
