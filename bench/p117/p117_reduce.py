#!/usr/bin/env python3
"""Lane P117's reducer (bench/p117/PREREG-p117.md; e4b#846): decode in one 32- or 64-row graph (W32, W64, W64pad)
against decode in pieces of at most 16 rows (R), judged against R's own arithmetically neutral perturbations. Derived
from ``bench/p110/p110_reduce.py`` (P110's rule) by named substitutions.

Input: ``box.json`` (``p117_box.py``) and ``summary.txt`` (the premise line) in the run directory.

Per window w and arm X, ``d_X(w)`` = the mean NLL of the continuation under X minus under R (positive: X is worse).
- **The floor:** ``half``, ``chunk`` and ``rev``, plus ``rep`` if R did not repeat bit for bit.
  - ``B_floor`` = the largest |mean over windows of d_f|;
  - ``S_floor`` = the largest mean over windows of |d_f|.
- **passes(X):** ``mean d_X <= B_floor + TOL`` and ``mean |d_X| <= SPREAD_X * max(S_floor, SPREAD_MIN)``.

Verdict:
- **NO_READING:** no premise line saying it held, or no box record.
- **VOID**, any of:
  - the record names another e4b / grouped-nf4-gemm commit, or the model is not at a registered revision;
  - an arm has other than its registered windows (W64pad: the first 48, or all if fewer);
  - engagement is wrong. Per pass, with G windows and its buckets' largest T:
    - decode attention calls = (cont - 1) x layers x ceil(G / T) (G64: none -- a replay runs no Python attention);
    - device grouping on in every pass;
    - every bucket ``eager: capture=False`` (G64: every bucket ``graph``);
    - the bucket statistics: (cont - 1) x ceil(G / T) pieces, each the runner's split, eager (G64: replays), with the
      registered padding rows;
  - **FUNCTION**: G64's emitted tokens differ from W64's at any position (the replay is not its padded eager step);
  - ``mutant_scale`` passes (the gate cannot fail).
- **AT_PARITY:** W32, W64 and W64pad all pass.
- **COST:** otherwise, naming every subject that fails.

Reported, never gated: every arm's bias, spread, standard error, max |d|, mean KL against R and argmax agreement; R's
bit-identity on repeat; the function gate's counts.

    python p117_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA]
    python p117_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

GNF4_SHA = "b4f93f1c62d1e3436ed45bec8ccd608c90433737"          # grouped-nf4-gemm v0.42.0, e4b CI's pin (SC2e's)
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}
WINDOWS = {"Qwen/Qwen3-30B-A3B": 64, "ibm-granite/granite-3.1-3b-a800m-instruct": 40}
PAD_WINDOWS = 48
TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005
B16, B8, B32, B64 = (1, 2, 4, 8, 16), (1, 2, 4, 8), (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64)
BUCKETS = {"R": B16, "rep": B16, "half": B8, "chunk": B16, "rev": B16, "W32": B32, "W64": B64, "W64pad": B64,
           "mutant_scale": B64, "G64": B64}
FLOORS = ("half", "chunk", "rev")
SUBJECTS = ("W32", "W64", "W64pad")
SCORED = ("R", "rep") + FLOORS + SUBJECTS + ("mutant_scale",)
ARMS = SCORED + ("G64",)


def _bucket_for(n, buckets):
    return next(b for b in buckets if n <= b)


def expected_stats(arm, g, cont):
    """The bucket statistics a pass of ``g`` windows registers: the runner's split into pieces of the largest bucket,
    each padded to its bucket, ``cont - 1`` decode steps."""
    bs = BUCKETS[arm]
    top, steps, out = bs[-1], cont - 1, {}
    left = g
    while left > 0:
        n = min(left, top)
        b = _bucket_for(n, bs)
        s = out.setdefault(str(b), {"pieces": 0, "rows": 0, "pad_rows": 0})
        s["pieces"] += steps
        s["rows"] += n * steps
        s["pad_rows"] += (b - n) * steps
        left -= n
    return out


def windows_of(arm, n):
    return min(n, PAD_WINDOWS) if arm == "W64pad" else n


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
    C, L, n = box["cont"], box["layers"], box["windows"]
    for arm in ARMS:
        for k, e in enumerate(box["engagement"].get(arm, [])):
            g = windows_of(arm, n)
            want_stats = expected_stats(arm, g, C)
            pieces = sum(s["pieces"] for s in want_stats.values()) // (C - 1)
            want_calls = 0 if arm == "G64" else (C - 1) * L * pieces
            if e.get("decode_calls") != want_calls:
                out.append(f"{arm} pass {k}: {e.get('decode_calls')} decode attention calls, expected {want_calls}")
            if not (e.get("grouping_flags_in_pass") or {}).get("device_grouping"):
                out.append(f"{arm} pass {k}: device grouping off")
            st = e.get("graph_status") or {}
            want_status = "graph" if arm == "G64" else "eager: capture=False"
            if [st.get(str(b)) for b in BUCKETS[arm]] != [want_status] * len(BUCKETS[arm]):
                out.append(f"{arm} pass {k}: graph_status {st}")
            gs = e.get("graph_stats") or {}
            for b, w in want_stats.items():
                got = gs.get(b, {})
                runs = (got.get("replays", 0), got.get("eager_steps", 0))
                want_runs = (w["pieces"], 0) if arm == "G64" else (0, w["pieces"])
                if runs != want_runs or got.get("rows") != w["rows"] or got.get("pad_rows") != w["pad_rows"]:
                    out.append(f"{arm} pass {k}: bucket {b} stats {got}, expected replays/eager {want_runs}, "
                               f"rows {w['rows']}, pad rows {w['pad_rows']}")
            extra = [b for b, v in gs.items() if b not in want_stats and (v.get("replays") or v.get("eager_steps"))]
            if extra:
                out.append(f"{arm} pass {k}: buckets {extra} ran, not in the registered split")
    return out


def reduce(run: Path, e4b_sha: str) -> dict:
    summ = (run / "summary.txt").read_text() if (run / "summary.txt").is_file() else ""
    out = {"lane": "P117"}
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
    if box.get("windows") != want:
        void.append(f"{box.get('windows')} windows, expected {want}")
    for arm in SCORED:
        n = len(per.get(arm, []))
        if n != windows_of(arm, want):
            void.append(f"arm {arm} has {n} windows, expected {windows_of(arm, want)}")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    void += engagement_faults(box)
    fn = box.get("function") or {}
    out["function"] = fn
    if fn.get("positions") != want * (box["cont"] - 1) or fn.get("differ") != 0:
        void.append(f"FUNCTION: G64's emitted tokens against W64's {fn} (want {want * (box['cont'] - 1)} positions, 0 differ)")
    st = {arm: stats(per, arm) for arm in SCORED if arm != "R"}
    fl = floor_of(st, box.get("rep_identical"))
    out.update(stats=st, floor=fl, rep_identical=box.get("rep_identical"),
               bar={"tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN,
                    "bias_bar": fl["B_floor"] + TOL, "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN)})
    if passes(st["mutant_scale"], fl):
        void.append("mutant_scale passes the bar: the gate cannot fail")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    bias_bar, spread_bar = fl["B_floor"] + TOL, SPREAD_X * max(fl["S_floor"], SPREAD_MIN)
    lines = [f"{a} bias {st[a]['bias']:+.5f} (bar {bias_bar:.5f}), spread {st[a]['spread']:.5f} (bar {spread_bar:.5f})"
             for a in SUBJECTS]
    failing = [a for a in SUBJECTS if not passes(st[a], fl)]
    out["subjects_pass"] = {a: passes(st[a], fl) for a in SUBJECTS}
    out.update(verdict="AT_PARITY" if not failing else "COST", reasons=lines, failing=failing)
    return out


# ------------------------------------------------------------------------------------------------ self-test --

def _box(model="Qwen/Qwen3-30B-A3B", cont=8, layers=4, d=None, e4b="a" * 40, differ=0):
    """A synthetic box record: per-arm NLL offsets ``d[arm]`` (default small and neutral), every engagement exact."""
    d = d or {}
    n = WINDOWS[model]
    per, eng = {}, {}
    for arm in ARMS:
        g = windows_of(arm, n)
        want = expected_stats(arm, g, cont)
        if arm != "G64":
            off = d.get(arm, 0.0 if arm in ("R", "rep") else 0.001)
            per[arm] = [{"window": w, "nll": 2.0 + 0.01 * (w % 5) + off, "argmax_agree": 1.0,
                         **({} if arm == "R" else {"kl": 1e-4})} for w in range(g)]
        pieces = sum(s["pieces"] for s in want.values()) // (cont - 1)
        st = {str(b): ("graph" if arm == "G64" else "eager: capture=False") for b in BUCKETS[arm]}
        gs = {b: {"replays": s["pieces"] if arm == "G64" else 0, "eager_steps": 0 if arm == "G64" else s["pieces"],
                  "rows": s["rows"], "pad_rows": s["pad_rows"]} for b, s in want.items()}
        eng[arm] = [{"decode_calls": 0 if arm == "G64" else (cont - 1) * layers * pieces, "graph_status": st,
                     "graph_stats": copy.deepcopy(gs), "grouping_flags_in_pass": {"device_grouping": True}}]
    return {"model": model, "revision": REVS[model], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "windows": n, "group": n,
            "cont": cont, "layers": layers, "rep_identical": True,
            "function": {"positions": n * (cont - 1), "differ": differ}, "per_window": per, "engagement": eng}


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
    r = run(_box(d={**big, "W64": 0.05}))
    cases.append(("W64 costs", r["verdict"] == "COST" and r["failing"] == ["W64"]))
    r = run(_box(d={**big, "W32": 0.05, "W64pad": -0.05}))
    cases.append(("two subjects fail, both named", r["verdict"] == "COST" and r["failing"] == ["W32", "W64pad"]))
    cases.append(("a negative bias passes the bias bar", run(_box(d={**big, "W64": -0.008}))["verdict"] == "AT_PARITY"))
    cases.append(("mutant passes", run(_box(d={"mutant_scale": 0.002}))["verdict"] == "VOID"))
    cases.append(("the replay is not its padded step", run(_box(d=big, differ=3))["verdict"] == "VOID"))
    cases.append(("wrong e4b", run(_box(d=big, e4b="b" * 40))["verdict"] == "VOID"))
    b = _box(d=big)
    b["revision"] = "main"
    cases.append(("wrong revision", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["per_window"]["W64pad"] = b["per_window"]["W64pad"] + [dict(b["per_window"]["W64pad"][0], window=60)]
    cases.append(("W64pad not its 48", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["half"][0]["decode_calls"] -= 1
    cases.append(("calls", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["W64"][0]["graph_stats"]["64"]["eager_steps"] = 0
    b["engagement"]["W64"][0]["graph_stats"]["64"]["replays"] = 7
    cases.append(("W64 replayed instead of its eager step", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["W64pad"][0]["graph_stats"]["64"]["pad_rows"] = 0
    cases.append(("W64pad unpadded", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["R"][0]["graph_stats"]["32"] = {"replays": 0, "eager_steps": 7, "rows": 224, "pad_rows": 0}
    cases.append(("R ran a bucket outside its split", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["G64"][0]["graph_status"]["64"] = "eager: OOM"
    cases.append(("G64 not captured", run(b)["verdict"] == "VOID"))
    b = _box(d=big)
    b["engagement"]["rev"][0]["grouping_flags_in_pass"]["device_grouping"] = False
    cases.append(("host grouping", run(b)["verdict"] == "VOID"))
    cases.append(("floor widens the bar", run(_box(d={**big, "half": 0.02, "W64": 0.025}))["verdict"] == "AT_PARITY"))
    b = _box(d=big)
    b["rep_identical"] = False
    b["per_window"]["rep"] = [dict(x, nll=x["nll"] + 0.004) for x in b["per_window"]["rep"]]
    r = run(b)
    cases.append(("rep joins the floor", "rep" in r["floor"]["draws"] and r["verdict"] == "AT_PARITY"))
    cases.append(("granite proof", run(_box(model="ibm-granite/granite-3.1-3b-a800m-instruct", d=big))["verdict"] == "AT_PARITY"))
    e = expected_stats("R", 64, 8)
    cases.append(("R's split", e == {"16": {"pieces": 28, "rows": 448, "pad_rows": 0}}))
    e = expected_stats("W64pad", 48, 8)
    cases.append(("W64pad's split", e == {"64": {"pieces": 7, "rows": 336, "pad_rows": 112}}))
    bad = [n for n, ok in cases if not ok]
    print(f"p117_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    print(f"P117_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("floor", "bar", "function", "rep_identical", "subjects_pass"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    for arm, s in (v.get("stats") or {}).items():
        print(f"  {arm}: bias {s['bias']:+.5f} spread {s['spread']:.5f} se {s['se']:.5f} max {s['max_abs']:.5f} "
              f"kl {s['mean_kl']} agree {s['argmax_agree']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
