#!/usr/bin/env python3
"""Lane P108's reducer (bench/p108/PREREG-p108.md; e4b#359): Gemma-4's paged path against transformers' forward, judged
against transformers' own chaos.

Input: ``box.json`` (``p108_box.py``) and ``summary.txt`` (the premise line) in the run directory.

Per window w and arm X, ``d_X(w)`` = the mean NLL of the continuation under X minus under the reference R.
- **The floor:** the arithmetically neutral draws of the same model: ``oneshot``, ``chunk`` and ``batch``, plus ``rep``
  if the reference did not repeat bit for bit.
  - ``B_floor`` = the largest |mean over windows of d_f|;
  - ``S_floor`` = the largest mean over windows of |d_f|.
- **The subject** (``paged``): bias ``b`` = the mean of d (signed: positive means paged is worse); spread ``s`` = the
  mean of |d|.
- ``passes(X)``: ``b_X <= B_floor + TOL`` and ``s_X <= SPREAD_X * max(S_floor, SPREAD_MIN)``.

Verdict:
- **NO_READING:** no premise line saying it held, or no box record.
- **VOID**, any of:
  - the loaded commit is not the pinned revision;
  - the engagement is wrong: the subject's decode attention calls must equal ``expected_calls``, with exactly
    ``expected_sliding_calls`` of them at the model's sliding window and the rest at 0;
  - an arm has fewer windows than registered;
  - ``mutant_scale`` passes (the gate cannot fail).
- **AT_PARITY:** the subject passes.
- **COST:** otherwise.

Reported, never gated: every arm's bias, spread, mean KL against R and argmax agreement; the reference's bit-identity on
repeat; ``mutant_window`` (whether a dropped window is visible at this context length).

    python p108_reduce.py --dir RUN_DIR --out verdict.json
    python p108_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

REV = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
WINDOWS = 32
TOL, SPREAD_X, SPREAD_MIN = 0.05, 2.0, 0.01
FLOORS = ("oneshot", "chunk", "batch")
ARMS = ("R", "rep") + FLOORS + ("paged", "mutant_scale", "mutant_window")


def stats(per, arm):
    r = {x["window"]: x["nll"] for x in per["R"]}
    d = [x["nll"] - r[x["window"]] for x in per[arm]]
    n = len(d)
    if not n:
        return {"n": 0}
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    kls = [x["kl"] for x in per[arm] if "kl" in x]
    return {"n": n, "bias": mean, "spread": sum(abs(v) for v in d) / n, "se": sd / math.sqrt(n) if n else None,
            "max_abs": max(abs(v) for v in d), "mean_kl": sum(kls) / len(kls) if kls else None,
            "argmax_agree": sum(x["argmax_agree"] for x in per[arm]) / n}


def floor_of(st, rep_identical):
    draws = list(FLOORS) + ([] if rep_identical else ["rep"])
    b = max(abs(st[f]["bias"]) for f in draws)
    s = max(st[f]["spread"] for f in draws)
    return {"draws": draws, "B_floor": b, "S_floor": s}


def passes(s, fl):
    return s["bias"] <= fl["B_floor"] + TOL and s["spread"] <= SPREAD_X * max(fl["S_floor"], SPREAD_MIN)


def reduce(run: Path) -> dict:
    summ = (run / "summary.txt").read_text() if (run / "summary.txt").is_file() else ""
    out = {"lane": "P108"}
    if "premise ok" not in summ or not (run / "box.json").is_file():
        out.update(verdict="NO_READING", reasons=["premise did not hold" if "premise ok" not in summ else "no box record"])
        return out
    box = json.loads((run / "box.json").read_text())
    per, eng = box["per_window"], box["engagement"]
    void = []
    if box.get("loaded_commit") != REV:
        void.append(f"loaded commit {box.get('loaded_commit')} is not {REV}")
    for arm in ARMS:
        want = len(per["R"]) if arm != "rep" else box.get("group")
        if arm == "R" and len(per["R"]) < WINDOWS:
            void.append(f"{len(per['R'])} windows, registered {WINDOWS}")
        elif arm != "R" and len(per.get(arm, [])) != want:
            void.append(f"arm {arm} has {len(per.get(arm, []))} windows, expected {want}")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    win = str(eng.get("sliding_window"))
    by = eng.get("by_window", {})
    if eng.get("calls") != eng.get("expected_calls"):
        void.append(f"decode attention calls {eng.get('calls')} != expected {eng.get('expected_calls')}")
    if by.get(win, 0) != eng.get("expected_sliding_calls") or set(by) - {win, "0"} \
            or by.get("0", 0) != eng.get("expected_calls", 0) - eng.get("expected_sliding_calls", 0):
        void.append(f"window engagement {by} against {eng.get('expected_sliding_calls')} sliding calls at window {win}")
    st = {arm: stats(per, arm) for arm in ARMS if arm != "R"}
    fl = floor_of(st, eng.get("rep_identical"))
    out.update(stats=st, floor=fl, rep_identical=eng.get("rep_identical"),
               bar={"tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN})
    if passes(st["mutant_scale"], fl):
        void.append("mutant_scale passes the bar: the gate cannot fail")
    out["mutant_window_passes"] = passes(st["mutant_window"], fl)
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    s = st["paged"]
    if passes(s, fl):
        out.update(verdict="AT_PARITY", reasons=[])
    else:
        out.update(verdict="COST", reasons=[f"paged bias {s['bias']:+.4f} (bar {fl['B_floor'] + TOL:.4f}), spread "
                                            f"{s['spread']:.4f} (bar {SPREAD_X * max(fl['S_floor'], SPREAD_MIN):.4f})"])
    return out


# ------------------------------------------------------------------------------------------------ self-test --
def _box(d=None, n=WINDOWS, group=8):
    """A synthetic record: per-arm d(w) values (constant per arm unless given) around a reference NLL of 2.0."""
    d = d or {}
    base = {"rep": 0.0, "oneshot": 0.04, "chunk": -0.03, "batch": 0.02, "paged": 0.03, "mutant_scale": 0.9,
            "mutant_window": 0.2}
    base.update(d)
    per = {"R": [{"window": i, "nll": 2.0 + 0.01 * i, "argmax_agree": 1.0} for i in range(n)]}
    for arm, v in base.items():
        k = group if arm == "rep" else n
        vals = v if isinstance(v, list) else [v] * k
        per[arm] = [{"window": i, "nll": 2.0 + 0.01 * i + vals[i], "argmax_agree": 0.9, "kl": abs(vals[i]) / 10}
                    for i in range(k)]
    steps = 255 * (n // group)
    eng = {"calls": steps * 30, "expected_calls": steps * 30, "expected_sliding_calls": steps * 25,
           "by_window": {"1024": steps * 25, "0": steps * 5}, "sliding_window": 1024, "rep_identical": True,
           "groups": n // group}
    return {"loaded_commit": REV, "group": group, "per_window": per, "engagement": eng}


def _run(tmp: Path, box, premise="premise ok"):
    for f in tmp.iterdir():
        f.unlink()
    (tmp / "summary.txt").write_text(premise + "\n")
    if box is not None:
        (tmp / "box.json").write_text(json.dumps(box))
    return reduce(tmp)


def self_test() -> int:
    import tempfile
    alt = [0.2 if i % 2 else -0.2 for i in range(WINDOWS)]            # floor spread 0.2, bias 0
    cases = [
        ("good", "AT_PARITY", _box(), "premise ok"),
        ("premise failed", "NO_READING", _box(), "premise failed"),
        ("no premise line", "NO_READING", _box(), ""),
        ("no box", "NO_READING", None, "premise ok"),
        ("wrong commit", "VOID", dict(_box(), loaded_commit="0" * 40), "premise ok"),
        ("too few windows", "VOID", _box(n=24), "premise ok"),
        ("mutant passes", "VOID", _box({"mutant_scale": 0.05}), "premise ok"),
        ("paged worse than the bar", "COST", _box({"paged": 0.0401 + 0.05 + 0.001}), "premise ok"),
        ("paged at the bias bar", "AT_PARITY", _box({"oneshot": alt, "paged": 0.03 + 0.05 - 1e-9}), "premise ok"),
        ("paged far BETTER is caught by the spread (a future-token leak reads so)", "COST", _box({"paged": -0.3}), "premise ok"),
        ("paged spread over 2x floor", "COST", _box({"paged": [0.3 if i % 2 else -0.3 for i in range(WINDOWS)],
                                                     "mutant_scale": 0.9}), "premise ok"),
        ("chaotic floor widens the spread bar", "AT_PARITY", _box({"oneshot": alt, "paged": [0.3 if i % 2 else -0.3 for i in range(WINDOWS)]}), "premise ok"),
        ("a floor's bias widens the bias bar", "AT_PARITY", _box({"batch": 0.15, "paged": 0.18}), "premise ok"),
        ("rep not identical joins the floor", "AT_PARITY", None, "premise ok"),
        ("calls short", "VOID", None, "premise ok"),
        ("window not engaged", "VOID", None, "premise ok"),
        ("stray window value", "VOID", None, "premise ok"),
    ]
    b = _box({"rep": 0.2, "paged": 0.24})
    b["engagement"]["rep_identical"] = False
    cases[13] = (cases[13][0], cases[13][1], b, cases[13][3])
    b = _box()
    b["engagement"]["calls"] -= 1
    cases[14] = (cases[14][0], cases[14][1], b, cases[14][3])
    b = _box()
    b["engagement"]["by_window"] = {"0": b["engagement"]["expected_calls"]}
    cases[15] = (cases[15][0], cases[15][1], b, cases[15][3])
    b = _box()
    b["engagement"]["by_window"] = dict(b["engagement"]["by_window"], **{"4096": 0})
    cases[16] = (cases[16][0], cases[16][1], b, cases[16][3])
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        for name, want, box, premise in cases:
            got = _run(Path(td), copy.deepcopy(box), premise)["verdict"]
            if got != want:
                fails += 1
                print(f"SELF-TEST FAIL {name}: got {got}, want {want}")
    if fails:
        print(f"self-test FAILED ({fails} of {len(cases)} cases)")
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
    out = reduce(Path(a.dir))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("VERDICT", out["verdict"], "|", "; ".join(out.get("reasons", [])) or "-")
    if "stats" in out:
        print("FLOOR", json.dumps(out["floor"]))
        for arm, s in out["stats"].items():
            print(f"ARM {arm}: bias {s['bias']:+.4f} spread {s['spread']:.4f} se {s['se']:.4f} kl {s['mean_kl']} agree "
                  f"{s['argmax_agree']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
