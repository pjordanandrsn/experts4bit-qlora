#!/usr/bin/env python3
"""Lane P87's reducer (bench/p87/PREREG-p87.md; e4b#564). Is K19 (E4B_INT4_GROUPED_SMALLM=1) faster on an RTX 5090
without moving the int4 stack's quality?

Reads, from the run directory:
- premise: rowexact.json, the on-box run of tests/test_k19_row_exact_gpu.py (K19's rows bit-equal alone and inside a
           B=16 step, which is what lets the B=1 K8 stand for the B=16 rows)
- speed:   e4b_b{16,1}_{off,on}{,_r2}.json (`step_ms_clean`, the graph-replay window), and the first draws' replay
           censuses logs/census_b{16,1}_{off,on}.txt (P42's table, parsed by bench/p42/p42_reduce.py's `parse_census`)
- quality: k8_{off,on}.json (`mean_nll`, PREREG-k8's 2,048 teacher-forced steps on window 9ef10d760ad9, the licensed
           recipe with the expert pack loaded by fingerprint); k8_build.json (the build that calibrated and dumped
           that pack, opt-in OFF) is reported beside them, not gated

  VOID          the premise did not hold or did not run; an arm is missing or failed; a census is missing; two draws
                of one arm differ by more than 3 %; ON did not route the experts through K19 (96 K19 calls per step,
                48 layers x gate_up + down, and the expert GEMV's 96 calls gone) or OFF ran K19; a K8 is off its
                window or step count; K8 ON equals K8 OFF to the bit (K19 changes the arithmetic, so it did not run)
  QUALITY_FAIL  |K8(on) - K8(off)| > 0.0095 nats (the K8 floor): K19 stays opt-in whatever its speed
  LICENSED      quality holds and B=16 is FASTER (median ON / median OFF <= 0.95)
  NOT_FASTER    quality holds and B=16 is not faster
Reported beside the verdict:
- B=1's ratio, read as FASTER (<= 0.97), SLOWER (>= 1.03) or NEUTRAL. The default a LICENSED read proposes depends on
  it: all decode rows, or only rows above B=1's.
- the K8 difference in nats, and whether the build equals OFF to the bit.

    python p87_reduce.py --dir <dir> --out verdict.json
    python p87_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

WINDOW = "9ef10d760ad9"
STEPS = 2048
K8_FLOOR = 0.0095
B16_FASTER = 0.95
B1_FASTER, B1_SLOWER = 0.97, 1.03
DRAW_SPREAD = 0.03
EXPERT_CALLS = 96                   # per decode step: 48 layers x (gate_up + down)
K19_KERNEL = "_gemm_int4_b32_grouped_smallm"
GEMV_KERNEL = "_gemv_int4_b32"      # the split-K GEMV; at B=1 it also serves the attention projections
BATCHES = (16, 1)


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


def _step(d: Path, name: str):
    r = _json(d / name)
    try:
        return float(r["step_ms_clean"])
    except (KeyError, TypeError, ValueError):
        return None


def _per_step(rows, replays, name):
    return sum(r["calls"] for r in rows if r["name"].strip() == name) / max(replays or 1, 1)


def load(d: Path) -> dict:
    p42 = _p42()
    data = {"premise": _json(d / "rowexact.json"), "speed": {}, "census": {}, "k8": {}}
    for B in BATCHES:
        for tag in ("off", "on"):
            data["speed"][(B, tag)] = [_step(d, f"e4b_b{B}_{tag}.json"), _step(d, f"e4b_b{B}_{tag}_r2.json")]
            c = d / "logs" / f"census_b{B}_{tag}.txt"
            if c.is_file():
                replays, rows = p42.parse_census(c.read_text())
                data["census"][(B, tag)] = {"replays": replays, "k19": _per_step(rows, replays, K19_KERNEL),
                                            "gemv": _per_step(rows, replays, GEMV_KERNEL)}
    for tag in ("build", "off", "on"):
        data["k8"][tag] = _json(d / f"k8_{tag}.json")
    return data


def _void(v, why):
    v["verdict"] = "VOID"
    v["reasons"].append(why)
    return v


def reduce(data: dict) -> dict:
    v: dict = {"lane": "P87", "verdict": None, "reasons": [], "speed": {}}
    pr = data.get("premise")
    if not isinstance(pr, dict) or pr.get("rc") != 0:
        return _void(v, f"the on-box premise (K19 rows bit-equal alone and inside B=16) did not hold or did not run: {pr}")
    for B in BATCHES:
        for tag in ("off", "on"):
            draws = data["speed"].get((B, tag))
            if not draws or None in draws:
                return _void(v, f"B={B} {tag}: a timed draw is missing or failed")
            if (max(draws) - min(draws)) / min(draws) > DRAW_SPREAD:
                return _void(v, f"B={B} {tag}: draws {draws} differ by more than {DRAW_SPREAD:.0%}")
            c = data["census"].get((B, tag))
            if not c or not c.get("replays"):
                return _void(v, f"B={B} {tag}: the first draw's census is missing")
        off, on = data["census"][(B, "off")], data["census"][(B, "on")]
        if off["k19"]:
            return _void(v, f"B={B} off: K19 ran {off['k19']:g} times per step (leaked into the control)")
        if abs(on["k19"] - EXPERT_CALLS) > 0.5:
            return _void(v, f"B={B} on: K19 ran {on['k19']:g} times per step, not {EXPERT_CALLS} (not engaged)")
        if abs((off["gemv"] - on["gemv"]) - EXPERT_CALLS) > 0.5:
            return _void(v, f"B={B}: the GEMV ran {off['gemv']:g} (off) vs {on['gemv']:g} (on) per step; the experts' "
                            f"{EXPERT_CALLS} did not all leave it")
    for tag in ("off", "on"):
        r = data["k8"].get(tag)
        if not isinstance(r, dict) or "mean_nll" not in r:
            return _void(v, f"K8 {tag}: missing or failed")
        if not str(r.get("text_sha", "")).startswith(WINDOW) or int(r.get("tokens_scored", -1)) != STEPS:
            return _void(v, f"K8 {tag}: not the registered window/steps")
    k_off, k_on = float(data["k8"]["off"]["mean_nll"]), float(data["k8"]["on"]["mean_nll"])
    if k_on == k_off:
        return _void(v, f"K8 on equals K8 off to the bit ({k_on!r}): K19 changes the arithmetic, so it did not run")
    for B in BATCHES:
        off, on = statistics.median(data["speed"][(B, "off")]), statistics.median(data["speed"][(B, "on")])
        v["speed"][B] = {"off_ms": off, "on_ms": on, "ratio": on / off}
    dk8 = k_on - k_off
    b = data["k8"].get("build")
    v["k8"] = {"off": k_off, "on": k_on, "delta_nats": dk8,
               "build": (float(b["mean_nll"]) if isinstance(b, dict) and "mean_nll" in b else None)}
    v["k8"]["build_equals_off"] = v["k8"]["build"] == k_off if v["k8"]["build"] is not None else None
    r1 = v["speed"][1]["ratio"]
    v["b1"] = "FASTER" if r1 <= B1_FASTER else ("SLOWER" if r1 >= B1_SLOWER else "NEUTRAL")
    r16 = v["speed"][16]["ratio"]
    if abs(dk8) > K8_FLOOR:
        v["verdict"] = "QUALITY_FAIL"
        v["reasons"].append(f"K8 moved {dk8:+.5f} nats (> {K8_FLOOR}): K19 stays opt-in")
    elif r16 <= B16_FASTER:
        v["verdict"] = "LICENSED"
        v["reasons"].append(f"B=16 {v['speed'][16]['on_ms']:.3f} vs {v['speed'][16]['off_ms']:.3f} ms (x{r16:.3f}); "
                            f"K8 {dk8:+.5f} nats; B=1 {v['b1']} (x{r1:.3f})")
    else:
        v["verdict"] = "NOT_FASTER"
        v["reasons"].append(f"B=16 ratio {r16:.3f} > {B16_FASTER}; K8 {dk8:+.5f} nats; B=1 {v['b1']} (x{r1:.3f})")
    return v


def _synthetic(off16=12.0, on16=10.0, off1=4.2, on1=4.2, k8_off=1.85, k8_on=1.851, k19_on=96.0, k19_off=0.0,
               gemv_off=(96.0, 192.0), gemv_on=(0.0, 96.0), premise=0):
    k8 = lambda x: {"mean_nll": x, "steps": STEPS, "tokens_scored": STEPS, "text_sha": WINDOW + "f" * 52}  # noqa: E731
    cen = {}
    for i, B in enumerate(BATCHES):
        cen[(B, "off")] = {"replays": 8, "k19": k19_off, "gemv": gemv_off[i]}
        cen[(B, "on")] = {"replays": 8, "k19": k19_on, "gemv": gemv_on[i]}
    return {"premise": {"rc": premise, "last": ["2 passed"]},
            "speed": {(16, "off"): [off16, off16], (16, "on"): [on16, on16], (1, "off"): [off1, off1], (1, "on"): [on1, on1]},
            "census": cen, "k8": {"build": k8(k8_off), "off": k8(k8_off), "on": k8(k8_on)}}


def self_test() -> None:
    got = reduce(_synthetic())
    assert got["verdict"] == "LICENSED" and got["b1"] == "NEUTRAL" and got["k8"]["build_equals_off"] is True, got
    assert reduce(_synthetic(on1=4.5))["b1"] == "SLOWER"
    assert reduce(_synthetic(on1=4.0))["b1"] == "FASTER"
    assert reduce(_synthetic(on16=11.6))["verdict"] == "NOT_FASTER"                         # 0.967 > 0.95
    assert reduce(_synthetic(k8_on=1.85 + 0.0096))["verdict"] == "QUALITY_FAIL"
    assert reduce(_synthetic(k8_on=1.85 - 0.0096, on16=9.0))["verdict"] == "QUALITY_FAIL"   # either sign, whatever the speed
    assert reduce(_synthetic(k8_on=1.85))["verdict"] == "VOID"                              # bit-equal: K19 did not run
    assert reduce(_synthetic(premise=1))["verdict"] == "VOID"
    s = _synthetic()
    s["premise"] = None
    assert reduce(s)["verdict"] == "VOID"
    assert reduce(_synthetic(k19_on=48.0))["verdict"] == "VOID"                             # not every projection
    assert reduce(_synthetic(k19_off=96.0))["verdict"] == "VOID"                            # leaked into the control
    assert reduce(_synthetic(gemv_on=(0.0, 192.0)))["verdict"] == "VOID"                    # B=1 experts still on the GEMV
    s = _synthetic()
    s["speed"][(16, "on")] = [10.0, 10.5]                                                    # 5 % apart
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    s["speed"][(1, "off")] = [4.2, None]
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    s["k8"]["on"]["text_sha"] = "0" * 64
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    del s["census"][(16, "off")]
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    s["k8"]["build"] = None                                                                  # the build is reported, not gated
    got = reduce(s)
    assert got["verdict"] == "LICENSED" and got["k8"]["build_equals_off"] is None, got
    print("p87_reduce self-test OK (17 cases)")


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
    print(f"P87_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    for B, s in v.get("speed", {}).items():
        print(f"  B={B}: off {s['off_ms']:.3f} ms, on {s['on_ms']:.3f} ms, ratio {s['ratio']:.3f}")
    if "k8" in v:
        k = v["k8"]
        print(f"  K8: off {k['off']!r}, on {k['on']!r}, delta {k['delta_nats']:+.5f} nats; build {k['build']!r} "
              f"(equals off: {k['build_equals_off']}); B=1 {v['b1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
