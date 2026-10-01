#!/usr/bin/env python3
"""Lane P90's reducer (bench/p90/PREREG-p90.md; e4b#564). Is K21 on gpt-oss-20b's native MXFP4 store
(E4B_MXFP4_GROUPED_SMALLM=1) faster on an RTX 5090 at B=16 without moving the store's quality?

Reads, from the run directory:
- premise: premise.json, the on-box run of tests/test_k21_row_exact_gpu.py (K21's rows bit-equal alone and inside a
           B=16 step through gpt-oss's epilogue, which is what lets the T == 1 KL stand for the B=16 rows)
- K0:      k0.json, the KL instrument's own controls on this host
- speed:   e4b_b{16,1}_{off,on}{,_r2}.json (`step_ms_clean`, the graph-replay window), and the first draws' replay
           censuses logs/census_b{16,1}_{off,on}.txt (P42's table, parsed by bench/p42/p42_reduce.py's `parse_census`)
- quality: gptoss_kl_{off,on}.json, P44's KL-from-reference rows for arm store_r12 (`kl_mean`, `top1_agreement`,
           `n_tokens_scored`, `scorer`), the reference being the dequant of the same shipped MXFP4 bytes

  VOID          the premise did not hold, did not run or skipped a test; K0 did not pass; an arm is missing or failed;
                a census is missing; two draws of one arm differ by more than 3 %; the routes are not the registered
                ones (per decode step: OFF B=16 = 48 NF4 grouped GEMMs and no K21, ON B=16 = 48 K21 calls and no NF4
                grouped GEMM; OFF B=1 = 48 MXFP4 GEMVs and no K21, ON B=1 = 48 K21 calls and no GEMV); a KL row is
                missing, not decode-shaped, or the two rows scored different token counts; OFF's KL does not reproduce
                P44's licensed store row (0.0019 nats) within 0.001
  QUALITY_FAIL  KL(on) - KL(off) > 0.0005 nats (the reference's own decode-vs-prefill floor, P44), or top-1 agreement
                drops by more than 0.002: K21 stays opt-in whatever its speed
  LICENSED      quality holds and B=16 is FASTER (median ON / median OFF <= 0.90)
  NOT_FASTER    quality holds and B=16 is not faster
Reported beside the verdict: B=1's ratio, read as FASTER (<= 0.97), SLOWER (>= 1.03) or NEUTRAL. The default a
LICENSED read proposes depends on it: all decode rows, or only rows above T == 1.

    python p90_reduce.py --dir <dir> --out verdict.json
    python p90_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

B16_FASTER = 0.90
B1_FASTER, B1_SLOWER = 0.97, 1.03
DRAW_SPREAD = 0.03
LAYER_CALLS = 48                    # per decode step: 24 layers x (gate_up + down)
K21_KERNEL = "_gemm_mxfp4_grouped_smallm"
NF4_KERNEL = "_gemm_nf4_grouped"
GEMV_KERNEL = "_gemv_mxfp4_b32"
P44_STORE_KL = 0.0019               # e4b.serve.p44.gptoss.store-r12.kl-vs-bf16.2026-09-19
INSTRUMENT_BAND = 0.001
KL_FLOOR = 0.0005                   # P44's control (i): the reference against itself, decode vs prefill
TOP1_DROP = 0.002
PREMISE_PASSED = "3 passed"
BATCHES = (16, 1)
# per (B, tag): kernel -> calls per step the registered route makes (0 = must be absent)
ROUTES = {(16, "off"): {NF4_KERNEL: LAYER_CALLS, K21_KERNEL: 0}, (16, "on"): {K21_KERNEL: LAYER_CALLS, NF4_KERNEL: 0},
          (1, "off"): {GEMV_KERNEL: LAYER_CALLS, K21_KERNEL: 0}, (1, "on"): {K21_KERNEL: LAYER_CALLS, GEMV_KERNEL: 0}}


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


def _kl_row(rec):
    if not isinstance(rec, dict):
        return None
    for r in rec.get("rows", []):
        if r.get("arm") == "store_r12":
            return r
    return None


def load(d: Path) -> dict:
    p42 = _p42()
    k0 = _json(d / "k0.json")
    data = {"premise": _json(d / "premise.json"), "k0": bool(isinstance(k0, dict) and k0.get("all_passed")),
            "speed": {}, "census": {}, "kl": {}}
    for B in BATCHES:
        for tag in ("off", "on"):
            data["speed"][(B, tag)] = [_step(d, f"e4b_b{B}_{tag}.json"), _step(d, f"e4b_b{B}_{tag}_r2.json")]
            c = d / "logs" / f"census_b{B}_{tag}.txt"
            if c.is_file():
                replays, rows = p42.parse_census(c.read_text())
                per = {}
                for r in rows:
                    per[r["name"].strip()] = per.get(r["name"].strip(), 0) + r["calls"]
                data["census"][(B, tag)] = {"replays": replays,
                                            "per_step": {n: c_ / max(replays or 1, 1) for n, c_ in per.items()}}
    for tag in ("off", "on"):
        data["kl"][tag] = _kl_row(_json(d / f"gptoss_kl_{tag}.json"))
    return data


def _void(v, why):
    v["verdict"] = "VOID"
    v["reasons"].append(why)
    return v


def reduce(data: dict) -> dict:
    v: dict = {"lane": "P90", "verdict": None, "reasons": [], "speed": {}}
    pr = data.get("premise")
    last = " ".join(pr.get("last") or []) if isinstance(pr, dict) else ""
    if not isinstance(pr, dict) or pr.get("rc") != 0 or PREMISE_PASSED not in last or "skipped" in last:
        return _void(v, f"the on-box premise (K21 rows bit-equal alone and inside B=16, {PREMISE_PASSED}) did not hold, "
                        f"did not run, or skipped a test: {pr}")
    if not data.get("k0"):
        return _void(v, "the KL instrument's K0 controls did not pass on this host")
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
            for kern, want in ROUTES[(B, tag)].items():
                got = c["per_step"].get(kern, 0)
                if abs(got - want) > 0.5:
                    return _void(v, f"B={B} {tag}: {kern} ran {got:g} times per step, not {want} -- not the registered route")
    off, on = data["kl"].get("off"), data["kl"].get("on")
    for tag, r in (("off", off), ("on", on)):
        if not isinstance(r, dict) or "kl_mean" not in r:
            return _void(v, f"KL {tag}: the store_r12 row is missing or failed")
        if r.get("scorer") != "decode":
            return _void(v, f"KL {tag}: scored {r.get('scorer')!r}, not decode-shaped -- at T == 1 is where the opt-in "
                            f"routes K21, so another shape would not read the kernel")
    if off.get("n_tokens_scored") != on.get("n_tokens_scored") or not off.get("n_tokens_scored"):
        return _void(v, f"KL: the rows scored different token counts ({off.get('n_tokens_scored')} vs "
                        f"{on.get('n_tokens_scored')})")
    k_off, k_on = float(off["kl_mean"]), float(on["kl_mean"])
    if abs(k_off - P44_STORE_KL) > INSTRUMENT_BAND:
        return _void(v, f"KL off {k_off:.5f} does not reproduce P44's licensed store row {P44_STORE_KL} within "
                        f"{INSTRUMENT_BAND}: the instrument is not reading as registered")
    for B in BATCHES:
        m_off, m_on = statistics.median(data["speed"][(B, "off")]), statistics.median(data["speed"][(B, "on")])
        v["speed"][B] = {"off_ms": m_off, "on_ms": m_on, "ratio": m_on / m_off}
    t_off, t_on = float(off.get("top1_agreement", 0.0)), float(on.get("top1_agreement", 0.0))
    v["kl"] = {"off": k_off, "on": k_on, "delta_nats": k_on - k_off, "top1_off": t_off, "top1_on": t_on,
               "tokens": off.get("n_tokens_scored")}
    r1 = v["speed"][1]["ratio"]
    v["b1"] = "FASTER" if r1 <= B1_FASTER else ("SLOWER" if r1 >= B1_SLOWER else "NEUTRAL")
    r16 = v["speed"][16]["ratio"]
    if k_on - k_off > KL_FLOOR or t_on < t_off - TOP1_DROP:
        v["verdict"] = "QUALITY_FAIL"
        v["reasons"].append(f"KL {k_off:.5f} -> {k_on:.5f} ({k_on - k_off:+.5f} nats; floor {KL_FLOOR}), top-1 "
                            f"{t_off:.4f} -> {t_on:.4f}: K21 stays opt-in")
    elif r16 <= B16_FASTER:
        v["verdict"] = "LICENSED"
        v["reasons"].append(f"B=16 {v['speed'][16]['on_ms']:.3f} vs {v['speed'][16]['off_ms']:.3f} ms (x{r16:.3f}); "
                            f"KL {k_off:.5f} -> {k_on:.5f}; B=1 {v['b1']} (x{r1:.3f})")
    else:
        v["verdict"] = "NOT_FASTER"
        v["reasons"].append(f"B=16 ratio {r16:.3f} > {B16_FASTER}; KL {k_off:.5f} -> {k_on:.5f}; B=1 {v['b1']} (x{r1:.3f})")
    return v


def _synthetic(off16=22.5, on16=14.0, off1=5.35, on1=5.6, kl_off=0.0019, kl_on=0.0018, top1_off=0.9814,
               top1_on=0.9816, premise=0, last="3 passed in 40.0s", k0=True, scorer="decode", tokens=(60000, 60000)):
    cen = {}
    for (B, tag), want in ROUTES.items():
        cen[(B, tag)] = {"replays": 8, "per_step": {k: float(n) for k, n in want.items() if n}}
    row = lambda k, t, s, n: {"arm": "store_r12", "kl_mean": k, "top1_agreement": t, "scorer": s, "n_tokens_scored": n}  # noqa: E731
    return {"premise": {"rc": premise, "last": [last]}, "k0": k0,
            "speed": {(16, "off"): [off16, off16], (16, "on"): [on16, on16], (1, "off"): [off1, off1], (1, "on"): [on1, on1]},
            "census": cen, "kl": {"off": row(kl_off, top1_off, "decode", tokens[0]), "on": row(kl_on, top1_on, scorer, tokens[1])}}


def self_test() -> None:
    got = reduce(_synthetic())
    assert got["verdict"] == "LICENSED" and got["b1"] == "SLOWER", got                      # 5.6 / 5.35 = 1.047
    assert reduce(_synthetic(on1=5.35))["b1"] == "NEUTRAL"
    assert reduce(_synthetic(on16=20.5))["verdict"] == "NOT_FASTER"                         # 0.911 > 0.90
    assert reduce(_synthetic(kl_on=0.0019 + 0.0006))["verdict"] == "QUALITY_FAIL"
    assert reduce(_synthetic(kl_on=0.0024, on16=10.0))["verdict"] == "LICENSED"             # +0.0005 is the floor itself
    assert reduce(_synthetic(top1_on=0.9814 - 0.0021, on16=10.0))["verdict"] == "QUALITY_FAIL"
    assert reduce(_synthetic(premise=1))["verdict"] == "VOID"
    assert reduce(_synthetic(last="1 passed, 2 skipped in 3.0s"))["verdict"] == "VOID"
    assert reduce(_synthetic(k0=False))["verdict"] == "VOID"
    assert reduce(_synthetic(scorer="prefill"))["verdict"] == "VOID"
    assert reduce(_synthetic(tokens=(60000, 59000)))["verdict"] == "VOID"
    assert reduce(_synthetic(kl_off=0.0031, kl_on=0.0031))["verdict"] == "VOID"              # instrument off its row
    s = _synthetic()
    s["census"][(16, "on")]["per_step"][NF4_KERNEL] = 48.0                                   # ON still on NF4
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    s["census"][(1, "on")]["per_step"][K21_KERNEL] = 0.0                                     # T == 1 never reached K21
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    s["census"][(16, "off")]["per_step"][K21_KERNEL] = 48.0                                  # leaked into the control
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    s["speed"][(16, "on")] = [14.0, 14.7]                                                    # 5 % apart
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    s["kl"]["on"] = None
    assert reduce(s)["verdict"] == "VOID"
    s = _synthetic()
    del s["census"][(1, "off")]
    assert reduce(s)["verdict"] == "VOID"
    print("p90_reduce self-test OK (18 cases)")


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
    print(f"P90_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    for B, s in v.get("speed", {}).items():
        print(f"  B={B}: off {s['off_ms']:.3f} ms, on {s['on_ms']:.3f} ms, ratio {s['ratio']:.3f}")
    if "kl" in v:
        k = v["kl"]
        print(f"  KL: off {k['off']:.6f}, on {k['on']:.6f}, delta {k['delta_nats']:+.6f} nats; top-1 {k['top1_off']:.4f} -> "
              f"{k['top1_on']:.4f}; tokens {k['tokens']}; B=1 {v['b1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
