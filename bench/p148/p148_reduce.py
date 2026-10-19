#!/usr/bin/env python3
"""p148_reduce.py -- lane P148's registered rule (bench/p148/PREREG-p148.md; e4b#1313), over ``p148.json``.

A microbench. A saving here is a PREMISE for kernel work, never the quoted gain: a GO starts a grouped-nf4-gemm kernel
and a served read of its own (and, for D2, a quality gate at both of the split premise's gates before anything flips).

The rule, first rung that applies (every bar compares unrounded values):
  VOID   the record is missing or not ok; not a 170-SM RTX 5090; a shape's contract, selection or read missing; a choice
         that is not the fastest contract-passing config of its own selection; a shape's cuBLAS medians (A1, A2) outside
         [0.8, 1.25] x P142's in-situ time (qkv 0.7146, o 0.5972, head 0.3770 ms): the harness does not transfer.
  NOISY  (a VOID, and a strike) any shape's self-pair, A2/A1 or B2/B1, outside [0.985, 1.015].
  then, per lever, independently:
  D1     GO iff qkv and o each have a contract-passing config (error <= 1.05 x cuBLAS's against fp64) and their savings
         sum to >= 2 % of P142's P (4.380166 ms): BAR = 0.087603 ms. A shape's saving is the SMALLER of its two pairs,
         A1 - B1 and A2 - B2. Otherwise PARK.
  D2     GO iff the int8 head kernel passes its contract (its error against its dequantized weights <= 1.05 x cuBLAS's
         against the bf16 head) and the head's saving is >= BAR. Otherwise PARK.

Reported, never ruling: every config's error and selection time, each shape's bandwidth (bytes / time against the
5090's 1.792 TB/s), and the pairs.

    python p148_reduce.py --bench p148.json --out verdict_p148.json [--e4b-sha SHA]
    python p148_reduce.py --self-test
"""
import argparse
import json
import sys

P_MS = 4.380166                       # P142's B=1 step (pername-5090-1)
BAR_SHARE = 0.02
BAR = round(BAR_SHARE * P_MS, 6)
ANCHOR_MS = {"qkv": 0.7146, "o": 0.5972, "head": 0.3770}   # P142's in-situ exclusive times
ANCHOR = (0.8, 1.25)
NOISY_BAND = (0.985, 1.015)
CONTRACT_RATIO = 1.05
SMS = 170
D1_SHAPES, D2_SHAPES = ("qkv", "o"), ("head",)
SHAPES = {"qkv": (5120, 2048, 48), "o": (2048, 4096, 48), "head": (151936, 2048, 1)}
PEAK_TBS = 1.792


def _bytes(s, int8=False):
    n, k, layers = SHAPES[s]
    return n * k * layers * (1 if int8 else 2)


def reduce(rec: dict) -> dict:
    out = {"lane": "P148", "verdict": None, "levers": {}, "reasons": [],
           "bars": {"bar_ms": BAR, "bar_share_of_P": BAR_SHARE, "P_ms": P_MS, "anchor": list(ANCHOR),
                    "noisy_band": list(NOISY_BAND), "contract_ratio": CONTRACT_RATIO},
           "note": "a microbench premise: a GO starts kernel work, never the quoted gain"}
    if not rec or rec.get("status") != "ok":
        out.update(verdict="VOID", reasons=["the record is missing or not ok"])
        return out
    void = []
    dev = rec.get("device") or {}
    if dev.get("sm_count") != SMS or "5090" not in str(dev.get("name")):
        void.append(f"device {dev}: registered a {SMS}-SM RTX 5090")
    con, sel, cho, rd = (rec.get(k) or {} for k in ("contract", "select", "choice", "reads"))
    for s in SHAPES:
        if s not in con:
            void.append(f"{s}: no contract record")
            continue
        passing = {c for c, e in con[s]["configs"].items() if e <= CONTRACT_RATIO * con[s]["cublas"]}
        if passing and s not in rd:
            void.append(f"{s}: a config passes the contract but there is no read")
            continue
        if not passing:
            continue
        if set(sel.get(s) or {}) != passing:
            void.append(f"{s}: the selection timed {sorted(sel.get(s) or {})}, not the passing set {sorted(passing)}")
        elif cho.get(s) != min(sel[s], key=sel[s].get):
            void.append(f"{s}: the choice {cho.get(s)} is not the fastest passing config")
        r = rd[s]
        if any(not isinstance(r.get(t), (int, float)) or r[t] <= 0 for t in ("A1", "B1", "B2", "A2")):
            void.append(f"{s}: read {r} incomplete")
            continue
        for t in ("A1", "A2"):
            if not ANCHOR[0] <= r[t] / ANCHOR_MS[s] <= ANCHOR[1]:
                void.append(f"{s}: cuBLAS {t} {r[t]:.4f} ms is {r[t] / ANCHOR_MS[s]:.3f}x P142's {ANCHOR_MS[s]} ms")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    noisy = []
    shapes = {}
    for s in SHAPES:
        if s not in rd:
            continue
        r = rd[s]
        sp = {"A2/A1": r["A2"] / r["A1"], "B2/B1": r["B2"] / r["B1"]}
        noisy += [f"{s} self-pair {k} = {v:.4f} outside {list(NOISY_BAND)}" for k, v in sp.items()
                  if not NOISY_BAND[0] <= v <= NOISY_BAND[1]]
        save = min(r["A1"] - r["B1"], r["A2"] - r["B2"])
        a, b = (r["A1"] + r["A2"]) / 2, (r["B1"] + r["B2"]) / 2
        shapes[s] = {"choice": cho[s], "save_ms": save, "pairs_ms": [r["A1"] - r["B1"], r["A2"] - r["B2"]],
                     "self_pairs": {k: round(v, 4) for k, v in sp.items()},
                     "cublas_tbs": round(_bytes(s) / (a * 1e-3) / 1e12, 3),
                     "kernel_tbs": round(_bytes(s, int8=s in D2_SHAPES) / (b * 1e-3) / 1e12, 3),
                     "cublas_err": con[s]["cublas"], "kernel_err": con[s]["configs"][cho[s]]}
    out["shapes"] = shapes
    if noisy:
        out.update(verdict="NOISY", reasons=noisy + ["NOISY is a VOID and a strike: no lever is read"])
        return out
    if all(s in shapes for s in D1_SHAPES):
        d1 = sum(shapes[s]["save_ms"] for s in D1_SHAPES)
        out["levers"]["D1"] = {"save_ms": d1, "share_of_P": d1 / P_MS, "verdict": "GO" if d1 >= BAR else "PARK"}
    else:
        out["levers"]["D1"] = {"verdict": "PARK", "why": "no config passes the contract at "
                               + ", ".join(s for s in D1_SHAPES if s not in shapes)}
    if "head" in shapes:
        d2 = shapes["head"]["save_ms"]
        out["levers"]["D2"] = {"save_ms": d2, "share_of_P": d2 / P_MS, "verdict": "GO" if d2 >= BAR else "PARK"}
    else:
        out["levers"]["D2"] = {"verdict": "PARK", "why": "the int8 head kernel fails its contract"}
    out["verdict"] = f"D1_{out['levers']['D1']['verdict']}+D2_{out['levers']['D2']['verdict']}"
    out["reasons"] = [f"{k}: {v.get('verdict')} ({v.get('save_ms', 0):.6f} ms vs bar {BAR})" for k, v in out["levers"].items()]
    return out


# ------------------------------------------------------------------ self-test --
def _fake(*, sms=170, name="NVIDIA GeForce RTX 5090", saves=(0.06, 0.06, 0.10), noise=0.0, cublas_x=1.0, err_cfg=1.0,
          drop=None, choice_slow=False, i8_err=1.0):
    con, sel, cho, rd = {}, {}, {}, {}
    for s, save in zip(("qkv", "o", "head"), saves):
        cfgs = {"4,512,4,1": err_cfg, "8,1024,8,2": err_cfg * 1.01} if s != "head" else {"8,1024,4,1": i8_err}
        con[s] = {"cublas": 1.0, "configs": cfgs}
        passing = {c: 0.5 + i * 0.01 for i, c in enumerate(c for c, e in cfgs.items() if e <= 1.05)}
        if passing:
            sel[s] = passing
            cho[s] = (max if choice_slow else min)(passing, key=passing.get)
            a = ANCHOR_MS[s] * cublas_x
            rd[s] = {"A1": a, "B1": a - save, "B2": (a - save) * (1 + noise), "A2": a}
    rec = {"status": "ok", "device": {"name": name, "sm_count": sms}, "contract": con, "select": sel, "choice": cho, "reads": rd}
    if drop:
        rec["reads"].pop(drop, None)
    return rec


def self_test() -> int:
    v = lambda **k: reduce(_fake(**k))["verdict"]  # noqa: E731
    cases = [
        ("both levers clear", v() == "D1_GO+D2_GO"),
        ("D1 sums its two shapes: 0.044 + 0.044 = 0.088 >= 0.087603", v(saves=(0.044, 0.044, 0.10)) == "D1_GO+D2_GO"),
        ("D1 just under the bar parks", v(saves=(0.0438, 0.0438, 0.10)) == "D1_PARK+D2_GO"),
        ("the bar reads the unrounded save (0.0876029 < 0.087603)", v(saves=(0.0438, 0.0438029, 0.10)) == "D1_PARK+D2_GO"),
        ("D2 under the bar parks", v(saves=(0.06, 0.06, 0.08)) == "D1_GO+D2_PARK"),
        ("a pair's saving is the SMALLER pair", (lambda r: r["shapes"]["qkv"]["save_ms"] < 0.06)(
            reduce(dict(_fake(), reads={**_fake()["reads"], "qkv": {"A1": 0.7146, "B1": 0.6546, "B2": 0.6600, "A2": 0.7146}})))),
        ("a contract failure at o parks D1", v(err_cfg=1.06) == "D1_PARK+D2_GO"),
        ("an int8 contract failure parks D2", v(i8_err=1.2) == "D1_GO+D2_PARK"),
        ("a 1.6 % self-pair is NOISY", v(noise=0.016) == "NOISY"),
        ("a 1.4 % self-pair is not", v(noise=0.014) == "D1_GO+D2_GO"),
        ("a card that is not the 5090", v(sms=188) == "VOID"),
        ("the harness does not transfer (cuBLAS 1.3x P142)", v(cublas_x=1.3) == "VOID"),
        ("cuBLAS at 0.85x still transfers", v(cublas_x=0.85) == "D1_GO+D2_GO"),
        ("a read missing", v(drop="o") == "VOID"),
        ("a choice that is not the fastest", v(choice_slow=True) == "VOID"),
        ("no record", reduce({})["verdict"] == "VOID"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p148_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha", default="")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    rec = json.load(open(a.bench))
    v = reduce(rec)
    if a.e4b_sha and rec.get("e4b_sha") != a.e4b_sha:
        v.update(verdict="VOID", reasons=[f"the box ran e4b {rec.get('e4b_sha')}, not {a.e4b_sha}"])
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P148_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
