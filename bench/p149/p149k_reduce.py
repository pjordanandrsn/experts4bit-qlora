#!/usr/bin/env python3
"""p149k_reduce.py -- lane P149's box K rule (bench/p149/PREREG-p149.md), over ``p149k.json``.

A microbench. A saving here is a PREMISE for the served read, never the quoted gain: K_GO lets the kernels move into
grouped-nf4-gemm and the quality-and-speed box be created; KERNEL_NO_GAIN is a reading, not a strike.

The rule, first rung that applies (every bar compares unrounded values):
  VOID   the record is missing or not ok; not a 170-SM RTX 5090; a shape's contract (at M == 1 or M == 16), selection
         or read missing; a choice that is not the fastest contract-passing config of its own selection; at M == 1 a
         shape's cuBLAS medians (A1, A2) outside [0.8, 1.25] x P145's in-situ time for its 64 calls (qo 1.4648, kv
         0.5675 ms): the harness does not transfer (that anchor was read at 450 W); at M == 1 a shape's profiled cuBLAS
         call launching no kernel on P145's in-situ grid (qo 1024, kv 256; x 1 x 1).
  NOISY  (a VOID, and a strike) any shape's M == 1 self-pair, A2/A1 or B2/B1, outside [0.985, 1.015].
  KERNEL_NO_GAIN  a shape's int8 kernel fails its contract at M == 1 or M == 16 (its error against its dequantized
         weights above 1.05 x cuBLAS's against the bf16 weights: no config passes), or the W1 projected in-situ saving,
         summed over qo and kv, is under BAR = 2 % of P145's W1 node period (8.125 ms): 0.1625 ms. Each shape's
         projection is the SMALLER of its two pairs', P145's in-situ time x (1 - B1/A1) and x (1 - B2/A2).
  K_GO   otherwise.

Reported, never ruling: every config's error and selection time; the M == 16 reads, pairs and self-pairs; each shape's
bandwidth (bytes / time); the raw harness saving in ms; the GPU state (power limits, clocks, temperature) at the start,
the end and around every read; each shape's chosen config at M == 1 and M == 16 (the engine's pinned table after K_GO).

    python p149k_reduce.py --bench p149k.json --out verdict_p149k.json [--e4b-sha SHA]
    python p149k_reduce.py --self-test
"""
import argparse
import json
import sys

P_MS = 8.125                          # P145's W1 node period
BAR_SHARE = 0.02
BAR = round(BAR_SHARE * P_MS, 6)
ANCHOR_MS = {"qo": 1.4648, "kv": 0.5675}   # P145's in-situ time for each shape's 64 calls (cuBLAS `kernel`, dur_ms)
ANCHOR = (0.8, 1.25)
GRID_P145 = {"qo": [1024, 1, 1], "kv": [256, 1, 1]}   # P145's cuBLAS `kernel` grids at one request, in situ
NOISY_BAND = (0.985, 1.015)
CONTRACT_RATIO = 1.05
SMS = 170
SHAPES = {"qo": (4096, 4096, 64), "kv": (1024, 4096, 64)}
MS = ("1", "16")
READ_TAGS = ("A1", "B1", "B2", "A2")


def _bytes(s, int8=False):
    n, k, layers = SHAPES[s]
    return n * k * layers * (1 if int8 else 2) + (n * layers * 4 if int8 else 0)


def _passing(con):
    return {c for c, e in con["configs"].items() if e <= CONTRACT_RATIO * con["cublas"]}


def reduce(rec: dict) -> dict:
    out = {"lane": "P149", "box": "K", "verdict": None, "reasons": [],
           "bars": {"bar_ms": BAR, "bar_share_of_P": BAR_SHARE, "P_ms": P_MS, "anchor": list(ANCHOR),
                    "noisy_band": list(NOISY_BAND), "contract_ratio": CONTRACT_RATIO},
           "note": "a microbench premise: K_GO creates the quality-and-speed box, never the quoted gain"}
    if not rec or rec.get("status") != "ok":
        out.update(verdict="VOID", reasons=["the record is missing or not ok"])
        return out
    void = []
    dev = rec.get("device") or {}
    if dev.get("sm_count") != SMS or "5090" not in str(dev.get("name")):
        void.append(f"device {dev}: registered a {SMS}-SM RTX 5090")
    con, sel, cho, rd = (rec.get(k) or {} for k in ("contract", "select", "choice", "reads"))
    kern = rec.get("cublas_kernels") or {}
    for s in SHAPES:
        grids = [list(k.get("grid") or []) for k in (kern.get(s) or {}).get("1") or []]
        if GRID_P145[s] not in grids:
            void.append(f"{s}: the harness's M == 1 cuBLAS call launched grids {grids}, not P145's in-situ {GRID_P145[s]}")
    fails = []
    for s in SHAPES:
        for m in MS:
            c = (con.get(s) or {}).get(m)
            if not c:
                void.append(f"{s} M={m}: no contract record")
                continue
            passing = _passing(c)
            if not passing:
                fails.append(f"{s} M={m}: no config passes the contract (error <= {CONTRACT_RATIO} x cuBLAS's)")
                continue
            r = (rd.get(s) or {}).get(m)
            if not r:
                void.append(f"{s} M={m}: a config passes the contract but there is no read")
                continue
            sm = (sel.get(s) or {}).get(m) or {}
            if set(sm) != passing:
                void.append(f"{s} M={m}: the selection timed {sorted(sm)}, not the passing set {sorted(passing)}")
            elif (cho.get(s) or {}).get(m) != min(sm, key=sm.get):
                void.append(f"{s} M={m}: the choice {(cho.get(s) or {}).get(m)} is not the fastest passing config")
            if any(not isinstance(r.get(t), (int, float)) or r[t] <= 0 for t in READ_TAGS):
                void.append(f"{s} M={m}: read incomplete")
                continue
            if m == "1":
                for t in ("A1", "A2"):
                    if not ANCHOR[0] <= r[t] / ANCHOR_MS[s] <= ANCHOR[1]:
                        void.append(f"{s}: cuBLAS {t} {r[t]:.4f} ms is {r[t] / ANCHOR_MS[s]:.3f}x P145's {ANCHOR_MS[s]} ms "
                                    "(the anchor was read at 450 W)")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    noisy, shapes = [], {}
    for s in SHAPES:
        shapes[s] = {}
        for m in MS:
            r = (rd.get(s) or {}).get(m)
            if not r:
                continue
            sp = {"A2/A1": r["A2"] / r["A1"], "B2/B1": r["B2"] / r["B1"]}
            if m == "1":
                noisy += [f"{s} M=1 self-pair {k} = {v:.4f} outside {list(NOISY_BAND)}" for k, v in sp.items()
                          if not NOISY_BAND[0] <= v <= NOISY_BAND[1]]
            a, b = (r["A1"] + r["A2"]) / 2, (r["B1"] + r["B2"]) / 2
            row = {"choice": cho[s][m], "harness_save_ms": min(r["A1"] - r["B1"], r["A2"] - r["B2"]),
                   "pairs_ms": [r["A1"] - r["B1"], r["A2"] - r["B2"]], "self_pairs": {k: round(v, 4) for k, v in sp.items()},
                   "cublas_tbs": round(_bytes(s) / (a * 1e-3) / 1e12, 3), "kernel_tbs": round(_bytes(s, int8=True) / (b * 1e-3) / 1e12, 3),
                   "cublas_err": con[s][m]["cublas"], "kernel_err": con[s][m]["configs"][cho[s][m]],
                   "gpu_before": r.get("gpu_before"), "gpu_after": r.get("gpu_after")}
            if m == "1":
                row.update(projected_save_ms=min(ANCHOR_MS[s] * (1 - r["B1"] / r["A1"]), ANCHOR_MS[s] * (1 - r["B2"] / r["A2"])),
                           anchor_ms=ANCHOR_MS[s])
            shapes[s][m] = row
    out["shapes"] = shapes
    out["gpu_state"] = rec.get("gpu_state")
    out["pinned_choices"] = {s: {m: (cho.get(s) or {}).get(m) for m in MS} for s in SHAPES}
    if noisy:
        out.update(verdict="NOISY", reasons=noisy + ["NOISY is a VOID and a strike: no premise is read"])
        return out
    if fails:
        out.update(verdict="KERNEL_NO_GAIN", reasons=fails + ["a reading, not a strike: no quality-and-speed box"])
        return out
    w1 = sum(shapes[s]["1"]["projected_save_ms"] for s in SHAPES if "1" in shapes[s])
    out["w1"] = {"projected_save_ms": w1, "share_of_P": w1 / P_MS,
                 "harness_save_ms": sum(shapes[s]["1"]["harness_save_ms"] for s in SHAPES)}
    if w1 >= BAR:
        out.update(verdict="K_GO", reasons=[f"W1 projected in situ {w1:.6f} ms >= bar {BAR} (harness "
                                           f"{out['w1']['harness_save_ms']:.6f} ms, reported)"])
    else:
        out.update(verdict="KERNEL_NO_GAIN", reasons=[f"W1 projected in situ {w1:.6f} ms < bar {BAR}: a reading, not a strike"])
    return out


# ------------------------------------------------------------------ self-test --
def _fake(*, sms=170, name="NVIDIA GeForce RTX 5090", saves=(0.12, 0.06), noise=0.0, noise16=0.0, cublas_x=1.0,
          err=1.0, err16=1.0, drop=None, choice_slow=False, grid_off=False):
    con, sel, cho, rd = {}, {}, {}, {}
    for s, save in zip(("qo", "kv"), saves):
        con[s], sel[s], cho[s], rd[s] = {}, {}, {}, {}
        for m in MS:
            e = err if m == "1" else err16
            cfgs = {"8,1024,4,1": e, "16,2048,8,2": e * 1.01} if m == "1" else {"64,128,4,1": e, "128,256,8,4": e * 1.01}
            con[s][m] = {"cublas": 1.0, "configs": cfgs}
            passing = {c: 0.5 + i * 0.01 for i, c in enumerate(c for c, ee in cfgs.items() if ee <= 1.05)}
            if passing:
                sel[s][m] = passing
                cho[s][m] = (max if choice_slow else min)(passing, key=passing.get)
                a = ANCHOR_MS[s] * cublas_x
                nz = noise if m == "1" else noise16
                rd[s][m] = {"A1": a, "B1": a - save, "B2": (a - save) * (1 + nz), "A2": a}
    rec = {"status": "ok", "device": {"name": name, "sm_count": sms}, "contract": con, "select": sel, "choice": cho, "reads": rd,
           "cublas_kernels": {s: {"1": [{"name": "kernel", "grid": list(GRID_P145[s])}], "16": [{"name": "k2", "grid": [8, 8, 1]}]}
                              for s in SHAPES}}
    if grid_off:
        rec["cublas_kernels"]["qo"]["1"] = [{"name": "gemvx_kernel", "grid": [512, 1, 1]}]
    if drop:
        rec["reads"][drop[0]].pop(drop[1], None)
    return rec


def self_test() -> int:
    def v(**k):
        try:
            return reduce(_fake(**k))["verdict"]
        except Exception as e:  # noqa: BLE001 - a crash is a failed case, never a pass
            return f"EXCEPTION {type(e).__name__}"
    cases = [
        ("a clear saving is K_GO", v() == "K_GO"),
        ("W1 sums its two shapes: 0.10 + 0.0625 = 0.1625 >= 0.1625", v(saves=(0.10, 0.0625)) == "K_GO"),
        ("just under the bar is KERNEL_NO_GAIN", v(saves=(0.10, 0.0624)) == "KERNEL_NO_GAIN"),
        ("the bar reads the unrounded save (0.16249999 < 0.1625)", v(saves=(0.10, 0.06249999)) == "KERNEL_NO_GAIN"),
        ("a shape's saving is the SMALLER pair's projection",
         (lambda r: r["shapes"]["qo"]["1"]["projected_save_ms"] < 0.115)(reduce(_fake(noise=0.01)))),
        ("an M == 1 contract failure is KERNEL_NO_GAIN, not a strike", v(err=1.06) == "KERNEL_NO_GAIN"),
        ("an M == 16 contract failure is KERNEL_NO_GAIN", v(err16=1.2) == "KERNEL_NO_GAIN"),
        ("a 1.6 % M == 1 self-pair is NOISY", v(noise=0.016) == "NOISY"),
        ("a 1.4 % M == 1 self-pair is not (0.15 + 0.08 project to 0.205 through the slower pair)",
         v(noise=0.014, saves=(0.15, 0.08)) == "K_GO"),
        ("an M == 16 self-pair is reported, never NOISY", v(noise16=0.05) == "K_GO"),
        ("a card that is not the 5090", v(sms=188) == "VOID"),
        ("the harness does not transfer (cuBLAS 1.3x P145)", v(cublas_x=1.3) == "VOID"),
        ("cuBLAS at 0.85x still transfers", v(cublas_x=0.85) == "K_GO"),
        ("an M == 1 read missing", v(drop=("kv", "1")) == "VOID"),
        ("an M == 16 read missing", v(drop=("qo", "16")) == "VOID"),
        ("a choice that is not the fastest", v(choice_slow=True) == "VOID"),
        ("no record", reduce({})["verdict"] == "VOID"),
        ("a cuBLAS kernel off P145's grid is VOID", v(grid_off=True) == "VOID"),
        ("no profiled kernel is VOID", reduce(dict(_fake(), cublas_kernels={}))["verdict"] == "VOID"),
        ("the bar rules in situ: a harness at 1.2x projects raw 0.12 + 0.06 to 0.15", v(saves=(0.12, 0.06), cublas_x=1.2) == "KERNEL_NO_GAIN"),
        ("the bar rules in situ: a harness at 0.85x projects raw 0.12 + 0.03 to 0.1765", v(saves=(0.12, 0.03), cublas_x=0.85) == "K_GO"),
        ("the pinned choices are reported for both M", (lambda r: set(r["pinned_choices"]["qo"]) == {"1", "16"})(reduce(_fake()))),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p149k_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    print(f"P149K_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
