"""bench/dq1/dq1_reduce.py -- lane DQ1's registered rule (bench/dq1/DQ1-PREREG.md, "The rule"). Pure: a receipt in, readings
and verdicts out; no GPU, no clock. ``python dq1_reduce.py receipt.json`` prints the read; ``--self-test`` runs the synthetic
cases; tests/test_dq1_lane.py pins both and kills the rule's mutants.

Definitions (all per decoder layer of the census model, linears only):
  arm time      per (shape, M, arm, phase): the median of each palindrome position's draws; the arm's time is their mean,
                and position 2 / position 1 is its self-pair.
  C             2 x fwd + dgrad: forward, the checkpoint recompute's forward, and the frozen-weight input gradient.
  LC(arm, M)    sum over the layer's seven linears (k/v and gate/up counted twice) of C; LCf / LCd the fwd / dgrad parts.
  H(M)          1 - LC(bf16, M) / LC(bnb, M): the share of bnb's base-linear time a perfect low-bit kernel could remove.
  G1(M)         LC(bnb, M) / LC(gnf4a, M); GF(M) the same for gnf4f (> 1 = gnf4 faster).
  L(M)          LoRA (2 x fwd + bwd, every linear) / LC(bnb, M).
  streaming     one layer's transfer X = layer NF4 bytes / pinned H2D GB/s under GEMM load. Each phase must hide one
                transfer behind the compute of that phase alone: R_fwd = LCf(bnb) / X (the forward), R_bwd = (LCf + LCd)(bnb)
                / X (the recompute + dgrad, which keep the layer resident); Rmin = min of the two. R = LC / 2X is reported.
"""
from __future__ import annotations

import json
import statistics
import sys

ARMS = ("bf16", "bnb", "dq", "gnf4a", "gnf4f")
REGISTERED_SHAPES = {"q_proj": (8192, 5120, 1), "kv_proj": (1024, 5120, 2), "o_proj": (5120, 8192, 1),
                     "gate_up_proj": (25600, 5120, 2), "down_proj": (5120, 25600, 1)}
REGISTERED_ROWS = [512, 1024, 2048, 4096, 8192]
REGISTERED_H2D_ROWS = [1024, 2048, 4096]
REGISTERED_DEVICE = "NVIDIA GeForce RTX 5090"
GRADED_G1_ROWS = (2048, 4096)
SELF_PAIR_BAND = (0.95, 1.05)
NOISY_FRACTION = 0.10
PARITY_MAX = 1e-2
# arm -> (route taken, DGRAD_STATS key that must move, ROUTE_STATS keys that must move)
EXPECT_ROUTE = {"gnf4a": ("dense", "dense", ("dense_fwd", "dense_dgrad")), "gnf4f": ("fused", "kernel", ())}


def _arm_time(series):
    """series: [[draws pos1], [draws pos2]] -> (mean of the two medians, pos2/pos1) or None."""
    if not series or len(series) != 2 or not all(series):
        return None
    m1, m2 = statistics.median(series[0]), statistics.median(series[1])
    return (m1 + m2) / 2.0, (m2 / m1 if m1 > 0 else float("inf"))


def _complete(receipt: dict) -> list:
    """Reasons the receipt is not the registered census, run to its end (a receipt cut short by the alarm is flushed per cell)."""
    why = []
    shapes = {k: tuple(v) for k, v in receipt.get("shapes", {}).items()}
    if shapes != REGISTERED_SHAPES:
        why.append(f"shapes {sorted(shapes)} are not the registered {sorted(REGISTERED_SHAPES)}")
    if list(receipt.get("rows", [])) != REGISTERED_ROWS:
        why.append(f"rows {receipt.get('rows')} are not {REGISTERED_ROWS}")
    have = {(c.get("shape"), c.get("M")) for c in receipt.get("cells", [])}
    missing = [(n, M) for n in REGISTERED_SHAPES for M in REGISTERED_ROWS if (n, M) not in have]
    if missing:
        why.append(f"{len(missing)} registered cells missing, first {missing[0]}")
    if sorted(h.get("M") for h in receipt.get("h2d", [])) != REGISTERED_H2D_ROWS:
        why.append(f"h2d rows {[h.get('M') for h in receipt.get('h2d', [])]} are not {REGISTERED_H2D_ROWS}")
    if not receipt.get("finished_at"):
        why.append("no finished_at: the census did not run to its end")
    return why


def reduce(receipt: dict) -> dict:
    out = {"void": [], "noisy": False, "function_fail": {}, "engagement_fail": {}, "run_fail": {}, "readings": {},
           "verdicts": {}}
    shapes = REGISTERED_SHAPES
    rows = REGISTERED_ROWS
    if receipt.get("rehearsal"):
        out["void"].append("rehearsal receipt: never graded")
    dev = receipt.get("forensics", {}).get("device", "")
    if dev != REGISTERED_DEVICE:
        out["void"].append(f"device {dev!r} is not the registered {REGISTERED_DEVICE!r}")
    out["void"] += _complete(receipt)
    cells = {(c["shape"], c["M"]): c for c in receipt.get("cells", [])}
    pairs_total = pairs_out = 0
    t, tf, td = {}, {}, {}      # (arm, shape, M) -> C, fwd, dgrad
    for (name, M), c in cells.items():
        if name not in shapes:
            continue
        for arm in ARMS:
            rec = c.get("arms", {}).get(arm)
            if rec is None:
                out["void"].append(f"{name} M={M}: arm {arm} missing")
                continue
            if rec.get("errors"):        # could not run here (e.g. resources): a reading, recorded per arm and row
                out["run_fail"].setdefault(arm, []).append((M, f"{name} M={M}: {rec['errors'][0]}"))
                continue
            ok_par = (rec.get("finite") is True and rec.get("rel_err_fwd", 1) <= PARITY_MAX
                      and rec.get("rel_err_dgrad", 1) <= PARITY_MAX)
            if not ok_par:
                out["function_fail"].setdefault(arm, []).append(
                    (M, f"{name} M={M}: parity fwd {rec.get('rel_err_fwd')} dgrad {rec.get('rel_err_dgrad')} "
                        f"finite {rec.get('finite')}"))
            if arm in EXPECT_ROUTE:
                route, key, rkeys = EXPECT_ROUTE[arm]
                if (rec.get("route") != route or rec.get("dgrad_stats_delta", {}).get(key, 0) < 1
                        or any(rec.get("route_stats_delta", {}).get(k, 0) < 1 for k in rkeys)):
                    out["engagement_fail"].setdefault(arm, []).append(
                        (M, f"{name} M={M}: route {rec.get('route')} dgrad {rec.get('dgrad_stats_delta')} "
                            f"routes {rec.get('route_stats_delta')}"))
            f, d = _arm_time(rec.get("fwd_ms")), _arm_time(rec.get("dgrad_ms"))
            if f is None or d is None:
                out["void"].append(f"{name} M={M}: arm {arm} has no timings")
                continue
            for _, sp in (f, d):
                pairs_total += 1
                if not (SELF_PAIR_BAND[0] <= sp <= SELF_PAIR_BAND[1]):
                    pairs_out += 1
            t[(arm, name, M)], tf[(arm, name, M)], td[(arm, name, M)] = 2 * f[0] + d[0], f[0], d[0]
    for need in ("bf16", "bnb"):
        for kind in ("run_fail", "function_fail", "engagement_fail"):
            if need in out[kind]:
                out["void"].append(f"reference arm {need} {kind}: {out[kind][need][0][1]}")
    out["self_pairs"] = {"total": pairs_total, "outside_band": pairs_out}
    if pairs_total and pairs_out / pairs_total > NOISY_FRACTION:
        out["noisy"] = True

    def lc(arm, M, table=t):
        vals = [table.get((arm, n, M)) for n in shapes]
        if any(v is None for v in vals):
            return None
        return sum(v * shapes[n][2] for v, n in zip(vals, shapes))

    R = out["readings"]
    for M in rows:
        b, f16 = lc("bnb", M), lc("bf16", M)
        r = R[str(M)] = {"LC_bnb_ms": b, "LC_bf16_ms": f16, "LCf_bnb_ms": lc("bnb", M, tf), "LCd_bnb_ms": lc("bnb", M, td)}
        if b and f16:
            r["H"] = 1 - f16 / b
        for arm, key in (("gnf4a", "G1"), ("gnf4f", "GF"), ("dq", "DQ")):
            v = lc(arm, M)
            r[f"LC_{arm}_ms"] = v
            if b and v:
                r[key] = b / v
        per = [(cells.get((n, M)) or {}) for n in shapes]
        if b and all(c.get("lora_fwd_ms") and c.get("lora_bwd_ms") for c in per):
            lt = sum((2 * statistics.median(c["lora_fwd_ms"]) + statistics.median(c["lora_bwd_ms"])) * shapes[n][2]
                     for c, n in zip(per, shapes))
            r["L"] = lt / b
        if b and all(c.get("dequant_bnb_ms") for c in per):
            r["dequant_share"] = sum(3 * statistics.median(c["dequant_bnb_ms"]) * shapes[n][2]
                                     for c, n in zip(per, shapes)) / b
        r["bnb_dispatch"] = sorted({str(c.get("bnb_dispatch")) for c in per})
    for h in receipt.get("h2d", []):
        M = h["M"]
        dr = h.get("draws", [])
        r = R.get(str(M))
        if not dr or not r or not r.get("LC_bnb_ms") or r.get("LCf_bnb_ms") is None:
            continue
        loaded = statistics.median(d["copy_loaded_gbs"] for d in dr)
        alone = statistics.median(d["copy_alone_gbs"] for d in dr)
        slow = statistics.median(d["gemm_loaded_ms"] / d["gemm_alone_ms"] for d in dr)
        x_ms = h["layer_bytes"] / (loaded * 1e6)
        r_fwd, r_bwd = r["LCf_bnb_ms"] / x_ms, (r["LCf_bnb_ms"] + r["LCd_bnb_ms"]) / x_ms
        r.update({"h2d_alone_gbs": alone, "h2d_loaded_gbs": loaded, "copy_ratio": loaded / alone, "gemm_slowdown": slow,
                  "xfer_ms": x_ms, "R_fwd": r_fwd, "R_bwd": r_bwd, "Rmin": min(r_fwd, r_bwd),
                  "R": r["LC_bnb_ms"] / (2 * x_ms),
                  "covered": all(d.get("copy_covered") is True and d.get("gemm_covered") is True for d in dr)})

    if out["void"] or out["noisy"]:
        out["verdicts"]["lane"] = "VOID" if out["void"] else "NOISY"
        return out
    out["verdicts"]["lane"] = "READ"

    def get(M, k):
        return R.get(str(M), {}).get(k)

    def failed(arm, kind, rows_=None):
        return any(rows_ is None or M in rows_ for M, _ in out[kind].get(arm, []))

    V = out["verdicts"]
    h2, h4 = get(2048, "H"), get(4096, "H")
    V["speed"] = ("S_ALIVE" if h2 is not None and h2 >= 0.15 else
                  "S_DEAD" if h2 is not None and h4 is not None and h2 < 0.10 and h4 < 0.07 else "S_MARGINAL")
    if failed("gnf4a", "function_fail", GRADED_G1_ROWS):
        V["g1"] = "G1_FUNCTION_FAIL"
    elif failed("gnf4a", "engagement_fail", GRADED_G1_ROWS):
        V["g1"] = "G1_ENGAGEMENT_FAIL"
    elif failed("gnf4a", "run_fail", GRADED_G1_ROWS):
        V["g1"] = "G1_COULD_NOT_RUN"
    else:
        g2, g4 = get(2048, "G1"), get(4096, "G1")
        V["g1"] = ("G1_LOSS" if (g2 is not None and g2 < 0.95) or (g4 is not None and g4 < 0.95) else
                   "G1_WIN" if g2 is not None and g4 is not None and g2 >= 1.05 and g4 >= 1.05 else "G1_PARITY")
    gf = [get(M, "GF") for M in rows]
    if failed("gnf4f", "function_fail"):
        V["fused"] = "GF_FUNCTION_FAIL"
    elif failed("gnf4f", "engagement_fail"):
        V["fused"] = "GF_ENGAGEMENT_FAIL"
    elif all(v is not None for v in gf):
        V["fused"] = "GF_LOSS" if all(v < 0.95 for v in gf) else "GF_NOT_LOSS"
    else:
        V["fused"] = "GF_PARTIAL"       # some cells could not run (e.g. resources); reported, not graded
    l2 = get(2048, "L")
    V["lora"] = None if l2 is None else ("L_HIGH" if l2 >= 0.15 else "L_LOW")
    m2, m4 = get(2048, "Rmin"), get(4096, "Rmin")
    if m2 is None or m4 is None:
        V["stream"] = None
    elif not all(get(M, "covered") for M in REGISTERED_H2D_ROWS):
        V["stream"] = "C_UNCOVERED"     # a loaded reading had an unloaded tail: the instrument, not the link
    elif m2 >= 1.25 and get(2048, "gemm_slowdown") <= 1.05 and get(2048, "copy_ratio") >= 0.8:
        V["stream"] = "C_ALIVE"
    elif m4 < 1.0:
        V["stream"] = "C_DEAD"
    else:
        V["stream"] = "C_MARGINAL"
    return out


# ------------------------------------------------------------------ self-test (synthetic receipts; no GPU)
SHAPES_T = REGISTERED_SHAPES


def synth(*, bf16=0.95, gnf4a=1.0, gnf4f=0.3, lora=0.05, jitter=1.0, device=REGISTERED_DEVICE, rehearsal=False,
          gbs=50.0, slow=1.01, layer_bytes=260_000_000, rows=tuple(REGISTERED_ROWS), break_arm=None, route_a="dense",
          jitter_shapes=None, loaded_frac=1.0, covered=True, finished=True):
    """A receipt whose every bnb fwd and dgrad is M/1024 ms per linear (so LC(bnb, 2048) = 3 x 2 x 7 = 42 ms, LCf = 14), the
    other arms scaled by the given ratios (arm time = bnb time x ratio, so G1 = 1/gnf4a and H = 1 - bf16; gnf4f's is GF)."""
    cells = []
    for n in SHAPES_T:
        for M in rows:
            base = M / 1024.0
            arms = {}
            for arm, k in (("bf16", bf16), ("bnb", 1.0), ("dq", 1.0), ("gnf4a", gnf4a), ("gnf4f", gnf4f)):
                k = k.get(M, 1.0) if isinstance(k, dict) else k
                v = base / k if arm == "gnf4f" else base * k
                j = jitter if jitter_shapes is None or n in jitter_shapes else 1.0
                rec = {"fwd_ms": [[v] * 5, [v * j] * 5], "dgrad_ms": [[v] * 5, [v * j] * 5],
                       "rel_err_fwd": 2e-3, "rel_err_dgrad": 2e-3, "finite": True,
                       "route": {"gnf4a": route_a, "gnf4f": "fused"}.get(arm, arm),
                       "dgrad_stats_delta": {"kernel": int(arm == "gnf4f"), "dense": int(arm == "gnf4a"), "grouped_mm": 0,
                                             "loop": 0},
                       "route_stats_delta": {"dense_fwd": int(arm == "gnf4a"), "dense_dgrad": int(arm == "gnf4a")}}
                if arm == break_arm:
                    rec["rel_err_fwd"] = 0.5
                arms[arm] = rec
            cells.append({"shape": n, "M": M, "arms": arms, "lora_fwd_ms": [base * lora] * 5,
                          "lora_bwd_ms": [base * lora] * 5, "dequant_bnb_ms": [base * 0.01] * 5, "bnb_dispatch": "dequant"})
    h2d = [{"M": M, "layer_bytes": layer_bytes,
            "draws": [{"copy_alone_gbs": gbs, "copy_loaded_gbs": gbs * loaded_frac, "gemm_alone_ms": 10.0,
                       "gemm_loaded_ms": 10.0 * slow, "copy_alone_ms": 1.0, "copy_covered": covered,
                       "gemm_covered": True}] * 3} for M in REGISTERED_H2D_ROWS]
    r = {"schema": "dq1-census/1", "rehearsal": rehearsal, "forensics": {"device": device},
         "shapes": {k: list(v) for k, v in SHAPES_T.items()}, "rows": list(rows), "cells": cells, "h2d": h2d}
    if finished:
        r["finished_at"] = "2026-10-04T00:00:00Z"
    return r


def _self_test() -> int:
    cases = []

    def case(name, receipt, check):
        out = reduce(receipt)
        ok = bool(check(out))
        cases.append((name, ok))
        if not ok:
            print("FAIL", name, json.dumps(out["verdicts"]), out["void"][:2], out["function_fail"].keys())

    def V(o):
        return o["verdicts"]
    case("dead speed, parity g1, fused loss", synth(bf16=0.95, gnf4a=1.0),
         lambda o: V(o) == {**V(o), "lane": "READ", "speed": "S_DEAD", "g1": "G1_PARITY", "fused": "GF_LOSS", "lora": "L_LOW"})
    case("alive speed", synth(bf16=0.80), lambda o: V(o)["speed"] == "S_ALIVE")
    case("marginal speed", synth(bf16=0.88), lambda o: V(o)["speed"] == "S_MARGINAL")
    case("boundary H=0.15 is alive", synth(bf16=0.85), lambda o: V(o)["speed"] == "S_ALIVE")
    case("g1 win", synth(gnf4a=0.90), lambda o: V(o)["g1"] == "G1_WIN")
    case("g1 loss", synth(gnf4a=1.10), lambda o: V(o)["g1"] == "G1_LOSS")
    case("g1 parity just inside", synth(gnf4a=0.96), lambda o: V(o)["g1"] == "G1_PARITY")
    case("fused not a loss", synth(gnf4f=1.2), lambda o: V(o)["fused"] == "GF_NOT_LOSS")
    case("lora high", synth(lora=0.2), lambda o: V(o)["lora"] == "L_HIGH")
    case("noisy self-pairs", synth(jitter=1.10), lambda o: V(o)["lane"] == "NOISY")
    # 1 of 5 shapes jittered = 20 % of pairs out of band (> 10 %): noisy; none = read
    case("20 % of pairs out is NOISY", synth(jitter=1.10, jitter_shapes={"q_proj"}), lambda o: V(o)["lane"] == "NOISY")
    case("in-band jitter is READ", synth(jitter=1.04), lambda o: V(o)["lane"] == "READ")
    case("g1 loss at 2048 only", synth(gnf4a={2048: 1 / 0.93}), lambda o: V(o)["g1"] == "G1_LOSS")
    case("g1 loss at 4096 only", synth(gnf4a={4096: 1 / 0.93}), lambda o: V(o)["g1"] == "G1_LOSS")
    case("g1 win needs both rows", synth(gnf4a={2048: 0.9}), lambda o: V(o)["g1"] == "G1_PARITY")
    case("H(2048)=0.12 is not dead even with H(4096) low", synth(bf16={512: 0.8, 1024: 0.85, 2048: 0.88, 4096: 0.95, 8192: 0.97}),
         lambda o: V(o)["speed"] == "S_MARGINAL")
    case("fused losing at one M only is GF_NOT_LOSS", synth(gnf4f={512: 0.3}), lambda o: V(o)["fused"] == "GF_NOT_LOSS")
    case("DMA keeping < 80 % of its bandwidth is not alive", synth(gbs=50.0, loaded_frac=0.7), lambda o: V(o)["stream"] == "C_MARGINAL")
    # LC weights k/v and gate/up twice: 7 linears x C (= 3 x M/1024 ms) -> 3 x 2 x 7 = 42 ms at M=2048
    case("LC counts each linear of the layer", synth(), lambda o: abs(o["readings"]["2048"]["LC_bnb_ms"] - 42.0) < 1e-9)
    r = synth()
    r["cells"] = [c for c in r["cells"] if c["M"] != 4096]
    case("a receipt missing a registered row is VOID", r, lambda o: V(o)["lane"] == "VOID")
    r = synth()
    r["cells"] = [c for c in r["cells"] if c["shape"] != "down_proj"]
    case("a receipt missing a shape's cells is VOID", r, lambda o: V(o)["lane"] == "VOID")
    r = synth()
    r["h2d"] = []
    case("a receipt without the streaming probe is VOID", r, lambda o: V(o)["lane"] == "VOID")
    case("a receipt the alarm cut short is VOID", synth(finished=False), lambda o: V(o)["lane"] == "VOID")
    r = synth()
    r["shapes"] = {"q_proj": [8192, 5120, 1]}
    case("off-spec shapes are VOID", r, lambda o: V(o)["lane"] == "VOID")
    case("an RTX 5090 D is not the registered card", synth(device="NVIDIA GeForce RTX 5090 D"),
         lambda o: V(o)["lane"] == "VOID")
    # R_fwd(2048) = 14 / X: at 22.3 GB/s X = 11.66 ms -> R_fwd 1.20 (< 1.25) while the pooled R = 42 / 23.3 = 1.80
    case("pooled R above 1.5 with R_fwd below 1.25 is MARGINAL", synth(gbs=22.3), lambda o: V(o)["stream"] == "C_MARGINAL")
    case("R_fwd at 1.30 is ALIVE", synth(gbs=0.26 * 1.30 / 0.014), lambda o: V(o)["stream"] == "C_ALIVE")
    case("an uncovered loaded draw withholds the stream verdict", synth(covered=False),
         lambda o: V(o)["stream"] == "C_UNCOVERED")
    r = synth()
    for c in r["cells"]:
        c["arms"]["gnf4a"]["route_stats_delta"]["dense_fwd"] = 0
    case("gnf4a whose dense forward never ran is G1_ENGAGEMENT_FAIL", r, lambda o: V(o)["g1"] == "G1_ENGAGEMENT_FAIL")
    case("a gnf4f parity failure is GF_FUNCTION_FAIL", synth(break_arm="gnf4f"), lambda o: V(o)["fused"] == "GF_FUNCTION_FAIL")
    r = synth()
    for c in r["cells"]:
        if c["M"] == 8192 and c["shape"] == "down_proj":
            c["arms"]["gnf4a"]["errors"] = ["pos3: OutOfMemoryError"]
    case("a gnf4a failure outside the graded rows does not grade G1", r, lambda o: V(o)["g1"] == "G1_PARITY")
    r = synth()
    for c in r["cells"]:
        if c["M"] == 2048 and c["shape"] == "down_proj":
            c["arms"]["gnf4a"]["errors"] = ["pos3: OutOfMemoryError"]
    case("a gnf4a failure inside the graded rows is G1_COULD_NOT_RUN", r, lambda o: V(o)["g1"] == "G1_COULD_NOT_RUN")
    r = synth()
    for c in r["cells"]:
        if c["M"] == 8192:
            c["arms"]["gnf4a"]["rel_err_fwd"] = 0.5
    case("a gnf4a parity failure outside the graded rows does not grade G1", r, lambda o: V(o)["g1"] == "G1_PARITY")
    case("speed dead needs H(4096) < 0.07", synth(bf16={512: 0.9, 1024: 0.9, 2048: 0.92, 4096: 0.92, 8192: 0.95}),
         lambda o: V(o)["speed"] == "S_MARGINAL")
    case("wrong device is void", synth(device="NVIDIA RTX A2000 12GB"), lambda o: V(o)["lane"] == "VOID")
    case("rehearsal is void", synth(rehearsal=True), lambda o: V(o)["lane"] == "VOID")
    case("broken reference arm is void", synth(break_arm="bnb"), lambda o: V(o)["lane"] == "VOID")
    case("broken gnf4a parity is G1_FUNCTION_FAIL", synth(break_arm="gnf4a"), lambda o: V(o)["g1"] == "G1_FUNCTION_FAIL")
    case("gnf4a on the wrong route is G1_ENGAGEMENT_FAIL", synth(route_a="fused"), lambda o: V(o)["g1"] == "G1_ENGAGEMENT_FAIL")
    # streaming: LC(bnb, 2048) = 6 * 8 = 48 ms; xfer = 2 * bytes / gbs
    case("stream alive", synth(gbs=50.0, layer_bytes=260_000_000), lambda o: V(o)["stream"] == "C_ALIVE")
    case("stream dead", synth(gbs=1.0, layer_bytes=260_000_000), lambda o: V(o)["stream"] == "C_DEAD")
    case("stream marginal (contention)", synth(gbs=50.0, slow=1.2), lambda o: V(o)["stream"] == "C_MARGINAL")
    r = synth()
    del r["cells"][0]["arms"]["gnf4f"]
    case("missing arm is void", r, lambda o: V(o)["lane"] == "VOID")
    r = synth()
    r["cells"][3]["arms"]["gnf4f"]["errors"] = ["pos4: OutOfResources"]
    case("a fused cell that cannot run is GF_PARTIAL, lane still read", r, lambda o: V(o)["lane"] == "READ")
    bad = sum(1 for _n, ok in cases if not ok)
    print(f"self-test {'OK' if not bad else 'FAILED'} ({len(cases)} cases{', ' + str(bad) + ' failed' if bad else ''})")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(_self_test())
    with open(sys.argv[1]) as fh:
        print(json.dumps(reduce(json.load(fh)), indent=1, default=str))
