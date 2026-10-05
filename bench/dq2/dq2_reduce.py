"""bench/dq2/dq2_reduce.py -- lane DQ2's registered rule (bench/dq2/DQ2-PREREG.md, "The rule"). Pure: a receipt in, readings
and a verdict out. ``python dq2_reduce.py receipt.json`` prints the read; ``--self-test`` runs the synthetic cases;
tests/test_dq2_lane.py pins both and kills the rule's mutants.

Per row M (one Qwen3-32B decoder layer, micro-batch 1 x M tokens):
  T_fwd, T_bwd   mean over the two positions (before / after the streaming probe) of each position's median draw;
                 position 2 / position 1 is the self-pair
  X              layer frozen bytes / pinned H2D GB/s measured while the layer's forward runs (copy under load)
  R_fwd = T_fwd / X ; R_bwd = T_bwd / X ; Rmin = min    -- each phase must hide one transfer behind its own compute
  slowdown       forward under DMA / forward alone ; copy_ratio = loaded / alone H2D GB/s
"""
from __future__ import annotations

import json
import statistics
import sys

REGISTERED_ROWS = [512, 1024, 2048, 4096, 8192]
REGISTERED_DEVICE = "NVIDIA GeForce RTX 5090"
REGISTERED_LINK = {"gen_max": 5, "width_max": 16}
GRADED_ROW = 2048
SELF_PAIR_BAND = (0.95, 1.05)
NOISY_MAX_OUT = 1                 # of the 10 self-pairs (fwd, bwd x 5 rows), more than this is NOISY (> 10 %)
MIN_GEN5_ALONE_GBS = 35.0         # a gen5 x16 link that cannot copy at least this fast alone is not the registered link
LORA_TARGETS = 7


def _pos_median(series):
    return statistics.median(series) if series else None


def reduce(r: dict) -> dict:
    out = {"void": [], "noisy": False, "readings": {}, "verdicts": {}}
    if r.get("rehearsal"):
        out["void"].append("rehearsal receipt: never graded")
    if r.get("forensics", {}).get("device") != REGISTERED_DEVICE:
        out["void"].append(f"device {r.get('forensics', {}).get('device')!r} is not {REGISTERED_DEVICE!r}")
    ln = r.get("link", {})
    if any(ln.get(k) != v for k, v in REGISTERED_LINK.items()):
        out["void"].append(f"link {ln} is not PCIe gen 5 x16")
    if list(r.get("rows", [])) != REGISTERED_ROWS:
        out["void"].append(f"rows {r.get('rows')} are not {REGISTERED_ROWS}")
    if not r.get("finished_at"):
        out["void"].append("no finished_at: the census did not run to its end")
    if not r.get("frozen_sha256_before") or r.get("frozen_sha256_before") != r.get("frozen_sha256_after"):
        out["void"].append("frozen 4-bit storage changed (or was not hashed) across the run")
    eng = r.get("engagement", {})
    wr = eng.get("wrappers", {})
    if (eng.get("linear4bit_modules") != LORA_TARGETS or eng.get("lora_params") != 2 * LORA_TARGETS
            or len(wr) != LORA_TARGETS or not all("peft" in v and "bnb" in v.lower() for v in wr.values())
            or eng.get("lora_dtypes") != ["torch.float32"]):
        out["void"].append(f"engagement: {eng}")
    cells = {c.get("M"): c for c in r.get("cells", [])}
    pairs_out = 0
    R = out["readings"]
    uncovered = False
    for M in REGISTERED_ROWS:
        c = cells.get(M)
        if c is None or c.get("error"):
            out["void"].append(f"row {M}: {'missing' if c is None else c['error']}")
            continue
        if not c.get("warm", {}).get("steady"):
            out["void"].append(f"row {M}: warm-up did not reach steady")
        p1, p2 = c.get("pos1", {}), c.get("pos2", {})
        bad = [p for p in (p1, p2) if not p.get("integrity", {}).get("finite")
               or p.get("integrity", {}).get("lora_grads_nonzero", 0) < LORA_TARGETS]
        if bad:
            out["void"].append(f"row {M}: integrity {[p.get('integrity') for p in bad]}")
        f1, f2 = _pos_median(p1.get("fwd_ms")), _pos_median(p2.get("fwd_ms"))
        b1, b2 = _pos_median(p1.get("bwd_ms")), _pos_median(p2.get("bwd_ms"))
        if None in (f1, f2, b1, b2):
            out["void"].append(f"row {M}: no timings")
            continue
        for sp in (f2 / f1, b2 / b1):
            if not SELF_PAIR_BAND[0] <= sp <= SELF_PAIR_BAND[1]:
                pairs_out += 1
        dr = c.get("stream", {}).get("draws", [])
        if not dr:
            out["void"].append(f"row {M}: no streaming probe")
            continue
        loaded = statistics.median(d["copy_loaded_gbs"] for d in dr)
        alone = statistics.median(d["copy_alone_gbs"] for d in dr)
        slow = statistics.median(d["fwd_loaded_ms"] / d["fwd_alone_ms"] for d in dr)
        uncovered |= not all(d.get("copy_covered") is True and d.get("fwd_covered") is True for d in dr)
        x_ms = r["layer_frozen_bytes"] / (loaded * 1e6)
        tf, tb = (f1 + f2) / 2, (b1 + b2) / 2
        R[str(M)] = {"T_fwd_ms": tf, "T_bwd_ms": tb, "self_pair_fwd": f2 / f1, "self_pair_bwd": b2 / b1,
                     "h2d_alone_gbs": alone, "h2d_loaded_gbs": loaded, "copy_ratio": loaded / alone,
                     "fwd_slowdown": slow, "X_ms": x_ms, "R_fwd": tf / x_ms, "R_bwd": tb / x_ms,
                     "Rmin": min(tf, tb) / x_ms}
    g = R.get(str(GRADED_ROW))
    if g and g["h2d_alone_gbs"] < MIN_GEN5_ALONE_GBS:
        out["void"].append(f"H2D alone {g['h2d_alone_gbs']:.1f} GB/s at M={GRADED_ROW}: not a gen 5 x16 link in practice")
    out["self_pairs_out"] = pairs_out
    out["noisy"] = pairs_out > NOISY_MAX_OUT
    # reported, not graded: the smallest registered row at which each bar is met
    out["break_even"] = {bar: next((M for M in REGISTERED_ROWS if R.get(str(M), {}).get("Rmin", 0) >= bar), None)
                         for bar in (1.0, 1.25)}
    if out["void"] or out["noisy"]:
        out["verdicts"]["lane"] = "VOID" if out["void"] else "NOISY"
        return out
    out["verdicts"]["lane"] = "READ"
    if uncovered:
        out["verdicts"]["stream"] = "C_UNCOVERED"
    elif g["Rmin"] >= 1.25 and g["fwd_slowdown"] <= 1.05 and g["copy_ratio"] >= 0.8:
        out["verdicts"]["stream"] = "C_ALIVE"
    elif g["Rmin"] < 1.0:
        out["verdicts"]["stream"] = "C_DEAD"
    else:
        out["verdicts"]["stream"] = "C_MARGINAL"
    return out


# ------------------------------------------------------------------ self-test (synthetic receipts; no GPU)
def synth(*, tf=12.0, tb=30.0, gbs=50.0, nbytes=251_539_136, jitter=1.0, jitter_rows=(), slow=1.01, loaded_frac=1.0,
          covered=True, device=REGISTERED_DEVICE, gen=5, width=16, rehearsal=False, finished=True, sha_after=None,
          err_row=None, nonzero=7, steady=True, wrappers_ok=True):
    """T_fwd = tf x M/2048 ms, T_bwd = tb x M/2048, X = nbytes / gbs (so Rmin(2048) = min(tf, tb) / X)."""
    cells = []
    for M in REGISTERED_ROWS:
        s = M / 2048
        j = jitter if M in jitter_rows else 1.0
        integ = {"finite": True, "lora_grads_nonzero": nonzero, "lora_grads": 14}
        c = {"M": M, "warm": {"steady": steady, "seconds": 4.1},
             "pos1": {"fwd_ms": [tf * s] * 5, "bwd_ms": [tb * s] * 5, "integrity": integ},
             "pos2": {"fwd_ms": [tf * s * j] * 5, "bwd_ms": [tb * s * j] * 5, "integrity": integ},
             "stream": {"draws": [{"copy_alone_gbs": gbs, "copy_loaded_gbs": gbs * loaded_frac, "fwd_alone_ms": 10.0,
                                   "fwd_loaded_ms": 10.0 * slow, "copy_covered": covered, "fwd_covered": True}] * 3}}
        if M == err_row:
            c = {"M": M, "error": "OutOfMemoryError: synthetic"}
        cells.append(c)
    w = "peft.tuners.lora.bnb.Linear4bit" if wrappers_ok else "torch.nn.modules.linear.Linear"
    r = {"schema": "dq2-layer/1", "rehearsal": rehearsal, "forensics": {"device": device},
         "link": {"gen_max": gen, "width_max": width}, "rows": list(REGISTERED_ROWS), "cells": cells,
         "layer_frozen_bytes": nbytes, "frozen_sha256_before": "aa", "frozen_sha256_after": sha_after or "aa",
         "engagement": {"linear4bit_modules": 7, "lora_params": 14, "wrappers": {f"m{i}": w for i in range(7)},
                        "lora_dtypes": ["torch.float32"]}}
    if finished:
        r["finished_at"] = "2026-10-05T00:00:00Z"
    return r


def _self_test() -> int:
    cases = []

    def case(name, receipt, check):
        o = reduce(receipt)
        ok = bool(check(o))
        cases.append(ok)
        if not ok:
            print("FAIL", name, o["verdicts"], o["void"][:2], o.get("self_pairs_out"))

    V = lambda o: o["verdicts"]  # noqa: E731
    X = 251_539_136 / 50e9 * 1e3                          # 5.03 ms at 50 GB/s
    case("alive", synth(), lambda o: V(o) == {"lane": "READ", "stream": "C_ALIVE"})
    case("Rmin at 1.30 is alive", synth(tf=1.30 * X), lambda o: V(o)["stream"] == "C_ALIVE")
    case("Rmin at 1.20 is marginal", synth(tf=1.20 * X), lambda o: V(o)["stream"] == "C_MARGINAL")
    case("Rmin at 0.95 is dead", synth(tf=0.95 * X), lambda o: V(o)["stream"] == "C_DEAD")
    case("the backward can bind", synth(tf=30.0, tb=1.1 * X), lambda o: V(o)["stream"] == "C_MARGINAL")
    case("forward slowdown under DMA > 1.05 is not alive", synth(slow=1.08), lambda o: V(o)["stream"] == "C_MARGINAL")
    case("DMA losing bandwidth is not alive", synth(loaded_frac=0.7), lambda o: V(o)["stream"] == "C_MARGINAL")
    case("uncovered", synth(covered=False), lambda o: V(o)["stream"] == "C_UNCOVERED")
    case("gen 4 host is void", synth(gen=4), lambda o: V(o)["lane"] == "VOID")
    case("x8 host is void", synth(width=8), lambda o: V(o)["lane"] == "VOID")
    case("slow 'gen5' link is void", synth(gbs=30.0, tf=30.0, tb=60.0), lambda o: V(o)["lane"] == "VOID")
    case("rehearsal is void", synth(rehearsal=True), lambda o: V(o)["lane"] == "VOID")
    case("wrong card is void", synth(device="NVIDIA GeForce RTX 5090 D"), lambda o: V(o)["lane"] == "VOID")
    case("cut short is void", synth(finished=False), lambda o: V(o)["lane"] == "VOID")
    case("frozen bytes changed is void", synth(sha_after="bb"), lambda o: V(o)["lane"] == "VOID")
    case("an errored row is void", synth(err_row=8192), lambda o: V(o)["lane"] == "VOID")
    case("dead LoRA grads are void", synth(nonzero=0), lambda o: V(o)["lane"] == "VOID")
    case("unsteady warm-up is void", synth(steady=False), lambda o: V(o)["lane"] == "VOID")
    case("not PEFT-over-bnb is void", synth(wrappers_ok=False), lambda o: V(o)["lane"] == "VOID")
    r = synth()
    r["engagement"]["lora_dtypes"] = ["torch.bfloat16"]
    case("bf16 adapters (not get_peft_model's fp32) are void", r, lambda o: V(o)["lane"] == "VOID")
    r = synth()
    r["engagement"]["wrappers"] = {f"m{i}": "peft.tuners.lora.layer.Linear" for i in range(7)}
    case("PEFT's generic Linear wrapper (the rehearsal's finding) is void", r, lambda o: V(o)["lane"] == "VOID")
    case("one jittered row moves both its pairs: NOISY", synth(jitter=1.10, jitter_rows=(512,)),
         lambda o: V(o)["lane"] == "NOISY")      # 2 of 10 > 1
    r = synth()
    r["cells"][0]["pos2"]["fwd_ms"] = [x * 1.10 for x in r["cells"][0]["pos2"]["fwd_ms"]]
    case("exactly one self-pair out is READ", r, lambda o: V(o)["lane"] == "READ" and o["self_pairs_out"] == 1)
    case("two rows jittered is NOISY", synth(jitter=1.10, jitter_rows=(512, 1024)), lambda o: V(o)["lane"] == "NOISY")
    case("in-band jitter is READ", synth(jitter=1.04, jitter_rows=tuple(REGISTERED_ROWS)), lambda o: V(o)["lane"] == "READ")
    case("break-even rows reported", synth(tf=1.1 * X),
         lambda o: o["break_even"] == {1.0: 2048, 1.25: 4096})
    case("readings computed", synth(), lambda o: abs(o["readings"]["2048"]["Rmin"] - 12.0 / X) < 1e-9)
    bad = sum(1 for ok in cases if not ok)
    print(f"self-test {'OK' if not bad else 'FAILED'} ({len(cases)} cases{', ' + str(bad) + ' failed' if bad else ''})")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(_self_test())
    with open(sys.argv[1]) as fh:
        print(json.dumps(reduce(json.load(fh)), indent=1, default=str))
