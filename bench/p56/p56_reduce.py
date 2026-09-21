#!/usr/bin/env python3
"""P56 reducer — read the four-arm ladder and say which of the two registered
outcomes the data is in. Nothing here is hand-transcribed; the results page is
generated from this.

Governed by ``bench/p56/P56-PREREG.md``. The arms, in increasing per-op error:

    reference_attn4        per-expert loop, 0 by definition
    batched_attn4          enable_batched_train, ~4e-3 composed at 48 layers
    fused_attn4_nodgrad    enable_fast_train(dgrad=False), forward fusion only
    fused_attn4            enable_fast_train(dgrad=True), ~5e-2 composed

A DEFECT in the fused path is a function of the arithmetic: the deltas order with
the ladder and the smallest rung is small. A TRAJECTORY FLOOR is not a function of
the arithmetic: every rung above the router's flip threshold lands in the same
place. Those are distinguishable, which is the point of running four arms instead
of re-running the failing pair.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

BAND = 0.05                       # tp1's B2/C2 rule, unchanged: |Δ final| and median step |Δ|
REFERENCE = "reference_attn4"
LADDER = ["batched_attn4", "fused_attn4_nodgrad", "fused_attn4"]
OK = {"ok", "OK"}


def load(run_dir: str) -> dict:
    """Every ``*.json`` receipt in the run dir, keyed by its ``tag``."""
    out = {}
    for fn in sorted(os.listdir(run_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            r = json.load(open(os.path.join(run_dir, fn)))
        except (ValueError, OSError):
            continue
        if isinstance(r, dict) and r.get("tag"):
            out[r["tag"]] = r
    return out


def valid(arm: dict, ref: dict | None) -> tuple[bool, list[str]]:
    """P1, plus P2 for the batched arm. Returns (ok, reasons-it-is-not)."""
    why = []
    if arm.get("status") not in OK:
        why.append(f"status {arm.get('status')!r}")
    if not arm.get("C1_bit_exact", False):
        why.append("C1 frozen bytes not bit-exact")
    if arm.get("tag") != REFERENCE and not arm.get("n_patched"):
        why.append("n_patched == 0 (the accelerated path was never installed)")
    if ref is not None:
        if arm.get("init_sha") != ref.get("init_sha"):
            why.append("init_sha differs from the reference (arms did not start identical)")
        if arm.get("trainable_params") != ref.get("trainable_params"):
            why.append(f"trainable {arm.get('trainable_params')} != reference {ref.get('trainable_params')}")
        if (arm.get("tokens") or {}).get("sha256") != (ref.get("tokens") or {}).get("sha256"):
            why.append("tokens sha differs from the reference")
        if len(arm.get("losses") or []) != len(ref.get("losses") or []):
            why.append("step counts differ")
    # P2. A fallback lands on the reference forward and is INVISIBLE in the output,
    # so a batched arm that fell back would read ~0.000 while measuring nothing.
    # This is the one check that keeps its number from being a lie.
    if arm.get("tag") == "batched_attn4":
        st = arm.get("batched_stats")
        if not st:
            why.append("batched arm carries no batched_stats (engagement unproven)")
        else:
            if not st.get("calls"):
                why.append("batched_stats records 0 calls")
            if st.get("fallback_calls"):
                why.append(f"batched path fell back to the reference on "
                           f"{st['fallback_calls']}/{st.get('calls')} calls "
                           f"({ {k: v for k, v in (st.get('by_reason') or {}).items() if v} }) "
                           f"— this arm is measuring the reference against itself")
    return (not why), why


def parity(arm: dict, ref: dict) -> tuple[float, float]:
    d_final = abs(arm["loss_last"] - ref["loss_last"])
    med = statistics.median(abs(a - b) for a, b in zip(arm["losses"], ref["losses"]))
    return d_final, med


def reduce_run(run_dir: str) -> dict:
    rec = load(run_dir)
    ref = rec.get(REFERENCE)
    rows, notes = [], []
    if ref is None or not valid(ref, None)[0]:
        return {"readable": False,
                "why": "no valid reference arm: " + ("missing" if ref is None else "; ".join(valid(ref, None)[1])),
                "rows": [], "verdict": "NO-REF"}
    for tag in LADDER:
        arm = rec.get(tag)
        if arm is None:
            rows.append({"arm": tag, "status": "NOT_RUN"})
            continue
        ok, why = valid(arm, ref)
        if not ok:
            rows.append({"arm": tag, "status": "VOID", "why": "; ".join(why)})
            continue
        d_final, med = parity(arm, ref)
        rows.append({"arm": tag, "status": "PASS" if (d_final <= BAND and med <= BAND) else "FAIL",
                     "d_final": round(d_final, 5), "median_step": round(med, 5),
                     "s_per_step": arm.get("s_per_step_median_11plus"),
                     "dgrad": arm.get("dgrad"),
                     "batched_stats": arm.get("batched_stats")})

    read = [r for r in rows if r.get("d_final") is not None]
    verdict, why = "UNREADABLE", "fewer than two rungs of the ladder produced a number"
    if len(read) >= 2:
        lo = min(r["d_final"] for r in read)
        hi = max(r["d_final"] for r in read)
        # The registered fork. "Ordered" means the smallest-perturbation rung that
        # ran is itself small (inside the band) while a larger rung is not: the
        # delta then tracks the arithmetic. "Flat" means every rung that ran is
        # outside the band and they sit within 2x of each other: the delta does
        # not track the arithmetic at all.
        smallest = next((r for r in rows if r.get("d_final") is not None), None)
        if smallest and smallest["d_final"] <= BAND < hi:
            verdict = "TRACKS-ARITHMETIC"
            why = (f"the smallest-perturbation rung that ran ({smallest['arm']}) is inside the band "
                   f"at {smallest['d_final']:.5f} while {hi:.5f} is outside it — the divergence is a "
                   f"function of the arithmetic, so the fused path's own error is implicated")
        elif lo > BAND and hi <= 2 * lo:
            verdict = "FLOOR-NOT-ARITHMETIC"
            why = (f"every rung that ran is outside the band ({lo:.5f}..{hi:.5f}) and they sit within "
                   f"2x of each other despite per-op errors an order of magnitude apart — the "
                   f"divergence does not track the arithmetic; it is this model's trajectory floor, "
                   f"and the 0.05 band cannot attribute any part of it to the fused path")
        else:
            verdict = "MIXED"
            why = f"rungs span {lo:.5f}..{hi:.5f}; neither registered pattern fits — report as measured"
    return {"readable": True, "rows": rows, "verdict": verdict, "why": why,
            "reference": {"loss_last": ref.get("loss_last"),
                          "s_per_step": ref.get("s_per_step_median_11plus"),
                          "trainable_params": ref.get("trainable_params")}}


def render(res: dict) -> str:
    out = ["| arm | Δ final | median step \\|Δ\\| | band 0.05 | s/step | note |",
           "|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if r.get("d_final") is None:
            out.append(f"| `{r['arm']}` | — | — | **{r['status']}** | — | {r.get('why', '')} |")
            continue
        note = ""
        if r.get("batched_stats"):
            st = r["batched_stats"]
            note = f"engaged {st.get('batched')}/{st.get('calls')} calls, {st.get('fallback_calls')} fallbacks"
        elif r.get("dgrad") is not None:
            note = f"dgrad={r['dgrad']}"
        out.append(f"| `{r['arm']}` | {r['d_final']:.5f} | {r['median_step']:.5f} | "
                   f"**{r['status']}** | {r.get('s_per_step')} | {note} |")
    out += ["", f"**Verdict: {res['verdict']}** — {res['why']}"]
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: p56_reduce.py <run-dir>")
    result = reduce_run(sys.argv[1])
    print(render(result))
    print()
    print(json.dumps(result, indent=2))
