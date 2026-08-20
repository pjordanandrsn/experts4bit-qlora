# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stage 3, gate 1: can cold routed work be admitted without proportional
growth in layer wall time?

Registered in grouped-nf4-gemm `bench/cold-engine/PREREG-tribrid-stage3.md`.
This harness runs that registration and nothing else: the arms, the sweep
points, and the pass thresholds are read from the prereg, not chosen here.

Four arms at each cold-mass point, same model / prompt / routing trace /
placement / box, differing in exactly one thing:

  control    VRAM + DRAM only — the Stage-2 hybrid, no forced cold mass
  cold-GPU   forced cold experts supplied only to the GPU
  cold-CPU   forced cold experts supplied only to the CPU
  dynamic    the destination rule chooses per step

**The dynamic arm is a threshold, and this harness says so in its own
receipt.** `cold_dest=<rows>` is the DRAM tier's `offload_rows` statistic
read the other way round, not a deadline estimate. Gate 2 is where a
deadline model would have to beat it; calling this arm "dynamic" is a name
for its position in the sweep, not a claim about its mechanism.

## The metric, and where it is approximate

    hide_ratio = 1 - exposed / isolated

*exposed* is measured as `T_step(arm) - T_step(control)` at matched
routing: the wall the cold work actually added. That is a difference of
two measured quantities and inherits both instruments' spread, so the
self-pair below is not optional — a hide ratio computed from a difference
inside the instrument's own noise is not a measurement.

*isolated* is the cold path's own serial cost: the tier's demand fill time
for cold rows plus the cold compute time, both from counters the engine
already keeps (`ColdTier.demand_fill_ns`, the amortization instrument's
`cold_cpu_ns` / `gpu_ns`). This is a **sum of stages**, not a separately
timed idle run, and it is therefore an approximation in one specific
direction: stages that already overlap each other inside the cold path
(a queued read landing while another computes) are counted twice, which
makes `isolated` an over-estimate and `hide_ratio` optimistic.

That bias is stated here rather than buried because it is the one number
the gate turns on. A stricter isolated measurement — replaying each step's
cold set with the layer otherwise idle — is the follow-up, and until it
exists this harness reports `hide_ratio_upper` alongside the raw halves so
a reader can recompute under their own definition.

Numerical equivalence is checked every point: an arm whose logits differ
from the resident reference fails the point outright, before any timing is
considered. A fast wrong answer is not a result.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SWEEP_DEFAULT = (0.01, 0.05, 0.10, 0.20)
ARMS = ("control", "cold-gpu", "cold-cpu", "dynamic")


def hide_ratio(exposed_ns: float, isolated_ns: float):
    """1 - exposed/isolated, or None when there was no cold work to hide.

    Returns None rather than 0.0 or 1.0 for an empty isolated term: a point
    with no cold path did not hide anything and did not fail to, and a
    number there would be read as a measurement of the mechanism.
    Negative exposure (the arm beat its control, inside noise) clamps to 1.0
    and is flagged by the caller's self-pair, not silently kept as >1.
    """
    if isolated_ns is None or isolated_ns <= 0:
        return None
    r = 1.0 - (exposed_ns / isolated_ns)
    return 1.0 if r > 1.0 else r


def summarize(steps_ns):
    """Median plus the spread a difference-of-medians has to clear."""
    if not steps_ns:
        return {"n": 0}
    s = sorted(steps_ns)
    med = statistics.median(s)
    return {
        "n": len(s), "median_ns": med,
        "p10_ns": s[max(0, int(0.10 * len(s)) - 1)],
        "p90_ns": s[min(len(s) - 1, int(0.90 * len(s)))],
        "iqr_ns": (s[min(len(s) - 1, int(0.75 * len(s)))]
                   - s[max(0, int(0.25 * len(s)) - 1)]),
    }


def self_pair(a_ns, b_ns):
    """Two runs of the SAME arm, as the instrument's own spread.

    House rule, and it has teeth here: `exposed` is a difference of two
    medians, so any hide ratio whose exposure is smaller than this pair's
    own disagreement is noise wearing a result's clothes.
    """
    ma, mb = statistics.median(a_ns), statistics.median(b_ns)
    lo = min(ma, mb)
    return {"median_a_ns": ma, "median_b_ns": mb,
            "ratio": (max(ma, mb) / lo) if lo > 0 else None,
            "abs_diff_ns": abs(ma - mb)}


def gate1_verdict(point: dict, *, hide_floor=0.70, slowdown_ceiling=0.05):
    """The prereg's gate-1 clauses, evaluated on one sweep point.

    Every clause is reported, including the ones that pass, and equivalence
    is not tradeable against timing: an arm that is numerically wrong fails
    regardless of how well it hid its I/O.
    """
    dyn = point["arms"].get("dynamic", {})
    clauses = {
        "numerically_equivalent": bool(dyn.get("equivalent")),
        "hide_ratio_ge_floor": (dyn.get("hide_ratio") is not None
                                and dyn["hide_ratio"] >= hide_floor),
        # A MISSING fixed arm must fail this clause, not satisfy it. Reading
        # an absent exposed_ns as +inf made "the dynamic arm beat a
        # measurement that does not exist" pass -- the exact inversion the
        # missing-numbers rule in this module forbids (Bugbot, e4b#170).
        "beats_both_fixed": (
            dyn.get("exposed_ns") is not None
            and all(point["arms"].get(a, {}).get("exposed_ns") is not None
                    for a in ("cold-gpu", "cold-cpu"))
            and all(dyn["exposed_ns"] <= point["arms"][a]["exposed_ns"]
                    for a in ("cold-gpu", "cold-cpu"))),
        "destination_flipped": (dyn.get("destination_flips", 0) >= 1),
        "slowdown_under_ceiling": (
            dyn.get("proportional_slowdown") is not None
            and dyn["proportional_slowdown"] < slowdown_ceiling),
    }
    return {"clauses": clauses,
            "verdict": "PASS" if all(clauses.values()) else "MISS",
            "thresholds": {"hide_floor": hide_floor,
                           "slowdown_ceiling": slowdown_ceiling}}


def engine_config(**kwargs) -> dict:
    """Every ``enable_hybrid_tier`` keyword this run will use, for the receipt.

    The gate-1 receipt originally carried the SWEEP's configuration and not the
    ENGINE's, and the runner was a disposable box script. That made an arm able
    to differ in a way no reader could see after the fact — and it did: the
    cold-CPU arm's residual could not be attributed, because whether
    ``offload_thin_uniq`` was set was unrecoverable from the artifact
    (grouped-nf4-gemm ``bench/cold-engine/RESULTS-tribrid-gate1.md``,
    *Reconciliation*).

    It is load-bearing for equivalence, not just provenance. ``dram_thin`` is
    decided at enable time from a layer's TOTAL DRAM population, and
    ``force_cold_mass`` shrinks that population — so with ``offload_thin_uniq``
    set, moving experts to NVMe can flip the destination of experts that did
    NOT move, and two arms stop being matched for a reason unrelated to
    ``cold_dest``. ``offload_rows`` reads per-step DRAM routing statistics that
    a placement change also perturbs, so it is the same class. Measured in e4b
    ``bench/hybrid-g9/issue171/RESULTS-issue171.md``.

    Defaults are `enable_hybrid_tier`'s own, so a receipt records what ran even
    when the runner passed nothing.
    """
    cfg = {"hot_rows": None, "qd": 4, "threads": 0, "pool": True,
           "offload_rows": None, "fused_ffn": False,
           "offload_thin_uniq": None, "cold_dest": "gpu", "prefetch": False}
    unknown = set(kwargs) - set(cfg)
    if unknown:
        raise ValueError(
            f"engine_config does not record {sorted(unknown)} — a receipt that "
            f"names a setting the engine never saw is worse than one that "
            f"records nothing. Add it here AND to enable_hybrid_tier's "
            f"signature check in tests/test_gate1_harness.py")
    cfg.update(kwargs)
    return cfg


def arm_config(arm: str, cold_rows_threshold: float):
    """`enable_hybrid_tier` kwargs for one arm. The control point carries no
    forced cold mass at all, which is what makes it a baseline rather than a
    fourth destination."""
    if arm == "control":
        return {"cold_dest": "gpu", "_forced_cold": False}
    if arm == "cold-gpu":
        return {"cold_dest": "gpu", "_forced_cold": True}
    if arm == "cold-cpu":
        return {"cold_dest": "cpu", "_forced_cold": True}
    if arm == "dynamic":
        return {"cold_dest": cold_rows_threshold, "_forced_cold": True}
    raise ValueError(f"unknown arm {arm!r} (have {ARMS})")


def build_plan(sweep, arms, *, threshold: float):
    """The full run plan, materialized before anything is measured.

    Emitted into the receipt so a reader can see every cell that was
    INTENDED, not only the ones that produced numbers — a harness that
    silently skipped a cell would look identical to one where that cell
    was never planned.
    """
    return [{"cold_frac": f, "arm": a, **arm_config(a, threshold)}
            for f in sweep for a in arms]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--profile", required=True,
                    help="routing profile (expert_profile JSONL)")
    ap.add_argument("--sweep", default=",".join(str(s) for s in SWEEP_DEFAULT))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--order", default="tail", choices=("tail", "head"))
    ap.add_argument("--threshold", type=float, default=4.0,
                    help="rows-per-unique-expert for the dynamic arm")
    ap.add_argument("--steps", type=int, default=64)
    # ENGINE configuration. Recorded in the receipt whether or not it is
    # passed, because "not set" and "not recorded" are different claims.
    ap.add_argument("--hot-rows", type=int, default=None,
                    help="ColdTier pinned rows; size from MEASURED free RAM")
    ap.add_argument("--offload-thin-uniq", type=int, default=None,
                    help="DRAM population at or below which a layer routes to "
                         "the GPU. NOTE: force_cold_mass shrinks that "
                         "population, so a non-None value here can flip the "
                         "destination of experts the arm did not move — leave "
                         "unset for an equivalence sweep")
    ap.add_argument("--offload-rows", type=float, default=None,
                    help="rows-per-unique DRAM expert at or above which the "
                         "DRAM tier routes to the GPU (same caveat)")
    ap.add_argument("--fused-ffn", action="store_true")
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cold-source", default="dram",
                    choices=("dram", "vram", "both"),
                    help="which tier force_cold_mass takes experts FROM. "
                         "'dram' (the default) moves experts off the CPU tier, "
                         "so a cold_dest='gpu' arm changes their arithmetic "
                         "and the equivalence clause reads that rounding "
                         "rather than the cold path; 'vram' holds the GPU "
                         "destination fixed. See #171")
    ap.add_argument("--warmup", type=int, default=8,
                    help="instrument law 1: warm BOTH shapes before any clock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan-only", action="store_true",
                    help="emit the run plan and exit — no model, no box")
    args = ap.parse_args()

    sweep = tuple(float(s) for s in args.sweep.split(","))
    arms = tuple(a.strip() for a in args.arms.split(","))
    plan = build_plan(sweep, arms, threshold=args.threshold)

    receipt = {
        "schema": "e4b-tribrid-gate1/1",
        "prereg": "grouped-nf4-gemm/bench/cold-engine/PREREG-tribrid-stage3.md",
        "plan": plan,
        "config": {"sweep": sweep, "arms": arms, "order": args.order,
                   "threshold": args.threshold, "steps": args.steps,
                   "warmup": args.warmup,
                   "cold_source": args.cold_source,
                   "dynamic_arm_is": "rows-per-unique-expert threshold, "
                                     "NOT a deadline estimate (gate 2)"},
        # the ENGINE's configuration, not just the sweep's — see engine_config
        "engine": engine_config(
            hot_rows=args.hot_rows,
            offload_thin_uniq=args.offload_thin_uniq,
            offload_rows=args.offload_rows,
            fused_ffn=args.fused_ffn,
            prefetch=args.prefetch,
            threads=args.threads),
        "isolated_term": "sum of cold-path stages; over-estimates when "
                         "stages overlap, so hide_ratio is an UPPER bound",
    }
    if args.plan_only:
        receipt["note"] = "plan only — nothing was measured"
        Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"plan: {len(plan)} cells -> {args.out}")
        return

    raise SystemExit(
        "gate-1 execution needs a stamped prereg and a box with CUDA + the "
        "native CPU kernels + a baked arena. Run --plan-only to emit the "
        "run plan, or wire the model path once those exist: this harness "
        "deliberately refuses to invent a measurement.")


if __name__ == "__main__":
    main()
