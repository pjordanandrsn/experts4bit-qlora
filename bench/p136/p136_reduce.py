#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""P136 R-e: conservative read of P123-format served census records.

A READ only licenses a recorded family/shape cell. It never changes the
portability matrix, a speed claim, or a kernel default by itself.
"""
import argparse
import json
from pathlib import Path

GNF4_SHA = "70bc03cb7a61b03c4e801e7423c58c103a1b6d7d"
FAMILIES = {
    "gemma4": ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52", (1, 16)),
    "mixtral": ("mistralai/Mixtral-8x7B-Instruct-v0.1", "eba92302a2861cdc0098cc54bc9f17cb2c47eb61", (1, 16)),
    "gptoss": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", (16,)),
}
MOE_LAYERS = {"gemma4": 30, "mixtral": 32, "gptoss": 24}
BLOCKING = {
    "CLASS_MAP_INCOMPLETE", "CLASS_MAP_SEGMENT_BROKEN", "NSYS_DIAGNOSTIC_ERRORS",
    "CLOCK_MISMATCH", "PROFILER_INFLATED", "NO_UNPROFILED_TIMING", "NODE_TRACE_INFLATED",
}
CLASSES = ("moe_expert", "moe_route", "attn", "dense_gemm", "norm_elem", "sample", "input_prep", "memcpy", "residual")


def load(root, name):
    try:
        return json.loads((root / name).read_text())
    except (OSError, ValueError):
        return None


def assess(root, family, e4b_sha):
    model, revision, batches = FAMILIES[family]
    faults = []
    arms = [load(root, f"arm_{tag}.json") for tag in ("S1a", "S1b")]
    for tag, arm in zip(("S1a", "S1b"), arms):
        if not arm:
            faults.append(f"missing or invalid arm {tag}")
            continue
        if (arm.get("model"), arm.get("revision"), arm.get("e4b_sha"), arm.get("gnf4_sha")) != (model, revision, e4b_sha, GNF4_SHA):
            faults.append(f"arm {tag}: wrong model or source pin")
        if arm.get("mode") != "speed" or arm.get("tag") != tag or arm.get("status") != "ok":
            faults.append(f"arm {tag}: wrong mode, tag or status")
        if any(v for v in (arm.get("knobs") or {}).values()):
            faults.append(f"arm {tag}: default knobs were set")
        if arm.get("max_seqs") != 16 or not arm.get("graph_stats"):
            faults.append(f"arm {tag}: 16-slot graph server unproven")
        if arm.get("moe_layers") != MOE_LAYERS[family]:
            faults.append(f"arm {tag}: MoE layer count differs")
        grouping = arm.get("grouping") or {}
        if grouping.get("device_grouping") is not True or grouping.get("force_singleton_groups") is not False:
            faults.append(f"arm {tag}: K25 device grouping not established")
        kinds = arm.get("int4_store_kinds") or []
        if ("mxfp4" if family == "gptoss" else "nf4") not in kinds:
            faults.append(f"arm {tag}: expert store kind differs")
        if family == "mixtral" and arm.get("max_tokens_per_seq") != 768:
            faults.append(f"arm {tag}: Mixtral token pool differs")
        if family != "mixtral" and arm.get("max_tokens_per_seq") != 4096:
            faults.append(f"arm {tag}: default token pool differs")
        # A capture can still VOID independently; this check catches an arm
        # whose server never built the required bucket before profiling.
        if not all(str(b) in (arm.get("graph_stats") or {}) for b in batches):
            faults.append(f"arm {tag}: registered graph buckets missing")
    rows = {}
    for batch in batches:
        work = f"W{batch}"
        vals = []
        for tag, arm in zip(("S1a", "S1b"), arms):
            cell = ((arm or {}).get("workloads") or {}).get(work) or {}
            value = cell.get("decode_ms_per_step")
            if cell.get("status") != "ok" or cell.get("batch") != batch or not isinstance(value, (int, float)) or value <= 0:
                faults.append(f"arm {tag}: {work} reference void")
            else:
                vals.append(value)
        if len(vals) == 2 and not 0.97 <= vals[1] / vals[0] <= 1.03:
            faults.append(f"{work}: unprofiled self-pair outside [0.97, 1.03]")
        drv = load(root, f"census_drv_b{batch}.json")
        census = load(root, f"census_b{batch}.json")
        if not drv or not census:
            faults.append(f"{work}: missing capture driver or census")
            continue
        if (drv.get("model"), drv.get("revision"), drv.get("e4b_sha"), drv.get("gnf4_sha")) != (model, revision, e4b_sha, GNF4_SHA):
            faults.append(f"{work}: capture source pin differs")
        if drv.get("batch") != batch or drv.get("skip") != 32 or drv.get("steps_bracketed") != 64:
            faults.append(f"{work}: wrong 32-skip/64-step window")
        if drv.get("moe_layers") != MOE_LAYERS[family] or not arms[0] or drv.get("grouping") != arms[0].get("grouping"):
            faults.append(f"{work}: capture build differs")
        if census.get("engine") != "e4b" or census.get("batch") != batch or census.get("status") not in ("ok", "labelled"):
            faults.append(f"{work}: census void or wrong engine/batch")
        labels = set(census.get("labels") or [])
        if labels & BLOCKING:
            faults.append(f"{work}: blocking labels {sorted(labels & BLOCKING)}")
        if not isinstance(census.get("P_ms"), (int, float)) or census["P_ms"] <= 0:
            faults.append(f"{work}: step missing")
            continue
        terms = census.get("terms") or {}
        if any(not isinstance(terms.get(k), (int, float)) for k in CLASSES + ("I_in", "idle_out", "overlap_in", "O")):
            faults.append(f"{work}: incomplete class/idle terms")
            continue
        closure = sum(terms[k] for k in CLASSES) + terms["I_in"] + terms["idle_out"] + terms["overlap_in"] - terms["O"]
        if abs(closure - census["P_ms"]) > 0.001:
            faults.append(f"{work}: class terms do not close the step")
        node = census.get("node") or {}
        if node.get("kernels_per_graph_modal") is None:
            faults.append(f"{work}: kernel count missing")
        if len(vals) == 2 and census.get("unprofiled_ms") is not None and not 0.9 <= census["unprofiled_ms"] / (sum(vals) / 2) <= 1.1:
            faults.append(f"{work}: reference does not match self-pair")
        p = census["P_ms"]
        rows[work] = {
            "P_ms_profiled": p,
            "unprofiled_ms": round(sum(vals) / 2, 6) if len(vals) == 2 else None,
            "classes_ms": {k: terms[k] for k in CLASSES},
            "I_in_ms": terms["I_in"], "idle_out_ms": terms["idle_out"],
            "overlap_in_ms": terms["overlap_in"], "O_ms": terms["O"],
            "busy_fraction": round((p - terms["I_in"] - terms["idle_out"]) / p, 4),
            "kernels_per_step": node.get("kernels_per_graph_modal"),
            "labels": sorted(labels),
        }
    verdict = "READ" if not faults else "VOID"
    return {"lane": "p136", "family": family, "model": model, "revision": revision,
            "e4b_sha": e4b_sha, "gnf4_sha": GNF4_SHA, "verdict": verdict,
            "why": faults, "rows": rows if verdict == "READ" else {}}


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for family in FAMILIES:
            rec = assess(Path(tmp), family, "a" * 40)
            assert rec["verdict"] == "VOID" and rec["why"]
    print("P136 reducer self-test OK (all missing family receipts VOID)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha")
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not all((args.dir, args.out, args.e4b_sha, args.family)):
        ap.error("--dir, --out, --e4b-sha and --family required")
    rec = assess(Path(args.dir), args.family, args.e4b_sha)
    Path(args.out).write_text(json.dumps(rec, indent=2) + "\n")
    print("P136 " + rec["verdict"] + " " + "; ".join(rec["why"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
