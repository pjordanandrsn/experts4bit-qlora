#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Reduce the registered Gemma-4 graph clock and repeat node-class shares.

Node-mode durations never become graph-time class milliseconds. A READ is an
instrument record for review, not a default, speed, or quality claim.
"""
import argparse
import json
import math
from pathlib import Path

MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
GNF4_SHA = "70bc03cb7a61b03c4e801e7423c58c103a1b6d7d"
TORCH = "2.8.0+cu128"
CLASSES = ("moe_expert", "moe_route", "attn", "dense_gemm", "norm_elem", "sample", "input_prep", "memcpy", "residual")
BLOCKING = {"CLASS_MAP_INCOMPLETE", "CLASS_MAP_SEGMENT_BROKEN", "NSYS_DIAGNOSTIC_ERRORS",
            "CLOCK_MISMATCH", "PROFILER_INFLATED", "NO_UNPROFILED_TIMING"}
ALLOWED_LABELS = BLOCKING | {"NODE_TRACE_INFLATED", "NSYS_GRAPH_ID_MAPPING"}


def _load(root, name):
    try:
        return json.loads((root / name).read_text())
    except (OSError, ValueError):
        return None


def _positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def _nonnegative(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _bucket_replayed(record, batch):
    stats = ((record.get("graph_stats") or {}).get(str(batch)) or {})
    return _positive(stats.get("replays")) and stats.get("eager_steps") == 0


def _pins(record, e4b_sha):
    return (record.get("model"), record.get("revision"), record.get("e4b_sha"), record.get("gnf4_sha")) == (
        MODEL, REVISION, e4b_sha, GNF4_SHA)


def _shares(census):
    node = census.get("node") or {}
    classes = node.get("class") or {}
    vals = {name: (classes.get(name) or {}).get("median_ms") for name in CLASSES}
    if not all(_nonnegative(v) for v in vals.values()):
        return None
    total = math.fsum(vals.values())
    if not _positive(total):
        return None
    return {name: vals[name] / total for name in CLASSES}


def assess(root, e4b_sha):
    faults = []
    if not isinstance(e4b_sha, str) or len(e4b_sha) != 40 or any(c not in "0123456789abcdef" for c in e4b_sha):
        faults.append("source pin is not a 40-character lowercase SHA")
    arms = [_load(root, f"arm_{tag}.json") for tag in ("S1a", "S1b")]
    for tag, arm in zip(("S1a", "S1b"), arms):
        if not isinstance(arm, dict):
            faults.append(f"missing/invalid arm {tag}")
            continue
        if (not _pins(arm, e4b_sha) or arm.get("mode") != "speed" or arm.get("tag") != tag
                or arm.get("status") != "ok" or arm.get("torch") != TORCH):
            faults.append(f"arm {tag}: source, subject or status differs")
        if any(v for v in (arm.get("knobs") or {}).values()) or arm.get("max_seqs") != 16 or not arm.get("graph_stats"):
            faults.append(f"arm {tag}: default 16-slot graph server unproven")
        grouping = arm.get("grouping") or {}
        if (arm.get("moe_layers") != 30 or arm.get("int4_store_kinds") != [] or arm.get("int4_expert_layers") != 0
                or grouping.get("device_grouping") is not True or grouping.get("force_singleton_groups") is not False
                or arm.get("max_tokens_per_seq") != 4096):
            faults.append(f"arm {tag}: default Gemma-4 configuration differs")
        if not all(_bucket_replayed(arm, b) for b in (1, 16)):
            faults.append(f"arm {tag}: W1/W16 graph buckets missing")
    if all(isinstance(a, dict) for a in arms) and arms[0].get("prompts_sha256") != arms[1].get("prompts_sha256"):
        faults.append("speed arms: prompt digests differ")
    rows = {}
    for batch in (1, 16):
        work = f"W{batch}"
        refs = []
        for tag, arm in zip(("S1a", "S1b"), arms):
            cell = ((arm or {}).get("workloads") or {}).get(work) or {}
            value = cell.get("decode_ms_per_step")
            if cell.get("status") != "ok" or cell.get("batch") != batch or not _positive(value):
                faults.append(f"arm {tag}: {work} unprofiled reference missing")
            else:
                refs.append(value)
        if len(refs) == 2 and not 0.97 <= refs[1] / refs[0] <= 1.03:
            faults.append(f"{work}: self-pair outside [0.97, 1.03]")
        drivers = {mode: _load(root, f"census_drv_b{batch}_{mode}.json") for mode in ("graph", "node_a", "node_b")}
        censuses = {side: _load(root, f"census_b{batch}_node_{side}.json") for side in ("a", "b")}
        ids = []
        for mode, drv in drivers.items():
            if not isinstance(drv, dict):
                faults.append(f"{work}: missing/invalid {mode} driver")
                continue
            if (not _pins(drv, e4b_sha) or drv.get("batch") != batch or drv.get("skip") != 32
                    or drv.get("steps_bracketed") != 64 or drv.get("mode") != "census"
                    or drv.get("moe_layers") != 30 or drv.get("int4_store_kinds") != []
                    or drv.get("int4_expert_layers") != 0 or drv.get("torch") != TORCH
                    or drv.get("max_tokens_per_seq") != 4096
                    or not _bucket_replayed(drv, batch)):
                faults.append(f"{work}: {mode} driver source/window/default differs")
            if arms[0] and drv.get("grouping") != arms[0].get("grouping"):
                faults.append(f"{work}: {mode} grouping differs")
            if arms[0] and drv.get("prompts_sha256") != arms[0].get("prompts_sha256"):
                faults.append(f"{work}: {mode} prompt digest differs")
            cid = drv.get("capture_id")
            if not isinstance(cid, str) or len(cid) != 32 or any(c not in "0123456789abcdef" for c in cid):
                faults.append(f"{work}: {mode} capture identity missing")
            else:
                ids.append(cid)
        if len(ids) != 3 or len(set(ids)) != 3:
            faults.append(f"{work}: graph/node captures are not separately identified")
        shares = {}
        node_diagnostics = {}
        clock = None
        labels = set()
        for side, census in censuses.items():
            if not isinstance(census, dict):
                faults.append(f"{work}: missing/invalid node-{side} census")
                continue
            if census.get("engine") != "e4b" or census.get("batch") != batch or census.get("status") not in ("ok", "labelled"):
                faults.append(f"{work}: node-{side} census invalid")
            labs = set(census.get("labels") or [])
            labels.update(labs)
            if labs & BLOCKING or labs - ALLOWED_LABELS:
                faults.append(f"{work}: node-{side} blocking labels {sorted((labs & BLOCKING) | (labs - ALLOWED_LABELS))}")
            p = census.get("P_ms")
            graph = census.get("graph") or {}
            s = (graph.get("S") or {}).get("median_ms")
            if not _positive(p) or not _positive(s) or s > p:
                faults.append(f"{work}: node-{side} graph clock missing")
            elif clock is None:
                clock = (p, s)
            elif (p, s) != clock:
                faults.append(f"{work}: node reductions disagree on graph clock")
            node = census.get("node") or {}
            if ((graph.get("steps_kept") or 0) < 56 or (node.get("steps_kept") or 0) < 56
                    or (node.get("segments") or {}).get("moe_layers") != 30):
                faults.append(f"{work}: node-{side} classified window incomplete")
            if not _positive(node.get("kernels_per_graph_modal")):
                faults.append(f"{work}: node-{side} kernel count missing")
            if (not _nonnegative(node.get("residual_fraction")) or node["residual_fraction"] > 0.02
                    or (node.get("segments") or {}).get("replays_broken")):
                faults.append(f"{work}: node-{side} class map or segments incomplete")
            share = _shares(census)
            if share is None:
                faults.append(f"{work}: node-{side} class durations incomplete/nonfinite")
            else:
                shares[side] = share
            gates = census.get("gates") or {}
            band = census.get("node_band") or {}
            if (not _positive(gates.get("node_span_over_span"))
                    or not _positive(gates.get("node_union_over_span"))
                    or not _nonnegative(band.get("omega_ms"))):
                faults.append(f"{work}: node-{side} trace diagnostics missing/nonfinite")
            node_diagnostics[side] = {
                "kernels_per_graph_modal": node.get("kernels_per_graph_modal"),
                "map_residual_fraction": node.get("residual_fraction"),
                "node_span_over_graph_span": gates.get("node_span_over_span"),
                "node_union_over_graph_span": gates.get("node_union_over_span"),
                "omega_ms_diagnostic_only": band.get("omega_ms"),
            }
        if len(refs) == 2 and clock is not None and abs(clock[0] / math.fsum(refs) * 2 - 1) > 0.05:
            faults.append(f"{work}: graph P outside frozen 5% unprofiled gate")
        if len(shares) == 2:
            spread = {name: abs(shares["a"][name] - shares["b"][name]) for name in CLASSES}
            if any(delta > 0.05 for delta in spread.values()):
                faults.append(f"{work}: repeat class shares differ by over five percentage points")
            if clock is not None:
                midpoint = {name: math.fsum((shares["a"][name], shares["b"][name])) / 2 for name in CLASSES}
                rows[work] = {"graph_P_ms_profiled": clock[0], "graph_S_ms": clock[1],
                              "graph_out_of_graph_interval_ms": clock[0] - clock[1],
                              "graph_span_fraction": clock[1] / clock[0],
                              "unprofiled_ms": math.fsum(refs) / 2 if len(refs) == 2 else None,
                              "node_duration_share_a": shares["a"], "node_duration_share_b": shares["b"],
                              "node_duration_share_midpoint": midpoint, "max_share_difference": max(spread.values()),
                              "node_diagnostics": node_diagnostics,
                              "labels": sorted(labels)}
    verdict = "READ" if not faults and len(rows) == 2 else "VOID"
    predictions = {}
    if verdict == "READ":
        w1 = rows["W1"]["node_duration_share_midpoint"]
        w16 = rows["W16"]["node_duration_share_midpoint"]
        ranked_w1 = sorted(CLASSES, key=lambda name: (-w1[name], name))
        ranked_w16 = sorted(CLASSES, key=lambda name: (-w16[name], name))
        predictions = {
            "W16_graph_P_exceeds_W1": "HELD" if rows["W16"]["graph_P_ms_profiled"] > rows["W1"]["graph_P_ms_profiled"] else "MISSED",
            "W1_dense_and_norm_top_two": "HELD" if set(ranked_w1[:2]) == {"dense_gemm", "norm_elem"} else "MISSED",
            "W16_moe_expert_largest": "HELD" if ranked_w16[0] == "moe_expert" else "MISSED",
            "residual_share_below_two_percent": "HELD" if all(rows[w][f"node_duration_share_{side}"]["residual"] < 0.02
                                                         for w in ("W1", "W16") for side in ("a", "b")) else "MISSED",
        }
    return {"lane": "g4census", "run_id": "g4census-5090-1", "model": MODEL, "revision": REVISION,
            "e4b_sha": e4b_sha, "gnf4_sha": GNF4_SHA, "verdict": verdict, "why": faults,
            "rows": rows if verdict == "READ" else {}, "predictions": predictions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            assert assess(Path(tmp), "a" * 40)["verdict"] == "VOID"
        print("g4census reducer missing-input self-test OK")
        return 0
    if not all((args.dir, args.out, args.e4b_sha)):
        ap.error("--dir, --out and --e4b-sha required")
    result = assess(Path(args.dir), args.e4b_sha)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("g4census " + result["verdict"] + " " + "; ".join(result["why"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
