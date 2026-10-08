"""Known-subject diagnostic only: paired cache intervention, never a capacity licence."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dq7_subject import CONFIG_HASHES

SUBJECT_ARMS = (("llama31_8b", "device", 2048), ("llama31_8b", "stream", 2048),
                ("llama31_8b", "device", 4096), ("llama31_8b", "stream", 4096),
                ("qwen3_32b", "device", 2048), ("qwen3_32b", "stream", 2048),
                ("qwen3_14b", "stream", 512), ("qwen3_14b", "stream", 4096))
ARMS = tuple((*arm, mode) for arm in SUBJECT_ARMS for mode in ("baseline", "empty"))
LAYERS = {"qwen3_14b": 40, "llama31_8b": 32, "qwen3_32b": 64}
SHAPE_HASH = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_SHA = re.compile(r"[0-9a-f]{40}\Z")


def valid_hash(value):
    return isinstance(value, str) and SHAPE_HASH.fullmatch(value) is not None


def positive(value):
    return type(value) is int and value > 0


def phase_valid(report, mode):
    if (report.get("schema") != "dq9-phases/1" or report.get("cache_mode") != mode
            or report.get("cache_calls") != int(mode == "empty") or report.get("new_sync_calls") != 0):
        return False
    expected = [(name, 0) for name in ("loader-start", "loader-done", "setup-done", "cache-boundary-done",
                                      "optimizer-create-start", "optimizer-created")]
    expected += [(name, step) for step in (1, 2) for name in ("forward-start", "forward-loss-done", "backward-start",
                 "backward-done", "clip-start", "clip-done", "optimizer-step-start", "optimizer-step-done")]
    rows = report.get("rows", [])
    if [(row.get("phase"), row.get("step")) for row in rows] != expected:
        return False
    for row in rows:
        if any(type(row.get(k)) is not int or row[k] < 0 for k in (
                "allocated", "reserved", "cached", "peak_allocated", "peak_reserved")):
            return False
        if (row["reserved"] < row["allocated"] or row["cached"] != row["reserved"]-row["allocated"]
                or row["peak_allocated"] < row["allocated"] or row["peak_reserved"] < row["reserved"]):
            return False
    inventories = [report.get("setup_storages", [])]
    optimizer = report.get("optimizer_storages", [])
    if [r.get("step") for r in optimizer] != [1, 2]:
        return False
    inventories += [r.get("storages", []) for r in optimizer]
    return all(entries and all(positive(r.get("storage_bytes")) and isinstance(r.get("name"), str)
                              and isinstance(r.get("device"), str) and isinstance(r.get("shape"), list)
                              and isinstance(r.get("dtype"), str) for r in entries) for entries in inventories)


def proof_valid(proofs, runtime, e4b_sha):
    seen, common = set(), None
    for proof in proofs:
        key = (proof.get("placement"), proof.get("cache_mode"))
        if key not in {(p, m) for p in ("device", "stream") for m in ("baseline", "empty")} or key in seen:
            return False
        seen.add(key)
        if (proof.get("schema") != "dq9-proof/1" or proof.get("status") != "PASS"
                or proof.get("frozen_unchanged") is not True or proof.get("reload_logits_bitwise") is not True
                or proof.get("gradient_tensors") != 28 or proof.get("seed") != 731
                or proof.get("algorithm") != "math SDPA, deterministic algorithms, TF32 off"
                or proof.get("loggetta_sha") != runtime["loggetta_sha"] or proof.get("e4b_sha") != e4b_sha
                or not valid_hash(proof.get("reload_logits_sha256")) or not phase_valid(proof.get("phases", {}), key[1])):
            return False
        steps = proof.get("steps", [])
        if len(steps) != 2:
            return False
        for step in steps:
            gradients, updates = step.get("gradients_sha256", {}), step.get("updates_sha256", {})
            if (not valid_hash(step.get("loss_sha256")) or len(gradients) != 28 or gradients.keys() != updates.keys()
                    or not all(valid_hash(x) for x in (*gradients.values(), *updates.values()))):
                return False
        math = (steps, proof["reload_logits_sha256"])
        if common is not None and math != common:
            return False
        common = math
    return len(seen) == 4


def reduce(proofs, receipts, runtime, e4b_sha):
    def void(cause):
        return {"schema": "dq9-diagnostic/1", "verdict": "VOID", "cause": cause,
                "capacity_licensed": False, "calibration_replacement_licensed": False}

    if (not SOURCE_SHA.fullmatch(runtime.get("loggetta_sha", "")) or not SOURCE_SHA.fullmatch(e4b_sha)
            or not proof_valid(proofs, runtime, e4b_sha)):
        return void("source pins or paired deterministic proof absent, failed, or different")
    seen, rows, data_hashes, checkpoint_hashes = set(), [], {}, {}
    for receipt in receipts:
        tag, setup, engaged = receipt.get("dq9", {}), receipt.get("setup", {}), receipt.get("engaged", {})
        subject, placement, seq, mode = (tag.get(k) for k in ("subject", "placement", "seq", "cache_mode"))
        key = (subject, placement, seq, mode)
        if key not in ARMS or key in seen:
            return void("unregistered or duplicate arm")
        seen.add(key)
        gpu, workload = receipt.get("hardware", {}).get("gpu", {}), receipt.get("workload", {})
        versions = tag.get("runtime", {})
        if (receipt.get("backend") != "dense" or receipt.get("status") != "OK"
                or gpu.get("name") != "NVIDIA GeForce RTX 5090" or not 30*2**30 <= gpu.get("memory_total", 0) <= 33*2**30
                or tag.get("config_sha256") != CONFIG_HASHES[subject] or tag.get("synthetic") is not True
                or tag.get("pretrained") is not False or tag.get("allocator") != "default"
                or tag.get("real_tokens_per_row") != seq or tag.get("padded_tokens") != 0
                or tag.get("loggetta_sha") != runtime["loggetta_sha"] or tag.get("e4b_sha") != e4b_sha
                or any(versions.get(k) != runtime[k] for k in ("torch", "bitsandbytes", "transformers", "peft"))
                or workload.get("steps") != 2 or workload.get("seq_len") != seq or workload.get("micro_batch") != 1
                or workload.get("grad_accum") != 1 or workload.get("learning_rate") != 2e-4
                or workload.get("lr_schedule") != "constant" or receipt.get("model", {}).get("n_layers") != LAYERS[subject]):
            return void("subject, source, runtime, hardware or workload differs")
        if (setup != {"base": "nf4", "placement": placement, "r": 16, "alpha": 32, "adapter_dtype": "fp32",
                      "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"], "attn_impl": "sdpa",
                      "loss_chunk": 0 if subject == "llama31_8b" else 512}
                or engaged.get("setup") != setup or engaged.get("chunked_loss_engaged") != (0 if subject == "llama31_8b" else 1)
                or len(engaged.get("adapter_targets", [])) != 7*LAYERS[subject]
                or not phase_valid(receipt.get("phases", {}), mode)):
            return void("setup, mechanism engagement or phase telemetry differs")
        offload = engaged.get("dense_offload", {})
        if placement == "stream" and (offload.get("layers") != LAYERS[subject] or offload.get("all_pinned") is not True
                                      or offload.get("late_bound_4bit") != 7*LAYERS[subject]):
            return void("streaming or late-bound backward did not engage")
        checkpoint = tag.get("checkpoint_manifest_sha256")
        if not valid_hash(checkpoint) or checkpoint_hashes.setdefault(subject, checkpoint) != checkpoint:
            return void("pair did not use identical synthetic checkpoint shards")
        digest = tag.get("row_tokens_sha256")
        if not valid_hash(digest) or data_hashes.setdefault((subject, seq), digest) != digest:
            return void("pair did not see identical real-token rows")
        comparison = receipt.get("comparison", {})
        allocated, driver = comparison.get("device_allocator", {}), comparison.get("device_driver", {})
        reserved = receipt.get("measured", {}).get("device_reserved_peak_bytes")
        if not all(positive(x) for x in (allocated.get("estimated"), allocated.get("measured"),
                                         driver.get("estimated"), driver.get("measured"), reserved)):
            return void("allocator, reserved or driver reading absent")
        phases = receipt["phases"]["rows"]
        before, after = phases[2], phases[3]
        released = before["reserved"]-after["reserved"]
        rows.append({"subject": subject, "placement": placement, "seq": seq, "cache_mode": mode,
                     "allocator_estimate_bytes": allocated["estimated"], "allocated_peak_bytes": allocated["measured"],
                     "allocator_residual_bytes": allocated["measured"]-allocated["estimated"],
                     "device_estimate_bytes": driver["estimated"], "driver_peak_bytes": driver["measured"],
                     "driver_residual_bytes": driver["measured"]-driver["estimated"], "reserved_peak_bytes": reserved,
                     "setup_cached_bytes": before["cached"], "cache_boundary_released_reserved_bytes": released,
                     "cache_boundary_allocated_delta_bytes": after["allocated"]-before["allocated"],
                     "cache_release_observed": mode == "empty" and released > 0 and before["allocated"] == after["allocated"],
                     "load_reserved_peak_bytes": phases[1]["peak_reserved"],
                     "load_allocated_peak_bytes": phases[1]["peak_allocated"]})
    if seen != set(ARMS):
        return void("missing arm")
    by_key = {(r["subject"], r["placement"], r["seq"], r["cache_mode"]): r for r in rows}
    pairs = []
    for arm in SUBJECT_ARMS:
        baseline, empty = (by_key[(*arm, mode)] for mode in ("baseline", "empty"))
        pairs.append({"subject": arm[0], "placement": arm[1], "seq": arm[2],
                      "allocated_peak_delta_bytes": empty["allocated_peak_bytes"]-baseline["allocated_peak_bytes"],
                      "reserved_peak_delta_bytes": empty["reserved_peak_bytes"]-baseline["reserved_peak_bytes"],
                      "driver_peak_delta_bytes": empty["driver_peak_bytes"]-baseline["driver_peak_bytes"],
                      "cache_release_observed": empty["cache_release_observed"]})
    return {"schema": "dq9-diagnostic/1", "verdict": "COMPLETE_DIAGNOSTIC", "rows": rows, "pairs": pairs,
            "known_subjects": True, "capacity_licensed": False, "calibration_replacement_licensed": False,
            "allocator_underestimate_arms": sum(r["allocator_residual_bytes"] > 0 for r in rows),
            "driver_underestimate_arms": sum(r["driver_residual_bytes"] > 0 for r in rows)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--runtime", required=True)
    p.add_argument("--proof-only", action="store_true")
    p.add_argument("--e4b-sha", required=True)
    a = p.parse_args()
    root = Path(a.directory)
    proofs = [json.loads(f.read_text()) for f in sorted(root.glob("proof-*.json"))]
    if a.proof_only:
        valid = proof_valid(proofs, json.loads(Path(a.runtime).read_text()), a.e4b_sha)
        print("DQ9 paired proof", "PASS" if valid else "FAIL")
        raise SystemExit(0 if valid else 11)
    result = reduce(proofs,
                    [json.loads(f.read_text()) for f in sorted(root.glob("read-*.json"))],
                    json.loads(Path(a.runtime).read_text()), a.e4b_sha)
    print(json.dumps(result, indent=2))
    raise SystemExit(12 if result["verdict"] == "VOID" else 0)
