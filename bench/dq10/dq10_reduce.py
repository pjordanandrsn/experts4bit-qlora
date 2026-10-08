"""Fixed new-family holdouts. Every byte miss survives; no derivation or retuning here."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from dq10_plan import SHAPES
from dq10_subject import CONFIG_HASHES
from dq9_reduce import phase_valid, positive, valid_hash

LAYERS = {"mistral7b_v03": 32, "smollm3_3b": 36}
STREAMED = {"mistral7b_v03": 224, "smollm3_3b": 180}
SOURCE_SHA = re.compile(r"[0-9a-f]{40}\Z")
SETUP = {"base": "nf4", "r": 16, "alpha": 32, "adapter_dtype": "fp32",
         "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"], "attn_impl": "sdpa", "loss_chunk": 0}


def proof_valid(proofs, runtime, e4b_sha):
    seen, math = set(), {}
    for proof in proofs:
        subject, placement = proof.get("subject"), proof.get("placement")
        key = (subject, placement)
        if subject not in ("tiny-mistral", "tiny-smollm3") or placement not in ("device", "stream") or key in seen:
            return False
        seen.add(key)
        tied = subject == "tiny-smollm3"
        if (proof.get("schema") != "dq10-proof/1" or proof.get("status") != "PASS"
                or proof.get("frozen_unchanged") is not True or proof.get("reload_logits_bitwise") is not True
                or proof.get("gradient_tensors") != 28 or proof.get("seed") != 731
                or proof.get("algorithm") != "math SDPA, deterministic algorithms, TF32 off"
                or proof.get("loggetta_sha") != runtime["loggetta_sha"] or proof.get("e4b_sha") != e4b_sha
                or proof.get("tied_parameter_identity") is not tied
                or not all(valid_hash(proof.get(k)) for k in ("reload_logits_sha256", "embedding_sha256", "head_sha256"))
                or (tied and proof["embedding_sha256"] != proof["head_sha256"])
                or not phase_valid(proof.get("phases", {}), "baseline")):
            return False
        engaged = proof.get("engaged", {})
        if (engaged.get("setup") != {**SETUP, "placement": placement}
                or engaged.get("chunked_loss_engaged") != 0 or len(engaged.get("quantized_linears", [])) != 14):
            return False
        stream = engaged.get("dense_offload", {})
        if placement == "stream" and (stream.get("layers") != 2 or stream.get("late_bound_4bit") != 6
                                       or stream.get("all_pinned") is not True):
            return False
        steps = proof.get("steps", [])
        if len(steps) != 2:
            return False
        for step in steps:
            gradients, updates = step.get("gradients_sha256", {}), step.get("updates_sha256", {})
            if (not valid_hash(step.get("loss_sha256")) or len(gradients) != 28 or gradients.keys() != updates.keys()
                    or not all(valid_hash(x) for x in (*gradients.values(), *updates.values()))):
                return False
        values = (steps, proof["reload_logits_sha256"], proof["embedding_sha256"], proof["head_sha256"])
        if math.setdefault(subject, values) != values:
            return False
    return len(seen) == 4


def residuals(e, a, r, d, hypothesis):
    n, den = hypothesis["reserve_numerator"], hypothesis["reserve_denominator"]
    charge = (e * n + den - 1) // den
    context = hypothesis["context_bytes"]
    return {"allocator_estimate_bytes": e, "allocated_peak_bytes": a, "reserved_peak_bytes": r,
            "driver_peak_bytes": d, "reserve_charge_bytes": charge, "context_charge_bytes": context,
            "plan_total_bytes": e + charge + context, "allocator_residual_bytes": a - e,
            "reserve_term_residual_bytes": r - a - charge, "reserved_total_residual_bytes": r - e - charge,
            "context_residual_bytes": d - r - context, "driver_residual_bytes": d - e - charge - context,
            "residual_charge_bytes": 0}


GATES = ("allocator_residual_bytes", "reserve_term_residual_bytes", "reserved_total_residual_bytes",
         "context_residual_bytes", "driver_residual_bytes")


def reduce(proofs, receipts, runtime, e4b_sha, policy_bytes):
    def void(cause):
        return {"schema": "dq10-holdouts/1", "verdict": "VOID", "cause": cause,
                "capacity_licensed": False, "default_change_eligible": False, "opt_in_removal_licensed": False}

    if (not SOURCE_SHA.fullmatch(runtime.get("loggetta_sha", "")) or not SOURCE_SHA.fullmatch(e4b_sha)
            or hashlib.sha256(policy_bytes).hexdigest() != runtime.get("policy_sha256")
            or not proof_valid(proofs, runtime, e4b_sha)):
        return void("source/policy pins or paired new-family deterministic proof differ")
    policy = json.loads(policy_bytes)
    seen, rows, hashes, checkpoints = set(), [], {}, {}
    for receipt in receipts:
        tag, setup, engaged = receipt.get("dq10", {}), receipt.get("setup", {}), receipt.get("engaged", {})
        subject, placement, seq = (tag.get(k) for k in ("subject", "placement", "seq"))
        key = (subject, placement, seq)
        if key not in SHAPES or key in seen:
            return void("unregistered or duplicate arm")
        seen.add(key)
        gpu, workload = receipt.get("hardware", {}).get("gpu", {}), receipt.get("workload", {})
        constraints = receipt.get("plan", {}).get("constraints", {})
        expected_setup = {**SETUP, "placement": placement}
        if (receipt.get("backend") != "dense" or receipt.get("status") != "OK"
                or gpu.get("name") != policy["hardware"]["name"] or gpu.get("driver") != policy["hardware"]["driver"]
                or not policy["hardware"]["memory_min"] <= gpu.get("memory_total", 0) <= policy["hardware"]["memory_max"]
                or len(gpu.get("pcie", [])) != 4 or gpu["pcie"][0] < 5 or gpu["pcie"][1] != 16
                or receipt.get("hardware", {}).get("ram_limit", 0) < 98_000_000_000
                or tag.get("config_sha256") != CONFIG_HASHES[subject] or tag.get("synthetic") is not True
                or tag.get("pretrained") is not False or tag.get("allocator") != "default"
                or tag.get("real_tokens_per_row") != seq or tag.get("padded_tokens") != 0
                or tag.get("loggetta_sha") != runtime["loggetta_sha"] or tag.get("e4b_sha") != e4b_sha
                or any(tag.get("runtime", {}).get(k) != runtime[k] for k in ("torch", "bitsandbytes", "transformers", "peft"))
                or workload.get("steps") != 2 or workload.get("seq_len") != seq or workload.get("micro_batch") != 1
                or workload.get("grad_accum") != 1 or workload.get("optimizer") != "adamw"
                or workload.get("learning_rate") != 2e-4 or workload.get("lr_schedule") != "constant"
                or constraints.get("dense_reserve_policy") != "dq10" or constraints.get("allocator_profile") != "default"
                or constraints.get("allow_development_executor") is not True
                or receipt.get("model", {}).get("n_layers") != LAYERS[subject]):
            return void("subject, source, runtime, hardware, workload or explicit policy differs")
        if (setup != expected_setup or engaged.get("setup") != setup or engaged.get("chunked_loss_engaged") != 0
                or len(engaged.get("adapter_targets", [])) != 7 * LAYERS[subject]
                or len(engaged.get("quantized_linears", [])) != 7 * LAYERS[subject]
                or engaged.get("memory_policy") != {"name": "dq10", "sha256": runtime["policy_sha256"],
                    "basis": "inferred", "licensed": False, "allocator_profile": "default"}
                or not phase_valid(receipt.get("phases", {}), "baseline")):
            return void("setup, mechanism engagement or phase telemetry differs")
        stream = engaged.get("dense_offload", {})
        if placement == "stream" and (stream.get("layers") != LAYERS[subject]
                or stream.get("late_bound_4bit") != STREAMED[subject] or stream.get("all_pinned") is not True):
            return void("registered streamed homes/late-bound projections differ")
        checkpoint, digest = tag.get("checkpoint_manifest_sha256"), tag.get("row_tokens_sha256")
        if (not valid_hash(checkpoint) or checkpoints.setdefault(subject, checkpoint) != checkpoint
                or not valid_hash(digest) or hashes.setdefault((subject, seq), digest) != digest):
            return void("placement pair checkpoint or real-token row differs")
        comparison = receipt.get("comparison", {})
        allocator, driver = comparison.get("device_allocator", {}), comparison.get("device_driver", {})
        e, a, r, d = (allocator.get("estimated"), allocator.get("measured"),
                      receipt.get("measured", {}).get("device_reserved_peak_bytes"), driver.get("measured"))
        if not all(positive(x) for x in (e, a, r, d)) or r < a or d < r:
            return void("allocator, reserved or sampled driver bytes invalid")
        row = {"subject": subject, "placement": placement, "seq": seq,
               **residuals(e, a, r, d, policy["hypotheses"][placement])}
        if driver.get("estimated") != row["plan_total_bytes"]:
            return void("executed plan is not the frozen itemized hypothesis")
        row["gates"] = {k: "PASS" if row[k] <= 0 else "UNDER" for k in GATES}
        row["unchanged_20_percent_charge_bytes"] = (e + 4) // 5
        row["allocator_overestimate_ratio"] = e / a
        rows.append(row)
    if seen != set(SHAPES):
        return void("missing arm")
    passed = all(row[k] <= 0 for row in rows for k in GATES)
    return {"schema": "dq10-holdouts/1", "verdict": "HOLDOUT_PASS" if passed else "HOLDOUT_UNDER",
            "policy_sha256": runtime["policy_sha256"], "rows": rows, "new_families": True,
            "capacity_licensed": passed, "default_change_eligible": passed,
            "observation_import_licensed": False, "opt_in_removal_licensed": False,
            "scope": "registered 5090/driver/runtime/configs/recipe/rungs only; DQ8 24GB remains required"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--e4b-sha", required=True)
    parser.add_argument("--proof-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.directory)
    proofs = [json.loads(path.read_text()) for path in sorted(root.glob("proof-*.json"))]
    runtime = json.loads(Path(args.runtime).read_text())
    if args.proof_only:
        ok = proof_valid(proofs, runtime, args.e4b_sha)
        print("DQ10 paired new-family proof", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 11)
    result = reduce(proofs, [json.loads(path.read_text()) for path in sorted(root.glob("read-*.json"))],
                    runtime, args.e4b_sha, Path(args.policy).read_bytes())
    print(json.dumps(result, indent=2))
    raise SystemExit(12 if result["verdict"] == "VOID" else 0)
