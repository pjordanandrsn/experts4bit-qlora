"""Pure byte-level DQ8 reading, with no fitted coefficients or calibration import."""
import argparse
import json
from pathlib import Path

from dq7_subject import CONFIG_HASHES

DEVICE = "NVIDIA GeForce RTX 4090"
LAYERS = {"qwen3_14b": 40, "llama31_8b": 32, "qwen3_32b": 64}
ARMS = {(name, placement) for name in ("qwen3_14b", "llama31_8b") for placement in ("device", "stream")}
ARMS.add(("qwen3_32b", "stream"))


def reduce(proof, refusal, receipts):
    def void(reason):
        return {"schema": "dq8-read/1", "verdict": "VOID", "pass_licensed": False, "reason": reason}

    keys = ("loss_bitwise", "all_gradients_bitwise", "optimizer_updates_bitwise", "reload_logits_bitwise", "frozen_unchanged")
    if proof.get("status") != "PASS" or any(proof.get(key) is not True for key in keys):
        return {**void("proof failed"), "verdict": "FUNCTION_FAIL"}
    if refusal.get("status") == "BOUNDARY_CHANGED":
        return {**void("resident control unexpectedly feasible"), "verdict": "BOUNDARY_CHANGED"}
    p = refusal.get("plan", {})
    gpu = p.get("hardware", {}).get("gpu", {})
    control_setup = {"base": "nf4", "placement": "device", "r": 16, "alpha": 32, "adapter_dtype": "fp32",
                     "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"], "attn_impl": "sdpa", "loss_chunk": 512}
    if (refusal.get("schema") != "dq8-refusal/1" or refusal.get("status") != "REFUSED"
        or refusal.get("subject") != "qwen3_32b" or refusal.get("placement") != "device"
        or refusal.get("seq") != 4096 or refusal.get("allocated_before") != 0 or refusal.get("allocated_after") != 0
        or refusal.get("config_sha256") != CONFIG_HASHES["qwen3_32b"]
        or p.get("status") != "refused" or gpu.get("name") != DEVICE
        or p.get("model", {}).get("n_layers") != 64 or p.get("model", {}).get("model_type") != "qwen3"
        or p.get("workload", {}).get("seq_len") != 4096 or p.get("workload", {}).get("steps") != 2
        or p.get("constraints", {}).get("fixed") != control_setup):
        return void("invalid boundary control")
    seen, rows, pairs, pins = set(), [], {}, set()
    for receipt in receipts:
        meta, setup = receipt.get("dq7", {}), receipt.get("setup", {})
        subject, placement = meta.get("subject"), meta.get("placement")
        key = subject, placement
        g = receipt.get("hardware", {}).get("gpu", {})
        expected = {"base": "nf4", "placement": placement, "r": 16, "alpha": 32, "adapter_dtype": "fp32",
                    "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"], "attn_impl": "sdpa",
                    "loss_chunk": 0 if subject == "llama31_8b" else 512}
        total = g.get("memory_total", 0)
        if (key not in ARMS or key in seen or receipt.get("backend") != "dense" or receipt.get("status") != "OK"
            or g.get("name") != DEVICE or not isinstance(total, int) or not 23*2**30 <= total <= 25*2**30
            or setup != expected or receipt.get("workload", {}).get("steps") != 2 or meta.get("seq") != 4096
            or meta.get("config_sha256") != CONFIG_HASHES.get(subject) or meta.get("synthetic") is not True
            or meta.get("pretrained") is not False or meta.get("allocator") != "default"
            or meta.get("real_tokens_per_row") != 4096 or meta.get("padded_tokens") != 0):
            return void("changed or duplicate reading")
        engaged = receipt.get("engaged", {})
        if (engaged.get("setup") != expected or bool(engaged.get("chunked_loss_engaged")) != bool(expected["loss_chunk"])
            or len(engaged.get("adapter_targets", [])) != 7*LAYERS[subject]):
            return void("setup did not engage")
        if placement == "stream":
            report = engaged.get("dense_offload", {})
            if (report.get("layers") != LAYERS[subject] or report.get("late_bound_4bit") != 7*LAYERS[subject]
                or report.get("all_pinned") is not True):
                return void("streaming did not engage")
        row_hash = meta.get("row_tokens_sha256")
        if not row_hash or subject in pairs and pairs[subject] != row_hash:
            return void("rows differ by placement")
        pairs[subject] = row_hash
        pin = meta.get("loggetta_sha"), meta.get("e4b_sha")
        if not all(pin):
            return void("missing pins")
        pins.add(pin)
        cmp, measured = receipt.get("comparison", {}), receipt.get("measured", {})
        a = cmp.get("device_allocator", {})
        estimate, peak = a.get("estimated"), a.get("measured")
        reserved, full = measured.get("device_reserved_peak_bytes"), cmp.get("device_driver", {}).get("estimated")
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (estimate, peak, reserved, full)) or reserved < peak:
            return void("missing or invalid peak bytes")
        lines = receipt.get("plan", {}).get("selected", {}).get("lines", [])
        context = sum(line["bytes"] for line in lines if line.get("where") == "device"
                      and line.get("name", "").startswith("CUDA context"))
        if not context:
            return void("missing priced context line")
        seen.add(key)
        rows.append({"subject": subject, "placement": placement, "seq": 4096, "estimated_bytes": estimate,
                     "allocator_peak_bytes": peak, "allocator_residual_bytes": peak-estimate,
                     "reserved_peak_bytes": reserved, "full_estimate_bytes": full,
                     "priced_context_bytes": context, "reserved_covered": reserved + context <= full, "measured": measured,
                     "itemized_estimate": lines})
    if seen != ARMS or len(pins) != 1:
        return void("incomplete arms or changed software")
    under = any(row["allocator_residual_bytes"] > 0 for row in rows)
    reserve_under = any(not row["reserved_covered"] for row in rows)
    verdict = "ESTIMATE_UNDER" if under else "RESERVE_UNDER" if reserve_under else "BOUNDARY_PASS"
    return {"schema": "dq8-read/1", "verdict": verdict, "pass_licensed": not (under or reserve_under), "rows": rows,
            "reserved_headroom_pass": all(row["reserved_covered"] for row in rows), "quality_read": False,
            "calibration_replacement_licensed": False, "refusal": refusal}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory)
    result = reduce(json.loads((directory/"proof.json").read_text()), json.loads((directory/"refusal.json").read_text()),
                    [json.loads(p.read_text()) for p in sorted(directory.glob("read-*.json"))])
    print(json.dumps(result, indent=2))
