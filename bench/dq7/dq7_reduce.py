"""Registered DQ7 decision: a never-under estimate on out-of-sample capacity arms. No loss/quality read."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dq7_subject import CONFIG_HASHES

LAYERS = {"qwen3_14b": 40, "llama31_8b": 32, "qwen3_32b": 64}
SEQS = {"qwen3_14b": (512, 2048), "llama31_8b": (512, 2048), "qwen3_32b": (2048, 4096)}
DEVICE = "NVIDIA GeForce RTX 5090"
# Exact DQ4 c_def ladder allocator peaks; spreads are ladder vs fresh confirmation at each arm's binding rung.
ANCHOR_BYTES = {("device", 2048): 23577589760, ("device", 4096): 26273331200,
                ("stream", 2048): 8459810816, ("stream", 4096): 11155552256}
ANCHOR_DRAW_SPREAD_BYTES = {"device": 65536, "stream": 3711488}


def reduce(proof, receipts):
    if proof.get("schema") != "dq7-proof/1" or proof.get("status") != "PASS" or not all(
        proof.get(key) is True for key in ("loss_bitwise", "all_gradients_bitwise", "optimizer_updates_bitwise",
                                           "reload_logits_bitwise", "frozen_unchanged")):
        return {"verdict": "VOID", "cause": "deterministic executor proof is absent or failed"}
    expected = {(subject, placement, seq) for subject in SEQS for placement in ("device", "stream") for seq in SEQS[subject]}
    seen, versions, rows, row_hashes = set(), set(), [], {}
    for receipt in receipts:
        tag, setup, engaged = receipt.get("dq7", {}), receipt.get("setup", {}), receipt.get("engaged", {})
        subject, placement, seq = tag.get("subject"), tag.get("placement"), tag.get("seq")
        key = (subject, placement, seq)
        if key not in expected or key in seen:
            return {"verdict": "VOID", "cause": "unregistered or duplicate arm"}
        seen.add(key)
        versions.add((tag.get("loggetta_sha"), tag.get("e4b_sha")))
        gpu = receipt.get("hardware", {}).get("gpu", {})
        if (receipt.get("backend") != "dense" or receipt.get("status") != "OK" or gpu.get("name") != DEVICE
                or not (30 * 2**30 <= gpu.get("memory_total", 0) <= 33 * 2**30)
                or tag.get("config_sha256") != CONFIG_HASHES[subject] or tag.get("synthetic") is not True
                or tag.get("pretrained") is not False or tag.get("real_tokens_per_row") != seq or tag.get("padded_tokens") != 0
                or tag.get("allocator") != "default" or receipt.get("workload", {}).get("steps") != 2
                or receipt.get("model", {}).get("n_layers") != LAYERS[subject]):
            return {"verdict": "VOID", "cause": "subject, device, workload or completed execution differs"}
        if (setup.get("base") != "nf4" or setup.get("placement") != placement or setup.get("r") != 16
                or setup.get("alpha") != 32 or setup.get("adapter_dtype") != "fp32"
                or setup.get("attn_impl") != "sdpa" or setup.get("loss_chunk") != (0 if subject == "llama31_8b" else 512)
                or set(setup.get("targets", [])) != {"attn_in", "attn_out", "mlp_in", "mlp_out"}
                or engaged.get("setup") != setup or engaged.get("chunked_loss_engaged") != (0 if subject == "llama31_8b" else 1)
                or len(engaged.get("adapter_targets", [])) != 7 * LAYERS[subject]):
            return {"verdict": "VOID", "cause": "planned setup or mechanism engagement differs"}
        offload = engaged.get("dense_offload", {})
        if placement == "stream" and (offload.get("layers") != LAYERS[subject] or offload.get("all_pinned") is not True
                                      or offload.get("late_bound_4bit") != 7 * LAYERS[subject]):
            return {"verdict": "VOID", "cause": "stream or late-bound backward did not engage"}
        fingerprint = tag.get("row_tokens_sha256")
        if not fingerprint or row_hashes.setdefault((subject, seq), fingerprint) != fingerprint:
            return {"verdict": "VOID", "cause": "placement pair saw different rows"}
        comparison = receipt.get("comparison", {}).get("device_allocator", {})
        estimate, peak = comparison.get("estimated"), comparison.get("measured")
        if not isinstance(estimate, int) or not isinstance(peak, int) or min(estimate, peak) <= 0:
            return {"verdict": "VOID", "cause": "allocator estimate or peak is absent"}
        residual = peak - estimate
        row = {"subject": subject, "placement": placement, "seq": seq, "estimate_bytes": estimate,
               "peak_bytes": peak, "estimate_GB": estimate / 1e9, "peak_GB": peak / 1e9,
               "residual_bytes": residual, "residual_GB": residual / 1e9, "never_under": residual <= 0,
               "overestimate_fraction": -residual / peak, "overestimate_prediction": 0.04 if placement == "device" else 0.10,
               "itemized_lines": receipt.get("plan", {}).get("selected", {}).get("lines", []),
               "reserved_peak_bytes": receipt.get("measured", {}).get("device_reserved_peak_bytes"),
               "driver_peak_bytes": receipt.get("measured", {}).get("driver_process_peak_bytes")}
        if subject == "qwen3_32b":
            anchor = ANCHOR_BYTES[(placement, seq)]
            spread = ANCHOR_DRAW_SPREAD_BYTES[placement]
            row.update(anchor_expected_bytes=anchor, anchor_spread_bytes=spread,
                       anchor_residual_bytes=peak-anchor, anchor_reproduced=abs(peak - anchor) <= spread)
        rows.append(row)
    if seen != expected or len(versions) != 1 or any(not value for value in next(iter(versions), ())):
        return {"verdict": "VOID", "cause": "missing arms or inconsistent software pins", "rows": rows}
    misses = [row for row in rows if row["subject"] != "qwen3_32b" and not row["never_under"]]
    oos = "ESTIMATE_UNDER" if misses else "NEVER_UNDER"
    anchored = all(row.get("anchor_reproduced", True) for row in rows)
    return {"verdict": oos if anchored else "ANCHOR_MISS", "out_of_sample_verdict": oos,
            "pass_licensed": anchored and not misses, "rows": rows,
            "decisive_subjects": ["qwen3_14b", "llama31_8b"], "underestimate_arms": len(misses),
            "quality_read": False, "calibration_replacement_licensed": False}


def self_test():
    # Missing proof must fail closed even when every purported peak looks good.
    assert reduce({}, [])['verdict'] == "VOID"
    proof = {"schema": "dq7-proof/1", "status": "PASS", **{key: True for key in (
        "loss_bitwise", "all_gradients_bitwise", "optimizer_updates_bitwise", "reload_logits_bitwise", "frozen_unchanged")}}
    assert reduce(proof, [])['verdict'] == "VOID"
    broken = copy.deepcopy(proof)
    broken["all_gradients_bitwise"] = False
    assert reduce(broken, [])['verdict'] == "VOID"
    print("dq7 reducer self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        directory = Path(args.directory)
        print(json.dumps(reduce(json.loads((directory / "proof.json").read_text()),
                                [json.loads(p.read_text()) for p in sorted(directory.glob("read-*.json"))]), indent=2))
