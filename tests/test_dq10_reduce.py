"""Falsify individual component gates and source/mechanism identity; fixtures are not CUDA evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_dq9_reduce import phases

ROOT = Path(__file__).parents[1]
for lane in ("dq10", "dq9", "dq7"):
    sys.path.insert(0, str(ROOT / "bench" / lane))
SPEC = importlib.util.spec_from_file_location("dq10_reduce_fixture", ROOT / "bench/dq10/dq10_reduce.py")
reducer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reducer)


def fixtures():
    runtime = json.loads((ROOT / "bench/dq10/runtime.json").read_text())
    policy_bytes = (ROOT / "bench/dq10/dq10_policy.json").read_bytes()
    assert hashlib.sha256(policy_bytes).hexdigest() == runtime["policy_sha256"]
    policy = json.loads(policy_bytes)
    sha = "a" * 40  # Synthetic fixture source identity; never a claimed commit.
    hashes = {f"gradient-{i}": "a" * 64 for i in range(28)}
    steps = [{"loss_sha256": "b" * 64, "gradients_sha256": hashes, "updates_sha256": hashes} for _ in range(2)]
    proofs = []
    for subject in ("tiny-mistral", "tiny-smollm3"):
        for placement in ("device", "stream"):
            engaged = {"setup": {**reducer.SETUP, "placement": placement}, "chunked_loss_engaged": 0,
                       "quantized_linears": list(range(14)),
                       "dense_offload": {"layers": 2, "late_bound_4bit": 6, "all_pinned": True}}
            proofs.append({"schema": "dq10-proof/1", "status": "PASS", "subject": subject, "placement": placement,
                           "steps": copy.deepcopy(steps), "frozen_unchanged": True, "reload_logits_bitwise": True,
                           "reload_logits_sha256": "c" * 64, "embedding_sha256": "d" * 64, "head_sha256": "d" * 64,
                           "tied_parameter_identity": subject == "tiny-smollm3", "gradient_tensors": 28, "seed": 731,
                           "algorithm": "math SDPA, deterministic algorithms, TF32 off", "phases": phases("baseline"),
                           "engaged": engaged, "loggetta_sha": runtime["loggetta_sha"], "e4b_sha": sha})
    receipts = []
    for subject, placement, seq in reducer.SHAPES:
        e, a, r, d = 1_000_000_000, 900_000_000, 950_000_000, 1_000_000_000
        total = reducer.residuals(e, a, r, d, policy["hypotheses"][placement])["plan_total_bytes"]
        setup = {**reducer.SETUP, "placement": placement}
        receipts.append({"backend": "dense", "status": "OK", "setup": setup,
            "model": {"n_layers": reducer.LAYERS[subject]},
            "hardware": {"gpu": {"name": policy["hardware"]["name"], "driver": policy["hardware"]["driver"],
                                   "memory_total": 32 * 2**30, "pcie": [5, 16, 5, 16]}, "ram_limit": 128 * 2**30},
            "workload": {"steps": 2, "seq_len": seq, "micro_batch": 1, "grad_accum": 1,
                         "optimizer": "adamw", "learning_rate": 2e-4, "lr_schedule": "constant"},
            "plan": {"constraints": {"dense_reserve_policy": "dq10", "allocator_profile": "default",
                                     "allow_development_executor": True}},
            "engaged": {"setup": setup, "chunked_loss_engaged": 0,
                        "adapter_targets": list(range(7 * reducer.LAYERS[subject])),
                        "quantized_linears": list(range(7 * reducer.LAYERS[subject])),
                        "memory_policy": {"name": "dq10", "sha256": runtime["policy_sha256"], "basis": "inferred",
                                          "licensed": False, "allocator_profile": "default"},
                        "dense_offload": {"layers": reducer.LAYERS[subject],
                                          "late_bound_4bit": reducer.STREAMED[subject], "all_pinned": True}},
            "dq10": {"subject": subject, "placement": placement, "seq": seq, "synthetic": True,
                     "pretrained": False, "allocator": "default", "real_tokens_per_row": seq, "padded_tokens": 0,
                     "config_sha256": reducer.CONFIG_HASHES[subject], "row_tokens_sha256": "f" * 64,
                     "checkpoint_manifest_sha256": "e" * 64, "loggetta_sha": runtime["loggetta_sha"], "e4b_sha": sha,
                     "runtime": {key: runtime[key] for key in ("torch", "bitsandbytes", "transformers", "peft")}},
            "phases": phases("baseline"), "measured": {"device_reserved_peak_bytes": r},
            "comparison": {"device_allocator": {"estimated": e, "measured": a},
                           "device_driver": {"estimated": total, "measured": d}}})
    return proofs, receipts, runtime, sha, policy_bytes


def test_complete_holdouts_only_eligibility_no_import_or_opt_in_removal():
    result = reducer.reduce(*fixtures())
    assert result["verdict"] == "HOLDOUT_PASS" and len(result["rows"]) == 12
    assert result["default_change_eligible"] is True
    assert result["observation_import_licensed"] is result["opt_in_removal_licensed"] is False


@pytest.mark.parametrize("gate", ["allocator", "reserve-term", "context", "driver"])
def test_one_byte_miss_survives_without_retuning_even_when_other_terms_can_hide_it(gate):
    args = fixtures()
    receipt = args[1][0]
    row = reducer.reduce(*args)["rows"][0]
    if gate == "allocator":
        receipt["comparison"]["device_allocator"]["measured"] = row["allocator_estimate_bytes"] + 1
        receipt["measured"]["device_reserved_peak_bytes"] = row["allocator_estimate_bytes"] + 1
        receipt["comparison"]["device_driver"]["measured"] = row["allocator_estimate_bytes"] + 1
        field = "allocator_residual_bytes"
    elif gate == "reserve-term":
        r = row["allocated_peak_bytes"] + row["reserve_charge_bytes"] + 1
        receipt["measured"]["device_reserved_peak_bytes"] = r
        receipt["comparison"]["device_driver"]["measured"] = r
        field = "reserve_term_residual_bytes"
    elif gate == "context":
        receipt["comparison"]["device_driver"]["measured"] = row["reserved_peak_bytes"] + row["context_charge_bytes"] + 1
        field = "context_residual_bytes"
    else:
        receipt["comparison"]["device_driver"]["measured"] = row["plan_total_bytes"] + 1
        field = "driver_residual_bytes"
    result = reducer.reduce(*args)
    assert result["verdict"] == "HOLDOUT_UNDER" and result["rows"][0][field] == 1
    assert result["default_change_eligible"] is result["capacity_licensed"] is False
    assert result["rows"][0]["residual_charge_bytes"] == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "proof-missing", "proof-gradient", "tied", "config",
                                      "policy-hash", "policy-selection", "context-price", "driver", "runtime", "rows",
                                      "checkpoint", "smol-stream-count", "phase-sync", "padding"])
def test_changed_source_scope_recipe_or_mechanism_is_void(mutation):
    args = fixtures()
    proofs, receipts, _, _, _ = args
    r = receipts[0]
    if mutation == "missing":
        receipts.pop()
    elif mutation == "duplicate":
        receipts.append(copy.deepcopy(r))
    elif mutation == "proof-missing":
        proofs.pop()
    elif mutation == "proof-gradient":
        proofs[0]["steps"][0]["gradients_sha256"]["gradient-0"] = "f" * 64
    elif mutation == "tied":
        proofs[2]["tied_parameter_identity"] = False
    elif mutation == "config":
        r["dq10"]["config_sha256"] = "f" * 64
    elif mutation == "policy-hash":
        r["engaged"]["memory_policy"]["sha256"] = "f" * 64
    elif mutation == "policy-selection":
        r["plan"]["constraints"]["dense_reserve_policy"] = None
    elif mutation == "context-price":
        r["comparison"]["device_driver"]["estimated"] += 1
    elif mutation == "driver":
        r["hardware"]["gpu"]["driver"] = "other"
    elif mutation == "runtime":
        r["dq10"]["runtime"]["torch"] = "other"
    elif mutation in ("rows", "checkpoint"):
        r["dq10"]["row_tokens_sha256" if mutation == "rows" else "checkpoint_manifest_sha256"] = "d" * 64
    elif mutation == "smol-stream-count":
        next(r for r in receipts if r["dq10"]["subject"] == "smollm3_3b" and
             r["dq10"]["placement"] == "stream")["engaged"]["dense_offload"]["late_bound_4bit"] = 252
    elif mutation == "phase-sync":
        r["phases"]["new_sync_calls"] = 1
    elif mutation == "padding":
        r["dq10"]["padded_tokens"] = 1
    assert reducer.reduce(*args)["verdict"] == "VOID"
