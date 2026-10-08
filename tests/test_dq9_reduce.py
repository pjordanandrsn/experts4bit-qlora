"""Fail-closed DQ9 decisions on synthetic phase metadata; fixtures are never CUDA evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/"bench/dq7"))
spec = importlib.util.spec_from_file_location("dq9_reduce_fixture", ROOT/"bench/dq9/dq9_reduce.py")
reducer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reducer)


def phases(mode):
    names = [(n, 0) for n in ("loader-start", "loader-done", "setup-done", "cache-boundary-done",
                              "optimizer-create-start", "optimizer-created")]
    names += [(n, step) for step in (1, 2) for n in ("forward-start", "forward-loss-done", "backward-start",
              "backward-done", "clip-start", "clip-done", "optimizer-step-start", "optimizer-step-done")]
    rows = [{"phase": name, "step": step, "allocated": 1024, "reserved": 2048, "cached": 1024,
             "peak_allocated": 1536, "peak_reserved": 2048} for name, step in names]
    if mode == "empty":
        for row in rows[3:]:
            row.update(reserved=1024, cached=0)
    inventory = [{"name": "fixture", "device": "cuda:0", "dtype": "torch.float32", "shape": [256], "storage_bytes": 1024}]
    return {"schema": "dq9-phases/1", "cache_mode": mode, "cache_calls": int(mode == "empty"), "new_sync_calls": 0,
            "rows": rows, "setup_storages": inventory,
            "optimizer_storages": [{"step": s, "storages": copy.deepcopy(inventory)} for s in (1, 2)]}


def fixtures():
    runtime = json.loads((ROOT/"bench/dq7/runtime.json").read_text())
    e4b_sha = "4ca5a3494e746b9aaa812270e03f4d0c1ec050a9"
    hashes = {f"gradient-{i}": "a"*64 for i in range(28)}
    steps = [{"loss_sha256": "b"*64, "gradients_sha256": hashes, "updates_sha256": hashes} for _ in range(2)]
    proofs = [{"schema": "dq9-proof/1", "status": "PASS", "placement": p, "cache_mode": m,
               "steps": copy.deepcopy(steps), "frozen_unchanged": True, "reload_logits_bitwise": True,
               "reload_logits_sha256": "c"*64, "gradient_tensors": 28, "seed": 731,
               "algorithm": "math SDPA, deterministic algorithms, TF32 off", "phases": phases(m),
               "loggetta_sha": runtime["loggetta_sha"], "e4b_sha": e4b_sha}
              for p in ("device", "stream") for m in ("baseline", "empty")]
    receipts = []
    for subject, placement, seq, mode in reducer.ARMS:
        r = json.loads((ROOT/f"bench/dq7/receipts/dq7-5090-1/read-{subject}-{placement}-{seq}.json").read_text())
        r["dq9"] = r.pop("dq7")
        r["dq9"].update(cache_mode=mode, runtime={k: runtime[k] for k in ("torch", "bitsandbytes", "transformers", "peft")},
                         checkpoint_manifest_sha256="d"*64)
        r["phases"] = phases(mode)
        receipts.append(r)
    return proofs, receipts, runtime, e4b_sha


def test_complete_diagnostic_retains_every_existing_miss_and_never_licenses_capacity():
    result = reducer.reduce(*fixtures())
    assert result["verdict"] == "COMPLETE_DIAGNOSTIC"
    assert len(result["rows"]) == 16 and len(result["pairs"]) == 8
    assert result["allocator_underestimate_arms"] == 8 and result["driver_underestimate_arms"] > 0
    assert result["capacity_licensed"] is result["calibration_replacement_licensed"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "proof-gradient", "proof-missing", "config", "runtime",
                                      "source", "rows", "checkpoint", "sync", "reset", "cache-calls", "inventory", "counter"])
def test_invalid_pair_or_proof_fails_closed(mutation):
    proofs, receipts, runtime, sha = fixtures()
    if mutation == "missing":
        receipts.pop()
    elif mutation == "duplicate":
        receipts.append(copy.deepcopy(receipts[0]))
    elif mutation == "proof-gradient":
        proofs[0]["steps"][0]["gradients_sha256"]["gradient-0"] = "f"*64
    elif mutation == "proof-missing":
        proofs.pop()
    elif mutation in ("config", "source", "rows", "checkpoint"):
        field = {"config": "config_sha256", "source": "loggetta_sha", "rows": "row_tokens_sha256",
                 "checkpoint": "checkpoint_manifest_sha256"}[mutation]
        receipts[0]["dq9"][field] = "f"*(40 if mutation == "source" else 64)
    elif mutation == "runtime":
        receipts[0]["dq9"]["runtime"]["peft"] = "wrong"
    elif mutation == "sync":
        receipts[0]["phases"]["new_sync_calls"] = 1
    elif mutation == "reset":
        receipts[0]["phases"]["rows"][2]["peak_allocated"] = 0
    elif mutation == "cache-calls":
        receipts[0]["phases"]["cache_calls"] = 1
    elif mutation == "inventory":
        receipts[0]["phases"]["optimizer_storages"] = []
    elif mutation == "counter":
        receipts[0]["phases"]["rows"][0]["cached"] += 1
    assert reducer.reduce(proofs, receipts, runtime, sha)["verdict"] == "VOID"


def test_one_byte_miss_survives_and_allocated_change_prevents_cache_only_attribution():
    args = fixtures()
    r = args[1][1]
    for kind in ("device_allocator", "device_driver"):
        comparison = r["comparison"][kind]
        comparison["measured"] = comparison["estimated"]+1
    r["phases"]["rows"][3].update(allocated=1023, cached=1)
    result = reducer.reduce(*args)
    row = result["rows"][1]
    assert row["allocator_residual_bytes"] == row["driver_residual_bytes"] == 1
    assert row["cache_release_observed"] is False
    assert row["cache_boundary_allocated_delta_bytes"] == -1
