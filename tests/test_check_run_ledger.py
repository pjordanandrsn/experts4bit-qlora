"""Tests for scripts/check_run_ledger.py

Three fixtures:
  1. One passing receipt (valid, within ceilings, proper approvals)
  2. One over-ceiling day (exceeds role daily ceiling)
  3. One missing approval (insufficient approvals for the cost threshold)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Compute absolute path to the script
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_run_ledger.py"


def test_passing_receipt(tmp_path: Path) -> None:
    """A valid receipt that passes all checks."""
    # Create directory structure
    runs = tmp_path / "bench" / "runs"
    runs.mkdir(parents=True)

    # Create a minimal policy
    policy = {
        "approval_thresholds": [
            {"max_usd": 2, "approver": "requesting-agent"},
            {"max_usd": 20, "approver": "one-of:CTO,CSO"},
            {"max_usd": 50, "approver": "two-of:CEO,CTO,CSO"},
            {"max_usd": None, "approver": "Jordan"},
        ],
        "per_run_hard_cap_usd": 35,
        "role_daily_ceiling_usd": {"CEO": 50, "CTO": 50, "CSO": 20, "COO": 10},
        "global_daily_budget_usd": 100,
    }
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "compute-policy.json").write_text(json.dumps(policy))

    # Create a valid receipt with all required fields
    receipt = {
        "experiment_id": "exp-test-001",
        "work_id": "experts4bit-qlora#123",
        "requested_by": "CTO/Cursor",
        "executed_by": "CTO/Cursor",
        "reviewed_by": None,
        "hypothesis": "Testing the check script",
        "expected_result": "Script validates receipt successfully",
        "success_criteria": "Receipt passes all validation checks",
        "failure_criteria": "Receipt fails validation or violates policy",
        "preregistration": "https://example.com/prereg",
        "approvals": [
            {
                "role": "CTO",
                "agent": "Cursor",
                "usd_estimate": 1.5,
                "slack_permalink": "https://example.com/slack/1",
            }
        ],
        "repo": "pjordanandrsn/experts4bit-qlora",
        "commit_sha": "abc123def456",
        "branch": "main",
        "dirty_tree": False,
        "container_image": "pytorch/pytorch:latest",
        "dependencies": {"torch": "2.0.0", "transformers": "4.30.0"},
        "command": "python train.py",
        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        "provider": "vast:verified-secure",
        "instance_id": "12345",
        "gpu_model": "RTX 5090",
        "gpu_count": 1,
        "cpu": "Intel Xeon 16 cores",
        "ram": "64GB",
        "storage": "500GB NVMe SSD",
        "started_at": "2026-09-06T10:00:00Z",
        "finished_at": "2026-09-06T11:30:00Z",
        "runtime_seconds": 5400,
        "cost_usd": {"estimated": 1.5, "actual": 1.45},
        "dataset": "test-dataset",
        "dataset_hash": "d" * 64,
        "model": "test-model",
        "model_revision": "main",
        "model_hash": None,
        "seed": 42,
        "configuration": {"batch_size": 4, "learning_rate": 0.001},
        "metrics": {"loss": 0.5, "accuracy": 0.95},
        "artifacts": [{"path": "model.pt", "sha256": "a" * 64, "bytes": 1024}],
        "teardown_proof": {
            "method": "vast-cli-destroy",
            "evidence": "Instance 12345 destroyed at 2026-09-06T11:31:00Z",
        },
        "status": "OK",
        "result": "pass",
        "decision": "merge https://github.com/example/repo/issues/1",
        "notes": "Test run completed successfully",
    }

    date_dir = runs / "2026-09-06" / "exp-test-001"
    date_dir.mkdir(parents=True)
    (date_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

    # Create ledger
    ledger_line = json.dumps(
        {"run_id": "exp-test-001", "date_utc": "2026-09-06", "role": "CTO", "cost_usd": 1.45}
    )
    (runs / "ledger.jsonl").write_text(f"# Ledger\n{ledger_line}\n")

    # Run the check
    result = subprocess.run(  # noqa: PLW1510
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Check failed: {result.stderr}"
    assert "OK: 1 receipts, 1 days" in result.stdout


def test_over_ceiling_day(tmp_path: Path) -> None:
    """Two receipts on the same day exceed the role daily ceiling."""
    runs = tmp_path / "bench" / "runs"
    runs.mkdir(parents=True)

    # COO ceiling is $10
    policy = {
        "approval_thresholds": [
            {"max_usd": 2, "approver": "requesting-agent"},
            {"max_usd": 20, "approver": "one-of:CTO,CSO"},
        ],
        "per_run_hard_cap_usd": 35,
        "role_daily_ceiling_usd": {"COO": 10},
        "global_daily_budget_usd": 100,
    }
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "compute-policy.json").write_text(json.dumps(policy))

    # Create two receipts totaling $12 (over the $10 ceiling)
    for i, cost in enumerate([6.0, 6.5], start=1):
        receipt = {
            "experiment_id": f"exp-over-{i:03d}",
            "work_id": "experts4bit-qlora#124",
            "requested_by": "COO/Forge",
            "executed_by": "COO/Forge",
            "reviewed_by": None,
            "hypothesis": "Testing ceiling check",
            "expected_result": "Daily ceiling enforcement triggers",
            "success_criteria": "Check script detects over-ceiling spend",
            "failure_criteria": "Check script fails to detect violation",
            "preregistration": "https://example.com/prereg",
            "approvals": [
                {
                    "role": "COO",
                    "agent": "Forge",
                    "usd_estimate": cost,
                    "slack_permalink": f"https://example.com/slack/{i}",
                }
            ],
            "repo": "pjordanandrsn/experts4bit-qlora",
            "commit_sha": "def456abc789",
            "branch": "test-branch",
            "dirty_tree": False,
            "container_image": "pytorch/pytorch:latest",
            "dependencies": {"torch": "2.0.0"},
            "command": "python bench.py",
            "environment": {},
            "provider": "vast:verified-secure",
            "instance_id": f"9999{i}",
            "gpu_model": "RTX 4090",
            "gpu_count": 1,
            "cpu": "AMD EPYC 8 cores",
            "ram": "32GB",
            "storage": "250GB SSD",
            "started_at": "2026-09-07T12:00:00Z",
            "finished_at": "2026-09-07T13:00:00Z",
            "runtime_seconds": 3600,
            "cost_usd": {"estimated": cost, "actual": cost},
            "dataset": "benchmark-data",
            "dataset_hash": None,
            "model": "test-model-v2",
            "model_revision": "v2.0",
            "model_hash": None,
            "seed": None,
            "configuration": {},
            "metrics": {},
            "artifacts": [],
            "teardown_proof": {"method": "vast-cli-destroy", "evidence": f"Instance destroyed {i}"},
            "status": "OK",
            "result": "pass",
            "decision": "continue https://github.com/example/repo/issues/2",
            "notes": f"Benchmark run {i}",
        }

        date_dir = runs / "2026-09-07" / f"exp-over-{i:03d}"
        date_dir.mkdir(parents=True)
        (date_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

    # Create ledger
    ledger = "\n".join(
        [
            "# Ledger",
            json.dumps(
                {"run_id": "exp-over-001", "date_utc": "2026-09-07", "role": "COO", "cost_usd": 6.0}
            ),
            json.dumps(
                {"run_id": "exp-over-002", "date_utc": "2026-09-07", "role": "COO", "cost_usd": 6.5}
            ),
        ]
    )
    (runs / "ledger.jsonl").write_text(ledger + "\n")

    # Run the check
    result = subprocess.run(  # noqa: PLW1510
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, "Check should fail on over-ceiling"
    assert "exceeds ceiling" in result.stderr.lower()


def test_missing_approval(tmp_path: Path) -> None:
    """A receipt with estimated cost $15 lacks the required CTO/CSO approval."""
    runs = tmp_path / "bench" / "runs"
    runs.mkdir(parents=True)

    policy = {
        "approval_thresholds": [
            {"max_usd": 2, "approver": "requesting-agent"},
            {"max_usd": 20, "approver": "one-of:CTO,CSO"},  # $15 needs CTO or CSO
        ],
        "per_run_hard_cap_usd": 35,
        "role_daily_ceiling_usd": {"COO": 50},
        "global_daily_budget_usd": 100,
    }
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "compute-policy.json").write_text(json.dumps(policy))

    # Receipt with $15 estimate but only self-approval (COO)
    receipt = {
        "experiment_id": "exp-no-approval",
        "work_id": "experts4bit-qlora#125",
        "requested_by": "COO/Forge",
        "executed_by": "COO/Forge",
        "reviewed_by": None,
        "hypothesis": "Check will catch missing approval",
        "expected_result": "Approval check fails for $15 run without CTO/CSO approval",
        "success_criteria": "Check script rejects receipt",
        "failure_criteria": "Check script accepts invalid approval",
        "preregistration": "https://example.com/prereg",
        "approvals": [
            {
                "role": "COO",
                "agent": "Forge",
                "usd_estimate": 15.0,
                "slack_permalink": "https://example.com/slack/99",
            }
        ],
        "repo": "pjordanandrsn/experts4bit-qlora",
        "commit_sha": "1234567890ab",
        "branch": "approval-test",
        "dirty_tree": False,
        "container_image": "pytorch/pytorch:2.0",
        "dependencies": {"torch": "2.0.1", "transformers": "4.31.0"},
        "command": "python train.py --epochs 10",
        "environment": {"WANDB_MODE": "offline"},
        "provider": "runpod:secure",
        "instance_id": "rp-8888",
        "gpu_model": "L40S",
        "gpu_count": 1,
        "cpu": "Intel Xeon 12 cores",
        "ram": "48GB",
        "storage": "400GB NVMe",
        "started_at": "2026-09-08T08:00:00Z",
        "finished_at": "2026-09-08T10:00:00Z",
        "runtime_seconds": 7200,
        "cost_usd": {"estimated": 15.0, "actual": 14.8},
        "dataset": "training-set-v1",
        "dataset_hash": "e" * 64,
        "model": "llama-7b",
        "model_revision": "main",
        "model_hash": "f" * 64,
        "seed": 12345,
        "configuration": {"epochs": 10, "batch_size": 8},
        "metrics": {"final_loss": 1.2},
        "artifacts": [{"path": "checkpoint.pt", "sha256": "b" * 64, "bytes": 2048}],
        "teardown_proof": {"method": "runpod-api-terminate", "evidence": "Terminated via API"},
        "status": "OK",
        "result": "pass",
        "decision": "revise https://github.com/example/repo/issues/3",
        "notes": "Training completed but approval was insufficient",
    }

    date_dir = runs / "2026-09-08" / "exp-no-approval"
    date_dir.mkdir(parents=True)
    (date_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

    ledger = "\n".join(
        [
            "# Ledger",
            json.dumps(
                {
                    "run_id": "exp-no-approval",
                    "date_utc": "2026-09-08",
                    "role": "COO",
                    "cost_usd": 14.8,
                }
            ),
        ]
    )
    (runs / "ledger.jsonl").write_text(ledger + "\n")

    # Run the check
    result = subprocess.run(  # noqa: PLW1510
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, "Check should fail on missing approval"
    assert "requires one of" in result.stderr.lower()
