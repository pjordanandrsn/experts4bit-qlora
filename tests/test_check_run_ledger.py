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


def test_global_daily_budget_violation(tmp_path):
    """Test that the global daily budget is enforced."""
    import sys

    runs = tmp_path / "bench" / "runs"
    runs.mkdir(parents=True)

    # Policy with $100 global daily budget
    policy = {
        "approval_thresholds": [
            {"max_usd": 2, "approver": "requesting-agent"},
            {"max_usd": 50, "approver": "two-of:CEO,CTO,CSO"},
        ],
        "per_run_hard_cap_usd": 35,
        "role_daily_ceiling_usd": {"CEO": 100, "CTO": 100},
        "global_daily_budget_usd": 100,
    }
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "compute-policy.json").write_text(json.dumps(policy))

    # Two receipts that together exceed the global budget ($100)
    receipt1 = {
        "experiment_id": "exp-budget-1",
        "work_id": "experts4bit-qlora#200",
        "requested_by": "CEO/Claude",
        "executed_by": "CEO/Claude",
        "reviewed_by": None,
        "hypothesis": "First run on the day",
        "expected_result": "Completes successfully",
        "success_criteria": "Under individual ceiling",
        "failure_criteria": "Exceeds ceiling",
        "preregistration": "https://example.com/prereg1",
        "approvals": [
            {
                "role": "CEO",
                "agent": "Claude",
                "usd_estimate": 30.0,
                "slack_permalink": "https://example.com/slack/10",
            },
            {
                "role": "CTO",
                "agent": "Cursor",
                "usd_estimate": 30.0,
                "slack_permalink": "https://example.com/slack/11",
            },
        ],
        "repo": "pjordanandrsn/experts4bit-qlora",
        "commit_sha": "aaa111",
        "branch": "main",
        "dirty_tree": False,
        "container_image": "pytorch/pytorch:2.1",
        "dependencies": {"torch": "2.1.0"},
        "command": "python benchmark.py",
        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        "provider": "vast:verified-secure",
        "instance_id": "vast-111",
        "gpu_model": "RTX 5090",
        "gpu_count": 1,
        "cpu": "AMD Ryzen 16 cores",
        "ram": "64GB",
        "storage": "1TB NVMe",
        "started_at": "2026-09-10T09:00:00Z",
        "finished_at": "2026-09-10T11:00:00Z",
        "runtime_seconds": 7200,
        "cost_usd": {"estimated": 30.0, "actual": 28.5},
        "dataset": "bench-dataset",
        "dataset_hash": "g" * 64,
        "model": "llama-13b",
        "model_revision": "v1.0",
        "model_hash": "h" * 64,
        "seed": 1111,
        "configuration": {"batch_size": 16},
        "metrics": {"throughput": 120},
        "artifacts": [{"path": "results.json", "sha256": "c" * 64, "bytes": 512}],
        "teardown_proof": {"method": "vast-destroy", "evidence": "Instance terminated"},
        "status": "OK",
        "result": "pass",
        "decision": "merge https://github.com/example/repo/pull/10",
        "notes": "First run of the day",
    }

    receipt2 = {
        "experiment_id": "exp-budget-2",
        "work_id": "experts4bit-qlora#201",
        "requested_by": "CTO/Cursor",
        "executed_by": "CTO/Cursor",
        "reviewed_by": None,
        "hypothesis": "Second run on the day",
        "expected_result": "Completes successfully",
        "success_criteria": "Under individual ceiling",
        "failure_criteria": "Exceeds ceiling",
        "preregistration": "https://example.com/prereg2",
        "approvals": [
            {
                "role": "CEO",
                "agent": "Claude",
                "usd_estimate": 35.0,
                "slack_permalink": "https://example.com/slack/12",
            },
            {
                "role": "CSO",
                "agent": "ChatGPT",
                "usd_estimate": 35.0,
                "slack_permalink": "https://example.com/slack/13",
            },
        ],
        "repo": "pjordanandrsn/experts4bit-qlora",
        "commit_sha": "bbb222",
        "branch": "main",
        "dirty_tree": False,
        "container_image": "pytorch/pytorch:2.1",
        "dependencies": {"torch": "2.1.0"},
        "command": "python train.py",
        "environment": {"CUDA_VISIBLE_DEVICES": "0,1"},
        "provider": "vast:verified-secure",
        "instance_id": "vast-222",
        "gpu_model": "RTX 5090",
        "gpu_count": 2,
        "cpu": "AMD Ryzen 32 cores",
        "ram": "128GB",
        "storage": "2TB NVMe",
        "started_at": "2026-09-10T14:00:00Z",
        "finished_at": "2026-09-10T17:00:00Z",
        "runtime_seconds": 10800,
        "cost_usd": {"estimated": 35.0, "actual": 73.0},
        "dataset": "train-dataset",
        "dataset_hash": "i" * 64,
        "model": "llama-70b",
        "model_revision": "v2.0",
        "model_hash": "j" * 64,
        "seed": 2222,
        "configuration": {"batch_size": 32, "epochs": 5},
        "metrics": {"final_loss": 0.8},
        "artifacts": [{"path": "model.pt", "sha256": "d" * 64, "bytes": 4096}],
        "teardown_proof": {"method": "vast-destroy", "evidence": "Instance terminated"},
        "status": "OK",
        "result": "pass",
        "decision": "merge https://github.com/example/repo/pull/11",
        "notes": "Second run of the day - pushes total to $101.5",
    }

    # Create receipt directories
    date_dir1 = runs / "2026-09-10" / "exp-budget-1"
    date_dir1.mkdir(parents=True)
    (date_dir1 / "receipt.json").write_text(json.dumps(receipt1, indent=2))

    date_dir2 = runs / "2026-09-10" / "exp-budget-2"
    date_dir2.mkdir(parents=True)
    (date_dir2 / "receipt.json").write_text(json.dumps(receipt2, indent=2))

    # Ledger with both runs (total: $28.5 + $73.0 = $101.5, exceeds $100)
    ledger = "\n".join(
        [
            "# Ledger",
            json.dumps(
                {
                    "run_id": "exp-budget-1",
                    "date_utc": "2026-09-10",
                    "role": "CEO",
                    "cost_usd": 28.5,
                }
            ),
            json.dumps(
                {
                    "run_id": "exp-budget-2",
                    "date_utc": "2026-09-10",
                    "role": "CTO",
                    "cost_usd": 73.0,
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

    assert result.returncode == 1, "Check should fail on global budget violation"
    assert "global daily total" in result.stderr.lower()
    assert "$101.5" in result.stderr or "$101.50" in result.stderr
