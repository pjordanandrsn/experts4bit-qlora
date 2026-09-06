"""Tests for scripts/check_run_ledger.py

Three fixtures:
  1. One passing receipt (valid, within ceilings, proper approvals)
  2. One over-ceiling day (exceeds role daily ceiling)
  3. One missing approval (insufficient approvals for the cost threshold)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


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

    # Create a valid receipt
    receipt = {
        "run_id": "exp-test-001",
        "requested_by": {"role": "CTO", "agent": "Cursor"},
        "why": "Test experiment",
        "hypothesis": "Testing the check script",
        "preregistration": "https://example.com/prereg",
        "approvals": [
            {
                "role": "CTO",
                "agent": "Cursor",
                "usd_estimate": 1.5,
                "slack_permalink": "https://example.com/slack/1",
            }
        ],
        "provider": "vast:verified-secure",
        "instance_id": "12345",
        "gpu": "RTX 5090",
        "image": "pytorch/pytorch:latest",
        "commits": {"experts4bit-qlora": "abc123def456"},
        "command": "python train.py",
        "start_utc": "2026-09-06T10:00:00Z",
        "end_utc": "2026-09-06T11:30:00Z",
        "cost_usd": {"estimated": 1.5, "actual": 1.45},
        "result": {"status": "OK", "summary": "Training completed successfully"},
        "artifacts": [{"path": "model.pt", "sha256": "a" * 64, "bytes": 1024}],
        "teardown_proof": {
            "method": "vast-cli-destroy",
            "evidence": "Instance 12345 destroyed at 2026-09-06T11:31:00Z",
        },
        "decision": {"status": "adopt", "link": "https://github.com/example/repo/issues/1"},
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
        ["python", "scripts/check_run_ledger.py"],
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
            "run_id": f"exp-over-{i:03d}",
            "requested_by": {"role": "COO", "agent": "Forge"},
            "why": f"Test experiment {i}",
            "hypothesis": "Testing ceiling check",
            "preregistration": "https://example.com/prereg",
            "approvals": [
                {
                    "role": "COO",
                    "agent": "Forge",
                    "usd_estimate": cost,
                    "slack_permalink": f"https://example.com/slack/{i}",
                }
            ],
            "provider": "vast:verified-secure",
            "instance_id": f"9999{i}",
            "gpu": "RTX 4090",
            "image": "pytorch/pytorch:latest",
            "commits": {"experts4bit-qlora": "def456abc789"},
            "command": "python bench.py",
            "start_utc": "2026-09-07T12:00:00Z",
            "end_utc": "2026-09-07T13:00:00Z",
            "cost_usd": {"estimated": cost, "actual": cost},
            "result": {"status": "OK", "summary": f"Run {i} completed"},
            "artifacts": [],
            "teardown_proof": {"method": "vast-cli-destroy", "evidence": f"Instance destroyed {i}"},
            "decision": {"status": "pending", "link": "https://github.com/example/repo/issues/2"},
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
        ["python", "scripts/check_run_ledger.py"],
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
        "run_id": "exp-no-approval",
        "requested_by": {"role": "COO", "agent": "Forge"},
        "why": "Test missing approval",
        "hypothesis": "Check will catch missing approval",
        "preregistration": "https://example.com/prereg",
        "approvals": [
            {
                "role": "COO",
                "agent": "Forge",
                "usd_estimate": 15.0,
                "slack_permalink": "https://example.com/slack/99",
            }
        ],
        "provider": "runpod:secure",
        "instance_id": "rp-8888",
        "gpu": "L40S",
        "image": "pytorch/pytorch:2.0",
        "commits": {"experts4bit-qlora": "1234567890ab"},
        "command": "python train.py --epochs 10",
        "start_utc": "2026-09-08T08:00:00Z",
        "end_utc": "2026-09-08T10:00:00Z",
        "cost_usd": {"estimated": 15.0, "actual": 14.8},
        "result": {"status": "OK", "summary": "Training done"},
        "artifacts": [{"path": "checkpoint.pt", "sha256": "b" * 64, "bytes": 2048}],
        "teardown_proof": {"method": "runpod-api-terminate", "evidence": "Terminated via API"},
        "decision": {"status": "void", "link": "https://github.com/example/repo/issues/3"},
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
        ["python", "scripts/check_run_ledger.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, "Check should fail on missing approval"
    assert "requires one of" in result.stderr.lower()
