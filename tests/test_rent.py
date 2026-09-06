"""Compute launcher (#430): policy refusals, guard-after-parent-death, receipt on fail."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from experts4bit_qlora.tools.rent import (
    FakeProvider, RentRefused, evaluate_launch, estimate_usd, main,
)

REPO = Path(__file__).resolve().parents[1]
POLICY = json.loads((REPO / "docs" / "compute-policy.json").read_text())
PERM = "https://cerin-amroth.slack.com/archives/C0BV5028SGM/p1"


def _cto(usd=5.0):
    return [{"role": "CTO", "agent": "Cursor", "usd_estimate": usd,
             "slack_permalink": PERM}]


def test_estimate_is_rate_times_wallclock():
    assert estimate_usd(usd_per_hour=2.0, wallclock_h=3.0) == 6.0


def test_over_ceiling_refuses_without_network():
    ledger = [{"run_id": "a", "date_utc": "2026-09-06", "role": "CTO", "cost_usd": 48.0}]
    with pytest.raises(RentRefused, match="exceeds ceiling"):
        evaluate_launch(POLICY, ledger, role="CTO", estimate=5.0,
                        provider="vast:verified-secure", gpu="RTX 5090",
                        wallclock_h=1.0, approvals=_cto(5), date_utc="2026-09-06")


def test_missing_approval_refuses():
    with pytest.raises(RentRefused, match="requires one of"):
        evaluate_launch(POLICY, [], role="CTO", estimate=10.0,
                        provider="vast:verified-secure", gpu="RTX 5090",
                        wallclock_h=1.0, approvals=[], date_utc="2026-09-06")


def test_disallowed_gpu_and_provider_refuse():
    with pytest.raises(RentRefused, match="gpu"):
        evaluate_launch(POLICY, [], role="CTO", estimate=1.0,
                        provider="vast:verified-secure", gpu="TITAN V",
                        wallclock_h=1.0, approvals=[], date_utc="2026-09-06")
    with pytest.raises(RentRefused, match="provider"):
        evaluate_launch(POLICY, [], role="CTO", estimate=1.0,
                        provider="aws", gpu="RTX 5090",
                        wallclock_h=1.0, approvals=[], date_utc="2026-09-06")


def test_cap_above_35_without_jordan_refuses():
    with pytest.raises(RentRefused, match="hard cap"):
        evaluate_launch(POLICY, [], role="CTO", estimate=36.0,
                        provider="vast:verified-secure", gpu="RTX 5090",
                        wallclock_h=1.0, approvals=_cto(36) + [
                            {"role": "CEO", "agent": "Claude", "usd_estimate": 36,
                             "slack_permalink": PERM}],
                        date_utc="2026-09-06")
    evaluate_launch(POLICY, [], role="CTO", estimate=36.0,
                    provider="vast:verified-secure", gpu="RTX 5090",
                    wallclock_h=1.0,
                    approvals=[{"role": "Jordan", "agent": "Jordan", "usd_estimate": 36,
                                "slack_permalink": PERM}],
                    date_utc="2026-09-06")


def test_self_approval_under_two_passes():
    evaluate_launch(POLICY, [], role="CTO", estimate=1.5,
                    provider="runpod:secure", gpu="L40S",
                    wallclock_h=1.0, approvals=[], date_utc="2026-09-06")


def test_cli_refusal_writes_receipt_and_ledger(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    runs = tmp_path / "runs"
    rc = main([
        "--policy", str(REPO / "docs" / "compute-policy.json"),
        "--ledger", str(ledger),
        "--runs-root", str(runs),
        "--run-id", "rent-refuse-1",
        "--dry-run",
        "--gpu", "TITAN V",
        "--usd-per-hour", "0.4",
        "--wallclock-h", "1",
    ])
    assert rc == 2
    receipts = list(runs.rglob("receipt.json"))
    assert len(receipts) == 1
    rec = json.loads(receipts[0].read_text())
    assert rec["status"] == "REFUSED"
    assert rec["complete"] is True
    assert rec["teardown_proof"]["method"] == "not-launched"
    assert "rent-refuse-1" in ledger.read_text()


def test_cli_dry_run_writes_complete_receipt(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    runs = tmp_path / "runs"
    rc = main([
        "--policy", str(REPO / "docs" / "compute-policy.json"),
        "--ledger", str(ledger),
        "--runs-root", str(runs),
        "--run-id", "rent-ok-1",
        "--dry-run",
        "--usd-per-hour", "0.4",
        "--wallclock-h", "1",
        "--fake-state", str(tmp_path / "fake.json"),
    ])
    assert rc == 0
    rec = json.loads(next((runs).rglob("receipt.json")).read_text())
    assert rec["status"] == "OK"
    assert rec["complete"] is True
    assert rec["teardown_proof"]["method"]
    assert rec["instance_id"] != "none"
    assert rec["cost_usd"]["estimated"] == 0.4


def test_failed_command_still_receipts_and_tears_down(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    runs = tmp_path / "runs"
    fake = tmp_path / "fake.json"
    rc = main([
        "--policy", str(REPO / "docs" / "compute-policy.json"),
        "--ledger", str(ledger),
        "--runs-root", str(runs),
        "--run-id", "rent-fail-1",
        "--dry-run",
        "--usd-per-hour", "0.4",
        "--wallclock-h", "1",
        "--fake-state", str(fake),
        "--command", "python -c 'raise SystemExit(7)'",
    ])
    rec = json.loads(next(runs.rglob("receipt.json")).read_text())
    assert rec["status"] == "HARNESS_ERROR"
    assert rec["teardown_proof"]["method"] != "pending"
    live = json.loads(fake.read_text()).get("live") if fake.exists() else []
    assert rec["instance_id"] not in (live or [])
    assert rc in (0, 1)


def test_guard_survives_parent_death(tmp_path: Path):
    fake = tmp_path / "fake.json"
    prov = FakeProvider(fake)
    iid = prov.launch(gpu="RTX 5090", wallclock_h=1, image="none")
    assert iid in prov.list_ids()
    hb = tmp_path / "heartbeat"
    hb.write_text("t")
    proof = tmp_path / "proof.json"
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from experts4bit_qlora.tools.rent import spawn_guard\n"
        "from pathlib import Path\n"
        "spawn_guard(python=sys.executable, module_args=["
        f"'--guard-worker', '--instance-id', {iid!r}, '--provider', 'fake',"
        f"'--fake-state', {str(fake)!r}, '--wallclock-s', '0.6',"
        f"'--heartbeat', {str(hb)!r}, '--proof', {str(proof)!r},"
        "'--heartbeat-timeout-s', '10'], log_path=Path("
        f"{str(tmp_path / 'g.log')!r}))\n"
        "os._exit(0)\n"
    )
    subprocess.run([sys.executable, str(parent)], check=True)
    deadline = time.time() + 8
    while time.time() < deadline:
        if proof.is_file() and iid not in FakeProvider(fake).list_ids():
            break
        time.sleep(0.1)
    assert proof.is_file(), "guard did not write teardown proof after parent exit"
    body = json.loads(proof.read_text())
    assert body["complete"] is True
    assert iid not in FakeProvider(fake).list_ids()
