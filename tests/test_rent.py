"""Compute launcher (#430): policy refusals, executor-conditional approvals, heartbeat, guard-after-parent-death,
schema-valid receipts on every path -- including the PR #440 review fixes."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from experts4bit_qlora.tools.rent import (
    FakeProvider, ReceiptInvalid, RentRefused, check_approvals, evaluate_launch, estimate_usd, git_facts,
    main, select_approver_spec, validate_receipt,
)

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "docs" / "compute-policy.json"
POLICY = json.loads(POLICY_PATH.read_text())
SCHEMA = REPO / "docs" / "run-receipt-schema.json"
PERM = "https://cerin-amroth.slack.com/archives/C0BV5028SGM/p1788680181409539"
GROK = {"CTO": "cursor-desktop-mini/grok"}


def _appr(role, agent="x", usd=5.0, perm=PERM):
    return {"role": role, "agent": agent, "usd_estimate": usd, "slack_permalink": perm}


def _cto(usd=5.0):
    return [_appr("CTO", "cursor-desktop-mini", usd)]


def _cli(tmp_path: Path, run_id: str, *extra: str, dry_run: bool = True) -> list[str]:
    """Every required identity field spelled out -- the launcher has no defaults for them."""
    args = [
        "--policy", str(POLICY_PATH), "--schema", str(SCHEMA),
        "--ledger", str(tmp_path / "ledger.jsonl"), "--runs-root", str(tmp_path / "runs"),
        "--run-id", run_id, "--role", "CTO", "--agent", "cursor-desktop-mini/grok",
        "--work-id", "experts4bit-qlora#430", "--preregistration", "docs/COMPUTE-GOVERNANCE.md#how-to-launch",
        "--hypothesis", "the launcher enforces the policy", "--expected-result", "a schema-valid receipt",
        "--success-criteria", "receipt complete with teardown proof", "--failure-criteria", "refusal or missing proof",
        "--usd-per-hour", "0.4", "--wallclock-h", "1", "--fake-state", str(tmp_path / f"{run_id}-fake.json"),
        *extra,
    ]
    if dry_run:
        args.append("--dry-run")
    return args


def _receipt(tmp_path: Path) -> dict:
    receipts = list((tmp_path / "runs").rglob("receipt.json"))
    assert len(receipts) == 1, receipts
    rec = json.loads(receipts[0].read_text())
    validate_receipt(rec, SCHEMA)  # every receipt the launcher writes satisfies the repo's schema
    return rec


# ---------------------------------------------------------------- policy refusals (no network)

def test_estimate_is_rate_times_wallclock():
    assert estimate_usd(usd_per_hour=2.0, wallclock_h=3.0) == 6.0


def test_over_ceiling_refuses_without_network():
    ledger = [{"run_id": "a", "date_utc": "2026-09-06", "role": "CTO", "cost_usd": 48.0}]
    with pytest.raises(RentRefused, match="exceeds ceiling"):
        evaluate_launch(POLICY, ledger, role="CTO", estimate=5.0, provider="vast:verified-secure", gpu="RTX 5090",
                        wallclock_h=1.0, approvals=_cto(5), date_utc="2026-09-06")


def test_global_daily_budget_refuses():
    ledger = [{"run_id": r, "date_utc": "2026-09-06", "role": role, "cost_usd": 48.0}
              for r, role in (("a", "CEO"), ("b", "CTO"))]
    with pytest.raises(RentRefused, match="exceeds budget"):
        evaluate_launch(POLICY, ledger, role="CSO", estimate=5.0, provider="vast:verified-secure", gpu="RTX 5090",
                        wallclock_h=1.0, approvals=[_appr("CSO")], date_utc="2026-09-06")


def test_role_without_ceiling_row_refuses():
    with pytest.raises(RentRefused, match="no numeric row"):
        evaluate_launch(POLICY, [], role="CXO", estimate=1.0, provider="vast:verified-secure", gpu="RTX 5090",
                        wallclock_h=1.0, approvals=[], date_utc="2026-09-06")


def test_missing_approval_refuses():
    kw = dict(provider="vast:verified-secure", gpu="RTX 5090", wallclock_h=1.0, date_utc="2026-09-06")
    with pytest.raises(RentRefused, match="requires all of"):      # undeclared CTO seat -> the override applies
        evaluate_launch(POLICY, [], role="CTO", estimate=10.0, approvals=[], **kw)
    with pytest.raises(RentRefused, match="requires one of"):      # seat declared and held elsewhere -> base spec
        evaluate_launch(POLICY, [], role="CTO", estimate=10.0, approvals=[], seat_executors={"CTO": "cursor-cloud-agent"}, **kw)


def test_two_of_band():
    kw = dict(provider="vast:verified-secure", gpu="RTX 5090", wallclock_h=1.0, date_utc="2026-09-06")
    with pytest.raises(RentRefused, match="requires two of"):
        evaluate_launch(POLICY, [], role="CTO", estimate=30.0, approvals=_cto(30), **kw)
    assert evaluate_launch(POLICY, [], role="CTO", estimate=30.0, approvals=_cto(30) + [_appr("CEO", "claude", 30)],
                           **kw) == "two-of:CEO,CTO,CSO"


def test_disallowed_gpu_and_provider_refuse():
    with pytest.raises(RentRefused, match="gpu"):
        evaluate_launch(POLICY, [], role="CTO", estimate=1.0, provider="vast:verified-secure", gpu="TITAN V",
                        wallclock_h=1.0, approvals=[], date_utc="2026-09-06")
    with pytest.raises(RentRefused, match="provider"):
        evaluate_launch(POLICY, [], role="CTO", estimate=1.0, provider="aws", gpu="RTX 5090",
                        wallclock_h=1.0, approvals=[], date_utc="2026-09-06")


def test_wallclock_above_policy_max_refuses():
    with pytest.raises(RentRefused, match="max_wallclock_h"):
        evaluate_launch(POLICY, [], role="CTO", estimate=1.0, provider="vast:verified-secure", gpu="RTX 5090",
                        wallclock_h=7.0, approvals=[], date_utc="2026-09-06")


def test_cap_above_35_without_jordan_refuses():
    kw = dict(provider="vast:verified-secure", gpu="RTX 5090", wallclock_h=1.0, date_utc="2026-09-06")
    with pytest.raises(RentRefused, match="hard cap"):
        evaluate_launch(POLICY, [], role="CTO", estimate=36.0, approvals=_cto(36) + [_appr("CEO", "claude", 36)], **kw)
    evaluate_launch(POLICY, [], role="CTO", estimate=36.0, approvals=[_appr("Jordan", "Jordan", 36)], **kw)


def test_jordan_approval_needs_role_agent_and_real_permalink():
    kw = dict(provider="vast:verified-secure", gpu="RTX 5090", wallclock_h=1.0, date_utc="2026-09-06")
    with pytest.raises(RentRefused, match="hard cap"):  # agent 'Jordan' under another role does not count
        evaluate_launch(POLICY, [], role="CTO", estimate=36.0, approvals=[_appr("CTO", "Jordan", 36)], **kw)
    with pytest.raises(RentRefused, match="hard cap"):  # a non-channel URL does not count
        evaluate_launch(POLICY, [], role="CTO", estimate=36.0,
                        approvals=[_appr("Jordan", "Jordan", 36, perm="https://slack.com/x")], **kw)


def test_permalink_must_be_a_channel_permalink():
    for bad in ("https://slack.com/x", "https://cerin-amroth.slack.com/", "https://cerin-amroth.slack.com/archives/C0BV5028SGM/p123"):
        with pytest.raises(RentRefused, match="permalink"):
            check_approvals(POLICY, 10.0, [_appr("CTO", "x", 10, perm=bad)], role="CTO")


def test_self_approval_under_two_passes():
    assert evaluate_launch(POLICY, [], role="CTO", estimate=1.5, provider="runpod:secure", gpu="L40S",
                           wallclock_h=1.0, approvals=[], date_utc="2026-09-06") == "requesting-agent"


# ---------------------------------------------------------------- HIGH-2: executor-conditional co-sign

def test_override_selects_all_of_while_grok_holds_the_cto_seat():
    assert select_approver_spec(POLICY, 10.0, GROK) == "all-of:CTO,CSO"
    assert select_approver_spec(POLICY, 10.0, {"CTO": "cursor-cloud-agent"}) == "one-of:CTO,CSO"   # declared, held elsewhere
    assert select_approver_spec(POLICY, 10.0, None) == "all-of:CTO,CSO"     # undeclared seat: fail closed (Warden 2a)
    assert select_approver_spec(POLICY, 10.0, {"CSO": "chatgpt"}) == "all-of:CTO,CSO"   # the CTO seat still undeclared
    assert select_approver_spec(POLICY, 2.0, GROK) == "requesting-agent"   # band is (2, 20]
    assert select_approver_spec(POLICY, 20.0, GROK) == "all-of:CTO,CSO"
    assert select_approver_spec(POLICY, 20.5, GROK) == "two-of:CEO,CTO,CSO"


def test_ten_dollars_needs_cto_and_cso_while_grok_holds_the_seat():
    with pytest.raises(RentRefused, match="requires all of"):
        check_approvals(POLICY, 10.0, _cto(10), role="CTO", seat_executors=GROK)
    assert check_approvals(POLICY, 10.0, _cto(10) + [_appr("CSO", "ChatGPT", 10)], role="CTO",
                           seat_executors=GROK) == "all-of:CTO,CSO"
    with pytest.raises(RentRefused, match="requires all of"):   # omitting --seat-executor is not the loophole
        check_approvals(POLICY, 10.0, _cto(10), role="CTO", seat_executors=None)
    assert check_approvals(POLICY, 10.0, _cto(10), role="CTO",
                           seat_executors={"CTO": "cursor-cloud-agent"}) == "one-of:CTO,CSO"


def test_ledger_check_applies_the_same_override():
    spec = importlib.util.spec_from_file_location("check_run_ledger", REPO / "scripts" / "check_run_ledger.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    th, ov = POLICY["approval_thresholds"], POLICY["approval_overrides"]
    with pytest.raises(SystemExit):
        mod.check_approval_threshold(Path("r.json"), 10.0, _cto(10), th, overrides=ov, seat_executors=GROK)
    mod.check_approval_threshold(Path("r.json"), 10.0, _cto(10) + [_appr("CSO", "ChatGPT", 10)], th,
                                 overrides=ov, seat_executors=GROK)
    with pytest.raises(SystemExit):   # undeclared seat -> the override applies in the ledger check too
        mod.check_approval_threshold(Path("r.json"), 10.0, _cto(10), th, overrides=ov, seat_executors=None)
    mod.check_approval_threshold(Path("r.json"), 10.0, _cto(10), th, overrides=ov,
                                 seat_executors={"CTO": "cursor-cloud-agent"})
    with pytest.raises(SystemExit):   # hi boundary: 20.0 is inside (2, 20]
        mod.check_approval_threshold(Path("r.json"), 20.0, _cto(20), th, overrides=ov, seat_executors=GROK)
    mod.check_approval_threshold(Path("r.json"), 20.0, _cto(20) + [_appr("CSO", "ChatGPT", 20)], th,
                                 overrides=ov, seat_executors=GROK)
    with pytest.raises(SystemExit):   # $30 is outside the override band: two-of applies whatever the seats say
        mod.check_approval_threshold(Path("r.json"), 30.0, _cto(30), th, overrides=ov, seat_executors=None)
    # Jordan in the ledger check: role AND agent "Jordan" with a valid permalink, nothing less
    mod.check_approval_threshold(Path("r.json"), 36.0, [_appr("Jordan", "Jordan", 36)], th, overrides=ov)
    with pytest.raises(SystemExit):
        mod.check_approval_threshold(Path("r.json"), 36.0, [_appr("CEO", "Jordan", 36)], th, overrides=ov)
    with pytest.raises(SystemExit):
        mod.check_approval_threshold(Path("r.json"), 36.0, [_appr("Jordan", "Jordan", 36, perm="https://slack.com/x")], th, overrides=ov)


# ---------------------------------------------------------------- receipts on every path

def test_cli_refusal_writes_schema_valid_receipt_and_ledger(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-refuse-1", "--gpu", "TITAN V"))
    assert rc == 2
    rec = _receipt(tmp_path)
    assert rec["status"] == "REFUSED" and rec["complete"] is True
    assert rec["teardown_proof"]["method"] == "not-launched"
    assert rec["requested_by"] == "CTO/cursor-desktop-mini/grok" and rec["work_id"] == "experts4bit-qlora#430"
    assert rec["commit_sha"] == git_facts(REPO)["commit_sha"] and rec["branch"] and isinstance(rec["command"], str)
    assert "rent-refuse-1" in (tmp_path / "ledger.jsonl").read_text()


def test_cli_refuses_ten_dollars_with_one_approval_while_grok_holds_the_seat(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-cosign-1", "--usd-per-hour", "5", "--wallclock-h", "2",
                   "--seat-executor", "CTO=cursor-desktop-mini/grok", "--approval", f"CTO/cursor-desktop-mini={PERM}"))
    assert rc == 2
    rec = _receipt(tmp_path)
    assert rec["status"] == "REFUSED" and "requires all of" in rec["notes"]
    assert rec["environment"]["approver_spec"] == "all-of:CTO,CSO"
    assert rec["environment"]["seat_executors"] == GROK


def test_cli_undeclared_seat_is_not_a_loophole(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-cosign-2", "--usd-per-hour", "5", "--wallclock-h", "2",
                   "--approval", f"CTO/cursor-desktop-mini={PERM}"))   # no --seat-executor at all
    assert rc == 2
    rec = _receipt(tmp_path)
    assert rec["status"] == "REFUSED" and rec["environment"]["approver_spec"] == "all-of:CTO,CSO"
    assert rec["environment"]["seat_executors"] == {}


def test_live_provider_refusal_writes_a_receipt(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-live-1", "--provider", "vast:verified-secure", dry_run=False))
    assert rc == 2
    rec = _receipt(tmp_path)
    assert rec["status"] == "REFUSED" and "not armed" in rec["notes"] and rec["instance_id"] == "none"


def test_cli_dry_run_writes_complete_receipt(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-ok-1"))
    assert rc == 0
    rec = _receipt(tmp_path)
    assert rec["status"] == "OK" and rec["result"] == "pass" and rec["complete"] is True
    assert rec["teardown_proof"]["reason"] == "completion" and rec["instance_id"] != "none"
    assert rec["cost_usd"] == {"estimated": 0.4, "actual": 0.0}
    assert rec["decision"] == "pending experts4bit-qlora#430" and rec["environment"]["approver_spec"] == "requesting-agent"
    assert rec["cpu"] == "unknown" and rec["teardown_proof"]["method"] == "fake-destroy"
    assert rec["artifacts"][0]["path"] == "receipt.json"


def test_failed_command_still_receipts_and_tears_down(tmp_path: Path):
    fake = tmp_path / "rent-fail-1-fake.json"
    rc = main(_cli(tmp_path, "rent-fail-1", "--command", f"{sys.executable} -c 'raise SystemExit(7)'"))
    rec = _receipt(tmp_path)
    assert rec["status"] == "HARNESS_ERROR" and rec["result"] == "fail" and "command exited 7" in rec["notes"]
    assert rec["teardown_proof"]["reason"] == "completion"
    assert rec["instance_id"] not in (json.loads(fake.read_text()).get("live") or [])
    assert rc == 1


def test_identity_fields_are_required(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["--policy", str(POLICY_PATH), "--dry-run", "--usd-per-hour", "0.4", "--wallclock-h", "1"])


def test_validate_receipt_rejects_placeholders():
    rec = json.loads(json.dumps({"commit_sha": "unknown", "command": ["a"]}))
    with pytest.raises(ReceiptInvalid):
        validate_receipt(rec, SCHEMA)


# ---------------------------------------------------------------- HIGH-1: heartbeat

def test_heartbeat_refresh_keeps_a_long_command_alive(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-hb-1", "--heartbeat-timeout-s", "1.0", "--heartbeat-refresh-s", "0.25",
                   "--command", "sleep 2.5"))
    rec = _receipt(tmp_path)
    assert rc == 0 and rec["status"] == "OK" and rec["result"] == "pass"
    assert rec["teardown_proof"]["reason"] == "completion"


def test_stalled_heartbeat_is_an_alarm_not_a_pass(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-hb-2", "--heartbeat-timeout-s", "1.0", "--heartbeat-refresh-s", "0",
                   "--command", "sleep 2.5"))
    rec = _receipt(tmp_path)
    assert rc == 1 and rec["status"] == "ALARM" and rec["result"] == "invalid"
    assert rec["teardown_proof"]["reason"] == "heartbeat-loss"
    assert rec["instance_id"] not in (json.loads((tmp_path / "rent-hb-2-fake.json").read_text()).get("live") or [])


def test_external_teardown_during_the_command_is_an_alarm(tmp_path: Path):
    """1a: the instance disappears before the launcher's own teardown and no guard proof exists yet -> never pass."""
    fake = tmp_path / "rent-ext-1-fake.json"
    wipe = f"{sys.executable} -c \"import json; p={str(fake)!r}; d=json.load(open(p)); d['live']=[]; json.dump(d, open(p,'w'))\""
    rc = main(_cli(tmp_path, "rent-ext-1", "--heartbeat-timeout-s", "30", "--command", wipe))
    rec = _receipt(tmp_path)
    assert rc == 1 and rec["status"] == "ALARM" and rec["result"] == "invalid"
    assert rec["teardown_proof"]["reason"] in ("already-gone", "torn-down-externally")


def test_validate_receipt_refuses_without_a_schema_file(tmp_path: Path):
    with pytest.raises(ReceiptInvalid, match="schema not found"):
        validate_receipt({"commit_sha": "abc1234"}, tmp_path / "missing.json")


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
    assert body["complete"] is True and body["reason"] == "wallclock"
    assert iid not in FakeProvider(fake).list_ids()
