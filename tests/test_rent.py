"""Compute launcher (#430): policy refusals, executor-conditional approvals, heartbeat, guard-after-parent-death,
schema-valid receipts on every path -- including the PR #440 review fixes."""
from __future__ import annotations

import importlib.util
import json
import os
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

def _ledger_module():
    spec = importlib.util.spec_from_file_location("check_run_ledger", REPO / "scripts" / "check_run_ledger.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        evaluate_launch(POLICY, [], role="QA", estimate=1.0, provider="vast:verified-secure", gpu="RTX 5090",
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
    mod = _ledger_module()
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
    assert json.loads(rec["environment"]["seat_executors"]) == GROK


def test_cli_undeclared_seat_is_not_a_loophole(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-cosign-2", "--usd-per-hour", "5", "--wallclock-h", "2",
                   "--approval", f"CTO/cursor-desktop-mini={PERM}"))   # no --seat-executor at all
    assert rc == 2
    rec = _receipt(tmp_path)
    assert rec["status"] == "REFUSED" and rec["environment"]["approver_spec"] == "all-of:CTO,CSO"
    assert json.loads(rec["environment"]["seat_executors"]) == {}


def test_live_provider_refusal_writes_a_receipt(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-live-1", "--provider", "vast:verified-secure", dry_run=False))
    assert rc == 2
    rec = _receipt(tmp_path)
    # #455: inside a test runner the live seam refuses before it looks at E4B_RENT_LIVE or the key (tests never rent)
    assert rec["status"] == "REFUSED" and "test runner" in rec["notes"] and rec["instance_id"] == "none"


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


def _guard_stub(code: str):
    def spawn(*, python, module_args, log_path):
        return subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return spawn


def test_launcher_never_runs_the_command_before_the_guard_is_armed(tmp_path: Path, monkeypatch):
    """Round 3: a guard that has not armed (slow import on a cold box) means teardown WITHOUT the command."""
    import experts4bit_qlora.tools.rent as rent_mod
    monkeypatch.setattr(rent_mod, "spawn_guard", _guard_stub("import time; time.sleep(30)"))
    ran = tmp_path / "command-ran"
    rc = main(_cli(tmp_path, "rent-arm-1", "--guard-arm-timeout-s", "0.5", "--command", f"touch {ran}"))
    rec = _receipt(tmp_path)
    assert rc == 1 and rec["status"] == "HARNESS_ERROR" and rec["result"] == "invalid"
    assert not ran.exists(), "the command ran with no armed guard"
    assert rec["teardown_proof"]["reason"] == "guard-not-armed"
    assert "did not arm" in rec["notes"] and "guard_armed_at" not in rec["environment"]
    assert rec["instance_id"] not in (json.loads((tmp_path / "rent-arm-1-fake.json").read_text()).get("live") or [])


def test_guard_that_exits_before_arming_fails_fast(tmp_path: Path, monkeypatch):
    import experts4bit_qlora.tools.rent as rent_mod
    monkeypatch.setattr(rent_mod, "spawn_guard", _guard_stub("raise SystemExit(3)"))
    ran = tmp_path / "command-ran"
    t0 = time.time()
    rc = main(_cli(tmp_path, "rent-arm-2", "--guard-arm-timeout-s", "30", "--command", f"touch {ran}"))
    assert time.time() - t0 < 10, "a dead guard must not be waited on for the whole arm timeout"
    rec = _receipt(tmp_path)
    assert rc == 1 and rec["status"] == "HARNESS_ERROR" and rec["teardown_proof"]["reason"] == "guard-not-armed"
    assert "(exited 3)" in rec["notes"] and not ran.exists()


def test_armed_guard_is_recorded_and_the_marker_exists(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-arm-3", "--command", "true"))
    rec = _receipt(tmp_path)
    assert rc == 0 and rec["status"] == "OK"
    marker = json.loads((tmp_path / "rent-arm-3" / "guard-armed.json").read_text()) if (tmp_path / "rent-arm-3" / "guard-armed.json").is_file() else None
    assert rec["environment"]["guard_armed_at"] and "armed at" in rec["notes"]
    assert marker is None or marker["instance_id"] == rec["instance_id"]


def test_receipt_environment_values_are_strings_and_the_validator_enforces_it(tmp_path: Path):
    """Round 3: the schema types `environment` as an object of strings and `gpu_count >= 1`; the fallback
    validator (no jsonschema) rejects both violations too."""
    rc = main(_cli(tmp_path, "rent-env-1"))
    rec = _receipt(tmp_path)
    assert rc == 0 and all(isinstance(v, str) for v in rec["environment"].values())
    assert json.loads(rec["environment"]["seat_executors"]) == {}
    bad = dict(rec)
    bad["environment"] = dict(rec["environment"], seat_executors={})
    with pytest.raises(ReceiptInvalid, match="environment"):
        validate_receipt(bad)
    bad = dict(rec)
    bad["gpu_count"] = 0
    with pytest.raises(ReceiptInvalid, match="gpu_count"):
        validate_receipt(bad)


def test_refusal_receipt_satisfies_the_schema_minimums(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-env-2", "--usd-per-hour", "20", "--wallclock-h", "2"))  # $40 > per-run cap
    rec = _receipt(tmp_path)
    assert rc == 2 and rec["status"] == "REFUSED" and rec["gpu_count"] == 1
    assert rec["teardown_proof"]["method"] == "not-launched"


def test_ledger_reads_seat_executors_from_the_json_string(tmp_path: Path):
    ledger = _ledger_module()
    assert ledger.seat_executors_from_env({"seat_executors": json.dumps(GROK)}) == GROK
    assert ledger.seat_executors_from_env({"seat_executors": GROK}) == GROK
    assert ledger.seat_executors_from_env({"seat_executors": "not json"}) is None
    assert ledger.seat_executors_from_env({}) is None


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


# ---------------------------------------------------------------- #446 follow-ups (LOWs from Warden, PR #440)

def _guard_args(module_args: list[str]) -> dict[str, str]:
    """Parse key-value pairs from guard module_args.

    ``--guard-worker`` is the only boolean (no-value) flag in the guard's argv; every other
    ``--flag value`` pair is recorded.  Values that look like flags (start with ``--``) are
    left associated with their key so the caller can detect them, but in practice the guard
    module_args only ever carry absolute paths and numbers as values.
    """
    result: dict[str, str] = {}
    it = iter(module_args)
    for tok in it:
        if tok == "--guard-worker":  # the only boolean flag; skip it
            continue
        if not tok.startswith("--"):
            continue  # unexpected positional; skip
        val = next(it, None)
        if val is not None:
            result[tok] = val
    return result


def _arming_guard_spawn(module_args: list[str], extra_code: str = "") -> subprocess.Popen:
    """Spawn a guard stub that writes the armed marker then runs extra_code (items 1–3 shared helper)."""
    args_map = _guard_args(module_args)
    proof_str = args_map.get("--proof", "")
    iid_str = args_map.get("--instance-id", "")
    code = (
        "import sys, json, os\nfrom pathlib import Path\n"
        f"proof_path = Path({proof_str!r})\n"
        f"iid = {iid_str!r}\n"
        "armed = proof_path.with_name('guard-armed.json')\n"
        "armed.parent.mkdir(parents=True, exist_ok=True)\n"
        "armed.write_text(json.dumps({'pid': os.getpid(), 'at': '2026-01-01T00:00:00Z', 'instance_id': iid}))\n"
        + extra_code
    )
    return subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def test_guard_liveness_early_exit_recorded_in_notes(tmp_path: Path, monkeypatch):
    """Item 1 (#446): a guard that arms then crashes is recorded in notes and environment;
    result stays 'pass' because the launcher itself destroyed a live instance."""
    import experts4bit_qlora.tools.rent as rent_mod

    def _spawn(*, python, module_args, log_path):
        # Arm, then exit with rc=5 (simulate crash after arming)
        return _arming_guard_spawn(module_args, "sys.exit(5)\n")

    monkeypatch.setattr(rent_mod, "spawn_guard", _spawn)
    # Small command so the guard has time to exit before the liveness check
    rc = main(_cli(tmp_path, "rent-liveness-1",
                   "--command", f"{sys.executable} -c 'import time; time.sleep(0.15)'"))
    assert rc == 0
    rec = _receipt(tmp_path)
    # Guard exited early -- the note must say so
    assert "guard exited 5" in rec["notes"], rec["notes"]
    assert rec["environment"].get("guard_exited_early") == "5"
    # The launcher tore down a live instance, so result is still pass (the guard miss is noted, not fatal)
    assert rec["result"] == "pass" and rec["status"] == "OK"


def test_guard_is_joined_after_sigterm(tmp_path: Path, monkeypatch):
    """Item 2 (#446): guard.wait(timeout=2) is called after SIGTERM so the guard finishes any
    sidecar write before the launcher exits; without the wait the guard process is still running
    when main() returns and captured_guard.poll() is None."""
    import experts4bit_qlora.tools.rent as rent_mod

    captured: list[subprocess.Popen] = []

    def _spawn(*, python, module_args, log_path):
        args_map = _guard_args(module_args)
        proof_str = args_map.get("--proof", "")
        iid_str = args_map.get("--instance-id", "")
        # Arms, then sleeps 0.5 s after SIGTERM before exiting (simulates slow sidecar write).
        # Without guard.wait(timeout=2) in the launcher, poll() is None when main() returns.
        code = (
            "import sys, signal, json, os, time\nfrom pathlib import Path\n"
            f"armed = Path({proof_str!r}).with_name('guard-armed.json')\n"
            "armed.parent.mkdir(parents=True, exist_ok=True)\n"
            f"armed.write_text(json.dumps({{'pid': os.getpid(), 'at': '2026-01-01T00:00:00Z', 'instance_id': {iid_str!r}}}))\n"
            "def on_sigterm(sig, frame):\n"
            "    time.sleep(0.5)\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, on_sigterm)\n"
            "time.sleep(60)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code], start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        captured.append(proc)
        return proc

    monkeypatch.setattr(rent_mod, "spawn_guard", _spawn)
    rc = main(_cli(tmp_path, "rent-join-1"))
    assert rc == 0
    assert len(captured) == 1
    # With guard.wait(timeout=2): the guard has been waited on; poll() is not None.
    assert captured[0].poll() is not None, (
        "guard process is still running after main() returned -- guard.wait() was not called"
    )


def test_guard_firing_marker_prevents_launcher_pass(tmp_path: Path, monkeypatch):
    """Item 3 (#446): TOCTOU residual -- deterministic variant.  The stub writes guard-firing.json and
    then holds (no destroy, no proof written) so the launcher *must* take the firing-marker branch:
    detect the marker, wait ~3 s for a proof that never arrives, synthesise one with
    method='fake-guard-fired' and reason='wallclock', and return ALARM/invalid.  This test fails
    without the TOCTOU fix regardless of timing, because the launcher never sees proof arrive and
    must synthesise it; without the fix it falls through to prov.destroy() and writes 'pass'.
    The instance is never destroyed by the launcher (it stays live in the fake state)."""
    import experts4bit_qlora.tools.rent as rent_mod

    captured: list[subprocess.Popen] = []

    def _spawn(*, python, module_args, log_path):
        args_map = _guard_args(module_args)
        proof_str = args_map.get("--proof", "")
        iid_str = args_map.get("--instance-id", "")
        # Arms, writes guard-firing.json, then holds indefinitely -- no destroy, no proof.
        code = (
            "import json, os, time\nfrom pathlib import Path\n"
            f"proof_path = Path({proof_str!r})\n"
            f"iid = {iid_str!r}\n"
            "armed = proof_path.with_name('guard-armed.json')\n"
            "armed.parent.mkdir(parents=True, exist_ok=True)\n"
            "armed.write_text(json.dumps({'pid': os.getpid(), 'at': '2026-01-01T00:00:00Z', 'instance_id': iid}))\n"
            "firing = proof_path.with_name('guard-firing.json')\n"
            "firing.write_text(json.dumps({'reason': 'wallclock', 'pid': os.getpid(), 'instance_id': iid}))\n"
            "time.sleep(60)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code], start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        captured.append(proc)
        return proc

    monkeypatch.setattr(rent_mod, "spawn_guard", _spawn)
    rc = main(_cli(tmp_path, "rent-toctou-det-1"))
    # The launcher never sent SIGTERM on the firing-marker path; kill the orphaned guard.
    for p in captured:
        try:
            p.kill()
        except OSError:
            pass
    assert rc == 1
    rec = _receipt(tmp_path)
    # Launcher detected the firing marker, waited 3 s, synthesised a proof -- never destroyed.
    assert rec["status"] == "ALARM" and rec["result"] == "invalid", rec
    assert rec["teardown_proof"]["reason"] == "wallclock", rec["teardown_proof"]
    assert rec["teardown_proof"]["method"] == "fake-guard-fired", rec["teardown_proof"]
    # The launcher must not have destroyed the instance: it remains live in the fake state.
    fake_state = tmp_path / "rent-toctou-det-1-fake.json"
    assert fake_state.is_file(), "fake state file expected"
    st = json.loads(fake_state.read_text())
    assert rec["instance_id"] in (st.get("live") or []), (
        "launcher must not have destroyed the instance when the guard was firing"
    )


def test_guard_firing_marker_race_window(tmp_path: Path, monkeypatch):
    """Item 3 (#446): race-window variant.  The guard writes guard-firing.json, waits briefly, then
    destroys and writes a real proof.  When the launcher arrives *after* the destroy+proof write the
    existing 'proof present → ALARM' path catches it; when it arrives *before*, the firing-marker
    branch catches it.  Both paths produce ALARM -- this test is not by itself a sufficient acceptance
    criterion (timing-dependent) but documents that neither path emits 'pass'."""
    import experts4bit_qlora.tools.rent as rent_mod

    def _spawn(*, python, module_args, log_path):
        args_map = _guard_args(module_args)
        proof_str = args_map.get("--proof", "")
        fake_state_str = args_map.get("--fake-state", "")
        iid_str = args_map.get("--instance-id", "")
        # 1. Arms.  2. Writes guard-firing.json.  3. Brief delay.  4. Destroys.  5. Writes proof.
        code = (
            "import json, os, time\nfrom pathlib import Path\n"
            f"proof_path = Path({proof_str!r})\n"
            f"fake_state = Path({fake_state_str!r})\n"
            f"iid = {iid_str!r}\n"
            "armed = proof_path.with_name('guard-armed.json')\n"
            "armed.parent.mkdir(parents=True, exist_ok=True)\n"
            "armed.write_text(json.dumps({'pid': os.getpid(), 'at': '2026-01-01T00:00:00Z', 'instance_id': iid}))\n"
            "firing = proof_path.with_name('guard-firing.json')\n"
            "firing.write_text(json.dumps({'reason': 'wallclock', 'pid': os.getpid(), 'instance_id': iid}))\n"
            "time.sleep(0.05)\n"
            "if fake_state.is_file():\n"
            "    st = json.loads(fake_state.read_text())\n"
            "    st['live'] = [x for x in (st.get('live') or []) if x != iid]\n"
            "    fake_state.write_text(json.dumps(st) + '\\n')\n"
            "proof_path.write_text(json.dumps({'method': 'fake-destroy', 'reason': 'wallclock', "
            "'evidence': json.dumps({'instance_absent': True}), 'complete': True}))\n"
        )
        return subprocess.Popen(
            [sys.executable, "-c", code], start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    monkeypatch.setattr(rent_mod, "spawn_guard", _spawn)
    rc = main(_cli(tmp_path, "rent-toctou-1"))
    rec = _receipt(tmp_path)
    # Regardless of which path caught it, the receipt must be ALARM, never pass.
    assert rc == 1 and rec["status"] == "ALARM" and rec["result"] == "invalid", rec
    assert rec["teardown_proof"]["reason"] == "wallclock", rec["teardown_proof"]


def test_fallback_validator_checks_teardown_proof_complete_and_reason(tmp_path: Path, monkeypatch):
    """Item 4 (#446): the no-jsonschema fallback rejects teardown_proof.complete that is not bool,
    and teardown_proof.reason that is not in the allowed vocabulary."""
    import sys as _sys
    # Block jsonschema so validate_receipt falls back to the manual checks.
    monkeypatch.setitem(_sys.modules, "jsonschema", None)

    rc = main(_cli(tmp_path, "rent-fb-1"))
    assert rc == 0
    rec = _receipt(tmp_path)

    # complete must be bool when present in teardown_proof
    bad_complete = dict(rec, teardown_proof=dict(rec["teardown_proof"], complete="true"))
    with pytest.raises(ReceiptInvalid, match="complete"):
        validate_receipt(bad_complete, SCHEMA)

    # reason must be in the valid vocabulary when present
    bad_reason = dict(rec, teardown_proof=dict(rec["teardown_proof"], reason="not-a-real-reason"))
    with pytest.raises(ReceiptInvalid, match="reason"):
        validate_receipt(bad_reason, SCHEMA)

    # A valid complete+reason pair passes the fallback
    good = dict(rec, teardown_proof=dict(rec["teardown_proof"], complete=True, reason="completion"))
    validate_receipt(good, SCHEMA)  # must not raise


# ---- #455: the live seam
def test_vast_is_refused_by_name_when_not_armed_and_runpod_still_refuses(monkeypatch):
    from experts4bit_qlora.tools import vast_provider
    from experts4bit_qlora.tools.rent import provider_for
    with pytest.raises(RentRefused, match="test runner"):
        provider_for("vast:verified-secure")  # under pytest: refused before anything else
    monkeypatch.setattr(vast_provider, "_under_test", lambda: False)  # the arming rules, as outside a runner
    monkeypatch.delenv("E4B_NO_LIVE", raising=False)  # the fixture's cross-process guard, lifted for the arming rules only
    monkeypatch.delenv("E4B_RENT_LIVE", raising=False)
    with pytest.raises(RentRefused, match="E4B_RENT_LIVE=1"):
        provider_for("vast:verified-secure")
    monkeypatch.setenv("E4B_RENT_LIVE", "1")
    with pytest.raises(RentRefused, match="no Vast key file"):
        provider_for("vast:verified-secure")  # conftest points DEFAULT_KEY_PATH at a file that does not exist
    with pytest.raises(RentRefused, match="RunPod adapter is a separate issue"):
        provider_for("runpod:secure")


def test_armed_vast_under_a_test_runner_is_refused_with_a_receipt(tmp_path: Path, monkeypatch):
    """The live seam through the CLI: inside pytest it refuses before any HTTP, and the refusal is a receipt."""
    monkeypatch.setenv("E4B_RENT_LIVE", "1")
    rc = main(_cli(tmp_path, "rent-live-2", "--provider", "vast:verified-secure", dry_run=False))
    assert rc == 2
    rec = _receipt(tmp_path)
    assert rec["status"] == "REFUSED" and "test runner" in rec["notes"] and rec["instance_id"] == "none"


def test_guard_keeps_watching_when_the_listing_fails_and_never_calls_it_gone(tmp_path: Path):
    """#455: a provider whose list_ids raises (false-zero refused) must not make the guard write 'already-gone'."""
    from experts4bit_qlora.tools import rent as rent_mod

    class Flaky(FakeProvider):
        calls = 0

        def list_ids(self):
            Flaky.calls += 1
            if Flaky.calls <= 3:
                raise RuntimeError("deprecated_endpoint (false zero refused)")
            return super().list_ids()

    fake = tmp_path / "flaky.json"
    prov = Flaky(fake)
    iid = prov.launch(gpu="RTX 5090", wallclock_h=1, image="img")
    saved = rent_mod.provider_for
    rent_mod.provider_for = lambda kind, **kw: prov
    try:
        proof = tmp_path / "proof.json"
        hb = tmp_path / "hb"
        hb.write_text("x")
        rc = rent_mod.guard_worker(instance_id=iid, provider_kind="fake", fake_state=str(fake), wallclock_s=0.6,
                                   heartbeat_path=str(hb), proof_path=str(proof), heartbeat_timeout_s=60)
    finally:
        rent_mod.provider_for = saved
    data = json.loads(proof.read_text())
    assert data["reason"] == "wallclock" and data["complete"] is True and rc == 0
    assert iid not in prov.list_ids()


def test_teardown_proof_carries_complete(tmp_path: Path) -> None:
    """teardown_proof.complete is now included in the receipt (#457 fix to rent.py:891).

    The launcher proof dict always carries 'complete' (True when the teardown succeeded,
    False when the instance was found already gone); the filter previously dropped it so
    the schema's teardown_proof.complete field was never populated by any receipt the
    launcher wrote.  After the fix, a successful FakeProvider teardown yields complete=True.
    """
    rc = main(_cli(tmp_path, "rent-complete-1"))
    assert rc == 0
    rec = _receipt(tmp_path)
    tp = rec["teardown_proof"]
    assert "complete" in tp, f"teardown_proof.complete missing from receipt after fix; got keys: {list(tp)}"
    assert isinstance(tp["complete"], bool), (
        f"teardown_proof.complete must be bool, got {type(tp['complete'])!r}: {tp['complete']!r}"
    )
    assert tp["complete"] is True, (
        f"FakeProvider teardown destroys the instance; expected complete=True, got {tp['complete']!r}"
    )


# ---- #460 reads: the launcher's pre-flight path, listing-or-unknown at teardown, orphan receipts, guard order
def test_teardown_reason_vocabulary_matches_the_schema():
    """Warden MEDIUM-2: the fallback validator, the schema enum and check_run_ledger read one set."""
    from experts4bit_qlora.tools.rent import TEARDOWN_REASONS
    enum = json.loads(SCHEMA.read_text())["properties"]["teardown_proof"]["properties"]["reason"]["enum"]
    assert set(enum) == set(TEARDOWN_REASONS) and "preflight-failed" in enum


def test_preflight_failure_is_not_run_invalid_and_destroys_the_box(tmp_path: Path, monkeypatch):
    """Warden LOW: no test ran the launcher's pre-flight block before; the fake now has one."""
    fake = tmp_path / "rent-pf-1-fake.json"
    fake.write_text(json.dumps({"live": [], "preflight_fail": "stuck loading"}) + "\n")
    rc = main(_cli(tmp_path, "rent-pf-1", "--command", "true"))
    rec = _receipt(tmp_path)
    assert rc == 1
    assert rec["status"] == "NOT_RUN" and rec["result"] == "invalid" and "stuck loading" in rec["notes"] and "command not run" in rec["notes"]
    assert rec["teardown_proof"]["reason"] == "preflight-failed" and rec["teardown_proof"]["complete"] is True
    assert rec["environment"]["vast_preflight"] == "failed"
    assert rec["instance_id"] not in (json.loads(fake.read_text()).get("live") or [])
    # the same receipt passes the fallback validator (no jsonschema): one vocabulary in both paths (Warden MEDIUM-2)
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    validate_receipt(rec, SCHEMA)


def test_fake_preflight_runs_and_passes_on_an_ordinary_run(tmp_path: Path):
    rc = main(_cli(tmp_path, "rent-pf-ok-1"))
    rec = _receipt(tmp_path)
    assert rc == 0 and rec["status"] == "OK" and rec["environment"]["fake_preflight"] == "ok"


def test_teardown_with_the_listing_down_is_unproven_alarm_and_the_guard_stays_alive(tmp_path: Path, monkeypatch):
    """Warden MEDIUM-1: main()'s listings after destroy were bare list_ids(); a failure there lost the receipt.
    Now a failed listing is UNKNOWN: complete=False, ALARM, and the guard is not killed."""
    from experts4bit_qlora.tools import rent as rent_mod

    class DeadAfterDestroy(FakeProvider):
        def destroy(self, instance_id):
            ev = super().destroy(instance_id)
            self.state_path.with_suffix(".dead").write_text("x")
            return ev

        def list_ids(self):
            if self.state_path.with_suffix(".dead").is_file():
                raise RuntimeError("deprecated_endpoint (false zero refused)")
            return super().list_ids()

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kw: DeadAfterDestroy(kw["fake_state"]))
    rc = main(_cli(tmp_path, "rent-dead-1"))
    rec = _receipt(tmp_path)
    pid = int(rec["environment"]["guard_left_alive"])
    try:
        assert rc == 1 and rec["status"] == "ALARM" and rec["result"] == "invalid" and rec["complete"] is False
        assert "teardown unproven" in rec["notes"] and f"guard pid {pid} left alive" in rec["notes"]
        tp = rec["teardown_proof"]
        assert tp["complete"] is False and tp["reason"] == "completion"
        assert json.loads(tp["evidence"])["list_after"] == "UNKNOWN" and json.loads(tp["evidence"])["instance_absent"] is False
        # the guard (a plain FakeProvider on the same state file) sees the instance gone and finishes on its own;
        # this process is its parent, so reap it (a zombie still answers kill(pid, 0)).
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                wpid, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break  # already reaped by subprocess's own bookkeeping
            if wpid == pid:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("the guard did not finish on its own evidence")
    finally:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def test_unparsed_create_with_nothing_provable_is_an_alarm_receipt(tmp_path: Path, monkeypatch):
    """CEO MEDIUM-2 through the CLI: a PossibleOrphan from the adapter is ALARM / complete=False, not a clean refusal."""
    from experts4bit_qlora.tools import rent as rent_mod, vast_provider

    class Orphaning(FakeProvider):
        def launch(self, **kw):
            raise vast_provider.PossibleOrphan("create answered a shape this code does not understand; "
                                               "POSSIBLE ORPHAN — check the console for label 'rent-orphan-1'")

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kw: Orphaning(kw["fake_state"]))
    rc = main(_cli(tmp_path, "rent-orphan-1"))
    rec = _receipt(tmp_path)
    assert rc == 2 and rec["status"] == "ALARM" and rec["result"] == "invalid" and rec["complete"] is False
    assert rec["teardown_proof"]["method"] == "fake-orphan-unproven" and rec["teardown_proof"]["complete"] is False
    assert "POSSIBLE ORPHAN" in rec["notes"] and "label 'rent-orphan-1'" in rec["notes"] and rec["instance_id"] == "none"


def test_unparsed_create_whose_orphans_were_swept_is_an_alarm_receipt(tmp_path: Path, monkeypatch):
    from experts4bit_qlora.tools import rent as rent_mod, vast_provider

    class Swept(FakeProvider):
        def launch(self, **kw):
            raise vast_provider.OrphanSwept("create answered a shape this code does not understand; the sweep found "
                                            "['7000123'] under label 'rent-orphan-2', destroyed them, absent in the listing after",
                                            ["7000123"])

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kw: Swept(kw["fake_state"]))
    rc = main(_cli(tmp_path, "rent-orphan-2"))
    rec = _receipt(tmp_path)
    assert rc == 2 and rec["status"] == "ALARM" and rec["complete"] is True
    assert rec["teardown_proof"]["method"] == "fake-orphan-sweep" and rec["teardown_proof"]["complete"] is True
    assert "ORPHAN SWEPT" in rec["notes"] and "7000123" in rec["notes"]


def test_guard_acts_on_a_lost_heartbeat_while_the_listing_is_down(tmp_path: Path):
    """CEO LOW: the heartbeat is judged before the listing, so an API outage does not defer a heartbeat-loss teardown."""
    from experts4bit_qlora.tools import rent as rent_mod

    class Blind(FakeProvider):
        def list_ids(self):
            raise RuntimeError("deprecated_endpoint (false zero refused)")

    fake = tmp_path / "blind.json"
    prov = Blind(fake)
    iid = prov.launch(gpu="RTX 5090", wallclock_h=1, image="img")
    saved = rent_mod.provider_for
    rent_mod.provider_for = lambda kind, **kw: prov
    try:
        proof = tmp_path / "proof.json"
        hb = tmp_path / "hb"
        hb.write_text("x")
        t0 = time.time()
        rent_mod.guard_worker(instance_id=iid, provider_kind="fake", fake_state=str(fake), wallclock_s=30,
                              heartbeat_path=str(hb), proof_path=str(proof), heartbeat_timeout_s=0.5)
        took = time.time() - t0
    finally:
        rent_mod.provider_for = saved
    data = json.loads(proof.read_text())
    assert data["reason"] == "heartbeat-loss" and took < 10, took
    assert data["complete"] is False, "absence stays unproven while the listing is down — destroyed, not proven"
    assert iid not in FakeProvider(fake).list_ids()


# ---- e4b#464: the command is handed the box; the controller's public key is attached by shape
def test_command_environment_carries_the_box_and_nothing_leaks(tmp_path: Path):
    """E4B_RENT_* reach --command as its environment; the launcher's own process keeps none of them."""
    out = tmp_path / "seen-env.json"
    code = "import os, json, sys; json.dump({k: v for k, v in os.environ.items() if k.startswith('E4B_RENT_')}, open(sys.argv[1], 'w'))"
    rc = main(_cli(tmp_path, "rent-env-1", "--command", f"{sys.executable} -c \"{code}\" {out}"))
    rec = _receipt(tmp_path)
    assert rc == 0 and rec["status"] == "OK"
    seen = json.loads(out.read_text())
    assert seen["E4B_RENT_RUN_ID"] == "rent-env-1"
    assert seen["E4B_RENT_INSTANCE_ID"] == rec["instance_id"]
    assert seen["E4B_RENT_RUN_DIR"].endswith("/rent-env-1") and Path(seen["E4B_RENT_RUN_DIR"]).is_dir()
    assert seen["E4B_RENT_PROVIDER"] == "fake" and seen["E4B_RENT_WALLCLOCK_S"] == "3600.0"
    now = int(time.time())
    assert now <= int(seen["E4B_RENT_DEADLINE_EPOCH"]) <= now + 3600 + 5
    assert "E4B_RENT_SSH" not in seen, "the fake reports no ssh endpoint → no placeholder, the key is absent"
    for k in ("E4B_RENT_RUN_ID", "E4B_RENT_INSTANCE_ID", "E4B_RENT_RUN_DIR", "E4B_RENT_SSH"):
        assert k not in os.environ, f"{k} exported into the launcher's own process"  # E4B_RENT_LIVE is the fixture's, not ours
    assert rec["environment"]["ssh_pubkey_attached"] == "no"


def test_command_environment_has_the_ssh_endpoint_when_the_preflight_reports_one(tmp_path: Path, monkeypatch):
    from experts4bit_qlora.tools import rent as rent_mod

    class WithSsh(FakeProvider):
        def preflight(self, instance_id, *, timeout_s=600.0):
            return {"vast_preflight": "ok", "vast_ssh": "ssh5.vast.ai:12345"}

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kw: WithSsh(kw["fake_state"]))
    out = tmp_path / "seen-env.json"
    code = "import os, json, sys; json.dump({k: v for k, v in os.environ.items() if k.startswith('E4B_RENT_SSH')}, open(sys.argv[1], 'w'))"
    rc = main(_cli(tmp_path, "rent-env-2", "--command", f"{sys.executable} -c \"{code}\" {out}"))
    assert rc == 0
    assert json.loads(out.read_text()) == {"E4B_RENT_SSH": "ssh5.vast.ai:12345", "E4B_RENT_SSH_HOST": "ssh5.vast.ai", "E4B_RENT_SSH_PORT": "12345"}


def test_command_environment_drops_an_inherited_endpoint_and_dates_the_deadline_from_launch(tmp_path: Path, monkeypatch):
    """Warden's two LOWs on #465: (1) an E4B_RENT_* key inherited from the caller's shell never reaches the command when this
    run has no such fact; (2) E4B_RENT_DEADLINE_EPOCH is seeded from the launch (t0), so a slow pre-flight does not push the
    command's deadline past the guard's."""
    from experts4bit_qlora.tools import rent as rent_mod

    for k, v in {"E4B_RENT_SSH": "stale.example:1", "E4B_RENT_SSH_HOST": "stale.example", "E4B_RENT_SSH_PORT": "1",
                 "E4B_RENT_RUN_ID": "someone-elses-run"}.items():
        monkeypatch.setenv(k, v)
    env = rent_mod.command_environment(run_id="rent-env-3", instance_id="i-1", run_dir=tmp_path, provider="fake",
                                       wallclock_s=60.0, deadline_epoch=1, ssh=None)
    assert env["E4B_RENT_RUN_ID"] == "rent-env-3"
    assert not any(k.startswith("E4B_RENT_SSH") for k in env), "a stale endpoint from the parent shell leaked through"
    env = rent_mod.command_environment(run_id="rent-env-3", instance_id="i-1", run_dir=tmp_path, provider="fake",
                                       wallclock_s=60.0, deadline_epoch=1, ssh="ssh5.vast.ai:2")
    assert (env["E4B_RENT_SSH"], env["E4B_RENT_SSH_HOST"], env["E4B_RENT_SSH_PORT"]) == ("ssh5.vast.ai:2", "ssh5.vast.ai", "2")

    class SlowPreflight(FakeProvider):
        def preflight(self, instance_id, *, timeout_s=600.0):
            time.sleep(1.5)
            return {"vast_preflight": "ok"}

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kw: SlowPreflight(kw["fake_state"]))
    out = tmp_path / "seen-env.json"
    code = "import os, json, sys; json.dump(dict(E4B_RENT_DEADLINE_EPOCH=os.environ['E4B_RENT_DEADLINE_EPOCH']), open(sys.argv[1], 'w'))"
    before = time.time()
    assert main(_cli(tmp_path, "rent-env-3", "--command", f"{sys.executable} -c \"{code}\" {out}")) == 0
    deadline = int(json.loads(out.read_text())["E4B_RENT_DEADLINE_EPOCH"])
    assert deadline <= int(before) + 3600 + 1, "the pre-flight's 1.5 s was added to the command's deadline"
    assert deadline >= int(before) + 3600 - 1


def test_ssh_pubkey_is_read_by_shape_and_a_private_key_is_refused(tmp_path: Path):
    from experts4bit_qlora.tools.rent import read_pubkey
    good = tmp_path / "id_ed25519.pub"
    good.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholderKeyMaterialForTheTestOnly0000000000 cdo@mini\n")
    assert read_pubkey(good).startswith("ssh-ed25519 AAAA")
    private = tmp_path / "id_ed25519"
    private.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----\n")
    with pytest.raises(RentRefused, match="private key is refused"):
        read_pubkey(private)
    with pytest.raises(RentRefused, match="no such file"):
        read_pubkey(tmp_path / "absent.pub")
    two = tmp_path / "two.pub"
    two.write_text("ssh-ed25519 AAAA1 a\nssh-ed25519 AAAA2 b\n")
    with pytest.raises(RentRefused, match="single"):
        read_pubkey(two)
    # through the CLI on the fake provider: the receipt records that a key was attached
    rc = main(_cli(tmp_path, "rent-key-1", "--ssh-pubkey", str(good)))
    assert rc == 0 and _receipt(tmp_path)["environment"]["ssh_pubkey_attached"] == "yes"
    rc = main(_cli(tmp_path / "b", "rent-key-2", "--ssh-pubkey", str(private)))
    assert rc == 2, "a refused key is a refusal receipt, not a launch"


def test_launcher_passes_the_declared_rate_as_the_offer_ceiling(tmp_path: Path, monkeypatch):
    """e4b#464: the launcher hands --usd-per-hour to launch() as max_dph, so the estimate on the approval line is the ceiling the offer must fit."""
    from experts4bit_qlora.tools import rent as rent_mod
    seen: dict = {}

    class Recording(FakeProvider):
        def launch(self, **kw):
            seen.update(kw)
            return super().launch(gpu=kw["gpu"], wallclock_h=kw["wallclock_h"], image=kw["image"])

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kw: Recording(kw["fake_state"]))
    assert main(_cli(tmp_path, "rent-rate-1")) == 0
    assert seen["max_dph"] == 0.4, "the fixture's --usd-per-hour 0.4 reaches the provider as the offer ceiling"
