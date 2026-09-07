"""Compute launcher (#430): policy refusals, executor-conditional approvals, heartbeat, guard-after-parent-death,
schema-valid receipts on every path -- including the PR #440 review fixes."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from experts4bit_qlora.tools.rent import (
    FakeProvider, ReceiptInvalid, RentRefused, acquire_live_lock, check_approvals, evaluate_launch, estimate_usd,
    _vast_anchor_exclusions, build_receipt, git_facts, main, select_approver_spec, spawn_guard, validate_receipt,
)

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "docs" / "compute-policy.json"
POLICY = json.loads(POLICY_PATH.read_text())
SCHEMA = REPO / "docs" / "run-receipt-schema.json"
PERM = "https://cerin-amroth.slack.com/archives/C0BV5028SGM/p1788680181409539"
GROK = {"CTO": "cursor-desktop-mini/grok"}
_MISSING = object()

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


def _canonical_anchor_refusal(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path / "receipt-repo" / "receipts" / "experts4bit-qlora"
    run_id, instance_id, machine_id = "p41-r1-granite-test", "50120000", "144661"
    run_dir = root / "2026-09-07" / run_id
    p41 = run_dir / "p41"
    (p41 / "logs").mkdir(parents=True)
    nonce = "a" * 64
    runner = b"#!/bin/bash\n# exact fixture runner\n"
    gate_runner = b"# exact fixture train-anchor gate\n"
    expected = {
        "repo": "pjordanandrsn/experts4bit-qlora",
        "receipt_repo_origin": "https://github.com/pjordanandrsn/adertha-agents.git",
        "executor": "CTO/claude-session-5c68525a",
        "work_id": "experts4bit-qlora#P41",
        "preregistration": "bench/p41/P41-PREREG.md",
        "gpu_model": "RTX 5090",
        "container_image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
        "dataset": "clinical",
        "dataset_hash": "d" * 64,
        "model": "ibm-granite/granite-3.1-3b-a800m-instruct",
        "model_revision": "a" * 40,
        "max_dph": 0.66,
        "p41_run_sha256": hashlib.sha256(runner).hexdigest(),
        "train_anchor_gate_sha256": hashlib.sha256(gate_runner).hexdigest(),
    }
    expected["approvals"] = [{"role": "Jordan", "agent": "Jordan", "usd_estimate": 2.64,
                              "slack_permalink": PERM}]
    teardown_evidence = {
        "destroy": {"method": "vast-destroy", "instance_id": instance_id, "http": 200},
        "instance_absent": True,
        "list_after": [],
    }
    receipt = build_receipt(
        experiment_id=run_id, work_id=expected["work_id"], requested_by=expected["executor"],
        executed_by=expected["executor"], hypothesis="registered hypothesis", expected_result="registered result",
        success_criteria="registered success", failure_criteria="registered failure",
        preregistration=expected["preregistration"], approvals=expected["approvals"],
        commit_sha="b" * 40, branch="main",
        dirty_tree=False, command="registered command", environment={"vast_machine_id": machine_id,
        "vast_offer_id": "49868233", "vast_contract_id": instance_id, "vast_verification": "verified",
        "vast_gpu_name": "RTX 5090", "vast_image": expected["container_image"],
        "vast_disk_space_gb": "320", "vast_cpu_ram_mb": str(128 * 1024), "vast_dph_total": "0.62",
        "vast_search_filter": json.dumps({"verified": {"eq": True}, "external": {"eq": False},
        "rentable": {"eq": True}, "type": "on-demand", "num_gpus": {"eq": 1},
        "gpu_name": {"eq": "RTX 5090"}, "disk_space": {"gte": 320},
        "cpu_ram": {"gte": 98 * 1024}, "order": [["dph_total", "asc"]]}, sort_keys=True)},
        provider="vast:verified-secure", instance_id=instance_id,
        gpu_model=expected["gpu_model"], gpu_count=1, started_at="2026-09-07T00:00:00Z",
        finished_at="2026-09-07T00:01:00Z", runtime_seconds=60, cost_estimated=2.64,
        cost_actual=0.02, teardown_proof={"method": "vast-destroy", "evidence": json.dumps(teardown_evidence),
        "complete": True, "reason": "completion"}, status="HARNESS_ERROR", result="fail",
        notes="strict anchor refusal", configuration={}, dataset=expected["dataset"],
        dataset_hash=expected["dataset_hash"], model=expected["model"],
        model_revision=expected["model_revision"], container_image=expected["container_image"], complete=True,
    )
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    files = {
        "P41_RUN_NONCE": nonce + "\n", "BOX_REFUSED": "", f"TP_DONE.{nonce}": "",
        f"P41_EXIT_CODE.{nonce}": "12\n", "INSTANCE_ID": instance_id + "\n",
        "box.json": json.dumps({"run_id": run_id, "instance_id": instance_id,
        "provider": "vast:verified-secure", "registered_gpu_class": "RTX 5090",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-4504a39e-e33d-bf3f-4b7a-792d3b07e62d"}) + "\n",
        "anchor.json": json.dumps({"status": "OK", "gpu": "NVIDIA GeForce RTX 5090"}) + "\n",
        "summary.txt": "ANCHOR rc=3 class=pcie-full/launch-fast\nBOX REFUSED by train anchor (rc=3)\n",
        "outer.log": "ANCHOR rc=3 class=pcie-full/launch-fast\nBOX REFUSED by train anchor (rc=3)\n",
    }
    for name, content in files.items():
        (p41 / name).write_text(content)
    (p41 / "logs" / "anchor_gate.log").write_text(
        "  launch.self_pair     1.0779  FATAL: >1.03\n"
        "  class                pcie-full/launch-fast  reported\n\nBOX REFUSED\n"
    )
    (p41 / "p41_run.sh").write_bytes(runner)
    (p41 / "train_anchor_gate.py").write_bytes(gate_runner)
    repo = tmp_path / "receipt-repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", expected["receipt_repo_origin"]], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"], check=True,
                            capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "update-ref", f"refs/remotes/origin/{branch}", head], check=True)
    subprocess.run(["git", "-C", str(repo), "branch", f"--set-upstream-to=origin/{branch}"], check=True,
                   capture_output=True)
    return root, run_dir / "receipt.json", expected


def _commit_anchor_fixture(root: Path, message: str) -> None:
    repo = root.parents[1]
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", message], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    upstream = subprocess.run(["git", "-C", str(repo), "rev-parse", "--symbolic-full-name", "@{upstream}"],
                              check=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "update-ref", upstream, head], check=True)


def test_vast_anchor_exclusions_accept_only_canonical_committed_strict_anchor_refusal(tmp_path: Path):
    root, receipt, expected = _canonical_anchor_refusal(tmp_path)
    machines, evidence = _vast_anchor_exclusions([str(receipt)], runs_root=root, expected=expected,
                                                  schema_path=SCHEMA)
    assert machines == {"144661"}
    assert evidence[0]["run_id"] == "p41-r1-granite-test"
    assert evidence[0]["machine_id"] == "144661"
    assert evidence[0]["path"] == "2026-09-07/p41-r1-granite-test/receipt.json"
    assert len(evidence[0]["git_blob"]) == 40 and len(evidence[0]["git_head"]) == 40


def test_vast_anchor_exclusions_empty_list_needs_no_receipt_git_repo(tmp_path: Path):
    assert _vast_anchor_exclusions([], runs_root=tmp_path / "absent", expected={}) == (set(), [])


@pytest.mark.parametrize("tamper", ["origin", "unpushed-head"])
def test_vast_anchor_exclusions_bind_canonical_origin_and_upstream(tmp_path: Path, tamper: str):
    root, receipt, expected = _canonical_anchor_refusal(tmp_path)
    repo = root.parents[1]
    if tamper == "origin":
        subprocess.run(["git", "-C", str(repo), "remote", "set-url", "origin", "https://example.com/fake.git"],
                       check=True)
    else:
        (repo / "extra").write_text("new commit\n")
        subprocess.run(["git", "-C", str(repo), "add", "extra"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "not in upstream"], check=True)
    with pytest.raises(RentRefused, match="origin mismatch|not the configured upstream"):
        _vast_anchor_exclusions([str(receipt)], runs_root=root, expected=expected, schema_path=SCHEMA)


def test_vast_anchor_exclusions_refuse_modified_or_non_anchor_evidence(tmp_path: Path):
    root, receipt, expected = _canonical_anchor_refusal(tmp_path)
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(RentRefused, match="differs from receipt Git HEAD"):
        _vast_anchor_exclusions([str(receipt)], runs_root=root, expected=expected, schema_path=SCHEMA)

    subprocess.run(["git", "-C", str(root.parents[1]), "checkout", "--", "."], check=True)
    gate = receipt.parent / "p41" / "logs" / "anchor_gate.log"
    gate.write_text("  class                pcie-full/launch-fast  reported\n\nBOX REFUSED\n")
    _commit_anchor_fixture(root, "remove refusal evidence")
    with pytest.raises(RentRefused, match="strict train-anchor refusal"):
        _vast_anchor_exclusions([str(receipt)], runs_root=root, expected=expected, schema_path=SCHEMA)


@pytest.mark.parametrize(("field", "value"), [("complete", 1), ("dirty_tree", 0)])
def test_vast_anchor_exclusions_require_exact_receipt_booleans(tmp_path: Path, field: str, value: int):
    root, receipt, expected = _canonical_anchor_refusal(tmp_path)
    payload = json.loads(receipt.read_text())
    payload[field] = value
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    _commit_anchor_fixture(root, "bad bool")
    with pytest.raises(RentRefused):
        _vast_anchor_exclusions([str(receipt)], runs_root=root, expected=expected, schema_path=SCHEMA)


def test_vast_anchor_exclusions_bind_the_original_approval_identity(tmp_path: Path):
    root, receipt, expected = _canonical_anchor_refusal(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["approvals"][0]["slack_permalink"] = (
        "https://cerin-amroth.slack.com/archives/C0BV5028SGM/p1788749828896999"
    )
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    _commit_anchor_fixture(root, "different approval")
    with pytest.raises(RentRefused, match="identity/status mismatch"):
        _vast_anchor_exclusions([str(receipt)], runs_root=root, expected=expected, schema_path=SCHEMA)


@pytest.mark.parametrize("tamper", ["verification", "gpu-name", "anchor-gpu", "numeric-bools", "nan-rate"])
def test_vast_anchor_exclusions_bind_verified_offer_and_exact_anchor_gpu(tmp_path: Path, tamper: str):
    root, receipt, expected = _canonical_anchor_refusal(tmp_path)
    if tamper == "anchor-gpu":
        anchor_path = receipt.parent / "p41" / "anchor.json"
        anchor = json.loads(anchor_path.read_text())
        anchor["gpu"] += " arbitrary suffix"
        anchor_path.write_text(json.dumps(anchor) + "\n")
    elif tamper in ("verification", "gpu-name"):
        payload = json.loads(receipt.read_text())
        key = "vast_verification" if tamper == "verification" else "vast_gpu_name"
        payload["environment"][key] = "unverified" if tamper == "verification" else "RTX 4090"
        receipt.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        payload = json.loads(receipt.read_text())
        if tamper == "nan-rate":
            payload["environment"]["vast_dph_total"] = "nan"
        else:
            recorded = json.loads(payload["environment"]["vast_search_filter"])
            recorded["verified"]["eq"] = 1
            recorded["external"]["eq"] = 0
            recorded["rentable"]["eq"] = 1
            payload["environment"]["vast_search_filter"] = json.dumps(recorded, sort_keys=True)
        receipt.write_text(json.dumps(payload, indent=2) + "\n")
    _commit_anchor_fixture(root, f"tamper {tamper}")
    with pytest.raises(RentRefused, match="offer/search contract mismatch|box/anchor identity mismatch"):
        _vast_anchor_exclusions([str(receipt)], runs_root=root, expected=expected, schema_path=SCHEMA)


@pytest.mark.parametrize("tamper", ["untracked-success", "success", "outcome", "done", "box-newline",
                                           "metrics", "foreign-run"])
def test_vast_anchor_exclusions_reject_adversarial_terminal_or_scope_evidence(tmp_path: Path, tamper: str):
    root, receipt, expected = _canonical_anchor_refusal(tmp_path)
    p41 = receipt.parent / "p41"
    nonce = (p41 / "P41_RUN_NONCE").read_text().strip()
    if tamper == "untracked-success":
        (p41 / f"P41_SUCCESS.{nonce}").write_text("")
    elif tamper == "success":
        (p41 / "P41_SUCCESS.foreign").write_text("")
    elif tamper == "outcome":
        (p41 / "P41_OUTCOME_COUNTS.foreign.json").write_text("{}\n")
    elif tamper == "done":
        (p41 / f"TP_DONE.{nonce}").write_text("foreign\n")
    elif tamper == "box-newline":
        (p41 / "BOX_REFUSED").write_text("\n")
    elif tamper == "metrics":
        payload = json.loads(receipt.read_text())
        payload["metrics"] = {"loss": 1.0}
        receipt.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        payload = json.loads(receipt.read_text())
        payload["experiment_id"] = "p41-r1-granite-foreign"
        receipt.write_text(json.dumps(payload, indent=2) + "\n")
        box = json.loads((p41 / "box.json").read_text())
        box["run_id"] = payload["experiment_id"]
        (p41 / "box.json").write_text(json.dumps(box) + "\n")
    if tamper != "untracked-success":
        _commit_anchor_fixture(root, f"tamper {tamper}")
    with pytest.raises(RentRefused):
        _vast_anchor_exclusions([str(receipt)], runs_root=root, expected=expected, schema_path=SCHEMA)


def test_invalid_anchor_exclusion_is_receipted_under_live_lock_before_provider(tmp_path: Path, monkeypatch):
    from experts4bit_qlora.tools import rent as rent_mod

    monkeypatch.setattr(rent_mod, "DEFAULT_LIVE_LOCK_PATH", tmp_path / "live.lock")
    invoked = []
    monkeypatch.setattr(rent_mod, "provider_for", lambda *args, **kwargs: invoked.append((args, kwargs)))
    missing = tmp_path / "not-canonical" / "receipt.json"
    rc = main(_cli(tmp_path, "bad-anchor-exclusion", "--provider", "fake",
                   "--exclude-vast-anchor-receipt", str(missing), dry_run=False))
    assert rc == 2 and invoked == []
    rec = _receipt(tmp_path)
    assert rec["status"] == "REFUSED" and rec["complete"] is True
    assert rec["teardown_proof"]["method"] == "not-launched"
    assert rec["environment"]["live_lock_path"] == str(tmp_path / "live.lock")
    assert "not inside the canonical receipt Git repository" in rec["notes"]


# ---------------------------------------------------------------- policy refusals (no network)

def test_estimate_is_rate_times_wallclock():
    assert estimate_usd(usd_per_hour=2.0, wallclock_h=3.0) == 6.0


def test_live_lock_refuses_a_second_controller_and_releases_on_close(tmp_path: Path):
    path = tmp_path / "rent-live.lock"
    first = acquire_live_lock(run_id="proof-one", who="CDO/one", path=path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["run_id"] == "proof-one"
    with pytest.raises(RentRefused, match=r"another live rental controller.*proof-one"):
        acquire_live_lock(run_id="proof-two", who="CDO/two", path=path)
    first.close()
    second = acquire_live_lock(run_id="proof-two", who="CDO/two", path=path)
    assert json.loads(path.read_text())["run_id"] == "proof-two"
    second.close()


def test_guard_inherits_the_live_lock_descriptor(tmp_path: Path, monkeypatch):
    captured = {}

    def popen(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(subprocess, "Popen", popen)
    spawn_guard(python=sys.executable, module_args=["--guard-worker"],
                log_path=tmp_path / "guard.log", lock_fd=17)
    assert captured["close_fds"] is True and captured["pass_fds"] == (17,)


def test_live_lock_filesystem_failure_is_a_named_refusal(tmp_path: Path):
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("x")
    with pytest.raises(RentRefused, match=r"live controller lock unavailable"):
        acquire_live_lock(run_id="proof-one", who="CDO/one", path=not_a_directory / "rent-live.lock")


def test_live_lock_environment_cannot_split_the_production_lock_domain(tmp_path: Path, monkeypatch):
    from experts4bit_qlora.tools import rent as rent_mod

    canonical = tmp_path / "canonical.lock"
    alternate = tmp_path / "alternate.lock"
    monkeypatch.setattr(rent_mod, "DEFAULT_LIVE_LOCK_PATH", canonical)
    monkeypatch.setenv("E4B_RENT_LOCK_PATH", str(alternate))
    handle = rent_mod.acquire_live_lock(run_id="proof-one", who="CDO/one")
    try:
        assert Path(handle.name) == canonical and canonical.is_file()
        assert not alternate.exists()
    finally:
        handle.close()


def test_live_lock_default_ignores_caller_home(monkeypatch):
    import pwd
    from experts4bit_qlora.tools import rent as rent_mod

    monkeypatch.setenv("HOME", "/tmp/controller-selected-home")
    expected = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".adertha" / "rent-live.lock"
    assert rent_mod.DEFAULT_LIVE_LOCK_PATH == expected


def test_same_run_id_lock_loser_cannot_clobber_owner_receipt_or_duplicate_ledger_id(tmp_path: Path, monkeypatch):
    from datetime import datetime, timezone
    from experts4bit_qlora.tools import rent as rent_mod

    run_id = "same-run"
    canonical = tmp_path / "canonical.lock"
    monkeypatch.setattr(rent_mod, "DEFAULT_LIVE_LOCK_PATH", canonical)
    owner = rent_mod.acquire_live_lock(run_id=run_id, who="CDO/owner")
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    owner_receipt = tmp_path / "runs" / date_utc / run_id / "receipt.json"
    owner_receipt.parent.mkdir(parents=True)
    owner_receipt.write_text("owner-in-progress\n")
    try:
        assert main(_cli(tmp_path, run_id, "--provider", "fake", dry_run=False)) == 2
        assert main(_cli(tmp_path, run_id, "--provider", "fake", dry_run=False)) == 2
    finally:
        owner.close()

    assert owner_receipt.read_text() == "owner-in-progress\n"
    refused_paths = sorted((tmp_path / "runs" / date_utc).glob(f"{run_id}-refused-*/receipt.json"))
    assert len(refused_paths) == 2
    refused = [json.loads(path.read_text()) for path in refused_paths]
    for rec in refused:
        validate_receipt(rec, SCHEMA)
        assert rec["status"] == "REFUSED" and rec["cost_usd"]["actual"] == 0
        assert rec["environment"]["requested_run_id"] == run_id
        assert rec["experiment_id"].startswith(f"{run_id}-refused-")
    ledger_ids = [json.loads(line)["run_id"] for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    assert len(ledger_ids) == len(set(ledger_ids)) == 2
    assert run_id not in ledger_ids


def test_sequential_live_reuse_of_run_id_is_refused_before_create(tmp_path: Path, monkeypatch):
    from datetime import datetime, timezone
    from experts4bit_qlora.tools import rent as rent_mod

    run_id = "sequential-run"
    monkeypatch.setattr(rent_mod, "DEFAULT_LIVE_LOCK_PATH", tmp_path / "canonical.lock")
    args = _cli(tmp_path, run_id, "--provider", "fake", "--command", "true", dry_run=False)
    assert main(args) == 0
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    owner_path = tmp_path / "runs" / date_utc / run_id / "receipt.json"
    owner_bytes = owner_path.read_bytes()
    owner = json.loads(owner_bytes)
    assert owner["status"] == "OK"

    assert main(args) == 2
    assert owner_path.read_bytes() == owner_bytes
    refusal_paths = sorted((tmp_path / "runs" / date_utc).glob(f"{run_id}-refused-*/receipt.json"))
    assert len(refusal_paths) == 1
    refusal = json.loads(refusal_paths[0].read_text())
    assert refusal["status"] == "REFUSED" and refusal["environment"]["requested_run_id"] == run_id
    assert "already exists in the ledger" in refusal["notes"]
    fake = tmp_path / f"{run_id}-fake.json"
    assert not (json.loads(fake.read_text()).get("live") or [])
    ledger_ids = [json.loads(line)["run_id"] for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    assert len(ledger_ids) == len(set(ledger_ids)) == 2


def test_sequential_dry_run_reuse_cannot_trust_stale_markers_or_overwrite_receipt(tmp_path: Path):
    run_id = "dry-sequential"
    command_count = tmp_path / "command-count"
    command = f"echo ran >> {command_count}"
    args = _cli(tmp_path, run_id, "--command", command)
    assert main(args) == 0
    owner = _receipt(tmp_path)
    owner_path = next((tmp_path / "runs").rglob(f"{run_id}/receipt.json"))
    owner_bytes = owner_path.read_bytes()
    assert command_count.read_text().splitlines() == ["ran"]

    assert main(args) == 2
    assert owner_path.read_bytes() == owner_bytes and json.loads(owner_bytes) == owner
    assert command_count.read_text().splitlines() == ["ran"], "stale arm marker authorized the repeated command"
    receipts = list((tmp_path / "runs").rglob("receipt.json"))
    assert len(receipts) == 2
    ids = [json.loads(path.read_text())["experiment_id"] for path in receipts]
    assert len(ids) == len(set(ids)) == 2


def test_concurrent_first_ledger_appends_preserve_every_row(tmp_path: Path):
    ledger = tmp_path / "new" / "ledger.jsonl"
    barrier = tmp_path / "go"
    code = (
        "import importlib.util,sys,time; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('rent_under_test',sys.argv[4]); "
        "rent=importlib.util.module_from_spec(spec); spec.loader.exec_module(rent); "
        "barrier=Path(sys.argv[2]); "
        "\nwhile not barrier.exists(): time.sleep(0.001)\n"
        "rent.append_ledger(Path(sys.argv[1]), run_id=sys.argv[3], date_utc='2026-09-06', role='CDO', cost_usd=0.0)"
    )
    rent_source = REPO / "experts4bit_qlora" / "tools" / "rent.py"
    procs = [subprocess.Popen([sys.executable, "-c", code, str(ledger), str(barrier), f"loser-{i}", str(rent_source)])
             for i in range(24)]
    barrier.touch()
    assert all(proc.wait(timeout=30) == 0 for proc in procs)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    assert len(rows) == 24 and {row["run_id"] for row in rows} == {f"loser-{i}" for i in range(24)}


def test_ledger_checker_rejects_duplicate_run_ids(tmp_path: Path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    row = {"run_id": "duplicate", "date_utc": "2026-09-06", "role": "CDO", "cost_usd": 0.0}
    ledger.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(SystemExit) as exc:
        _ledger_module().load_ledger_entries(ledger)
    assert exc.value.code == 1
    assert "duplicate ledger run_id 'duplicate'" in capsys.readouterr().err


def test_live_budget_snapshot_is_loaded_after_lock_acquisition(tmp_path: Path, monkeypatch):
    from datetime import datetime, timezone
    from experts4bit_qlora.tools import rent as rent_mod

    ledger = tmp_path / "ledger.jsonl"
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rent_mod.append_ledger(ledger, run_id="earlier", date_utc=date_utc, role="CTO", cost_usd=49.4)
    canonical = tmp_path / "canonical.lock"
    real_acquire = rent_mod.acquire_live_lock

    def acquire_after_prior_owner_finishes(*, run_id, who):
        handle = real_acquire(run_id=run_id, who=who, path=canonical)
        rent_mod.append_ledger(ledger, run_id="prior-owner", date_utc=date_utc, role="CTO", cost_usd=0.3)
        return handle

    monkeypatch.setattr(rent_mod, "acquire_live_lock", acquire_after_prior_owner_finishes)
    rc = main(_cli(tmp_path, "fresh-budget", "--provider", "fake", dry_run=False))
    rec = _receipt(tmp_path)
    assert rc == 2 and rec["status"] == "REFUSED" and "exceeds ceiling" in rec["notes"]
    fake = tmp_path / "fresh-budget-fake.json"
    assert not fake.exists() or not (json.loads(fake.read_text()).get("live") or [])


def test_ledger_reader_waits_for_writer_lock_and_reads_complete_row(tmp_path: Path):
    import fcntl

    ledger = tmp_path / "ledger.jsonl"
    result = tmp_path / "result.json"
    rent_source = REPO / "experts4bit_qlora" / "tools" / "rent.py"
    code = (
        "import importlib.util,json,sys; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('rent_under_test',sys.argv[3]); "
        "rent=importlib.util.module_from_spec(spec); spec.loader.exec_module(rent); "
        "Path(sys.argv[2]).write_text(json.dumps(rent.load_ledger(Path(sys.argv[1]))))"
    )
    with ledger.open("a+", encoding="utf-8") as writer:
        fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
        writer.write("# ledger\n")
        writer.flush()
        reader = subprocess.Popen([sys.executable, "-c", code, str(ledger), str(result), str(rent_source)])
        time.sleep(0.2)
        assert reader.poll() is None, "reader ignored the writer's exclusive flock"
        writer.write(json.dumps({"run_id": "prior", "date_utc": "2026-09-06", "role": "CTO", "cost_usd": 7.25}) + "\n")
        writer.flush()
        os.fsync(writer.fileno())
        fcntl.flock(writer.fileno(), fcntl.LOCK_UN)
    assert reader.wait(timeout=10) == 0
    assert json.loads(result.read_text()) == [
        {"run_id": "prior", "date_utc": "2026-09-06", "role": "CTO", "cost_usd": 7.25}
    ]


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


@pytest.mark.parametrize(
    "complete_value",
    [False, None, 0, "false", pytest.param(_MISSING, id="missing")],
    ids=["false", "null", "zero", "string", "missing"],
)
def test_ok_pass_receipt_requires_complete_in_both_validators(
    tmp_path: Path, monkeypatch, complete_value: object,
):
    assert main(_cli(tmp_path, "rent-complete-invariant-1")) == 0
    rec = _receipt(tmp_path)
    incomplete = dict(rec)
    if complete_value is _MISSING:
        incomplete.pop("complete")
    else:
        incomplete["complete"] = complete_value

    with pytest.raises(ReceiptInvalid, match="OK/pass receipts require complete=true"):
        validate_receipt(incomplete, SCHEMA)
    checker = _ledger_module()
    with pytest.raises(SystemExit):
        checker.validate_receipt(tmp_path / "incomplete.json", incomplete)

    monkeypatch.setitem(sys.modules, "jsonschema", None)
    with pytest.raises(ReceiptInvalid, match="OK/pass receipts require complete=true"):
        validate_receipt(incomplete, SCHEMA)
    with pytest.raises(SystemExit):
        checker.validate_receipt(tmp_path / "incomplete-fallback.json", incomplete)


@pytest.mark.parametrize("proof_complete", [False, None, 0, "false"], ids=["false", "null", "zero", "string"])
def test_ok_pass_receipt_requires_true_nested_teardown_complete_when_present(
    tmp_path: Path, monkeypatch, proof_complete: object,
):
    assert main(_cli(tmp_path, "rent-proof-complete-invariant-1")) == 0
    rec = _receipt(tmp_path)
    contradictory = dict(rec, teardown_proof=dict(rec["teardown_proof"], complete=proof_complete))

    with pytest.raises(ReceiptInvalid):
        validate_receipt(contradictory, SCHEMA)
    checker = _ledger_module()
    with pytest.raises(SystemExit):
        checker.validate_receipt(tmp_path / "contradictory.json", contradictory)

    monkeypatch.setitem(sys.modules, "jsonschema", None)
    with pytest.raises(ReceiptInvalid):
        validate_receipt(contradictory, SCHEMA)
    with pytest.raises(SystemExit):
        checker.validate_receipt(tmp_path / "contradictory-fallback.json", contradictory)


def test_ok_pass_receipt_allows_historical_nested_teardown_complete_omission(tmp_path: Path, monkeypatch):
    assert main(_cli(tmp_path, "rent-proof-complete-historical-1")) == 0
    rec = _receipt(tmp_path)
    historical = dict(rec, teardown_proof=dict(rec["teardown_proof"]))
    historical["teardown_proof"].pop("complete")

    validate_receipt(historical, SCHEMA)
    checker = _ledger_module()
    checker.validate_receipt(tmp_path / "historical.json", historical)

    monkeypatch.setitem(sys.modules, "jsonschema", None)
    validate_receipt(historical, SCHEMA)
    checker.validate_receipt(tmp_path / "historical-fallback.json", historical)


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


def test_stale_arm_marker_for_another_instance_is_not_trusted(tmp_path: Path):
    from experts4bit_qlora.tools import rent as rent_mod

    marker = tmp_path / "guard-armed.json"
    marker.write_text(json.dumps({"pid": 1, "instance_id": "old-instance", "at": "2000-01-01T00:00:00Z"}))
    guard = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    armed = rent_mod.wait_for_guard_armed(marker, guard, timeout_s=5, instance_id="current-instance")
    assert armed is None and guard.wait(timeout=2) == 3


def test_guard_spawn_failure_tears_down_without_running_the_command(tmp_path: Path, monkeypatch):
    import experts4bit_qlora.tools.rent as rent_mod

    def fail_to_spawn(**kwargs):
        raise OSError("synthetic process-table failure")

    monkeypatch.setattr(rent_mod, "spawn_guard", fail_to_spawn)
    ran = tmp_path / "command-ran"
    rc = main(_cli(tmp_path, "rent-arm-4", "--command", f"touch {ran}"))
    rec = _receipt(tmp_path)
    assert rc == 1 and rec["status"] == "HARNESS_ERROR" and rec["result"] == "invalid"
    assert rec["complete"] is True and rec["teardown_proof"]["reason"] == "guard-spawn-failed"
    assert "guard failed to start" in rec["notes"] and "command not run" in rec["notes"] and not ran.exists()
    assert rec["instance_id"] not in (json.loads((tmp_path / "rent-arm-4-fake.json").read_text()).get("live") or [])


def test_guard_spawn_and_first_destroy_failure_hold_lock_until_recovery(tmp_path: Path, monkeypatch):
    import threading
    import experts4bit_qlora.tools.rent as rent_mod

    first_failure = threading.Event()
    allow_recovery = threading.Event()

    class RecoveringProvider(FakeProvider):
        destroy_calls = 0

        def destroy(self, instance_id):
            self.destroy_calls += 1
            if self.destroy_calls == 1:
                first_failure.set()
                raise RuntimeError("synthetic first destroy failure")
            assert allow_recovery.wait(timeout=10)
            return super().destroy(instance_id)

    canonical = tmp_path / "canonical.lock"
    fake = tmp_path / "rent-arm-5-fake.json"
    provider = RecoveringProvider(fake)
    monkeypatch.setattr(rent_mod, "DEFAULT_LIVE_LOCK_PATH", canonical)
    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kwargs: provider)
    monkeypatch.setattr(rent_mod, "spawn_guard", lambda **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))
    ran = tmp_path / "command-ran"
    result: dict[str, int] = {}
    controller = threading.Thread(
        target=lambda: result.setdefault(
            "rc", main(_cli(tmp_path, "rent-arm-5", "--provider", "fake", "--command", f"touch {ran}", dry_run=False))
        )
    )
    controller.start()
    assert first_failure.wait(timeout=10)
    contender_code = (
        "import sys; from pathlib import Path; "
        "from experts4bit_qlora.tools.rent import acquire_live_lock, RentRefused; "
        "\ntry: acquire_live_lock(run_id='contender', who='CDO/two', path=Path(sys.argv[1]))\n"
        "except RentRefused: raise SystemExit(0)\n"
        "raise SystemExit(3)"
    )
    contender = subprocess.run([sys.executable, "-c", contender_code, str(canonical)], check=False)
    assert contender.returncode == 0, "the recovery controller released its lock while the instance was still live"
    allow_recovery.set()
    controller.join(timeout=15)
    assert not controller.is_alive() and result["rc"] == 1
    rec = _receipt(tmp_path)
    assert rec["status"] == "ALARM" and rec["result"] == "invalid" and rec["complete"] is True
    assert rec["teardown_proof"]["reason"] == "guard-spawn-failed"
    assert rec["environment"]["teardown_attempts"] == "2" and "command not run" in rec["notes"]
    assert not ran.exists() and rec["instance_id"] not in provider.list_ids()
    reacquired = acquire_live_lock(run_id="successor", who="CDO/two", path=canonical)
    reacquired.close()


def test_dead_unarmed_guard_and_incomplete_proof_cannot_bypass_teardown(tmp_path: Path, monkeypatch):
    import experts4bit_qlora.tools.rent as rent_mod

    class FailDestroyOnce(FakeProvider):
        destroy_calls = 0

        def destroy(self, instance_id):
            self.destroy_calls += 1
            if self.destroy_calls == 1:
                raise RuntimeError("first destroy failed")
            return super().destroy(instance_id)

    provider = FailDestroyOnce(tmp_path / "rent-arm-6-fake.json")
    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kwargs: provider)

    def dead_guard_with_incomplete_proof(*, python, module_args, log_path, **kwargs):
        parsed = _guard_args(module_args)
        proof = Path(parsed["--proof"])
        iid = parsed["--instance-id"]
        code = (
            "import json; from pathlib import Path; "
            f"Path({str(proof)!r}).write_text(json.dumps({{'method':'fake-destroy-failed','reason':'wallclock',"
            f"'evidence':'failed','complete':False,'instance_id':{iid!r},'at':'2000-01-01T00:00:00Z'}})); "
            "raise SystemExit(3)"
        )
        return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    monkeypatch.setattr(rent_mod, "spawn_guard", dead_guard_with_incomplete_proof)
    ran = tmp_path / "command-ran"
    rc = main(_cli(tmp_path, "rent-arm-6", "--guard-arm-timeout-s", "5", "--command", f"touch {ran}"))
    rec = _receipt(tmp_path)
    assert rc == 1 and rec["status"] == "ALARM" and rec["result"] == "invalid" and rec["complete"] is True
    assert rec["teardown_proof"]["reason"] == "guard-not-armed" and provider.destroy_calls == 2
    assert "ignored teardown proof without authenticated absence" in rec["notes"] and not ran.exists()
    assert rec["instance_id"] not in provider.list_ids()


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
    method='fake-guard-fired' and reason='wallclock'. Because that proof is incomplete, the launcher
    must then destroy the instance itself, authenticate absence, and still return ALARM/invalid.
    This test fails if an incomplete proof bypasses teardown or is allowed to produce a pass."""
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
    # The launcher should join the guard after its own authenticated teardown; clean up defensively.
    for p in captured:
        try:
            p.kill()
        except OSError:
            pass
    assert rc == 1
    rec = _receipt(tmp_path)
    # Launcher detected the firing marker, rejected its incomplete proof, and tore down itself.
    assert rec["status"] == "ALARM" and rec["result"] == "invalid", rec
    assert rec["teardown_proof"]["reason"] == "completion", rec["teardown_proof"]
    assert rec["teardown_proof"]["method"] == "fake-destroy", rec["teardown_proof"]
    fake_state = tmp_path / "rent-toctou-det-1-fake.json"
    assert fake_state.is_file(), "fake state file expected"
    st = json.loads(fake_state.read_text())
    assert rec["instance_id"] not in (st.get("live") or [])
    assert rec["complete"] is True and "ignored teardown proof without authenticated absence" in rec["notes"]


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
                "'evidence': json.dumps({'instance_absent': True}), 'complete': True, 'instance_id': iid}))\n"
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


def _guard_on(prov, rent_mod, tmp_path: Path, iid: str, fake: Path):
    """Run guard_worker against `prov` with a fresh heartbeat; return (rc, proof dict)."""
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
    return rc, json.loads(proof.read_text())


def test_the_already_gone_branch_never_asks_for_a_second_listing(tmp_path: Path):
    """#510: the absence is established by the loop's own listing, so no second request is made.

    The branch used to call `_list_or_unknown(prov)` again. That helper answers None when the
    backend declines (#455: never an empty set), and `sorted(None)` raises inside the dict being
    built for `_write_proof` -- so a declining API destroyed the proof of a teardown the guard had
    already observed, and the run ended ALARM/invalid for want of a listing incidental to the
    conclusion.  This provider raises on any listing after the decisive one, so on the old code the
    guard died with a TypeError and wrote nothing.
    """
    from experts4bit_qlora.tools import rent as rent_mod

    class AbsentThenBroken(FakeProvider):
        calls = 0

        def list_ids(self):
            AbsentThenBroken.calls += 1
            if AbsentThenBroken.calls > 1:
                raise AssertionError("#510: the already-gone branch must not request a second listing")
            return {"someone-elses-box"}

    fake = tmp_path / "gone.json"
    prov = AbsentThenBroken(fake)
    iid = prov.launch(gpu="RTX 5090", wallclock_h=1, image="img")
    rc, data = _guard_on(prov, rent_mod, tmp_path, iid, fake)

    assert AbsentThenBroken.calls == 1, "the decisive listing is the only one the branch needs"
    assert rc == 0
    assert data["reason"] == "already-gone" and data["complete"] is True
    assert json.loads(data["evidence"])["instance_absent"] is True


def test_the_already_gone_proof_records_the_listing_that_established_the_absence(tmp_path: Path):
    """#510: `list_after` is the set the instance was observed absent from, not a later re-read.

    Calibrated to actually detect the defect: the second listing returns a DIFFERENT set, so the
    old code (which re-read) records `["box-c"]` and the fixed code records the decisive
    `["box-a", "box-b"]`.  An identical second listing would have passed either way -- which is
    what an earlier draft of this test did, and it proved nothing.
    """
    from experts4bit_qlora.tools import rent as rent_mod

    class AbsentThenDifferent(FakeProvider):
        calls = 0

        def list_ids(self):
            AbsentThenDifferent.calls += 1
            if AbsentThenDifferent.calls == 1:
                return {"box-b", "box-a"}      # the listing that establishes the absence
            return {"box-c"}                    # any later listing: a different, irrelevant world

    fake = tmp_path / "gone2.json"
    prov = AbsentThenDifferent(fake)
    iid = prov.launch(gpu="RTX 5090", wallclock_h=1, image="img")
    rc, data = _guard_on(prov, rent_mod, tmp_path, iid, fake)

    assert rc == 0 and data["reason"] == "already-gone"
    ev = json.loads(data["evidence"])
    assert ev["list_after"] == ["box-a", "box-b"], "sorted, and the listing the conclusion rests on"
    assert ev["instance_absent"] is True


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
    assert tp["instance_id"] == rec["instance_id"]
    with pytest.raises(ReceiptInvalid, match="teardown_proof.instance_id must equal receipt instance_id"):
        validate_receipt(dict(rec, teardown_proof=dict(tp, instance_id="another-instance")), SCHEMA)

    historical_tp = dict(tp)
    historical_tp.pop("instance_id")
    validate_receipt(dict(rec, teardown_proof=historical_tp), SCHEMA)


def test_ledger_checker_binds_optional_teardown_instance_id(tmp_path: Path, monkeypatch, capsys) -> None:
    rc = main(_cli(tmp_path, "rent-proof-binding-1"))
    assert rc == 0
    rec = _receipt(tmp_path)
    checker = _ledger_module()

    historical_tp = dict(rec["teardown_proof"])
    historical_tp.pop("instance_id")
    checker.validate_receipt(tmp_path / "historical.json", dict(rec, teardown_proof=historical_tp))

    contradictory = dict(rec, teardown_proof=dict(rec["teardown_proof"], instance_id="another-instance"))
    with pytest.raises(SystemExit) as strict_exc:
        checker.validate_receipt(tmp_path / "strict.json", contradictory)
    assert strict_exc.value.code == 1
    assert "teardown_proof.instance_id must equal receipt instance_id" in capsys.readouterr().err

    monkeypatch.setitem(sys.modules, "jsonschema", None)
    with pytest.raises(SystemExit) as fallback_exc:
        checker.validate_receipt(tmp_path / "fallback.json", contradictory)
    assert fallback_exc.value.code == 1
    assert "teardown_proof.instance_id must equal receipt instance_id" in capsys.readouterr().err


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


def test_teardown_listing_outage_holds_lock_and_retries_until_absence(tmp_path: Path, monkeypatch):
    """A transient listing outage cannot produce false absence or hand off through a guard-exit race."""
    from experts4bit_qlora.tools import rent as rent_mod

    class DeadAfterDestroy(FakeProvider):
        failures_left = 0

        def destroy(self, instance_id):
            ev = super().destroy(instance_id)
            if not self.state_path.with_suffix(".dead").is_file():
                self.state_path.with_suffix(".dead").write_text("x")
                self.failures_left = 2
            return ev

        def list_ids(self):
            if self.failures_left:
                self.failures_left -= 1
                raise RuntimeError("deprecated_endpoint (false zero refused)")
            return super().list_ids()

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kw: DeadAfterDestroy(kw["fake_state"]))
    rc = main(_cli(tmp_path, "rent-dead-1"))
    rec = _receipt(tmp_path)
    assert rc == 1 and rec["status"] == "ALARM" and rec["result"] == "invalid" and rec["complete"] is True
    assert "holds the live lock until authenticated absence" in rec["notes"]
    assert rec["environment"]["teardown_attempts"] == "3" and "guard_left_alive" not in rec["environment"]
    tp = rec["teardown_proof"]
    assert tp["complete"] is True and tp["reason"] == "completion"
    assert json.loads(tp["evidence"])["list_after"] == [] and json.loads(tp["evidence"])["instance_absent"] is True


def test_receipt_reuses_authenticated_teardown_absence_when_next_listing_would_429(
    tmp_path: Path, monkeypatch,
):
    """The listing that closes the own-teardown loop is final evidence; a redundant next request cannot void it."""
    import experts4bit_qlora.tools.rent as rent_mod

    class NextListingWouldRateLimit(FakeProvider):
        def __init__(self, state_path: Path):
            super().__init__(state_path)
            self.list_calls = 0

        def list_ids(self):
            self.list_calls += 1
            if self.list_calls >= 3:
                raise RuntimeError("HTTP 429 after authenticated teardown absence")
            return super().list_ids()

    provider = NextListingWouldRateLimit(tmp_path / "rent-final-list-429-fake.json")
    guards: list[subprocess.Popen] = []

    def _spawn(*, python, module_args, log_path, **kwargs):
        proc = _arming_guard_spawn(module_args, "import time; time.sleep(60)\n")
        guards.append(proc)
        return proc

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kwargs: provider)
    monkeypatch.setattr(rent_mod, "spawn_guard", _spawn)
    try:
        rc = main(_cli(tmp_path, "rent-final-list-429", "--command", "true"))
    finally:
        for guard in guards:
            if guard.poll() is None:
                guard.kill()

    rec = _receipt(tmp_path)
    assert provider.list_calls == 2, "the controller made a redundant listing after proving absence"
    assert rc == 0 and rec["status"] == "OK" and rec["result"] == "pass" and rec["complete"] is True
    proof = rec["teardown_proof"]
    assert proof["instance_id"] == rec["instance_id"] and proof["complete"] is True
    evidence = json.loads(proof["evidence"])
    assert evidence["instance_absent"] is True and evidence["list_after"] == []


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
    rate, est = float(seen["E4B_RENT_USD_PER_HOUR"]), float(seen["E4B_RENT_EST_USD"])   # one source for a box-side budget rule
    assert abs(est - rate * float(seen["E4B_RENT_WALLCLOCK_S"]) / 3600) < 1e-6 and rate > 0
    for k in ("E4B_RENT_RUN_ID", "E4B_RENT_INSTANCE_ID", "E4B_RENT_RUN_DIR", "E4B_RENT_SSH"):
        assert k not in os.environ, f"{k} exported into the launcher's own process"  # E4B_RENT_LIVE is the fixture's, not ours
    assert rec["environment"]["ssh_pubkey_given"] == "no"


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


def test_a_failed_key_attach_leaves_no_attached_claim_in_the_receipt(tmp_path: Path, monkeypatch):
    """#465 MEDIUM-1: the launcher records that a key was GIVEN; only a pre-flight that got the 200 records ATTACHED."""
    from experts4bit_qlora.tools import rent as rent_mod

    class AttachFails(FakeProvider):
        def preflight(self, instance_id, *, timeout_s=600.0):
            raise RuntimeError("attaching the ssh key to i failed: HTTP 500")

    monkeypatch.setattr(rent_mod, "provider_for", lambda kind, **kw: AttachFails(kw["fake_state"]))
    pub = tmp_path / "k.pub"
    pub.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGQyMDI2LXRlc3Qta2V5LW5vdC1yZWFsLWJ5dGVz test@e4b\n")
    rc = main(_cli(tmp_path, "rent-key-2", "--ssh-pubkey", str(pub), "--command", "true"))
    rec = _receipt(tmp_path)
    assert rc != 0 and rec["status"] != "OK" and rec["environment"]["vast_preflight"] == "failed"
    assert rec["environment"]["ssh_pubkey_given"] == "yes"
    assert "vast_ssh_key_attached" not in rec["environment"] and "ssh_pubkey_attached" not in rec["environment"]


def test_ssh_pubkey_is_read_by_shape_and_a_private_key_is_refused(tmp_path: Path):
    from experts4bit_qlora.tools.rent import read_pubkey
    good = tmp_path / "id_ed25519.pub"
    good.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholderKeyMaterialForTheTestOnly0000000000 cdo@mini\n")
    assert read_pubkey(good).startswith("ssh-ed25519 AAAA")
    privkey = tmp_path / "id_ed25519"
    privkey.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----\n")
    with pytest.raises(RentRefused, match="private key is refused"):
        read_pubkey(privkey)
    with pytest.raises(RentRefused, match="no such file"):
        read_pubkey(tmp_path / "absent.pub")
    two = tmp_path / "two.pub"
    two.write_text("ssh-ed25519 AAAA1 a\nssh-ed25519 AAAA2 b\n")
    with pytest.raises(RentRefused, match="single"):
        read_pubkey(two)
    # through the CLI on the fake provider: the receipt records that a key was attached
    rc = main(_cli(tmp_path, "rent-key-1", "--ssh-pubkey", str(good)))
    assert rc == 0 and _receipt(tmp_path)["environment"]["ssh_pubkey_given"] == "yes"
    rc = main(_cli(tmp_path / "b", "rent-key-2", "--ssh-pubkey", str(privkey)))
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
