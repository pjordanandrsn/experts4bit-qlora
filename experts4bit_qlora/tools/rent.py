# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""GPU rental launcher that enforces ``docs/compute-policy.json`` at rent time (#430).

``python -m experts4bit_qlora.tools.rent`` (or ``scripts/rent_run.py``) is the
entry point. It estimates cost, refuses policy violations, requires the
approvals the threshold demands, launches (or dry-runs) on an allowed
provider, arms a teardown guard that lives on the *controller* (survives
ssh/parent death), proves the instance is gone, and writes a receipt +
ledger line even when the run fails.

No gate, threshold, floor or claim value is moved by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SLACK_PERMALINK = (
    "https://cerin-amroth.slack.com/",
    "https://slack.com/",
)


class RentRefused(RuntimeError):
    """Policy refusal before any instance is created."""


def load_policy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(json.loads(line))
    return out


def estimate_usd(*, usd_per_hour: float, wallclock_h: float) -> float:
    if usd_per_hour < 0 or wallclock_h <= 0:
        raise RentRefused("usd_per_hour must be >= 0 and wallclock_h must be > 0")
    return round(usd_per_hour * wallclock_h, 4)


def _today(entries: list[dict[str, Any]], date_utc: str, *, role: str | None = None) -> float:
    total = 0.0
    for e in entries:
        if e.get("date_utc") != date_utc:
            continue
        if role is not None and e.get("role") != role:
            continue
        total += float(e.get("cost_usd") or 0)
    return total


def _jordan_approved(approvals: list[dict[str, Any]]) -> bool:
    return any(a.get("agent") == "Jordan" or a.get("role") == "Jordan" for a in approvals)


def _permalink_ok(url: str) -> bool:
    return isinstance(url, str) and url.startswith(SLACK_PERMALINK)


def check_provider_gpu(policy: dict[str, Any], provider: str, gpu: str) -> None:
    allowed_p = list(policy.get("allowed_providers") or [])
    allowed_g = list(policy.get("allowed_gpus") or [])
    if provider not in allowed_p:
        raise RentRefused(f"provider {provider!r} is not allowed (policy: {allowed_p})")
    if gpu not in allowed_g:
        raise RentRefused(f"gpu {gpu!r} is not allowed (policy: {allowed_g})")


def check_wallclock(policy: dict[str, Any], wallclock_h: float) -> None:
    td = policy.get("teardown") or {}
    max_h = float(td.get("max_wallclock_h") or 6)
    if wallclock_h > max_h:
        raise RentRefused(f"wallclock {wallclock_h}h exceeds teardown.max_wallclock_h {max_h}")


def check_hard_cap(policy: dict[str, Any], estimate: float, approvals: list[dict[str, Any]]) -> None:
    cap = float(policy["per_run_hard_cap_usd"])
    if estimate > cap and not _jordan_approved(approvals):
        raise RentRefused(
            f"estimate ${estimate:.2f} exceeds per-run hard cap ${cap:.2f} without Jordan's approval")


def check_ceilings(policy: dict[str, Any], ledger: list[dict[str, Any]], *,
                   role: str, estimate: float, date_utc: str) -> None:
    ceilings = policy["role_daily_ceiling_usd"]
    ceiling = ceilings.get(role)
    if ceiling is not None and not isinstance(ceiling, (int, float)):
        ceiling = None
    if ceiling is not None:
        spent = _today(ledger, date_utc, role=role)
        if spent + estimate > float(ceiling):
            raise RentRefused(
                f"{role} daily spend ${spent:.2f} + estimate ${estimate:.2f} "
                f"exceeds ceiling ${float(ceiling):.2f}")
    global_budget = float(policy["global_daily_budget_usd"])
    gspent = _today(ledger, date_utc)
    if gspent + estimate > global_budget:
        raise RentRefused(
            f"global daily spend ${gspent:.2f} + estimate ${estimate:.2f} "
            f"exceeds budget ${global_budget:.2f}")


def check_approvals(policy: dict[str, Any], estimate: float,
                    approvals: list[dict[str, Any]], *, role: str) -> None:
    applicable = None
    for th in policy["approval_thresholds"]:
        max_usd = th["max_usd"]
        if max_usd is None or estimate <= float(max_usd):
            applicable = th
            break
    if applicable is None:
        raise RentRefused(f"no approval threshold for ${estimate:.2f}")
    spec = applicable["approver"]
    if spec == "requesting-agent":
        return
    for a in approvals:
        if a.get("slack_permalink") and not _permalink_ok(str(a["slack_permalink"])):
            raise RentRefused(f"approval permalink is not a Slack URL: {a.get('slack_permalink')!r}")
    if _jordan_approved(approvals):
        return
    if spec == "Jordan":
        if not _jordan_approved(approvals):
            raise RentRefused(f"estimate ${estimate:.2f} requires Jordan approval")
        return
    if spec.startswith("one-of:"):
        need = set(spec.split(":", 1)[1].split(","))
        got = {a.get("role") for a in approvals}
        if not (need & got):
            raise RentRefused(f"estimate ${estimate:.2f} requires one of {sorted(need)}, got {sorted(got)}")
        return
    if spec.startswith("two-of:"):
        need = set(spec.split(":", 1)[1].split(","))
        got = {a.get("role") for a in approvals if a.get("role") in need}
        if len(got) < 2:
            raise RentRefused(f"estimate ${estimate:.2f} requires two of {sorted(need)}, got {sorted(got)}")
        return
    raise RentRefused(f"unknown approver spec {spec!r}")


def evaluate_launch(policy: dict[str, Any], ledger: list[dict[str, Any]], *,
                    role: str, estimate: float, provider: str, gpu: str,
                    wallclock_h: float, approvals: list[dict[str, Any]],
                    date_utc: str) -> None:
    check_provider_gpu(policy, provider, gpu)
    check_wallclock(policy, wallclock_h)
    check_hard_cap(policy, estimate, approvals)
    check_ceilings(policy, ledger, role=role, estimate=estimate, date_utc=date_utc)
    check_approvals(policy, estimate, approvals, role=role)


class FakeProvider:
    """File-backed fake so a detached guard process shares instance state."""

    kind = "fake"

    def __init__(self, state_path: Path):
        self.state_path = Path(state_path)

    def _load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"live": []}
        return json.loads(self.state_path.read_text())

    def _save(self, st: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(st) + "\n")

    def launch(self, *, gpu: str, wallclock_h: float, image: str) -> str:
        st = self._load()
        iid = f"fake-{int(time.time())}-{os.getpid()}"
        live = list(st.get("live") or [])
        live.append(iid)
        st["live"] = live
        self._save(st)
        return iid

    def list_ids(self) -> set[str]:
        return set(self._load().get("live") or [])

    def destroy(self, instance_id: str) -> dict[str, Any]:
        st = self._load()
        live = [x for x in (st.get("live") or []) if x != instance_id]
        st["live"] = live
        self._save(st)
        return {"method": "fake-destroy", "instance_id": instance_id,
                "remaining": live, "at": _utc()}


def provider_for(kind: str, *, fake_state: Path | None = None):
    if kind == "fake":
        if fake_state is None:
            raise RentRefused("fake provider needs --fake-state")
        return FakeProvider(fake_state)
    if kind in ("vast:verified-secure", "runpod:secure"):
        raise RentRefused(
            f"live provider {kind} is not armed in this process; pass --dry-run "
            "(fake provider) or implement the API adapter behind E4B_RENT_LIVE=1")
    raise RentRefused(f"unknown provider {kind!r}")


def spawn_guard(*, python: str, module_args: list[str],
                log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    return subprocess.Popen(
        [python, "-m", "experts4bit_qlora.tools.rent", *module_args],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )


def guard_worker(*, instance_id: str, provider_kind: str, fake_state: str | None,
                 wallclock_s: float, heartbeat_path: str, proof_path: str,
                 heartbeat_timeout_s: float) -> int:
    prov = provider_for(provider_kind, fake_state=Path(fake_state) if fake_state else None)
    deadline = time.time() + wallclock_s
    hb = Path(heartbeat_path)
    reason = "wallclock"
    while time.time() < deadline:
        time.sleep(min(0.2, max(0.05, wallclock_s / 20)))
        if hb.is_file():
            age = time.time() - hb.stat().st_mtime
            if age > heartbeat_timeout_s:
                reason = "heartbeat-loss"
                break
        elif time.time() + 0.01 >= deadline:
            break
    evidence = prov.destroy(instance_id)
    remaining = prov.list_ids()
    gone = instance_id not in remaining
    proof = {
        "method": evidence.get("method", f"{provider_kind}-destroy"),
        "evidence": json.dumps({"destroy": evidence, "list_after": sorted(remaining),
                                "instance_absent": gone, "reason": reason},
                               sort_keys=True),
        "complete": gone,
        "reason": reason,
        "at": _utc(),
    }
    Path(proof_path).parent.mkdir(parents=True, exist_ok=True)
    Path(proof_path).write_text(json.dumps(proof, indent=2) + "\n")
    return 0 if gone else 1


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_receipt(dir_path: Path, receipt: dict[str, Any]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def append_ledger(path: Path, *, run_id: str, date_utc: str, role: str, cost_usd: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Append-only ledger: one JSON object per line. "
            'Each entry: {"run_id": str, "date_utc": str, "role": str, "cost_usd": float}\n')
    with path.open("a") as f:
        f.write(json.dumps({"run_id": run_id, "date_utc": date_utc,
                            "role": role, "cost_usd": cost_usd}) + "\n")


def skeleton_receipt(**kw) -> dict[str, Any]:
    now = _utc()
    return {
        "experiment_id": kw.get("experiment_id", "rent-unset"),
        "work_id": kw.get("work_id", "experts4bit-qlora#430"),
        "requested_by": kw.get("requested_by", "CTO/Cursor"),
        "executed_by": kw.get("executed_by", kw.get("requested_by", "CTO/Cursor")),
        "reviewed_by": None,
        "hypothesis": kw.get("hypothesis", "rent launcher policy gate"),
        "expected_result": kw.get("expected_result", "instance launched under policy"),
        "success_criteria": kw.get("success_criteria", "receipt complete with teardown proof"),
        "failure_criteria": kw.get("failure_criteria", "policy refusal or missing teardown proof"),
        "preregistration": kw.get("preregistration", "docs/COMPUTE-GOVERNANCE.md"),
        "approvals": kw.get("approvals") or [],
        "repo": kw.get("repo", "pjordanandrsn/experts4bit-qlora"),
        "commit_sha": kw.get("commit_sha", "unknown"),
        "branch": kw.get("branch", "unknown"),
        "dirty_tree": bool(kw.get("dirty_tree", False)),
        "container_image": kw.get("container_image", "none"),
        "dependencies": kw.get("dependencies") or {},
        "command": kw.get("command") or ["python", "-m", "experts4bit_qlora.tools.rent"],
        "environment": kw.get("environment") or {},
        "provider": kw.get("provider", "fake"),
        "instance_id": kw.get("instance_id", "none"),
        "gpu_model": kw.get("gpu_model", "none"),
        "gpu_count": int(kw.get("gpu_count", 0)),
        "cpu": kw.get("cpu", "unknown"),
        "ram": kw.get("ram", "unknown"),
        "storage": kw.get("storage", "unknown"),
        "started_at": kw.get("started_at", now),
        "finished_at": kw.get("finished_at", now),
        "runtime_seconds": float(kw.get("runtime_seconds", 0)),
        "cost_usd": kw.get("cost_usd") or {"estimated": 0, "actual": 0},
        "dataset": kw.get("dataset", "none"),
        "dataset_hash": kw.get("dataset_hash", "none"),
        "model": kw.get("model", "none"),
        "model_revision": kw.get("model_revision", "none"),
        "model_hash": kw.get("model_hash", "none"),
        "seed": int(kw.get("seed", 0)),
        "configuration": kw.get("configuration") or {},
        "metrics": kw.get("metrics") or {},
        "artifacts": kw.get("artifacts") or [],
        "teardown_proof": kw.get("teardown_proof") or {"method": "not-launched", "evidence": "no instance"},
        "status": kw.get("status", "NOT_RUN"),
        "result": kw.get("result", "invalid"),
        "decision": kw.get("decision", "abandon https://github.com/pjordanandrsn/experts4bit-qlora/issues/430"),
        "notes": kw.get("notes", ""),
        "complete": kw.get("complete", False),
    }


def maybe_slack(text: str) -> None:
    url = os.environ.get("E4B_SLACK_WEBHOOK")
    if not url:
        return
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except (urllib.error.URLError, TimeoutError):
        pass


def _parse_approval(raw: str, estimate: float) -> dict[str, Any]:
    if "=" not in raw:
        raise RentRefused("--approval must be ROLE/AGENT=https://…slack.com/…")
    who, permalink = raw.split("=", 1)
    if "/" not in who:
        raise RentRefused("--approval identity must be ROLE/AGENT")
    role, agent = who.split("/", 1)
    return {"role": role, "agent": agent, "usd_estimate": estimate,
            "slack_permalink": permalink}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guard-worker", action="store_true")
    ap.add_argument("--instance-id")
    ap.add_argument("--provider", default="fake")
    ap.add_argument("--fake-state")
    ap.add_argument("--wallclock-s", type=float)
    ap.add_argument("--heartbeat")
    ap.add_argument("--proof")
    ap.add_argument("--heartbeat-timeout-s", type=float, default=2.0)
    ap.add_argument("--policy", default="docs/compute-policy.json")
    ap.add_argument("--ledger", default="bench/runs/ledger.jsonl")
    ap.add_argument("--runs-root", default="bench/runs")
    ap.add_argument("--run-id")
    ap.add_argument("--work-id", default="experts4bit-qlora#430")
    ap.add_argument("--role", default="CTO")
    ap.add_argument("--agent", default="Cursor")
    ap.add_argument("--gpu", default="RTX 5090")
    ap.add_argument("--usd-per-hour", type=float, default=0.5)
    ap.add_argument("--wallclock-h", type=float, default=1.0)
    ap.add_argument("--image", default="none")
    ap.add_argument("--approval", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--command", default="")
    ap.add_argument("--hypothesis", default="rent launcher")
    args = ap.parse_args(argv)

    if args.guard_worker:
        return guard_worker(
            instance_id=args.instance_id, provider_kind=args.provider,
            fake_state=args.fake_state, wallclock_s=float(args.wallclock_s),
            heartbeat_path=args.heartbeat, proof_path=args.proof,
            heartbeat_timeout_s=float(args.heartbeat_timeout_s))

    policy = load_policy(Path(args.policy))
    ledger = load_ledger(Path(args.ledger))
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    estimate = estimate_usd(usd_per_hour=args.usd_per_hour, wallclock_h=args.wallclock_h)
    approvals = [_parse_approval(a, estimate) for a in args.approval]
    run_id = args.run_id or f"rent-{date_utc}-{os.getpid()}"
    who = f"{args.role}/{args.agent}"
    provider = "fake" if args.dry_run else args.provider
    rec_dir = Path(args.runs_root) / date_utc / run_id
    policy_provider = "vast:verified-secure" if provider == "fake" else provider

    try:
        evaluate_launch(
            policy, ledger, role=args.role, estimate=estimate,
            provider=policy_provider, gpu=args.gpu, wallclock_h=args.wallclock_h,
            approvals=approvals, date_utc=date_utc)
    except RentRefused as e:
        rec = skeleton_receipt(
            experiment_id=run_id, work_id=args.work_id, requested_by=who,
            executed_by=who, approvals=approvals, gpu_model=args.gpu,
            cost_usd={"estimated": estimate, "actual": 0},
            status="REFUSED", result="invalid",
            teardown_proof={"method": "not-launched", "evidence": f"refused before launch: {e}"},
            notes=str(e), complete=True,
            command=[sys.executable, "-m", "experts4bit_qlora.tools.rent"],
            provider=provider,
            configuration={"dry_run": args.dry_run, "wallclock_h": args.wallclock_h},
            hypothesis=args.hypothesis,
        )
        write_receipt(rec_dir, rec)
        append_ledger(Path(args.ledger), run_id=run_id, date_utc=date_utc, role=args.role, cost_usd=0)
        maybe_slack(f"REFUSED {run_id}: {e}")
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    fake_state = Path(args.fake_state) if args.fake_state else rec_dir / "fake-state.json"
    rec_dir.mkdir(parents=True, exist_ok=True)
    prov = FakeProvider(fake_state) if (args.dry_run or provider == "fake") else provider_for(
        provider, fake_state=fake_state)
    t0 = time.time()
    iid = prov.launch(gpu=args.gpu, wallclock_h=args.wallclock_h, image=args.image)
    maybe_slack(f"LAUNCHED {run_id} instance={iid}")
    hb = rec_dir / "heartbeat"
    hb.write_text(_utc())
    proof_path = rec_dir / "teardown-proof.json"
    wallclock_s = args.wallclock_h * 3600
    guard = spawn_guard(
        python=sys.executable,
        module_args=[
            "--guard-worker", "--instance-id", iid, "--provider", prov.kind,
            "--fake-state", str(fake_state), "--wallclock-s", str(wallclock_s),
            "--heartbeat", str(hb), "--proof", str(proof_path),
            "--heartbeat-timeout-s", "30",
        ],
        log_path=rec_dir / "guard.log",
    )
    status, result, notes = "OK", "pass", f"guard pid {guard.pid}"
    if args.command:
        try:
            hb.write_text(_utc())
            subprocess.run(args.command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            status, result, notes = "HARNESS_ERROR", "fail", f"command exited {e.returncode}"
    if args.dry_run and not args.command:
        evidence = prov.destroy(iid)
        proof = {
            "method": evidence.get("method", "fake-destroy"),
            "evidence": json.dumps({"destroy": evidence,
                                    "list_after": sorted(prov.list_ids()),
                                    "instance_absent": iid not in prov.list_ids()}),
        }
        try:
            os.kill(guard.pid, signal.SIGTERM)
        except OSError:
            pass
    else:
        deadline = time.time() + 2
        proof = {"method": "pending", "evidence": "guard still running"}
        while time.time() < deadline:
            if proof_path.is_file():
                proof = json.loads(proof_path.read_text())
                break
            time.sleep(0.05)
        if not proof_path.is_file() and iid in prov.list_ids():
            evidence = prov.destroy(iid)
            proof = {"method": evidence.get("method", "fake-destroy"),
                     "evidence": json.dumps({"destroy": evidence,
                                             "list_after": sorted(prov.list_ids()),
                                             "instance_absent": iid not in prov.list_ids()})}
    gone = iid not in prov.list_ids()
    rec = skeleton_receipt(
        experiment_id=run_id, work_id=args.work_id, requested_by=who, executed_by=who,
        approvals=approvals, gpu_model=args.gpu, gpu_count=1, instance_id=iid,
        provider=prov.kind, cost_usd={"estimated": estimate, "actual": 0 if args.dry_run else estimate},
        status=status, result=result, notes=notes, teardown_proof=proof,
        complete=gone and bool(proof.get("method")) and proof.get("method") != "pending",
        runtime_seconds=time.time() - t0,
        command=[sys.executable, "-m", "experts4bit_qlora.tools.rent"],
        configuration={"dry_run": args.dry_run, "wallclock_h": args.wallclock_h,
                       "usd_per_hour": args.usd_per_hour},
        hypothesis=args.hypothesis,
    )
    rec_path = write_receipt(rec_dir, rec)
    rec["artifacts"] = [{"path": str(rec_path), "sha256": _sha256_file(rec_path),
                         "bytes": rec_path.stat().st_size}]
    write_receipt(rec_dir, rec)
    append_ledger(Path(args.ledger), run_id=run_id, date_utc=date_utc, role=args.role,
                  cost_usd=rec["cost_usd"]["actual"])
    maybe_slack(f"DONE {run_id} status={status}")
    maybe_slack(f"TORN DOWN {run_id} instance={iid} gone={gone}")
    print(f"{'DRY-RUN' if args.dry_run else 'LAUNCHED'} {run_id} instance={iid} "
          f"complete={rec['complete']} receipt={rec_path}")
    return 0 if rec["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
