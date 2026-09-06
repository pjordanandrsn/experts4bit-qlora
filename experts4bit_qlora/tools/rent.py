# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""GPU rental launcher that enforces ``docs/compute-policy.json`` at rent time (#430).

``python -m experts4bit_qlora.tools.rent`` (or ``scripts/rent_run.py``) is the
entry point. It estimates cost, refuses policy violations, requires the
approvals the threshold (and any executor-conditional override) demands,
launches (or dry-runs) on an allowed provider, arms a teardown guard that lives
on the *controller* (survives ssh/parent death), keeps that guard's heartbeat
fresh while the command runs, proves the instance is gone, and writes a
schema-valid receipt + ledger line even when the run fails or is refused.

No gate, threshold, floor or claim value is moved by this module.

Review fixes (PR #440 review, 2026-09-06): the heartbeat is refreshed while the
command runs and a guard-initiated teardown is recorded as ``ALARM`` /
``invalid``; ``approval_overrides`` (``all-of:``) from the policy; ``commit_sha``
/ ``branch`` / ``dirty_tree`` from git and a schema check before every write;
no identity defaults on the command line; a live-provider refusal writes a
receipt; a role without a ceiling row is refused; approval permalinks must be
real Slack permalinks of the coordination channel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PERMALINK_RE = re.compile(r"^https://cerin-amroth\.slack\.com/archives/C[A-Z0-9]{8,12}/p\d{16}(\?\S*)?$")
KNOWN_ROLES = ("CEO", "CTO", "CSO", "CDO", "COO", "CXO", "Jordan")
COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
SCHEMA_PATH = Path("docs/run-receipt-schema.json")
STATUS_ENUM = ("OK", "REFUSED", "OOM", "INSTALL_FAILED", "LOAD_FAULT", "HARNESS_ERROR", "ALARM", "NOT_RUN")
RESULT_ENUM = ("pass", "fail", "inconclusive", "invalid")


class RentRefused(RuntimeError):
    """Policy refusal before any instance is created."""


class ReceiptInvalid(RuntimeError):
    """A receipt that does not satisfy docs/run-receipt-schema.json (a launcher bug, never written)."""


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


def _permalink_ok(url: Any) -> bool:
    return isinstance(url, str) and bool(PERMALINK_RE.match(url))


def _jordan_approved(approvals: list[dict[str, Any]]) -> bool:
    """Jordan's approval lifts every threshold and the hard cap -- only as role AND agent 'Jordan'
    with a real permalink of the coordination channel (the launcher cannot read Slack; the receipt
    records the permalink so the ledger check can resolve it against the org-corpus raw layer)."""
    return any(a.get("role") == "Jordan" and a.get("agent") == "Jordan" and _permalink_ok(a.get("slack_permalink"))
               for a in approvals)


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
    if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool):
        # Fail closed: a role without a confirmed ceiling row does not rent under the global budget alone.
        raise RentRefused(
            f"role {role!r} has no numeric row in role_daily_ceiling_usd -- Jordan adds the row before this role rents")
    spent = _today(ledger, date_utc, role=role)
    if spent + estimate > float(ceiling):
        raise RentRefused(
            f"{role} daily spend ${spent:.2f} + estimate ${estimate:.2f} exceeds ceiling ${float(ceiling):.2f}")
    global_budget = float(policy["global_daily_budget_usd"])
    gspent = _today(ledger, date_utc)
    if gspent + estimate > global_budget:
        raise RentRefused(
            f"global daily spend ${gspent:.2f} + estimate ${estimate:.2f} exceeds budget ${global_budget:.2f}")


def select_approver_spec(policy: dict[str, Any], estimate: float,
                         seat_executors: dict[str, str] | None = None) -> str:
    """The approver spec for this estimate: the first threshold whose max_usd covers it, then any
    ``approval_overrides`` entry whose ``band_usd`` (lo, hi] contains the estimate and whose
    ``while_role_executor`` seats are either undeclared or held by the named executor. Overrides only
    ever tighten; a matching override replaces the spec (e.g. ``one-of:CTO,CSO`` -> ``all-of:CTO,CSO``).
    Declaring a seat with a different holder (``--seat-executor CTO=<other>``) is the only way to the
    base spec -- omitting the declaration is not (PR #440 Warden read, finding 2a)."""
    applicable = None
    for th in policy["approval_thresholds"]:
        max_usd = th["max_usd"]
        if max_usd is None or estimate <= float(max_usd):
            applicable = th
            break
    if applicable is None:
        raise RentRefused(f"no approval threshold for ${estimate:.2f}")
    spec = str(applicable["approver"])
    seats = seat_executors or {}
    for ov in policy.get("approval_overrides") or []:
        lo, hi = ov["band_usd"]
        if not (float(lo) < estimate <= float(hi)):
            continue
        cond = ov.get("while_role_executor") or {}
        if not cond:
            continue
        # Fail closed: an undeclared seat is never the loophole. The override is skipped only when every
        # seat it names is declared (--seat-executor) AND held by someone other than the named executor.
        declared_elsewhere = all(r in seats and seats[r] != ex for r, ex in cond.items())
        if not declared_elsewhere:
            spec = str(ov["approver"])
    return spec


def _roles_of(approvals: list[dict[str, Any]]) -> set[str]:
    return {str(a.get("role")) for a in approvals}


def check_approvals(policy: dict[str, Any], estimate: float,
                    approvals: list[dict[str, Any]], *, role: str,
                    seat_executors: dict[str, str] | None = None) -> str:
    """Refuse unless the approvals satisfy the spec; return the spec that applied (recorded in the receipt)."""
    for a in approvals:
        if a.get("role") not in KNOWN_ROLES:
            raise RentRefused(f"approval role {a.get('role')!r} is not one of {list(KNOWN_ROLES)}")
        if not _permalink_ok(a.get("slack_permalink")):
            raise RentRefused(
                f"approval permalink is not a #ml-packages Slack permalink "
                f"(https://cerin-amroth.slack.com/archives/C…/p<16 digits>): {a.get('slack_permalink')!r}")
    spec = select_approver_spec(policy, estimate, seat_executors)
    if spec == "requesting-agent":
        return spec
    if _jordan_approved(approvals):
        return spec
    if spec == "Jordan":
        raise RentRefused(f"estimate ${estimate:.2f} requires Jordan's approval (role and agent 'Jordan')")
    got = _roles_of(approvals)
    if spec.startswith("one-of:"):
        need = set(spec.split(":", 1)[1].split(","))
        if not (need & got):
            raise RentRefused(f"estimate ${estimate:.2f} requires one of {sorted(need)}, got {sorted(got)}")
        return spec
    if spec.startswith("two-of:"):
        need = set(spec.split(":", 1)[1].split(","))
        if len(need & got) < 2:
            raise RentRefused(f"estimate ${estimate:.2f} requires two of {sorted(need)}, got {sorted(got)}")
        return spec
    if spec.startswith("all-of:"):
        need = set(spec.split(":", 1)[1].split(","))
        missing = need - got
        if missing:
            raise RentRefused(
                f"estimate ${estimate:.2f} requires all of {sorted(need)} (executor-conditional override), "
                f"missing {sorted(missing)}")
        return spec
    raise RentRefused(f"unknown approver spec {spec!r}")


def evaluate_launch(policy: dict[str, Any], ledger: list[dict[str, Any]], *,
                    role: str, estimate: float, provider: str, gpu: str,
                    wallclock_h: float, approvals: list[dict[str, Any]],
                    date_utc: str, seat_executors: dict[str, str] | None = None) -> str:
    check_provider_gpu(policy, provider, gpu)
    check_wallclock(policy, wallclock_h)
    check_hard_cap(policy, estimate, approvals)
    check_ceilings(policy, ledger, role=role, estimate=estimate, date_utc=date_utc)
    return check_approvals(policy, estimate, approvals, role=role, seat_executors=seat_executors)


def git_facts(cwd: Path | None = None) -> dict[str, Any]:
    """commit_sha / branch / dirty_tree of the repository the launcher runs in. Refuses when git cannot
    answer: a receipt without a real commit_sha is not a record (schema pattern ^[0-9a-f]{7,40}$)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
    try:
        sha = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD") or "HEAD"
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        raise RentRefused(f"cannot read commit_sha/branch from git (cwd={cwd or os.getcwd()}): {e}") from e
    if not COMMIT_SHA_RE.match(sha):
        raise RentRefused(f"git returned a non-hex commit sha {sha!r}")
    return {"commit_sha": sha, "branch": branch, "dirty_tree": dirty}


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


def _write_proof(proof_path: str | Path, proof: dict[str, Any]) -> None:
    """Atomic write (tmp + os.replace) so a SIGTERM mid-write never leaves a truncated proof."""
    target = Path(proof_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(proof, indent=2) + "\n")
    os.replace(tmp, target)


def guard_worker(*, instance_id: str, provider_kind: str, fake_state: str | None,
                 wallclock_s: float, heartbeat_path: str, proof_path: str,
                 heartbeat_timeout_s: float) -> int:
    """Controller-side teardown guard. Destroys the instance on wallclock or heartbeat loss and writes
    the proof with the reason; exits quietly (reason 'already-gone') once the launcher has torn down."""
    prov = provider_for(provider_kind, fake_state=Path(fake_state) if fake_state else None)
    deadline = time.time() + wallclock_s
    hb = Path(heartbeat_path)
    poll = min(0.2, max(0.05, wallclock_s / 20))
    reason = "wallclock"
    while time.time() < deadline:
        time.sleep(poll)
        if instance_id not in prov.list_ids():
            reason = "already-gone"
            break
        if hb.is_file():
            age = time.time() - hb.stat().st_mtime
            if age > heartbeat_timeout_s:
                reason = "heartbeat-loss"
                break
    if reason == "already-gone":
        # The launcher (or someone) destroyed it first; nothing to prove beyond the absence.
        if not Path(proof_path).is_file():
            _write_proof(proof_path, {"method": f"{provider_kind}-observed-absent", "reason": reason,
                                      "evidence": json.dumps({"list_after": sorted(prov.list_ids()),
                                                              "instance_absent": True}, sort_keys=True),
                                      "complete": True, "at": _utc()})
        return 0
    evidence = prov.destroy(instance_id)
    remaining = prov.list_ids()
    gone = instance_id not in remaining
    proof = {
        "method": evidence.get("method", f"{provider_kind}-destroy"),
        "evidence": json.dumps({"destroy": evidence, "list_after": sorted(remaining),
                                "instance_absent": gone, "reason": reason}, sort_keys=True),
        "complete": gone,
        "reason": reason,
        "at": _utc(),
    }
    # The launcher's own proof (reason "completion") is never overwritten; a guard proof that lands
    # second goes to a sidecar so the on-disk trail stays consistent with the receipt.
    target = Path(proof_path)
    _write_proof(target if not target.is_file() else target.with_name(target.stem + ".guard.json"), proof)
    return 0 if gone else 1


class HeartbeatRefresher:
    """Touches the heartbeat file every ``interval`` seconds while the command runs (HIGH-1 fix)."""

    def __init__(self, path: Path, interval: float):
        self.path = Path(path)
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rent-heartbeat", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.path.write_text(_utc())
            except OSError:
                pass

    def __enter__(self) -> "HeartbeatRefresher":
        self.path.write_text(_utc())
        if self.interval > 0:
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def validate_receipt(receipt: dict[str, Any], schema_path: Path = SCHEMA_PATH) -> None:
    """Raise ReceiptInvalid unless the receipt satisfies docs/run-receipt-schema.json. Uses jsonschema
    when importable; otherwise the required keys, the commit_sha pattern, the string/number types the
    ledger check relies on, and the status/result enums."""
    if not Path(schema_path).is_file():
        # Refuse rather than degrade: without the schema there is no statement of what a receipt is.
        raise ReceiptInvalid(f"receipt schema not found at {schema_path}; refusing to write an unvalidated receipt")
    schema = json.loads(Path(schema_path).read_text())
    problems: list[str] = []
    try:
        import jsonschema  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency
        jsonschema = None
    if jsonschema is not None:
        validator = jsonschema.Draft202012Validator(schema)
        problems = [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in validator.iter_errors(receipt)]
    else:
        required = schema.get("required") or []
        problems += [f"missing required field {k!r}" for k in required if k not in receipt]
        for a in receipt.get("approvals") or []:
            if not (isinstance(a, dict) and all(k in a for k in ("role", "agent", "usd_estimate", "slack_permalink"))):
                problems.append("approvals[] items need role/agent/usd_estimate/slack_permalink")
                break
        tp = receipt.get("teardown_proof")
        if not (isinstance(tp, dict) and isinstance(tp.get("method"), str) and tp["method"] and isinstance(tp.get("evidence"), str)):
            problems.append("teardown_proof needs string method and evidence")
        for k in ("started_at", "finished_at"):
            try:
                datetime.fromisoformat(str(receipt.get(k, "")).replace("Z", "+00:00"))
            except ValueError:
                problems.append(f"{k} must be an ISO-8601 timestamp")
        for art in receipt.get("artifacts") or []:
            if not (isinstance(art, dict) and all(k in art for k in ("path", "sha256", "bytes"))):
                problems.append("artifacts[] items need path/sha256/bytes")
                break
        if not isinstance(receipt.get("gpu_count"), int) or isinstance(receipt.get("gpu_count"), bool):
            problems.append("gpu_count must be an integer")
        if not isinstance(receipt.get("runtime_seconds"), (int, float)):
            problems.append("runtime_seconds must be a number")
        if not isinstance(receipt.get("dirty_tree"), bool):
            problems.append("dirty_tree must be a boolean")
        if not COMMIT_SHA_RE.match(str(receipt.get("commit_sha", ""))):
            problems.append("commit_sha must match ^[0-9a-f]{7,40}$")
        if not isinstance(receipt.get("command"), str) or not receipt["command"]:
            problems.append("command must be a non-empty string")
        cu = receipt.get("cost_usd")
        if not (isinstance(cu, dict) and all(isinstance(cu.get(k), (int, float)) and cu[k] >= 0 for k in ("estimated", "actual"))):
            problems.append("cost_usd.estimated/actual must be numbers >= 0")
        if receipt.get("status") not in STATUS_ENUM:
            problems.append(f"status must be one of {STATUS_ENUM}")
        if receipt.get("result") not in RESULT_ENUM:
            problems.append(f"result must be one of {RESULT_ENUM}")
        for k in ("branch", "requested_by", "executed_by", "preregistration", "hypothesis", "notes", "decision"):
            if not isinstance(receipt.get(k), str) or not receipt[k]:
                problems.append(f"{k} must be a non-empty string")
    if problems:
        raise ReceiptInvalid("receipt does not satisfy the schema: " + "; ".join(problems[:8]))


def write_receipt(dir_path: Path, receipt: dict[str, Any], *, schema_path: Path = SCHEMA_PATH) -> Path:
    validate_receipt(receipt, schema_path)
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


def build_receipt(*, experiment_id: str, work_id: str, requested_by: str, executed_by: str,
                  hypothesis: str, expected_result: str, success_criteria: str, failure_criteria: str,
                  preregistration: str, approvals: list[dict[str, Any]], commit_sha: str, branch: str,
                  dirty_tree: bool, command: str, environment: dict[str, Any], provider: str,
                  instance_id: str, gpu_model: str, gpu_count: int, started_at: str, finished_at: str,
                  runtime_seconds: float, cost_estimated: float, cost_actual: float,
                  teardown_proof: dict[str, Any], status: str, result: str, notes: str,
                  configuration: dict[str, Any], dataset: str = "none", dataset_hash: str = "none",
                  model: str = "none", model_revision: str = "none", model_hash: str = "none",
                  seed: int = 0, container_image: str = "none", decision: str | None = None,
                  cpu: str = "unknown", ram: str = "unknown", storage: str = "unknown",
                  complete: bool = False) -> dict[str, Any]:
    """A receipt with no identity defaults: who/what/why come from the caller (the command line), the
    repository facts from git, the outcome from what happened. Fields that do not apply to a launch
    without a dataset or model say so ("none") -- they are not guesses; cpu/ram/storage are "unknown" until a
    live adapter reports them (override with --cpu/--ram/--storage)."""
    return {
        "experiment_id": experiment_id, "work_id": work_id, "requested_by": requested_by,
        "executed_by": executed_by, "reviewed_by": None, "hypothesis": hypothesis,
        "expected_result": expected_result, "success_criteria": success_criteria,
        "failure_criteria": failure_criteria, "preregistration": preregistration,
        "approvals": approvals, "repo": "pjordanandrsn/experts4bit-qlora", "commit_sha": commit_sha,
        "branch": branch, "dirty_tree": bool(dirty_tree), "container_image": container_image,
        "dependencies": {"python": sys.version.split()[0]}, "command": command, "environment": environment,
        "provider": provider, "instance_id": instance_id, "gpu_model": gpu_model, "gpu_count": int(gpu_count),
        "cpu": cpu, "ram": ram, "storage": storage,
        "started_at": started_at, "finished_at": finished_at, "runtime_seconds": float(runtime_seconds),
        "cost_usd": {"estimated": float(cost_estimated), "actual": float(cost_actual)},
        "dataset": dataset, "dataset_hash": dataset_hash, "model": model, "model_revision": model_revision,
        "model_hash": model_hash, "seed": int(seed), "configuration": configuration, "metrics": {},
        "artifacts": [], "teardown_proof": teardown_proof, "status": status, "result": result,
        # docs/COMPUTE-GOVERNANCE.md vocabulary: adopt|refute|void|pending + link -- a launch is never a decision
        "decision": decision if decision else f"pending {work_id}", "notes": notes, "complete": bool(complete),
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
        raise RentRefused("--approval must be ROLE/AGENT=https://cerin-amroth.slack.com/archives/C…/p…")
    who, permalink = raw.split("=", 1)
    if "/" not in who:
        raise RentRefused("--approval identity must be ROLE/AGENT")
    role, agent = who.split("/", 1)
    return {"role": role, "agent": agent, "usd_estimate": estimate,
            "slack_permalink": permalink}


def _parse_seat_executors(raw: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise RentRefused("--seat-executor must be ROLE=<executor>, e.g. CTO=cursor-desktop-mini/grok")
        role, ex = item.split("=", 1)
        out[role.strip()] = ex.strip()
    return out


def _guard_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="rent guard worker (internal)")
    ap.add_argument("--guard-worker", action="store_true", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--provider", default="fake")
    ap.add_argument("--fake-state")
    ap.add_argument("--wallclock-s", type=float, required=True)
    ap.add_argument("--heartbeat", required=True)
    ap.add_argument("--proof", required=True)
    ap.add_argument("--heartbeat-timeout-s", type=float, default=30.0)
    args = ap.parse_args(argv)
    return guard_worker(
        instance_id=args.instance_id, provider_kind=args.provider,
        fake_state=args.fake_state, wallclock_s=float(args.wallclock_s),
        heartbeat_path=args.heartbeat, proof_path=args.proof,
        heartbeat_timeout_s=float(args.heartbeat_timeout_s))


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_argument_group("who / what / why (required -- a receipt carries no defaults for these)")
    who.add_argument("--role", required=True, help="requesting role, e.g. CTO (must have a ceiling row)")
    who.add_argument("--agent", required=True, help="requesting executor, e.g. cursor-desktop-mini/grok")
    who.add_argument("--work-id", required=True, help="the issue this run serves, e.g. experts4bit-qlora#433")
    who.add_argument("--preregistration", required=True, help="path/URL of the pre-registration, e.g. bench/p41/P41-PREREG.md")
    who.add_argument("--hypothesis", required=True)
    who.add_argument("--expected-result", required=True)
    who.add_argument("--success-criteria", required=True)
    who.add_argument("--failure-criteria", required=True)
    ap.add_argument("--seat-executor", action="append", default=[], metavar="ROLE=EXECUTOR",
                    help="who currently holds a seat, for executor-conditional approval overrides "
                         "(e.g. CTO=cursor-desktop-mini/grok); repeatable")
    ap.add_argument("--policy", default="docs/compute-policy.json")
    ap.add_argument("--ledger", default="bench/runs/ledger.jsonl")
    ap.add_argument("--runs-root", default="bench/runs")
    ap.add_argument("--schema", default=str(SCHEMA_PATH))
    ap.add_argument("--run-id")
    ap.add_argument("--provider", default="vast:verified-secure")
    ap.add_argument("--fake-state")
    ap.add_argument("--gpu", default="RTX 5090")
    ap.add_argument("--usd-per-hour", type=float, required=True)
    ap.add_argument("--wallclock-h", type=float, required=True)
    ap.add_argument("--image", default="none")
    ap.add_argument("--approval", action="append", default=[], metavar="ROLE/AGENT=PERMALINK")
    ap.add_argument("--dry-run", action="store_true", help="fake provider; still writes a receipt")
    ap.add_argument("--command", default="", help="the workload to run while the instance is up")
    ap.add_argument("--heartbeat-timeout-s", type=float, default=30.0,
                    help="the guard tears down when the heartbeat file is older than this")
    ap.add_argument("--heartbeat-refresh-s", type=float, default=None,
                    help="how often the launcher refreshes the heartbeat while --command runs "
                         "(default: timeout / 3; 0 disables -- tests only)")
    ap.add_argument("--cpu", default="unknown", help="host cpu as the provider reports it")
    ap.add_argument("--ram", default="unknown", help="host ram as the provider reports it")
    ap.add_argument("--storage", default="unknown", help="host storage as the provider reports it")
    ap.add_argument("--dataset", default="none")
    ap.add_argument("--dataset-hash", default="none")
    ap.add_argument("--model", default="none")
    ap.add_argument("--model-revision", default="none")
    ap.add_argument("--model-hash", default="none")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--guard-worker" in argv:
        return _guard_main(argv)
    args = _parser().parse_args(argv)

    schema_path = Path(args.schema)
    policy = load_policy(Path(args.policy))
    ledger = load_ledger(Path(args.ledger))
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id = args.run_id or f"rent-{date_utc}-{os.getpid()}"
    who = f"{args.role}/{args.agent}"
    provider = "fake" if args.dry_run else args.provider
    rec_dir = Path(args.runs_root) / date_utc / run_id
    command_str = args.command or " ".join([sys.executable, "-m", "experts4bit_qlora.tools.rent", *argv])
    t0 = time.time()
    started_at = _utc()

    try:
        facts = git_facts()
    except RentRefused as e:
        # Without a real commit sha there is no valid receipt to write (schema: ^[0-9a-f]{7,40}$).
        print(f"REFUSED (no receipt written): {e}", file=sys.stderr)
        return 2

    try:
        estimate = estimate_usd(usd_per_hour=args.usd_per_hour, wallclock_h=args.wallclock_h)
        approvals = [_parse_approval(a, estimate) for a in args.approval]
        seat_executors = _parse_seat_executors(args.seat_executor)
    except RentRefused as e:
        print(f"REFUSED (no receipt written -- the request itself is malformed): {e}", file=sys.stderr)
        return 2

    environment: dict[str, Any] = {"seat_executors": seat_executors, "approver_spec": None,
                                   "heartbeat_timeout_s": args.heartbeat_timeout_s, "python": sys.version.split()[0]}
    configuration = {"dry_run": bool(args.dry_run), "wallclock_h": args.wallclock_h,
                     "usd_per_hour": args.usd_per_hour, "gpu": args.gpu, "image": args.image}

    def refused(msg: str, spec: str | None) -> int:
        environment["approver_spec"] = spec
        rec = build_receipt(
            experiment_id=run_id, work_id=args.work_id, requested_by=who, executed_by=who,
            hypothesis=args.hypothesis, expected_result=args.expected_result,
            success_criteria=args.success_criteria, failure_criteria=args.failure_criteria,
            preregistration=args.preregistration, approvals=approvals, command=command_str,
            environment=environment, provider=provider, instance_id="none", gpu_model=args.gpu,
            gpu_count=0, started_at=started_at, finished_at=_utc(), runtime_seconds=time.time() - t0,
            cost_estimated=estimate, cost_actual=0.0,
            teardown_proof={"method": "not-launched", "evidence": f"refused before launch: {msg}"},
            status="REFUSED", result="invalid", notes=msg, configuration=configuration,
            dataset=args.dataset, dataset_hash=args.dataset_hash, model=args.model,
            model_revision=args.model_revision, model_hash=args.model_hash, seed=args.seed,
            container_image=args.image, cpu=args.cpu, ram=args.ram, storage=args.storage, complete=True, **facts)
        write_receipt(rec_dir, rec, schema_path=schema_path)
        append_ledger(Path(args.ledger), run_id=run_id, date_utc=date_utc, role=args.role, cost_usd=0.0)
        maybe_slack(f"REFUSED {run_id}: {msg}")
        print(f"REFUSED: {msg}", file=sys.stderr)
        return 2

    policy_provider = "vast:verified-secure" if provider == "fake" else provider
    try:  # the spec that applies is part of the record even when the run is refused
        spec: str | None = select_approver_spec(policy, estimate, seat_executors)
    except RentRefused:
        spec = None
    environment["approver_spec"] = spec
    try:
        spec = evaluate_launch(
            policy, ledger, role=args.role, estimate=estimate,
            provider=policy_provider, gpu=args.gpu, wallclock_h=args.wallclock_h,
            approvals=approvals, date_utc=date_utc, seat_executors=seat_executors)
        environment["approver_spec"] = spec
        fake_state = Path(args.fake_state) if args.fake_state else rec_dir / "fake-state.json"
        prov = provider_for(provider, fake_state=fake_state)  # MEDIUM-3: a live-provider refusal is a receipt too
    except RentRefused as e:
        return refused(str(e), spec)

    rec_dir.mkdir(parents=True, exist_ok=True)
    iid = prov.launch(gpu=args.gpu, wallclock_h=args.wallclock_h, image=args.image)
    maybe_slack(f"LAUNCHED {run_id} instance={iid}")
    hb = rec_dir / "heartbeat"
    proof_path = rec_dir / "teardown-proof.json"
    wallclock_s = args.wallclock_h * 3600
    refresh = args.heartbeat_refresh_s if args.heartbeat_refresh_s is not None else args.heartbeat_timeout_s / 3
    hb.write_text(_utc())
    guard = spawn_guard(
        python=sys.executable,
        module_args=[
            "--guard-worker", "--instance-id", iid, "--provider", prov.kind,
            "--fake-state", str(fake_state), "--wallclock-s", str(wallclock_s),
            "--heartbeat", str(hb), "--proof", str(proof_path),
            "--heartbeat-timeout-s", str(args.heartbeat_timeout_s),
        ],
        log_path=rec_dir / "guard.log",
    )
    status, result, notes = "OK", "pass", f"guard pid {guard.pid}"
    if args.command:
        with HeartbeatRefresher(hb, refresh):  # HIGH-1: the heartbeat stays fresh for the whole command
            try:
                subprocess.run(args.command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                status, result, notes = "HARNESS_ERROR", "fail", f"command exited {e.returncode}"

    # Teardown. `pass` is written only when the launcher itself destroyed a live instance; an instance that
    # is already gone -- by the guard (heartbeat-loss / wallclock) or by anyone else -- is ALARM / invalid.
    def _read_proof() -> dict[str, Any] | None:
        return json.loads(proof_path.read_text()) if proof_path.is_file() else None
    proof = _read_proof()
    if proof is None and iid not in prov.list_ids():
        # Gone without a proof: the guard may be between destroy and write (1a) -- give it a moment.
        deadline = time.time() + 3
        while time.time() < deadline and (proof := _read_proof()) is None:
            time.sleep(0.05)
        if proof is None:
            proof = {"method": f"{prov.kind}-observed-absent", "reason": "torn-down-externally",
                     "evidence": json.dumps({"instance_absent": True, "list_after": sorted(prov.list_ids())},
                                            sort_keys=True), "complete": True, "at": _utc()}
            _write_proof(proof_path, proof)
    if proof is not None:
        reason = proof.get("reason") or "unknown"
        status, result = "ALARM", "invalid"
        notes = f"instance torn down before the launcher's own teardown: {reason} (" + notes + ")"
    else:
        try:
            evidence = prov.destroy(iid)
        except Exception as e:  # noqa: BLE001 - a live adapter may raise; the receipt still records it (1b)
            evidence = {"method": f"{prov.kind}-destroy-failed", "error": repr(e)}
            status, result = "ALARM", "invalid"
            notes = f"teardown failed: {e!r} (" + notes + ")"
        remaining = prov.list_ids()
        proof = {"method": evidence.get("method", f"{prov.kind}-destroy"), "reason": "completion",
                 "evidence": json.dumps({"destroy": evidence, "list_after": sorted(remaining),
                                         "instance_absent": iid not in remaining}, sort_keys=True),
                 "complete": iid not in remaining, "at": _utc()}
        _write_proof(proof_path, proof)
        try:
            os.kill(guard.pid, signal.SIGTERM)
        except OSError:
            pass
    gone = iid not in prov.list_ids()
    complete = gone and proof.get("method") not in (None, "pending")
    actual_cost = 0.0  # fake provider bills nothing; a live adapter must read the provider's billing, never copy the estimate
    rec = build_receipt(
        experiment_id=run_id, work_id=args.work_id, requested_by=who, executed_by=who,
        hypothesis=args.hypothesis, expected_result=args.expected_result,
        success_criteria=args.success_criteria, failure_criteria=args.failure_criteria,
        preregistration=args.preregistration, approvals=approvals, command=command_str,
        environment=environment, provider=prov.kind, instance_id=iid, gpu_model=args.gpu, gpu_count=1,
        started_at=started_at, finished_at=_utc(), runtime_seconds=time.time() - t0,
        cost_estimated=estimate, cost_actual=actual_cost,
        teardown_proof={k: v for k, v in proof.items() if k in ("method", "evidence", "reason", "at")},
        status=status, result=result, notes=notes, configuration=configuration,
        dataset=args.dataset, dataset_hash=args.dataset_hash, model=args.model,
        model_revision=args.model_revision, model_hash=args.model_hash, seed=args.seed,
        container_image=args.image, cpu=args.cpu, ram=args.ram, storage=args.storage, complete=complete, **facts)
    rec_path = write_receipt(rec_dir, rec, schema_path=schema_path)
    rec["artifacts"] = [{"path": "receipt.json", "sha256": _sha256_file(rec_path), "bytes": rec_path.stat().st_size}]
    write_receipt(rec_dir, rec, schema_path=schema_path)
    append_ledger(Path(args.ledger), run_id=run_id, date_utc=date_utc, role=args.role, cost_usd=actual_cost)
    maybe_slack(f"DONE {run_id} status={status}")
    maybe_slack(f"TORN DOWN {run_id} instance={iid} gone={gone}")
    print(f"{'DRY-RUN' if args.dry_run else 'LAUNCHED'} {run_id} instance={iid} status={status} "
          f"complete={rec['complete']} receipt={rec_path}")
    return 0 if (rec["complete"] and status == "OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
