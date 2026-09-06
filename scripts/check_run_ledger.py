#!/usr/bin/env python3
"""Compute governance check: validate run receipts against the policy.

Enforces the compute policy from docs/COMPUTE-GOVERNANCE.md:
  1. Every receipt.json under bench/runs/ validates against docs/run-receipt-schema.json
  2. Each receipt's experiment_id and cost_usd.actual appear in bench/runs/ledger.jsonl
     (ledger uses run_id for backwards compat; experiment_id == run_id)
  3. Per-role daily sums <= role ceilings; global daily sums <= global budget
     (from docs/compute-policy.json)
  4. Each receipt's approvals satisfy the threshold for its cost_usd.estimated,
     UNLESS the receipt carries incident: "<repo>#<n>" — in that case the threshold
     check is replaced by the incident reference itself (format verified; in CI the
     issue is resolved via the GitHub API and must have label 'incident' or title
     starting with '[incident]'; locally a NOT VERIFIED line is printed to stderr —
     never a silent pass; status ALARM / result invalid / decision abandon enforced
     in validate_receipt). The cost still counts toward daily ceilings and global budget.
  5. Each receipt has non-empty teardown_proof

Exit 1 with file:line-style messages on any violation; exit 0 with
"OK: N receipts, M days" on success.

    python scripts/check_run_ledger.py

Schema validation uses jsonschema.Draft202012Validator when jsonschema is importable
(true in lint-and-test via .[test]; the discoverability job also installs
jsonschema>=4.18 before this step so the strict path runs there too).
The manual fallback derives required fields from docs/run-receipt-schema.json["required"]
so the schema is the single source of truth — the hardcoded REQUIRED_FIELDS set is gone.
ISO-8601 timestamp format is always validated by a fromisoformat() loop run outside the
if/else because Draft 2020-12 does not enforce format: date-time without a format checker,
so `started_at: "yesterday"` would otherwise pass the strict path.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path("bench/runs")
LEDGER = RUNS_DIR / "ledger.jsonl"
POLICY = Path("docs/compute-policy.json")
PERMALINK_RE = re.compile(r"^https://cerin-amroth\.slack\.com/archives/C[A-Z0-9]{8,12}/p\d{16}(\?\S*)?$")
# incident field format: <owner>/<repo>#<n> or <repo>#<n>
INCIDENT_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?#[0-9]+$")
SCHEMA = Path("docs/run-receipt-schema.json")

STATUS_VALUES = {
    "OK",
    "REFUSED",
    "OOM",
    "INSTALL_FAILED",
    "LOAD_FAULT",
    "HARNESS_ERROR",
    "ALARM",
    "NOT_RUN",
}

RESULT_VALUES = {"pass", "fail", "inconclusive", "invalid"}


def fail(path: Path | str, line: int | None, message: str) -> None:
    """Print a file:line finding and exit 1."""
    loc = f"{path}:{line}" if line else str(path)
    print(f"{loc} — {message}", file=sys.stderr)
    sys.exit(1)


def load_json(path: Path) -> Any:
    """Load JSON; fail with file:0 on parse error."""
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        fail(path, e.lineno, f"JSON parse error: {e.msg}")
    except FileNotFoundError:
        fail(path, None, "file not found")


def _verify_incident_exists(path: Path, incident: str) -> None:
    """Verify the incident issue exists on GitHub and is labelled as an incident.

    Owner-less form (``<repo>#<n>``) defaults to ``pjordanandrsn/``.

    In CI (``GITHUB_TOKEN`` present and non-empty): calls the GitHub REST API.
    The issue must have the label ``incident`` **or** a title starting with
    ``[incident]``.  Any HTTP error (404, rate-limit, network timeout) causes
    ``fail()``; never a silent pass.

    Locally (``GITHUB_TOKEN`` absent or empty): prints a ``NOT VERIFIED`` line
    to stderr and returns.  Operators must verify the reference manually.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()

    # Parse: <owner>/<repo>#<n>  or  <repo>#<n>
    hash_idx = incident.rindex("#")
    issue_num = incident[hash_idx + 1 :]
    owner_repo_part = incident[:hash_idx]
    owner_repo = owner_repo_part if "/" in owner_repo_part else f"pjordanandrsn/{owner_repo_part}"

    if not token:
        print(
            f"{path} — NOT VERIFIED — incident {incident!r} not checked "
            f"(GITHUB_TOKEN absent); verify manually before accepting",
            file=sys.stderr,
        )
        return

    api_url = f"https://api.github.com/repos/{owner_repo}/issues/{issue_num}"
    req = urllib.request.Request(
        api_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            issue_data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        fail(
            path,
            0,
            f"incident {incident!r}: GitHub API returned HTTP {exc.code} — "
            f"issue not found or access denied",
        )
    except Exception as exc:  # noqa: BLE001
        fail(path, 0, f"incident {incident!r}: GitHub API call failed: {exc}")

    labels = {lbl.get("name", "") for lbl in issue_data.get("labels") or []}
    title = str(issue_data.get("title", ""))
    if "incident" not in labels and not title.startswith("[incident]"):
        fail(
            path,
            0,
            f"incident {incident!r}: issue #{issue_num} in {owner_repo!r} has neither "
            f"label 'incident' nor title starting with '[incident]' "
            f"(title={title!r}, labels={sorted(labels)!r})",
        )


def validate_receipt(path: Path, data: dict[str, Any]) -> None:
    """Validate a single receipt against docs/run-receipt-schema.json.

    Strict path (jsonschema importable): jsonschema.Draft202012Validator against
    the full schema — catches required-field drift and enum violations automatically.
    Manual fallback: derives required fields from schema["required"] so the schema
    file is the single source of truth; no hardcoded REQUIRED_FIELDS list.
    Business-logic constraints not expressed in the schema (Slack permalink pattern,
    ISO-8601 timestamp format, incident status/result/decision requirements) are
    enforced in both paths; Draft 2020-12 does not enforce format: date-time without
    a format checker.
    Incident receipts: when incident is present the format is enforced by the schema
    pattern (strict) or INCIDENT_RE (fallback); the issue is verified to exist via
    _verify_incident_exists() (API in CI, NOT VERIFIED locally); status ALARM, result
    invalid, and decision starting with 'abandon' are required in both paths.
    """
    if not SCHEMA.exists():
        fail(SCHEMA, None, "run-receipt-schema.json not found; cannot validate receipts")
    schema: dict[str, Any] = load_json(SCHEMA)

    try:
        import jsonschema  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001
        jsonschema = None

    if jsonschema is not None:
        # Strict path: full Draft 2020-12 validation catches required fields,
        # enum constraints (status, result, teardown_proof.reason, …) and types.
        validator = jsonschema.Draft202012Validator(schema)
        errs = sorted(validator.iter_errors(data), key=lambda e: [str(p) for p in e.path])
        if errs:
            msgs = [
                f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                for e in errs[:8]
            ]
            fail(path, 0, "schema: " + "; ".join(msgs))
    else:
        # Manual fallback: required fields from the schema (single source of truth).
        required = set(schema.get("required") or [])
        missing = required - data.keys()
        if missing:
            fail(path, 0, f"missing required fields: {', '.join(sorted(missing))}")

        # Validate requested_by (string, non-empty)
        if not isinstance(data.get("requested_by"), str) or not data["requested_by"]:
            fail(path, 0, "requested_by must be a non-empty string")

        # Validate approvals array items
        if not isinstance(data.get("approvals"), list):
            fail(path, 0, "approvals must be an array")
        for i, appr in enumerate(data["approvals"]):
            if not isinstance(appr, dict):
                fail(path, 0, f"approvals[{i}] must be an object")
            for field in ["role", "agent", "usd_estimate", "slack_permalink"]:
                if field not in appr:
                    fail(path, 0, f"approvals[{i}] missing '{field}'")
            if not isinstance(appr["usd_estimate"], (int, float)):
                fail(path, 0, f"approvals[{i}].usd_estimate must be a number")
            if appr["usd_estimate"] < 0:
                fail(path, 0, f"approvals[{i}].usd_estimate must be >= 0")

        # Validate repo and commit_sha
        if not isinstance(data.get("repo"), str) or not data["repo"]:
            fail(path, 0, "repo must be a non-empty string")
        if not re.fullmatch(r"[0-9a-f]{7,40}", str(data.get("commit_sha", ""))):
            fail(path, 0, "commit_sha must be a hex sha (7-40 chars)")

        # Validate cost_usd
        if not isinstance(data.get("cost_usd"), dict):
            fail(path, 0, "cost_usd must be an object")
        cu = data["cost_usd"]
        if "estimated" not in cu or "actual" not in cu:
            fail(path, 0, "cost_usd must have 'estimated' and 'actual'")
        if not isinstance(cu["estimated"], (int, float)) or cu["estimated"] < 0:
            fail(path, 0, "cost_usd.estimated must be a number >= 0")
        if not isinstance(cu["actual"], (int, float)) or cu["actual"] < 0:
            fail(path, 0, "cost_usd.actual must be a number >= 0")

        # Validate status and result enums
        if data.get("status") not in STATUS_VALUES:
            fail(
                path,
                0,
                f"status must be one of {sorted(STATUS_VALUES)}, got {data.get('status')!r}",
            )
        if data.get("result") not in RESULT_VALUES:
            fail(
                path,
                0,
                f"result must be one of {sorted(RESULT_VALUES)}, got {data.get('result')!r}",
            )

        # Validate artifacts array
        if not isinstance(data.get("artifacts"), list):
            fail(path, 0, "artifacts must be an array")
        for i, art in enumerate(data["artifacts"]):
            if not isinstance(art, dict):
                fail(path, 0, f"artifacts[{i}] must be an object")
            for field in ["path", "sha256", "bytes"]:
                if field not in art:
                    fail(path, 0, f"artifacts[{i}] missing '{field}'")

        # Validate teardown_proof structure
        if not isinstance(data.get("teardown_proof"), dict):
            fail(path, 0, "teardown_proof must be an object")
        tp = data["teardown_proof"]
        if "method" not in tp or "evidence" not in tp:
            fail(path, 0, "teardown_proof must have 'method' and 'evidence'")
        if not tp["method"] or not tp["evidence"]:
            fail(path, 0, "teardown_proof.method and evidence must be non-empty")
        # teardown_proof.reason enum — read from schema (single source of truth)
        if "reason" in tp:
            _tp_props = (schema.get("properties") or {}).get("teardown_proof", {})
            valid_reasons = set(
                (_tp_props.get("properties") or {}).get("reason", {}).get("enum") or []
            )
            if valid_reasons and tp["reason"] not in valid_reasons:
                fail(
                    path,
                    0,
                    f"teardown_proof.reason must be one of {sorted(valid_reasons)}, "
                    f"got {tp['reason']!r}",
                )

        # Validate decision (non-empty string)
        if not isinstance(data.get("decision"), str) or not data["decision"]:
            fail(path, 0, "decision must be a non-empty string")

        # Validate incident format in the fallback path (strict path enforces via schema pattern)
        if "incident" in data and not INCIDENT_RE.fullmatch(str(data["incident"])):
            fail(
                path,
                0,
                f"incident must match <repo>#<n> or <owner>/<repo>#<n>, "
                f"got {data['incident']!r}",
            )

    # ISO-8601 timestamp validation: run in both paths because Draft 2020-12 does not
    # enforce format: date-time without a format checker, so "yesterday" passes strict.
    for _ts_field in ["started_at", "finished_at"]:
        try:
            datetime.fromisoformat(str(data.get(_ts_field, "")).replace("Z", "+00:00"))
        except ValueError:
            fail(path, 0, f"{_ts_field} must be a valid ISO 8601 timestamp")

    # Business-logic constraint not expressible in the schema: Slack permalink pattern.
    # Enforced in both paths because the schema only constrains minLength: 1.
    if isinstance(data.get("approvals"), list):
        for i, appr in enumerate(data["approvals"]):
            if isinstance(appr, dict) and not PERMALINK_RE.match(
                str(appr.get("slack_permalink", ""))
            ):
                fail(
                    path,
                    0,
                    f"approvals[{i}].slack_permalink is not a #ml-packages permalink: "
                    f"{appr.get('slack_permalink')!r}",
                )

    # Incident receipt constraints (both paths):
    # 1. Existence: the issue must exist on GitHub with label 'incident' or title
    #    '[incident]…'; in CI this is verified via the API; locally NOT VERIFIED is
    #    printed to stderr — never a silent pass.
    # 2. Business logic: status ALARM, result invalid, decision starts with 'abandon'.
    #    Conditional constraints not expressible in Draft 2020-12 without if/then;
    #    enforced here so both the strict and fallback paths agree.
    if "incident" in data:
        _verify_incident_exists(path, str(data["incident"]))
        if data.get("status") != "ALARM":
            fail(
                path,
                0,
                f"an incident receipt requires status ALARM, got {data.get('status')!r}",
            )
        if data.get("result") != "invalid":
            fail(
                path,
                0,
                f"an incident receipt requires result 'invalid', got {data.get('result')!r}",
            )
        if not isinstance(data.get("decision"), str) or not data["decision"].startswith("abandon"):
            fail(
                path,
                0,
                f"an incident receipt requires decision starting with 'abandon', "
                f"got {data.get('decision')!r}",
            )


def select_approver_spec(
    estimated: float,
    thresholds: list[dict[str, Any]],
    overrides: list[dict[str, Any]] | None = None,
    seat_executors: dict[str, str] | None = None,
) -> str | None:
    """The approver spec for this estimate: the first threshold whose max_usd covers it, then any
    approval_overrides entry whose while_role_executor seats all match seat_executors and whose
    band_usd (lo, hi] contains the estimate. Same semantics as experts4bit_qlora.tools.rent."""
    applicable = None
    for th in thresholds:
        max_usd = th["max_usd"]
        if max_usd is None or estimated <= max_usd:
            applicable = th
            break
    if applicable is None:
        return None
    spec = str(applicable["approver"])
    seats = seat_executors or {}
    for ov in overrides or []:
        lo, hi = ov["band_usd"]
        if not (float(lo) < estimated <= float(hi)):
            continue
        cond = ov.get("while_role_executor") or {}
        if not cond:
            continue
        # Fail closed (same rule as the launcher): the override is skipped only when every seat it names
        # is declared in the receipt's environment.seat_executors AND held by someone else.
        declared_elsewhere = all(r in seats and seats[r] != ex for r, ex in cond.items())
        if not declared_elsewhere:
            spec = str(ov["approver"])
    return spec


def seat_executors_from_env(env: dict) -> dict[str, str] | None:
    """`environment.seat_executors` as the launcher writes it: a JSON string (the schema types environment
    values as strings). A dict is accepted for older receipts; anything else is None (= undeclared)."""
    raw = env.get("seat_executors") if isinstance(env, dict) else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    return {str(k): str(v) for k, v in raw.items()}


def check_approval_threshold(
    path: Path,
    estimated: float,
    approvals: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    overrides: list[dict[str, Any]] | None = None,
    seat_executors: dict[str, str] | None = None,
) -> None:
    """Check that approvals satisfy the threshold (and any executor-conditional override) for the estimated cost."""
    approver_spec = select_approver_spec(estimated, thresholds, overrides, seat_executors)
    if approver_spec is None:
        fail(path, 0, f"no threshold found for estimated cost ${estimated}")

    # Self-approval case
    if approver_spec == "requesting-agent":
        # No external approval needed
        return

    # Jordan's approval lifts any spec -- as role AND agent "Jordan" with a valid channel permalink (same as the launcher)
    if any(a.get("role") == "Jordan" and a.get("agent") == "Jordan" and PERMALINK_RE.match(str(a.get("slack_permalink", "")))
           for a in approvals):
        return
    if approver_spec == "Jordan":
        fail(path, 0, f"estimated ${estimated} requires Jordan approval (role and agent 'Jordan'), none found")

    # one-of:CTO,CSO case
    if approver_spec.startswith("one-of:"):
        required_roles = set(approver_spec.split(":", 1)[1].split(","))
        approving_roles = {a.get("role") for a in approvals}
        if not (required_roles & approving_roles):
            fail(
                path,
                0,
                f"estimated ${estimated} requires one of {required_roles}, got {approving_roles}",
            )
        return

    # two-of:CEO,CTO,CSO case
    if approver_spec.startswith("two-of:"):
        required_roles = set(approver_spec.split(":", 1)[1].split(","))
        approving_roles = [a.get("role") for a in approvals if a.get("role") in required_roles]
        if len(set(approving_roles)) < 2:
            fail(
                path,
                0,
                f"estimated ${estimated} requires two of {required_roles}, got {approving_roles}",
            )
        return

    # all-of:CTO,CSO case (executor-conditional override, e.g. while Grok holds the CTO seat)
    if approver_spec.startswith("all-of:"):
        required_roles = set(approver_spec.split(":", 1)[1].split(","))
        approving_roles = {a.get("role") for a in approvals}
        missing = required_roles - approving_roles
        if missing:
            fail(
                path,
                0,
                f"estimated ${estimated} requires all of {sorted(required_roles)} (override), missing {sorted(missing)}",
            )
        return

    fail(path, 0, f"unknown approver spec: {approver_spec}")


def main() -> None:
    # Load policy
    if not POLICY.exists():
        fail(POLICY, None, "policy file not found")
    policy = load_json(POLICY)

    # Collect all receipts
    receipts: list[tuple[Path, dict[str, Any]]] = []
    if RUNS_DIR.exists():
        for receipt_path in RUNS_DIR.rglob("receipt.json"):
            data = load_json(receipt_path)
            validate_receipt(receipt_path, data)
            receipts.append((receipt_path, data))

    # Load ledger
    ledger_entries: dict[str, dict[str, Any]] = {}
    if LEDGER.exists():
        for line_no, line in enumerate(LEDGER.read_text().splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                fail(LEDGER, line_no, f"JSON parse error: {e.msg}")
            if "run_id" not in entry:
                fail(LEDGER, line_no, "ledger entry missing run_id")
            ledger_entries[entry["run_id"]] = entry

    # Check that each receipt appears in the ledger
    for path, data in receipts:
        exp_id = data["experiment_id"]
        # Ledger still uses run_id for backwards compat; we accept experiment_id == run_id
        if exp_id not in ledger_entries:
            fail(path, 0, f"experiment_id {exp_id!r} not found in ledger (as run_id)")

        # Check cost matches
        actual = data["cost_usd"]["actual"]
        ledger_cost = ledger_entries[exp_id].get("cost_usd")
        if ledger_cost != actual:
            fail(
                path,
                0,
                f"cost_usd.actual {actual} != ledger entry {ledger_cost}",
            )

    # Check daily ceilings and budget
    role_daily: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    global_daily: dict[str, float] = defaultdict(float)

    for path, data in receipts:
        # Extract date from started_at
        start = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
        date_str = start.strftime("%Y-%m-%d")
        # Extract role from requested_by string (format: "ROLE/Agent")
        role = data["requested_by"].split("/")[0] if "/" in data["requested_by"] else data["requested_by"]
        cost = data["cost_usd"]["actual"]

        role_daily[role][date_str] += cost
        global_daily[date_str] += cost

    # Check against ceilings
    role_ceilings = policy["role_daily_ceiling_usd"]
    global_budget = policy["global_daily_budget_usd"]

    for role, dates in role_daily.items():
        ceiling = role_ceilings.get(role)
        if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool):
            # Fail closed: a role without a numeric ceiling row must not have rented (the launcher refuses it too)
            for path, data in receipts:
                if (data["requested_by"].split("/")[0] if "/" in data["requested_by"] else data["requested_by"]) == role:
                    fail(path, 0, f"role {role!r} has no numeric row in role_daily_ceiling_usd")
            continue
        for date_str, total in dates.items():
            if total > ceiling:
                # Find a receipt from that role on that date to report
                for path, data in receipts:
                    start = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
                    if (
                        start.strftime("%Y-%m-%d") == date_str
                        and (data["requested_by"].split("/")[0] if "/" in data["requested_by"] else data["requested_by"]) == role
                    ):
                        fail(
                            path,
                            0,
                            f"{role} daily total on {date_str}: ${total:.2f} exceeds ceiling ${ceiling}",
                        )

    for date_str, total in global_daily.items():
        if total > global_budget:
            # Find any receipt from that date to report
            for path, data in receipts:
                start = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
                if start.strftime("%Y-%m-%d") == date_str:
                    fail(
                        path,
                        0,
                        f"global daily total on {date_str}: ${total:.2f} exceeds budget ${global_budget}",
                    )

    # Check approval thresholds.
    # Incident receipts (those carrying incident: "<repo>#<n>") are exempt: the threshold
    # check is replaced by the incident reference itself.  Format and required
    # status/result/decision constraints have already been enforced by validate_receipt.
    # The cost still counts toward daily ceilings and the global budget (handled above).
    thresholds = policy["approval_thresholds"]
    overrides = policy.get("approval_overrides") or []
    for path, data in receipts:
        if "incident" in data:
            continue  # approval threshold replaced by incident reference; validated above
        estimated = data["cost_usd"]["estimated"]
        approvals = data["approvals"]
        env = data.get("environment") if isinstance(data.get("environment"), dict) else {}
        check_approval_threshold(path, estimated, approvals, thresholds, overrides, seat_executors_from_env(env))

    # All checks passed
    num_days = len(global_daily) if global_daily else 0
    print(f"OK: {len(receipts)} receipts, {num_days} days")


if __name__ == "__main__":
    main()
