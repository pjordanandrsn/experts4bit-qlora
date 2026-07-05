"""Controller-side provenance gate: classify each job dir before its result may back a doc claim.

A result without sufficient provenance can still be useful for debugging, but it must not support
a documented claim. This validator reads job-local files ONLY (never mutates the run tree), checks
each job against the provenance contract (docs/provenance_contract.md), and classifies it:

  claim_usable  — complete, self-consistent, fully attributed: may back a doc claim
  debug_only    — ran and produced parseable output but a provenance check failed (e.g. missing
                  commit / GPU / versions, or a manifest-vs-result mode/offload/seed mismatch)
  invalid       — status/result present but unparseable or internally contradictory
  missing       — claimed (lock exists) or listed in the manifest but no result.json

Filename reconciliation (contract §"as-built"): the runner writes ``env.json`` (environment
record), ``command.sh`` + the shared manifest line (manifest-of-record), ``status.json``,
``result.json``, and payload files (``result_rows.jsonl`` / ``profile.jsonl`` / ``adapter*``).
The idealized names in the plan (manifest.json, environment.json, metrics.jsonl) map onto these;
this validator checks the as-built set and says so.

Usage:
    python scripts/validate_job_provenance.py --manifest runs/job_manifest/olmoe_repeat_jobs.jsonl \\
      --jobs-root /workspace/runs/jobs [--locks-root /workspace/runs/locks] \\
      --out runs/results/provenance_report.json
Exit code is nonzero if any manifest job is missing or invalid (claim_usable + debug_only are OK).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mode_matrix_common import read_jsonl  # noqa: E402

# result fields that must be present and non-null for a claim (attribution + environment).
_REQUIRED_RESULT = ("job_id", "status", "commit", "pod_id", "hostname",
                    "torch_version", "started_at", "finished_at")
# fields whose absence downgrades to debug_only rather than invalid (environment completeness).
_ENV_FIELDS = ("gpu_name", "gpu", "cuda_version", "bitsandbytes_version")
# manifest params that must match the result when both are present (mode/offload/seed integrity).
_INTEGRITY = ("storage_mode", "offload", "seed", "query_storage_mode", "query_offload")


def _load(path):
    try:
        return json.load(open(path)), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as e:
        return None, f"unparseable: {type(e).__name__}: {e}"


def classify(job_id, job_dir, manifest_params):
    """Return (classification, reasons[]). manifest_params: the manifest's params for this job
    (or {} if not in the manifest — a job dir present without a manifest entry is still checked)."""
    reasons = []
    if not os.path.isdir(job_dir):
        return "missing", ["no job directory"]

    result, err = _load(os.path.join(job_dir, "result.json"))
    status, serr = _load(os.path.join(job_dir, "status.json"))
    if result is None:
        # ran-but-no-result vs never-ran: a status/run.log presence means debug_only, else missing.
        if os.path.exists(os.path.join(job_dir, "status.json")) or os.path.exists(os.path.join(job_dir, "run.log")):
            return "debug_only", [f"result.json {err}; job dir exists (ran or partially ran)"]
        return "missing", [f"result.json {err}"]

    # Internal consistency: job_id must match the directory + the result.
    if result.get("job_id") != job_id:
        reasons.append(f"result job_id {result.get('job_id')!r} != dir {job_id!r}")
        return "invalid", reasons
    if status is not None and status.get("status") and result.get("status") \
            and status["status"] != result["status"]:
        reasons.append(f"status.json {status['status']!r} != result {result['status']!r}")

    if result.get("status") != "pass":
        return "debug_only", [f"status={result.get('status')!r}: {result.get('fail_or_skip_reason')}"]

    # Required attribution fields.
    for k in _REQUIRED_RESULT:
        if not result.get(k):
            reasons.append(f"missing {k}")
    # Environment completeness (at least one GPU-name field + versions).
    if not (result.get("gpu_name") or result.get("gpu")):
        reasons.append("no GPU name")
    if not result.get("cuda_version"):
        reasons.append("no cuda_version")
    if not result.get("bitsandbytes_version"):
        reasons.append("no bitsandbytes_version")
    # env.json presence (the environment record).
    if not os.path.exists(os.path.join(job_dir, "env.json")):
        reasons.append("no env.json")
    # command.sh presence (manifest-of-record for what actually ran).
    if not os.path.exists(os.path.join(job_dir, "command.sh")):
        reasons.append("no command.sh")

    # Manifest-vs-result integrity: mode/offload/seed must agree where both declare them.
    for k in _INTEGRITY:
        if k in manifest_params and k in result and manifest_params[k] != result[k]:
            reasons.append(f"manifest {k}={manifest_params[k]!r} != result {result[k]!r}")

    return ("claim_usable" if not reasons else "debug_only"), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--out", default=None, help="write the full JSON report here")
    args = ap.parse_args()

    manifest = {j["job_id"]: j.get("params", {}) for j in read_jsonl(args.manifest)}
    report = {}
    counts = {"claim_usable": 0, "debug_only": 0, "invalid": 0, "missing": 0}

    # Every manifest job (may be missing) + any extra job dirs present (unexpected residue).
    job_ids = list(manifest)
    if os.path.isdir(args.jobs_root):
        for name in sorted(os.listdir(args.jobs_root)):
            if name not in manifest and os.path.isdir(os.path.join(args.jobs_root, name)):
                job_ids.append(name)

    for job_id in job_ids:
        cls, reasons = classify(job_id, os.path.join(args.jobs_root, job_id), manifest.get(job_id, {}))
        report[job_id] = {"classification": cls, "reasons": reasons, "in_manifest": job_id in manifest}
        counts[cls] += 1

    print(f"provenance report: {counts}")
    for job_id, r in sorted(report.items()):
        tag = "" if r["in_manifest"] else " (not in manifest)"
        line = f"  [{r['classification']}] {job_id}{tag}"
        if r["reasons"]:
            line += " — " + "; ".join(r["reasons"])
        print(line)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"counts": counts, "jobs": report}, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.out}")

    # Nonzero exit if anything is missing or invalid — claim_usable/debug_only are acceptable states.
    return 1 if (counts["missing"] or counts["invalid"]) else 0


if __name__ == "__main__":
    sys.exit(main())
