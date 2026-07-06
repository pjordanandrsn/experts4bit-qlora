"""Rev2 manifest for the five certificate trios (post-fix rerun, cites the rev1 red).

Same five configs as make_post_audit_manifest.py's cert jobs; new `_rev2` job ids so the
failed rev1 locks/artifacts stay untouched as the pre-fix record (Q4 lineage:
runs/results/postaudit_cert_rev1_FAILED.md -> the _sha fix commit -> this rerun).

Usage:
    python scripts/make_cert_rev2_manifest.py --out runs/job_manifest/post_audit_cert_rev2_jobs.jsonl
"""

import argparse
import json
import sys

from make_post_audit_manifest import cert_job


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [
        cert_job("bf16", 0.0, False),
        cert_job("bf16", 0.0, True),
        cert_job("bf16", 0.1, False),
        cert_job("int8", 0.0, False),
        cert_job("int8", 0.0, True),
    ]
    for j in jobs:
        j["params"]["supersedes_failed_job"] = j["job_id"]
        j["job_id"] = j["job_id"] + "_rev2"
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    for j in jobs:
        print(f"  {j['job_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
