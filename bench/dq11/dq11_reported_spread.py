"""Last, optional DQ11 spread processes; deadline skips and failures never VOID science."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import time


def reported_spread(directory, env, arm):
    """Preserve a status and actual cap/duration even if the child is truncated."""
    if arm not in {"L", "U", "U0"}:
        raise ValueError("unknown spread arm")
    deadline = int(env["TC1_DEADLINE_EPOCH"])
    started, monotonic = time.time(), time.monotonic()
    left = deadline - started - 300
    row = {"schema": "dq11-reported-spread-phase/1", "arm": arm, "phase": "spread-" + arm,
           "nonce": env["TC1_RUN_NONCE"], "source": env["E4B_SHA"],
           "science_eligible": env.get("DQ11_REHEARSAL") != "1", "gate": False,
           "started_epoch": started, "deadline_epoch": deadline, "reserve_seconds": 300,
           "requested_cap_seconds": 900, "effective_cap_seconds": min(900, left) if left > 0 else None,
           "time_left_before_reserve_seconds": left, "status": "RUNNING"}
    path = directory / f"receipts/spread-status-{arm}.json"
    path.write_text(json.dumps(row, indent=2) + "\n")
    try:
        if left <= 0:
            row.update(status="spread: skipped for time", reason="deadline reserve")
        else:
            with (directory / f"logs/spread-{arm}.log").open("w") as log:
                subprocess.run(["python", "dq11_arm.py", "--kind", "spread", "--arm", arm,
                                "--repetition", "0", "--out", f"receipts/spread-{arm}.json"],
                               cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=row["effective_cap_seconds"], check=True)
            if not (directory / f"receipts/spread-{arm}.json").is_file():
                raise ValueError("spread child exited without a receipt")
            row["status"] = "complete"
    except (subprocess.SubprocessError, TimeoutError, OSError, ValueError) as error:
        row.update(status="incomplete", error_type=type(error).__name__, reason=str(error))
    finally:
        row.update(elapsed_seconds=time.monotonic() - monotonic, finished_epoch=time.time())
        row["time_left_after_reserve_seconds"] = deadline - row["finished_epoch"] - 300
        path.write_text(json.dumps(row, indent=2) + "\n")
        ledger = directory / "receipts/phase-timings.json"
        if ledger.exists():
            record = json.loads(ledger.read_text())
            record["phases"].append(row)
            ledger.write_text(json.dumps(record, indent=2) + "\n")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--arm", choices=("L", "U", "U0"), required=True)
    args = parser.parse_args()
    print(json.dumps(reported_spread(args.directory, dict(os.environ), args.arm)))
