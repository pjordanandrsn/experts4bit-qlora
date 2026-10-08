"""One fresh process, bounded by the reviewed deadline, with retained failures."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path


def clock():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_digest(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("input must be an absolute regular file without symlink traversal")
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_input(pin):
    if set(pin) != {"path", "sha256"} or file_digest(pin["path"]) != pin["sha256"]:
        raise ValueError("input bytes differ from manifest")
    return Path(pin["path"])


def window(deadline, timeout):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (deadline, timeout)) or timeout <= 0:
        raise ValueError("invalid deadline/timeout")
    if time.time() + timeout + 600 > deadline:
        raise ValueError("phase timeout does not leave ten minutes for retrieval/teardown")


def run(argv, *, env, cwd, log, receipt, deadline, timeout):
    """No shell or retry. Kill only the process group created by this call.

    The private controller must prove ledger/clearance before this executor is
    invoked; a deadline here is not spending authority or teardown evidence.
    """
    window(deadline, timeout)
    if not argv or not Path(argv[0]).is_absolute() or not cwd.is_absolute() or not cwd.is_dir():
        raise ValueError("absolute executable and existing cwd required")
    if not log.is_absolute() or not receipt.is_absolute() or log == receipt:
        raise ValueError("distinct absolute process record paths required")
    record = {"argv": argv, "cwd": str(cwd), "started_at": clock(), "timeout_s": timeout,
              "deadline_epoch_s": deadline, "status": "STARTING", "attempts": 1}
    # Exclusive creation prevents overwriting a failed attempt on rerun.
    with receipt.open("x") as dst:
        dst.write(json.dumps(record, indent=2) + "\n")
    process = None
    try:
        with log.open("xb") as output:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            record["pid"] = process.pid
            try:
                record["returncode"] = process.wait(timeout=timeout)
                record["status"] = "OK" if record["returncode"] == 0 else "PROCESS_FAILED"
            except subprocess.TimeoutExpired:
                record["status"] = "TIMEOUT"
    except Exception as exc:
        record["status"] = "LAUNCH_FAILED"
        record["error_type"] = type(exc).__name__
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            record["returncode"] = process.returncode
        record["finished_at"] = clock()
        if log.is_file():
            record["log_sha256"] = file_digest(log)
        receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if record["status"] != "OK":
        raise RuntimeError("component failed; retained process receipt and log")
    return record
