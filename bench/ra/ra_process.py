"""One fresh process, bounded by the reviewed deadline, with retained failures."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
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


def guard_evidence(record, guard, evidence, argv, pid):
    """Retain guard failures without replacing an earlier process failure."""
    try:
        observation = json.loads(evidence.read_bytes())
        record["parent_death_guard"]["observed"] = observation
        expected_hash = hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
        if file_digest(guard) != record["parent_death_guard"]["script_sha256"] or \
                observation.get("status") != "ARMED" or observation.get("pid") != pid or \
                any(observation.get(k) != os.getpid() for k in ("expected_parent", "parent_before", "parent_after")) or \
                observation.get("signal") != signal.SIGKILL or observation.get("exec_argv_sha256") != expected_hash:
            raise ValueError("owned parent-death guard evidence differs")
        record["parent_death_guard"]["verified"] = True
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        record["parent_death_guard"].update(verified=False, error_type=type(exc).__name__)
        if record["status"] == "OK":
            record["status"] = "GUARD_UNVERIFIED"


def run(argv, *, env, cwd, log, receipt, deadline, timeout):
    """Selected Python worker only; no shell or retry. Kill this owned group.

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
    launch = argv
    guard_path = receipt.with_name(receipt.name + ".guard.json")
    if sys.platform == "linux":
        if threading.current_thread() is not threading.main_thread():
            raise ValueError("guarded launch requires the parent main thread")
        guard = Path(__file__).with_name("ra_child_guard.py")
        record["parent_death_guard"] = {"required": True, "script_sha256": file_digest(guard),
                                        "evidence_path": str(guard_path), "expected_parent": os.getpid()}
        launch = [argv[0], "-I", "-S", "-B", str(guard), str(os.getpid()), str(guard_path), "--", *argv]
    else:
        record["parent_death_guard"] = {"required": False, "reason": "CPU non-Linux control only"}
    # Exclusive creation prevents overwriting a failed attempt on rerun.
    with receipt.open("x") as dst:
        dst.write(json.dumps(record, indent=2) + "\n")
    process = None
    try:
        with log.open("xb") as output:
            process = subprocess.Popen(launch, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
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
        record["cleanup_complete"] = process is None
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                record["cleanup_error_type"] = type(exc).__name__
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                record["kill_wait_timeout"] = True
            except OSError as exc:
                record["cleanup_wait_error_type"] = type(exc).__name__
            record["returncode"] = process.returncode
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                record["cleanup_complete"] = process.returncode is not None and not any(k in record for k in
                    ("kill_wait_timeout", "cleanup_error_type", "cleanup_wait_error_type"))
            except OSError as exc:
                record["cleanup_probe_error_type"] = type(exc).__name__
            if not record["cleanup_complete"] and record["status"] == "OK":
                record["status"] = "CLEANUP_INCOMPLETE"
            if record["parent_death_guard"]["required"]:
                guard_evidence(record, guard, guard_path, argv, process.pid)
        record["finished_at"] = clock()
        if log.is_file():
            record["log_sha256"] = file_digest(log)
        if guard_path.is_file():
            record["guard_sha256"] = file_digest(guard_path)
        receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if record["status"] != "OK":
        raise RuntimeError("component failed; retained process receipt and log")
    return record
