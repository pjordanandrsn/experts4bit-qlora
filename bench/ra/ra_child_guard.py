"""Linux RA child bootstrap: arm SIGKILL on parent death before target exec.

Called only by ra_process under the selected Python's -I -S -B startup. This
protects cooperating direct children, not arbitrary forked descendants.
"""
from __future__ import annotations

import ctypes
import datetime
import errno
import hashlib
import json
import os
import signal
import stat
import sys
from pathlib import Path


def arm(expected_parent):
    if sys.platform != "linux" or type(expected_parent) is not int or expected_parent < 1:
        raise ValueError("Linux and a positive expected parent PID required")
    before = os.getppid()
    if before != expected_parent:
        raise RuntimeError("expected parent exited before guard setup")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "parent-death signal refused")
    observed = ctypes.c_int()
    if libc.prctl(2, ctypes.cast(ctypes.byref(observed), ctypes.c_void_p).value, 0, 0, 0) != 0:  # PR_GET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "parent-death signal readback refused")
    after = os.getppid()
    if after != expected_parent or observed.value != signal.SIGKILL:
        raise RuntimeError("parent changed or parent-death signal did not engage")
    return {"expected_parent": expected_parent, "parent_before": before, "parent_after": after,
            "signal": observed.value, "pid": os.getpid(), "platform": sys.platform}


def execute(expected_parent, evidence, argv):
    evidence = Path(evidence)
    if not evidence.is_absolute() or any(p.is_symlink() for p in (evidence, *evidence.parents)):
        raise ValueError("absolute owned guard evidence without symlinks required")
    if not argv or not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file() or \
            Path(argv[0]).stat().st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise ValueError("absolute non-privileged target executable required")
    if sys.platform == "linux":
        try:
            os.getxattr(argv[0], "security.capability")
        except OSError as exc:
            if exc.errno != errno.ENODATA:
                raise ValueError("target file capabilities cannot be established absent") from exc
        else:
            raise ValueError("target file capabilities would clear parent-death protection")
    record = {"schema": 1, "status": "STARTING", "exec_argv_sha256":
              hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()}
    # Refuse an old guard receipt rather than overwrite a prior attempt.
    with evidence.open("x") as dst:
        dst.write(json.dumps(record) + "\n")
    try:
        record.update(arm(expected_parent), status="ARMED",
                      clock=datetime.datetime.now(datetime.timezone.utc).isoformat())
        evidence.write_text(json.dumps(record, indent=2) + "\n")
        # Re-check after the receipt write; SIGKILL also closes the later race.
        if os.getppid() != expected_parent:
            raise RuntimeError("parent exited before target exec")
        os.execv(argv[0], argv)
    except BaseException as exc:
        record.update(status="FAILED", error_type=type(exc).__name__)
        evidence.write_text(json.dumps(record, indent=2) + "\n")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[3] != "--":
        raise ValueError("expected parent, evidence and -- target argv required")
    execute(int(sys.argv[1]), Path(sys.argv[2]), sys.argv[4:])
