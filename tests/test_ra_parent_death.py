"""CPU guard sensitivity controls; native Linux orphan tests are separate receipts."""
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/ra" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load("ra_child_guard")
process = load("ra_process")


class Prctl:
    def __init__(self, *, set_error=False, get_error=False, signal_value=signal.SIGKILL):
        self.calls = []
        self.set_error, self.get_error, self.signal_value = set_error, get_error, signal_value

    def __call__(self, option, value, *rest):
        self.calls.append(option)
        if option == 1:
            return -1 if self.set_error else 0
        if self.get_error:
            return -1
        ctypes.cast(value, ctypes.POINTER(ctypes.c_int))[0] = self.signal_value
        return 0


def mocked_linux(monkeypatch, parents=(42, 42), **kwargs):
    monkeypatch.setattr(guard.sys, "platform", "linux")
    values = iter(parents)
    monkeypatch.setattr(guard.os, "getppid", lambda: next(values))
    libc = type("Libc", (), {"prctl": Prctl(**kwargs)})()
    monkeypatch.setattr(guard.ctypes, "CDLL", lambda *a, **k: libc)
    return libc


def test_signal_set_and_readback_with_stable_parent(monkeypatch):
    libc = mocked_linux(monkeypatch)
    rec = guard.arm(42)
    assert libc.prctl.calls == [1, 2]
    assert rec["parent_before"] == rec["parent_after"] == rec["expected_parent"] == 42
    assert rec["signal"] == signal.SIGKILL


@pytest.mark.parametrize("parents,options,error", [
    ((41,), {}, RuntimeError),
    ((42,), {"set_error": True}, OSError),
    ((42,), {"get_error": True}, OSError),
    ((42, 41), {}, RuntimeError),
    ((42, 42), {"signal_value": 0}, RuntimeError),
])
def test_signal_and_parent_mutants_refuse(monkeypatch, parents, options, error):
    mocked_linux(monkeypatch, parents, **options)
    with pytest.raises(error):
        guard.arm(42)


@pytest.mark.parametrize("value", [0, -1, True, "42"])
def test_parent_identity_type_refused(value):
    with pytest.raises(ValueError):
        guard.arm(value)


def test_nonlinux_cannot_claim_parent_death(monkeypatch):
    monkeypatch.setattr(guard.sys, "platform", "darwin")
    with pytest.raises(ValueError):
        guard.arm(42)


def test_privileged_exec_refused_before_arm(tmp_path, monkeypatch):
    exe = tmp_path / "python"
    exe.write_bytes(b"synthetic never executed")
    exe.chmod(0o4755)
    monkeypatch.setattr(guard, "arm", lambda p: pytest.fail("must refuse before arming"))
    with pytest.raises(ValueError, match="non-privileged"):
        guard.execute(42, tmp_path / "guard.json", [str(exe)])


@pytest.mark.parametrize("capability", [b"present", "unreadable"])
def test_capability_present_or_unreadable_refused(tmp_path, monkeypatch, capability):
    exe = tmp_path / "python"
    exe.write_bytes(b"synthetic never executed")
    monkeypatch.setattr(guard.sys, "platform", "linux")

    def get(*a):
        if capability == "unreadable":
            raise OSError(errno.EACCES, "fixture")
        return capability

    monkeypatch.setattr(guard.os, "getxattr", get, raising=False)
    with pytest.raises(ValueError, match="capabilit"):
        guard.execute(42, tmp_path / "guard.json", [str(exe)])


def test_exec_preserves_original_argv_and_exclusive_evidence(tmp_path, monkeypatch):
    exe = tmp_path / "python"
    exe.write_bytes(b"synthetic never executed")
    mocked_linux(monkeypatch, (42, 42, 42))
    monkeypatch.setattr(guard.os, "getxattr",
                        lambda *a: (_ for _ in ()).throw(OSError(errno.ENODATA, "absent")), raising=False)
    calls = []
    monkeypatch.setattr(guard.os, "execv", lambda *a: calls.append(a))
    argv = [str(exe), "-B", "worker.py", "space bearing argument"]
    receipt = tmp_path / "guard.json"
    guard.execute(42, receipt, argv)
    rec = json.loads(receipt.read_bytes())
    assert rec["status"] == "ARMED" and calls == [(str(exe), argv)]
    assert rec["exec_argv_sha256"] == hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
    before = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        guard.execute(42, receipt, argv)
    assert receipt.read_bytes() == before


@pytest.mark.parametrize("bad", ["missing", "signal", "pid", "parent", "argv", "status", "script"])
def test_resealed_guard_evidence_cannot_pass(tmp_path, bad):
    script = tmp_path / "guard.py"
    script.write_text("fixture")
    argv = [sys.executable, "-c", "pass"]
    rec = {"status": "OK", "parent_death_guard": {"script_sha256": process.file_digest(script)}}
    obs = {"status": "ARMED", "pid": 123, "expected_parent": os.getpid(), "parent_before": os.getpid(),
           "parent_after": os.getpid(), "signal": signal.SIGKILL,
           "exec_argv_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()}
    evidence = tmp_path / "guard.json"
    if bad == "script":
        script.write_text("changed")
    elif bad == "parent":
        obs["parent_after"] += 1
    elif bad != "missing":
        obs[{"signal": "signal", "pid": "pid", "argv": "exec_argv_sha256", "status": "status"}[bad]] = None
    if bad != "missing":
        evidence.write_text(json.dumps(obs))
    process.guard_evidence(rec, script, evidence, argv, 123)
    assert rec["status"] == "GUARD_UNVERIFIED" and rec["parent_death_guard"]["verified"] is False


@pytest.mark.parametrize("outcome", ["success", "timeout", "nonzero", "launch"])
def test_missing_guard_retained_without_masking_original_failure(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(process.sys, "platform", "linux")

    class Child:
        pid = 999999
        returncode = None
        n = 0

        def wait(self, timeout):
            self.n += 1
            if self.n == 1:
                if outcome == "timeout":
                    raise subprocess.TimeoutExpired("fixture", timeout)
                if outcome == "launch":
                    raise OSError("original failure")
                self.returncode = 0 if outcome == "success" else 7
            else:
                self.returncode = -9 if self.returncode is None else self.returncode
            return self.returncode

    seen = []
    monkeypatch.setattr(process.subprocess, "Popen", lambda argv, **kw: seen.append(argv) or Child())
    monkeypatch.setattr(process.os, "killpg",
                        lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    receipt = tmp_path / "process.json"
    with pytest.raises((RuntimeError, OSError)):
        process.run([sys.executable, "-B", "worker.py"], env={}, cwd=tmp_path,
                    log=tmp_path / "worker.log", receipt=receipt, deadline=time.time() + 1000, timeout=1)
    rec = json.loads(receipt.read_bytes())
    assert rec["status"] == {"success": "GUARD_UNVERIFIED", "timeout": "TIMEOUT",
                             "nonzero": "PROCESS_FAILED", "launch": "LAUNCH_FAILED"}[outcome]
    assert rec["parent_death_guard"]["verified"] is False and rec["cleanup_complete"] is True
    assert seen[0][1:4] == ["-I", "-S", "-B"]
    assert seen[0][8:] == [sys.executable, "-B", "worker.py"]


def test_parent_thread_refused_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(process.sys, "platform", "linux")
    monkeypatch.setattr(process.threading, "current_thread", lambda: object())
    with pytest.raises(ValueError, match="main thread"):
        process.run([sys.executable, "-c", "pass"], env={}, cwd=tmp_path,
                    log=tmp_path / "worker.log", receipt=tmp_path / "process.json",
                    deadline=time.time() + 1000, timeout=1)
    assert not (tmp_path / "process.json").exists()
