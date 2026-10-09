"""CPU ordering/mutation/cleanup controls, never GPU or launch evidence."""
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import test_ra_inputs as controls

controls.load("ra_env")
process = controls.load("ra_process")
abba = controls.load("ra_abba")


def pin(path):
    return {"path": str(path), "sha256": process.file_digest(path)}


def encoded(path, obj):
    path.write_text(json.dumps(obj))
    return pin(path)


@pytest.fixture
def plan(tmp_path, abba_stage):
    stage = abba_stage
    spec = controls.fixture(tmp_path / "inputs")
    lock = controls.inputs.inspect(spec, stage)
    p = {"schema": 1, "battery": "proof", "stage": {"path": str(stage),
         "sha256": process.file_digest(stage / "stage-manifest.json")},
         "inputs": {"spec": encoded(tmp_path / "inputs-spec.json", spec),
                    "lock": encoded(tmp_path / "inputs-lock.json", lock)},
         "versions": {}, "deadline_epoch_s": time.time() + 10000}
    for slot in ("old", "new"):
        root = tmp_path / slot
        (root / "bin").mkdir(parents=True)
        (root / "bin/python").write_text("synthetic interpreter bytes; never executed")
        cache = tmp_path / (slot + "-cache")
        cache.mkdir()
        base = {"venv": str(root), "cache": str(cache), "threads": 1, "allocator": "expandable_segments:True"}
        v = {**base, "python": pin(root / "bin/python"), "jobs": {}}
        fixture = {"E4B_PAGED_MODEL": lock["model"], "E4B_PAGED_REVISION": lock["revision"],
                   "E4B_PAGED_ARENA": str(tmp_path / (slot + ".arena")),
                   "E4B_PAGED_CALIB": str(tmp_path / (slot + ".calib"))}
        training = {**base, "battery": "proof", "data": pin(Path(spec["files"]["train_data"])),
                    "tokens": pin(Path(spec["files"]["train_tokens"])), "expect_trainable": 1,
                    "prereg": pin(controls.ROOT / "bench/ra/PREREG-ra.md"),
                    "deadline_epoch_s": p["deadline_epoch_s"], "timeout_s": 10}
        for phase in abba.PHASES:
            env = {}
            if phase.startswith("training"):
                native = copy.deepcopy(training)
            elif phase == "decode":
                native = {"kind": "PROOF", "short": 8, "long": 24, "reps": 1,
                          "prompts": spec["files"]["decode_prompts"]}
                env = dict(fixture)
            elif phase == "capacity":
                env = {**fixture, "E4B_PAGED_MAX_TOKENS_PER_SEQ": "2048", "E4B_PAGED_CHUNK_TOKENS": "512"}
                native = {**base, "battery": "proof", "fixture": env,
                          "prompts": pin(Path(spec["files"]["decode_prompts"])),
                          "deadline_epoch_s": p["deadline_epoch_s"], "startup_s": 1, "driver_s": 1}
            else:
                env = {**fixture, "E4B_PAGED_GRAPHS": "0"}
                native = {"kind": "PROOF", "group": int(phase.split("_")[1]), "cont": 32,
                          "windows": spec["files"]["quality_windows"], "ref_dir": str(root / "unused")}
            v["jobs"][phase] = {"spec": encoded(tmp_path / (slot + "-" + phase + ".json"), native),
                                  "fixture": env, "timeout_s": 30}
        p["versions"][slot] = v
    return p


@pytest.fixture(scope="session")
def abba_stage(tmp_path_factory):
    yield from controls.stage.__wrapped__(tmp_path_factory)


def fake_run(argv, **kw):
    record = {"status": "OK", "cleanup_complete": True, "returncode": 0, "attempts": 1}
    kw["log"].write_text("CPU substituted worker; no native/GPU execution\n")
    kw["receipt"].write_text(json.dumps(record))
    return record


def test_exact_24_process_order_and_owned_quality_refs(plan, tmp_path, monkeypatch):
    seen = []

    def run(argv, **kw):
        seen.append((kw["cwd"].parent.name, kw["cwd"].name, argv, kw["env"]))
        return fake_run(argv, **kw)

    monkeypatch.setattr(abba.ra_process, "run", run)
    out = tmp_path / "run"
    rec = abba.supervise(plan, out, tmp_path / "box.lock")
    assert [(a, b) for a, b, *_ in seen] == [(t, p) for t, _ in abba.POSITIONS for p in abba.PHASES]
    assert rec["status"] == "ORDERED_COMPONENTS_RECORDED_PENDING_GATES"
    assert rec["proves_gpu_engagement"] is rec["release_cleared"] is False
    for tag, phase, argv, env in seen:
        slot = dict(abba.POSITIONS)[tag]
        assert argv[0] == plan["versions"][slot]["python"]["path"]
        assert argv[1] == "-B" and "--instruments" in argv
        assert ("--profile" in argv) == (phase == "training_profile")
        assert "PYTHONPATH" not in env
        if phase.startswith("quality_"):
            materialized = json.loads((out / "raw" / tag / phase / "spec.json").read_text())
            assert materialized["ref_dir"] == str(out / "raw" / tag / phase / "reference")
    with pytest.raises(FileExistsError):
        abba.supervise(plan, out, tmp_path / "box.lock")


@pytest.mark.parametrize("bad", ["missing", "extra", "slots", "battery", "stage", "threads", "venv", "cache"])
def test_plan_mutations_refuse(plan, bad):
    if bad == "missing":
        del plan["versions"]["old"]["jobs"]["decode"]
    elif bad == "extra":
        plan["versions"]["old"]["jobs"]["retry"] = {}
    elif bad == "slots":
        plan["versions"]["third"] = {}
    elif bad == "battery":
        plan["battery"] = "reading"
    elif bad == "stage":
        plan["stage"]["sha256"] = "0" * 64
    elif bad == "threads":
        plan["versions"]["new"]["threads"] = 2
    else:
        plan["versions"]["new"][bad] = plan["versions"]["old"][bad]
    with pytest.raises(ValueError):
        abba.check(plan)


@pytest.mark.parametrize("phase,key,value", [("decode", "short", 9), ("quality_1", "group", 12),
    ("quality_12", "cont", 128), ("training_profile", "expect_trainable", 2),
    ("capacity", "startup_s", 30), ("training", "deadline_epoch_s", 0)])
def test_resealed_spec_battery_mutants_refuse(plan, phase, key, value):
    job = plan["versions"]["old"]["jobs"][phase]
    path = Path(job["spec"]["path"])
    native = json.loads(path.read_text())
    native[key] = value
    job["spec"] = encoded(path, native)
    with pytest.raises(ValueError):
        abba.check(plan)


@pytest.mark.parametrize("fail_at", [0, 5, 6, 12, 23])
def test_failure_stops_without_retry_and_retains_prefix(plan, tmp_path, monkeypatch, fail_at):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        fake_run(argv, **kw)
        if len(calls) == fail_at + 1:
            raise RuntimeError("CPU worker failed")
        return {"status": "OK", "cleanup_complete": True}

    monkeypatch.setattr(abba.ra_process, "run", run)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="CPU worker"):
        abba.supervise(plan, out, tmp_path / "box.lock")
    rec = json.loads((out / "supervisor.json").read_text())
    assert rec["status"] == "FAILED" and len(rec["completed"]) == fail_at
    assert len(calls) == fail_at + 1
    assert (out / "raw" / rec["active"]["tag"] / rec["active"]["phase"] / "process.json").exists()


@pytest.mark.parametrize("cleanup", [False, None])
def test_unknown_or_failed_cleanup_never_advances(plan, tmp_path, monkeypatch, cleanup):
    monkeypatch.setattr(abba.ra_process, "run", lambda *a, **k: {"status": "OK", "cleanup_complete": cleanup})
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        abba.supervise(plan, tmp_path / "run", tmp_path / "box.lock")
    assert json.loads((tmp_path / "run/supervisor.json").read_text())["completed"] == []


def test_runtime_input_drift_stops_after_first_worker(plan, tmp_path, monkeypatch):
    def run(argv, **kw):
        spec = abba.pinned(plan["inputs"]["spec"])
        (Path(spec["trees"]["checkpoint"]) / "bytes.bin").write_bytes(b"mutant")
        return fake_run(argv, **kw)

    monkeypatch.setattr(abba.ra_process, "run", run)
    with pytest.raises(ValueError, match="common inputs"):
        abba.supervise(plan, tmp_path / "run", tmp_path / "box.lock")
    assert json.loads((tmp_path / "run/supervisor.json").read_text())["completed"] == []


def test_shared_lock_refuses_concurrent_ra(plan, tmp_path, monkeypatch):
    path = tmp_path / "box.lock"
    with path.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            abba.supervise(plan, tmp_path / "run", path)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("initial", ["timeout", "success", "launch_failure"])
def test_post_kill_wait_timeout_retains_original_failure(tmp_path, monkeypatch, initial):
    class Child:
        pid = 999999
        returncode = None
        n = 0

        def wait(self, timeout):
            self.n += 1
            if self.n == 1 and initial == "success":
                self.returncode = 0
                return 0
            if self.n == 1 and initial == "launch_failure":
                raise OSError("original worker failure")
            raise subprocess.TimeoutExpired("CPU fixture", timeout)

    monkeypatch.setattr(process.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(process.os, "killpg", lambda *a: None)
    with pytest.raises((RuntimeError, OSError)) as exc:
        process.run([sys.executable, "-c", "pass"], env=os.environ.copy(), cwd=tmp_path,
                    log=tmp_path / "worker.log", receipt=tmp_path / "worker.json",
                    deadline=time.time() + 1000, timeout=1)
    rec = json.loads((tmp_path / "worker.json").read_text())
    assert rec["kill_wait_timeout"] is True and rec["cleanup_complete"] is False
    assert rec["status"] == {"timeout": "TIMEOUT", "success": "CLEANUP_INCOMPLETE",
                             "launch_failure": "LAUNCH_FAILED"}[initial]
    assert not isinstance(exc.value, subprocess.TimeoutExpired)
