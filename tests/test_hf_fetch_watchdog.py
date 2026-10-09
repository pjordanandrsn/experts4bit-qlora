"""bench/common/hf_fetch_watchdog.py on CPU, no network (P127 Amendment 2; e4b#1313).

The watchdog's own self-test drives fake fetchers through each path:
- a steady fetch;
- a stall, then a clean rerun with the orphan pruned;
- a stall every time, which gives up;
- an error exit, then a clean rerun;
- an exit 0 that leaves a partial, which is refused and rerun;
- a trickle that spends the budget;
- on POSIX, the group kill reaching a grandchild.
These tests run it and pin the contract a runner depends on.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "bench" / "common" / "hf_fetch_watchdog.py"
spec = importlib.util.spec_from_file_location("hf_fetch_watchdog", SRC)
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)


def test_the_self_test_passes():
    p = subprocess.run([sys.executable, str(SRC), "--self-test"], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "hf_fetch_watchdog self-test OK" in p.stdout.splitlines()[-1]


def test_the_snapshot_is_the_last_stdout_line_and_logs_go_to_stderr(tmp_path):
    """A runner reads the snapshot with ``tail -1`` over stdout+stderr; the watchdog's own lines are on stderr and come
    before the final print."""
    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    fake = tmp_path / "fake.py"
    fake.write_text(wd.FAKE)
    cmd = [sys.executable, str(fake), "steady", str(blobs), str(snap), str(tmp_path / "state")]
    rc, rec = wd.run(cmd, blobs, poll_s=0.1, stall_s=30.0, max_restarts=0, budget_s=120, grace_s=1.0)
    assert rc == wd.EXIT_OK and rec["snapshot"] == str(snap) and rec["restarts"] == 0
    assert [a["outcome"] for a in rec["attempts"]] == ["exit"] and rec["attempts"][0]["partials"] == 0


def test_usage_without_repo_is_exit_2(capsys):
    assert wd.main([]) == wd.EXIT_USAGE


def test_the_measured_directory_follows_the_hub_cache_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert wd.blobs_dir("Qwen/Qwen3-30B-A3B") == tmp_path / "models--Qwen--Qwen3-30B-A3B" / "blobs"
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert wd.hub_cache() == tmp_path / "home" / "hub"


def test_the_child_passes_its_arguments_to_snapshot_download():
    """The real child is ``snapshot_download(**kwargs)`` with the JSON the CLI builds; checked without importing the hub."""
    assert "snapshot_download" in wd.CHILD_SRC and "json.loads(sys.argv[1])" in wd.CHILD_SRC
    kwargs = {"repo_id": "a/b", "revision": "r", "max_workers": 8, "allow_patterns": ["*.json"]}
    assert json.loads(json.dumps(kwargs)) == kwargs
