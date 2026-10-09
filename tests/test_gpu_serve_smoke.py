# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The smoke controller must fail loudly; GPU evidence is a separate real run."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    "gpu_serve_smoke", Path(__file__).resolve().parents[1] / "bench/smoke/gpu_serve_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_supported_graphs_cannot_silently_fall_back():
    for enabled, status, stats, error in (
        (False, None, None, "resolved to eager"),
        (True, {1: "eager: CUDA error"}, {1: {"replays": 1}}, "failed to capture"),
        (True, {1: "graph"}, {1: {"replays": 0}}, "never replayed"),
    ):
        with pytest.raises(RuntimeError, match=error):
            smoke.check_graphs((8, 9), enabled, status, stats)
    smoke.check_graphs((8, 6), False, None, None)
    smoke.check_graphs((12, 0), True, {1: "graph"}, {1: {"replays": 3}})


@pytest.mark.parametrize("fault", ("family_fail", "missing_result", "bad_exit", "timeout"))
def test_one_bad_worker_fails_the_suite_and_other_cells_still_report(tmp_path, monkeypatch, fault):
    calls = []

    def run(cmd, **kwargs):
        family = cmd[cmd.index("--family") + 1]
        stack = cmd[cmd.index("--stack") + 1]
        calls.append((family, stack))
        bad = (family, stack) == ("mixtral", "default")
        if bad and fault == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        if not (bad and fault == "missing_result"):
            row = {"family": family, "stack": stack, "status": "FAIL" if bad and fault == "family_fail" else "PASS"}
            if row["status"] == "FAIL":
                row["exception"] = "TypeError: residency residual"
            Path(cmd[cmd.index("--result") + 1]).write_text(json.dumps(row))
        return subprocess.CompletedProcess(cmd, 1 if bad else 0)

    monkeypatch.setattr(smoke.subprocess, "run", run)
    assert smoke.run_suite(tmp_path, 300) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert calls == [(f, s) for f in smoke.FAMILIES for s in smoke.STACKS]
    assert len(summary["cells"]) == 8
    failures = [r for r in summary["cells"] if r["status"] == "FAIL"]
    assert len(failures) == 1 and failures[0]["exception"]


def test_worker_preserves_the_real_exception(tmp_path, monkeypatch):
    def fail(*args):
        raise TypeError("_HybridTier.forward() got an unexpected keyword argument 'residual'")
    monkeypatch.setattr(smoke, "run_cell", fail)
    monkeypatch.setattr(smoke, "source_identity", lambda: {})
    args = type("Args", (), dict(family="qwen3_moe", stack="default", output_dir=str(tmp_path),
                                 result=str(tmp_path / "result.json")))()
    assert smoke.worker(args) == 1
    row = json.loads((tmp_path / "result.json").read_text())
    assert row["status"] == "FAIL" and "unexpected keyword argument 'residual'" in row["exception"]
    assert "TypeError" in row["traceback"]
