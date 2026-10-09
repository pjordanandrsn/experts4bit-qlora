# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The smoke controller must fail loudly; GPU evidence is a separate real run."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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
    assert calls == list(smoke.CELLS)
    assert len(summary["cells"]) == 10
    failures = [r for r in summary["cells"] if r["status"] == "FAIL"]
    assert len(failures) == 1 and failures[0]["exception"]


def test_worker_preserves_the_real_exception(tmp_path, monkeypatch):
    def fail(*args):
        raise TypeError("_HybridTier.forward() got an unexpected keyword argument 'residual'")
    monkeypatch.setattr(smoke, "run_cell", fail)
    monkeypatch.setattr(smoke, "source_identity", lambda: {})
    args = type("Args", (), dict(family="qwen3_moe", stack="default", output_dir=str(tmp_path),
                                 result=str(tmp_path / "result.json"), int4_source="offline-hub"))()
    assert smoke.worker(args) == 1
    row = json.loads((tmp_path / "result.json").read_text())
    assert row["status"] == "FAIL" and "unexpected keyword argument 'residual'" in row["exception"]
    assert "TypeError" in row["traceback"]


def test_partial_or_failed_residual_licence_cannot_pass():
    good = {"moe_residual": {"licensed": 2, "partial": 0, "probe_errors": []},
            "fuse_t1_glue_n": 9, "fuse_t1_glue_r2_n": [2, 2]}
    smoke.check_residual(good)
    for residual in ({"licensed": 0, "partial": 0, "probe_errors": []},
                     {"licensed": 2, "partial": 1, "probe_errors": []},
                     {"licensed": 2, "partial": 0, "probe_errors": ["TypeError: residual"]}):
        with pytest.raises(RuntimeError, match="not fully licensed"):
            smoke.check_residual(dict(good, moe_residual=residual))
    with pytest.raises(RuntimeError, match="did not engage"):
        smoke.check_residual(dict(good, fuse_t1_glue_r2_n=[0, 0]))


def test_two_expert_targets_cannot_include_mtp_or_duplicate_text_layer():
    a, b = object(), object()
    for names in (("model.layers.0.mlp.experts.base", "mtp.layers.0.mlp.experts.base"),
                  ("model.layers.0.mlp.experts.base", "model.layers.0.block_sparse_moe.experts.base")):
        model = SimpleNamespace(named_modules=lambda: zip(names, (a, b)))
        with pytest.raises(RuntimeError, match="text-tower"):
            smoke.check_text_targets(model, [a, b])
    names = ("model.layers.0.mlp.experts.base", "model.layers.1.mlp.experts.base")
    assert smoke.check_text_targets(SimpleNamespace(named_modules=lambda: zip(names, (a, b))), [a, b]) == list(names)


def test_kernel_digest_tracks_imported_source_and_native_c_without_metadata(tmp_path, monkeypatch):
    kernels = tmp_path / "override"
    native = kernels / "gnf4_native"
    native.mkdir(parents=True)
    module = kernels / "nf4_grouped.py"
    module.write_text("imported_source = 43\n")
    c = native / "kernel.c"
    c.write_text("actual native source\n")
    monkeypatch.setitem(sys.modules, "nf4_grouped", SimpleNamespace(__file__=str(module)))
    first = smoke.imported_kernel_sources({kernels}, native)
    assert first["modules"]["nf4_grouped"]["path"] == str(module)
    assert first["native_c_sources"]["kernel.c"]
    module.write_text("imported_source = 44\n")
    second = smoke.imported_kernel_sources({kernels}, native)
    assert second["imported_source_sha256"] != first["imported_source_sha256"]
    c.write_text("changed native source\n")
    assert smoke.imported_kernel_sources({kernels}, native)["imported_source_sha256"] != second["imported_source_sha256"]


def test_local_int4_option_reaches_every_isolated_worker(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[cmd.index("--int4-source") + 1])
        Path(cmd[cmd.index("--result") + 1]).write_text(json.dumps({"status": "PASS"}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(smoke.subprocess, "run", run)
    assert smoke.run_suite(tmp_path, 300, "local") == 0
    assert calls == ["local"] * len(smoke.CELLS)
