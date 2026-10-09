# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CPU checks that missing execution and false-positive controls fail admission."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("gpu_path_smoke", Path(__file__).resolve().parents[1] / "bench/smoke/gpu_path_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def probe(cell="hybrid_backward", mutation="none", error=None, hits=0):
    nodes = list(smoke.TESTS[cell])
    return SimpleNamespace(cell=cell, mutation=mutation, collected=nodes, collection_errors=[],
                           stats={"mutation_hits": hits}, engagement_error=lambda: error,
                           reports=[{"nodeid": n, "when": "call", "outcome": "passed", "detail": None} for n in nodes])


@pytest.mark.parametrize("fault", ("skip", "missing_body", "missing_collection", "setup_failure", "unengaged"))
def test_baseline_cannot_pass_without_real_test_body_and_path(fault):
    p = probe()
    if fault == "skip":
        p.reports[0].update(outcome="skipped", when="setup")
    elif fault == "missing_body":
        p.reports = []
    elif fault == "missing_collection":
        p.collected = []
    elif fault == "setup_failure":
        p.reports[0].update(outcome="failed", when="setup", detail="ImportError")
    else:
        p.engagement_error = lambda: "three tiers did not engage"
    with pytest.raises(RuntimeError):
        smoke.assess(p, 0)


def test_numerical_mutation_needs_one_called_assertion_failure():
    p = probe(mutation="wrong_dgrad", hits=1)
    with pytest.raises(RuntimeError, match="exactly one numerical assertion"):
        smoke.assess(p, 0)                    # mutation left the numerical contract passing
    p.reports[0].update(outcome="failed", detail="RuntimeError: backend unavailable")
    with pytest.raises(RuntimeError, match="numerical assertion"):
        smoke.assess(p, 1)                    # an unrelated failure is not a positive control
    p.reports[0]["detail"] = "AssertionError: gradients differ"
    assert smoke.assess(p, 1)["verdict"] == "MUTATION_DETECTED"
    p.stats["mutation_hits"] = 0
    with pytest.raises(RuntimeError, match="never called"):
        smoke.assess(p, 1)


def test_checkpoint_bypass_must_fail_the_host_path_while_gradients_still_pass():
    p = probe(cell="checkpoint_offload", mutation="bypass_host_save", hits=1)
    with pytest.raises(RuntimeError, match="isolate"):
        smoke.assess(p, 0)
    p.engagement_error = lambda: "checkpoint pinned host input save/restore did not engage"
    result = smoke.assess(p, 0)
    assert result["observed"] == "FAIL" and result["verdict"] == "MUTATION_DETECTED"
    p.reports[0].update(outcome="failed", detail="AssertionError: unrelated gradient mismatch")
    with pytest.raises(RuntimeError, match="isolate"):
        smoke.assess(p, 1)


@pytest.mark.parametrize("fault", ("timeout", "missing", "exit", "skip_pass", "wrong_cell", "corrupt"))
def test_one_bad_worker_fails_and_all_baselines_and_controls_still_report(tmp_path, monkeypatch, fault):
    seen = []

    def run(cmd, **kwargs):
        cell, mutation = cmd[cmd.index("--cell") + 1], cmd[cmd.index("--mutation") + 1]
        seen.append((cell, mutation))
        bad = len(seen) == 1
        if bad and fault == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        path = Path(cmd[cmd.index("--result") + 1])
        expected = "PASS" if mutation == "none" else "FAIL"
        row = dict(cell=cell, mutation=mutation, expected=expected, observed=expected, status="PASS")
        if bad and fault == "skip_pass":
            row["observed"] = "SKIP"
        if bad and fault == "wrong_cell":
            row["cell"] = "another path"
        if not (bad and fault == "missing"):
            path.write_text("not JSON" if bad and fault == "corrupt" else json.dumps(row))
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert not any(k.startswith("E4B_") for k in kwargs["env"])
        return subprocess.CompletedProcess(cmd, 1 if bad and fault == "exit" else 0)

    monkeypatch.setenv("E4B_CKPT_OFFLOAD", "0")
    monkeypatch.setattr(smoke.subprocess, "run", run)
    assert smoke.run_suite(tmp_path, 300) == 1
    j = json.loads((tmp_path / "summary.json").read_text())
    assert seen == list(smoke.CASES) and len(j["cells"]) == 7
    assert sum(r["status"] == "FAIL" for r in j["cells"]) == 1


def test_missing_cuda_fails_worker_with_the_real_exception(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(smoke, "cuda_environment", fail)
    monkeypatch.setattr(smoke, "source_identity", lambda: {"source": "CPU unit stub"})
    args = SimpleNamespace(cell="hybrid_backward", mutation="none", result=str(tmp_path / "result.json"))
    assert smoke.worker(args) == 1
    row = json.loads(Path(args.result).read_text())
    assert row["status"] == "FAIL" and "RuntimeError: CUDA unavailable" in row["exception"]
    assert "CUDA unavailable" in row["traceback"]


def test_real_path_probe_requires_pinned_save_and_restore():
    p = smoke.PathProbe("checkpoint_offload", "none")
    p.stats.update(pinned_cuda_inputs=0, restored_cuda_inputs=0)
    assert p.engagement_error()
    p.stats["pinned_cuda_inputs"] = 1
    assert p.engagement_error()
    p.stats["restored_cuda_inputs"] = 1
    assert p.engagement_error() is None


def test_all_three_real_backward_buses_are_required():
    p = smoke.PathProbe("hybrid_backward", "none")
    p.stats.update(backward_calls=2, backward_buses=["hot", "dram"])
    assert p.engagement_error()
    p.stats["backward_buses"].append("cold")
    assert p.engagement_error() is None
