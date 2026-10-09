# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Three tiny CUDA correctness cells, each with a planted path mutation.

Use the existing tests' numerical oracles and unchanged tolerances. A missing
backend, skipped case, absent path engagement, or undetected mutation fails.
Each baseline/control runs offline in its own process; no package file is edited.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TESTS = {
    "hybrid_backward": ("tests/test_hybrid_train.py::test_qlora_adapter_grads_match_reference_across_tiers",),
    "checkpoint_offload": ("tests/test_ckpt_offload.py::test_fused_experts_under_the_offloaded_checkpoint",),
    "alternate_epilogues": (
        "tests/test_hot_residency_gptoss.py::test_gptoss_mixed_split_and_bias_matters",
        "tests/test_hot_residency_v4.py::test_v4_clamps_are_live_in_the_fused_path"),
}
MUTATIONS = {"hybrid_backward": ("wrong_dgrad",), "checkpoint_offload": ("bypass_host_save",),
             "alternate_epilogues": ("drop_bias", "drop_clamp")}
CASES = tuple((cell, mutation) for cell in TESTS for mutation in ("none", *MUTATIONS[cell]))


def source_identity():
    spec = importlib.util.spec_from_file_location("serve_smoke_identity", ROOT / "bench/smoke/gpu_serve_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity = module.source_identity()
    files = [Path(__file__), *(ROOT / n.split("::")[0] for nodes in TESTS.values() for n in nodes)]
    identity["instrument_files_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return identity


def cuda_environment():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: this correctness smoke cannot pass by skipping")
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        raise RuntimeError(f"bf16 grouped kernels require sm80+, found {capability}")
    return {"gpu": torch.cuda.get_device_name(), "capability": list(capability), "torch": torch.__version__,
            "graphs_exercised": False, "sm89_padded_decode": "hardware-blocked on sm86; outside these cells"}


class PathProbe:
    """Observe real calls, and plant one narrowly scoped fault inside a worker."""
    def __init__(self, cell, mutation):
        self.cell, self.mutation = cell, mutation
        self.stack = ExitStack()
        self.reports = []
        self.collected = []
        self.collection_errors = []
        self.stats = {"mutation_hits": 0}

    def __enter__(self):
        if self.cell == "hybrid_backward":
            from experts4bit_qlora.engines import hybrid_train
            original = hybrid_train._HybridProjFn.backward
            self.stats.update(backward_calls=0, backward_buses=[])

            def backward(ctx, grad):
                self.stats["backward_calls"] += 1
                self.stats["backward_buses"].extend(b["bus"] for b in ctx.plan.buses)
                gx, *rest = original(ctx, grad)
                if self.mutation == "wrong_dgrad":
                    self.stats["mutation_hits"] += 1
                    gx = gx + 1                    # corrupt the frozen projection's input gradient
                return (gx, *rest)
            self.stack.enter_context(patch.object(hybrid_train._HybridProjFn, "backward", staticmethod(backward)))
        elif self.cell == "checkpoint_offload":
            import torch
            from experts4bit_qlora.engines import ckpt_offload
            original = torch.autograd.graph.save_on_cpu
            self.stats.update(pinned_cuda_inputs=0, restored_cuda_inputs=0)
            stats = self.stats

            class SavedInputs(original):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    pack, unpack = self.pack_hook, self.unpack_hook

                    def counted_pack(tensor):
                        result = pack(tensor)
                        if tensor.is_cuda and tuple(tensor.shape) == (64, 256):
                            if result[1].device.type != "cpu" or not result[1].is_pinned():
                                raise AssertionError("checkpoint input was not saved in pinned host memory")
                            stats["pinned_cuda_inputs"] += 1
                        return result

                    def counted_unpack(saved):
                        result = unpack(saved)
                        if saved[0].type == "cuda" and tuple(saved[1].shape) == (64, 256):
                            if not result.is_cuda:
                                raise AssertionError("checkpoint input was not restored to CUDA")
                            stats["restored_cuda_inputs"] += 1
                        return result
                    self.pack_hook, self.unpack_hook = counted_pack, counted_unpack
            self.stack.enter_context(patch.object(torch.autograd.graph, "save_on_cpu", SavedInputs))
            if self.mutation == "bypass_host_save":
                def bypass(fn, *args, **kwargs):
                    self.stats["mutation_hits"] += 1
                    return ckpt_offload.reentrant_checkpoint(fn, *args, **kwargs)
                self.stack.enter_context(patch.object(ckpt_offload, "offloaded_checkpoint", bypass))
        elif self.cell == "alternate_epilogues":
            from experts4bit_qlora.engines import hot_residency
            original = hot_residency._fused_over_stack
            signature = inspect.signature(original)
            self.stats.update(gptoss_calls=0, v4_calls=0)

            def fused(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                if bound.arguments["gptoss"] is not None:
                    self.stats["gptoss_calls"] += 1
                    if self.mutation == "drop_bias":
                        gu, dn, alpha, limit = bound.arguments["gptoss"]
                        bound.arguments["gptoss"] = (gu * 0, dn * 0, alpha, limit)
                        self.stats["mutation_hits"] += 1
                if bound.arguments["clamp_limit"] is not None:
                    self.stats["v4_calls"] += 1
                    if self.mutation == "drop_clamp":
                        bound.arguments["clamp_limit"] = None
                        self.stats["mutation_hits"] += 1
                return original(*bound.args, **bound.kwargs)
            self.stack.enter_context(patch.object(hot_residency, "_fused_over_stack", fused))
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection_errors.append({"nodeid": report.nodeid, "outcome": report.outcome, "detail": str(report.longrepr)})

    def pytest_runtest_logreport(self, report):
        self.reports.append({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
                             "detail": str(report.longrepr) if report.longrepr else None})

    def engagement_error(self):
        if self.cell == "hybrid_backward":
            if self.stats["backward_calls"] < 2 or set(self.stats["backward_buses"]) != {"hot", "dram", "cold"}:
                return "both projection backward calls did not exercise all three tiers"
        elif self.cell == "checkpoint_offload":
            if not self.stats["pinned_cuda_inputs"] or not self.stats["restored_cuda_inputs"]:
                return "checkpoint pinned host input save/restore did not engage"
        elif not self.stats["gptoss_calls"] or not self.stats["v4_calls"]:
            return "both alternate-family fused epilogues did not engage"
        return None


def assess(probe, exit_code):
    """Only a called, failing contract can license a mutation control."""
    expected = list(TESTS[probe.cell])
    if sorted(probe.collected) != sorted(expected) or probe.collection_errors:
        raise RuntimeError("selected tests did not all collect: " + repr(probe.collection_errors))
    if any(r["outcome"] == "skipped" or (r["outcome"] == "failed" and r["when"] != "call") for r in probe.reports):
        raise RuntimeError("a selected test skipped or failed outside its test body")
    calls = [r for r in probe.reports if r["when"] == "call"]
    if sorted(r["nodeid"] for r in calls) != sorted(expected):
        raise RuntimeError("not every selected test body executed")
    errors = [r for r in calls if r["outcome"] == "failed"]
    engagement = probe.engagement_error()
    if probe.mutation == "none":
        if exit_code != 0 or errors or engagement:
            raise RuntimeError(engagement or f"baseline pytest failed (exit {exit_code})")
        return {"observed": "PASS", "verdict": "PATH_PASS"}
    if not probe.stats["mutation_hits"]:
        raise RuntimeError("planted mutation was never called")
    if probe.mutation == "bypass_host_save":
        # The gradients intentionally still agree. The pinned-host path must fail its separate engagement check.
        if exit_code != 0 or errors or engagement != "checkpoint pinned host input save/restore did not engage":
            raise RuntimeError("checkpoint bypass did not isolate the missing pinned-host path")
    elif exit_code != 1 or len(errors) != 1 or "AssertionError" not in (errors[0]["detail"] or "") or engagement:
        raise RuntimeError("mutation did not produce exactly one numerical assertion failure on its engaged path")
    return {"observed": "FAIL", "verdict": "MUTATION_DETECTED", "engagement_error": engagement}


def worker(args):
    row = {"cell": args.cell, "mutation": args.mutation, "status": "FAIL", "expected": "PASS" if args.mutation == "none" else "FAIL"}
    probe = None
    try:
        row["environment"] = cuda_environment()
        import pytest
        with PathProbe(args.cell, args.mutation) as probe:
            code = pytest.main([*TESTS[args.cell], "-q", "-s"], plugins=[probe])
            row.update(assess(probe, code), pytest_exit=code)
        row["status"] = "PASS"
    except Exception as exc:
        row.update(exception=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if probe is not None:
            row.update(engagement=probe.stats, test_reports=probe.reports, collection_errors=probe.collection_errors)
        try:
            row["identity"] = source_identity()
        except Exception as exc:
            row["identity_error"] = f"{type(exc).__name__}: {exc}"
            row["status"] = "FAIL"
        Path(args.result).write_text(json.dumps(row, indent=2, default=str) + "\n")
    return 0 if row["status"] == "PASS" else 1


def run_suite(root, timeout):
    deadline, rows = time.monotonic() + timeout, []
    for cell, mutation in CASES:
        directory = root / f"{cell}-{mutation}"
        directory.mkdir()
        result = directory / "result.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--cell", cell,
               "--mutation", mutation, "--result", str(result)]
        row = {"cell": cell, "mutation": mutation, "status": "FAIL"}
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout)
            env = {k: v for k, v in os.environ.items() if not k.startswith("E4B_")}
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                       PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""))
            with (directory / "worker.log").open("w") as log:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, env=env)
            if result.exists():
                row = json.loads(result.read_text())
                expected = "PASS" if mutation == "none" else "FAIL"
                if row.get("cell") != cell or row.get("mutation") != mutation:
                    raise ValueError("worker result belongs to a different cell/mutation")
                if row.get("status") == "PASS" and (row.get("observed") != expected or row.get("expected") != expected):
                    raise ValueError("worker PASS did not establish the required baseline/mutation outcome")
                if proc.returncode != 0 and row.get("status") == "PASS":
                    row.update(status="FAIL", exception=f"worker exited {proc.returncode} after reporting PASS")
            else:
                row["exception"] = f"worker exited {proc.returncode} without a result"
        except subprocess.TimeoutExpired:
            row["exception"] = f"suite exceeded its {timeout:g}s deadline"
        except (OSError, ValueError) as exc:
            row.update(status="FAIL", exception=f"invalid worker execution/result: {exc}")
        rows.append(row)
        print(f"{row['status']} {cell}/{mutation}: {row.get('verdict', row.get('exception'))}", flush=True)
    summary = {"schema": "e4b-gpu-path-smoke/1", "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
               "correctness_only": True, "downloads": False, "cells": rows}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return 0 if summary["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="new directory for worker logs and summary.json")
    parser.add_argument("--timeout", type=float, default=300, help="whole-suite deadline in seconds")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cell", choices=TESTS, help=argparse.SUPPRESS)
    parser.add_argument("--mutation", choices=("none", *(m for values in MUTATIONS.values() for m in values)), help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.cell is None or args.mutation not in ("none", *MUTATIONS[args.cell]) or not args.result:
            parser.error("worker needs a cell, its supported mutation and result path")
        return worker(args)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.output_dir:
        root = Path(args.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        return run_suite(root, args.timeout)
    with tempfile.TemporaryDirectory(prefix="e4b-gpu-path-smoke-") as tmp:
        return run_suite(Path(tmp), args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
