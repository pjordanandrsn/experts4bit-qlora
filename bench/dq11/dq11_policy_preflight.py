"""Fresh exact-wheel CPU policy evidence before DQ11 quotes; no GPU or model work."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from dq11_common import file_sha
from dq11_proof_policy import settings, validate_shipped_policy


def apply_registered_tf32(directory, framework):
    """Execute the two actual sealed prepare assignments, without importing/loading a model."""
    source = directory / "dq11_arm.py"
    tree = ast.parse(source.read_text())
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    targets = {"torch.backends.cuda.matmul.allow_tf32", "torch.backends.cudnn.allow_tf32"}
    nodes = [node for node in prepare.body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and ast.unparse(node.targets[0]) in targets]
    if (len(nodes) != 2 or {ast.unparse(node.targets[0]) for node in nodes} != targets
            or any(not isinstance(node.value, ast.Constant) or node.value.value is not False for node in nodes)):
        raise ValueError("registered prepare TF32 assignments changed")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), {"torch": framework})
    return {"source_sha256": file_sha(source), "lines": [node.lineno for node in nodes]}


def worker(directory, arm):
    """Unsloth CPU support is isolated here; it is never an environment for scientific arms."""
    if any(k.startswith("UNSLOTH_") or k.startswith("DQ11_REHEARSAL") for k in os.environ):
        raise ValueError("inherited arm/rehearsal switches refused by CPU policy probe")
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        raise ValueError("inherited proof workspace refused by CPU policy probe")
    if arm in {"U", "U0"}:
        os.environ.update(UNSLOTH_ALLOW_CPU="1", UNSLOTH_COMPILE_DISABLE="1", UNSLOTH_RETURN_LOGITS="1")
        from unsloth import FastLanguageModel  # noqa: F401 -- actual locked import, no model call
    import torch
    from loggetta import Constraints, Workload, plan  # noqa: F401
    from loggetta.backends import dense, dense_train  # noqa: F401
    from loggetta.hardware import probe  # noqa: F401
    import experts4bit_qlora  # noqa: F401

    if str(torch.__version__) != "2.12.1+cu130" or torch.version.cuda != "13.0":
        raise ValueError("CPU policy probe requires the exact science Torch cu130 build")
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise ValueError("CPU policy probe must have no available or initialized CUDA device")
    authority = json.loads((directory / "source_authority.json").read_text())
    if file_sha(Path(torch.__file__)) not in authority["sha256"]:
        raise ValueError("loaded Torch Python source differs from hash-verified wheels")
    before = settings()
    assignments = apply_registered_tf32(directory, torch)
    actual = settings()
    validate_shipped_policy(actual)
    return {"arm": arm, "status": "PASS", "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "cpu_only": True, "cuda_initialized": torch.cuda.is_initialized(),
            "unsloth_cpu_supported_import": arm in {"U", "U0"},
            "after_library_imports": before, "execution_settings": actual, "tf32_assignments": assignments,
            "limitation": "CPU-supported Unsloth import and flag check; no model, CUDA kernel or GPU-path certification"}


def probe(directory, source, nonce):
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 11):
        raise ValueError("exact science wheel policy probe requires Linux x86_64 Python 3.11")
    wheels = json.loads((directory / "wheels.json").read_text())["packages"]
    versions = {row["name"]: importlib.metadata.version(row["name"]) for row in wheels}
    if len(wheels) != 101 or any(versions[row["name"]] != row["version"] for row in wheels):
        raise ValueError("CPU policy runtime differs from the full 101-package science lock")
    authority = json.loads((directory / "source_authority.json").read_text())
    if authority["e4b_sha"] != source or authority["wheel_lock_sha256"] != file_sha(directory / "wheels.json"):
        raise ValueError("CPU policy source authority does not bind the requested science source")
    commits = {name: json.loads(importlib.metadata.distribution(name).read_text("direct_url.json"))["vcs_info"]["commit_id"]
               for name in ("experts4bit-qlora", "loggetta")}
    if commits != {"experts4bit-qlora": source, "loggetta": "34ecb6cec6f43a6f8607ff9f192749fdc7b587e9"}:
        raise ValueError("installed source packages differ from requested commits")
    report = {"schema": "dq11-cpu-policy-gate/1", "source": source, "nonce": nonce,
              "science_sha256": file_sha(directory / "science.sha256"), "wheel_lock_sha256": file_sha(directory / "wheels.json"),
              "preflight_sha256": file_sha(Path(__file__)), "arm_sha256": file_sha(directory / "dq11_arm.py"),
              "utc": datetime.now(timezone.utc).isoformat(), "cpu_only": True, "packages": versions, "arms": [],
              "passed": False, "gpu_work": False, "source_commits": commits}
    for arm in ("L", "U", "U0"):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        child = subprocess.run([sys.executable, str(Path(__file__)), "--directory", str(directory), "--worker", arm],
                               env=env, capture_output=True, text=True, timeout=180, check=False)
        row = {"arm": arm, "stdout": child.stdout, "stderr": child.stderr, "returncode": child.returncode}
        try:
            if child.returncode:
                raise ValueError("CPU policy library import or settings check refused")
            result = json.loads(child.stdout.splitlines()[-1])
            if result["arm"] != arm or result["status"] != "PASS":
                raise ValueError("CPU policy child result changed")
            row["result"] = result
        except (ValueError, KeyError, IndexError) as error:
            row["error"] = str(error)
        report["arms"].append(row)
    report["passed"] = all("error" not in row for row in report["arms"])
    return report


def validate_report(report, directory, source, nonce):
    wheels = json.loads((directory / "wheels.json").read_text())["packages"]
    if (report["schema"] != "dq11-cpu-policy-gate/1" or report["source"] != source or report["nonce"] != nonce
            or report["passed"] is not True or report["cpu_only"] is not True or report["gpu_work"] is not False
            or report["science_sha256"] != file_sha(directory / "science.sha256")
            or report["wheel_lock_sha256"] != file_sha(directory / "wheels.json")
            or report["preflight_sha256"] != file_sha(directory / "dq11_policy_preflight.py")
            or report["arm_sha256"] != file_sha(directory / "dq11_arm.py")
            or report["packages"] != {row["name"]: row["version"] for row in wheels}
            or report["source_commits"] != {"experts4bit-qlora": source, "loggetta": "34ecb6cec6f43a6f8607ff9f192749fdc7b587e9"}
            or [row["arm"] for row in report["arms"]] != ["L", "U", "U0"]):
        raise ValueError("CPU policy gate source/closure/runtime/nonce/result mismatch")
    for row in report["arms"]:
        actual = row["result"]
        if (row["returncode"] != 0 or actual["arm"] != row["arm"] or actual["status"] != "PASS"
                or actual["torch"] != "2.12.1+cu130" or actual["cuda_build"] != "13.0"
                or actual["cpu_only"] is not True or actual["cuda_initialized"] is not False
                or actual["unsloth_cpu_supported_import"] is not (row["arm"] in {"U", "U0"})
                or actual["tf32_assignments"]["source_sha256"] != report["arm_sha256"]):
            raise ValueError("CPU policy arm lacks exact-build, CPU-only actual evidence")
        validate_shipped_policy(actual["execution_settings"])


def preflight_policy(directory, source):
    """Execute a fresh local or transported exact-wheel probe; stale receipts cannot replace it."""
    nonce = uuid.uuid4().hex
    command = json.loads(os.environ["DQ11_POLICY_COMMAND"]) if "DQ11_POLICY_COMMAND" in os.environ else [sys.executable, str(directory / "dq11_policy_preflight.py")]
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        raise ValueError("CPU policy command must be a nonempty JSON argument vector")
    child = subprocess.run([*command, "--source", source, "--nonce", nonce], capture_output=True, text=True,
                           timeout=600, check=False)
    record = {"passed": False, "nonce": nonce, "stdout": child.stdout, "stderr": child.stderr, "returncode": child.returncode}
    try:
        if child.returncode:
            raise ValueError("exact-wheel CPU policy probe refused")
        report = json.loads(child.stdout.splitlines()[-1])
        validate_report(report, directory, source, nonce)
        record.update(passed=True, evidence=report)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        record["error"] = str(error)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source")
    parser.add_argument("--nonce")
    parser.add_argument("--worker", choices=("L", "U", "U0"))
    args = parser.parse_args()
    try:
        result = worker(args.directory, args.worker) if args.worker else probe(args.directory, args.source, args.nonce)
        print(json.dumps(result))
        raise SystemExit(0 if args.worker or result["passed"] else 78)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError, importlib.metadata.PackageNotFoundError) as error:
        parser.exit(78, str(error) + "\n")
