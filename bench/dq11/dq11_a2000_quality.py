"""Initial-only real Mistral quality on A2000/cu128; never SCI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SCHEMA = "dq11-a2000-real-initial-quality/1"
PINS = {
    "model_files.json": "c21aac15939dd1c6a8c8c367b2887c44668335927f81368afa996690470a0ff5",
    "locked_inputs.json": "9937024a50d3d44b28016d47b85e3e2feba159ce9d50f82403120e17581a4144",
    "wheels.json": "8adb525bfe995969eec9dfa3afd5c7c9f68d4ed2cc1fb15781057a6b170152c1",
}


def mode(directory):
    """A separate sealed real-model contract, never the tiny-model override."""
    if any(v for k, v in os.environ.items() if k.startswith("DQ11_REHEARSAL")):
        raise ValueError("real initial-quality read cannot use tiny rehearsal mode")
    directory = Path(directory)
    marker = json.loads((directory / "A2000-INITIAL-QUALITY.json").read_text())
    if (
        marker.get("schema") != SCHEMA
        or marker.get("science_eligible") is not False
        or marker.get("optimizer_updates") != 0
        or type(marker.get("optimizer_updates")) is not int
        or marker.get("source") != os.environ.get("E4B_SHA")
        or not re.fullmatch("[a-f0-9]{40}", marker.get("source", ""))
        or marker.get("nonce") != os.environ.get("TC1_RUN_NONCE")
        or not re.fullmatch("[A-Za-z0-9_-]{1,128}", marker.get("nonce", ""))
        or marker.get("pinned_files") != PINS
    ):
        raise ValueError("real initial-quality marker changed")
    for name, expected in PINS.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("registered real model/input/runtime manifest changed: " + name)
    model = json.loads((directory / "model_files.json").read_text())
    if (
        model.get("model") != "mistralai/Mistral-7B-v0.1"
        or model.get("revision") != "27d67f1b5f57dc0953326b2601d68371d40ea8da"
    ):
        raise ValueError("registered real Mistral required")
    return marker


def require_ownership(directory):
    marker = mode(directory)
    if os.environ.get("DQ11_GPU_CLAIM_CONFIRMED") != "1":
        raise ValueError("fresh exclusive bus GPU ownership required before imports")
    return marker


def require_hardware(directory):
    marker = require_ownership(directory)
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name() != "NVIDIA RTX A2000 12GB"
        or torch.cuda.get_device_capability() != (8, 6)
        or torch.version.cuda != "12.8"
        or str(torch.__version__) != "2.11.0+cu128"
    ):
        raise ValueError("real initial-quality read requires A2000/cu128/sm86")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    if not 11 * (1 << 30) <= total <= 13 * (1 << 30) or free < 9 * (1 << 30):
        raise ValueError("real initial-quality memory guard refused")
    # Bound only the owned process, leaving space beyond the household baseline.
    cap = min(9 * (1 << 30), int(total * 0.75))
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return {
        "schema": SCHEMA,
        "science_eligible": False,
        "source": marker["source"],
        "nonce": marker["nonce"],
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "sm": list(torch.cuda.get_device_capability()),
        "free_before_bytes": free,
        "total_bytes": total,
        "process_cap_bytes": cap,
        "optimizer_updates": 0,
    }


def worker(directory, arm):
    """Zero-update score via registered preparation/scorer, no optimizer or proof."""
    directory = Path(directory)
    marker = require_ownership(directory)
    if arm not in ("L", "U", "U0"):
        raise ValueError("unknown initial-quality arm")
    from dq11_arm import prepare, read_inputs, runtime
    from dq11_common import adapter_slots, base_census, file_sha, install_initial, object_sha, path_census, score
    from dq11_proof_policy import process_environment, settings, validate_shipped_policy

    with process_environment("read"):
        versions = runtime(directory)
        locked, tokens = read_inputs(directory)
        authority = json.loads((directory / "source_authority.json").read_text())
        if authority["e4b_sha"] != marker["source"]:
            raise ValueError("installed source authority changed")
        sources = set(authority["sha256"])
        model, _, _, engagement = prepare(directory, arm, quality_read=True)
        if arm in ("U", "U0"):
            engagement["token_0_scored_inputs"] = {
                name: sum(block.count(0) for block in tokens[name])
                for name in ("alpaca-heldout", "wikitext-test")
            }
        initial = install_initial(model, directory / "adapters/adapter_init.safetensors")
        before = path_census(model, arm, sources)
        base = base_census(model)
        execution_settings = settings()
        validate_shipped_policy(execution_settings)
        scores = score(model, tokens)
        if (
            path_census(model, arm, sources) != before
            or base_census(model) != base
            or settings() != execution_settings
            or any(p.grad is not None for p in adapter_slots(model)[0].values())
        ):
            raise ValueError("initial-only read changed frozen values/bindings/policy or produced gradients")
        from dq11_reduce import quality

        row = {
            "schema": SCHEMA,
            "science_eligible": False,
            "arm": arm,
            "source": marker["source"],
            "nonce": marker["nonce"],
            "runtime": versions,
            "initial": initial,
            "tokens": locked["tokens"],
            "initial_quality": scores,
            "base": base,
            "path_before": before,
            "path_after": before,
            "execution_settings": execution_settings,
            "engaged": engagement,
            "optimizer_updates": 0,
            "frozen_unchanged": True,
            "provenance": {
                name: file_sha(directory / name)
                for name in (
                    "science.sha256",
                    "wheels.json",
                    "model_files.json",
                    "locked_inputs.json",
                    "source_authority.json",
                )
            },
            "interpretation": "A2000/cu128/sm86 initial-only quality; loaded binding census, no observed backward proof, speed or SCI licence",
        }
        row["identity_sha256"] = object_sha({"initial": initial, "tokens": row["tokens"], "runtime": versions})
        assert quality(row, row, "initial_quality")
        return row


def report(directory, *, include_U0=True):
    """Re-derive paired K8 only from completed matched initial-only receipts."""
    directory = Path(directory)
    marker = mode(directory)
    from dq11_common import file_sha, object_sha, value_identity
    from dq11_reduce import TEXTS, quality
    from dq11_proof_policy import validate_shipped_policy

    rows = {arm: json.loads((directory / f"receipts/initial-{arm}.json").read_text()) for arm in ("L", "U")}
    optional = directory / "receipts/initial-U0.json"
    if include_U0 and optional.exists():
        rows["U0"] = json.loads(optional.read_text())
    locked = json.loads((directory / "locked_inputs.json").read_text())
    versions = {p["name"]: p["version"] for p in json.loads((directory / "wheels.json").read_text())["packages"]}
    versions.update(loggetta="34ecb6cec6f43a6f8607ff9f192749fdc7b587e9", **{"experts4bit-qlora": marker["source"]})
    anchor = rows["L"]
    for arm, row in rows.items():
        if (
            row["schema"] != SCHEMA
            or row["science_eligible"] is not False
            or row["arm"] != arm
            or row["source"] != marker["source"]
            or row["nonce"] != marker["nonce"]
            or type(row["optimizer_updates"]) is not int
            or row["optimizer_updates"] != 0
            or row["frozen_unchanged"] is not True
            or row["path_before"] != row["path_after"]
            or row["tokens"] != locked["tokens"]
            or row["runtime"] != versions
            or len(row["initial"]) != 448
            or row["initial"] != anchor["initial"]
            or value_identity(row["base"]) != value_identity(anchor["base"])
        ):
            raise ValueError("initial-only matched receipt identity changed")
        validate_shipped_policy(row["execution_settings"])
        hardware = row["engaged"]["a2000_initial_quality"]
        if (
            hardware["schema"] != SCHEMA
            or hardware["science_eligible"] is not False
            or hardware["source"] != marker["source"]
            or hardware["nonce"] != marker["nonce"]
            or hardware["gpu"] != "NVIDIA RTX A2000 12GB"
            or hardware["cuda"] != "12.8"
            or hardware["sm"] != [8, 6]
            or hardware["optimizer_updates"] != 0
            or hardware["process_cap_bytes"] != min(9 * (1 << 30), int(hardware["total_bytes"] * 0.75))
            or not 11 * (1 << 30) <= hardware["total_bytes"] <= 13 * (1 << 30)
            or hardware["free_before_bytes"] < 9 * (1 << 30)
        ):
            raise ValueError("initial-only hardware receipt changed")
        for name, digest in row["provenance"].items():
            if (
                name
                not in (
                    "science.sha256",
                    "wheels.json",
                    "model_files.json",
                    "locked_inputs.json",
                    "source_authority.json",
                )
                or file_sha(directory / name) != digest
            ):
                raise ValueError("initial-only source provenance changed")
        if set(row["provenance"]) != {
            "science.sha256",
            "wheels.json",
            "model_files.json",
            "locked_inputs.json",
            "source_authority.json",
        }:
            raise ValueError("initial-only source provenance missing")
        if row["identity_sha256"] != object_sha(
            {"initial": row["initial"], "tokens": row["tokens"], "runtime": versions}
        ):
            raise ValueError("initial-only identity hash inconsistent")
    comparisons = []
    for base in ("L", "U0") if "U0" in rows else ("L",):
        passed = quality(rows[base], rows["U"], "initial_quality")
        texts = []
        for text in TEXTS:
            b, u = rows[base]["initial_quality"][text], rows["U"]["initial_quality"][text]
            texts.append(
                {
                    "text": text,
                    "base_ppl": b["ppl"],
                    "U_ppl": u["ppl"],
                    "U_minus_base_ppl": u["ppl"] - b["ppl"],
                    "absolute_ppl_delta": abs(u["ppl"] - b["ppl"]),
                    "U_minus_base_mean_NLL": (u["nll"] - b["nll"]) / b["targets"],
                    "base_mean_NLL": b["nll"] / b["targets"],
                    "U_mean_NLL": u["nll"] / u["targets"],
                    "targets": u["targets"],
                }
            )
        comparisons.append({"base": base, "candidate": "U", "k8_pass": passed, "texts": texts})
    return {
        "schema": SCHEMA,
        "science_eligible": False,
        "source": marker["source"],
        "nonce": marker["nonce"],
        "optimizer_updates": 0,
        "recommendation": None,
        "paid_GO": False,
        "status": "INITIAL_QUALITY_PASS" if comparisons[0]["k8_pass"] else "INITIAL_QUALITY_FAIL",
        "U0_status": "complete" if "U0" in rows else "not_included",
        "k8_absolute_ppl_budget": 0.05,
        "comparisons": comparisons,
        "scope": "A2000/cu128/sm86 initial-only real Mistral; no5090/cu130/SM120 quality, speed, capacity or shipping licence",
    }


def run(directory, assets):
    """Owned initial-only pipeline; actual phase helper shared with the $0 run."""
    import shutil

    from dq11_rehearsal import timed_phase
    from dq11_science_stage import stage

    directory, assets = Path(directory).resolve(), Path(assets).resolve()
    here = Path(__file__).resolve().parent
    source, nonce = os.environ.get("E4B_SHA", ""), os.environ.get("TC1_RUN_NONCE", "")
    if (
        not re.fullmatch("[a-f0-9]{40}", source)
        or not re.fullmatch("[A-Za-z0-9_-]{1,128}", nonce)
        or os.environ.get("DQ11_GPU_CLAIM_CONFIRMED") != "1"
        or any(v for k, v in os.environ.items() if k.startswith("DQ11_REHEARSAL"))
    ):
        raise ValueError("merged source, unique nonce and GPU ownership required; no tiny mode")
    files = stage(here, assets)
    directory.mkdir(exist_ok=False)
    for name in ("logs", "receipts", "adapters", "data", "hf-cache", "tmp"):
        (directory / name).mkdir()
    for path in files:
        if path.name == "science.sha256":
            target = directory / "science-reference.sha256"
        elif path.name == "adapter_init.safetensors":
            target = directory / "adapters" / path.name
        elif path.name == "tokens.json":
            target = directory / "data" / path.name
        else:
            target = directory / path.name
        shutil.copy2(path, target)
    for name in ("wheels.json", "requirements.lock"):
        shutil.copy2(directory / name, directory / ("science-" + name))
        shutil.copy2(here / "rehearsal" / name, directory / name)
    marker = {
        "schema": SCHEMA,
        "science_eligible": False,
        "source": source,
        "nonce": nonce,
        "optimizer_updates": 0,
        "pinned_files": PINS,
    }
    (directory / "A2000-INITIAL-QUALITY.json").write_text(json.dumps(marker, indent=2) + "\n")
    mode(directory)
    # The owned workspace seal is explicitly non-scientific; original SCI seal retained.
    payloads = sorted(p for p in directory.iterdir() if p.is_file())
    (directory / "science.sha256").write_text(
        "".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n" for p in payloads)
    )
    deadline = int(os.environ["TC1_DEADLINE_EPOCH"])
    env = dict(
        os.environ,
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        TMPDIR=str(directory / "tmp"),
    )
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(key, None)
    ledger = {
        "schema": SCHEMA,
        "run_kind": "real_initial_quality",
        "science_eligible": False,
        "nonce": nonce,
        "deadline_epoch": deadline,
        "reserve_seconds": 300,
        "phases": [],
    }
    (directory / "receipts/phase-timings.json").write_text(json.dumps(ledger, indent=2) + "\n")

    def phase(name, cap, *command):
        timed_phase(directory, env, name, cap, *command)

    status = "REFUSED"
    try:
        phase("git-prerequisite", 240, "bash", "dq11_require_git.sh")
        venv = directory / "venv-dq11"
        phase("create-venv", 120, "python3.11", "-m", "venv", str(venv))
        env["PATH"] = str(venv / "bin") + os.pathsep + env["PATH"]
        phase("bootstrap", 1800, "python", "dq11_bootstrap.py", "--cache", str(directory.parent / "wheel-cache"))
        phase("prepare-real-model", 3600, "python", "dq11_a2000_quality.py", "prepare", "--directory", str(directory))
        for arm in ("L", "U"):
            phase(
                "initial-" + arm,
                900,
                "python",
                "dq11_a2000_quality.py",
                "score",
                "--arm",
                arm,
                "--directory",
                str(directory),
            )
        phase(
            "report", 120, "python", "dq11_a2000_quality.py", "report", "--directory", str(directory), "--primary-only"
        )
        status = "COMPLETE_INITIAL_ONLY"
        # Optional U0 and its report run AFTER the complete primary L/U report.
        # Failure/deadline leaves that required result and the reserve intact.
        try:
            phase(
                "initial-U0",
                900,
                "python",
                "dq11_a2000_quality.py",
                "score",
                "--arm",
                "U0",
                "--directory",
                str(directory),
            )
            phase(
                "optional-U0-report", 120, "python", "dq11_a2000_quality.py", "report", "--directory", str(directory)
            )
        except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
            (directory / "receipts/optional-U0-refusal.json").write_text(
                json.dumps({"schema": SCHEMA, "science_eligible": False, "error": str(error)}, indent=2) + "\n"
            )
    finally:
        (directory / ("QUALITY_STATUS." + nonce)).write_text(status + "\n")
        (directory / ("QUALITY_DONE." + nonce)).touch()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "score", "report"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--arm", choices=("L", "U", "U0"))
    args = parser.parse_args()
    if args.action == "run":
        if args.assets is None:
            parser.error("run requires canonical --assets")
        run(args.directory, args.assets)
    elif args.action == "prepare":
        from dq11_prepare import prepare

        prepare(args.directory, quality_read=True)
    else:
        if args.action == "score" and args.arm is None:
            parser.error("score requires --arm")
        value = (
            worker(args.directory, args.arm)
            if args.action == "score"
            else report(args.directory, include_U0=not args.primary_only)
        )
        target = (
            args.directory
            / "receipts"
            / (
                f"initial-{args.arm}.json"
                if args.action == "score"
                else ("initial-quality-primary.json" if args.primary_only else "initial-quality-with-U0.json")
            )
        )
        target.write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(value, indent=2))
