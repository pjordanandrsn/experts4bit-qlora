"""L-only census worker; import the sealed13b science path supplied by the caller."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from dq11_l_census_capture import Capture
from dq11_l_census_reduce import (
    INPUT_SHA,
    LOGGETTA,
    MODEL_SHA,
    SCIENCE,
    SCIENCE_SHA,
    STEP_PREFIX,
    normalize_trace,
    sha,
)


def instrument_seal():
    names = ("dq11_l_census_run.py", "dq11_l_census_capture.py", "dq11_l_census_reduce.py")
    return {name: sha(Path(__file__).with_name(name)) for name in names}


def rehearse_guard(directory):
    """Explicit real-model A2000 contract; no zero-update or tiny-model marker reuse."""
    marker = json.loads((directory / "L-CENSUS-REHEARSAL.json").read_text())
    if (
        marker.get("schema") != "dq11-l-census-rehearsal/1"
        or marker.get("science_eligible") is not False
        or marker.get("source") != SCIENCE
        or marker.get("nonce") != os.environ.get("TC1_RUN_NONCE")
        or marker.get("instrument") != instrument_seal()
        or marker.get("optimizer_updates") != 40
        or marker.get("input_sha256") != INPUT_SHA
        or marker.get("model_sha256") != MODEL_SHA
        or marker.get("wheel_lock_sha256") != sha(directory / "wheels.json")
        or os.environ.get("DQ11_GPU_CLAIM_CONFIRMED") != "1"
    ):
        raise ValueError("explicit sealed real-L census rehearsal/ownership required")
    return marker


def rehearse_hardware(directory):
    rehearse_guard(directory)
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name() != "NVIDIA RTX A2000 12GB"
        or torch.cuda.get_device_capability() != (8, 6)
        or torch.version.cuda != "12.8"
        or str(torch.__version__) != "2.11.0+cu128"
    ):
        raise ValueError("real-L rehearsal requires A2000/sm86/Torch2.11.0+cu128")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    if not 11 * (1 << 30) <= total <= 13 * (1 << 30) or free < 9 * (1 << 30):
        raise ValueError("A2000 memory guard refused")
    cap = min(9 * (1 << 30), int(total * 0.75))
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return {
        "schema": "dq11-l-census-rehearsal-hardware/1",
        "science_eligible": False,
        "gpu": torch.cuda.get_device_name(),
        "sm": [8, 6],
        "torch": str(torch.__version__),
        "process_cap_bytes": cap,
        "free_before_bytes": free,
        "optimizer_updates": 40,
    }


@contextmanager
def rehearsal_admission(directory, enabled):
    if not enabled:
        yield
        return
    # Pinned prepare delegates ONLY its hardware admission through these two
    # functions when quality_read=True. Retain every model/plan/cast operation.
    # This separate marker authorizes40updates; the old zero-update marker does not.
    import dq11_a2000_quality as hardware

    prior = hardware.require_ownership, hardware.require_hardware
    hardware.require_ownership = rehearse_guard
    hardware.require_hardware = rehearse_hardware
    try:
        yield
    finally:
        hardware.require_ownership, hardware.require_hardware = prior


def run(args):
    directory = args.directory.resolve()
    if os.environ.get("E4B_SHA") != SCIENCE or not os.environ.get("TC1_RUN_NONCE"):
        raise ValueError("registered source and fresh run nonce required")
    if args.out.exists():
        raise ValueError("refuse overwrite of a census receipt")
    if sha(directory / "locked_inputs.json") != INPUT_SHA or sha(directory / "model_files.json") != MODEL_SHA:
        raise ValueError("real canonical Mistral/input seal changed")
    science = args.science.resolve()
    if sha(science / "science.sha256") != SCIENCE_SHA:
        raise ValueError("science manifest is not the13b closure")
    for line in (science / "science.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (science / name).resolve()
        if not path.is_relative_to(science) or sha(path) != digest:
            raise ValueError("staged13b science bytes changed: " + name)
    sys.path.insert(0, str(args.science.resolve()))
    from dq11_arm import prepare, read_inputs, runtime
    from dq11_common import adapter_slots, base_census, install_initial, object_sha, path_census, score, tensor_sha
    from dq11_proof_policy import capture_pass, observer_policy, process_environment, settings, validate_shipped_policy
    from dq11_reduce import quality
    from loggetta.measure import DriverMemorySampler

    kind = "proof" if args.kind == "proof" else "read"
    with process_environment(kind):
        if args.rehearsal:
            rehearse_guard(directory)  # ownership gate before framework import
        versions = runtime(directory)
        locked, tokens = read_inputs(directory)
        authority = json.loads((directory / "source_authority.json").read_text())
        if authority.get("e4b_sha") != SCIENCE:
            raise ValueError("source authority is not the13bL path")
        with rehearsal_admission(directory, args.rehearsal):
            model, trainable, workload, engagement = prepare(directory, "L", quality_read=args.rehearsal)
        import torch

        expected_torch = "2.11.0+cu128" if args.rehearsal else "2.12.1+cu130"
        if str(torch.__version__) != expected_torch:
            raise ValueError("profiler/runtime version differs from registered mode")
        initial = install_initial(model, directory / "adapters/adapter_init.safetensors")
        slots, modules = adapter_slots(model)
        before = path_census(model, "L", set(authority["sha256"]))
        base = base_census(model)
        initial_quality = score(model, tokens)
        execution_before = settings()
        row = {
            "schema": "dq11-l-census-run/1",
            "kind": args.kind,
            "science": SCIENCE,
            "science_eligible": not args.rehearsal,
            "loggetta": LOGGETTA,
            "torch": str(torch.__version__),
            "nonce": os.environ["TC1_RUN_NONCE"],
            "input_sha256": INPUT_SHA,
            "model_sha256": MODEL_SHA,
            "science_manifest_sha256": SCIENCE_SHA,
            "instrument": instrument_seal(),
            "instrument_sha256": object_sha(instrument_seal()),
            "source_authority_sha256": sha(directory / "source_authority.json"),
            "identity_sha256": object_sha({"initial": initial, "tokens": locked["tokens"], "runtime": versions}),
            "initial": initial,
            "tokens": locked["tokens"],
            "runtime": versions,
            "path_before": before,
            "initial_quality": initial_quality,
            "engaged": engagement,
            "capture_steps": [],
            "profiler_removed": True,
        }
        if args.kind == "proof":
            ids = torch.tensor([tokens["train"][0]], device="cuda", dtype=torch.long)
            observer = Capture(model, slots, modules, args.trace)
            with observer_policy() as policy:
                with observer.installed():
                    observer.begin_profile()
                    observer.step = 0
                    try:
                        from torch.profiler import record_function

                        with record_function(STEP_PREFIX + "0"):
                            observed, _ = capture_pass(model, slots, ids)
                    finally:
                        observer.end_profile()
                clean, _ = capture_pass(model, slots, ids)
                if observed != clean or not observer.removed:
                    raise ValueError("reporting observer changed C loss/gradients or failed removal")
                model.zero_grad(set_to_none=True)
            cuda_probe = normalize_trace(json.loads(args.trace.read_text()), expected_steps=(0,))
            row.update(
                cuda_canary_device_events=cuda_probe["captured_device_events"],
                observer_same_arm_bitwise=True,
                observer_policy=policy,
                proof=clean,
                trace_sha256=sha(args.trace),
                gradients=dict(observer.gradients),
            )
        else:
            validate_shipped_policy(settings())
            if args.proof is None:
                raise ValueError("fresh proof receipt is required")
            proof = json.loads(args.proof.read_text())
            row["proof_sha256"] = sha(args.proof)
            if (
                proof.get("kind") != "proof"
                or proof.get("observer_same_arm_bitwise") is not True
                or proof.get("nonce") != row["nonce"]
                or proof.get("science_eligible") is not row["science_eligible"]
                or any(proof.get(k) != row[k] for k in ("identity_sha256", "instrument_sha256", "path_before"))
                or not quality(proof, row, "initial_quality")
            ):
                raise ValueError("fresh C proof/initial quality missing or mismatched")
            from dq11_stream_witness import require_train_prefetch, train_prefetch_snapshot
            from loggetta.backends import dense_train
            from loggetta.backends.experts4bit_train import train_loop

            route_before = train_prefetch_snapshot(model)
            data = SimpleNamespace(
                blocks=tokens["train"], info={"source": "DQ11 sealed tokens", "tokens": locked["tokens"]["train"]}
            )
            with DriverMemorySampler() as sampler:
                if args.kind == "reference":
                    row["training"] = train_loop(
                        model,
                        trainable,
                        str(directory / "hf-cache/model"),
                        workload,
                        sampler,
                        {},
                        seed=3407,
                        warmup=5,
                        prepared_data=data,
                        frozen_digest=dense_train.frozen_digest,
                        frozen_kind="dense",
                    )
                else:
                    capture = Capture(model, slots, modules, args.trace)
                    with capture.installed(), capture.trainer_clock(train_loop):
                        row["training"] = train_loop(
                            model,
                            trainable,
                            str(directory / "hf-cache/model"),
                            workload,
                            sampler,
                            {},
                            seed=3407,
                            warmup=5,
                            prepared_data=data,
                            frozen_digest=dense_train.frozen_digest,
                            frozen_kind="dense",
                        )
                    row.update(
                        capture_steps=[6, 7, 8],
                        gradients=dict(capture.gradients),
                        calls=dict(capture.calls),
                        trace_sha256=sha(args.trace),
                        profiler_removed=capture.removed,
                    )
            row["train_prefetch"] = require_train_prefetch(
                directory, model, route_before, 40, rehearsal=args.rehearsal
            )
            row["final_quality"] = score(model, tokens)
            row["final_adapters_sha256"] = {key: tensor_sha(p) for key, p in slots.items()}
            if row["training"]["status"] != "OK":
                raise ValueError("shipped training integrity failed")
        if path_census(model, "L", set(authority["sha256"])) != before or base_census(model) != base:
            raise ValueError("reporting changed loaded callables/frozen bases")
        if runtime(directory) != versions or settings() != execution_before:
            raise ValueError("runtime or execution policy changed during census")
        row.update(path_after=before, frozen_unchanged=True, status="OK", quality_pass=True)
        if not quality(row, row, "initial_quality"):
            raise ValueError("initial scoring integrity failed")
        if args.rehearsal:
            row["interpretation"] = "Real Mistral L/A2000 correctness only; no speed/BUILD/science inference"
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--science", type=Path, required=True)
    parser.add_argument("--kind", choices=("proof", "reference", "capture"), required=True)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rehearsal", action="store_true")
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
