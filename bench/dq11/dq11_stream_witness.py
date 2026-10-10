"""Read-only live CUDA streaming-route witness shared by science and rehearsal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

ROUTE_COUNTERS = ("uses", "fwd_prefetch_issued", "bwd_prefetch_issued")


def train_prefetch_snapshot(model):
    """Read live counters from the real CUDA schedule, never a prepare report."""
    from experts4bit_qlora.engines.dense_offload import decoder_layers, dense_offload_report

    handles = [getattr(layer, "_dense_offload", None) for _, layer in decoder_layers(model)]
    if (len(handles) != 32 or any(h is None or h.device.type != "cuda" or h._train is None for h in handles)
            or len({id(h._train) for h in handles}) != 1):
        raise ValueError("DQ11 requires all 32 handles on one live CUDA train-prefetch schedule")
    schedules = dense_offload_report(handles)["train_prefetch"]
    if not isinstance(schedules, dict) or len(schedules) != 1:
        raise ValueError("missing live CUDA train-prefetch counters")
    return {"handles": len(handles), "streamed_bytes": sum(h.bytes for h in handles),
            "devices": {device: {key: counts.get(key) for key in ROUTE_COUNTERS}
                        for device, counts in schedules.items()}}


def validate_train_prefetch_witness(witness, *, rehearsal=False):
    """The worker and final checker use the same strictly positive delta gate."""
    schema = "dq11-rehearsal-train-prefetch/1" if rehearsal else "dq11-train-prefetch/1"
    if (not isinstance(witness, dict) or witness.get("schema") != schema
            or witness.get("science_eligible") is not (not rehearsal) or witness.get("status") != "PASS"
            or type(witness.get("updates")) is not int or witness["updates"] != 40):
        raise ValueError("CUDA train-prefetch route witness refused")
    before, after = witness.get("before", {}), witness.get("after", {})
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("malformed CUDA train-prefetch snapshots")
    devices = before.get("devices", {})
    if (not isinstance(devices, dict) or not isinstance(after.get("devices"), dict)
            or not isinstance(witness.get("delta"), dict)):
        raise ValueError("malformed CUDA train-prefetch counters")
    if (before.get("handles") != 32 or after.get("handles") != 32
            or type(before.get("streamed_bytes")) is not int or before["streamed_bytes"] <= 0
            or after.get("streamed_bytes") != before["streamed_bytes"]
            or len(devices) != 1 or set(devices) != set(after.get("devices", {}))
            or set(devices) != set(witness.get("delta", {}))):
        raise ValueError("CUDA train-prefetch live schedule changed or was empty")
    for device, counts in devices.items():
        if not isinstance(device, str) or not re.fullmatch(r"cuda(?::0)?", device):
            raise ValueError("train-prefetch route witness is not the single CUDA device")
        if (not isinstance(counts, dict) or not isinstance(after["devices"][device], dict)
                or not isinstance(witness["delta"][device], dict)):
            raise ValueError("malformed CUDA train-prefetch per-device counters")
        for key in ROUTE_COUNTERS:
            old, new = counts.get(key), after["devices"][device].get(key)
            if (type(old) is not int or type(new) is not int or old < 0 or new <= old
                    or witness["delta"][device].get(key) != new - old):
                raise ValueError("CUDA train-prefetch did not execute during the forty training updates")


def require_train_prefetch(directory, model, before, updates, *, rehearsal=False):
    witness = {"schema": "dq11-rehearsal-train-prefetch/1" if rehearsal else "dq11-train-prefetch/1",
               "science_eligible": not rehearsal, "status": "PASS",
               "source": os.environ.get("E4B_SHA"), "nonce": os.environ.get("TC1_RUN_NONCE"),
               "phase": "after forty training updates, before final evaluation",
               "updates": updates, "before": before, "after": None, "delta": {}}
    path = Path(directory) / "receipts" / f"train-prefetch-read-{os.getpid()}.json"
    try:
        after = train_prefetch_snapshot(model)
        witness["after"] = after
        for device, counts in after["devices"].items():
            old_counts = before.get("devices", {}).get(device, {})
            witness["delta"][device] = {
                key: counts[key] - old_counts[key] if type(counts.get(key)) is int and type(old_counts.get(key)) is int else None
                for key in ROUTE_COUNTERS}
        validate_train_prefetch_witness(witness, rehearsal=rehearsal)
    except (ValueError, AttributeError) as error:
        witness.update(status="REFUSED", reason=str(error))
        path.write_text(json.dumps(witness, indent=2) + "\n")
        raise
    path.write_text(json.dumps(witness, indent=2) + "\n")
    return witness
