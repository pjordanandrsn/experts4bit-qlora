"""Draft DQ11 box protocol: CPU stub events only, never scientific evidence.

The instrument amendment must replace these events with reviewed real workers.
Live invocation refuses before any operation; no GPU/network/framework imports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def phases(protocol):
    if protocol["status"] != "DRAFT_NOT_LAUNCHABLE":
        raise ValueError("draft runner cannot interpret a launchable registration")
    if protocol["arms"] != ["L", "U", "U0", "U0", "U", "L"]:
        raise ValueError("missing, duplicate or reordered registered arm")
    fixed = {"updates": 40, "timing_warmup": 5, "sequence": 2048,
             "micro_batch": 1, "grad_accum": 1, "rank": 16, "alpha": 32,
             "seed": 3407, "adapter_storage": "float32"}
    if any(protocol[key] != value for key, value in fixed.items()):
        raise ValueError("changed registered workload")
    yield "preflight", None
    yield "prepare_seal", None
    yield "binding_precision_proof", None
    yield "initial_quality", None
    for repetition, order in enumerate((protocol["arms"][:3], protocol["arms"][3:]), 1):
        for arm in order:
            tag = f"{repetition}-{arm}"
            yield "audit_before", tag
            yield "train", tag
            yield "audit_after", tag
            yield "final_quality", tag
    yield "quality_gate", None
    yield "decision", None


def dry_run(protocol, worker, *, nonce):
    """Exercise ordering/refusal via an injected stub, keeping teardown on failure.

    Only event identity is validated here. No synthetic loss/timing/quality value
    can become a scientific pass. The real instrument needs independent checks.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", nonce):
        raise ValueError("unsafe nonce")
    events = []
    failure = None

    def call(phase, arm):
        result = worker(phase, arm)
        if result != {"stub": True, "phase": phase, "arm": arm}:
            raise ValueError("missing, changed or non-stub phase result")
        events.append(result)

    try:
        for phase, arm in phases(protocol):
            call(phase, arm)
    except (RuntimeError, ValueError) as error:
        failure = str(error)
    finally:
        try:
            call("teardown_proof", None)
        except (RuntimeError, ValueError) as error:
            failure = str(error)
    return {"schema": "dq11-dry-run/1", "nonce": nonce,
            "status": "DRY_RUN_FAILED" if failure else "DRY_RUN_COMPLETE",
            "scientific_evidence": False, "launchable": False,
            "events": events, "failure": failure}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("dq11_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    if not args.dry_run:
        parser.exit(78, "DQ11 DRAFT_NOT_LAUNCHABLE: reviewed science instrument amendment required\n")
    protocol = json.loads(args.protocol.read_text())
    receipt = dry_run(protocol, lambda phase, arm: {"stub": True, "phase": phase, "arm": arm}, nonce=args.nonce)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    return 11 if receipt["failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
