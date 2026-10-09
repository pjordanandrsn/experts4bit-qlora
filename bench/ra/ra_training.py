#!/usr/bin/env python3
"""Execute the registered TC1 field fixture; retain its native receipt unchanged.

This component does not manufacture a reducer-ready features dictionary.
Independent wheel/import/default/fallback evidence is still a supervisor gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import ra_env
import ra_process
import ra_stage

MODEL = {"proof": ("granite", "ibm-granite/granite-3.1-3b-a800m-instruct"),
         "reading": ("qwen3", "Qwen/Qwen3-30B-A3B")}
FIXTURE = {"framework": "e4b", "arm": "fused", "template": "alpaca", "pack": 0,
           "steps": 20, "seq": 2048, "micro-batch": 2, "accum": 4, "autocast": 0,
           "r": 16, "alpha": 16, "lr": "2e-4", "weight-decay": "0.001",
           "lr-schedule": "linear", "warmup-steps": 5, "optim": "adamw_8bit",
           "seed": 3407, "eval-n": 8, "eval-every": 20, "attn-4bit": 1,
           "offload": 0, "adapter-dtype": "fp32", "lora-init": "matched:3407", "dgrad": 1}
SPEC = {"battery", "venv", "cache", "threads", "allocator", "data", "tokens", "prereg",
        "expect_trainable", "deadline_epoch_s", "timeout_s"}


def token_digest(spec, pins):
    """TC1 hashes compact train/eval JSON, separately from the whole input file."""
    path = ra_process.check_input(spec["tokens"])
    tk = json.loads(path.read_bytes())
    fam, model = MODEL[spec["battery"]]
    if (tk["fam"], tk["tokenizer"], tk["revision"], tk["dataset_sha256"], tk["template"], tk["seq"]) != \
            (fam, model, pins["models"][model], spec["data"]["sha256"], "alpaca", 2048) or tk.get("pack", False):
        raise ValueError("prepared token fixture identity")
    rows = tk["train"] + tk["eval"]
    if not tk["train"] or len(tk["eval"]) < 8 or any(not 8 <= len(row) <= 2048 or
            any(type(t) is not int or t < 0 for t in row) for row in rows):
        raise ValueError("prepared token shape")
    body = json.dumps({"train": tk["train"], "eval": tk["eval"]}, separators=(",", ":")).encode()
    digest = hashlib.sha256(body).hexdigest()
    if tk["sha256"] != digest:
        raise ValueError("prepared token payload digest")
    return digest


def command(spec, stage, out, *, profile):
    if set(spec) != SPEC or spec["battery"] not in MODEL:
        raise ValueError("training spec fields/battery")
    for key in ("venv", "cache"):
        if not Path(spec[key]).is_absolute():
            raise ValueError("absolute venv/cache required")
    if type(spec["expect_trainable"]) is not int or spec["expect_trainable"] <= 0:
        raise ValueError("family-derived trainable count required")
    ra_process.window(spec["deadline_epoch_s"], spec["timeout_s"])
    data, tokens, prereg = [ra_process.check_input(spec[key]) for key in ("data", "tokens", "prereg")]
    pins = json.loads((stage / "source-pins.json").read_bytes())
    tokens_sha = token_digest(spec, pins)
    fam, model = MODEL[spec["battery"]]
    opts = {**FIXTURE, "fam": fam, "model": model, "revision": pins["models"][model],
            "tag": "training_profile" if profile else "training", "data": data,
            "data-sha": spec["data"]["sha256"], "tokens": tokens, "tokens-sha": tokens_sha,
            "prereg": prereg, "expect-trainable": spec["expect_trainable"],
            "profile-warm": 10, "profile-steps": 10 if profile else 0,
            "out": out / "native", "adapter-dir": out / "adapters"}
    return [str(Path(spec["venv"]) / "bin/python"), "-B", str(stage / "tc1_arm.py"),
            *(item for k, v in opts.items() for item in ("--" + k, str(v)))], fam, model


def check_native(native, spec, pins, *, profile):
    """Bind command settings and common token bytes to TC1's actual receipt.

    Keep function failures intact for the reducer rather than converting them
    into successful rows. This check establishes fixture identity only.
    """
    fam, model = MODEL[spec["battery"]]
    fixed = {"framework": "e4b", "arm": "fused", "fam": fam, "model": model,
             "revision": pins["models"][model], "tag": "training_profile" if profile else "training",
             "steps": 20, "seq": 2048, "micro_batch": 2, "accum": 4, "autocast": False,
             "r": 16, "alpha": 16, "lr": 2e-4, "seed": 3407, "offload": False,
             "attn_4bit": True, "adapter_dtype": "fp32", "lora_init": "matched:3407", "dgrad": True,
             "template": "alpaca", "expect_trainable": spec["expect_trainable"],
             "profile_steps": 10 if profile else 0, "profile_warm": 10,
             "prereg": spec["prereg"]["path"]}
    if any(native.get(k) != v for k, v in fixed.items()):
        raise ValueError("native TC1 fixture differs from command")
    tk = native["tokens"]
    if tk["sha256"] != token_digest(spec, pins) or tk["pack"] is not False or tk["eval_rows_used"] != 8:
        raise ValueError("native training token/eval fixture")
    if native["optimizer"] != "adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5" or \
            [r["step"] for r in native["eval_curve"]] != [0, 20]:
        raise ValueError("native optimizer/eval schedule")
    # TC1 already records the trajectory; don't replace it with requested LR values.
    expected_lr = [2e-4 * min(i / 5, (20 - i) / 15) for i in range(20)]
    if native["lr_per_step"] != [round(v, 8) for v in expected_lr]:
        raise ValueError("native learning-rate schedule")
    if profile:
        if native["profile"]["profiled_steps"] != 10:
            raise ValueError("native profile window")
    elif native["profile"] is not None:
        raise ValueError("profile entered wall arm")


def execute(spec, stage, out, *, profile=False):
    if not stage.is_absolute() or not out.is_absolute():
        raise ValueError("absolute instrument/output paths required")
    if any(p.is_symlink() for path in (stage, out) for p in (path, *path.parents)):
        raise ValueError("instrument/output symlink traversal")
    ra_stage.verify(stage)
    argv, fam, _ = command(spec, stage, out, profile=profile)
    component = "training_profile" if profile else "training"
    env, removed = ra_env.clean(os.environ, component=component, fixture={}, venv=Path(spec["venv"]),
                               cache=Path(spec["cache"]), threads=spec["threads"], allocator=spec["allocator"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "native").mkdir()
    (out / "adapters").mkdir()
    (out / "inputs.json").write_text(json.dumps({k: spec[k] for k in ("data", "tokens", "prereg")}, indent=2) + "\n")
    verified = sys.modules.get('ra_verified_worker')
    child_binding, native_out = None, out
    if verified is not None and verified.CURRENT is not None:
        argv, native_out, child_binding = verified.tc1_child(spec, stage, out, profile=profile)
    process = ra_process.run(argv, env=env, cwd=out, log=out / "process.log", receipt=out / "process.json",
                             deadline=spec["deadline_epoch_s"], timeout=spec["timeout_s"])
    # A changed input or instrument during a run is retained as a failed binding.
    native_path = native_out / "native" / f"{fam}_e4b_{component}.json"
    try:
        child_evidence = verified.check_tc1_child(child_binding, process, profile=profile) if child_binding else None
        ra_stage.verify(stage)
        for key in ("data", "tokens", "prereg"):
            ra_process.check_input(spec[key])
        native = json.loads(native_path.read_bytes())
        check_native(native, spec, json.loads((stage / "source-pins.json").read_bytes()), profile=profile)
    except Exception as exc:
        (out / "component.json").write_text(json.dumps({"status": "NATIVE_BINDING_FAILED",
                                                      "error_type": type(exc).__name__,
                                                      "proves_gpu_engagement": False}, indent=2) + "\n")
        raise
    result = {"status": "NATIVE_RECORDED_PENDING_ENGAGEMENT", "component": component,
              "native_path": str(native_path), "native_sha256": ra_process.file_digest(native_path),
              "process": process, "removed_environment_keys": removed, "verified_child_startup": child_evidence, "proves_gpu_engagement": False}
    (out / "component.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--instruments", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    if not args.spec.is_absolute():
        ap.error("absolute spec required")
    execute(json.loads(args.spec.read_bytes()), args.instruments, args.out, profile=args.profile)


if __name__ == "__main__":
    main()
