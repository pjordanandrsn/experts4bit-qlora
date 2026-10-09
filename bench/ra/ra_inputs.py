#!/usr/bin/env python3
"""Common input bytes and prepared-fixture projections, without release imports.

A reviewed lock is an expected inventory, not download or launch authority.
This gate cannot establish tokenizer regeneration, runtime consumption or GPU
engagement. Arenas and caches are per-release outputs and are excluded.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import stat
from pathlib import Path

import ra_stage

REGISTERED_PINS = Path(__file__).with_name("source-pins.json")
MODELS = {"proof": ("granite", "ibm-granite/granite-3.1-3b-a800m-instruct"),
          "reading": ("qwen3", "Qwen/Qwen3-30B-A3B")}
TREES = {"checkpoint", "datasets", "calibration_inputs"}
FILES = {"train_data", "train_tokens", "corpus_tokens", "decode_prompts", "quality_windows"}
SPEC = {"schema", "battery", "trees", "files"}
POINTS = (("warm", "serial", 0, 4, 999), ("burst", "poisson", 1000, 64, 998),
          ("end", "poisson", 12, 120, 112))


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def absolute(path):
    p = Path(path)
    if not p.is_absolute() or p != Path(os.path.normpath(p)) or any(q.is_symlink() for q in (p, *p.parents)):
        raise ValueError("absolute input without traversal/symlinks required")
    return p


def file_record(path):
    p = absolute(path)
    if not stat.S_ISREG(p.lstat().st_mode):
        raise ValueError("regular input file required")
    h = hashlib.sha256()
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), "rb") as src:
        before = os.fstat(src.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("regular input file required")
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
        after = os.fstat(src.fileno())
    def signature(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    if signature(before) != signature(after) or signature(after) != signature(p.stat()):
        raise ValueError("input changed while hashing")
    return {"sha256": h.hexdigest(), "size": after.st_size}


def inventory(root):
    root = absolute(root)
    if not root.is_dir():
        raise ValueError("input tree required")
    out = {}
    def failed(exc):
        raise exc

    for base, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        for name in dirs:
            p = absolute(Path(base) / name)
            if not p.is_dir():
                raise ValueError("regular input directory required")
        for name in files:
            p = Path(base) / name
            out[p.relative_to(root).as_posix()] = file_record(p)
    if not out:
        raise ValueError("nonempty input tree required")
    return dict(sorted(out.items()))


def unique(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate input JSON key")
        out[k] = v
    return out


def read_json(path):
    p = absolute(path)
    before = file_record(p)
    if before["size"] > 128 * 1024 * 1024:
        raise ValueError("input JSON too large")
    value = json.loads(p.read_bytes(), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite input JSON")))
    if file_record(p) != before:
        raise ValueError("input changed while reading")
    return value


def tokens(rows, count=None, width=None):
    if not isinstance(rows, list) or not rows or (count is not None and len(rows) != count):
        raise ValueError("input row count")
    if any(not isinstance(r, list) or not r or (width is not None and len(r) != width) or
           any(type(t) is not int or t < 0 for t in r) for r in rows):
        raise ValueError("input token shape/type")


def plans(stage):
    """Execute only the registered stdlib plan function, with its own RNG."""
    src = ast.parse((stage / "sc2_driver.py").read_bytes())
    nodes = [n for n in src.body if isinstance(n, ast.FunctionDef) and n.name == "plan"]
    if len(nodes) != 1:
        raise ValueError("registered plan function absent/ambiguous")
    scope = {"random": random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "registered-sc2-plan", "exec"), scope)
    return {name: [list(r) for r in scope["plan"](mode, rate, n, seed, 16, 64, 256)]
            for name, mode, rate, n, seed in POINTS}


def inspect(spec, stage):
    stage = absolute(stage)
    ra_stage.verify(stage)
    if set(spec) != SPEC or type(spec["schema"]) is not int or spec["schema"] != 1 or spec["battery"] not in MODELS:
        raise ValueError("input spec schema/battery")
    if set(spec["trees"]) != TREES or set(spec["files"]) != FILES:
        raise ValueError("input roles differ")
    roots = {k: absolute(v) for k, v in spec["trees"].items()}
    paths = {k: absolute(v) for k, v in spec["files"].items()}
    if len(set(paths.values())) != len(paths) or any(a == b or a in b.parents or b in a.parents
            for i, a in enumerate(roots.values()) for b in list(roots.values())[i + 1:]):
        raise ValueError("distinct nonoverlapping input roots/files required")
    if any(p == r or r in p.parents or p in r.parents for p in paths.values() for r in roots.values()):
        raise ValueError("prepared files must be separate from input trees")
    trees = {k: inventory(v) for k, v in roots.items()}
    files = {k: file_record(v) for k, v in paths.items()}
    pins = read_json(stage / "source-pins.json")
    if file_record(stage / "source-pins.json") != file_record(REGISTERED_PINS):
        raise ValueError("stage pins differ from registered RA pins")
    fam, model = MODELS[spec["battery"]]
    revision = pins["models"][model]
    train = read_json(paths["train_tokens"])
    if (train.get("fam"), train.get("tokenizer"), train.get("revision"), train.get("dataset_sha256"),
        train.get("template"), train.get("seq")) != (fam, model, revision, files["train_data"]["sha256"], "alpaca", 2048) \
            or train.get("pack", False) is not False:
        raise ValueError("TC1 prepared identity")
    tokens(train["train"])
    tokens(train["eval"])
    if len(train["eval"]) < 8 or any(not 8 <= len(r) <= 2048 for r in train["train"] + train["eval"]):
        raise ValueError("TC1 prepared lengths")
    training_sha = digest({"train": train["train"], "eval": train["eval"]})
    if train.get("sha256") != training_sha:
        raise ValueError("TC1 compact payload digest")
    corpus = read_json(paths["corpus_tokens"])
    if set(corpus) != {"model", "revision", "dataset", "ids"} or \
            (corpus["model"], corpus["revision"], corpus["dataset"]) != (model, revision, "wikitext-2-raw-v1:test"):
        raise ValueError("retained corpus token identity")
    tokens([corpus["ids"]])
    ids = corpus["ids"]
    expected_rows = [ids[k * 4096:k * 4096 + 512] for k in range(16)]
    tokens(expected_rows, 16, 512)
    if len({tuple(r) for r in expected_rows}) != 16:
        raise ValueError("distinct decode rows required")
    prompts = read_json(paths["decode_prompts"])
    if set(prompts) != {"model", "revision", "rows", "prompts_sha256", "rows_sha256", "offset", "prompt"} or \
            (prompts["model"], prompts["revision"], prompts["offset"], prompts["prompt"]) != (model, revision, 4096, 512) \
            or prompts["rows"] != expected_rows or prompts["prompts_sha256"] != digest(expected_rows) or \
            prompts["rows_sha256"] != [digest(r) for r in expected_rows]:
        raise ValueError("P109 prepared projection")
    cont = 32 if spec["battery"] == "proof" else 128
    windows = read_json(paths["quality_windows"])
    expected_windows = [ids[k * 4096:k * 4096 + 512 + cont] for k in range(12)]
    tokens(expected_windows, 12, 512 + cont)
    if set(windows) != {"wikitext"} or windows["wikitext"] != expected_windows:
        raise ValueError("P115 prepared projection")
    request_plans = plans(stage)
    projection = {"training_payload_sha256": training_sha, "decode_prompts_sha256": digest(expected_rows),
                  # Native P115 hashes default JSON spacing, unlike TC1/P109.
                  "quality_windows_sha256": hashlib.sha256(json.dumps(expected_windows).encode()).hexdigest(),
                  "quality_window_ids": [f"wikitext:{k}" for k in range(12)], "capacity_plans": request_plans}
    # Repeat complete inventory after all parsing/native helper execution.
    if files != {k: file_record(v) for k, v in paths.items()} or trees != {k: inventory(v) for k, v in roots.items()}:
        raise ValueError("input drift during projection")
    ra_stage.verify(stage)
    return {"schema": 1, "battery": spec["battery"], "model": model, "revision": revision,
            "registered_source": pins["source_commit"], "trees": trees, "files": files, "projection": projection}


def verify(spec, stage, lock_path, lock_sha256):
    if file_record(lock_path)["sha256"] != lock_sha256:
        raise ValueError("reviewed input lock bytes differ")
    expected = read_json(lock_path)
    observed = inspect(spec, stage)
    if identity(observed) != identity(expected) or file_record(lock_path)["sha256"] != lock_sha256:
        raise ValueError("common inputs differ from reviewed lock")
    return {"status": "INPUT_BYTES_AND_PROJECTIONS_MATCH", "common_identity": identity(observed),
            "input_lock_sha256": lock_sha256, "runtime_consumption_verified": False,
            "tokenizer_regeneration_verified": False, "gpu_verified": False, "launch_authority": False}


def bind_native(lock, native, component, *, server_evidence=None):
    """Receipt projection equality only; retain the original native record.

    Call after verify and before envelope normalization, in addition to the
    component's own complete shape/default/engagement checks. This alone does
    not prove which file a release loader opened.
    """
    capacity = component in ("warm", "burst", "end")
    config_identity = server_evidence.get("config", {}) if capacity and isinstance(server_evidence, dict) else native
    if (config_identity.get("model"), config_identity.get("revision")) != (lock["model"], lock["revision"]):
        raise ValueError("native model/input revision differs")
    projection = lock["projection"]
    if component in ("training", "training_profile"):
        if native["tokens"]["sha256"] != projection["training_payload_sha256"] or \
                native["tokens"]["pack"] is not False or native["tokens"]["eval_rows_used"] != 8:
            raise ValueError("native TC1 input projection differs")
    elif component == "decode":
        if native["prompts_sha256"] != projection["decode_prompts_sha256"]:
            raise ValueError("native decode input projection differs")
    elif component in ("warm", "burst", "end"):
        if native.get("model") != lock["model"] or native["prompts_sha256"] != projection["decode_prompts_sha256"] or \
                identity(native["plan"]) != identity(projection["capacity_plans"][component]):
            raise ValueError("native capacity input projection differs")
    elif component == "quality":
        if native["windows_sha256"] != {"wikitext": projection["quality_windows_sha256"]}:
            raise ValueError("native quality input projection differs")
    else:
        raise ValueError("unknown native input component")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("candidate", "verify"))
    p.add_argument("--spec", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--lock-sha256")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    spec = read_json(a.spec)
    if a.mode == "candidate":
        if a.lock_sha256 or absolute(a.lock) != absolute(a.out):
            p.error("candidate requires --out equal --lock, without --lock-sha256")
        record = inspect(spec, a.stage)
    else:
        if not a.lock_sha256:
            p.error("verify requires reviewed --lock-sha256")
        record = verify(spec, a.stage, a.lock, a.lock_sha256)
    with absolute(a.out).open("x") as dst:
        dst.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
