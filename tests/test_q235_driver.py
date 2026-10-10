# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's box driver (bench/q235/): the sealed closure, the locks, the fetch's content checks, and the runner's refusals.

No network and no GPU: the runner is driven up to its host floors with a stand-in ``nvidia-smi``, which is where a box that is
not the registered class, or a host below the RAM floor, refuses with the launcher's host code before any install or fetch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "bench" / "q235"


def _mod(name, path):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(path.parent))


def test_the_committed_manifest_is_the_closure():
    stage = _mod("q235_stage", Q / "q235_stage.py")
    want = {}
    for line in (Q / "q235.sha256").read_text().splitlines():
        digest, name = line.split()
        want[name] = digest
    for p in stage.CLOSURE:
        assert want[Path(p).name] == stage.sha(ROOT / p), f"{p} changed: rewrite q235.sha256 (q235_stage.py --write) after review"
    lock = json.loads((Q / "locked_inputs.json").read_text())["assets"]
    assert set(want) == {Path(p).name for p in stage.CLOSURE} | set(lock)


def test_staging_refuses_an_asset_that_is_not_the_locked_bytes(tmp_path):
    stage = _mod("q235_stage", Q / "q235_stage.py")
    (tmp_path / "ds_alpaca.json").write_text("{}")
    with pytest.raises(ValueError, match="asset ds_alpaca.json"):
        stage.stage(tmp_path)
    with pytest.raises(ValueError, match="asset ds_alpaca.json"):
        stage.stage(tmp_path / "absent")


def test_the_wheel_lock_is_complete_hashed_and_from_allowlisted_origins():
    sys.path.insert(0, str(ROOT / "bench" / "dq11"))
    try:
        boot = _mod("q235_bootstrap", Q / "q235_bootstrap.py")
    finally:
        sys.path.remove(str(ROOT / "bench" / "dq11"))
    lock = json.loads((Q / "wheels.json").read_text())
    rows = boot.check_lock(lock)
    versions = {r["name"]: r["version"] for r in rows}
    assert versions["torch"] == "2.11.0+cu128" and versions["transformers"] == "5.19.0" and versions["bitsandbytes"] == "0.50.2"
    bad = json.loads(json.dumps(lock))
    bad["packages"][0]["sha256"] = ""
    with pytest.raises(ValueError, match="no sha256"):
        boot.check_lock(bad)
    bad = json.loads(json.dumps(lock))
    bad["packages"][0]["url"] = "https://example.com/x.whl"
    with pytest.raises(ValueError, match="origin"):
        boot.check_lock(bad)
    bad = json.loads(json.dumps(lock))
    bad["packages"].append(dict(bad["packages"][0]))
    with pytest.raises(ValueError, match="repeated"):
        boot.check_lock(bad)


def test_the_model_lock_is_the_registered_revision():
    lock = json.loads((Q / "model_files.json").read_text())
    files = lock["files"]
    assert lock["revision"] == "ac9c66cc9b46af7306746a9250f23d47083d689e" and len(files) == lock["n_files"] == 129
    shards = [f for f in files if f["file"].endswith(".safetensors")]
    assert len(shards) == 118 and all(f["lfs_sha256"] for f in shards)
    assert sum(f["size"] for f in shards) == lock["safetensors_bytes"] == 470191875040
    assert all(f["blob_id"] for f in files) and "model.safetensors.index.json" in {f["file"] for f in files}


def test_fetch_checks_content_by_sha256_or_git_blob_id(tmp_path):
    fetch = _mod("q235_fetch", Q / "q235_fetch.py")
    small, big = tmp_path / "config.json", tmp_path / "model-00001-of-00002.safetensors"
    small.write_text('{"a": 1}\n')
    big.write_bytes(os.urandom(4096))
    blob = subprocess.run(["git", "hash-object", str(small)], capture_output=True, text=True).stdout.strip()
    lock = {"files": [{"file": "config.json", "size": small.stat().st_size, "blob_id": blob, "lfs_sha256": None},
                      {"file": big.name, "size": 4096, "blob_id": "x", "lfs_sha256": hashlib.sha256(big.read_bytes()).hexdigest()}]}
    assert fetch.verify(lock, tmp_path, ["config.json", big.name], 2) == []
    big.write_bytes(os.urandom(4096))
    (tmp_path / "config.json").write_text('{"a": 2}\n')
    out = fetch.verify(lock, tmp_path, ["config.json", big.name], 2)
    assert len(out) == 2 and all("content" in x for x in out), out
    assert "absent" in fetch.verify(lock, tmp_path / "none", ["config.json"], 1)[0]


def test_a_slice_fetches_the_shards_its_layers_and_the_non_layer_tensors_need():
    fetch = _mod("q235_fetch", Q / "q235_fetch.py")
    index = {"weight_map": {"model.embed_tokens.weight": "s1", "model.layers.0.a": "s1", "model.layers.1.a": "s2",
                            "model.layers.2.a": "s3", "model.norm.weight": "s4", "lm_head.weight": "s4"}}
    lock = {"files": [{"file": n} for n in ("config.json", "s1.safetensors", "s2.safetensors", "s3.safetensors", "s4.safetensors")]}
    index = {"weight_map": {k: v + ".safetensors" for k, v in index["weight_map"].items()}}
    assert fetch.slice_files(lock, index, 2) == ["config.json", "s1.safetensors", "s2.safetensors", "s4.safetensors"]


@pytest.fixture
def box(tmp_path):
    """A box workspace with the sealed closure staged flat, a stand-in nvidia-smi, and the TC1 lifecycle's variables."""
    stage = _mod("q235_stage", Q / "q235_stage.py")
    w = tmp_path / "w"
    w.mkdir()
    for p in stage.CLOSURE:
        shutil.copy(ROOT / p, w / Path(p).name)
    lock = json.loads((Q / "locked_inputs.json").read_text())["assets"]
    lines = [ln for ln in (Q / "q235.sha256").read_text().splitlines() if ln.split()[1] not in lock]
    (w / "q235.sha256").write_text("\n".join(lines) + "\n")          # the closure without the 1 MB asset (not needed to the floors)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("Q235_", "TC1_"))}
    env.update(PATH=f"{bin_}:{env['PATH']}", Q235_W=str(w), TC1_RUN_NONCE="n0nce", E4B_SHA="a" * 40, GNF4_SHA="b" * 40,
               TC1_DEADLINE_EPOCH=str(int(__import__("time").time()) + 3600))
    return w, bin_, env


def _smi(bin_, line):
    (bin_ / "nvidia-smi").write_text(f"#!/bin/sh\necho '{line}'\n")
    (bin_ / "nvidia-smi").chmod(0o755)


def _run(w, env):
    return subprocess.run(["bash", str(w / "q235_run.sh")], env=env, capture_output=True, text=True, timeout=120)


def test_the_runner_refuses_a_card_that_is_not_the_registered_class(box):
    w, bin_, env = box
    _smi(bin_, "NVIDIA GeForce RTX 5090, 32607, 575.64.05, 5, 5, 16, 16, 575.00 W")
    env["Q235_CARD_RE"] = "RTX 5090"                                       # a science box takes no override
    r = _run(w, env)
    assert r.returncode == 18, (r.returncode, r.stdout[-800:], r.stderr[-800:])
    assert (w / "REFUSAL").read_text().strip() == "card class" and (w / "TC1_EXIT_CODE.n0nce").read_text().strip() == "18"
    assert (w / "TP_DONE.n0nce").exists() and not (w / "TC1_SUCCESS.n0nce").exists()
    timing = [json.loads(ln) for ln in (w / "receipts" / "phase-timings.jsonl").read_text().splitlines()]
    assert timing[0]["phase"] == "require-git" and timing[0]["rc"] == 0
    assert json.loads((w / "receipts" / "run.json").read_text())["science_eligible"] is True


def test_the_runner_refuses_a_host_below_the_ram_floor(box):
    w, bin_, env = box
    with open("/proc/meminfo") if os.path.exists("/proc/meminfo") else open(os.devnull) as f:
        total = next((int(ln.split()[1]) for ln in f if ln.startswith("MemTotal:")), None)
    if total is None or total * 1024 >= 192e9:
        pytest.skip("needs a Linux host with less than 192 GB of RAM")
    _smi(bin_, "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 575.64.05, 5, 5, 16, 16, 600.00 W")
    r = _run(w, env)
    assert r.returncode == 18 and (w / "REFUSAL").read_text().strip() == "host ram floor", (r.returncode, r.stdout[-800:])


def test_the_runner_refuses_an_unknown_mode_and_missing_pins(box):
    w, bin_, env = box
    r = _run(w, {**env, "Q235_REHEARSAL": "paid"})
    assert r.returncode == 78
    r = _run(w, {k: v for k, v in env.items() if k != "GNF4_SHA"})
    assert r.returncode == 78 and "GNF4_SHA unset" in r.stdout


def test_the_runner_voids_a_closure_that_does_not_match_its_manifest(box):
    w, bin_, env = box
    (w / "q235_box.py").write_text("# changed\n")
    r = _run(w, env)
    assert r.returncode == 9 and "q235.sha256" in r.stdout
