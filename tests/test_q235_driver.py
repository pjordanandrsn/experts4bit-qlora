# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's box driver (bench/q235/): the sealed closure, the locks, the fetch's content checks, and the runner's refusals.

No network and no GPU: the runner is driven up to its host floors with a stand-in ``nvidia-smi``, which is where a box that is
not the registered class, or a host below the RAM floor, refuses with the launcher's host code before any install or fetch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
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
    boot = _mod("q235_bootstrap", Q / "q235_bootstrap.py")
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


def science_ram_floor_gb():
    import re
    return int(re.search(r"^  science\).*MIN_RAM_GB=(\d+);", (Q / "q235_run.sh").read_text(), re.M).group(1))


def test_the_science_host_ram_floor_is_amendment_5s():
    """Amendment 5 derives the floor from the measured scaling read (gate2/memslope.json): 177.74 GB of anonymous and shared host
    memory at the load end at 94 layers, plus the four host check copies (40.24 GB) and the held layer outputs (2.04 GB), plus the
    18.5 GB allowance: 238.52, so 239."""
    assert science_ram_floor_gb() == 239


def test_the_runner_refuses_a_host_below_the_ram_floor(box):
    w, bin_, env = box
    with open("/proc/meminfo") if os.path.exists("/proc/meminfo") else open(os.devnull) as f:
        total = next((int(ln.split()[1]) for ln in f if ln.startswith("MemTotal:")), None)
    floor = science_ram_floor_gb()
    if total is None or total * 1024 >= floor * 1e9:
        pytest.skip(f"needs a Linux host with less than {floor} GB of RAM")
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


def test_the_host_rate_probe_refuses_below_its_floors_and_without_o_direct_when_required(tmp_path):
    """No CUDA here: the link reads as unmeasured, which a link floor refuses (18); the disk is read back from a fresh file, and
    --require-direct either reads it with O_DIRECT or refuses (13) -- it never falls back to a possibly cached read."""
    probe = [sys.executable, str(Q / "q235_hostrate.py"), "--dir", str(tmp_path), "--disk-gib", "1", "--copies", "1"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(probe + ["--out", str(tmp_path / "a.json")], env=env, capture_output=True, text=True, timeout=300)
    if "No module named 'torch'" in r.stderr:
        pytest.skip("the link probe needs torch")
    rec = json.loads((tmp_path / "a.json").read_text())
    assert r.returncode == 0 and rec["disk_read_gb_s"] > 0 and rec["disk_method"], (r.returncode, r.stderr[-400:])
    r = subprocess.run(probe + ["--out", str(tmp_path / "b.json"), "--min-h2d-gbs", "1"], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 18 and "H2D" in r.stderr
    r = subprocess.run(probe + ["--out", str(tmp_path / "c.json"), "--min-disk-gbs", "1e9"], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 13 and "disk read" in r.stderr
    r = subprocess.run(probe + ["--out", str(tmp_path / "d.json"), "--require-direct"], env=env, capture_output=True, text=True,
                       timeout=300)
    rec = json.loads((tmp_path / "d.json").read_text())
    assert (r.returncode == 0 and rec["disk_method"].startswith("O_DIRECT")) or (r.returncode == 13 and rec["disk_read_gb_s"] is None)
    assert not list(tmp_path.glob(".q235-hostrate.bin"))                 # the probe file never outlives the probe


def _fake_hub(tmp_path):
    """A stand-in huggingface_hub for the fetch (parent and child process): it copies locked files from FAKE_SRC, after FAKE_SLEEP s."""
    pkg = tmp_path / "fakehub" / "huggingface_hub"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "import os, shutil, time\n"
        "def snapshot_download(repo, revision=None, local_dir=None, allow_patterns=None, max_workers=8):\n"
        "    time.sleep(float(os.environ.get('FAKE_SLEEP', '0')))\n"
        "    for n in allow_patterns:\n"
        "        shutil.copyfile(os.path.join(os.environ['FAKE_SRC'], n), os.path.join(str(local_dir), n))\n"
        "def hf_hub_download(*a, **k):\n"
        "    raise RuntimeError('not used')\n")
    return pkg.parent


def _fake_checkpoint(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text('{"a": 1}\n')
    files = [{"file": "config.json", "size": (src / "config.json").stat().st_size, "lfs_sha256": None,
              "blob_id": subprocess.run(["git", "hash-object", str(src / "config.json")], capture_output=True, text=True).stdout.strip()}]
    for i in (1, 2):
        name = f"model-0000{i}-of-00002.safetensors"
        (src / name).write_bytes(os.urandom(1 << 20))
        files.append({"file": name, "size": 1 << 20, "blob_id": "x", "lfs_sha256": hashlib.sha256((src / name).read_bytes()).hexdigest()})
    lock = {"schema": "q235-model-files/1", "repo": "r/m", "revision": "0" * 40, "n_files": len(files), "files": files}
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    return src


def test_the_fetch_times_its_first_slice_and_refuses_a_slow_pull_by_its_deadline(tmp_path):
    src = _fake_checkpoint(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_fake_hub(tmp_path)), "FAKE_SRC": str(src)}
    run = [sys.executable, str(Q / "q235_fetch.py"), "--lock", str(tmp_path / "lock.json"), "--probe-shards", "1", "--workers", "2"]
    dest = tmp_path / "fast"
    dest.mkdir()
    r = subprocess.run(run + ["--dest", str(dest), "--out", str(tmp_path / "fast.json"), "--min-gb-s", "1e-6"], env=env,
                       capture_output=True, text=True, timeout=120)
    rec = json.loads((tmp_path / "fast.json").read_text())
    assert r.returncode == 0 and rec["ok"] and rec["probe"]["finished"] and rec["probe"]["files"] == 2, (r.stdout, r.stderr[-400:])
    dest = tmp_path / "slow"
    dest.mkdir()
    t0 = time.time()
    r = subprocess.run(run + ["--dest", str(dest), "--out", str(tmp_path / "slow.json"), "--min-gb-s", "0.002"],
                       env={**env, "FAKE_SLEEP": "30"}, capture_output=True, text=True, timeout=120)
    rec = json.loads((tmp_path / "slow.json").read_text())
    assert r.returncode == 14 and not rec["probe"]["finished"] and "deadline" in rec["refusals"][0], (r.stdout, r.stderr[-400:])
    assert time.time() - t0 < 25                                          # refused at the ~0.5 s deadline, not after the 30 s pull

# ---- the pre-rental fetch gate (gates 3 and 5) -------------------------------------------------------------------------

def _gate():
    return _mod("q235_fetch_gate", Q / "q235_fetch_gate.py")


def test_the_gate_compares_the_hub_listing_with_the_lock_row_for_row():
    g = _gate()
    lock = {"files": [{"file": "a.json", "size": 3, "blob_id": "b1", "lfs_sha256": None},
                      {"file": "m.safetensors", "size": 9, "blob_id": "b2", "lfs_sha256": "s2"}]}
    assert g.check_against_lock([dict(r) for r in lock["files"]], lock) == []
    listing = [dict(lock["files"][0], size=4), {"file": "extra.txt", "size": 1, "blob_id": "x", "lfs_sha256": None}]
    out = g.check_against_lock(listing, lock)
    assert any("m.safetensors: locked, not listed" in x for x in out) and any("extra.txt: listed" in x for x in out)
    assert any("a.json: size 4" in x for x in out)


def test_the_gate_needs_every_wheel_to_answer_its_one_byte_probe():
    g = _gate()
    rows = [{"name": "a", "url": "https://files.pythonhosted.org/a.whl"}, {"name": "b", "url": "https://files.pythonhosted.org/b.whl"}]
    ok = {r["url"]: {"status": 206, "bytes": 1} for r in rows}
    assert g.check_probes(rows, ok) == []
    bad = {**ok, rows[1]["url"]: {"status": 404, "error": "Not Found"}}
    assert g.check_probes(rows, bad) == ["b: 404 Not Found"]
    assert g.check_probes(rows, {}) and len(g.check_probes(rows, {})) == 2


def test_the_placeholder_scan_finds_unfilled_values_and_the_registration_has_none(tmp_path):
    g = _gate()
    f = tmp_path / "x.sh"
    f.write_text("CAP=TBD\nSHA=<E4B_SHA>\nok=1\nname={{ run_id }}\nnote='a placeholder in prose'\n")
    out = _scan_outside_repo(g, f)
    assert len(out) == 3 and all("unfilled value" in x for x in out), out
    files = g.placeholder_files()
    names = {p.name for p in files}
    assert {"q235_run.sh", "wheels.json", "model_files.json", "q235.sha256", "PREREG-q235.md", "q235_drive.sh"} <= names
    assert g.scan_placeholders(files) == [], "an unfilled value in a file the box runs or the registration"


def _scan_outside_repo(g, path):
    """scan_placeholders reports paths relative to the repository; a fixture outside it is scanned line by line here."""
    return [f"{path.name}:{i}: unfilled value {m.group(0)!r}" for i, line in enumerate(path.read_text().splitlines(), 1)
            for m in g.PLACEHOLDER.finditer(line)]


def test_a_gate_report_admits_only_its_own_commit_passed_and_fresh():
    g = _gate()
    now = 1_000_000.0
    rep = {"schema": g.SCHEMA, "passed": True, "refusals": [], "e4b_sha": "a" * 40, "generated_at": now - 60}
    assert g.report_ok(rep, "a" * 40, now) is None
    assert "not " in g.report_ok(rep, "b" * 40, now)
    assert "old" in g.report_ok({**rep, "generated_at": now - g.MAX_AGE_S - 1}, "a" * 40, now)
    assert "refused" in g.report_ok({**rep, "passed": False, "refusals": ["x"]}, "a" * 40, now)
    assert "schema" in g.report_ok({**rep, "schema": "p130-fetch-gate/1"}, "a" * 40, now)


def test_the_gnf4_pin_is_one_value_in_the_drive_the_gate_and_the_lock():
    g = _gate()
    lock = json.loads((Q / "wheels.json").read_text())
    drive = (Q / "q235_drive.sh").read_text()
    assert lock["git"]["grouped-nf4-gemm"]["sha"] == g.GNF4_SHA and f"GNF4_SHA={g.GNF4_SHA}" in drive


def test_the_drive_refuses_to_launch_without_a_gate_report(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("Q235_", "TC1_", "E4B_"))}
    env.update(TC1_DRIVE_DRYRUN="1", E4B_RENT_RUN_DIR=str(tmp_path))
    r = subprocess.run(["bash", str(Q / "q235_drive.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 78 and "Q235_FETCH_GATE unset" in r.stdout, (r.returncode, r.stdout, r.stderr)
    stale = tmp_path / "gate.json"
    stale.write_text(json.dumps({"schema": "q235-fetch-gate/1", "passed": True, "refusals": [], "e4b_sha": "0" * 40, "generated_at": 0}))
    r = subprocess.run(["bash", str(Q / "q235_drive.sh")], env={**env, "Q235_FETCH_GATE": str(stale)}, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 78 and "does not let" in r.stdout, (r.returncode, r.stdout, r.stderr)


def test_the_gate_probes_wheels_with_the_box_s_own_opener(monkeypatch):
    """A host that answers only the box's client (its User-Agent) must pass the gate, and one that refuses it must fail it."""
    import urllib.request
    g = _gate()
    boot = g._bootstrap()
    box_ua, seen = boot.USER_AGENT, []

    class _Resp:
        status = 206

        def read(self, n=-1):
            return b"x"

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        seen.append(req.get_header("User-agent"))
        if req.get_header("User-agent") != box_ua:                       # this host serves the box's client only
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    rows = [{"name": "a", "url": "https://files.pythonhosted.org/a.whl"}]
    assert g.check_probes(rows, g.probe_wheels(rows, boot)) == [] and seen == [box_ua]
    monkeypatch.setattr(boot, "USER_AGENT", "someone-else/1")
    out = g.check_probes(rows, g.probe_wheels(rows, boot))
    assert len(out) == 1 and "403" in out[0], out


def _smi_leg(bin_, card, apps="", util="0"):
    """A stand-in nvidia-smi for Amendment 11's leg: the card line, the compute apps, and a utilization stream under -l."""
    (bin_ / "nvidia-smi").write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f'  *query-compute-apps*) [ -n "{apps}" ] && echo "{apps}"; exit 0 ;;\n'
        f'  *utilization.gpu*) while :; do echo "{util}"; sleep 1; done ;;\n'
        f"  *) echo '{card}' ;;\n"
        "esac\n")
    (bin_ / "nvidia-smi").chmod(0o755)


PRO6000 = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 575.64.05, 5, 5, 16, 16, 600.00 W"


def test_a_leg_run_id_runs_the_pro6000_mode_never_science(box):
    """Amendment 11: the launcher's run id q235-leg-pro6000-N selects the leg; its card class is the PRO 6000 and it is never a draw."""
    w, bin_, env = box
    _smi_leg(bin_, "NVIDIA GeForce RTX 5090, 32607, 575.64.05, 5, 5, 16, 16, 575.00 W")
    r = _run(w, {**env, "TC1_RUN_ID": "q235-leg-pro6000-1"})
    assert r.returncode == 18 and (w / "REFUSAL").read_text().strip() == "card class", (r.returncode, r.stdout[-800:])
    run = json.loads((w / "receipts" / "run.json").read_text())
    assert run["mode"] == "pro6000" and run["science_eligible"] is False


def test_the_leg_mode_comes_from_the_run_id_only(box):
    w, bin_, env = box
    r = _run(w, {**env, "TC1_RUN_ID": "q235-leg-pro6000-1", "Q235_REHEARSAL": "a2000"})
    assert r.returncode == 78 and "with leg run id" in r.stdout
    r = _run(w, {**env, "Q235_REHEARSAL": "pro6000"})
    assert r.returncode == 78 and "leg run id only" in r.stdout
    _smi_leg(bin_, "NVIDIA GeForce RTX 5090, 32607, 575.64.05, 5, 5, 16, 16, 575.00 W")
    r = _run(w, {**env, "TC1_RUN_ID": "q235-pro6000-4"})            # a science id stays science
    assert json.loads((w / "receipts" / "run.json").read_text())["mode"] == "science"


@pytest.mark.parametrize("apps,util,why", [("4242", "0", "1 compute apps"), ("", "37", "max 37 %")])
def test_the_leg_refuses_a_card_that_is_not_quiet(box, apps, util, why):
    w, bin_, env = box
    _smi_leg(bin_, PRO6000, apps=apps, util=util)
    r = _run(w, {**env, "TC1_RUN_ID": "q235-leg-pro6000-1"})
    assert r.returncode == 18 and (w / "REFUSAL").read_text().strip() == "not quiet" and why in r.stdout, (r.returncode, r.stdout[-600:])
    q = json.loads((w / "receipts" / "quiet.json").read_text())
    assert q["util_samples"] >= 5 and q["nproc"] >= 1 and "loadavg" in q


def test_a_quiet_leg_passes_its_gate_and_records_it(box):
    w, bin_, env = box
    _smi_leg(bin_, PRO6000, util="3")
    r = _run(w, {**env, "TC1_RUN_ID": "q235-leg-pro6000-1"})
    q = json.loads((w / "receipts" / "quiet.json").read_text())
    assert q["compute_apps"] == 0 and q["utilization_gpu_max_pct"] == 3, q
    assert "not quiet" not in r.stdout                               # the next check is the host-RAM floor (239 GB) or beyond


def test_the_leg_takes_a_second_host_rate_sample_after_the_repeat():
    text = (Q / "q235_run.sh").read_text()
    assert text.index("must hostrate2") > text.index("for run in box repeat")
    assert re.search(r"^  pro6000\) CARD_RE='\^NVIDIA RTX PRO 6000'; MIN_VRAM_MIB=94000; MIN_RAM_GB=239;", text, re.M)


def _repeat_cap(tmp_path, box, pinned=""):
    """Amendment 15: the runner's own repeat_cap (the text between its markers), run against a box record in a box workspace."""
    text = (Q / "q235_run.sh").read_text()
    fn = text[text.index("# --- repeat_cap (Amendment 15) ---"):text.index("# --- end repeat_cap ---")]
    margin = re.search(r"^REPEAT_MARGIN=([0-9.]+) ", text, re.M).group(1)
    w = tmp_path / "w"
    (w / "receipts" / "box").mkdir(parents=True)
    (w / "logs").mkdir()
    if box is not None:
        (w / "receipts" / "box" / "box.json").write_text(json.dumps(box))
    pybin = tmp_path / "pybin"
    pybin.mkdir()
    (pybin / "python").symlink_to(sys.executable)
    r = subprocess.run(["bash", "-c", f'{fn}\nREPEAT_MARGIN={margin}; CAP_REPEAT="{pinned}"; repeat_cap'], cwd=w, capture_output=True,
                       text=True, env={**os.environ, "PATH": f"{pybin}:{os.environ['PATH']}"}, timeout=60)
    rec = json.loads((w / "receipts" / "repeat-cap.json").read_text()) if r.returncode == 0 else None
    return r, rec


def test_the_repeat_cap_is_sized_from_the_box_s_own_load_and_step(tmp_path):
    """Amendment 15 on q235-pro6000-6's own box record (load 907.9 s, step 0 46.551 s): 1.5 x 954.451 = 1,431.68 -> 1,432 s. That
    box's repeat met Amendment 12's leg-scaled 705 s cap inside its 661 s expert fuse."""
    r, rec = _repeat_cap(tmp_path, {"load": {"seconds": 907.9}, "step": {"elapsed_s": 46.551}})
    assert r.returncode == 0 and r.stdout.strip() == "1432", (r.stdout, r.stderr)
    assert rec == {"amendment": 15, "box_load_s": 907.9, "box_step_s": 46.551, "margin": 1.5, "derived_cap_s": 1432}


def test_the_repeat_cap_has_a_60_s_floor_and_a_rehearsal_pin_is_recorded_beside_it(tmp_path):
    r, rec = _repeat_cap(tmp_path / "a", {"load": {"seconds": 0.2}, "step": {"elapsed_s": 0.05}})       # a CPU rehearsal's tiny model
    assert r.stdout.strip() == "60" and rec["derived_cap_s"] == 60 and "rehearsal_cap_s" not in rec
    r, rec = _repeat_cap(tmp_path / "b", {"load": {"seconds": 0.2}, "step": {"elapsed_s": 0.05}}, pinned="900")
    assert r.stdout.strip() == "900" and rec["derived_cap_s"] == 60 and rec["rehearsal_cap_s"] == 900


@pytest.mark.parametrize("box", [None, {"load": {}, "step": {"elapsed_s": 46.5}}, {"load": {"seconds": 907.9}},
                                 {"load": {"seconds": 907.9}, "step": {"elapsed_s": 0}},
                                 {"load": {"seconds": float("nan")}, "step": {"elapsed_s": 46.5}}])
def test_a_box_record_that_cannot_size_the_repeat_fails_the_cap(tmp_path, box):
    r, rec = _repeat_cap(tmp_path, box)
    assert r.returncode != 0 and r.stdout.strip() == "" and rec is None, (r.returncode, r.stdout)


def test_the_runner_sizes_the_repeat_on_the_box_and_voids_when_it_cannot():
    """Amendment 15's wiring: no sealed repeat cap; the repeat's cap is repeat_cap's, taken after the box phase and bounded by the time
    left like every phase (phase -> budget_cap); an unsized repeat is a VOID (11) before the repeat starts; only a rehearsal pins it."""
    text = (Q / "q235_run.sh").read_text()
    assert not re.search(r"\bCAP_REPEAT=\d", text) and re.search(r'^REPEAT_MARGIN=1\.5 CAP_REPEAT=""$', text, re.M)
    loop = text[text.index("for run in box repeat; do"):text.index("if [ \"$MODE\" = pro6000 ]; then   # Amendment 11: the second")]
    assert 'cap=$(repeat_cap) && [ -n "$cap" ] || {' in loop and "Amendment 15)'; finish 11; }" in loop
    assert loop.index("cap=$(repeat_cap)") < loop.index('phase "$run" "$cap" python q235_box.py')
    rehearsal = text[text.index('if [ "$MODE" != science ]; then'):text.index("must require-git")]
    assert "CAP_REPEAT=${Q235_CAP_REPEAT:-}" in rehearsal
    assert text.count("Q235_CAP_REPEAT") == 1
