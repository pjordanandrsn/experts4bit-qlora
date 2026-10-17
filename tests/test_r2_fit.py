# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2's time-left fit (``bench/fam235/r2_fit.py``): Q235's fit run unchanged with R2's depth, checkpoint bytes, lock and runner.

- The checkpoint's tensor bytes from its config equal the index's ``total_size`` for all 56 layers.
- The fit reads R2's lock and runner, and Q235's wheels, each where ``r2_drift.WRAPPED`` classifies it.
- The box and the repeat scale by 56 / 4; the fixed phases do not."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "bench" / "fam235"
Q = ROOT / "bench" / "q235"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_checkpoint_bytes_are_the_index_s_total():
    r2 = _mod("r2_fit", F / "r2_fit.py")
    assert r2.LAYERS * r2.LAYER_BYTES + r2.OUTSIDE_BYTES == 281_260_142_592          # model.safetensors.index.json at cc88a6cc
    lock = json.loads((F / "model_files.json").read_text())
    header_bytes = lock["safetensors_bytes"] - 281_260_142_592
    assert 0 < header_bytes < 59 * 8192                                                 # the 59 shards' JSON headers, nothing else


def test_the_fit_reads_each_file_where_the_drift_check_classifies_it():
    drift = _mod("r2_drift", F / "r2_drift.py")
    g = _mod("r2_fit", F / "r2_fit.py").fit_module()
    assert (g.LAYERS, g.SLICE_LAYERS) == (56, 4)
    for name, where in drift.WRAPPED["bench/q235/q235_fit.py"]["names"].items():
        assert (g.HERE / name).parent == {"r2": F, "q235": Q}[where], name
    assert g.HERE / "q235_run.sh" == F / "r2_run.sh"
    assert g.sealed_caps(g.HERE / "q235_run.sh") == g.sealed_caps(F / "r2_run.sh") != {}


def test_the_box_and_the_repeat_scale_by_the_depth_and_the_fixed_phases_do_not():
    g = _mod("r2_fit", F / "r2_fit.py").fit_module()
    timings = {"require-git": 1, "create-venv": 5, "tokens": 10, "hostrate": 30, "box": 100, "repeat": 50}
    fetch = {"bytes": 10 ** 9, "hash_seconds": 1.0}
    rep = g.fit(timings, 100.0, fetch, 10 ** 9, 281_263_818_738, 1000.0, 1.0, g.sealed_caps(F / "r2_run.sh"))
    assert rep["layers"] == 56 and rep["slice_layers"] == 4
    assert rep["phases"]["box"]["scaled_s"] == 1400.0 and rep["phases"]["repeat"]["scaled_s"] == 700.0
    assert rep["phases"]["tokens"]["scaled_s"] == 10.0 and rep["phases"]["hostrate"]["scaled_s"] == 30.0


def _profile(read_s, read_bytes, assemble=24.0, offload=5.3):
    return {"read": read_s, "read_bytes": read_bytes, "assemble": assemble, "offload": offload}


def test_a_warm_read_is_slowed_to_the_disk_floor_and_the_floor_wins():
    r2 = _mod("r2_fit", F / "r2_fit.py")
    host = {"h2d_gb_s": 6.12, "host_copy_gb_s": 7.23}
    warm = r2.slowed(112.0, _profile(0.71, 20.84e9), host, 0)               # 29 GB/s: the page cache, not the disk
    assert abs(warm["read_s"] - (20.84e9 / 0.743e9 - 0.71)) < 0.01 and warm["slowed_s"] > 112.0
    slow = r2.slowed(112.0, _profile(60.0, 20.84e9), host, 0)               # slower than the floor already: no read added
    assert slow["read_s"] == 0.0


def test_a_slowed_time_is_never_below_its_measurement():
    r2 = _mod("r2_fit", F / "r2_fit.py")
    slow_host = {"h2d_gb_s": 1.0, "host_copy_gb_s": 1.0}                   # a rehearsal host slower than every floor
    out = r2.slowed(100.0, _profile(100.0, 1e9), slow_host, 9.5e9)
    assert out["slowed_s"] == 100.0 and out["host_copy_s"] == 0.0 and out["staging_s"] == 0.0
    assert out["with_spread_s"] == round(100.0 * r2.SPREAD, 2)


def test_only_the_box_and_the_repeat_are_slowed_and_spread():
    r2 = _mod("r2_fit", F / "r2_fit.py")
    host = {"h2d_gb_s": 6.12, "host_copy_gb_s": 7.23}
    box = {"load": {"profile": _profile(7.7, 20.84e9)}, "step": {"h2d_bytes_offload": 9.51e9}, "step1": {"h2d_bytes_offload": 9.51e9}}
    repeat = {"load": {"profile": _profile(0.71, 20.84e9, 12.1, 4.9)}}
    t = {"require-git": 0, "tokens": 7, "box": 112, "repeat": 46}
    out, rec = r2.cold_timings(t, box, repeat, host)
    assert out["tokens"] == 7 and out["require-git"] == 0
    assert out["box"] == rec["box"]["with_spread_s"] > 112 * r2.SPREAD and out["repeat"] == rec["repeat"]["with_spread_s"] > 46 * r2.SPREAD


def test_the_pull_floor_at_r2_s_fetch_cap():
    """The leg's cold fetch (25.25 GB, hashed in 19.3 s) and a 7,080 s fetch cap: the checkpoint must pull at 0.063 GB/s or more."""
    g = _mod("r2_fit", F / "r2_fit.py").fit_module()
    fetch = {"bytes": 25_245_419_050, "hash_seconds": 19.3}
    assert g.fetch_floor(281_263_818_738, fetch, 7080) == 0.063
