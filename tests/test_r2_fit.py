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
