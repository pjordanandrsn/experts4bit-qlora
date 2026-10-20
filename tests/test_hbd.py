"""bench/hbd (PREREG-hbd.md): the reducer's registered outcome table, read in its order, on cells in the arm's receipt format."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HBD = REPO / "bench" / "hbd"
CELLS = ("E0a", "E0b", "E2", "E1", "U0a", "U0b", "U1", "U2")
LIB12 = {"path": "/v/bitsandbytes/libbitsandbytes_cuda128.so", "sha256": "1" * 64}
LIB13 = {"path": "/v/bitsandbytes/libbitsandbytes_cuda130.so", "sha256": "3" * 64}
SLOTS = {"0|gate_up|0": "a" * 64, "0|down|0": "b" * 64, "0|q|-1": "c" * 64, "-1|embed|-1": "d" * 64}
DQ = {"0|gate_up|0": [8, 1, 2], "0|down|0": [8, 3, 4], "0|q|-1": [8, 5, 6], "-1|embed|-1": [8, 7, 8]}


def _red():
    spec = importlib.util.spec_from_file_location("hbd_reduce", HBD / "hbd_reduce.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cells():
    """Eight cells that read SAME everywhere: the t28 cells on the cuda128 library, the cu130 cells on the cuda130 one."""
    out = {}
    for c in CELLS:
        cu130 = c in ("E1", "U1", "U2")
        out[c] = {"status": "ok", "digest": "f" * 64, "stored": "e" * 64, "n_slots": len(SLOTS), "keys": "k" * 64,
                  "lib": LIB13 if cu130 else LIB12, "probe": LIB13 if cu130 else LIB12, "per_slot": dict(SLOTS), "per_slot_dequant": dict(DQ),
                  "versions": {"torch": "2.12.1+cu130" if cu130 else "2.8.0+cu128",
                               "transformers": "5.18.0" if c in ("E0a", "E0b") else "5.5.0"}}
    return out


def _write(tmp, cells):
    spec = {"cells": {}}
    (tmp / "hbd").mkdir(exist_ok=True)
    for c, v in cells.items():
        if v is None:
            continue
        hp1 = {"train_base": {"digest": v["digest"], "n_slots": v["n_slots"]}}
        if v.get("train_base_error"):
            hp1["train_base_error"] = v["train_base_error"]
        hp1["hbd"] = {"stored_digest": v["stored"], "n_slots": v["n_slots"], "keys_sha256": v["keys"], "versions": v["versions"],
                      "bnb": {"libraries": v["lib"] if isinstance(v["lib"], list) else [v["lib"]]}}
        (tmp / f"r_{c}.json").write_text(json.dumps({"status": v["status"], "hp1": hp1}))
        (tmp / "hbd" / f"rec_{c}.json").write_text(json.dumps({"per_slot": v["per_slot"], "per_slot_dequant": v["per_slot_dequant"]}))
        (tmp / "hbd" / f"probe_{c}.json").write_text(json.dumps({"libraries": v["probe"] if isinstance(v["probe"], list) else [v["probe"]]}))
        spec["cells"][c] = {"receipt": f"r_{c}.json", "record": f"hbd/rec_{c}.json", "probe": f"hbd/probe_{c}.json"}
    (tmp / "spec.json").write_text(json.dumps(spec))
    return _red().reduce(spec, str(tmp))


def test_all_cells_equal_read_same_for_both_frameworks_and_say_nothing_about_depth(tmp_path):
    out = _write(tmp_path, _cells())
    assert out["outcome"] == "READ" and out["summary"] == {"e4b": "SAME", "unsloth": "SAME"}
    assert "says nothing about depth" in out["depth_note"] and out["claim"].startswith("none")
    assert out["build"]["e4b"]["pair"] == ["E2", "E1"] and out["build"]["unsloth"]["pair"] == ["U0a", "U1"]


def test_a_repeat_that_differs_voids_before_any_build_comparison(tmp_path):
    for a in ("E0b", "U0b"):
        cells = _cells()
        cells[a]["digest"] = "0" * 64
        out = _write(tmp_path, cells)
        assert out["outcome"] == "VOID" and any("determinism" in r for r in out["reasons"]) and "build" not in out


def test_a_build_pair_on_the_same_library_file_is_void(tmp_path):
    cells = _cells()
    cells["E1"]["lib"] = cells["E1"]["probe"] = LIB12
    out = _write(tmp_path, cells)
    assert out["outcome"] == "VOID" and any("same libbitsandbytes" in r for r in out["reasons"])


def test_a_probe_that_disagrees_with_the_cell_or_two_libraries_in_one_cell_is_void(tmp_path):
    cells = _cells()
    cells["U1"]["probe"] = LIB12
    assert any("sibling probe disagrees" in r for r in _write(tmp_path, cells)["reasons"])
    cells = _cells()
    cells["E2"]["lib"] = [LIB12, LIB13]
    assert any("2 libbitsandbytes files" in r for r in _write(tmp_path, cells)["reasons"])


def test_a_missing_or_failed_cell_is_void(tmp_path):
    cells = _cells()
    cells["U2"] = None
    assert any("U2: missing" in r for r in _write(tmp_path, cells)["reasons"])
    cells = _cells()
    cells["E1"]["train_base_error"] = "OutOfMemoryError: CUDA out of memory"
    assert any("E1: train_base_error" in r for r in _write(tmp_path, cells)["reasons"])


def test_a_different_slot_set_is_structure_not_a_value_change(tmp_path):
    cells = _cells()
    cells["E1"].update(n_slots=5, keys="z" * 64, per_slot=dict(SLOTS, **{"0|router|-1": "r" * 64}), digest="9" * 64)
    out = _write(tmp_path, cells)
    assert out["build"]["e4b"]["reading"] == "STRUCTURE" and out["build"]["e4b"]["only_in"][1] == ["0|router|-1"]


def test_different_stored_bytes_read_different_quantized_and_name_the_slot_kind(tmp_path):
    cells = _cells()
    cells["U1"].update(stored="7" * 64, digest="8" * 64, per_slot=dict(SLOTS, **{"0|gate_up|0": "x" * 64}))
    cells["U2"].update(stored="7" * 64, digest="8" * 64, per_slot=dict(SLOTS, **{"0|gate_up|0": "x" * 64}))
    out = _write(tmp_path, cells)
    r = out["build"]["unsloth"]
    assert r["reading"] == "DIFFERENT_QUANTIZED" and r["stored_slots_differ_by_kind"] == {"experts": 1}
    assert out["summary"] == {"e4b": "SAME", "unsloth": "DIFFERENT_QUANTIZED"} and "build effect" in out["depth_note"]


def test_same_bytes_decoded_differently_read_different_dequant_and_localize_by_the_dequantized_fingerprints(tmp_path):
    cells = _cells()
    cells["E1"].update(digest="6" * 64, per_slot_dequant=dict(DQ, **{"0|q|-1": [8, 5, 9]}))
    r = _write(tmp_path, cells)["build"]["e4b"]
    assert r["reading"] == "DIFFERENT_DEQUANT" and r["dequant_equal"] is False and r["stored_slots_differ_by_kind"] == {}
    assert r["dequant_slots_differ_by_kind"] == {"attention": 1}


def test_cross_framework_pairs_read_on_the_dequantized_digest_only(tmp_path):
    """e4b's q/k/v are nf4/64 and Unsloth's nf4/64+dq, so their stored digests differ by construction: a cross-framework pair
    with equal dequantized digests (HPN's finding) reads SAME_DEQUANT, not DIFFERENT_QUANTIZED."""
    cells = _cells()
    for c in ("U0a", "U0b", "U1", "U2"):
        cells[c].update(stored="5" * 64, per_slot=dict(SLOTS, **{"0|q|-1": "u" * 64}))
    out = _write(tmp_path, cells)
    cross = [r for r in out["reported"] if r.get("read_on") == "dequantized digest"]
    assert [r["pair"] for r in cross] == [["E0a", "U0a"], ["E1", "U1"]] and {r["reading"] for r in cross} == {"SAME_DEQUANT"}
    cells["U1"].update(digest="4" * 64, per_slot_dequant=dict(DQ, **{"0|down|0": [8, 3, 0]}))
    cells["U2"].update(digest="4" * 64)
    r = [x for x in _write(tmp_path, cells)["reported"] if x["pair"] == ["E1", "U1"]][0]
    assert r["reading"] == "DIFFERENT_DEQUANT" and r["dequant_slots_differ_by_kind"] == {"experts": 1}


def test_two_mapped_paths_with_one_sha_are_two_libraries_and_void(tmp_path):
    cells = _cells()
    cells["E0a"]["lib"] = [LIB12, dict(LIB12, path="/other/libbitsandbytes_cuda128.so")]
    assert any("E0a: the cell mapped 2 libbitsandbytes files" in r for r in _write(tmp_path, cells)["reasons"])


def test_every_recorded_version_that_differs_is_listed_for_a_pair(tmp_path):
    out = _write(tmp_path, _cells())
    assert out["venv_at_fixed_build"]["pair"] == ["E0a", "E2"]
    assert out["venv_at_fixed_build"]["versions_differ"] == {"transformers": ["5.18.0", "5.5.0"]}
    assert out["build"]["e4b"]["versions_differ"] == {"torch": ["2.8.0+cu128", "2.12.1+cu130"]}


def test_the_reported_rows_are_the_registered_four(tmp_path):
    out = _write(tmp_path, _cells())
    assert [r["pair"] for r in out["reported"]] == [["E0a", "E1"], ["U1", "U2"], ["E0a", "U0a"], ["E1", "U1"]]
    assert [r.get("read_on") for r in out["reported"]] == [None, None, "dequantized digest", "dequantized digest"]


def test_the_cli_writes_the_result_and_prints_the_outcome(tmp_path):
    _write(tmp_path, copy.deepcopy(_cells()))
    p = subprocess.run([sys.executable, str(HBD / "hbd_reduce.py"), str(tmp_path / "spec.json"), "--out", str(tmp_path / "o.json")],
                       capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.startswith("HBD=READ") and json.loads((tmp_path / "o.json").read_text())["outcome"] == "READ"


def test_the_probe_reads_its_own_memory_maps_and_never_imports_unsloth_for_an_e4b_cell():
    src = (HBD / "hbd_probe.py").read_text()
    assert "/proc/self/maps" in src and 'if a.framework == "unsloth":\n        import unsloth' in src
    assert src.index("import unsloth") < src.index("    import torch")
