"""CPU refusals for the new share-only census contract."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("g4census_reduce", HERE / "g4census_reduce.py")
reduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reduce)
SHA = "a" * 40


def _write(root, name, value):
    (root / name).write_text(json.dumps(value))


def _arm(tag):
    return {
        "model": reduce.MODEL, "revision": reduce.REVISION, "e4b_sha": SHA, "gnf4_sha": reduce.GNF4_SHA,
        "mode": "speed", "tag": tag, "status": "ok", "torch": reduce.TORCH, "knobs": {}, "max_seqs": 16,
        "graph_stats": {"1": {"replays": 64, "eager_steps": 0}, "16": {"replays": 64, "eager_steps": 0}}, "moe_layers": 30, "int4_store_kinds": [],
        "int4_expert_layers": 0, "grouping": {"device_grouping": True, "force_singleton_groups": False},
        "max_tokens_per_seq": 4096, "prompts_sha256": "prompt-digest",
        "workloads": {f"W{b}": {"status": "ok", "batch": b, "decode_ms_per_step": 10.0} for b in (1, 16)},
    }


def _driver(batch, cid):
    return {
        "model": reduce.MODEL, "revision": reduce.REVISION, "e4b_sha": SHA, "gnf4_sha": reduce.GNF4_SHA,
        "mode": "census", "batch": batch, "skip": 32, "steps_bracketed": 64,
        "capture_id": cid, "moe_layers": 30, "int4_store_kinds": [], "int4_expert_layers": 0, "torch": reduce.TORCH,
        "grouping": {"device_grouping": True, "force_singleton_groups": False},
        "max_tokens_per_seq": 4096, "prompts_sha256": "prompt-digest", "graph_stats": {str(batch): {"replays": 64, "eager_steps": 0}},
    }


def _census(batch):
    class_ms = {name: {"median_ms": 1.0} for name in reduce.CLASSES}
    return {
        "engine": "e4b", "batch": batch, "status": "labelled", "labels": ["NODE_TRACE_INFLATED"],
        "P_ms": 10.0, "graph": {"S": {"median_ms": 9.0}, "steps_kept": 61},
        "gates": {"node_span_over_span": 1.03, "node_union_over_span": 1.00},
        "node_band": {"omega_ms": 0.3},
        "node": {"class": class_ms, "residual_fraction": 0.001, "kernels_per_graph_modal": 4000,
                 "steps_kept": 61, "segments": {"replays_broken": 0, "moe_layers": 30}},
    }


def _fixture(root):
    for tag in ("S1a", "S1b"):
        _write(root, f"arm_{tag}.json", _arm(tag))
    for batch in (1, 16):
        for mode, cid in (("graph", "a" * 32), ("node_a", "b" * 32), ("node_b", "c" * 32)):
            _write(root, f"census_drv_b{batch}_{mode}.json", _driver(batch, cid))
        for side in ("a", "b"):
            _write(root, f"census_b{batch}_node_{side}.json", _census(batch))


def _mutate(root, name, mutate):
    p = root / name
    data = json.loads(p.read_text())
    mutate(data)
    p.write_text(json.dumps(data))


def test_complete_repeat_reads_but_does_not_turn_node_duration_into_graph_ms(tmp_path):
    _fixture(tmp_path)
    result = reduce.assess(tmp_path, SHA)
    assert result["verdict"] == "READ", result["why"]
    row = result["rows"]["W1"]
    assert row["graph_P_ms_profiled"] == 10.0
    assert row["node_duration_share_midpoint"]["moe_expert"] == pytest.approx(1 / 9)
    assert "classes_ms" not in row and "busy_fraction" not in row
    assert row["labels"] == ["NODE_TRACE_INFLATED"]
    assert set(result["predictions"].values()) == {"MISSED"}  # predictions do not gate the READ


@pytest.mark.parametrize("target,edit,reason", [
    ("census_drv_b1_node_b.json", lambda d: d.update(capture_id="b" * 32), "separately identified"),
    ("census_drv_b1_node_b.json", lambda d: d.update(steps_bracketed=63), "window"),
    ("census_drv_b16_node_a.json", lambda d: d.update(prompts_sha256="other"), "prompt"),
    ("census_b1_node_b.json", lambda d: d["node"]["class"]["moe_expert"].update(median_ms=3.0), "five percentage"),
    ("census_b16_node_a.json", lambda d: d["node"].update(residual_fraction=0.03), "map or segments"),
    ("census_b16_node_a.json", lambda d: d.update(labels=["CLASS_MAP_SEGMENT_BROKEN"]), "blocking labels"),
    ("census_b1_node_a.json", lambda d: d.update(P_ms=11.0), "graph clock"),
    ("census_b1_node_a.json", lambda d: d["node"]["class"].pop("memcpy"), "incomplete/nonfinite"),
    ("arm_S1b.json", lambda d: d["workloads"]["W1"].update(decode_ms_per_step=11.0), "self-pair"),
    ("arm_S1a.json", lambda d: d.update(int4_expert_layers=1), "configuration"),
    ("arm_S1b.json", lambda d: d.update(prompts_sha256="other"), "prompt digests"),
    ("census_drv_b16_node_a.json", lambda d: d["graph_stats"]["16"].update(eager_steps=1), "window"),
    ("census_b16_node_a.json", lambda d: d["node"].update(steps_kept=40), "classified window"),
])
def test_registered_refusals(tmp_path, target, edit, reason):
    _fixture(tmp_path)
    _mutate(tmp_path, target, edit)
    result = reduce.assess(tmp_path, SHA)
    assert result["verdict"] == "VOID"
    assert any(reason in fault for fault in result["why"]), result["why"]


def test_second_node_is_required(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "census_b16_node_b.json").unlink()
    result = reduce.assess(tmp_path, SHA)
    assert result["verdict"] == "VOID" and result["rows"] == {}


def test_prior_p136_driver_without_capture_identity_refuses(tmp_path):
    _fixture(tmp_path)
    _mutate(tmp_path, "census_drv_b1_node_a.json", lambda d: d.pop("capture_id"))
    assert reduce.assess(tmp_path, SHA)["verdict"] == "VOID"


def test_graph_clock_remains_tied_to_unprofiled_reference(tmp_path):
    _fixture(tmp_path)
    for side in ("a", "b"):
        _mutate(tmp_path, f"census_b1_node_{side}.json", lambda d: d.update(P_ms=10.6))
    result = reduce.assess(tmp_path, SHA)
    assert result["verdict"] == "VOID"
    assert any("frozen 5%" in fault for fault in result["why"])


def test_staged_manifest_pins_each_actual_source():
    root = HERE.parents[1]
    sources = {
        "g4census_run.sh": HERE / "g4census_run.sh",
        "g4census_reduce.py": HERE / "g4census_reduce.py",
        "g4census_census.py": HERE / "g4census_census.py",
        "p136_box.py": root / "bench/p136/p136_box.py",
        "kernel_classes_nf4.json": root / "bench/p136/kernel_classes_nf4.json",
        "sc1b_census.py": root / "bench/sc1b/sc1b_census.py",
        "sc1b_e4b_census.py": root / "bench/sc1b/sc1b_e4b_census.py",
        "p109_box.py": root / "bench/p109/p109_box.py",
        "k8_bake.py": root / "bench/p39/k8_bake.py",
        "calib.json": root / "bench/p39/calib.json",
        "test_decode_graph_buckets.py": root / "tests/test_decode_graph_buckets.py",
        "test_kv_step_select.py": root / "tests/test_kv_step_select.py",
        "test_fused_glue_decode_graphs_gpu.py": root / "bench/p136/premise/test_fused_glue_decode_graphs_gpu.py",
        "test_gemv_bw_served_gpu.py": root / "tests/test_gemv_bw_served_gpu.py",
    }
    rows = [line.split() for line in (HERE / "staged-g4census.sha256").read_text().splitlines()]
    assert {name for _, name in rows} == set(sources)
    for want, name in rows:
        assert hashlib.sha256(sources[name].read_bytes()).hexdigest() == want, name
