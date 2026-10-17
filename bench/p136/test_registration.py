"""CPU contract for P136's registered stage and fail-closed census rule."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGED = {
    "p136_run.sh": HERE / "p136_run.sh",
    "p136_reduce.py": HERE / "p136_reduce.py",
    "p136_box.py": HERE / "p136_box.py",
    "p136_census.py": HERE / "p136_census.py",
    "kernel_classes_nf4.json": HERE / "kernel_classes_nf4.json",
    "sc1b_census.py": ROOT / "bench/sc1b/sc1b_census.py",
    "sc1b_e4b_census.py": ROOT / "bench/sc1b/sc1b_e4b_census.py",
    "p109_box.py": ROOT / "bench/p109/p109_box.py",
    "k8_bake.py": ROOT / "bench/p39/k8_bake.py",
    "calib.json": ROOT / "bench/p39/calib.json",
    "test_decode_graph_buckets.py": ROOT / "tests/test_decode_graph_buckets.py",
    "test_kv_step_select.py": ROOT / "tests/test_kv_step_select.py",
    "test_fused_glue_decode_graphs_gpu.py": ROOT / "tests/test_fused_glue_decode_graphs_gpu.py",
    "test_gemv_bw_served_gpu.py": ROOT / "tests/test_gemv_bw_served_gpu.py",
}


def reducer():
    spec = importlib.util.spec_from_file_location("p136_reduce", HERE / "p136_reduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_staged_bytes_and_closed_source_map():
    rows = [line.split() for line in (HERE / "staged-p136.sha256").read_text().splitlines() if line.strip()]
    assert {name for _, name in rows} == set(STAGED)
    for want, name in rows:
        assert hashlib.sha256(STAGED[name].read_bytes()).hexdigest() == want, name
    drive = (HERE / "p136_drive.sh").read_text()
    run = (HERE / "p136_run.sh").read_text()
    assert "case \"$RUN_ID\"" in drive and "unregistered run ID" in drive
    assert "case \"$P136_RUN_ID\"" in run and "unregistered run ID" in run
    assert "P136_DRIVE_DRYRUN" in drive and "P136_DRY" in run
    assert "70bc03cb7a61b03c4e801e7423c58c103a1b6d7d" in run
    assert "--cuda-graph-trace=$MODE" in run and "SKIP=32; STEPS=64" in run
    assert "for B in $BATCHES" in run and "BATCHES='16'" in run
    assert "--exclude 'census/*.nsys-rep' --exclude 'census/*.sqlite'" in drive


def test_missing_receipts_void(tmp_path):
    r = reducer()
    for family in r.FAMILIES:
        result = r.assess(tmp_path, family, "a" * 40)
        assert result["verdict"] == "VOID"
        assert result["rows"] == {}


def test_gptoss_w16_read_and_blocking_label(tmp_path):
    r = reducer()
    model, rev, batches = r.FAMILIES["gptoss"]
    assert batches == (16,)
    for tag in ("S1a", "S1b"):
        arm = {"model": model, "revision": rev, "e4b_sha": "a" * 40,
               "gnf4_sha": r.GNF4_SHA, "mode": "speed", "tag": tag, "status": "ok",
               "knobs": {}, "max_seqs": 16, "max_tokens_per_seq": 4096,
               "moe_layers": 24, "int4_store_kinds": ["mxfp4"],
               "grouping": {"device_grouping": True, "force_singleton_groups": False},
               "graph_stats": {"16": {"captured": 1}},
               "workloads": {"W16": {"batch": 16, "status": "ok", "decode_ms_per_step": 1.0}}}
        (tmp_path / f"arm_{tag}.json").write_text(json.dumps(arm))
    drv = {"model": model, "revision": rev, "e4b_sha": "a" * 40,
           "gnf4_sha": r.GNF4_SHA, "batch": 16, "skip": 32, "steps_bracketed": 64,
           "moe_layers": 24, "grouping": {"device_grouping": True, "force_singleton_groups": False}}
    (tmp_path / "census_drv_b16.json").write_text(json.dumps(drv))
    terms = dict.fromkeys(r.CLASSES + ("I_in", "idle_out", "overlap_in", "O"), 0.1)
    terms["O"] = 0.2
    census = {"engine": "e4b", "batch": 16, "status": "ok", "labels": [],
              "P_ms": 1.0, "terms": terms, "unprofiled_ms": 1.0,
              "node": {"kernels_per_graph_modal": 1200}}
    path = tmp_path / "census_b16.json"
    path.write_text(json.dumps(census))
    assert r.assess(tmp_path, "gptoss", "a" * 40)["verdict"] == "READ"
    census["labels"] = ["CLASS_MAP_INCOMPLETE"]
    path.write_text(json.dumps(census))
    assert r.assess(tmp_path, "gptoss", "a" * 40)["verdict"] == "VOID"
