"""CPU contract for P136's registered stage and fail-closed census rule."""
import hashlib
import importlib.util
import json
import re
import subprocess
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
    "test_fused_glue_decode_graphs_gpu.py": HERE / "premise/test_fused_glue_decode_graphs_gpu.py",  # Amendment 1
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
    assert "2025.6.1.190-256136895201v0" in run
    assert "3bf863cc.pub" in run and "7fa2af80.pub" in run
    assert "AE09FE4BBD223A84B2CCFCE3F60F4B3D7FA2AF80" in run
    assert "EB693B3035CD5710E231E123A4B469963BF863CC" in run
    assert "NSYS STATUS REFUSED" not in run
    assert run.index("install_nsys\n# ---- the checkpoint") < run.index("snapshot_download")
    assert "for B in $BATCHES" in run and "BATCHES='16'" in run
    assert "--exclude 'census/*.nsys-rep' --exclude 'census/*.sqlite'" in drive


def test_missing_receipts_void(tmp_path):
    r = reducer()
    for family in r.FAMILIES:
        result = r.assess(tmp_path, family, "a" * 40)
        assert result["verdict"] == "VOID"
        assert result["rows"] == {}


def test_premise_summary_accepts_pytest_elapsed_suffix():
    run = (HERE / "p136_run.sh").read_text()
    pattern = re.search(r"grep -qE '([^']+)'", run).group(1)
    for summary in ("19 passed in 17.18s",
                    "19 passed, 5 warnings in 17.18s",
                    "19 passed, 5 warnings in 61.40s (0:01:01)"):
        assert subprocess.run(["grep", "-qE", pattern], input=summary, text=True,
                              check=False).returncode == 0
    for summary in ("18 passed in 17.18s", "19 passed, 1 skipped in 17.18s",
                    "19 passed, 1 failed in 17.18s"):
        assert subprocess.run(["grep", "-qE", pattern], input=summary, text=True,
                              check=False).returncode != 0


def test_default_store_read_and_lever_store_void(tmp_path):
    r = reducer()
    for family, (model, rev, batches) in r.FAMILIES.items():
        root = tmp_path / family
        root.mkdir()
        grouping = {"device_grouping": True, "force_singleton_groups": False}
        for tag in ("S1a", "S1b"):
            arm = {"model": model, "revision": rev, "e4b_sha": "a" * 40,
                   "gnf4_sha": r.GNF4_SHA, "mode": "speed", "tag": tag, "status": "ok",
                   "knobs": {}, "max_seqs": 16,
                   "max_tokens_per_seq": 768 if family == "mixtral" else 4096,
                   "moe_layers": r.MOE_LAYERS[family], "int4_store_kinds": [],
                   "int4_expert_layers": 0, "grouping": grouping,
                   "graph_stats": {str(batch): {"captured": 1} for batch in batches},
                   "workloads": {f"W{batch}": {"batch": batch, "status": "ok", "decode_ms_per_step": 1.0}
                                 for batch in batches}}
            (root / f"arm_{tag}.json").write_text(json.dumps(arm))
        for batch in batches:
            drv = {"model": model, "revision": rev, "e4b_sha": "a" * 40,
                   "gnf4_sha": r.GNF4_SHA, "batch": batch, "skip": 32, "steps_bracketed": 64,
                   "moe_layers": r.MOE_LAYERS[family], "grouping": grouping}
            (root / f"census_drv_b{batch}.json").write_text(json.dumps(drv))
            terms = dict.fromkeys(r.CLASSES + ("I_in", "idle_out", "overlap_in", "O"), 0.1)
            terms["O"] = 0.2
            census = {"engine": "e4b", "batch": batch, "status": "ok", "labels": [],
                      "P_ms": 1.0, "terms": terms, "unprofiled_ms": 1.0,
                      "node": {"kernels_per_graph_modal": 1200}}
            (root / f"census_b{batch}.json").write_text(json.dumps(census))
        assert r.assess(root, family, "a" * 40)["verdict"] == "READ"

        arm_path = root / "arm_S1a.json"
        arm = json.loads(arm_path.read_text())
        for kinds, layers in ((["mxfp4"], 0), (["int4_b32"], 0), ([], 1), (None, 0)):
            changed = dict(arm, int4_store_kinds=kinds, int4_expert_layers=layers)
            arm_path.write_text(json.dumps(changed))
            result = r.assess(root, family, "a" * 40)
            assert result["verdict"] == "VOID"
            assert "arm S1a: default expert store differs" in result["why"]
        arm_path.write_text(json.dumps(arm))
        census_path = root / f"census_b{batches[-1]}.json"
        census = json.loads(census_path.read_text())
        census["labels"] = ["CLASS_MAP_INCOMPLETE"]
        census_path.write_text(json.dumps(census))
        assert r.assess(root, family, "a" * 40)["verdict"] == "VOID"


RUN_ID = re.compile(r"p136-(?:gemma4|mixtral|gptoss)-5090-\d|p136-a2000-dry-\d")


def _case_ids(script, var):
    """The run IDs one ``case "$VAR" in ... esac`` block accepts (its pattern lines only)."""
    block = re.search(r'^case "\$' + var + r'" in\n(.*?)^esac', script, re.DOTALL | re.MULTILINE).group(1)
    return {i for line in block.splitlines() if ")" in line for i in RUN_ID.findall(line.split(")", 1)[0])}


def test_the_runner_and_the_driver_accept_the_same_registered_ids():
    """Amendment 2: the box and the controller must agree on every run ID. A drift in either direction is paid for after
    the create (an ID the drive accepts and the box refuses starts a box; the reverse creates an instance for nothing), and
    every accepted ID is one the preregistration names."""
    run = _case_ids((HERE / "p136_run.sh").read_text(), "P136_RUN_ID")
    drive = _case_ids((HERE / "p136_drive.sh").read_text(), "RUN_ID")
    named = set(RUN_ID.findall((HERE / "PREREG-p136.md").read_text()))
    want = {f"p136-{f}-5090-{n}" for f in ("gemma4", "mixtral", "gptoss") for n in (1, 2)} | {"p136-a2000-dry-1"}
    assert run == drive == want, (sorted(run ^ want), sorted(drive ^ want))
    assert named == want, (sorted(named - want), sorted(want - named))
