# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P152, the per-name census on Qwen3-30B-A3B at W16 with a W1 re-read (bench/p152/PREREG-p152.md): the staged pins, with
the per-name table and class map at P142's bytes and P136's pieces at P136's; the runner on Qwen3 at W1 and W16 with
grouped-nf4-gemm at e4b CI's pin, the power floor before the install, the GPU state per phase and the eager routing pass;
the reducer and the routing driver's self-tests and the mutants they must kill; the reducer on the REAL W1 record of
pername-5090-1 (sha-pinned) first; the driver's run ids."""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p152"
PN = B / "pername"
SOURCES = {"p152_run.sh": LANE, "p152_reduce.py": LANE, "p152_route.py": LANE, "pername_table.py": PN,
           "kernel_classes_pername.json": PN, "p136_reduce.py": B / "p136", "p136_box.py": B / "p136",
           "p136_census.py": B / "p136", "sc1b_census.py": B / "sc1b", "sc1b_e4b_census.py": B / "sc1b",
           "p109_box.py": B / "p109", "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "test_decode_graph_buckets.py": REPO / "tests", "test_kv_step_select.py": REPO / "tests",
           "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise", "test_gemv_bw_served_gpu.py": REPO / "tests"}
PYPATH = os.pathsep.join(str(B / d) for d in ("p136", "sc1b", "p109"))
RUN = (LANE / "p152_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p152_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p152.md").read_text(encoding="utf-8")
REAL_W1 = REPO / "tests" / "fixtures" / "p152" / "pername_b1.json"
REAL_W1_SHA256 = "0923a7b45681ccd6c9790176b584beccf362c0a63833bef3686903c88a888b77"   # pername-5090-1's table, verbatim


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300, cwd=cwd,
                          env=dict(os.environ, PYTHONPATH=PYPATH))


def _reducer():
    sys.path.insert(0, str(B / "p136"))
    spec = importlib.util.spec_from_file_location("p152_reduce_under_test", LANE / "p152_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p152.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_shared_pieces_are_at_p142s_and_p136s_pins():
    mine, pn, p136 = (_pins(LANE / "staged-p152.sha256"), _pins(PN / "staged-pername.sha256"),
                      _pins(B / "p136" / "staged-p136.sha256"))
    for name in ("pername_table.py", "kernel_classes_pername.json"):
        assert mine[name] == pn[name], name
    shared = set(mine) & set(p136)
    assert shared == set(SOURCES) - {"p152_run.sh", "p152_reduce.py", "p152_route.py", "pername_table.py",
                                     "kernel_classes_pername.json"}
    assert all(mine[n] == p136[n] for n in shared)


def test_the_runner_reads_qwen3_at_w1_and_w16_on_the_default():
    assert ("p152-5090-1|p152-5090-2|p152-5090-3) FAMILY=qwen3; MODEL=Qwen/Qwen3-30B-A3B; "
            "REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; MOE_LAYERS=48; MIN_DISK_GB=150; BATCHES='1 16';;") in RUN
    gnf4 = re.search(r"^GNF4_SHA=([0-9a-f]{40}) ", RUN, re.M)
    assert gnf4, "the grouped-nf4-gemm pin is a full commit"
    red = (LANE / "p152_reduce.py").read_text(encoding="utf-8")
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert f'GNF4_PIN = "{gnf4.group(1)}"' in red and gnf4.group(1) in PREREG and gnf4.group(1) in ci, "e4b CI's pin"
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    for pin in ('{(1536, 2048), (2048, 768)} <= set(ng._BW_SHAPES)', 'md.version("grouped-nf4-gemm") == "0.47.1"',
                'sp.resolve_fusion_modes({k: sp.FUSION_UNSET for k in sp.FUSION_KNOBS}, "qwen3_moe")',
                'hasattr(ib, "rope_norm_qk")'):
        assert pin in trip, pin
    assert "mixtral" not in RUN.lower().replace("mixtral-8x7b", "")
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("GNF4_GEMV_BW", "E4B_PAGED_GRAPHS", "E4B_FUSE_T1_GLUE", "E4B_PAGED_MAX_TOKENS_PER_SEQ"):
        assert re.search(rf"\b{knob}\b", unset), knob
    arm = RUN.index("python $W/sc1b_census.py arm")
    table = RUN.index('python $W/pername_table.py --node "$N"')
    route = RUN.index("python $W/p152_route.py --prompts")
    reduce_ = RUN.index("python $W/p152_reduce.py --dir")
    assert arm < table < route < reduce_


def test_the_power_floor_refuses_before_the_install():
    floor = RUN.index('[ "$PL" -ge "$MIN_POWER_W" ] || {')
    assert "MIN_POWER_W=575" in RUN and "finish 17; }" in RUN[floor:floor + 260]
    assert "power.limit unreadable" in RUN and floor < RUN.index('say "install e4b @')
    assert "${P152_MIN_POWER_W:-}" in RUN                       # a frozen reading knob


def test_the_gpu_state_brackets_every_phase():
    for tag in ('gpu_snap "arm_$TAG start"', 'gpu_snap "arm_$TAG end"', 'gpu_snap "$S start"', 'gpu_snap "$S end"',
                'gpu_snap "route_b16 start"', 'gpu_snap "route_b16 end"'):
        assert tag in RUN, tag
    assert "clocks_event_reasons.active" in RUN and "gpu_phases.csv" in RUN


def test_the_routing_pass_is_the_census_window_eager_at_w16():
    route = RUN[RUN.index("# ---- the ROUTING census"):RUN.index("python $W/p152_route.py --prompts") + 200]
    assert "E4B_PAGED_GRAPHS=0" in route and "P136_BATCH=16" in route
    assert "--skip $SKIP --steps $STEPS" in route and "$ENGINE_ENV" in route
    src = (LANE / "p152_route.py").read_text(encoding="utf-8")
    assert "census_window(sched, batch, a.skip, a.steps, rec.start, rec.stop" in src
    assert 'os.environ.get("E4B_PAGED_GRAPHS") != "0"' in src and "default_env_ok(env)" in src


def test_every_time_left_check_fits_the_guard():
    assert "guard 1.25 h" in PREREG
    need = {k: int(v) for k, v in re.findall(r"NEED_(FETCH|BAKE|ARM|CAPTURE|ROUTE)=(\d+)", RUN)}
    assert set(need) == {"FETCH", "BAKE", "ARM", "CAPTURE", "ROUTE"}
    for v in need.values():
        assert v + 600 <= 1.25 * 3600, need


@pytest.mark.parametrize("script", ["p152_reduce.py", "p152_route.py"])
def test_the_self_test_passes(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_reducer_on_the_real_w1_record_first():
    assert hashlib.sha256(REAL_W1.read_bytes()).hexdigest() == REAL_W1_SHA256
    red = _reducer()
    tab = json.loads(REAL_W1.read_text())
    assert red.table_faults(tab, 1) == []                       # the W1 expert path at 48 launches, the table closes
    w1 = red.rank(tab, 1, 3.9166)                               # pername-5090-1's unprofiled W1 step (S1a/S1b mean)
    assert all(x["class"] != "dense_gemm" for x in w1["ranked"]) and len(w1["closed"]) >= 4
    top = w1["candidates"][0]
    assert top["key"] == "_fp8_paged_decode_split_f8dot|128x1x1" and abs(top["share_unprofiled"] - 0.5558 / 3.9166) < 1e-3
    gu = next(x for x in w1["ranked"] if x["key"].endswith("|gate_up"))
    assert gu["floor_ms"] == 0.6068 and gu["addressable_ms"] == 0.0     # in situ under K33's isolated time
    ef = red.expert_floor(None, None, w1["class_removable_ms"]["moe_expert"])
    assert abs(ef["W1"]["floor_ms"] - 0.6800) < 1e-3 and ef["W16"]["status"] == "UNREAD"


REDUCE_MUTANTS = [
    # shares over the profiled P, not the unprofiled step (the maintainer's rule)
    ('"share_unprofiled": round(r["removable_ms"] / u_ms, 5)', '"share_unprofiled": round(r["removable_ms"] / tab["P_ms"], 5)'),
    # the W16 floor from uniform routing, not the routing census
    ("    mean_unique = sum(totals) / len(totals)\n", "    mean_unique = 82.4 * LAYERS\n"),
    # dense_gemm not closed at W1 (stream (A)'s lines proposed by stream (B))
    ('W1_CLOSED = {"dense_gemm"}', "W1_CLOSED = set()"),
    # the head named as a stream (B) candidate
    ('owner = "stream (A): P150\'s int8 head" if', 'owner = None if'),
    # the W1 expert-path count skipped
    ("    if batch != 1:\n        return faults\n", "    return faults\n"),
    # P136's grouped-nf4-gemm pin left in place
    ("pr.GNF4_SHA = GNF4_PIN\n", "\n"),
]
ROUTE_MUTANTS = [
    # the layer-order check dropped
    ("        if layers != list(range(n_layers)):\n", "        if False:\n"),
    # a step's rows not checked against the batch
    ("            if rows != batch or k != top_k:\n", "            if False:\n"),
]


@pytest.mark.parametrize("script,old,new", [("p152_reduce.py", o, n) for o, n in REDUCE_MUTANTS]
                         + [("p152_route.py", o, n) for o, n in ROUTE_MUTANTS])
def test_each_mutant_fails_the_self_test(tmp_path, script, old, new):
    src = (LANE / script).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / script).write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / script, "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P152_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p152_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p152-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p152" in out.stdout and "bash p152_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p152_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p145-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p152-5090-N" in out.stdout + out.stderr
