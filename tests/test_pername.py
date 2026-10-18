# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The per-name served census (bench/pername/PREREG-pername.md; e4b#1313): the staged pins and P136's pieces
byte-identical, the class map = P136's plus one name, the runner on Qwen3 at W1 with the table step, the self-tests,
the mutants the self-tests must kill, and the driver's run ids."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "pername"
SOURCES = {"pername_run.sh": LANE, "pername_table.py": LANE, "pername_reduce.py": LANE, "kernel_classes_pername.json": LANE,
           "p136_reduce.py": B / "p136", "p136_box.py": B / "p136", "p136_census.py": B / "p136",
           "sc1b_census.py": B / "sc1b", "sc1b_e4b_census.py": B / "sc1b", "p109_box.py": B / "p109",
           "k8_bake.py": B / "p39", "calib.json": B / "p39", "test_decode_graph_buckets.py": REPO / "tests",
           "test_kv_step_select.py": REPO / "tests", "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise",
           "test_gemv_bw_served_gpu.py": REPO / "tests"}
PYPATH = os.pathsep.join(str(B / d) for d in ("p136", "sc1b", "p109"))


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


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-pername.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_p136s_pieces_are_staged_at_p136s_pins():
    mine, p136 = _pins(LANE / "staged-pername.sha256"), _pins(B / "p136" / "staged-p136.sha256")
    shared = set(mine) & set(p136)
    assert shared == set(SOURCES) - {"pername_run.sh", "pername_table.py", "pername_reduce.py", "kernel_classes_pername.json"}
    assert all(mine[n] == p136[n] for n in shared)


def test_the_map_is_p136s_plus_rope_norm_qk():
    mine = json.loads((LANE / "kernel_classes_pername.json").read_text(encoding="utf-8"))["e4b"]
    p136 = json.loads((B / "p136" / "kernel_classes_nf4.json").read_text(encoding="utf-8"))["e4b"]
    added = [(a["class"], sorted(set(a["match"]) - set(b["match"]))) for a, b in zip(mine["rules"], p136["rules"])
             if a["match"] != b["match"]]
    assert added == [("norm_elem", ["_rope_norm_qk"])]
    assert {k: v for k, v in mine.items() if k != "rules"} == {k: v for k, v in p136.items() if k != "rules"}
    assert [a["class"] for a in mine["rules"]] == [b["class"] for b in p136["rules"]]


def test_the_runner_reads_qwen3_at_w1_with_the_table():
    run = (LANE / "pername_run.sh").read_text(encoding="utf-8")
    assert "pername-5090-1|pername-5090-2) FAMILY=qwen3; MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; " \
           "MOE_LAYERS=48; MIN_DISK_GB=150; BATCHES='1';;" in run
    assert "GNF4_SHA=70bc03cb7a61b03c4e801e7423c58c103a1b6d7d" in run          # P136's v0.46.1 pin, unchanged
    arm = run.index("python $W/sc1b_census.py arm")
    table = run.index("python $W/pername_table.py --node \"$N\"")
    reduce_ = run.index("python $W/pername_reduce.py --dir")
    assert arm < table < reduce_
    assert "p136-" not in run and "verdict_p136" not in run


@pytest.mark.parametrize("script", ["pername_table.py", "pername_reduce.py"])
def test_the_self_tests_pass(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


MUTANTS = {
    # the table bills each item its whole duration instead of its own advance of the frontier
    "pername_table.py": [('excl = max(0, r["end"] - max(r["start"], frontier))', 'excl = r["end"] - r["start"]')],
    "pername_reduce.py": [
        # the expert GEMVs ranked on removable time with no K33 floor
        ('floor = K33_MS.get(r.get("role"), 0.0) if r["name"] == GEMV else 0.0', "floor = 0.0"),
        # the swiglu line read from its billed duration
        ('sw = round(sum(x["removable_ms"] for x in lines', 'sw = round(sum(x["dur_ms"] for x in lines'),
        # dense_gemm (closed) ranked like any other line
        ('"closed": r["class"] in CLOSED})', '"closed": False})'),
    ],
}


@pytest.mark.parametrize("script,old,new", [(s, o, n) for s, ms in MUTANTS.items() for o, n in ms])
def test_each_mutant_fails_the_self_test(tmp_path, script, old, new):
    src = (LANE / script).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / script).write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / script, "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "PN_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "pername_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "pername-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/pername" in out.stdout and "bash pername_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "pername_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p136-mixtral-5090-9"), timeout=120)
    assert out.returncode == 78 and "is not pername-5090-N" in out.stdout + out.stderr
