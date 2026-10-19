# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P143 Amendment 3 (bench/p143/PREREG-p143.md; e4b#1313), the confirmatory served read of E4B_ATTN_SPLIT_CAP: the
staged pins (A2's box at A2's registered bytes, P136's premise tests and P115 Phase D's served pieces at their own
registrations), the runner's order (no kernel step; eight arms at five reps), the self-tests run as the box runs them
(one directory), the mutants the reducer's self-test must kill, and the driver's run ids."""
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p143"
SOURCES = {"p143a3_run.sh": LANE, "p143a2_box.py": LANE, "p143a3_reduce.py": LANE,
           "p115_quality.py": B / "p115", "p109_box.py": B / "p109", "p110_box.py": B / "p110", "p108_box.py": B / "p108",
           "p97_box.py": B / "p97", "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "test_decode_graph_buckets.py": REPO / "tests", "test_kv_step_select.py": REPO / "tests",
           "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise", "test_gemv_bw_served_gpu.py": REPO / "tests"}


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=180, cwd=cwd)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p143a3.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_a2s_box_and_every_shared_piece_are_at_a2s_registered_bytes():
    """A3 runs A2's box unchanged, and every piece the two kits share (the served pieces, the premise) is A2's."""
    mine, a2 = _pins(LANE / "staged-p143a3.sha256"), _pins(LANE / "staged-p143a2.sha256")
    shared = set(mine) & set(a2)
    assert "p143a2_box.py" in shared and len(shared) == 12
    assert all(mine[n] == a2[n] for n in shared)


def test_the_runner_has_no_kernel_step_eight_arms_at_five_reps_and_reads_only_a_rule_verdict():
    run = (LANE / "p143a3_run.sh").read_text(encoding="utf-8")
    assert "GNF4_SHA=38b308887688ea5bda7dd449bd3cc3dc1426b995" in run and 'md.version("grouped-nf4-gemm") == "0.47.1"' in run
    assert "p143a3-5090-1|p143a3-5090-2|p143a3-5090-3) MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;" in run
    assert "SHORT=32; LONG=160; REPS=5; CONT=128; WINDOWS=12" in run
    for gone in ("p143a2_kernel", "kernel.json", "NEED_KERNEL", "p143a2_reduce"):
        assert gone not in run, gone
    order = ["grep -qE '^19 passed", "snapshot_download", "python $W/k8_bake.py", "--prompts-only",
             "for TAG in S0a S1a S1b S0b S0c S1c S1d S0d; do", "for PH in off on; do", "python $W/p143a3_reduce.py --dir"]
    at = [run.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert run.count("E4B_ATTN_SPLIT_CAP=$CAPV P143A2_ARM=$ARM") == 2, "A2's box, its own arm variable"
    assert 'case "$V" in FLIP|NO_FLIP|FUNCTION_FAIL|QUALITY_FAIL) ;;' in run, "NOISY and VOID are not readings"


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p143a3_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


@pytest.mark.parametrize("script", ["p143a2_box.py", "p143a3_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_a2s_engagement_checks_are_carried_verbatim():
    a2 = (LANE / "p143a2_reduce.py").read_text(encoding="utf-8")
    a3 = (LANE / "p143a3_reduce.py").read_text(encoding="utf-8")
    helpers = a2[a2.index("def _rate(r, w):"):a2.index("def kernel_faults(k) -> tuple:")].rstrip()
    assert helpers in a3


MUTANTS = [
    # g1 from the best pair, not the worst of four
    ('"g1": round(min(pairs["W1"].values()), 4)', '"g1": round(max(pairs["W1"].values()), 4)'),
    # only the first ABBA block's pairs
    ('PAIRS = (("S1a", "S0a"), ("S1b", "S0b"), ("S1c", "S0c"), ("S1d", "S0d"))', 'PAIRS = (("S1a", "S0a"), ("S1b", "S0b"))'),
    # the NOISY band widened
    ("NOISY_BAND = (0.985, 1.015)", "NOISY_BAND = (0.98, 1.02)"),
    # the second block's self-pairs dropped
    ('SELF_PAIRS = (("S0b", "S0a"), ("S1b", "S1a"), ("S0d", "S0c"), ("S1d", "S1c"))', 'SELF_PAIRS = (("S0b", "S0a"), ("S1b", "S1a"))'),
    # NOISY never read
    ("    if noisy:\n        out.update(verdict=\"NOISY\"", "    if False:\n        out.update(verdict=\"NOISY\""),
    # the five reps never checked
    ('        if r.get("reps") != REPS:', "        if False:"),
    # W16's no-regression line dropped from the flip
    ("    elif g1 >= FLIP_G1 and g16 >= NOREG_G16:", "    elif g1 >= FLIP_G1:"),
    # the old rule: S1 arms held to S0a's tokens (the W16 prefill ramp runs the capped buckets)
    ('        first = "S0a" if t.startswith("S0") else "S1a"', '        first = "S0a"'),
    # the SANE gate's argmax bar dropped
    ('    elif abs(sane["bias"]) > SANE_BIAS or sane["argmax_agree"] < SANE_ARGMAX:', '    elif abs(sane["bias"]) > SANE_BIAS:'),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p143a3_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p143a3_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p143a3_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P143A3_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p143a3_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p143a3-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p143a3" in out.stdout and "bash p143a3_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p143a3_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p143a2-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p143a3-5090-N" in out.stdout + out.stderr
