# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P151 (bench/p151/PREREG-p151.md; e4b#1313), D1's served read of grouped-nf4-gemm K35 through
E4B_DENSE_GEMV_BW: the staged pins (every piece shared with P143 Amendment 3's kit at A3's pins, K35's compiled contract
at grouped-nf4-gemm's bytes), the runner's order (the power floor before the install, the premise at 31, eight arms at
five reps), the self-tests run as the box runs them (one directory), A2's and A3's pieces carried verbatim, the mutants
the self-tests must kill, and the driver's run ids."""
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p151"
P143 = B / "p143"
SOURCES = {"p151_run.sh": LANE, "p151_box.py": LANE, "p151_reduce.py": LANE,
           "p115_quality.py": B / "p115", "p109_box.py": B / "p109", "p110_box.py": B / "p110", "p108_box.py": B / "p108",
           "p97_box.py": B / "p97", "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "test_decode_graph_buckets.py": REPO / "tests", "test_kv_step_select.py": REPO / "tests",
           "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise", "test_gemv_bw_served_gpu.py": REPO / "tests",
           "test_dense_gemv.py": LANE / "premise"}
#: grouped-nf4-gemm kernel/test_dense_gemv.py at K35's merge (9b824155ace19796e14ff182ed1ac8e4dc406086), its sha256
K35_COMPILED_TEST_SHA256 = "e21de6e51ef4f920fe26e9611f3a415ff064360678480ff1a3349f639fa2ca05"


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
    pins = _pins(LANE / "staged-p151.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_every_piece_shared_with_a3_is_at_a3s_pins_and_the_premise_is_k35s_file():
    mine, a3 = _pins(LANE / "staged-p151.sha256"), _pins(P143 / "staged-p143a3.sha256")
    shared = set(mine) & set(a3)
    assert len(shared) == 11 and "p143a2_box.py" not in mine
    assert all(mine[n] == a3[n] for n in shared)
    assert mine["test_dense_gemv.py"] == K35_COMPILED_TEST_SHA256


def test_the_runner_refuses_below_575_w_before_the_install_and_reads_only_a_rule_verdict():
    run = (LANE / "p151_run.sh").read_text(encoding="utf-8")
    assert "GNF4_SHA=9b824155ace19796e14ff182ed1ac8e4dc406086" in run and 'md.version("grouped-nf4-gemm") == "0.47.1"' in run
    assert "p151-5090-1|p151-5090-2|p151-5090-3) MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;" in run
    assert "SHORT=32; LONG=160; REPS=5; CONT=128; WINDOWS=12" in run and "MIN_POWER_W=575 " in run
    assert 'finish 17; }' in run and 'echo "refused: power.limit unreadable" > REFUSAL; finish 15; }' in run
    assert "E4B_ATTN_SPLIT_CAP E4B_DENSE_GEMV_BW\n" in run, "both knobs start unset"
    assert 'assert sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES == {"qwen3_moe": 16}' in run
    order = ["for f in p151_box.py", "PL=$(nvidia-smi --query-gpu=power.limit", "pipx logs/pip_e4b.log",
             "grep -qE '^31 passed", "snapshot_download", "python $W/k8_bake.py", "--prompts-only",
             "for TAG in S0a S1a S1b S0b S0c S1c S1d S0d; do", "for PH in off on; do", "python $W/p151_reduce.py --dir"]
    at = [run.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert "test_gemv_bw_served_gpu.py test_dense_gemv.py -q -rs" in run, "K35's compiled contract joins the premise"
    assert run.count("E4B_DENSE_GEMV_BW=$DV P151_ARM=$ARM") == 2 and "E4B_ATTN_SPLIT_CAP=" not in run
    assert 'case "$V" in FLIP|NO_FLIP|FUNCTION_FAIL|QUALITY_FAIL) ;;' in run, "NOISY and VOID are not readings"


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p151_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


@pytest.mark.parametrize("script", ["p151_box.py", "p151_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_a3s_checks_and_rungs_are_carried_verbatim():
    a3 = (P143 / "p143a3_reduce.py").read_text(encoding="utf-8")
    mine = (LANE / "p151_reduce.py").read_text(encoding="utf-8")
    helpers = a3[a3.index("def _rate(r, w):"):a3.index("def cap_faults(")]
    rungs = a3[a3.index("    noisy = [f\"W1 self-pair"):a3.index("    if fn:\n")]
    assert helpers in mine and rungs in mine


def test_a2s_served_kv_is_carried_verbatim():
    a2 = (P143 / "p143a2_box.py").read_text(encoding="utf-8")
    mine = (LANE / "p151_box.py").read_text(encoding="utf-8")
    assert a2[a2.index("class ServedKV:"):a2.index("def speed_main(a) -> int:")] in mine


REDUCER_MUTANTS = [
    # g1 from the best pair, not the worst of four
    ('    g1_min, g16_min = min(pairs["W1"].values()), min(pairs["W16"].values())',
     '    g1_min, g16_min = max(pairs["W1"].values()), min(pairs["W16"].values())'),
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
    ("    elif g1_min >= FLIP_G1 and g16_min >= NOREG_G16:", "    elif g1_min >= FLIP_G1:"),
    # the bars read the ROUNDED mins
    ("    elif g1_min >= FLIP_G1 and g16_min >= NOREG_G16:",
     "    elif round(g1_min, 4) >= FLIP_G1 and round(g16_min, 4) >= NOREG_G16:"),
    # S1 arms held to S0a's tokens (the kernel rounds differently from cuBLAS: S1 may decode other tokens than S0)
    ('        first = "S0a" if t.startswith("S0") else "S1a"', '        first = "S0a"'),
    # the SANE gate's argmax bar dropped
    ('    elif abs(sane["bias"]) > SANE_BIAS or sane["argmax_agree"] < SANE_ARGMAX:', '    elif abs(sane["bias"]) > SANE_BIAS:'),
    # the knob's engagement never checked at the speed build
    ("        void += dense_faults(t, r, t[:2])\n", ""),
    # ... nor in SANE
    ("        void += dense_faults(name, rec, arm, measure=True)\n", ""),
    # a one-row fallback to F.linear accepted (docs-upkeep's review: routed everywhere, cuBLAS in the capture)
    (' or routed != {"1": DENSE_MODULES} or "1" in passed:', ' or routed != {"1": DENSE_MODULES}:'),
    # a zero count accepted
    ('    if eng.get("modules") != DENSE_MODULES or routed != {"1": DENSE_MODULES} or "1" in passed:',
     '    if routed and routed != {"1": DENSE_MODULES} or "1" in passed:'),
    # an unreadable power limit accepted
    ("                if pl is None or pl < MIN_POWER_W:", "                if pl is not None and pl < MIN_POWER_W:"),
    # the floor lowered to a capped board's
    ("MIN_POWER_W = 575 ", "MIN_POWER_W = 450 "),
    # an explicit cap of 16 taken for the default
    ('    if sc.get("cap") != CAP[arm] or sc.get("source") != "default-allowlisted":', '    if sc.get("cap") != CAP[arm]:'),
]


@pytest.mark.parametrize("old,new", REDUCER_MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p151_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p151_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p151_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


BOX_MUTANTS = [
    # the split cap knob may be set in an arm
    ("    bad.update({k: env.get(k) for k in (CAP_KNOB,) + FUSION_KNOBS + GEMV_KNOBS if (env.get(k) or \"\").strip()})",
     "    bad.update({k: env.get(k) for k in FUSION_KNOBS + GEMV_KNOBS if (env.get(k) or \"\").strip()})"),
    # the arms' values swapped
    ('ARM_VALUE = {"S0": "0", "S1": "1"}', 'ARM_VALUE = {"S0": "1", "S1": "0"}'),
    # an unreadable telemetry field read as zero
    ("        except (IndexError, ValueError):\n            out[k] = None", "        except (IndexError, ValueError):\n            out[k] = 0.0"),
]


@pytest.mark.parametrize("old,new", BOX_MUTANTS)
def test_each_box_mutant_fails_the_self_test(box_dir, tmp_path, old, new):
    src = (LANE / "p151_box.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p151_box.py").write_bytes(src.replace(old, new).encode("utf-8"))
    shutil.copyfile(box_dir / "p109_box.py", tmp_path / "p109_box.py")
    out = _run(tmp_path / "p151_box.py", "--self-test", cwd=tmp_path)
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P151_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p151_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p151-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p151" in out.stdout and "bash p151_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p151_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p143a3-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p151-5090-N" in out.stdout + out.stderr
