"""P159 (bench/p159/PREREG-p159.md): the staged pins; the box module's and the reducer's self-tests; the block's calls
against the registered SMT fuse block; the registered constants against the PREREG; the guard fit on both sides; the
drive's declared run ids and its fit refusal; the runner's refusal order. Stdlib only, no GPU."""

import ast
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p159"
SMT = REPO / "bench" / "smt-fuse"
PREREG = (LANE / "PREREG-p159.md").read_text()
RUN = (LANE / "p159_run.sh").read_text()
DRIVE = (LANE / "p159_drive.sh").read_text()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOX = _load("p159_box_under_test", LANE / "p159_box.py")
RED = _load("p159_reduce_under_test", LANE / "p159_reduce.py")


def _entries():
    out = {}
    for line in (LANE / "staged-p159.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def test_staged_files_match_their_pins_and_the_drive_stages_exactly_them():
    src = {"p159_run.sh": LANE, "p159_box.py": LANE, "p159_reduce.py": LANE, "smt_fuse_box.py": SMT,
           "q235_bootstrap.py": SMT, "wheels.json": SMT, "dq11_require_git.sh": SMT}
    entries = _entries()
    assert set(entries) == set(src), entries
    for name, want in entries.items():
        assert hashlib.sha256((src[name] / name).read_bytes()).hexdigest() == want, name
        assert name in DRIVE, name
    assert "sha256sum -c staged-p159.sha256" in RUN
    assert entries["wheels.json"] == "84f361ed35fff9a857f340b65488ad582dc4270aef7e67cbeb6b46b8370021f3"   # Q235's lock


def test_both_self_tests_pass():
    for script, want in (("p159_box.py", "p159_box self-test OK (15 cases)"),
                         ("p159_reduce.py", "p159_reduce self-test OK (28 cases)")):
        p = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True,
                           timeout=120)
        assert p.returncode == 0 and want in p.stdout, p.stdout + p.stderr


def _calls(fn_src):
    """The block's calls that define what it measures, normalized (whitespace-free)."""
    keep = ("safe_open(", "get_tensor(", "fuse_experts(", '.to("cuda")', "from_float(", "ExpertsLoRA(",
            "enable_expert_offload(", "release_cached_pinned_memory(", "empty_cache(", "set_num_threads(")
    out = []
    for node in ast.walk(ast.parse(fn_src)):
        if isinstance(node, ast.Call):
            s = re.sub(r"\s+", "", ast.unparse(node))
            if any(k.replace(" ", "") in s for k in keep):
                out.append(s.replace("SF.", "").replace("KEY.format", "KEY.format"))
    return sorted(out)


def test_the_block_makes_the_registered_blocks_calls():
    smt = (SMT / "smt_fuse_box.py").read_text()
    box = (LANE / "p159_box.py").read_text()
    one_rep = ast.get_source_segment(smt, next(n for n in ast.parse(smt).body
                                               if isinstance(n, ast.FunctionDef) and n.name == "one_rep"))
    one_block = ast.get_source_segment(box, next(n for n in ast.parse(box).body
                                                 if isinstance(n, ast.FunctionDef) and n.name == "one_block"))
    assert _calls(one_rep) == _calls(one_block)


def test_the_registered_constants_are_the_prereg_s():
    assert BOX.FUSED_SHA in PREREG and BOX.NF4_SHA in PREREG
    assert BOX.FUSE_FRESH_BYTES == RED.FUSE_FRESH_BYTES == int(7.5 * 2 ** 30)
    assert BOX.FAULTS_4K == 1966080 and BOX.FAULTS_2M == 3840 and "**1,966,080**" in PREREG and "**3,840**" in PREREG
    assert (BOX.FAST_S, BOX.SLOW_S) == (RED.FAST_S, RED.SLOW_S) == (1.5, 3.0)
    assert "FAST at or below **1.5 s**" in PREREG and "SLOW at or above **3.0 s**" in PREREG
    assert (RED.HI, RED.LO) == (0.8, 0.2) and RED.COUNTER_MULTIPLE == 5.0 and "at least **5x**" in PREREG
    assert (RED.HUGE_SLOW_MAX, RED.HUGE_FAST_MIN) == (0.1, 0.5) and RED.B_RATIO == 1.25
    assert (RED.MIN_STATE_BLOCKS, RED.MIN_SLOW_TRIPLES, RED.MIN_B) == (12, 6, 3)
    assert RED.N_HUGE_MAX == 64 << 20 and RED.P_MINFLT_MAX == 400_000
    assert abs(RED.NULL_LOG_MAX - __import__("math").log(1.5)) < 1e-12 and "exceeds log 1.5" in PREREG
    assert BOX.P_TUNABLES.startswith("glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=")
    assert BOX.PR_SET_THP_DISABLE == 41 and BOX.BALLOON_MAX == 96 * 2 ** 30 and BOX.FLOOR == 1.3
    assert BOX.DRY_BALLOON_MAX == 2 ** 30 and "at most 1 GiB" in PREREG


def test_the_guard_fits_with_25_percent_headroom():
    """The cross-lane rule on both sides: the runner's pre-install check carries the box module's fit, and the
    registered guard holds the boot plus that fit with 25 % headroom."""
    k = {n: int(v) for n, v in re.findall(r"^(INSTALL_TO_GATE_S|NEED_PROBE|RESERVE)=(\d+)", RUN, re.M)}
    f = BOX.time_fit()
    assert k == {"INSTALL_TO_GATE_S": BOX.FIT["install_to_gate_s"], "NEED_PROBE": int(f["need_s"]),
                 "RESERVE": BOX.FIT["reserve_s"]} == {"INSTALL_TO_GATE_S": 345, "NEED_PROBE": 11530, "RESERVE": 600}
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_PROBE + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))" in RUN
    assert re.search(r"^FIT_HEADROOM_PCT=125 ", RUN, re.M)
    fit = sum(k.values())
    assert fit == int(f["fit_s"]) == 12475 and fit * 125 // 100 == 15593
    assert (BOX.BOOT_S + fit) * 1.25 <= BOX.GUARD_S == 5 * 3600
    for s in ("**NEED = 11,530 s**", "FIT = 12,475 s", "**15,593 s**", "15,969 s", "**guard of 5.0 h**"):
        assert s in PREREG, s


def test_the_runner_refuses_before_it_installs_in_the_registered_order():
    order = ["sha256sum -c staged-p159.sha256", '[ "$LEFT" -ge "$FIT_NEED" ]', "CUDA_PROBE=$(python -",
             "python3 $W/p159_box.py --plan", "FREE_GB=$(df", "python3.11 -m venv", "python $W/q235_bootstrap.py",
             "python $W/$s --self-test", "can_run $NEED_PROBE probe", "python $W/p159_box.py --out",
             "python $W/p159_reduce.py --result"]
    pos = [RUN.index(s) for s in order]
    assert pos == sorted(pos), list(zip(order, pos))
    assert set(re.findall(r"finish (\d+)", RUN)) == {"0", "3", "9", "10", "13", "16", "18", "21", "22", "23", "40",
                                                      "41", "78", "130"}
    assert "unset OMP_NUM_THREADS MKL_NUM_THREADS TRITON_INTERPRET GLIBC_TUNABLES" in RUN


def _drive(run_id, deadline_s, dry=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E4B_RENT_", "P159_"))}
    env.update(E4B_RENT_RUN_DIR="/tmp/p159-test", E4B_RENT_RUN_ID=run_id, E4B_RENT_INSTANCE_ID="0",
               E4B_RENT_DEADLINE_EPOCH=str(int(__import__("time").time()) + deadline_s), P159_DRIVE_DRYRUN="1")
    if dry:
        env.update(P159_DRY="1", P159_BOX_EXEC="true")
    else:
        env.update(E4B_RENT_SSH_HOST="h", E4B_RENT_SSH_PORT="1", E4B_RENT_SSH_OPTS="-q")
    return subprocess.run(["bash", str(LANE / "p159_drive.sh")], capture_output=True, text=True, env=env, timeout=60)


def test_the_drive_takes_only_declared_ids_and_refuses_a_short_deadline():
    ok = _drive("p159-1", 18000)
    assert ok.returncode == 0 and "DRYRUN stage" in ok.stdout, ok.stdout + ok.stderr
    for bad in ("p159-5", "p159-0", "smt-sib-3"):
        p = _drive(bad, 18000)
        assert p.returncode == 78 and "not declared" in p.stdout, (bad, p.stdout)
    p = _drive("p159-dry-1", 18000)
    assert p.returncode == 78 and "dry run's id" in p.stdout, p.stdout
    assert _drive("p159-dry-1", 600, dry=True).returncode == 0          # the dry run carries no fit check
    p = _drive("p159-2", 15000)                                           # under 12,475 x 1.25 = 15,593 s
    assert p.returncode == 78 and "the registered need 15593s" in p.stdout, p.stdout


# Each mutation must fail the reducer's self-test: the bands' edges, the order of the refutations, each READ-2 clause,
# each minimum count, the two fail-closed engagement checks, and the registered constants.
REDUCER_MUTANTS = [
    ("HI, LO = 0.8, 0.2", "HI, LO = 0.7, 0.2"),
    ("HI, LO = 0.8, 0.2", "HI, LO = 0.8, 0.3"),
    ('    elif n_ps and f_ps <= LO:\n        r1 = "REFUTED_FAULTS"\n'
     '    elif n_nf and s_nf <= LO:\n        r1 = "REFUTED_THP"',
     '    elif n_nf and s_nf <= LO:\n        r1 = "REFUTED_THP"\n'
     '    elif n_ps and f_ps <= LO:\n        r1 = "REFUTED_FAULTS"'),
    (" and hs_s <= HUGE_SLOW_MAX and hs_f >= HUGE_FAST_MIN", ""),
    (" and tf_s > tf_f", ""),
    ("MIN_STATE_BLOCKS, MIN_SLOW_TRIPLES, MIN_B = 12, 6, 3", "MIN_STATE_BLOCKS, MIN_SLOW_TRIPLES, MIN_B = 12, 6, 1"),
    ("MIN_STATE_BLOCKS, MIN_SLOW_TRIPLES, MIN_B = 12, 6, 3", "MIN_STATE_BLOCKS, MIN_SLOW_TRIPLES, MIN_B = 12, 1, 3"),
    ("MIN_STATE_BLOCKS, MIN_SLOW_TRIPLES, MIN_B = 12, 6, 3", "MIN_STATE_BLOCKS, MIN_SLOW_TRIPLES, MIN_B = 1, 6, 3"),
    ("return h is None or h * FUSE_FRESH_BYTES", "return h is not None and h * FUSE_FRESH_BYTES"),
    ("(not control or any(", "(False or any("),
    ("COUNTER_MULTIPLE = 5.0", "COUNTER_MULTIPLE = 1.0"),
    ("B_RATIO = 1.25", "B_RATIO = 1.7"),
    ("NULL_LOG_MAX = math.log(1.5)", "NULL_LOG_MAX = math.log(2.0)"),
    ('if not slow_seen:\n        r1 = "NOT_READ_STATE_NOT_OBSERVED"',
     'if False:\n        r1 = "NOT_READ_STATE_NOT_OBSERVED"'),
]


@pytest.mark.parametrize("old,new", REDUCER_MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p159_reduce.py").read_text()
    assert src.count(old) == 1, old
    m = tmp_path / "p159_reduce.py"
    m.write_text(src.replace(old, new))
    p = subprocess.run([sys.executable, str(m), "--self-test"], capture_output=True, text=True, timeout=120)
    assert p.returncode != 0, f"mutant survived: {old!r} -> {new!r}"
