# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P158 (bench/p158/PREREG-p158.md; e4b#1313), the served read of K39 (grouped-nf4-gemm v0.50.0's K25
``lut="prmt"``) through E4B_K25_LUT: the staged pins (every piece shared with P153's kit at P153's pins, K39's compiled
contract at grouped-nf4-gemm's bytes, e4b's served-route contract), the runner's order (the power floor and the guard
fit before the install, the premise at 49, nine processes at five reps, K39's kernel bytes in the tripwire, no SANE
phase), the guard fit with 25 % headroom on both sides, the self-tests run as the box runs them (one directory), P153's
pieces carried verbatim, the mutants the self-tests must kill, and the driver's run ids (-1..-3)."""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p158"
P153 = B / "p153"
PREREG = (LANE / "PREREG-p158.md").read_text(encoding="utf-8")
SOURCES = {"p158_run.sh": LANE, "p158_box.py": LANE, "p158_reduce.py": LANE, "p109_box.py": B / "p109",
           "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "test_decode_graph_buckets.py": REPO / "tests", "test_kv_step_select.py": REPO / "tests",
           "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise", "test_gemv_bw_served_gpu.py": REPO / "tests",
           "test_nf4_smallm_prmt_gpu.py": LANE / "premise", "test_k25_lut_gpu.py": REPO / "tests"}
#: grouped-nf4-gemm kernel/test_nf4_smallm_prmt_gpu.py at v0.50.0 (41ef086e6a3e2be5b158ff1bb38e0de67862a949), its sha256
K39_COMPILED_TEST_SHA256 = "9f6366287957f75bdfe53ff8165b401c30fb9e97d4da582254ad7e98bf702868"
GUARD_H = 1.5
BOOT_S = 300          # the launcher's create-to-runner allowance (P152's, P154's and P155's, kept)
HEADROOM = 1.25


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
    pins = _pins(LANE / "staged-p158.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_every_piece_shared_with_p153_is_at_p153s_pins_and_the_premise_is_k39s_file():
    mine, p153 = _pins(LANE / "staged-p158.sha256"), _pins(P153 / "staged-p153.sha256")
    shared = set(mine) & set(p153)
    assert shared == {"p109_box.py", "k8_bake.py", "calib.json", "test_decode_graph_buckets.py", "test_kv_step_select.py",
                      "test_fused_glue_decode_graphs_gpu.py", "test_gemv_bw_served_gpu.py"}
    assert all(mine[n] == p153[n] for n in shared)
    assert mine["test_nf4_smallm_prmt_gpu.py"] == K39_COMPILED_TEST_SHA256
    assert not {"p115_quality.py", "p110_box.py", "p108_box.py", "p97_box.py"} & set(mine), "no SANE phase"


def test_the_runner_refuses_below_575_w_and_without_the_fit_before_the_install_and_reads_only_a_rule_verdict():
    run = (LANE / "p158_run.sh").read_text(encoding="utf-8")
    assert "GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949" in run and 'md.version("grouped-nf4-gemm") == "0.50.0"' in run
    assert "p158-5090-1|p158-5090-2|p158-5090-3) MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;" in run
    assert "SHORT=32; LONG=160; REPS=5\n" in run and "MIN_POWER_W=575 " in run
    assert 'finish 17; }' in run and 'echo "refused: power.limit unreadable" > REFUSAL; finish 15; }' in run
    assert "E4B_ATTN_SPLIT_CAP E4B_DENSE_GEMV_BW E4B_INT8_HEAD_BW E4B_K25_LUT\n" in run, "every knob starts unset"
    assert "ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568" in run, "K25/K39's kernel bytes in the tripwire"
    assert 'assert sp.INT8_HEAD_BW_DEFAULT_FAMILIES == {"qwen3_moe": ((151936, 2048),)}' in run, "the int8 head's default"
    assert 'assert sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES == {"qwen3_moe": 16}' in run
    order = ["for f in p158_box.py", "PL=$(nvidia-smi --query-gpu=power.limit", '[ "$LEFT" -ge "$FIT_NEED" ]',
             "pipx logs/pip_e4b.log", "grep -qE '^49 passed", "snapshot_download", "python $W/k8_bake.py", "--prompts-only",
             "for TAG in S0w S0a S1a S1b S0b S0c S1c S1d S0d; do", "python $W/p158_reduce.py --dir"]
    at = [run.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert "test_gemv_bw_served_gpu.py test_nf4_smallm_prmt_gpu.py test_k25_lut_gpu.py -q -rs" in run
    assert run.count("E4B_K25_LUT=$DV P158_ARM=$ARM") == 1 and 'DV=tree; [ "$ARM" = S1 ] && DV=prmt' in run
    for knob in ("E4B_ATTN_SPLIT_CAP=", "E4B_DENSE_GEMV_BW=", "E4B_INT8_HEAD_BW=", "E4B_NF4_GROUPED_SMALLM="):
        assert knob not in run, f"{knob} stays unset (its default) in both arms"
    assert "--mode sane" not in run and "p115_quality" not in run, "no SANE phase"
    assert 'case "$V" in FLIP|NO_FLIP|FUNCTION_FAIL) ;;' in run, "NOISY and VOID are not readings"


def test_the_guard_fits_with_25_percent_headroom():
    """e4b#1845's rule on both sides: the runner refuses before the install below FIT x 125 %, and the registered guard
    holds the boot plus that fit with 25 % headroom."""
    run = (LANE / "p158_run.sh").read_text(encoding="utf-8")
    val = {k: int(re.search(rf"^{k}=(\d+)", run, re.M).group(1)) for k in ("INSTALL_TO_GATE_S", "NEED_BENCH", "RESERVE")}
    fit = val["INSTALL_TO_GATE_S"] + val["NEED_BENCH"] + val["RESERVE"]
    assert (val["INSTALL_TO_GATE_S"], val["NEED_BENCH"], val["RESERVE"]) == (600, 2100, 600) and fit == 3300
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE))" in run and "FIT_HEADROOM_PCT=125" in run
    assert (BOOT_S + fit) * HEADROOM <= GUARD_H * 3600
    assert "(300 + 3300) x 1.25 = 4500 s <= 5400 s (1.5 h)" in PREREG and "**guard 1.5 h**" in PREREG
    # NEED_BENCH: the larger of the two measured served reads' fetch-to-reduce walls, at nine processes, x1.5
    assert math_ceil_to(1.5 * max(1347, 1039), 100) == val["NEED_BENCH"]


def math_ceil_to(x, step):
    return -(-int(round(x)) // step) * step


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p158_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


@pytest.mark.parametrize("script", ["p158_box.py", "p158_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_p153s_checks_are_carried_verbatim():
    p153 = (P153 / "p153_reduce.py").read_text(encoding="utf-8")
    mine = (LANE / "p158_reduce.py").read_text(encoding="utf-8")
    for a, b in (("def _rate(r, w):", "def record_faults("), ("def record_faults(", "def cap_faults(")):
        piece = p153[p153.index(a):p153.index(b)]
        if a == "def _rate(r, w):":            # divergence's docstring dropped "reported, never ruled": here it is a gate
            piece = piece.replace('    """Rows identical out of the rows, and each row\'s first divergent step (None when '
                                  'identical): reported, never ruled."""',
                                  '    """Rows identical out of the rows, and each row\'s first divergent step (None when '
                                  'identical)."""')
        assert piece in mine, a


def test_the_k25_build_count_is_derived_from_the_capture_code():
    """576 = 48 MoE layers x the four multi-row buckets x (enable_decode_graphs' two warm-ups + one capture): the warm-up
    default, the bucket set and T == 1's route are read from the code the box installs."""
    runner = (REPO / "experts4bit_qlora" / "engines" / "paged_runner.py").read_text(encoding="utf-8")
    assert re.search(r"def enable_decode_graphs\(self, buckets=DEFAULT_BUCKETS, \*, capture: bool = True,\s+"
                     r"warmup: int = 2,", runner)
    assert "for _ in range(max(1, warmup)):" in runner and "with torch.cuda.graph(g):\n                self._padded_step(b)" in runner
    hr = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
    assert "    singleton = T == 1 or" in hr and "R_rows <= 256 and _k25_mode != \"0\"" in hr
    serve = (REPO / "experts4bit_qlora" / "serve_paged.py").read_text(encoding="utf-8")
    assert "runner.enable_decode_graphs(cfg.buckets) if cfg.graphs else None" in serve
    sys.path.insert(0, str(LANE))
    try:
        import p158_reduce as r
    finally:
        sys.path.remove(str(LANE))
    assert r.K25_BUILD_CALLS == 48 * 4 * 3 == 576 and r.K25_BUCKETS == (2, 4, 8, 16)


REDUCER_MUTANTS = [
    # g16 from the best pair, not the worst of four
    ('    g16_min, g1_min = min(pairs["W16"].values()), min(pairs["W1"].values())',
     '    g16_min, g1_min = max(pairs["W16"].values()), min(pairs["W1"].values())'),
    # only the first ABBA block's pairs
    ('PAIRS = (("S1a", "S0a"), ("S1b", "S0b"), ("S1c", "S0c"), ("S1d", "S0d"))', 'PAIRS = (("S1a", "S0a"), ("S1b", "S0b"))'),
    # the NOISY band widened
    ("NOISY_BAND = (0.985, 1.015)", "NOISY_BAND = (0.98, 1.02)"),
    # NOISY read at W1 only (P153's form): the bar is on W16
    ("             for w in WORKLOADS for k, v in selfp[w].items() if not NOISY_BAND[0]",
     "             for w in (\"W1\",) for k, v in selfp[w].items() if not NOISY_BAND[0]"),
    # NOISY never read
    ("    if noisy:\n        out.update(verdict=\"NOISY\"", "    if False:\n        out.update(verdict=\"NOISY\""),
    # the warm-up process's absence accepted
    ('    missing = [t for t in TAGS + (WARMUP,) if t not in arms', '    missing = [t for t in TAGS if t not in arms'),
    # the five reps never checked
    ('        if r.get("reps") != REPS:', "        if False:"),
    # the W1 guard dropped from the flip
    ("    elif g16_min >= FLIP_G16 and g1_min >= NOREG_G1:", "    elif g16_min >= FLIP_G16:"),
    # the bars read the ROUNDED mins
    ("    elif g16_min >= FLIP_G16 and g1_min >= NOREG_G1:",
     "    elif round(g16_min, 4) >= FLIP_G16 and round(g1_min, 4) >= NOREG_G1:"),
    # THE NULL never read: a non-deterministic tree reads as a contract break against K39
    ('    if null:\n        out.update(verdict="VOID"', '    if False:\n        out.update(verdict="VOID"'),
    # the null without the warm-up
    ('    null = _token_diffs(arms, ("S0b", "S0c", "S0d", WARMUP), "S0a", lens)',
     '    null = _token_diffs(arms, ("S0b", "S0c", "S0d"), "S0a", lens)'),
    # the identity gate never read
    ('    if fn:\n        out["contract"]', '    if False:\n        out["contract"]'),
    # S1 held to S1a's tokens only (the contract is S1 == S0)
    ('    fn = _token_diffs(arms, ("S1a", "S1b", "S1c", "S1d"), "S0a", lens)',
     '    fn = _token_diffs(arms, ("S1b", "S1c", "S1d"), "S1a", lens)'),
    # S1's own timed reps unchecked
    ('lens) + _rep_diffs(arms, ("S1a", "S1b", "S1c", "S1d"))', 'lens)'),
    # K25's engagement never checked
    ("        void += k25_faults(t, r, t[:2])\n", ""),
    # the build count not exact
    ('    if r.get("k25_seen_build") != {key: K25_BUILD_CALLS}:',
     '    if not (r.get("k25_seen_build") or {}).get(key):'),
    # the decode's resolution unchecked
    ('    if (k.get("env"), k.get("lut"), k.get("why")) != (want, want, None):', '    if k.get("lut") != want:'),
    # another decode accepted after the runs
    ('    if set(after) != {key} or after.get(key, 0) < K25_BUILD_CALLS:', '    if after.get(key, 0) < K25_BUILD_CALLS:'),
    # the run counts not compared across processes
    ('        if len(set(runs.values())) != 1:', '        if False:'),
    # the int8 head's default unchecked
    ("        void += default_faults(t, r)\n", ""),
    # an unreadable enforced limit accepted
    ("                    if pl is None or pl < MIN_POWER_W:", "                    if pl is not None and pl < MIN_POWER_W:"),
    # only power.limit read, not the enforced limit
    ('                for f in ("power.limit", "enforced.power.limit"):', '                for f in ("power.limit",):'),
    # the floor lowered to a capped board's
    ("MIN_POWER_W = 575 ", "MIN_POWER_W = 450 "),
    # an explicit cap of 16 taken for the default
    ('    if sc.get("cap") != CAP[arm] or sc.get("source") != "default-allowlisted":', '    if sc.get("cap") != CAP[arm]:'),
    # the clock label one-sided
    ("        elif 100 * abs(ca - cb) > CLOCK_TOL_PCT * cb:", "        elif 100 * (ca - cb) > CLOCK_TOL_PCT * cb:"),
    # an unread clock read as a value
    ("    return statistics.median(xs) if xs else None", "    return statistics.median(xs) if xs else 0.0"),
]


@pytest.mark.parametrize("old,new", REDUCER_MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p158_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p158_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p158_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


BOX_MUTANTS = [
    # K25's mode may be set in an arm
    ("    bad.update({k: env.get(k) for k in (K25_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS",
     "    bad.update({k: env.get(k) for k in (HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS"),
    # the int8 head may be set in an arm
    ("    bad.update({k: env.get(k) for k in (K25_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS",
     "    bad.update({k: env.get(k) for k in (K25_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS"),
    # the arms' values swapped
    ('ARM_VALUE = {"S0": "tree", "S1": "prmt"}', 'ARM_VALUE = {"S0": "prmt", "S1": "tree"}'),
    # an unreadable field read as zero
    ("        return float(v)\n    except (TypeError, ValueError):\n        return None",
     "        return float(v)\n    except (TypeError, ValueError):\n        return 0.0"),
    # a malformed sample line kept
    ("        if len(vals) != len(SAMPLE):\n            continue", "        if not vals:\n            continue"),
]


@pytest.mark.parametrize("old,new", BOX_MUTANTS)
def test_each_box_mutant_fails_the_self_test(box_dir, tmp_path, old, new):
    src = (LANE / "p158_box.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p158_box.py").write_bytes(src.replace(old, new).encode("utf-8"))
    shutil.copyfile(box_dir / "p109_box.py", tmp_path / "p109_box.py")
    out = _run(tmp_path / "p158_box.py", "--self-test", cwd=tmp_path)
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P158_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


@pytest.mark.parametrize("run_id,ok", [("p158-5090-1", True), ("p158-5090-3", True), ("p158-5090-4", False),
                                       ("p153-5090-1", False)])
def test_the_driver_takes_three_run_ids_and_no_other(tmp_path, run_id, ok):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p158_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, run_id), timeout=120)
    if ok:
        assert out.returncode == 0 and "DRYRUN stage -> root@h:/root/p158" in out.stdout and "bash p158_run.sh" in out.stdout, \
            out.stdout + out.stderr
    else:
        assert out.returncode == 78 and "is not p158-5090-N" in out.stdout + out.stderr
