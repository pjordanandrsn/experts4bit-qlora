# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P230, the PDL served read on the default NF4 server (bench/p230/PREREG-p230.md). The self-tests; every runner call
through its script's own argument check; the staged pins; the run ids across runner, driver and PREREG; the
grouped-nf4-gemm pin; the arms' palindrome; the pre-install fit, its boot slack against the launcher's ssh wait, and the
start marker's place; every pre-start exit code classified. (Private names are the repository guards' job, not this
file's.)"""
import hashlib
import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p230"
PREREG = (LANE / "PREREG-p230.md").read_text(encoding="utf-8")
RUN = (LANE / "p230_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p230_drive.sh").read_text(encoding="utf-8")
SOURCES = {"p230_run.sh": LANE, "p230_box.py": LANE, "p230_reduce.py": LANE, "p109_box.py": REPO / "bench" / "p109",
           "k8_bake.py": REPO / "bench" / "p39", "calib.json": REPO / "bench" / "p39",
           "test_decode_graph_buckets.py": REPO / "tests"}
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(LANE), str(REPO / "bench" / "p109")])}
GNF4_V050 = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"      # grouped-nf4-gemm v0.50.0
RUN_IDS = ["p230-5090-1", "p230-5090-2"] + [f"p230-5090-r{i}" for i in range(1, 9)]
SSH_READY_S, STAGING_MARGIN_S = 420, 300                     # the launch manifest's ssh_ready_s; staging and the checks
GUARD_H = 2.1


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p109")]
    try:
        spec.loader.exec_module(mod)
    finally:
        del sys.path[:2]
    return mod


@pytest.mark.parametrize("name", ["p230_reduce", "p230_box"])
def test_self_tests_pass(name):
    r = subprocess.run([sys.executable, str(LANE / f"{name}.py"), "--self-test"], capture_output=True, text=True,
                       timeout=300, cwd=str(LANE), env=ENV)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def _joined_lines(text):
    out, cur = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    return out


def _calls(script):
    calls = []
    for line in _joined_lines(RUN):
        m = re.search(r"python \$W/" + re.escape(script) + r"(?P<rest>.*)", line)
        if not m:
            continue
        rest = re.split(r"\s(?:\d?>|>>|\||;|&&|\|\|)", " " + m["rest"])[0]
        calls.append([re.sub(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", "X", tok) for tok in shlex.split(rest)])
    return calls


@pytest.mark.parametrize("script", ["p230_box.py", "p230_reduce.py"])
def test_every_runner_call_passes_the_scripts_own_argument_check(script):
    calls = [c for c in _calls(script) if "--self-test" not in c]
    assert calls, script
    for argv in calls:
        argv = ["32" if a == "X" and i and argv[i - 1] == "--short" else "160" if a == "X" and i and argv[i - 1] == "--long"
                else "3" if a == "X" and i and argv[i - 1] == "--reps" else a for i, a in enumerate(argv)]
        r = subprocess.run([sys.executable, str(LANE / script), *argv, "--check-args"], capture_output=True, text=True,
                           timeout=60, cwd=str(LANE), env=ENV)
        assert r.returncode == 0 and "ARGS OK" in r.stdout, (argv, r.stdout + r.stderr)


def test_staged_pins_match_their_sources():
    lines = [ln.split() for ln in (LANE / "staged-p230.sha256").read_text(encoding="ascii").splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert sorted(n for _, n in lines) == sorted(SOURCES)
    for want, name in lines:
        got = hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: staged {want}, source {got}"


def test_run_ids_agree_across_runner_driver_and_prereg():
    case = "|".join(RUN_IDS) + ")"
    assert case in RUN and case in DRIVE
    for bad in ("p230-5090-3", "p230-5090-r9"):
        assert bad not in RUN and bad not in DRIVE, bad
    assert "`p230-5090-1`" in PREREG and "`p230-5090-r1` to `p230-5090-r8`" in PREREG


def test_the_runner_and_reducer_pin_gnf4_v050_and_the_shipped_switch():
    R = _load("p230_reduce")
    assert f"GNF4_SHA={GNF4_V050}" in RUN and R.GNF4_SHA == GNF4_V050
    assert 'md.version("grouped-nf4-gemm") == "0.50.0"' in RUN
    assert "int4_b32.PDL_DEFAULT is True and int4_b32.PDL_MAX_ROWS_DEFAULT == 8" in RUN
    assert '"_pdl_kw(dev, T)" in inspect.getsource(nf4_grouped._gemv_nf4_bw_launch)' in RUN
    assert "nf4_grouped._bw_engages(1536, 2048" in RUN and "nf4_grouped._bw_engages(2048, 768" in RUN


def test_the_arms_are_a_palindrome_and_differ_only_in_the_switch():
    R, B = _load("p230_reduce"), _load("p230_box")
    assert R.TAGS == ("OFF1", "DEF1", "DEF2", "OFF2") and "for TAG in OFF1 DEF1 DEF2 OFF2; do" in RUN
    assert 'case "$ARM" in OFF) SS="GNF4_PDL=0";; DEF) SS="";; esac' in RUN
    assert B.ARMS == {"OFF": {"GNF4_PDL": "0", "GNF4_PDL_MAX_ROWS": None},
                      "DEF": {"GNF4_PDL": None, "GNF4_PDL_MAX_ROWS": None}}
    assert "GNF4_PDL GNF4_PDL_MAX_ROWS" in RUN                          # unset before the arms: DEF is the shipped default
    assert set(R.MUST_LAUNCH) <= set(B.SWITCHED) and "_gemv_nf4_bw" in B.SWITCHED_NF4


def test_function_fail_is_read_before_the_timing_rungs():
    src = (LANE / "p230_reduce.py").read_text(encoding="utf-8")
    body = src[src.index("    if fn:\n        out.update(verdict=\"FUNCTION_FAIL\""):]
    assert body.index("FUNCTION_FAIL") < body.index("NOISY") < body.index("KEEP")
    assert PREREG.index("**FUNCTION_FAIL.**") < PREREG.index("**NOISY.**") < PREREG.index("**KEEP.**")


def test_the_box_tags_each_launch_with_its_bucket_for_the_build_only():
    src = (LANE / "p230_box.py").read_text(encoding="utf-8")
    assert "paged_runner.PagedModelRunner._capture_bucket = _tagged_capture" in src
    assert src.index("_capture_bucket = _tagged_capture") < src.index("parts = build_engine(cfg)") \
        < src.index("paged_runner.PagedModelRunner._capture_bucket = orig_capture")
    assert '"pdl_by_bucket": by_bucket' in src


def test_the_pre_install_fit_meets_the_guard_with_boot_slack():
    fit = 450 + 1210 + 550 + 4 * 600 + 600
    fit_need = fit * 125 // 100
    guard_s = round(GUARD_H * 3600)
    assert fit == 5210 and fit_need == 6512 and guard_s == 7560
    assert guard_s - fit_need >= SSH_READY_S + STAGING_MARGIN_S, guard_s - fit_need       # slack 1,048 s
    assert GUARD_H <= 5.9 and abs(GUARD_H * 1.05 + 66 * 0.011 - 2.93) < 0.005
    assert abs(2 * 2.93 + (len(RUN_IDS) - 2) * 0.31 - 8.34) < 0.005          # the ids bound the boxes at 10
    for s in ("5,210 s", "6,512 s", "Guard 2.1 h (7,560 s)", "1,048 s", "USD 2.93", "USD 8.34"):
        assert s in PREREG, s
    for need in ("NEED_FETCH=1210", "NEED_BAKE=550", "NEED_ARM=600", "INSTALL_TO_GATE_S=450", "RESERVE=600"):
        assert need in RUN, need
    i_fit, i_start = RUN.index('[ "$LEFT" -ge "$FIT_NEED" ]'), RUN.index(": > $W/STARTED")
    assert i_fit < i_start < RUN.index("python -m pip install") and i_start < RUN.index("python $W/p230_reduce.py")


def test_every_exit_before_the_start_is_classified():
    R = _load("p230_reduce")
    pre = RUN[:RUN.index(": > $W/STARTED")]
    codes = {int(c) for c in re.findall(r"finish (\d+)", pre)} - {130}   # 130 is the trap's
    assert codes, "no pre-start exits found"
    unknown = sorted(c for c in codes if c not in R.HOST_CODES and c not in R.FIT_CODES and c not in R.KIT_CODES)
    assert not unknown, f"pre-start exit codes the classification does not name: {unknown}"
    assert {13, 15, 16, 18} == set(R.HOST_CODES) and {41} == set(R.FIT_CODES) and {9, 10, 28, 78, 130} <= set(R.KIT_CODES)
    for rc in R.HOST_CODES:
        assert R.classify(rc, False)["event"] == "HOST_REFUSED"
    fit = R.classify(41, False)
    assert fit["event"] == "FIT_REFUSED" and not fit["halt"] and not fit["started"]     # a slow boot never halts the lane
    assert R.classify(0, True)["event"] == "ANSWER"
    assert R.classify(27, True)["event"] == "NO_ANSWER"


def test_the_runner_finishes_on_the_verdict():
    assert 'case "$V" in KEEP|NULL|SLOWER|FUNCTION_FAIL) finish 0;; *) say "P230 $V -- no reading"; finish 27;; esac' in RUN
    R = _load("p230_reduce")
    assert R.ANSWERS == ("KEEP", "NULL", "SLOWER", "FUNCTION_FAIL")
    assert (R.SELF_LO, R.SELF_HI, R.KEEP_BAR, R.SLOWER_BAR) == (0.995, 1.005, 1.005, 0.995)
    for s in ("[0.995, 1.005]", "1.005", "0.995"):
        assert s in PREREG, s
