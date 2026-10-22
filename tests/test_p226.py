# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P226, the K25 plan at bucket 32 (bench/p226/PREREG-p226.md). The self-tests; every runner call through its script's
own argument check; the staged pins; the run ids across runner, driver and PREREG; the control equal to e4b's c1; the
256-row bound that keeps bucket 32 on K25; the selection and confirmation steps disjoint; the grouped-nf4-gemm pin; the
capture's window fitting its row budget. (Private names are the repository guards' job, not this file's.)"""
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
LANE = REPO / "bench" / "p226"
PREREG = (LANE / "PREREG-p226.md").read_text(encoding="utf-8")
RUN = (LANE / "p226_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p226_drive.sh").read_text(encoding="utf-8")
SOURCES = {"p226_run.sh": LANE, "p226_capture.py": LANE, "p226_bench.py": LANE, "p226_reduce.py": LANE,
           "p154_route.py": REPO / "bench" / "p154", "p136_box.py": REPO / "bench" / "p136",
           "sc1b_e4b_census.py": REPO / "bench" / "sc1b", "p109_box.py": REPO / "bench" / "p109",
           "k8_bake.py": REPO / "bench" / "p39", "calib.json": REPO / "bench" / "p39",
           "test_k25_row_exact_gpu.py": REPO / "tests"}
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(LANE), str(REPO / "bench" / "p154")])}
GNF4_V050 = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"      # grouped-nf4-gemm v0.50.0: K39's prmt decode


def _load(name):
    d = LANE if (LANE / f"{name}.py").exists() else REPO / "bench" / "p154"
    spec = importlib.util.spec_from_file_location(name, d / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p154")]
    try:
        spec.loader.exec_module(mod)
    finally:
        del sys.path[:2]
    return mod


@pytest.mark.parametrize("name", ["p226_reduce", "p226_bench", "p226_capture"])
def test_self_tests_pass(name):
    if name == "p226_capture":
        pytest.importorskip("torch")
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


@pytest.mark.parametrize("script", ["p226_capture.py", "p226_bench.py", "p226_reduce.py"])
def test_every_runner_call_passes_the_scripts_own_argument_check(script):
    calls = [c for c in _calls(script) if "--self-test" not in c]
    assert calls, script
    for argv in calls:
        r = subprocess.run([sys.executable, str(LANE / script), *argv, "--check-args"], capture_output=True, text=True,
                           timeout=60, cwd=str(LANE), env=ENV)
        assert r.returncode == 0 and "ARGS OK" in r.stdout, (argv, r.stdout + r.stderr)


def test_staged_pins_match_their_sources():
    lines = [ln.split() for ln in (LANE / "staged-p226.sha256").read_text(encoding="ascii").splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert sorted(n for _, n in lines) == sorted(SOURCES)
    for want, name in lines:
        got = hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: staged {want}, source {got}"


def test_run_ids_agree_across_runner_driver_and_prereg():
    for rid in (f"p226-5090-{i}" for i in (1, 2, 3)):
        assert rid in RUN and rid in DRIVE
    assert "`p226-5090-1`" in PREREG and "p154-5090" not in RUN + DRIVE


def test_the_control_is_e4bs_c1_and_the_grid_holds_it():
    R = _load("p226_reduce")
    hr = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
    m = re.search(r'"c1": \((\{[^}]*\}), (\{[^}]*\})\)', hr)
    assert m, "e4b's _K25_PLANS['c1']"
    import ast
    assert (ast.literal_eval(m[1]), ast.literal_eval(m[2])) == (R.C1["gu"], R.C1["dn"])
    assert R.C1["gu"] in R.GRID and R.C1["dn"] in R.GRID and len(R.GRID) == 144


def test_bucket_32_stays_on_k25_and_bucket_64_does_not():
    hr = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
    assert "device_grouping and R_rows <= 256 and _k25_mode" in hr          # K25's row bound in the served route
    R, C = _load("p226_reduce"), _load("p226_capture")
    assert R.TOKENS * R.TOPK == 256 and C.BATCH == R.TOKENS and 64 * R.TOPK > 256
    assert C.ROUTE_KEY == "nf4_k25|le256" and C.PLAN_KEY == "c1|le256"


def test_selection_and_confirmation_are_disjoint_and_registered():
    R = _load("p226_reduce")
    assert R.SELECT_STEPS == tuple(range(0, 64, 8)) and R.CONFIRM_STEPS == tuple(range(64, 128))
    assert not set(R.SELECT_STEPS) & set(R.CONFIRM_STEPS)
    assert (R.CARRY, R.ITERS, R.GO_BAR, R.SELF_BAND, R.MIN_POWER_W) == (3, 10, 1.05, (0.99, 1.01), 575)
    assert R.ARMS == ["C", "C2", "K1", "K2", "K3", "D"] and R.LUT == "prmt"
    for text in ("**1.05**", "[0.99, 1.01]", "575 W", "Steps 64-127", "0, 8, ..., 56", "seed 226"):
        assert text in PREREG, text


def test_the_runner_pins_gnf4_v050_and_asserts_the_served_decode_and_plan():
    assert f"GNF4_SHA={GNF4_V050}" in RUN and 'md.version("grouped-nf4-gemm") == "0.50.0"' in RUN
    assert 'lut.get("lut") == "prmt"' in RUN and 'plan.get("plan") == "c1"' in RUN
    assert "E4B_PAGED_MAX_SEQS=32" in RUN and "E4B_PAGED_GRAPHS=0" in RUN and "E4B_INT8_HEAD_BW=0" not in RUN


def test_the_capture_window_fits_its_row_budget_and_uses_new_prompts():
    C = _load("p226_capture")
    assert C.SKIP + C.STEPS + C.BATCH - 1 < C.TOKENS                      # the first-admitted row cannot retire inside
    assert (C.FIRST_ROW, C.N_ROWS, C.STEPS, C.TOKENS) == (16, 32, 128, 256)
    assert "wikitext_rows(tok, n=48)[16:48]" in PREREG


def test_the_runner_captures_before_it_times_and_finishes_on_the_verdict():
    i_cap, i_bench = RUN.index("python $W/p226_capture.py --prompts"), RUN.index("python $W/p226_bench.py --picks")
    assert i_cap < i_bench < RUN.index("python $W/p226_reduce.py --bench")
    assert 'case "$V" in GO|NO_GO) finish 0;; *) say "P226 $V -- no reading"; finish 27;; esac' in RUN
    for need in ("NEED_FETCH=1210", "NEED_BAKE=550", "NEED_CAPTURE=900", "NEED_BENCH=2400"):
        assert need in RUN, need
