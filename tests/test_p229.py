# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P229, K2b's microbench (bench/p229/PREREG-p229.md). The staged pins (the premise's files in the repo's layout); the
registered constants as the PREREG states them; the runner's order, installs, tripwire, premise and guard fit; the
drive's refusals; the runner's verdict cases; the self-tests and the reducer mutants they must kill."""
import hashlib
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p229"
#: staged path (as the box sees it) -> the file in this checkout
SOURCES = {"p229_bench.py": LANE / "p229_bench.py", "p229_reduce.py": LANE / "p229_reduce.py",
           "p229_run.sh": LANE / "p229_run.sh",
           "t/tests/test_int8_head_bw.py": REPO / "tests" / "test_int8_head_bw.py",
           "t/tests/test_int8_head_rows_interp.py": REPO / "tests" / "test_int8_head_rows_interp.py",
           "t/tests/conftest.py": REPO / "tests" / "conftest.py",
           "t/bench/i8dense/i8_fakequant.py": REPO / "bench" / "i8dense" / "i8_fakequant.py",
           "t/bench/p148/p148_kernels.py": REPO / "bench" / "p148" / "p148_kernels.py"}
RUN = (LANE / "p229_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p229_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p229.md").read_text(encoding="utf-8")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


R = _load("p229_reduce_under_test", LANE / "p229_reduce.py")


def test_every_staged_file_is_pinned_and_matches():
    pins = dict(reversed(line.split()) for line in (LANE / "staged-p229.sha256").read_text(encoding="ascii").splitlines())
    assert set(pins) == set(SOURCES)
    for name, path in SOURCES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins[name], name


def test_the_premise_counts_are_the_suites_own():
    """45 and 11 are what the staged suites hold: the CPU suite's items, and the interpreter file's 7 plus its 4
    CUDA-graph replays (each parametrized test counted once per case)."""
    cpu =subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          str(REPO / "tests" / "test_int8_head_bw.py")], capture_output=True, text=True, cwd=REPO)
    assert re.search(r"^45 tests collected", cpu.stdout, re.M), cpu.stdout[-500:]
    src = (REPO / "tests" / "test_int8_head_rows_interp.py").read_text(encoding="utf-8")
    assert src.count("\ndef test_") == 5 and '@pytest.mark.parametrize("rows", [1, 2, 3, 8])' in src \
        and '@pytest.mark.parametrize("rows", [1, 2, 4, 8])' in src            # 1 + 4 + 1 + 1 + 4 = 11


def test_the_self_tests_pass():
    R.self_test()
    r = subprocess.run([sys.executable, str(LANE / "p229_bench.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def test_the_registered_constants_and_the_prereg_agree():
    assert (R.BAR_D, R.BAR_K, R.TOL, R.MS, R.SELF_BAND, R.MIN_POWER_W, R.CLOCK_TOL) == \
        (0.70, 1.25, 1.5e-2, (2, 4, 8), (0.99, 1.01), 575, 0.02)
    assert R.HEAD == {"N": 151936, "K": 2048} and R.SHIPPED_PLAN == [32, 256, 4]
    assert R.ROWS_RECORD == {"rows_max": 8, "kernel": "dense_i8.linear_i8 (K37)",
                             "plan": {str(m): [32, 256, 4] for m in range(2, 9)}}
    assert R.GNF4_COMMIT == "41ef086e6a3e2be5b158ff1bb38e0de67862a949"
    assert R.ARMS == ["D", "D2", "I", "K"] and R.ITERS == 10 and R.READ_STEPS == tuple(range(60))
    for s in ("**0.70**", "**1.25x**", "**max |I - D| / max |D| <= 1.5e-2**", "seed 229", "seed 22900 + M",
              "`41ef086e6a3e2be5b158ff1bb38e0de67862a949`", "plan: (32, 256, 4) at EVERY M in 2..8", "under 575 W",
              "exactly **11 passed**", "exactly **45 passed**", "**$1.45**", "GO_ROWS_8 / GO_ROWS_4 / GO_ROWS_2",
              "A miss at 4 caps it at 2 even if 8 passes",
              "a code PR first lowers the route's `MAX_ROWS` to N, refusing larger values by name"):
        assert s in PREREG, s


def test_the_runner_installs_the_release_then_e4b_checks_the_route_and_runs_the_premise_on_the_card():
    assert "GNF4=41ef086e6a3e2be5b158ff1bb38e0de67862a949" in RUN and R.GNF4_COMMIT in RUN
    assert '--no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4"' in RUN
    assert 'assert g.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"]' in RUN
    assert "assert all(tuple(dense_i8.plan(151936, 2048, m)) == (32, 256, 4) for m in range(2, 9))" in RUN
    assert "TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest tests/test_int8_head_rows_interp.py" in RUN
    assert "grep -qE '^11 passed" in RUN and "grep -qE '^45 passed" in RUN and "cd $W/t &&" in RUN
    order = ["sha256sum -c staged-p229.sha256", "PL=$(nvidia-smi --query-gpu=power.limit", '[ "$LEFT" -ge "$FIT_NEED" ]',
             "pipx logs/pip_gnf4.log", "pipx logs/pip_e4b.log", "TRIPWIRE FAIL", "grep -qE '^11 passed",
             "p229_bench.py --out", "p229_reduce.py --bench"]
    at = [RUN.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert "MIN_POWER_W=575 " in RUN and "finish 17; }" in RUN
    assert "p229-5090-1|p229-5090-2|p229-5090-3|" in RUN and "p229-shake-1|p229-shake-2|p229-shake-3) SHAKE=1;;" in RUN
    assert "--shakedown --reps 4)" in RUN
    for knob in ("E4B_INT8_HEAD_ROWS", "E4B_INT8_HEAD_BW", "E4B_PAGED_SPEC", "TRITON_INTERPRET"):
        assert re.search(rf"^unset .*|^\s+.*\b{knob}\b", RUN, re.M) and f" {knob}" in RUN.split("unset ", 1)[1].split(
            "\n: > summary.txt")[0], knob


def test_the_runners_verdict_cases():
    assert "  1/SHAKEDOWN) finish 0;;" in RUN
    assert "  0/GO_ROWS_8|0/GO_ROWS_4|0/GO_ROWS_2|0/NO_GO|0/TOLERANCE_FAIL) finish 0;;" in RUN
    assert '  0/*) say "P229 $V -- not a reading"; finish 27;;' in RUN
    assert set(R.READ_VERDICTS) == {"GO_ROWS_8", "GO_ROWS_4", "GO_ROWS_2", "NO_GO", "TOLERANCE_FAIL"}


def test_the_guard_fit():
    get = lambda v: int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))  # noqa: E731
    install, need, reserve = get("INSTALL_TO_GATE_S"), get("NEED_BENCH"), get("RESERVE")
    assert (install, need, reserve) == (900, 600, 600)
    fit = install + need + reserve
    assert fit == 2100 and int(fit * 1.25) == 2625 and (300 + fit) * 1.25 == 3000 <= 3600
    for s in ("`NEED_BENCH` = **600 s**", "`INSTALL_TO_GATE_S` = **900 s**", "`FIT` = 900 + 600 + 600 = 2100 s",
              "2100 x 1.25 = 2625 s", "(boot 300 s + 2100 s) x 1.25 = 3000 s <= 3600 s (1.0 h)"):
        assert s in PREREG, s
    assert abs(1.0 * 1.05 + 10 * 0.04 - 1.45) < 1e-9
    assert 3 * 60 * 80 * 1.0 / 1000 == 14.4


SSH_READY_S, STAGING_MARGIN_S = 420, 120   # pod-launch's longest wait for ssh on Vast; the drive's staging margin


def test_the_slack_clears_a_slow_ssh_host():
    """The deadline runs from the create, so the guard minus the runner's FIT_NEED must hold pod-launch's ssh wait plus
    the staging, or a slow-ssh host halts the lane at the fit check (rc 41) before the install (P225's r1 took 443 s)."""
    get = lambda v: int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))  # noqa: E731
    fit_need = (get("INSTALL_TO_GATE_S") + get("NEED_BENCH") + get("RESERVE")) * get("FIT_HEADROOM_PCT") // 100
    guard_s = round(float(re.search(r"\*\*guard ([\d.]+) h\*\*", PREREG).group(1)) * 3600)
    assert (guard_s, fit_need) == (3600, 2625)
    assert guard_s - fit_need == 975 >= SSH_READY_S + STAGING_MARGIN_S
    assert "guard - FIT_NEED = 3600 - 2625 = 975 s" in PREREG


def test_the_drive_never_creates_compute_and_stages_the_premise_in_the_repos_layout():
    assert "p229-5090-[1-3]|p229-5090-r[1-8]|p229-shake-[1-3]) :;;" in DRIVE and "the checkout is dirty" in DRIVE
    assert 'mkdir -p "$STAGE/t/$(dirname "$f")" && cp "$REPO/$f" "$STAGE/t/$f"' in DRIVE
    for f in ("tests/test_int8_head_bw.py", "tests/test_int8_head_rows_interp.py", "tests/conftest.py",
              "bench/i8dense/i8_fakequant.py", "bench/p148/p148_kernels.py"):
        assert f in DRIVE, f
    assert "scp -q -r" in DRIVE
    for word in ("vast.py", "destroy", "/api/", "rent-watch"):
        assert word not in DRIVE.split("set -uo pipefail", 1)[1], word


def _accepted(src, pattern_re, ids):
    """The ids a script's own reading case accepts, run through bash with the pattern as written."""
    pat = re.search(pattern_re, src, re.M).group(1)
    script = f'case "$1" in {pat}) exit 0;; *) exit 9;; esac'
    return {rid for rid in ids if subprocess.run(["bash", "-c", script, "_", rid]).returncode == 0}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_amendment_4_adds_exactly_the_retry_ids_r1_to_r8():
    """Amendment 4: the runner and the drive accept exactly -1..-3 and r1..r8 (labels, not budget); -4 and r9 refuse."""
    probe = [f"p229-5090-{x}" for x in ("0", "1", "2", "3", "4", "10", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
                                       "r8", "r9", "r10", "r01", "R1", "r", "r1x")]
    want = {f"p229-5090-{x}" for x in ("1", "2", "3")} | {f"p229-5090-r{i}" for i in range(1, 9)}
    assert _accepted(RUN, r"^  (p229-5090-1\|[^)]*)\) :;;", probe) == want
    assert _accepted(DRIVE, r'^case "\$E4B_RENT_RUN_ID" in ([^)]*)\) :;;', probe) == want
    assert "**`p229-5090-r1` to `p229-5090-r8`**, for ANY box" in PREREG
    assert "**The ids are labels, not budget.**" in PREREG and "no code parses an id's number" in PREREG
    assert ("| **refused at create** (Amendment 4) | the provider refused the create, so nothing was created ($0) | no |"
            " no; counts toward no cap, uses one id |") in PREREG
    assert "**The stop counts runs whose runner ran.**" in PREREG
    assert "count a refusal at create or a pre-flight teardown: neither is a run for it." in PREREG
    assert "not a strike, under the registered rc 17 rule)" in PREREG


def test_the_drive_follows_pod_launchs_contract():
    """pod-launch runs a reading's drive without E4B_SHA (it proves the checkout HEAD is heads.e4b) and owns the run
    directory's receipt files: the drive derives the launch commit from the clean HEAD, refuses an override, and fetches
    the box tree into $RUN_DIR/p229/ (P226's form)."""
    assert 'if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi' in DRIVE
    assert 'E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)' in DRIVE
    assert "E4B_RENT_DEADLINE_EPOCH E4B_SHA; do" not in DRIVE
    assert 'OUT="$RUN_DIR/p229"' in DRIVE and '"$RUN_DIR/"' not in DRIVE
    assert 'RC=$(cat "$OUT/P229_EXIT_CODE.$NONCE"' in DRIVE


REDUCER_MUTANTS = [
    ("BAR_D = 0.70 ", "BAR_D = 0.80 "),
    ("BAR_K = 1.25 ", "BAR_K = 1.40 "),
    ('out["passes"] = d["ub"] <= BAR_D and d["ub_clock_adj"] <= BAR_D and k["ub"] <= BAR_K and k["ub_clock_adj"] <= BAR_K',
     'out["passes"] = d["median"] <= BAR_D and k["median"] <= BAR_K'),
    ('s["ub_clock_adj"] = s["ub"] * max(1.0, ratio)', 's["ub_clock_adj"] = s["ub"] * min(1.0, ratio)'),
    ("        if not passes[M]:\n            break", "        if not passes[M]:\n            continue"),
    ('if rec.get("rows_record") != ROWS_RECORD:', 'if False:'),
    ('if rec.get("gnf4_commit") != GNF4_COMMIT:', 'if False:'),
    ('if rep.get("d2_identical") is not True:', 'if False:'),
    ('bad = [M for M, c in cells.items() if c["i_rel_max"] > TOL]', 'bad = []'),
    ('if cell.get("engaged") != engaged_want(M):', 'if False:'),
]


@pytest.mark.parametrize("old, new", REDUCER_MUTANTS, ids=[m[0][:28] for m in REDUCER_MUTANTS])
def test_the_self_test_kills_each_reducer_mutant(tmp_path, old, new):
    src = (LANE / "p229_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p229_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    r = subprocess.run([sys.executable, str(tmp_path / "p229_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode != 0 and "FAILED" in r.stdout, (old, r.stdout)


# ---- the real record first: p229-shake-1's bench record (an RTX A4000, explore tier), sha256-pinned
FIXTURE = LANE / "fixtures" / "p229-shake-1" / "bench_p229.json"
FIXTURE_SHA256 = "584b1b61146f5864efc8c5d7dfa1ea72dd11e84c120e0b30e2ec84c297b2659b"


def _real():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, "the pinned shakedown record changed"
    return json.loads(raw)


def test_the_real_shakedown_record_reduces_to_shakedown_with_every_check_passed():
    rec = _real()
    assert rec["shakedown"] is True and rec["gnf4_commit"] == R.GNF4_COMMIT and rec["rows_record"] == R.ROWS_RECORD
    v = R.verdict(rec)
    assert v["verdict"] == "SHAKEDOWN" and v["read"] is False, v
    for M in R.MS:
        assert rec["cells"][str(M)]["engaged"] == R.engaged_want(M)
        assert all(rep["d2_identical"] for rep in rec["cells"][str(M)]["reps"])


DEPARTURES = [
    (lambda r: r["rows_record"]["plan"].update({"8": [128, 128, 4]}), "a plan other than the shipped one at M = 8"),
    (lambda r: r.update(gnf4_commit="a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7"), "another package commit"),
    (lambda r: r["cells"]["4"]["engaged"].update(I={}), "I never routed"),
    (lambda r: r["cells"]["2"]["reps"][1].update(d2_identical=False), "D2 moved a bit"),
    (lambda r: r["cells"]["8"]["reps"][2].update(order=["K", "I", "D2", "D"]), "the order"),
]


@pytest.mark.parametrize("depart, what", DEPARTURES, ids=[d[1] for d in DEPARTURES])
def test_each_departure_from_the_real_record_voids_it(depart, what):
    rec = _real()
    depart(rec)
    v = R.verdict(rec)
    assert v["verdict"] == "VOID", (what, v.get("why"))


def test_a_mutant_without_the_route_record_check_accepts_the_real_records_departure(tmp_path):
    """The rule that the bar reads the plan the shipped route runs: without it, a record off that plan passes."""
    src = (LANE / "p229_reduce.py").read_text(encoding="utf-8")
    old = 'if rec.get("rows_record") != ROWS_RECORD:'
    assert src.count(old) == 1
    (tmp_path / "p229_reduce.py").write_bytes(src.replace(old, "if False:").encode("utf-8"))
    mut = _load("p229_reduce_no_route_check", tmp_path / "p229_reduce.py")
    rec = _real()
    DEPARTURES[0][0](rec)
    assert mut.verdict(rec)["verdict"] == "SHAKEDOWN"
    assert R.verdict(rec)["verdict"] == "VOID"
