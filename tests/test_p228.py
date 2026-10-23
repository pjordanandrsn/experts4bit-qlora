# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P228, K1's selection and microbench (bench/p228/PREREG-p228.md). The staged pins; the registered constants as the
PREREG states them; the runner's order, premise and guard fit; the drive's refusals; the runner's READ line on a mixed
record; the self-tests and the reducer mutants they must kill."""
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
LANE = REPO / "bench" / "p228"
SOURCES = {"p228_bench.py": LANE, "p228_reduce.py": LANE, "p228_run.sh": LANE,
           "test_dense_lt_pin_gpu.py": REPO / "tests", "test_dense_lt_pin.py": REPO / "tests"}
RUN = (LANE / "p228_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p228_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p228.md").read_text(encoding="utf-8")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


R = _load("p228_reduce_under_test", LANE / "p228_reduce.py")


def test_every_staged_file_is_pinned_and_matches():
    pins = dict(reversed(line.split()) for line in (LANE / "staged-p228.sha256").read_text(encoding="ascii").splitlines())
    assert set(pins) == set(SOURCES)
    for name, where in SOURCES.items():
        assert hashlib.sha256((where / name).read_bytes()).hexdigest() == pins[name], name


def test_the_self_tests_pass():
    R.self_test()
    r = subprocess.run([sys.executable, str(LANE / "p228_bench.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def test_the_registered_constants_and_the_prereg_agree():
    assert (R.BAR, R.TOL, R.TIE, R.RULED_MS, R.MS, R.SELF_BAND, R.MIN_POWER_W) == \
        (0.85, 1.5e-2, 1.02, (2, 4), (2, 4, 8), (0.99, 1.01), 575)
    assert R.SITES == {"o_proj": {"N": 2048, "K": 4096, "copies": 48}, "router": {"N": 128, "K": 2048, "copies": 384}}
    assert R.ARMS == ["D", "D2", "P", "T1"] and R.ITERS == 10 and R.READ_STEPS == tuple(range(60))
    assert (R.SELECT_STEPS, R.SELECT_ITERS) == (8, 5) and R.REDUCTION_OK == (0, 2, 4)
    for s in ("**0.85**", "**within 1.5e-2**", "**1.02x**", "| `o_proj` | (2048, 4096) | 48 |", "| router gate | (128, 2048) | 384 |",
              "exactly **37 passed**", "under 575 W", "seed 2280 + M", "seed 22800 + M", "**$1.45**"):
        assert s in PREREG, s


def test_the_runner_installs_e4b_alone_checks_k1_and_runs_the_premise_on_the_card():
    assert "grouped-nf4-gemm.git" not in RUN and "GNF4_SHA" not in RUN
    assert 'assert dl.LT_PINS == {}, "the commit ships no pins: this lane selects them"' in RUN
    assert "assert re_.router_linear is dl.router_linear" in RUN
    assert "test_dense_lt_pin_gpu.py test_dense_lt_pin.py -q -rs" in RUN and "grep -qE '^37 passed" in RUN
    order = ["sha256sum -c staged-p228.sha256", "PL=$(nvidia-smi --query-gpu=power.limit", '[ "$LEFT" -ge "$FIT_NEED" ]',
             "pipx logs/pip_e4b.log", "TRIPWIRE FAIL", "grep -qE '^37 passed", "p228_bench.py --out", "p228_reduce.py --bench"]
    at = [RUN.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert "MIN_POWER_W=575 " in RUN and "finish 17; }" in RUN
    assert "p228-5090-1|p228-5090-2|p228-5090-3|" in RUN and "p228-shake-1|p228-shake-2|p228-shake-3) SHAKE=1;;" in RUN
    assert "--shakedown --reps 4 --copies-o 8 --copies-r 32" in RUN


def test_the_guard_fit():
    get = lambda v: int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))  # noqa: E731
    install, need, reserve = get("INSTALL_TO_GATE_S"), get("NEED_BENCH"), get("RESERVE")
    assert (install, need, reserve) == (600, 1200, 600)
    fit = install + need + reserve
    assert fit == 2400 and int(fit * 1.25) == 3000 and (300 + fit) * 1.25 == 3375 <= 3600
    for s in ("`NEED_BENCH` = **1200 s**", "`FIT` = 600 + 1200 + 600 = 2400 s", "2400 x 1.25 = 3000 s",
              "(boot 300 s + 2400 s) x 1.25 = 3375 s <= 3600 s (1.0 h)", "up to 64 candidates"):
        assert s in PREREG, s
    assert abs(1.0 * 1.05 + 10 * 0.04 - 1.45) < 1e-9


def test_the_drive_never_creates_compute_and_refuses_what_is_not_registered():
    assert "p228-5090-[1-3]|p228-5090-r[1-8]|p228-shake-[1-3]) :;;" in DRIVE and "the checkout is dirty" in DRIVE
    assert "tests/test_dense_lt_pin_gpu.py tests/test_dense_lt_pin.py; do" in DRIVE
    for word in ("vast.py", "destroy", "/api/", "rent-watch"):
        assert word not in DRIVE.split("set -uo pipefail", 1)[1], word


SSH_READY_S, STAGING_MARGIN_S = 420, 120   # pod-launch's longest wait for ssh on Vast; the drive's staging margin


def test_the_slack_clears_a_slow_ssh_host():
    """The deadline runs from the create, so the guard minus the runner's FIT_NEED must hold pod-launch's ssh wait plus
    the staging, or a slow-ssh host halts the lane at the fit check (rc 41) before the install (P225's r1 took 443 s)."""
    get = lambda v: int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))  # noqa: E731
    fit_need = (get("INSTALL_TO_GATE_S") + get("NEED_BENCH") + get("RESERVE")) * get("FIT_HEADROOM_PCT") // 100
    guard_s = round(float(re.search(r"\*\*guard ([\d.]+) h\*\*", PREREG).group(1)) * 3600)
    assert (guard_s, fit_need) == (3600, 3000)
    assert guard_s - fit_need == 600 >= SSH_READY_S + STAGING_MARGIN_S
    assert "guard - FIT_NEED = 3600 - 3000 = 600 s" in PREREG


def _accepted(src, pattern_re, ids):
    """The ids a script's own reading case accepts, run through bash with the pattern as written."""
    pat = re.search(pattern_re, src, re.M).group(1)
    script = f'case "$1" in {pat}) exit 0;; *) exit 9;; esac'
    return {rid for rid in ids if subprocess.run(["bash", "-c", script, "_", rid]).returncode == 0}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_amendment_5_adds_exactly_the_retry_ids_r1_to_r8():
    """Amendment 5: the runner and the drive accept exactly -1..-3 and r1..r8 (labels, not budget); -4 and r9 refuse."""
    probe = [f"p228-5090-{x}" for x in ("0", "1", "2", "3", "4", "10", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
                                       "r8", "r9", "r10", "r01", "R1", "r", "r1x")]
    want = {f"p228-5090-{x}" for x in ("1", "2", "3")} | {f"p228-5090-r{i}" for i in range(1, 9)}
    assert _accepted(RUN, r"^  (p228-5090-1\|[^)]*)\) :;;", probe) == want
    assert _accepted(DRIVE, r'^case "\$E4B_RENT_RUN_ID" in ([^)]*)\) :;;', probe) == want
    assert "**`p228-5090-r1` to `p228-5090-r8`**, for ANY box" in PREREG
    assert "**The ids are labels, not budget.**" in PREREG and "no code parses an id's number" in PREREG
    assert ("| **refused at create** (Amendment 5) | the provider refused the create, so nothing was created ($0) | no |"
            " no; counts toward no cap, uses one id |") in PREREG


#: Amendment 5's host and fit refusals before the install; a box that passes them and begins the install STARTS
PRE_INSTALL_REFUSALS = {10, 13, 15, 16, 17, 18, 28, 41}


def _pre_install_refusals():
    """Every `finish N` the runner can reach before its first install, minus the kit's own refusals (9 a staging
    tripwire, 78 an unregistered id) and the signal trap (130)."""
    head = RUN.split("\npipx logs/", 1)[0]
    return {int(n) for n in re.findall(r"finish (\d+)", head)} - {9, 78, 130}


def _counts(box):
    """Amendment 5's rule, as the hand table applies it: (started, toward the stop)."""
    started = box.get("runner_rc") is not None and box.get("began_install", False) \
        and box["runner_rc"] not in PRE_INSTALL_REFUSALS
    return started, started and not box.get("site_read", False)


def test_amendment_5_one_start_definition_for_the_cap_and_the_stop():
    """Ruled pre-data (P227's and P229's final form): a box STARTS when its runner passes the pre-install refusals and
    begins the install; only started boxes count toward the cap (3) and the stop; the four non-start cases count toward
    nothing; a mid-install fault starts and counts; the rc 41 pair rule is unchanged."""
    assert _pre_install_refusals() == PRE_INSTALL_REFUSALS            # the runner refuses all of them before any install
    flat = " ".join(PREREG.split("## Amendment 5", 1)[1].split())
    for case in ("**A box STARTS when its runner passes the refusals before the install (rc 17, rc 41, and the host"
                 " refusals rc 10, 13, 15, 16, 18 and 28) and begins the install**",
                 "**Only started boxes count toward the cap and toward any stop; a box that did not start (a refusal at"
                 " create, a pre-flight teardown, rc 17, a single rc 41) counts toward none of them; the registered rc 41"
                 " pair rule is unchanged.**",
                 "**At most three boxes start.**", "3 x $1.45",
                 "This PREREG has one counting rule for boxes, the stop (\"Two consecutive runs with no site read stop"
                 " the lane\", Budget)",
                 "An rc 17 box never counts toward the stop, whatever label the launcher's receipt carries.",
                 "A single rc 41 is not a strike; two consecutive rc 41 count as one strike under the cross-lane rule"
                 " (Amendment 3), unchanged.",
                 "A refusal at create and a pre-flight teardown count toward nothing.",
                 "Decided pre-data: no P228 box has run."):
        assert case in flat, case
    assert "Two consecutive runs with no site read stop the lane." in PREREG.split("## Amendment 1", 1)[0]
    not_started = [{"runner_rc": None},                                  # a refusal at create: nothing created
                   {"runner_rc": None, "created": True},                 # a pre-flight teardown: the runner never ran
                   {"runner_rc": 17}, {"runner_rc": 41}]                 # the box's own refusals before the install
    for box in not_started:
        assert _counts(box) == (False, False), box
    assert _counts({"runner_rc": 9, "began_install": True}) == (True, True)     # a mid-install fault (PIP FAIL, rc 9)
    assert _counts({"runner_rc": 27, "began_install": True}) == (True, True)    # no site read: started, counts
    assert _counts({"runner_rc": 0, "began_install": True, "site_read": True}) == (True, False)
    assert "PIP FAIL (e4b)\"; finish 9; }" in RUN.split("\npipx logs/", 1)[1][:400]
    a3 = " ".join(PREREG.split("## Amendment 3", 1)[1].split("## Amendment 4", 1)[0].split())
    assert "two consecutive headroom refusals count as one strike" in a3
    assert "# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike" \
        in RUN


def _rc17_stop(boxes):
    """Amendment 6's stop, as the owner applies it: True once three boxes whose runner ran end in rc 17 in a row. A
    refusal at create or a pre-flight teardown (runner_rc None) neither adds to the count nor resets it; any other
    runner outcome resets it."""
    run = 0
    for rc in boxes:
        if rc is None:
            continue
        run = run + 1 if rc == 17 else 0
        if run >= 3:
            return True
    return False


def test_amendment_6_admits_by_the_latest_power_record_and_the_box_reads_the_floor():
    """Amendment 6 admits a machine whose latest registry record is >= 575 W at any age, or that has no record, so the
    box's own read is the floor for those hosts. The runner prints the read the registry parses, and refuses below 575 W
    (rc 17), both before its first install. The floor and the caps are unchanged, and three consecutive rc 17 stop the
    lane."""
    head = RUN.split("\npipx logs/", 1)[0]
    assert "MIN_POWER_W=575" in RUN
    printed = head.index('echo "POWER_LIMIT_W $PL (floor $MIN_POWER_W)"')       # the registry's source for the cap
    refused = head.index("finish 17; }", printed)                               # rc 17 after the print, before install
    assert printed < refused
    assert 17 in _pre_install_refusals()
    assert 'say "REFUSED: power.limit unreadable"' in head and "finish 15; }" in head      # an unreadable cap is rc 15
    flat = " ".join(PREREG.split("## Amendment 6 ", 1)[1].split())
    for case in ("is never pinned, whatever the record's age",
                 "A machine whose latest record is 575 W or more, at any age, is admitted, and so is a machine with no"
                 " record.",
                 "**The stop for them (new): three consecutive rc 17 boxes stop the lane.**",
                 "A refusal at create or a pre-flight teardown neither adds to the count nor resets it; any other runner"
                 " outcome resets it.",
                 "**Unchanged:** the 575 W floor, the measurement, the rows, the rule",
                 "at most $1.05/h, guard 1.0 h, $1.45 worst for a started box, and at most three started boxes",
                 "No staged pin moves and no runner byte changes.",
                 "each template sets it to `\"eligible\"` (pod-launch's default), keeps the rule `\"latest\"`, and sets no"
                 " `max_age_days`",
                 "**Who counts:** the lane's owner, from each box's receipt (the runner's exit code).",
                 "So at most **$0.56** a box, and three in a row cost at most **$1.68** (3 x $0.56) before the stop."):
        assert case in flat, case
    assert _rc17_stop([17, 17, 17])
    assert _rc17_stop([17, None, 17, None, 17])                  # refusals at create neither add nor reset
    assert not _rc17_stop([17, 17, 0, 17, 17])                   # a started box resets the count
    assert not _rc17_stop([17, 17, 41, 17])                      # so does any other runner outcome
    assert not _rc17_stop([None, None, None])                    # refusals at create never stop the lane on their own
    assert _counts({"runner_rc": 17}) == (False, False)          # an rc 17 box still counts toward no cap and no stop


def test_the_drive_follows_pod_launchs_contract():
    """pod-launch runs a reading's drive without E4B_SHA (it proves the checkout HEAD is heads.e4b) and owns the run
    directory's receipt files: the drive derives the launch commit from the clean HEAD, refuses an override, and fetches
    the box tree into $RUN_DIR/p228/ (P226's form)."""
    assert 'if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi' in DRIVE
    assert 'E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)' in DRIVE
    assert "E4B_RENT_DEADLINE_EPOCH E4B_SHA; do" not in DRIVE
    assert 'OUT="$RUN_DIR/p228"' in DRIVE and '"$RUN_DIR/"' not in DRIVE
    assert 'RC=$(cat "$OUT/P228_EXIT_CODE.$NONCE"' in DRIVE


def test_a_record_with_no_site_read_is_not_a_reading(tmp_path):
    code = re.search(r'^READ=\$\(python -c "(.*)"\)$', RUN, re.M).group(1)

    def read_count(v):
        p = tmp_path / "verdict_p228.json"
        p.write_text(json.dumps(v), encoding="utf-8")
        r = subprocess.run([sys.executable, "-c", code.replace("$W/verdict_p228.json", p.as_posix())],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return int(r.stdout.strip())
    q = R._fake()
    q["sites"]["o_proj"]["cells"]["2"]["reps"][4]["p_rel"] = 0.05                    # o_proj: TOLERANCE_FAIL (read)
    q["sites"]["router"]["cells"]["8"]["reps"][1]["d2_identical"] = False             # router: VOID
    v = R.verdict(q)
    assert v["verdict"] == "o_proj=TOLERANCE_FAIL router=VOID" and read_count(v) == 1
    q["sites"]["o_proj"]["cells"]["2"]["reps"][4]["d2_identical"] = False
    v = R.verdict(q)
    assert v["verdict"] == "o_proj=VOID router=VOID" and read_count(v) == 0
    assert '"1/o_proj=SHAKEDOWN router=SHAKEDOWN") finish 0;;' in RUN


REDUCER_MUTANTS = [
    ("BAR = 0.85 ", "BAR = 0.95 "),
    ("RULED_MS = (2, 4) ", "RULED_MS = (2,) "),
    ('ok = all(cells[M]["p_over_d"]["ub"] <= BAR and cells[M]["p_over_d"]["ub_clock_adj"] <= BAR for M in RULED_MS)',
     'ok = all(cells[M]["p_over_d"]["median"] <= BAR for M in RULED_MS)'),
    ('p["ub_clock_adj"] = p["ub"] * max(1.0, ratio)', 'p["ub_clock_adj"] = p["ub"] * min(1.0, ratio)'),
    ('two = [c for c in ok if c["config"]["reduction_scheme"] == 2 and select_ms(c["step_ms"]) <= TIE * best]', 'two = []'),
    ('return (c.get("check_ok") is True and c.get("bitwise") is True', 'return (c.get("check_ok") is True'),
    ('if rep.get("d2_identical") is not True:', 'if False:'),
    ('bad = [M for M, c in cells.items() if c["p_rel_max"] > TOL]', 'bad = []'),
    ('if sel.get("chosen") != cfg:', 'if False:'),
]


@pytest.mark.parametrize("old, new", REDUCER_MUTANTS, ids=[m[0][:28] for m in REDUCER_MUTANTS])
def test_the_self_test_kills_each_reducer_mutant(tmp_path, old, new):
    src = (LANE / "p228_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p228_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    r = subprocess.run([sys.executable, str(tmp_path / "p228_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode != 0 and "FAILED" in r.stdout, (old, r.stdout)


# ---- the real record first: p228-shake-1's bench record (an RTX A4000, explore tier), sha256-pinned
FIXTURE = LANE / "fixtures" / "p228-shake-1" / "bench_p228.json"
FIXTURE_SHA256 = "06adab439ed025df7edbe06b3bba85cd135409ece94874076cc7981b621d25dc"


def _real():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, "the pinned shakedown record changed"
    return json.loads(raw)


def test_the_real_shakedown_record_reduces_to_shakedown_with_every_check_passed():
    rec = _real()
    assert rec["shakedown"] is True and isinstance(rec["key"]["cublaslt"], int)
    v = R.verdict(rec)
    assert v["verdict"] == "o_proj=SHAKEDOWN router=SHAKEDOWN", v
    for site in R.SITE_ORDER:
        L = rec["sites"][site]["copies"]
        for M in R.MS:
            sel = rec["sites"][site]["select"][str(M)]
            assert sel["chosen"] == R.select(sel["candidates"])[0] is not None
            assert rec["sites"][site]["cells"][str(M)]["engaged"]["P"] == {f"{site}|M{M}|eager|pinned": L}


DEPARTURES = [
    (lambda r: r["sites"]["o_proj"]["select"]["2"].update(chosen=dict(r["sites"]["o_proj"]["select"]["2"]["chosen"],
                                                                      split_k=99)), "a pin the rule did not name"),
    (lambda r: r["sites"]["o_proj"]["cells"]["4"]["engaged"].update(D={"o_proj|M4|eager|pinned": 1}), "D pinned"),
    (lambda r: r["sites"]["o_proj"]["cells"]["8"]["engaged"].update(P={}), "P never pinned"),
    (lambda r: r["sites"]["o_proj"]["cells"]["2"]["reps"][1].update(d2_identical=False), "D2 moved a bit"),
    (lambda r: r["sites"]["o_proj"]["cells"]["4"]["reps"][2].update(order=["T1", "P", "D2", "D"]), "the order"),
]


@pytest.mark.parametrize("depart, what", DEPARTURES, ids=[d[1] for d in DEPARTURES])
def test_each_departure_from_the_real_record_voids_that_site_only(depart, what):
    rec = _real()
    depart(rec)
    sv = R.verdict(rec)["sites"]
    assert sv["o_proj"]["verdict"] == "VOID" and sv["router"]["verdict"] == "SHAKEDOWN", (what, sv["o_proj"]["why"])


def test_a_mutant_without_the_chosen_pin_check_accepts_the_real_records_departure(tmp_path):
    """The rule that block B timed the pin block A's rule names: without it, a hand-edited pin passes."""
    src = (LANE / "p228_reduce.py").read_text(encoding="utf-8")
    old = 'if sel.get("chosen") != cfg:'
    assert src.count(old) == 1
    (tmp_path / "p228_reduce.py").write_bytes(src.replace(old, "if False:").encode("utf-8"))
    mut = _load("p228_reduce_no_chosen_check", tmp_path / "p228_reduce.py")
    rec = _real()
    DEPARTURES[0][0](rec)
    assert mut.verdict(rec)["sites"]["o_proj"]["verdict"] == "SHAKEDOWN"
    assert R.verdict(rec)["sites"]["o_proj"]["verdict"] == "VOID"


def _fetches_without_no_links(drive):
    """(rsync lines, the ones without --no-links): the drive's live rsync invocations, comments skipped."""
    lines = [ln for ln in drive.splitlines() if re.search(r"\brsync\b", ln) and not ln.lstrip().startswith("#")]
    return lines, [ln for ln in lines if not re.search(r"\brsync\s(?:[^|;&]*\s)?--no-links\b", ln)]


def test_the_fetch_never_imports_a_link(tmp_path):
    """Amendment 7 (bench/common rule 8): every rsync in the drive carries --no-links, so no link on a rented box reaches
    the controller. A drive with no rsync fails (the check needs its input), and the old fetch line fails."""
    lines, bad = _fetches_without_no_links(DRIVE)
    assert len(lines) == 1 and not bad, (lines, bad)
    assert _fetches_without_no_links(DRIVE.replace(" --no-links", ""))[1]          # the old line is refused
    assert not _fetches_without_no_links("echo fetch\n")[0]                          # no rsync -> the count above fails
    if shutil.which("rsync") is None:
        return
    src, dst = tmp_path / "box", tmp_path / "out"
    src.mkdir()
    (src / "f.json").write_text("{}", encoding="utf-8")
    try:
        (src / "secret").symlink_to("../../../../etc/hosts")
        (src / "etc").symlink_to("/etc", target_is_directory=True)
    except OSError:
        pytest.skip("this host cannot make symlinks")
    flags = re.search(r"rsync (-az --no-links) -e", lines[0]).group(1).split()
    subprocess.run(["rsync", *flags, f"{src}/", f"{dst}/"], check=True, capture_output=True)
    assert (dst / "f.json").is_file()
    assert not [p for p in dst.rglob("*") if p.is_symlink()] and not (dst / "secret").exists() and not (dst / "etc").exists()
