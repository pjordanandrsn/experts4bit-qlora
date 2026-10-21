# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P223 (bench/p223/PREREG-p223.md): the staged pins, the arms as the box, the reducer and the runner each state
them, the runner's order (the host floors and the guard fit before any install, the compiled premises before the fetch,
the eight processes in the registered order with each arm's first read before the next, the reduce), the fit table
against the registration's and its 3.5 h guard, the self-tests run as the box runs them (one directory), the mutants the
self-tests must kill, and the driver's run ids."""
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
LANE = B / "p223"
SOURCES = {"p223_run.sh": LANE, "p223_box.py": LANE, "p223_reduce.py": LANE, "test_dense_i8.py": B / "p149",
           "test_int8_attn_interp.py": REPO / "tests", "p109_box.py": B / "p109", "k8_bake.py": B / "p39",
           "calib.json": B / "p39", "phases.sh": B / "common"}
RUN = (LANE / "p223_run.sh").read_text(encoding="utf-8")
PRE = (LANE / "PREREG-p223.md").read_text(encoding="utf-8")


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, cwd=None, env=None):
    """A staged script in its own directory, as the box runs it; the package resolves from this tree when it is not
    installed (the box installs it)."""
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300, cwd=cwd,
                          env=env)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p223.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_the_helpers_are_p149s_registered_bytes():
    """The kit is P149's box Q+S kit by substitution: every helper it shares is staged at P149's pinned bytes."""
    p149 = _pins(B / "p149" / "staged-p149.sha256")
    pins = _pins(LANE / "staged-p223.sha256")
    for name in ("test_dense_i8.py", "test_int8_attn_interp.py", "p109_box.py", "k8_bake.py", "calib.json"):
        assert pins[name] == p149[name], name


def test_the_runner_the_reducer_and_ci_pin_the_same_grouped_nf4_gemm():
    red = (LANE / "p223_reduce.py").read_text(encoding="utf-8")
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    sha = re.search(r"^GNF4_SHA=([0-9a-f]{40})", RUN, re.M).group(1)
    assert f'GNF4_SHA = "{sha}"' in red and f"grouped-nf4-gemm.git@{sha}" in ci


def test_the_arms_are_the_registrations_in_box_reducer_and_runner():
    """PREREG "The arms": D0 none; I0 the split cap at 1 and the buckets 2,4,8,16; D1 the int8 lever; I1 all three. The
    D arms run the auto buckets, so the engine environment never sets E4B_PAGED_BUCKETS."""
    sys.path.insert(0, str(LANE))
    try:
        import p223_box as box
        import p223_reduce as red
    finally:
        sys.path.pop(0)
    i = {"E4B_ATTN_SPLIT_CAP": "1", "E4B_PAGED_BUCKETS": "2,4,8,16"}
    assert box.ARM_KNOBS == {"D0": {}, "I0": i, "D1": {"E4B_SERVE_ATTN_INT8": "1"}, "I1": {"E4B_SERVE_ATTN_INT8": "1", **i}}
    assert box.TAGS == red.TAGS == ("D0a", "I0a", "I1a", "D1a", "D1b", "I1b", "I0b", "D0b")
    assert box.ARM_BUCKETS == red.ARM_BUCKETS == {"D": (1, 2, 4, 8, 16), "I": (2, 4, 8, 16)}
    assert "| **D0** | the default (bf16 attention projections) | none |" in PRE
    assert "| **I1** | `E4B_SERVE_ATTN_INT8=1` | `E4B_ATTN_SPLIT_CAP=1`, `E4B_PAGED_BUCKETS=2,4,8,16` |" in PRE
    assert "D0a, I0a, I1a, D1a, D1b, I1b, I0b, D0b" in PRE
    knobs = RUN[RUN.index("arm_knobs(){"):RUN.index("run_arm(){")]
    assert 'D0) echo "";;' in knobs and 'I0) echo "E4B_ATTN_SPLIT_CAP=1 E4B_PAGED_BUCKETS=2,4,8,16";;' in knobs
    assert 'D1) echo "E4B_SERVE_ATTN_INT8=1";;' in knobs
    assert 'I1) echo "E4B_SERVE_ATTN_INT8=1 E4B_ATTN_SPLIT_CAP=1 E4B_PAGED_BUCKETS=2,4,8,16";;' in knobs
    engine = re.search(r'^ENGINE_ENV="([^"]*)"', RUN, re.M).group(1)
    assert "E4B_PAGED_BUCKETS" not in engine and "E4B_PAGED_MAX_SEQS=16 E4B_PAGED_MAX_TOKENS_PER_SEQ=768" in engine
    assert red.NOISY_BAND == (0.985, 1.015) and red.MIN_POWER_W == 575 and red.CC == "12.0"
    assert (red.P16_BAND, red.P1_BAND) == ((0.97, 1.00), (0.60, 0.95))
    assert "p16 in 0.97-1.00" in PRE and "p1 in 0.60-0.95" in PRE and "outside [0.985, 1.015]" in PRE


def test_the_runner_refuses_before_any_install_and_runs_the_registered_order():
    assert "p223-5090-1|p223-5090-2|p223-5090-3) ;;" in RUN and "`p223-5090-1` is the reading" in PRE
    assert "MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1; REV=eba92302a2861cdc0098cc54bc9f17cb2c47eb61" in RUN
    assert "MIN_POWER_W=575" in RUN and 'assert triton.__version__ == "3.4.0"' in RUN and "TORCH_FULL\" = 2.8.0+cu128" in RUN
    assert 'transformers.__version__ == "5.17.0"' in RUN and "Mixtral at `eba92302`" in PRE
    order = ['[ "$CC" = 12.0 ]', '[ "$PL" -ge "$MIN_POWER_W" ] && [ "$EPL" -ge "$MIN_POWER_W" ]', "finish 17; }",
             '[ "$LEFT" -ge "$FIT_NEED" ]', "finish 19; }", ". ./phases.sh", "pipx(){", "phase_start install setup",
             "pipx logs/pip_e4b.log",
             "p223_reduce.py --self-test", "p223_box.py --self-test",
             "python -m pytest test_dense_i8.py", "python -m pytest test_int8_attn_interp.py", "grep -qE '^28 passed",
             "grep -qE '^4 passed", "can_run $NEED_FETCH fetch", "can_run $NEED_BAKE bake", "--prompts-only",
             "for TAG in D0a I0a I1a D1a D1b I1b I0b D0b; do", "case \"$TAG\" in D0a|I0a|I1a|D1a) gate_check $TAG;; esac",
             'reduce_all || { say "REDUCER FAILED"; finish 22; }', 'case "$V" in READ) ;;']
    at = []
    for x in order:
        assert x in RUN, x
        at.append(RUN.index(x, at[-1] if at else 0))
    assert at == sorted(at), dict(zip(order, at))
    assert "finish 27;; esac" in RUN, "NOISY and VOID are not readings"
    for gone in ("--mode gate", "--mode tf", "--mode sane", "--windows", "--q ", "--q-only", "P223_DRY", "slice"):
        assert gone not in RUN, gone
    gate = RUN[RUN.index("gate_check(){"):RUN.index("for TAG in")]
    assert "phase_start reduce reduce; reduce_all; phase_end reduce; finish 45" in gate, "an early VOID still reduces"


def test_the_phase_stamps_are_the_common_writers_with_a_kind_each():
    """bench/common/README.md "Phase stamps": the runner sources the staged phases.sh (after the fit check, as another
    file's script must come) and keeps no local writer; every phase opens with a registered kind and closes."""
    assert "stamp(){" not in RUN and re.search(r"^\s*stamp ", RUN, re.M) is None
    starts = re.findall(r"phase_start ([^\s;]+) ([a-z]+)", RUN)
    ends = set(re.findall(r"phase_end ([^\s;]+)", RUN))
    assert {n for n, _k in starts} == ends == {"install", "fetch", "bake", '"$TAG"', "reduce"}, (starts, ends)
    assert {(n, k) for n, k in starts} == {("install", "setup"), ("fetch", "setup"), ("bake", "setup"),
                                           ('"$TAG"', "measure"), ("reduce", "reduce")}
    assert RUN.index(". ./phases.sh") > RUN.index('[ "$LEFT" -ge "$FIT_NEED" ]')


TERMS = r"\b(INSTALL|NEED_FETCH|NEED_BAKE|NEED_ARM|NEED_REDUCE|RESERVE)=(\d+)"


def test_the_fit_table_is_the_registrations_and_fits_three_and_a_half_hours():
    """PREREG "The fit": boot 300 + install 450 + fetch 3,000 + bake 1,500 + 8 x 480 + reduce 120 + reserve 600 = 9,810 s;
    x 1.25 = 12,263 s inside the 3.5 h guard; the runner's own check before the install needs (9,810 - 300) x 1.25 =
    11,888 s left."""
    v = {k: int(x) for k, x in re.findall(TERMS, RUN)}
    assert v == {"INSTALL": 450, "NEED_FETCH": 3000, "NEED_BAKE": 1500, "NEED_ARM": 480, "NEED_REDUCE": 120,
                 "RESERVE": 600}, v
    for term, s in (("boot", "300"), ("install", "450"), ("fetch", "3,000"), ("bake", "1,500"),
                    ("8 arm processes × 480", "3,840"), ("reduce", "120"), ("reserve", "600")):
        assert f"| {term} | {s} |" in PRE, term
    assert "| **total** | **9,810** |" in PRE and "inside the **3.5 h guard** (12,600 s)" in PRE
    assert "(9,810 − 300) × 1.25 = 11,888 s" in PRE
    assert "FIT=$((INSTALL + NEED_FETCH + NEED_BAKE + 8 * NEED_ARM + NEED_REDUCE + RESERVE))" in RUN
    assert "FIT_HEADROOM_PCT=125" in RUN and "FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))" in RUN     # the canonical form
    boot, guard = 300, int(3.5 * 3600)
    fit = v["INSTALL"] + v["NEED_FETCH"] + v["NEED_BAKE"] + 8 * v["NEED_ARM"] + v["NEED_REDUCE"] + v["RESERVE"]
    assert boot + fit == 9810 and (boot + fit) * 1.25 <= guard
    need = fit * 125 // 100                                  # the runner's check in integer arithmetic
    assert need == 11887 and 11888 - need == 1               # 1 s under the registration's 11,888 (the half second rounded up)
    assert guard - need >= boot                              # the guard leaves boot its allowance after that check


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p223_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


def _engine():
    try:
        import experts4bit_qlora.engines.paged_runner  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("script", ["p223_box.py", "p223_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and re.search(r"self-test: (\d+)/\1 OK", out.stdout), out.stdout + out.stderr
    if script == "p223_box.py" and _engine():                # the DecodeCensus cases need the package: none skipped
        assert "self-test: 19/19 OK" in out.stdout, out.stdout


BOX_MUTANTS = [
    ('ARM_KNOBS = {"D0": {}, "I0": {SPLIT: "1", BUCKETS: "2,4,8,16"},', 'ARM_KNOBS = {"D0": {}, "I0": {SPLIT: "1"},'),
    ('ARM_BUCKETS = {"D": (1, 2, 4, 8, 16), "I": (2, 4, 8, 16)}', 'ARM_BUCKETS = {"D": (1, 2, 4, 8, 16), "I": (1, 2, 4, 8, 16)}'),
    ('    if (arm[0] == "I") != (split == "1"):', "    if False:"),
    ('                 "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING")', '                 "E4B_NF4_T1_DEVICE_GROUPING",)'),
    ('        b = self.bucket_for(n, self.r._buckets) if n <= self.r._buckets[-1] else "chunked"', "        b = n"),
    ("        return self.orig(rids)", "        return {}"),
]


@pytest.mark.parametrize("old,new", BOX_MUTANTS)
def test_each_box_mutant_fails_the_self_test(tmp_path, old, new):
    if not _engine():
        pytest.skip("the DecodeCensus cases need torch and the package")
    for name, d in SOURCES.items():
        shutil.copyfile(d / name, tmp_path / name)
    src = (LANE / "p223_box.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p223_box.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p223_box.py", "--self-test", cwd=tmp_path)
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


MUTANTS = [
    ("NOISY_BAND = (0.985, 1.015)", "NOISY_BAND = (0.98, 1.02)"),
    ("P16_BAND, P1_BAND = (0.97, 1.00), (0.60, 0.95)", "P16_BAND, P1_BAND = (0.90, 1.00), (0.60, 0.95)"),
    ("P16_BAND, P1_BAND = (0.97, 1.00), (0.60, 0.95)", "P16_BAND, P1_BAND = (0.97, 1.00), (0.60, 0.98)"),
    # VOID: the card
    ("        if not isinstance(epl, (int, float)) or epl < MIN_POWER_W:", "        if False:"),
    ('    for key in ("card", "card_after"):', '    for key in ("card",):'),
    ('        if GPU_CLASS not in str(c.get("name") or ""):', "        if False:"),
    ('    if str(r.get("cc")) != CC:', "    if False:"),
    # VOID: the engagement
    ('    if r.get("runner_buckets") != want_b or r.get("buckets") != want_b:', "    if False:"),
    ('    if [g.get(str(b)) for b in want_b] != ["graph"] * len(want_b) or set(g) - {str(b) for b in want_b}:', "    if False:"),
    ('        if sc.get("cap") != 1 or sc.get("source") != "explicit":', "        if False:"),
    ('        if any(str(b) not in plans for b in want_b) or any((p or {}).get("n_split") != 1 for p in plans.values()):',
     "        if False:"),
    ('    elif sc.get("cap") is not None or sc.get("source") != "default-off" or plans:', "    elif False:"),
    ('    if any((s or {}).get("eager_steps") for s in gs.values()):', "    if False:"),
    ("        if rep1:\n", "        if False:\n"),
    ('        if not lone or set(lone) != {"2"}:', "        if not lone:"),
    ('        elif lone["2"] != window_pad:', "        elif False:"),
    ("    elif not rep1:", "    elif False:"),
    # VOID: the int8 counters
    ('        if r.get("int8_modules") or a8.get("projections"):', "        if False:"),
    ('    if not r.get("int8_modules") or not a8.get("projections"):', "    if False:"),
    ("        if not sum(routed.values()):", "        if False:"),
    # VOID: P149's kit checks
    ('    if r.get("reps") != REPS or (r.get("short"), r.get("long")) != (SHORT, LONG):', "    if False:"),
    ('    if cen != CENSUS:', "    if False:"),
    ('    if len({arms[t].get("prompts_sha256") for t in TAGS}) != 1:', "    if False:"),
    # the reads
    ("            at = {i: first_diff(short[i][k], long_[i][k][:SHORT]) for i in range(len(short))}",
     "            at = {i: first_diff(short[i][k], long_[i][k][:SHORT]) for i in range(1)}"),
    ('        elif lone["2"] != window_pad:', '        elif lone["2"] != (gs.get("2") or {}).get("pad_rows"):'),
    ("                at = {i: first_diff(reps[i][k], reps[0][k]) for i in range(1, len(reps))}",
     "                at = {i: first_diff(reps[i][k], reps[0][k]) for i in range(1, 2)}"),
    ('        labels[s] = "NOT_SCHEDULE_INVARIANT" if bad else "SCHEDULE_INVARIANT"',
     '        labels[s] = "SCHEDULE_INVARIANT"'),
    ('        p1, p16 = min(pairs["W1"].values()), min(pairs["W16"].values())',
     '        p1, p16 = max(pairs["W1"].values()), min(pairs["W16"].values())'),
    ("    steps = [first_diff(a[0], b[0]) for a, b in zip(w1, w16)]", "    steps = [first_diff(a[0], b[0]) for a, b in zip(w1[:1], w16[:1])]"),
    ('    d_arms = {a: any(inv[t]["w16_rows_differing"] for t in TAGS if t[:2] == a) for a in ("D0", "D1")}',
     '    d_arms = {a: True for a in ("D0", "D1")}'),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p223_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p223_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p223_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P223_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


@pytest.mark.parametrize("run_id,ok", [("p223-5090-1", True), ("p223-5090-3", True), ("p223-5090-4", False),
                                       ("p149-5090-1", False), ("p223-dry-1", False)])
def test_the_driver_takes_the_registered_ids_only(tmp_path, run_id, ok):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p223_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, run_id), timeout=120)
    if ok:
        assert out.returncode == 0 and "DRYRUN stage -> root@h:/root/p223 " in out.stdout + " ", out.stdout + out.stderr
        assert "bash p223_run.sh" in out.stdout
    else:
        assert out.returncode == 78 and "is not p223-5090-N" in out.stdout + out.stderr


def test_the_driver_stages_what_the_pins_name():
    drive = (LANE / "p223_drive.sh").read_text(encoding="utf-8")
    stage = drive[drive.index('STAGE="'):drive.index('for f in $STAGE')]
    names = {pathlib.Path(p).name for p in re.findall(r"\$\w+(?:/\w+)?/([\w.\-]+)", stage)}
    assert names == set(SOURCES) | {"staged-p223.sha256"}, names
    assert "--exclude 'work/*'" in drive and "P223_RUN_NONCE" in drive and "P149_" not in drive and "p149-" not in drive
