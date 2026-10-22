# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P223 (bench/p223/PREREG-p223.md): the staged pins, the arms as the box, the reducer and the runner each state
them, the runner's order (the host floors and the guard fit before any install, the compiled premises before the fetch,
the eight processes in the registered order with each arm's first read before the next, the reduce), the fit table
against the registration's and its 3.5 h guard, the self-tests run as the box runs them (one directory), the mutants the
self-tests must kill, and the driver's run ids. Amendment 1's shake mode: its table against the 1.75 h guard, the switch
refusing a mismatch either way, a shake reduce that can never read READ, and every check kept."""
import json
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
           "p149_slice.py": B / "p149",
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
    for name in ("test_dense_i8.py", "test_int8_attn_interp.py", "p109_box.py", "k8_bake.py", "calib.json", "p149_slice.py"):
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
    assert "p223-5090-1|p223-5090-2|p223-5090-3) SHAKE=0;;" in RUN and "`p223-5090-1` is the reading" in PRE
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
    for gone in ("--mode gate", "--mode tf", "--mode sane", "--windows", "--q ", "--q-only", "P223_DRY"):
        assert gone not in RUN, gone
    gate = RUN[RUN.index("gate_check(){"):RUN.index("for TAG in")]
    assert "phase_start reduce reduce; reduce_all; phase_end reduce; finish 45" in gate, "an early VOID still reduces"


def test_the_phase_stamps_are_the_common_writers_with_a_kind_each():
    """bench/common/README.md "Phase stamps": the runner sources the staged phases.sh (after the fit check, as another
    file's script must come) and keeps no local writer; every phase opens with a registered kind and closes."""
    assert "stamp(){" not in RUN and re.search(r"^\s*stamp ", RUN, re.M) is None
    starts = re.findall(r"phase_start ([^\s;]+) ([a-z]+)", RUN)
    ends = set(re.findall(r"phase_end ([^\s;]+)", RUN))
    assert {n for n, _k in starts} == ends == {"install", "fetch", "slice", "bake", '"$TAG"', "reduce"}, (starts, ends)
    assert {(n, k) for n, k in starts} == {("install", "setup"), ("fetch", "setup"), ("slice", "setup"), ("bake", "setup"),
                                           ('"$TAG"', "measure"), ("reduce", "reduce")}
    assert RUN.index(". ./phases.sh") > RUN.index('[ "$LEFT" -ge "$FIT_NEED" ]')


TERMS = r"\b(INSTALL|NEED_FETCH|NEED_BAKE|NEED_ARM|NEED_REDUCE|RESERVE|MIN_DISK_GB|REPS)=(\d+)"
SHAKE_BLOCK = 'if [ "$SHAKE" = 1 ]; then   # the shake table (Amendment 1)'


def _tables():
    """The reading's table (every assignment before the shake block) and the shake's (the reading's, overridden in it)."""
    head, shake = RUN.split(SHAKE_BLOCK, 1)
    box = {k: int(x) for k, x in re.findall(TERMS, head)}
    return box, {**box, **{k: int(x) for k, x in re.findall(TERMS, shake.split("\nfi\n", 1)[0])}}


def _fit(v):
    return v["INSTALL"] + v["NEED_FETCH"] + v["NEED_BAKE"] + 8 * v["NEED_ARM"] + v["NEED_REDUCE"] + v["RESERVE"]


def test_the_fit_table_is_the_registrations_and_fits_three_and_a_half_hours():
    """PREREG "The fit": boot 300 + install 450 + fetch 3,000 + bake 1,500 + 8 x 480 + reduce 120 + reserve 600 = 9,810 s;
    x 1.25 = 12,263 s inside the 3.5 h guard; the runner's own check before the install needs (9,810 - 300) x 1.25 =
    11,888 s left."""
    v, _shake = _tables()
    assert v == {"INSTALL": 450, "NEED_FETCH": 3000, "NEED_BAKE": 1500, "NEED_ARM": 480, "NEED_REDUCE": 120,
                 "RESERVE": 600, "MIN_DISK_GB": 200, "REPS": 5}, v
    for term, s in (("boot", "300"), ("install", "450"), ("fetch", "3,000"), ("bake", "1,500"),
                    ("8 arm processes × 480", "3,840"), ("reduce", "120"), ("reserve", "600")):
        assert f"| {term} | {s} |" in PRE, term
    assert "| **total** | **9,810** |" in PRE and "inside the **3.5 h guard** (12,600 s)" in PRE
    assert "(9,810 − 300) × 1.25 = 11,888 s" in PRE
    assert "FIT=$((INSTALL + NEED_FETCH + NEED_BAKE + 8 * NEED_ARM + NEED_REDUCE + RESERVE))" in RUN
    assert "FIT_HEADROOM_PCT=125" in RUN and "FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))" in RUN     # the canonical form
    boot, guard = 300, int(3.5 * 3600)
    fit = _fit(v)
    assert boot + fit == 9810 and (boot + fit) * 1.25 <= guard
    need = fit * 125 // 100                                  # the runner's check in integer arithmetic
    assert need == 11887 and 11888 - need == 1               # 1 s under the registration's 11,888 (the half second rounded up)
    assert guard - need >= boot                              # the guard leaves boot its allowance after that check


def test_the_shake_table_is_amendment_1s_and_fits_its_guard():
    """Amendment 1's shake table: boot 300 + install 450 + fetch 900 + the cut and the bake 600 + 8 x 240 + reduce 120 +
    reserve 600 = 4,890 s; x 1.25 = 6,112 s inside the 1.75 h shake guard; the runner's own check needs (4,890 - 300) x
    1.25 = 5,737 s left before the install. Two timed reps, a 60 GB disk; everything else the reading's."""
    box, v = _tables()
    assert v == {"INSTALL": 450, "NEED_FETCH": 900, "NEED_BAKE": 600, "NEED_ARM": 240, "NEED_REDUCE": 120,
                 "RESERVE": 600, "MIN_DISK_GB": 60, "REPS": 2}, v
    assert {k: x for k, x in box.items() if k in ("INSTALL", "NEED_REDUCE", "RESERVE")} == \
        {k: x for k, x in v.items() if k in ("INSTALL", "NEED_REDUCE", "RESERVE")}
    for term, s in (("fetch (two shards, 9.1 GB)", "900"), ("the cut and the bake", "600"),
                    ("8 arm processes × 240 (2 timed reps)", "1,920"), ("install, self-tests and the compiled suites", "450")):
        assert f"| {term} | {s} |" in PRE, term
    assert "| **total** | **4,890** |" in PRE and "inside the **1.75 h shake guard** (6,300 s)" in PRE
    assert re.search(r"\(4,890 − 300\) ×\s+1\.25 = 5,737 s", PRE)
    boot, guard = 300, int(1.75 * 3600)
    fit = _fit(v)
    assert boot + fit == 4890 and (boot + fit) * 1.25 <= guard
    need = fit * 125 // 100                                  # the runner's canonical check, the same line as the reading's
    assert need == 5737 and guard - need >= boot
    assert RUN.count("FIT=$((") == 1 and RUN.index(SHAKE_BLOCK) < RUN.index("FIT=$((")


def _switch(text, run_id, shake):
    """The runner's id switch, cut from ``text`` and run on its own: rc and the table it leaves."""
    snippet = ('say(){ echo "$*"; }\nfinish(){ echo "rc=$1"; exit "$1"; }\n'
               "REPS=5; NEED_FETCH=3000; NEED_BAKE=1500; NEED_ARM=480; MIN_DISK_GB=200\n"
               + text[text.index('case "$P223_RUN_ID" in'):text.index('RS=""; [ "$SHAKE" = 1 ] && RS=--shake')]
               + 'echo "SHAKE=$SHAKE REPS=$REPS NEED_ARM=$NEED_ARM"\n')
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "P223_RUN_ID": run_id}
    if shake is not None:
        env["P223_SHAKE"] = shake
    out = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, env=env, timeout=30)
    return out.returncode, out.stdout


SWITCH = [("p223-5090-1", None, 0, "SHAKE=0 REPS=5 NEED_ARM=480"), ("p223-5090-3", "0", 0, "SHAKE=0 REPS=5"),
          ("p223-shake-1", "1", 0, "SHAKE=1 REPS=2 NEED_ARM=240"), ("p223-shake-3", "1", 0, "SHAKE=1 REPS=2"),
          ("p223-5090-2", "1", 78, "does not match"), ("p223-shake-2", None, 78, "does not match"),
          ("p223-shake-1", "0", 78, "does not match"), ("p223-shake-6", "1", 0, "SHAKE=1 REPS=2"),
          ("p223-shake-5", None, 78, "does not match"), ("p223-shake-7", "1", 78, "unregistered"),
          ("p223-dry-1", None, 78, "unregistered")]


@pytest.mark.parametrize("run_id,shake,rc,want", SWITCH)
def test_the_switch_refuses_a_mismatch_either_way_before_any_install(run_id, shake, rc, want):
    if shutil.which("bash") is None:
        pytest.skip("the runner needs a POSIX bash")
    got, out = _switch(RUN, run_id, shake)
    assert got == rc and want in out, (got, out)
    check = RUN.index('[ "${P223_SHAKE:-0}" = "$SHAKE" ] || {')
    assert check < RUN.index("pipx logs/pip_e4b.log") and check < RUN.index("FIT=$((") and "finish 78; }" in RUN[check:check + 200]


SWITCH_MUTANTS = [('[ "${P223_SHAKE:-0}" = "$SHAKE" ] ||', "true ||"),
                  ("  p223-shake-1|p223-shake-2|p223-shake-3|p223-shake-4|p223-shake-5|p223-shake-6) SHAKE=1;;",
                   "  p223-shake-1|p223-shake-2|p223-shake-3|p223-shake-4|p223-shake-5|p223-shake-6) SHAKE=0;;"),
                  ("  NEED_FETCH=900; NEED_BAKE=600; NEED_ARM=240; MIN_DISK_GB=60; REPS=2",
                   "  NEED_FETCH=900; NEED_BAKE=600; NEED_ARM=240; MIN_DISK_GB=60")]


@pytest.mark.parametrize("old,new", SWITCH_MUTANTS)
def test_each_switch_mutant_fails_the_switch_cases(old, new):
    if shutil.which("bash") is None:
        pytest.skip("the runner needs a POSIX bash")
    assert RUN.count(old) == 1, old
    bad = RUN.replace(old, new)
    failed = []
    for rid, sh, rc, want in SWITCH:
        got, out = _switch(bad, rid, sh)
        if got != rc or want not in out:
            failed.append((rid, sh))
    assert failed, "the mutant passes every switch case"


def _reducer():
    sys.path.insert(0, str(LANE))
    try:
        import p223_reduce as red
    finally:
        sys.path.pop(0)
    return red


def test_a_shake_reduce_never_reads_READ_even_on_all_pass_records():
    """Amendment 1: the verdict is SHAKE, hard-coded. On clean one-layer records it is SHAKE with no instrument fault;
    the reading's rule would read them (labels and all), and the shake still writes no READ."""
    red = _reducer()
    saved = red._consts()
    try:
        red.use_shake()
        arms = red._shake_fixture()
        v = red.reduce_shake(arms, red.E)
        assert v["verdict"] == "SHAKE" and v["ok"] and v["instrument_faults"] == [], v["instrument_faults"]
        assert '"READ"' not in json.dumps(v, default=str)
        assert v["reported"]["labels"] == {"S0": "SCHEDULE_INVARIANT", "S1": "SCHEDULE_INVARIANT"}
        assert {p["graded"] for p in v["reported"]["predictions"]} == {"not graded (shake)"}
        assert red.reduce(arms, red.E)["verdict"] == "READ"       # the same records under the reading's rule would read
    finally:
        red._restore(saved)
    sh = RUN[RUN.index('if [ "$SHAKE" = 1 ]; then   # never a reading'):RUN.index('case "$V" in READ)')]
    assert '[ "$V" = SHAKE ]' in sh and "finish 0; fi" in sh and sh.rstrip().endswith("finish 29\nfi")


def _shake_dir(tmp_path):
    red = _reducer()
    saved = red._consts()
    try:
        red.use_shake()
        arms = red._shake_fixture()
    finally:
        red._restore(saved)
    for t, r in arms.items():
        (tmp_path / f"arm_{t}.json").write_text(json.dumps(r), encoding="utf-8")
    return red.E


CLI_MUTANTS = [("    v = reduce_shake(load(a.dir), a.e4b_sha) if a.shake else reduce(load(a.dir), a.e4b_sha)",
                "    v = reduce(load(a.dir), a.e4b_sha)"),
               ("    if a.shake:\n        use_shake()", "    if a.shake:\n        pass")]


@pytest.mark.parametrize("mutant", [None] + CLI_MUTANTS)
def test_the_shake_cli_reads_shake_on_clean_one_layer_records(tmp_path, mutant):
    """The runner's call (--shake) on a clean shake directory: SHAKE, ok, no READ. Each CLI mutant breaks it."""
    e = _shake_dir(tmp_path)
    src = (LANE / "p223_reduce.py").read_text(encoding="utf-8")
    if mutant:
        assert src.count(mutant[0]) == 1, mutant[0]
        src = src.replace(*mutant)
    (tmp_path / "red").mkdir()
    (tmp_path / "red" / "p223_reduce.py").write_text(src, encoding="utf-8")
    out = _run(tmp_path / "red" / "p223_reduce.py", "--dir", str(tmp_path), "--out", str(tmp_path / "v.json"),
               "--e4b-sha", e, "--shake")
    v = json.loads((tmp_path / "v.json").read_text(encoding="utf-8")) if (tmp_path / "v.json").exists() else {}
    good = out.returncode == 0 and v.get("verdict") == "SHAKE" and v.get("ok") is True and '"READ"' not in json.dumps(v)
    assert good == (mutant is None), (out.stdout[-600:], out.stderr[-600:])


def test_a_shake_keeps_every_check():
    """Amendment 1, "every check kept": the shake runs the reading's own per-process checks (process_faults) and the
    reading's gate; it reports and goes on where the reading stops; every VOID case of the reducer's self-test also faults
    a shake on its own one-layer records; and the runner's checks carry no shake branch."""
    src = (LANE / "p223_reduce.py").read_text(encoding="utf-8")
    shake = src[src.index("def reduce_shake("):src.index("def gate_faults(")]
    assert "faults += process_faults(t, r, e4b_sha)" in shake and "void += process_faults(t, arms[t], e4b_sha)" in src
    gate = src[src.index("def gate_faults("):src.index("def _json(")]
    assert "return process_faults(tag, r, e4b_sha)" in gate and "shake" not in gate.lower()
    red = _reducer()
    cases = []
    red._cases_engagement(cases)
    void = [c for c in cases if not c[0].endswith("; the shake lists it")]
    shook = [c for c in cases if c[0].endswith("; the shake lists it")]
    assert len(void) == len(shook) >= 20 and all(ok for _d, ok, _x in cases), [c[0] for c in cases if not c[1]]
    run_gate = RUN[RUN.index("gate_check(){"):RUN.index("for TAG in")]
    assert 'p223_reduce.py --gate $W/arm_$1.json --e4b-sha "$E4B_SHA" $RS' in run_gate
    assert 'if [ "$SHAKE" = 1 ]; then echo "SHAKE_GATE_FAULT $1"' in run_gate and "finish 45" in run_gate
    assert 'p223_reduce.py --dir $W --out $W/verdict_p223.json --e4b-sha "$E4B_SHA" $RS' in RUN
    for x in ("sha256sum -c staged-p223.sha256", '[ "$CC" = 12.0 ]', '[ "$EPL" -ge "$MIN_POWER_W" ]', "p223_box.py --self-test",
              "p223_reduce.py --self-test", "grep -qE '^28 passed", "grep -qE '^4 passed", "TRIPWIRE FAIL"):
        assert RUN.count(x) == 1, x                           # one unconditional line, no shake branch around it


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
    # Amendment 1: the shake
    ('    out = {"lane": "P223 shake (Amendment 1)", "verdict": "SHAKE",', '    out = {"lane": "P223 shake (Amendment 1)", "verdict": "READ",'),
    ("        faults += process_faults(t, r, e4b_sha)", "        faults += []"),
    ('            faults.append(f"{t}: missing or not ok")', "            pass"),
    ("    PROJ, CENSUS = 4 * LAYERS, census(LAYERS)", "    PROJ, CENSUS = 4 * LAYERS, census(32)"),
    ('    MODEL, REV, LAYERS, REPS = SHAKE_SLICE, "", SHAKE_LAYERS, SHAKE_REPS',
     '    MODEL, REV, LAYERS, REPS = SHAKE_SLICE, "", SHAKE_LAYERS, 5'),
    ("            if len(reps) != REPS or any(", "            if len(reps) != 5 or any("),
    ('    if r.get("reps") != REPS or (r.get("short"), r.get("long")) != (SHORT, LONG):',
     '    if r.get("reps") != 5 or (r.get("short"), r.get("long")) != (SHORT, LONG):'),
    ('"graded": "not graded (shake)"}', '"graded": p["graded"]}'),
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
                                       ("p223-shake-1", True), ("p223-shake-3", True), ("p223-shake-6", True), ("p223-shake-7", False),
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
        assert ("P223_SHAKE=1" in out.stdout) == run_id.startswith("p223-shake-"), out.stdout
    else:
        assert out.returncode == 78 and "is not p223-5090-N" in out.stdout + out.stderr


def test_the_driver_stages_what_the_pins_name():
    drive = (LANE / "p223_drive.sh").read_text(encoding="utf-8")
    stage = drive[drive.index('STAGE="'):drive.index('for f in $STAGE')]
    names = {pathlib.Path(p).name for p in re.findall(r"\$\w+(?:/\w+)?/([\w.\-]+)", stage)}
    assert names == set(SOURCES) | {"staged-p223.sha256"}, names
    assert "--exclude 'work/*'" in drive and "P223_RUN_NONCE" in drive and "P149_" not in drive and "p149-" not in drive
