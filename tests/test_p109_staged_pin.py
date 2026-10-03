"""The p109 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P109).

`bench/p109/p109_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p109/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- P39's NF4 bake and host calibration at the bytes SC1 stages;
- the order: refusals, install and tripwire, the self-tests, the premise on the card, then (and only then) the fetch, the
  bake, the prompts, the five arms and the reducer;
- the subject: the default server, with only each arm's switch set;
- the rule's constants and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p109"
P39 = REPO / "bench" / "p39"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p109_run.sh": LANE / "p109_run.sh",
    "p109_box.py": LANE / "p109_box.py",
    "p109_reduce.py": LANE / "p109_reduce.py",
    "k8_bake.py": P39 / "k8_bake.py",
    "calib.json": P39 / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
}
RUN = (LANE / "p109_run.sh").read_text()
REDUCE = (LANE / "p109_reduce.py").read_text()
BOX = (LANE / "p109_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p109 launch refuses ON A RENTED BOX"


def test_the_bake_and_calibration_are_the_bytes_sc1_stages():
    sc1 = dict((n, w) for w, n in _entries(REPO / "bench" / "sc1" / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    assert mine["k8_bake.py"] == sc1["k8_bake.py"] and mine["calib.json"] == sc1["calib.json"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p109_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p109_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver           # the arena stays on the box


def test_the_self_tests_pass():
    for script, want in (("p109_reduce.py", "self-test OK (18 cases)"), ("p109_box.py", "self-test OK (5/5 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True)
        assert out.returncode == 0 and want in out.stdout, out.stdout + out.stderr


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("E1", "G1", "G2", "E2", "D1", "P1")' in REDUCE                  # Amendment 2: P1
    assert "SELF_LO, SELF_HI = 0.93, 1.07" in REDUCE and "S16_MIN, S1_MIN = 1.25, 0.97" in REDUCE
    assert "SANE_ROWS, SANE_TOKENS = 12, 16" in REDUCE
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == "51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00"   # v0.35.0, CI's pin
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p109_reduce.py --self-test", "p109_box.py --self-test", "python -m pytest test_decode_graph_buckets.py",
             'echo "premise ok"', 'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "p109_box.py --prompts-only",
             "for TAG in E1 G1 G2 E2 D1 P1; do", "python $W/p109_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "7 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert "serve_paged.PagedServeConfig().graphs is False" in trip               # the subject is the eager default
    assert 'md.version("grouped-nf4-gemm") == "0.35.0"' in trip and "fp8_kv_append_bt1" in trip


def test_the_subject_is_the_default_server():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(': > summary.txt')]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS", "E4B_PAGED_PLACEMENT", "E4B_PAGED_FUSE_QKV",
                 "E4B_NF4_GROUPED_SMALLM", "E4B_SERVE_EXP_INT4"):
        assert knob in unset, knob
    assert 'ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"' in RUN
    assert 'case "$ARM" in G|P) GR="E4B_PAGED_GRAPHS=1";; esac' in RUN
    assert '(cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, "all-vram", (1, 2, 4, 8, 16))' in BOX
    assert "hr.DEVICE_GROUPING[0] = True" in BOX and "hr.FORCE_SINGLETON_GROUPS[0] = False" in BOX
    assert 'kw["capture"] = False' in BOX and "paged_runner.PagedModelRunner.enable_decode_graphs = _padded_eager" in BOX
    assert '"FUNCTION_FAIL"' in REDUCE and 'fn.append(f"{t} != P1 on {w} at {n} tokens, rows {rows}")' in REDUCE


def test_the_box_reads_the_registered_shape():
    assert "ROWS, PROMPT, OFFSET = 16, 512, 4096" in BOX and 'WORKLOADS = {"W16": ROWS, "W1": 1}' in BOX
    for flag in ('"--short", type=int, default=32', '"--long", type=int, default=160', '"--reps", type=int, default=3'):
        assert flag in BOX, flag
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3" in RUN and "SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1" in RUN


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes
    assert {16, 25, 27} <= codes                                             # host RAM, premise, an unproved proof


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p109-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P109_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p109_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p109"), out.stdout + out.stderr


def test_every_time_left_check_fits_its_own_guard():
    """Amendment 1 (p109-prove-1): the proof's checks were sized for the reading and could never pass in its guard."""
    prereg = (LANE / "PREREG-p109.md").read_text()
    assert "guard 0.75 h" in prereg and "guard 2.5 h" in prereg
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|ARM)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|ARM)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == {"FETCH", "BAKE", "ARM"}
    for need in prove.values():                       # after 15 min of install and premise, with the 600 s margin
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 2.5 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM"]
