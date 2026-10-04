"""The p111 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P111).

`bench/p111/p111_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p111/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- P109's box (whose prompts, passes and slope P111's box imports) at its registered bytes, and P39's bake and
  calibration at SC1's;
- the order: refusals, install and tripwire, the self-tests, the premise on the card, then the fetch, the bake, the
  prompts, the four arms and the reducer;
- the subject: the default graph server, with only E4B_KV_STEP_SELECT differing between arms;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p111"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p111_run.sh": LANE / "p111_run.sh",
    "p111_box.py": LANE / "p111_box.py",
    "p111_reduce.py": LANE / "p111_reduce.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_kv_step_select.py": REPO / "tests" / "test_kv_step_select.py",
}
RUN = (LANE / "p111_run.sh").read_text()
REDUCE = (LANE / "p111_reduce.py").read_text()
BOX = (LANE / "p111_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p111 launch refuses ON A RENTED BOX"


def test_imported_and_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p109 = dict((n, w) for w, n in _entries(REPO / "bench" / "p109" / "staged.sha256"))
    assert mine["p109_box.py"] == p109["p109_box.py"]
    assert mine["k8_bake.py"] == p109["k8_bake.py"] and mine["calib.json"] == p109["calib.json"]
    assert mine["test_decode_graph_buckets.py"] == p109["test_decode_graph_buckets.py"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p111_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p111_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver


def test_the_self_tests_pass():
    out = subprocess.run([sys.executable, str(LANE / "p111_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (11 cases)" in out.stdout, out.stdout + out.stderr
    env = {"PYTHONPATH": str(REPO / "bench" / "p109"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, str(LANE / "p111_box.py"), "--self-test"], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "p111_box self-test OK (3/3 cases)" in out.stdout, out.stdout + out.stderr


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("S0a", "S1a", "S1b", "S0b")' in REDUCE
    assert "SELF_LO, SELF_HI = 0.96, 1.04" in REDUCE and "GAIN_MIN = 1.00" in REDUCE
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == "51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00"
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p111_reduce.py --self-test", "p111_box.py --self-test",
             "python -m pytest test_decode_graph_buckets.py test_kv_step_select.py", 'echo "premise ok"',
             'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "p111_box.py --prompts-only",
             "for TAG in S0a S1a S1b S0b; do", "python $W/p111_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "13 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'serve_paged._graphs_env("", "cuda", "all-vram") is True' in trip
    assert "fp8_paged_kv._step_select_env(None) is False" in trip and '"graph_bucket_publish"' in trip


def test_the_subject_is_the_default_graph_server_and_only_the_switch_differs():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(': > summary.txt')]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_KV_STEP_SELECT", "E4B_PAGED_MAX_SEQS", "E4B_NF4_GROUPED_SMALLM"):
        assert knob in unset, knob
    assert 'ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"' in RUN
    assert 'ARM=${TAG:0:2}; SS="E4B_KV_STEP_SELECT=0"; [ "$ARM" = S1 ] && SS="E4B_KV_STEP_SELECT=1"' in RUN
    assert "if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, \"all-vram\", (1, 2, 4, 8, 16)):" in BOX
    assert '"step_select": bool(getattr(kv, "_step_select", False))' in BOX
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3" in RUN and "SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1" in RUN


def test_every_time_left_check_fits_its_own_guard():
    prereg = (LANE / "PREREG-p111.md").read_text()
    assert "guard 0.75 h" in prereg and "guard 1.25 h" in prereg
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|ARM)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|ARM)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == {"FETCH", "BAKE", "ARM"}
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 1.25 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes
    assert {16, 25, 27} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p111-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P111_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p111_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p111"), out.stdout + out.stderr
