"""The p110 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P110).

`bench/p110/p110_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p110/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test and pins the lane's shape:
- P108's and P97's boxes (whose helpers P110's box imports) at their registered bytes, and P39's bake and calibration at
  SC1's;
- the order: refusals, install and tripwire, the self-test, the premise, then (and only then) the fetch, the bake, the box
  and the reducer;
- the subject: the default server built eager, and every arm's switches;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p110"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p110_run.sh": LANE / "p110_run.sh",
    "p110_box.py": LANE / "p110_box.py",
    "p110_reduce.py": LANE / "p110_reduce.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
}
RUN = (LANE / "p110_run.sh").read_text()
REDUCE = (LANE / "p110_reduce.py").read_text()
BOX = (LANE / "p110_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p110 launch refuses ON A RENTED BOX"


def test_imported_boxes_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p108 = dict((n, w) for w, n in _entries(REPO / "bench" / "p108" / "staged.sha256"))
    sc1 = dict((n, w) for w, n in _entries(REPO / "bench" / "sc1" / "staged.sha256"))
    assert mine["p108_box.py"] == p108["p108_box.py"] and mine["p97_box.py"] == p108["p97_box.py"]
    assert mine["k8_bake.py"] == sc1["k8_bake.py"] and mine["calib.json"] == sc1["calib.json"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p110_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p110_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver           # the arena stays on the box


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p110_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (18 cases)" in out.stdout, out.stdout + out.stderr
    assert "TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005" in REDUCE
    assert 'FLOORS = ("half", "chunk", "rev")' in REDUCE and 'DEVICE_ARMS = ("D", "P", "mutant_scale")' in REDUCE
    assert 'WINDOWS = {"Qwen/Qwen3-30B-A3B": 48, "ibm-granite/granite-3.1-3b-a800m-instruct": 12}' in REDUCE
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == "51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00"
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p110_reduce.py --self-test", "python -m pytest test_decode_graph_buckets.py", 'echo "premise ok"',
             'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "python $W/p110_box.py --out $W/box.json",
             "python $W/p110_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "7 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert "serve_paged.PagedServeConfig().graphs is False" in trip and '"capture" in inspect.signature' in trip
    assert "import p108_box" in trip


def test_the_subject_is_the_default_server_and_every_arm_its_registered_switches():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(': > summary.txt')]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_PLACEMENT", "E4B_NF4_GROUPED_SMALLM", "E4B_SERVE_EXP_INT4"):
        assert knob in unset, knob
    assert 'ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"' in RUN
    assert "if cfg.graphs or cfg.placement != \"all-vram\":" in BOX and "model = parts.runner.model" in BOX
    for arm, kw in (('"R": {}', ""), ('"half": {"halves": True}', ""), ('"rev": {"reverse": True}', ""),
                    ('"D": {"device_grouping": True}', ""), ('"P": {"device_grouping": True, "padded": True}', ""),
                    ('"mutant_scale": {"device_grouping": True, "padded": True, "mutant": "scale"}', "")):
        assert arm in BOX, arm
    assert "runner.enable_decode_graphs(BUCKETS, capture=False, verbose=False)" in BOX
    assert 'floor_chunk if arm == "chunk" else chunk' in BOX
    for flag in ('"--windows", type=int, default=48', '"--group", type=int, default=12', '"--prompt", type=int, default=512',
                 '"--cont", type=int, default=128', '"--chunk", type=int, default=512', '"--floor-chunk", type=int, default=256'):
        assert flag in BOX, flag
    assert "WINDOWS_DEF=48; CONT_DEF=128" in RUN and "WINDOWS_DEF=12; CONT_DEF=32" in RUN


def test_every_time_left_check_fits_its_own_guard():
    prereg = (LANE / "PREREG-p110.md").read_text()
    assert "guard 0.75 h" in prereg and "guard 1.5 h" in prereg
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|BOX)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|BOX)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == {"FETCH", "BAKE", "BOX"}
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 1.5 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_BOX"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes
    assert {16, 25, 26, 27} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p110-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P110_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p110_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p110"), out.stdout + out.stderr
