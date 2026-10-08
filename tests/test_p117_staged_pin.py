"""The p117 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P117 from P110's).

`bench/p117/p117_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p117/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test and pins the lane's shape:
- P108's and P97's boxes (whose helpers P117's box imports) at their registered bytes, P39's bake and calibration at
  SC1's, and the premise test at the bytes P110 registered;
- the order: refusals, install and tripwire, the self-test, the premise, then (and only then) the fetch, the bake, the box
  and the reducer;
- the subject: SC2e's int4 stack built eager with one slot, and every arm's buckets;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p117"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p117_run.sh": LANE / "p117_run.sh",
    "p117_box.py": LANE / "p117_box.py",
    "p117_reduce.py": LANE / "p117_reduce.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
}
RUN = (LANE / "p117_run.sh").read_text()
REDUCE = (LANE / "p117_reduce.py").read_text()
BOX = (LANE / "p117_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p117 launch refuses ON A RENTED BOX"


def test_imported_boxes_and_the_premise_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p108 = dict((n, w) for w, n in _entries(REPO / "bench" / "p108" / "staged.sha256"))
    sc1 = dict((n, w) for w, n in _entries(REPO / "bench" / "sc1" / "staged.sha256"))
    p110 = dict((n, w) for w, n in _entries(REPO / "bench" / "p110" / "staged.sha256"))
    assert mine["p108_box.py"] == p108["p108_box.py"] and mine["p97_box.py"] == p108["p97_box.py"]
    assert mine["k8_bake.py"] == sc1["k8_bake.py"] and mine["calib.json"] == sc1["calib.json"]
    assert mine["test_decode_graph_buckets.py"] == p110["test_decode_graph_buckets.py"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p117_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p117_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver           # the arena stays on the box


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p117_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (22 cases)" in out.stdout, out.stdout + out.stderr
    assert "TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005" in REDUCE
    assert 'FLOORS = ("half", "chunk", "rev")' in REDUCE and 'SUBJECTS = ("W32", "W64", "W64pad")' in REDUCE
    assert 'WINDOWS = {"Qwen/Qwen3-30B-A3B": 64, "ibm-granite/granite-3.1-3b-a800m-instruct": 40}' in REDUCE
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == "b4f93f1c62d1e3436ed45bec8ccd608c90433737"   # v0.42.0, SC2e's
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model


def test_the_box_and_the_reducer_register_the_same_arms():
    for line in ('B16, B8, B32, B64 = (1, 2, 4, 8, 16), (1, 2, 4, 8), (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64)',
                 "PAD_WINDOWS = 48", 'FLOORS = ("half", "chunk", "rev")', 'SUBJECTS = ("W32", "W64", "W64pad")'):
        assert line in BOX and line in REDUCE, line
    for arm, kw in (("R", '{"buckets": B16}'), ("half", '{"buckets": B8}'), ("rev", '{"buckets": B16, "reverse": True}'),
                    ("W32", '{"buckets": B32}'), ("W64", '{"buckets": B64}'),
                    ("W64pad", '{"buckets": B64, "first": PAD_WINDOWS}'),
                    ("mutant_scale", '{"buckets": B64, "mutant": "scale"}'), ("G64", '{"buckets": B64, "capture": True}')):
        assert f'"{arm}": {kw}' in BOX, arm
    assert ('BUCKETS = {"R": B16, "rep": B16, "half": B8, "chunk": B16, "rev": B16, "W32": B32, "W64": B64, "W64pad": B64,'
            in REDUCE)


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p117_reduce.py --self-test", "python -m pytest test_decode_graph_buckets.py", 'echo "premise ok"',
             'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "python $W/p117_box.py --out $W/box.json",
             "python $W/p117_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "7 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert "default_buckets(64) == (1, 2, 4, 8, 16, 32, 64)" in trip and '"capture" in inspect.signature' in trip
    assert "Int4Linear.SMALLM_ROWS_MAX == 16" in trip and "import int4_b32, int4_smallm" in trip
    assert 'md.version("grouped-nf4-gemm") == "0.42.0"' in trip and "import p108_box" in trip


def test_the_subject_is_sc2es_int4_stack_built_eager_with_one_slot():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(': > summary.txt')]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS", "E4B_PAGED_PREFILL_GRAPH", "E4B_PAGED_BULK_KV",
                 "E4B_SERVE_EXP_INT4", "E4B_PAGED_FUSE_QKV"):
        assert knob in unset, knob
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
    for var in ("FOLDS", "SPEEDENV", "ROUTEENV"):                      # SC2e's stack, byte for byte
        line = next(x for x in sc1.splitlines() if x.startswith(f"{var}="))
        assert line.split("   #")[0] in RUN, var
    assert 'else LEVERS="$ROUTEENV $SPEEDENV E4B_PAGED_FUSE_QKV=1"; fi' in RUN
    assert 'ENGINE_KNOBS="E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0"' in RUN
    assert "$ENGINE_KNOBS $LEVERS\"" in RUN
    assert 'cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1' in BOX and "model = parts.runner.model" in BOX
    assert "runner.enable_decode_graphs(buckets, capture=capture, verbose=False)" in BOX
    assert "hr.DEVICE_GROUPING[0] = True" in BOX and 'floor_chunk if arm == "chunk" else chunk' in BOX
    for flag in ('"--windows", type=int, default=64', '"--prompt", type=int, default=512', '"--cont", type=int, default=128',
                 '"--chunk", type=int, default=512', '"--floor-chunk", type=int, default=256', '"--stride", type=int, default=3072'):
        assert flag in BOX, flag
    assert "WINDOWS_DEF=64; CONT_DEF=128" in RUN and "WINDOWS_DEF=40; CONT_DEF=32" in RUN


def test_every_time_left_check_fits_its_own_guard():
    prereg = (LANE / "PREREG-p117.md").read_text()
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
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p117-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P117_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p117_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p117"), out.stdout + out.stderr
