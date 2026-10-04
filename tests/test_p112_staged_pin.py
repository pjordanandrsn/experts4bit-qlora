"""The p112 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P112).

`bench/p112/p112_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p112/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- P109's box (whose prompts, passes and slope P112's box imports) at its registered bytes, and P39's bake and
  calibration at SC1's;
- the order: refusals, install and tripwire, the self-tests, the premise on the card (e4b's graph buckets and
  grouped-nf4-gemm's GNF4_PDL contract), then the fetch, the bake, the prompts, the four arms and the reducer;
- the subject: the default graph server with SC1's int4 configuration, with only GNF4_PDL differing between arms;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p112"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p112_run.sh": LANE / "p112_run.sh",
    "p112_box.py": LANE / "p112_box.py",
    "p112_reduce.py": LANE / "p112_reduce.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
}
RUN = (LANE / "p112_run.sh").read_text()
REDUCE = (LANE / "p112_reduce.py").read_text()
BOX = (LANE / "p112_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p112 launch refuses ON A RENTED BOX"


def test_imported_and_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p109 = dict((n, w) for w, n in _entries(REPO / "bench" / "p109" / "staged.sha256"))
    assert mine["p109_box.py"] == p109["p109_box.py"]
    assert mine["k8_bake.py"] == p109["k8_bake.py"] and mine["calib.json"] == p109["calib.json"]
    assert mine["test_decode_graph_buckets.py"] == p109["test_decode_graph_buckets.py"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p112_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p112_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver


def test_the_self_tests_pass():
    out = subprocess.run([sys.executable, str(LANE / "p112_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (17 cases)" in out.stdout, out.stdout + out.stderr
    env = {"PYTHONPATH": str(REPO / "bench" / "p109"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, str(LANE / "p112_box.py"), "--self-test"], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "p112_box self-test OK (5/5 cases)" in out.stdout, out.stdout + out.stderr


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("P0a", "P1a", "P1b", "P0b")' in REDUCE
    assert "SELF_LO, SELF_HI = 0.96, 1.04" in REDUCE and "GAIN_MIN = 1.00" in REDUCE
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == "951a97fbb489b806d238922559b55724b55db8fd"
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p112_reduce.py --self-test", "p112_box.py --self-test",
             "python -m pytest test_decode_graph_buckets.py", "python -m pytest test_pdl.py", 'echo "premise ok"',
             'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "p112_box.py --prompts-only",
             "for TAG in P0a P1a P1b P0b; do", "python $W/p112_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "7 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    assert 'grep -q "23 passed" && ! echo "$LASTL2" | grep -q skipped' in RUN
    assert "git -C $W/gnf4src checkout -q $GNF4_SHA" in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'serve_paged._graphs_env("", "cuda", "all-vram") is True' in trip
    assert '"griddepcontrol.wait" in inspect.getsource(int4_b32._pdl_enter.fn)' in trip
    assert '"launch_pdl" in CUDAOptions.__dataclass_fields__' in trip and "torch_cc() >= (9, 0)" in trip


def test_the_subject_is_the_default_graph_server_and_only_the_switch_differs():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(': > summary.txt')]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_KV_STEP_SELECT", "GNF4_PDL", "E4B_PAGED_MAX_SEQS", "E4B_NF4_GROUPED_SMALLM"):
        assert knob in unset, knob
    int4 = ('INT4_ENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 '
            'E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1 E4B_INT4_GROUPED_SMALLM=auto '
            'E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto E4B_PAGED_FUSE_QKV=1"')
    assert int4 in RUN and "E4B_SERVE_EXP_INT4_CALIB" not in int4
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
    for lever in ("E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4",
                  "E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1",
                  "E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto",
                  "E4B_PAGED_FUSE_QKV=1"):
        assert lever in sc1 and lever in int4, lever                    # SC1's int4_sched levers, verbatim
    assert "env PYTHONPATH= $ENGINE_ENV $INT4_ENV $SS P112_ARM=$ARM" in RUN
    assert 'ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"' in RUN
    assert 'ARM=${TAG:0:2}; SS="GNF4_PDL=0"; [ "$ARM" = P1 ] && SS="GNF4_PDL=1"' in RUN and "for TAG in P0a P1a P1b P0b; do" in RUN
    assert "if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, \"all-vram\", (1, 2, 4, 8, 16)):" in BOX
    assert '"pdl_active": bool(int4_b32.pdl_active("cuda"))' in BOX and "triton.knobs.runtime.launch_enter_hook = None" in BOX
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3" in RUN and "PROVE" not in RUN.split("set -uo pipefail")[1]


def test_every_time_left_check_fits_its_own_guard():
    prereg = (LANE / "PREREG-p112.md").read_text()
    assert "guard 1.0 h" in prereg and "no proving rental" in prereg
    need = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|CENSUS)=(\d+)", RUN))
    assert set(need) == {"FETCH", "BAKE", "ARM", "CENSUS"}
    for v in need.values():
        assert int(v) + 600 <= 1.0 * 3600 - 900, need
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM", "$NEED_CENSUS"]
    assert "python $W/p112_box.py --census --out $W/census_default.json" in RUN
    census = RUN[RUN.index("# ---- the default-server census"):]
    assert "$INT4_ENV" not in census and "GNF4_PDL=1" in census                 # the default server, no levers


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes
    assert {16, 25} <= codes and 27 not in codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p112-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P112_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p112_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p112"), out.stdout + out.stderr
