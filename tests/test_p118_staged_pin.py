"""The p118 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P118; e4b#1313).

`bench/p118/p118_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p118/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- P109's box, P39's bake and calibration and the graph premise tests at the bytes P115 and P116 pinned;
- the order: refusals, install and tripwire, the self-tests, the premise on the card (the graphs with the lookahead
  through them, 16; the protocol, 18), then the fetch, the bake, the prompts, the four arms and the reducer;
- the subject: the default graph server at 16 slots, with only the lookahead switch differing between arms;
- the rule's constants against the runner and the PREREG, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p118"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p118_run.sh": LANE / "p118_run.sh",
    "p118_box.py": LANE / "p118_box.py",
    "p118_reduce.py": LANE / "p118_reduce.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_kv_step_select.py": REPO / "tests" / "test_kv_step_select.py",
    "test_decode_lookahead.py": REPO / "tests" / "test_decode_lookahead.py",
    "test_decode_lookahead_gpu.py": REPO / "tests" / "test_decode_lookahead_gpu.py",
}
PREMISE_GRAPHS = ("test_decode_graph_buckets.py", "test_kv_step_select.py", "test_decode_lookahead_gpu.py")
RUN = (LANE / "p118_run.sh").read_text()
DRIVE = (LANE / "p118_drive.sh").read_text()
REDUCE = (LANE / "p118_reduce.py").read_text()
BOX = (LANE / "p118_box.py").read_text()
PREREG = (LANE / "PREREG-p118.md").read_text(encoding="utf-8")


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _env():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}   # Windows needs it
    return {"PYTHONPATH": str(REPO / "bench" / "p109"), **base}


def _import(name, *dirs):
    sys.path[:0] = [str(d) for d in dirs]
    try:
        return __import__(name)
    finally:
        del sys.path[:len(dirs)]


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p118 launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    for lane in ("p115", "p116"):
        theirs = dict((n, w) for w, n in _entries(REPO / "bench" / lane / "staged.sha256"))
        for name in ("p109_box.py", "k8_bake.py", "calib.json", "test_decode_graph_buckets.py", "test_kv_step_select.py"):
            assert mine[name] == theirs[name], f"{name} is not at {lane.upper()}'s bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p118_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in DRIVE and "traces/" in DRIVE


def test_the_self_tests_pass():
    out = subprocess.run([sys.executable, str(LANE / "p118_reduce.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p118_reduce self-test OK (28 cases)" in out.stdout, out.stdout + out.stderr
    out = subprocess.run([sys.executable, str(LANE / "p118_box.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p118_box self-test OK (12/12 cases)" in out.stdout, out.stdout + out.stderr
    assert "self-tested on 28 cases" in PREREG and "self-test (12 cases)" in PREREG


def test_the_rule_is_the_registered_rule():
    r = _import("p118_reduce", LANE)
    assert r.TAGS == ("L0a", "L1a", "L1b", "L0b") and r.WORKLOADS == ("W16", "W1")
    assert (r.SELF_LO, r.SELF_HI, r.SELF_W1_LO, r.SELF_W1_HI) == (0.96, 1.04, 0.99, 1.01)
    assert (r.GAIN_MIN_W1, r.GAIN_MIN_W16, r.OVERLAP_MIN, r.GAP_MIN_MS) == (1.02, 0.99, 0.9, 0.2)
    assert "**UNTESTED (premise unmet):** L0's traced W1 host gap is **< 0.2 ms**" in PREREG
    assert "[0.96, 1.04] at W16" in PREREG and "[0.99, 1.01] at W1" in PREREG
    assert "**< 1.02**" in PREREG and "**< 0.99**" in PREREG and "at least **90 %**" in PREREG
    # the registered constant (0.43.0, CI's pin when this page was registered); the box installs it itself, so CI's pin
    # moving later is not a change to the lane's rule (the coupling P115's pin test dropped in #1341)
    assert f"GNF4_SHA={r.GNF4_SHA}" in RUN and r.GNF4_SHA == "6ee2e10408161a9d3c874975c9191a7f2957e6f4"
    assert "**`6ee2e10`** (0.43.0" in PREREG
    for model, rev in r.REVS.items():
        assert rev in RUN, model
    b = _import("p118_box", LANE, REPO / "bench" / "p109")
    assert b.ARMS == tuple(sorted({t[:2] for t in r.TAGS})) and b.KNOB == "E4B_PAGED_DECODE_LOOKAHEAD"


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "python - <<'PYT'", "p118_reduce.py --self-test", "p118_box.py --self-test",
             "python -m pytest " + " ".join(PREMISE_GRAPHS), 'python -m pytest test_decode_lookahead.py -k "not server_answers"',
             'echo "premise ok"', 'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "p118_box.py --prompts-only",
             "for TAG in L0a L1a L1b L0b; do", "python $W/p118_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'echo "$LASTG" | grep -q "16 passed"' in RUN and 'echo "$LASTP" | grep -q "18 passed"' in RUN
    assert "**16 passed**" in PREREG and "**18 passed**" in PREREG
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'serve_paged._lookahead_env("") is False and serve_paged._lookahead_env("1") is True' in trip
    assert '"lookahead=cfg.decode_lookahead" in inspect.getsource(serve_paged.build_engine)' in trip
    assert 'getattr(PagedModelRunner, "issue_decode", None)' in trip and "StepTrace" in trip
    assert 'md.version("grouped-nf4-gemm") == "0.43.0"' in trip and 'transformers.__version__ == "5.17.0"' in trip


def test_the_premise_collects_the_registered_counts():
    """16 graph cases on a CUDA card (none skip at sm_89+); the protocol file collects 13 without CUDA and its five
    device-parametrized cases double on a CUDA card: 18 (its HTTP case deselected, the box installs no fastapi)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          *[str(REPO / "tests" / f) for f in PREMISE_GRAPHS]], capture_output=True, text=True, cwd=REPO)
    assert re.search(r"\b16 tests? collected\b", out.stdout), out.stdout[-600:] + out.stderr[-600:]
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          str(REPO / "tests" / "test_decode_lookahead.py"), "-k", "not server_answers"],
                         capture_output=True, text=True, cwd=REPO)
    m = re.search(r"\b(\d+)/(\d+) tests? collected \(1 deselected\)", out.stdout)
    assert m, out.stdout[-600:] + out.stderr[-600:]
    lines = [x for x in out.stdout.splitlines() if "::" in x]
    on_cuda = sum(2 if re.search(r"[[-]cpu]$", x) else 1 for x in lines)      # cuda doubles the device-parametrized ones
    import torch
    if torch.cuda.is_available():
        assert int(m.group(1)) == 18, out.stdout[-600:]
    else:
        assert int(m.group(1)) == 13 and on_cuda == 18, (m.group(0), on_cuda)


def test_the_subject_is_the_default_graph_server_and_only_the_switch_differs():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_KV_STEP_SELECT", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE",
                 "E4B_PAGED_DECODE_LOOKAHEAD", "E4B_PAGED_STEP_TRACE", "GNF4_PDL", "GNF4_GEMV_BW", "TRITON_INTERPRET"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert ('ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena '
            'E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"') in RUN
    assert 'LA="E4B_PAGED_DECODE_LOOKAHEAD=1"' in RUN and 'ARM=${TAG:0:2}; KN=""; [ "$ARM" = L1 ] && KN=$LA' in RUN
    assert "--trace-dir $W/traces" in RUN
    assert "(cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, \"all-vram\", (1, 2, 4, 8, 16))" in BOX
    assert 'if cfg.decode_lookahead != (arm == "L1"):' in BOX and 'if parts.scheduler.lookahead != (arm == "L1"):' in BOX
    # the traced passes run after every timed pass and outside them
    assert BOX.index('rec["graph_stats"] =') < BOX.index("traced_pass(parts, torch, rows[:b]")
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3" in RUN and "SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1" in RUN


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 0.75 h" in PREREG and "guard 1.25 h" in PREREG
    keys = {"FETCH", "BAKE", "ARM"}
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|ARM)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|ARM)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == keys
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 1.25 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM"]
    assert "Lane ceiling $2.50, hard stop $3.00" in PREREG and "over $15 needs the maintainer lane" in PREREG


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    # 13 the disk floor and 18 the CUDA host floor; 14 and 17 are the launcher's. 18 appears once.
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 15, 16, 21, 22, 25, 27, 40} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p118-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P118_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p118_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p118"), out.stdout + out.stderr
