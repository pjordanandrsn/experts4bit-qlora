"""Lane P121's staged-file pin must match the repo (the e4b#642 check, mirrored from P115 Phase D's; e4b#846).

`bench/p121/p121_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p121/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- P115's quality box, P109's, P110's, P108's and P97's boxes, P39's bake and calibration and the decode-graph premise
  test at the bytes P115 Phase D pinned; the K25 premise test at the bytes P96 pinned;
- the order: refusals, install and tripwire, the self-tests, the premise on the card (11), then the fetch, the bake,
  the prompts, the four speed arms, the two quality phases (skipped when a speed arm failed) and the reducer;
- the subject: the default graph server at 16 slots, only ``E4B_NF4_GROUPED_SMALLM`` differing (0 / auto); quality at
  16 windows a pass on both texts;
- the rule's constants against the runner and the PREREG, every time-left check inside its own guard, the exit codes.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p121"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p121_run.sh": LANE / "p121_run.sh",
    "p121_box.py": LANE / "p121_box.py",
    "p121_reduce.py": LANE / "p121_reduce.py",
    "p115_quality.py": REPO / "bench" / "p115" / "p115_quality.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "p110_box.py": REPO / "bench" / "p110" / "p110_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_k25_row_exact_gpu.py": REPO / "tests" / "test_k25_row_exact_gpu.py",
}
PREMISE = ("test_decode_graph_buckets.py", "test_k25_row_exact_gpu.py")
RUN = (LANE / "p121_run.sh").read_text()
DRIVE = (LANE / "p121_drive.sh").read_text()
BOX = (LANE / "p121_box.py").read_text()
PREREG = (LANE / "PREREG-p121.md").read_text(encoding="utf-8")


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _env():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}   # Windows needs it
    sep = ";" if sys.platform.startswith("win") else ":"
    return {"PYTHONPATH": sep.join(str(REPO / "bench" / d) for d in ("p109", "p110", "p108", "p97", "p115")), **base}


def _import(name, *dirs):
    sys.path[:0] = [str(d) for d in dirs]
    try:
        return __import__(name)
    finally:
        del sys.path[:len(dirs)]


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next P121 launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    phase_d = dict((n, w) for w, n in _entries(REPO / "bench" / "p115" / "staged-d.sha256"))
    for name in ("p115_quality.py", "p109_box.py", "p110_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json",
                 "test_decode_graph_buckets.py"):
        assert mine[name] == phase_d[name], f"{name} is not at P115 Phase D's bytes"
    p96 = dict((n, w) for w, n in _entries(REPO / "bench" / "p96" / "staged.sha256"))
    assert mine["test_k25_row_exact_gpu.py"] == p96["test_k25_row_exact_gpu.py"], "not at P96's bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p121_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in DRIVE and 'done < "$HERE/staged.sha256"' in DRIVE


def test_the_self_tests_pass():
    out = subprocess.run([sys.executable, str(LANE / "p121_reduce.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p121_reduce self-test OK (36 cases)" in out.stdout, out.stdout + out.stderr
    out = subprocess.run([sys.executable, str(LANE / "p121_box.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p121_box self-test OK (21/21 cases)" in out.stdout, out.stdout + out.stderr


def test_the_rule_is_the_registered_rule():
    r = _import("p121_reduce", LANE)
    b = _import("p121_box", LANE, REPO / "bench" / "p109")
    assert r.TAGS == ("K0a", "K1a", "K1b", "K0b") and b.ARMS == ("K0", "K1", "Cm", "Ct") and b.ARM_VALUE == {"K0": "0", "K1": "auto"}
    assert b.ARM_ENV["Cm"] == {b.KNOB: "0", b.T1: "1"} and b.ARM_ENV["Ct"] == {b.KNOB: "1", b.T1: "0"}   # P96's m and t
    assert (r.SELF_LO, r.SELF_HI, r.GAIN_MIN_W16) == (0.96, 1.04, 1.00)
    assert (r.TOL, r.SPREAD_X, r.SPREAD_MIN, r.K8_BUDGET, r.K8_GATED) == (0.01, 2.0, 0.005, 0.05, ())   # ppl: reported
    assert (r.CONT_WINDOWS, r.CONT_GROUP) == (12, 1)
    assert r.GROUP == 16 and r.WINDOWS == {r.QWEN: 48, r.GRAN: 16} and r.LAYERS == {r.QWEN: 48, r.GRAN: 32}
    assert r.OFF_ARMS == b.QUALITY_ARMS["K0"] and b.QUALITY_ARMS["K1"] == ("ON",)
    assert f"GNF4_SHA={r.GNF4_SHA}" in RUN and r.GNF4_SHA == "56f90e3555b089cdc8fb508e42ea0d2c069ce8e5"
    for model, rev in r.REVS.items():
        assert rev in RUN, model
    assert "WINDOWS_DEF=48" in RUN and "WINDOWS_DEF=16" in RUN and '"--group", type=int, default=16' in BOX
    for words in ("LICENSED", "QUALITY_FAIL", "SLOWER", "NOISY", "FUNCTION_FAIL", "`56f90e3`", "16 windows a pass"):
        assert words in PREREG, words


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "python - <<'PYT'", "p121_reduce.py --self-test", "p121_box.py --self-test", "p115_quality.py --self-test",
             "python -m pytest " + " ".join(PREMISE), 'echo "premise ok"', 'say "fetch $MODEL @ $REV"',
             "python $W/k8_bake.py", "p121_box.py --prompts-only", "for TAG in K0a K1a K1b K0b; do",
             'if [ "$ARMS_OK" = 1 ]; then', "for PH in off on; do", "--out $W/quality_$PH.json",
             "--out $W/continuity_$PH.json --ref-dir $W/work/ref_c --texts wikitext --windows 12 --group 1",
             "python $W/p121_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'echo "$LASTL" | grep -q "11 passed"' in RUN and "11 passed" in PREREG
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'hr._k25_mode_env() == "auto"' in trip and 'R_rows <= 256 and _k25_mode != "0"' in trip
    assert 'k25_t1 = T == 1 and int4_stores is None and _k25_mode_env() == "1"' in trip
    assert 'hasattr(nf4_smallm, "gemm_nf4_grouped_smallm")' in trip and 'hasattr(ng, "gemm_4bit_grouped_captured")' in trip
    assert 'serve_paged._fusion_env(k, "") == "0" for k in serve_paged.FUSION_KNOBS' in trip
    assert 'md.version("grouped-nf4-gemm") == "0.43.0"' in trip and 'transformers.__version__ == "5.17.0"' in trip


def test_the_premise_files_collect_11_cases():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          *[str(REPO / "tests" / f) for f in PREMISE]], capture_output=True, text=True, cwd=REPO)
    assert re.search(r"\b11 tests? collected\b", out.stdout), out.stdout[-600:] + out.stderr[-600:]


def test_the_subject_is_the_default_graph_server_and_only_the_knob_differs():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING", "E4B_INT4_LEAN_GLUE", "E4B_PAGED_GRAPHS",
                 "E4B_PAGED_MAX_SEQS", "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "GNF4_GEMV_BW", "TRITON_INTERPRET"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert ('ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena '
            'E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"') in RUN
    assert 'KN0="E4B_NF4_GROUPED_SMALLM=0"' in RUN and 'KN1="E4B_NF4_GROUPED_SMALLM=auto"' in RUN
    assert "E4B_PAGED_GRAPHS=0 P121_ARM=$ARM" in RUN and "--mode quality" in RUN
    assert "(cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, \"all-vram\", (1, 2, 4, 8, 16))" in BOX
    assert BOX.index("rc = RouteCounter().install()") < BOX.index("parts = build_engine(cfg)\n    load_s")
    assert BOX.index("rc = RouteCounter().install()\n    t0 = time.time()") < BOX.index("parts = build_engine(cfg)\n    model")
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; WINDOWS_DEF=48; CONT_DEF=128" in RUN
    assert "SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; WINDOWS_DEF=16; CONT_DEF=32" in RUN


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 0.75 h" in PREREG and "guard 2.0 h" in PREREG
    keys = {"FETCH", "BAKE", "ARM", "QOFF", "QON", "CONT"}
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|QOFF|QON|CONT)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|QOFF|QON|CONT)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == keys
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 2.0 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM", "$NEED", "$NEED_CONT"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 15, 16, 21, 22, 25, 27, 40} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p121-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P121_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p121_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p121"), out.stdout + out.stderr
