"""P115 Phase D's staged-file pin must match the repo (the e4b#642 check, mirrored for Amendment 3; e4b#1313).

`bench/p115/p115d_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p115/staged-d.sha256`; this
test runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the phase's shape:
- P115's quality box, P109's, P110's, P108's and P97's boxes, P39's bake and calibration and the premise tests at the
  bytes Phase C and P116 pinned;
- the order: refusals, install and tripwire, the self-tests, the premise on the card (19), then the fetch, the bake,
  the prompts, the four speed arms, the two SANE phases (skipped when a speed arm failed) and the reducer;
- the subject: the default graph server at 16 slots with grouped-nf4-gemm 0.43.0's bandwidth GEMV at its default, only
  the four fusion knobs differing (0 / auto); SANE at one window per pass (T == 1);
- the rule's constants against the runner and the PREREG, every time-left check inside its own guard, the exit codes.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p115"
PIN = LANE / "staged-d.sha256"
SOURCES = {
    "p115d_run.sh": LANE / "p115d_run.sh",
    "p115d_box.py": LANE / "p115d_box.py",
    "p115d_reduce.py": LANE / "p115d_reduce.py",
    "p115_quality.py": LANE / "p115_quality.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "p110_box.py": REPO / "bench" / "p110" / "p110_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_kv_step_select.py": REPO / "tests" / "test_kv_step_select.py",
    "test_fused_glue_decode_graphs_gpu.py": REPO / "tests" / "test_fused_glue_decode_graphs_gpu.py",
    "test_gemv_bw_served_gpu.py": REPO / "tests" / "test_gemv_bw_served_gpu.py",
}
PREMISE = ("test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py",
           "test_gemv_bw_served_gpu.py")
RUN = (LANE / "p115d_run.sh").read_text()
DRIVE = (LANE / "p115d_drive.sh").read_text()
BOX = (LANE / "p115d_box.py").read_text()
PREREG = (LANE / "PREREG-p115.md").read_text(encoding="utf-8")


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _env():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}   # Windows needs it
    sep = ";" if sys.platform.startswith("win") else ":"
    return {"PYTHONPATH": sep.join(str(REPO / "bench" / d) for d in ("p109", "p110", "p108", "p97")), **base}


def _import(name, *dirs):
    sys.path[:0] = [str(d) for d in dirs]
    try:
        return __import__(name)
    finally:
        del sys.path[:len(dirs)]


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next Phase D launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    phase_c = dict((n, w) for w, n in _entries(LANE / "staged-c.sha256"))
    for name in ("p115_quality.py", "p109_box.py", "p110_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json",
                 "test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py"):
        assert mine[name] == phase_c[name], f"{name} is not at Phase C's bytes"
    p116 = dict((n, w) for w, n in _entries(REPO / "bench" / "p116" / "staged.sha256"))
    assert mine["test_gemv_bw_served_gpu.py"] == p116["test_gemv_bw_served_gpu.py"], "not at P116's bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged-d.sha256"]:
        if name != "p115d_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in DRIVE and 'done < "$HERE/staged-d.sha256"' in DRIVE


def test_the_self_tests_pass():
    out = subprocess.run([sys.executable, str(LANE / "p115d_reduce.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p115d_reduce self-test OK (23 cases)" in out.stdout, out.stdout + out.stderr
    out = subprocess.run([sys.executable, str(LANE / "p115d_box.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p115d_box self-test OK (12/12 cases)" in out.stdout, out.stdout + out.stderr


def test_the_rule_is_the_registered_rule():
    r = _import("p115d_reduce", LANE)
    assert r.TAGS == ("D0a", "D1a", "D1b", "D0b")
    assert (r.SANE_WINDOWS, r.SANE_BIAS, r.SANE_ARGMAX) == (12, 0.02, 0.95), "Phase C's SANE gate, unchanged"
    pc = _import("p115c_reduce", LANE)
    assert (pc.SANE_WINDOWS, pc.SANE_BIAS, pc.SANE_ARGMAX) == (r.SANE_WINDOWS, r.SANE_BIAS, r.SANE_ARGMAX)
    assert r.PREDICTED[r.QWEN] == {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48],
                                   "fuse_router_epilogue_n": 48}, "Phase A/B's Qwen3 census"
    assert r.PREDICTED[r.GRAN] == pc.PREDICTED["granite"], "Phase C's Granite census"
    assert f"GNF4_SHA={r.GNF4_SHA}" in RUN and r.GNF4_SHA == "6ee2e10408161a9d3c874975c9191a7f2957e6f4"
    for model, rev in r.REVS.items():
        assert rev in RUN, model
    assert "WINDOWS=12" in RUN and "group=1," in BOX
    assert "COMBINED_SANE" in PREREG and "COMBINED_FAIL" in PREREG and "**`6ee2e10`**" in PREREG


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "python - <<'PYT'", "p115d_reduce.py --self-test", "p115d_box.py --self-test", "p115_quality.py --self-test",
             "python -m pytest " + " ".join(PREMISE), 'echo "premise ok"', 'say "fetch $MODEL @ $REV"',
             "python $W/k8_bake.py", "p115d_box.py --prompts-only", "for TAG in D0a D1a D1b D0b; do",
             'if [ "$ARMS_OK" = 1 ]; then', "for PH in off on; do", "python $W/p115d_reduce.py --dir $W --out $W/verdict_d.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'echo "$LASTL" | grep -q "19 passed"' in RUN and "19 passed" in PREREG
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'ng._bw() == "auto" and {(1536, 2048), (2048, 768)} <= set(ng._BW_SHAPES)' in trip
    assert 'serve_paged._fusion_env(k, "") == "0" for k in serve_paged.FUSION_KNOBS' in trip
    assert "hr._collapsed_grouping(1, None) == (True, False) and hr._collapsed_grouping(4, None) == (False, True)" in trip
    assert 'md.version("grouped-nf4-gemm") == "0.43.0"' in trip and 'transformers.__version__ == "5.17.0"' in trip


def test_the_premise_files_collect_19_cases():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          *[str(REPO / "tests" / f) for f in PREMISE]], capture_output=True, text=True, cwd=REPO)
    if "nf4_grouped" not in sys.modules:
        try:
            import nf4_grouped  # noqa: F401
        except ImportError:                   # test_gemv_bw_served_gpu skips at collection without grouped-nf4-gemm
            return
    assert re.search(r"\b19 tests? collected\b", out.stdout), out.stdout[-600:] + out.stderr[-600:]


def test_the_subject_is_the_default_graph_server_and_only_the_knobs_differ():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2",
                 "E4B_FUSE_ROUTER_EPI", "E4B_PAGED_DECODE_LOOKAHEAD", "GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_PDL",
                 "TRITON_INTERPRET"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert ('ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena '
            'E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"') in RUN
    assert 'KN0="E4B_PAGED_FUSE_QKV=0 E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0"' in RUN
    assert 'KN1="E4B_PAGED_FUSE_QKV=auto E4B_FUSE_T1_GLUE=auto E4B_FUSE_T1_GLUE_R2=auto E4B_FUSE_ROUTER_EPI=auto"' in RUN
    assert "E4B_PAGED_GRAPHS=0 P115D_ARM=$ARM" in RUN
    assert "(cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, \"all-vram\", (1, 2, 4, 8, 16))" in BOX
    assert BOX.index("q.KernelCounters().install()") < BOX.index("parts = build_engine(cfg)\n    model = parts.runner.model")
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; CONT_DEF=128" in RUN and "SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; CONT_DEF=32" in RUN


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 0.75 h" in PREREG and "guard 1.5 h" in PREREG
    keys = {"FETCH", "BAKE", "ARM", "SANE"}
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|SANE)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|SANE)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nWINDOWS=")]))
    assert set(prove) == set(reading) == keys
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 1.5 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM", "$NEED_SANE"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 15, 16, 21, 22, 25, 27, 40} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p115d-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P115D_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p115d_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p115d"), out.stdout + out.stderr
