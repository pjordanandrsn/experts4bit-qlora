"""The p133 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P133 from test_p116_staged_pin.py).

`bench/p133/p133_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p133/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- P116's borrowed pieces (P115's reducer and quality box, P109's, P110's, P108's and P97's boxes, P39's bake and
  calibration, the premise tests) at P116's registered bytes;
- the pins: e4b E4B_PIN, grouped-nf4-gemm 0.46.0 exact, torch 2.8.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2;
- the declared run ids, and the order: refusals, install and tripwire, the self-tests, the premise (exact pass counts and
  skipped ids), the fetch, the bake, the prompts, the fit check, the four speed arms, the two quality phases and the
  reducer;
- the subject: the default server, with only ``GNF4_GEMV_BW`` differing between arms, the shipped plan table;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p133"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p133_run.sh": LANE / "p133_run.sh",
    "p133_box.py": LANE / "p133_box.py",
    "p133_reduce.py": LANE / "p133_reduce.py",
    "p133_tiny.py": LANE / "p133_tiny.py",
    "p115_reduce.py": REPO / "bench" / "p115" / "p115_reduce.py",
    "p115_quality.py": REPO / "bench" / "p115" / "p115_quality.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "p110_box.py": REPO / "bench" / "p110" / "p110_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_kv_step_select.py": REPO / "tests" / "test_kv_step_select.py",
    "test_gemv_bw_served_gpu.py": REPO / "tests" / "test_gemv_bw_served_gpu.py",
}
RUN = (LANE / "p133_run.sh").read_text()
DRIVE = (LANE / "p133_drive.sh").read_text()
REDUCE = (LANE / "p133_reduce.py").read_text()
BOX = (LANE / "p133_box.py").read_text()
PREREG = (LANE / "PREREG-p133.md").read_text(encoding="utf-8")
E4B_PIN = "c8d7e629e4c8431fc86970cf7195d8e67f72e22a"
GNF4_PIN = "6b7c2e7474d20c8843814dd46dad6cfc3d94b440"


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _env():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}   # Windows needs it
    sep = ";" if sys.platform.startswith("win") else ":"
    return {"PYTHONPATH": sep.join(str(REPO / "bench" / d) for d in ("p109", "p115")), **base}


def _drive(tmp_path, run_id, **extra):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "P133_DRIVE_DRYRUN": "1", **extra}
    return subprocess.run(["bash", str(LANE / "p133_drive.sh")], capture_output=True, text=True, env=env)


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p133 launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_p116s_bytes():
    mine = dict((n, w) for w, n in _entries())
    p116 = dict((n, w) for w, n in _entries(REPO / "bench" / "p116" / "staged.sha256"))
    for name in SOURCES:
        if not name.startswith("p133_"):
            assert mine[name] == p116[name], f"{name} is not at P116's bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p133_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in DRIVE and "--exclude 'gnf4src'" in DRIVE


def test_the_self_tests_pass():
    out = subprocess.run([sys.executable, str(LANE / "p133_reduce.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p133_reduce self-test OK (41 cases)" in out.stdout, out.stdout + out.stderr
    out = subprocess.run([sys.executable, str(LANE / "p133_box.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p133_box self-test OK (27/27 cases)" in out.stdout, out.stdout + out.stderr


def test_the_pins_are_exact_and_agree():
    assert f"E4B_PIN={E4B_PIN}" in RUN and f"E4B_PIN={E4B_PIN}" in DRIVE and E4B_PIN in PREREG
    assert f"GNF4_SHA={GNF4_PIN}" in RUN and f'GNF4_SHA = "{GNF4_PIN}"' in REDUCE and GNF4_PIN in PREREG
    assert '[ "$E4B_SHA" = "$E4B_PIN" ] ||' in RUN and "TORCH_WANT=2.8.0+cu128" in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    for pin in ('md.version("grouped-nf4-gemm") == "0.46.0"', 'md.version("bitsandbytes") == "0.50.2"',
                'transformers.__version__ == "5.17.0"', 'torch.__version__ == os.environ["WANT_TORCH"]',
                'ng._BW_SHAPES == frozenset({(1536, 2048), (2048, 768)}) and ng._bw() == "auto"',
                "assert (cc, sm) == ((8, 9), 128), (cc, sm)", "ng._bw_engages(1536, 2048, dev) is (sm >= 160)"):
        assert pin in trip, pin
    for model, rev in re.findall(r'(QWEN|GRAN): "([0-9a-f]{40})"', REDUCE):
        assert rev in RUN, model


def test_the_declared_run_ids_and_nothing_else(tmp_path):
    ok = [("p133-4090-prove-1", {"P133_PROVE": "1"}), ("p133-4090-1", {}), ("p133-4090-2", {}),
          ("p133-dry-1", {"P133_DRY": "1", "P133_BOX_EXEC": "/bin/false"})]
    for run_id, extra in ok:
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> "), (run_id, out.stdout + out.stderr)
        assert f"E4B_SHA={E4B_PIN}" in out.stdout, out.stdout
    refused = [("p133-4090-3", {}), ("p133-4090-prove-2", {"P133_PROVE": "1"}), ("p133-4090-prove-1", {}),
               ("p133-4090-1", {"P133_PROVE": "1"}), ("p133-4090-1", {"P133_DRY": "1", "P133_BOX_EXEC": "/bin/false"}),
               ("p133-dry-1", {}), ("p116-5090-1", {}), ("p133-4090-1", {"E4B_SHA": "0" * 40})]
    for run_id, extra in refused:
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 78 and "refusing" in out.stdout, (run_id, extra, out.stdout + out.stderr)
    assert "p133-4090-prove-1" in PREREG and "p133-4090-1" in PREREG and "p133-4090-2" in PREREG


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("B0a", "B1a", "B1b", "B0b")' in REDUCE
    assert "SELF_LO, SELF_HI = 0.96, 1.04" in REDUCE and "GAIN_MIN_W1, GAIN_MIN_WCTL = 1.03, 0.99" in REDUCE
    assert "SELF_W1_LO, SELF_W1_HI = 0.985, 1.015" in REDUCE
    assert 'FLOORS = ("chunk",)' in REDUCE and 'OFF_ARMS = ("R", "rep", "chunk", "mutant_scale")' in REDUCE
    assert "TOL, SPREAD_X, SPREAD_MIN, K8_BUDGET, K8_GATED = pb.TOL, pb.SPREAD_X, pb.SPREAD_MIN, pb.K8_BUDGET, pb.K8_GATED" in REDUCE
    p110 = (REPO / "bench" / "p110" / "p110_box.py").read_text()
    assert "INSTRUMENT_BUCKETS = _instrument_buckets()" in REDUCE, "Amendment 1: read from p110_box.py, never typed"
    import ast
    p110_buckets = next(ast.literal_eval(n.value) for n in ast.parse(p110).body if isinstance(n, ast.Assign)
                        and any(getattr(x, "id", None) == "BUCKETS" for x in n.targets))
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p115")]
    try:
        import p133_reduce as r
    finally:
        del sys.path[:2]
    assert r.INSTRUMENT_BUCKETS == tuple(p110_buckets) == (1, 2, 4, 8, 16), (r.INSTRUMENT_BUCKETS, p110_buckets)
    assert "for b in INSTRUMENT_BUCKETS" in REDUCE and "Amendment 1" in PREREG
    assert 'INCUMBENT = {QWEN: "scalar", GRAN: "scalar", TINY: "scalar"}' in REDUCE
    assert "WINDOWS = {QWEN: 24, GRAN: 12, TINY: 4}" in REDUCE
    for mode, w in (("DRY", 4), ("PROVE", 12), ("reading", 24)):
        assert f"WINDOWS_DEF={w};" in RUN, mode
    assert "1.03" in PREREG and "0.99" in PREREG and "[0.985, 1.015]" in PREREG and "[0.96, 1.04]" in PREREG


def test_the_premise_is_exact_pass_counts_and_skipped_ids():
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p109")]
    try:
        import p133_box as b
    finally:
        del sys.path[:2]
    assert b.PREMISE["kernel"][0] == 33 and set(b.PREMISE["kernel"][1].values()) == {b.PDL}
    assert b.PREMISE["served"] == (18, {}) and b.PREMISE["served_dry"][0] == 13
    assert len(b.PREMISE["served_dry"][1]) == 5 and set(b.PREMISE["served_dry"][1].values()) == {b.FP8}
    for f in ("test_decode_graph_buckets.py", "test_kv_step_select.py"):
        assert f'reason="{b.FP8}")' in (REPO / "tests" / f).read_text(), f
    for tid in b.PREMISE["kernel"][1]:
        assert tid in PREREG, tid
    assert "--junitxml=$W/logs/premise_kernel.xml" in RUN and "--expect kernel" in RUN and "--expect served" in RUN


def test_the_order_puts_every_refusal_before_the_fetch_and_the_fit_before_the_arms():
    order = ["CUDA_PROBE=$(python -", "REFUSED: the image's torch", "REFUSED: card is", "the licence covers sm_89",
             "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4src", "python - <<'PYT'",
             "p133_reduce.py --self-test", "p133_box.py --self-test", "p115_quality.py --self-test",
             "python -m pytest test_nf4_gemv_bw.py", "--expect kernel", "--expect served", 'echo "premise ok"',
             "python $W/p133_tiny.py", 'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "p133_box.py --prompts-only",
             "p133_box.py --mode fit", "finish 31", "for TAG in B0a B1a B1b B0b; do", 'if [ "$ARMS_OK" = 1 ]; then',
             "for PH in off on; do", 'say "reduce"; python $W/p133_reduce.py --dir $W --out $W/verdict.json']
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert "STOP-5" in RUN[RUN.index('if [ "$ARMS_OK" = 1 ]; then'):]


def test_the_subject_is_the_default_server_and_only_the_switch_differs():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_KV_STEP_SELECT", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE",
                 "GNF4_PDL", "GNF4_GEMV_DOTPAD", "GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "TRITON_INTERPRET"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert 'ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"' in RUN
    assert 'BW="GNF4_GEMV_BW=1"\n' in RUN and "GNF4_GEMV_BW_PLAN=" not in RUN[RUN.index('BW="GNF4_GEMV_BW=1"'):]
    assert 'ARM=${TAG:0:2}; KN=""; [ "$ARM" = B1 ] && KN=$BW' in RUN
    assert 'NEED=$NEED_QOFF; ARM=B0; KN=""; [ "$PH" = on ] && { NEED=$NEED_QON; ARM=B1; KN=$BW; }' in RUN
    assert "E4B_PAGED_GRAPHS=0 P133_ARM=$ARM" in RUN and "--windows $WINDOWS --cont $CONT --group 1" in RUN
    assert 'if not dry() and (not cfg.graphs or cfg.placement != "all-vram"):' in BOX
    assert 'if cfg.graphs or cfg.placement != "all-vram":' in BOX and "if a.group != 1:" in BOX
    assert BOX.index("q.KernelCounters().install()") < BOX.index("parts = build_engine(cfg)\n    model = parts.runner.model")
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; WINDOWS_DEF=24; CONT_DEF=128" in RUN
    assert "SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; WINDOWS_DEF=12; CONT_DEF=32" in RUN


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 0.75 h" in PREREG and "guard 1.25 h" in PREREG
    keys = {"FETCH", "BAKE", "FIT", "ARM", "QOFF", "QON"}
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|FIT|ARM|QOFF|QON)=(\d+)",
                            RUN[RUN.index('elif [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|FIT|ARM|QOFF|QON)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == keys
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 1.25 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_FIT", "$NEED_ARM", "$NEED"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    # 13 the disk floor and 18 the CUDA host floor; 14 and 17 are the launcher's. 18 appears once.
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 15, 16, 21, 22, 25, 27, 28, 31, 40} <= codes


PRE_AMENDMENT = "7896ce0e9f70b9023e8a480aa9c50699c826975e"   # #1695's merge, the reducer p133-4090-prove-1 ran


def test_amendment_1_the_old_reducer_voids_the_proofs_shape_and_the_new_one_passes_it(tmp_path):
    """p133-4090-prove-1's shape: server buckets 1-64 (max_seqs 64), instrument passes at 1-16. The pre-amendment reducer
    VOIDs it on the pass graph_status alone; the amended one passes it; a pass that genuinely misses bucket 8 still VOIDs."""
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_AMENDMENT}:bench/p133/p133_reduce.py"],
                         capture_output=True, text=True)
    assert src.returncode == 0, src.stderr                       # CI checks out full history (fetch-depth: 0)
    old = tmp_path / "old"
    old.mkdir()
    (old / "p133_reduce.py").write_text(src.stdout)
    shutil.copy(REPO / "bench" / "p115" / "p115_reduce.py", old)
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p115")]
    try:
        import p133_reduce as r
    finally:
        del sys.path[:2]
    rec = tmp_path / "rec"
    rec.mkdir()
    bk = [1, 2, 4, 8, 16, 32, 64]
    toks = [[i, i + 1] for i in range(16)]
    for t, (rc, r1) in {"B0a": (800.0, 110.0), "B1a": (801.0, 125.0), "B1b": (800.5, 124.8), "B0b": (799.0, 110.2)}.items():
        (rec / f"arm_{t}.json").write_text(json.dumps(r._fake_arm(t, rc, r1, toks, buckets=bk)))
    (rec / "quality_off.json").write_text(json.dumps(r._fake_quality("off", buckets=bk)))
    (rec / "quality_on.json").write_text(json.dumps(r._fake_quality("on", buckets=bk, bias={"wikitext": 0.001, "c4val1": 0.001})))
    (rec / "fit.json").write_text(json.dumps(r.FIT))

    def run(script):
        out = tmp_path / "v.json"
        p = subprocess.run([sys.executable, str(script), "--dir", str(rec), "--out", str(out), "--e4b-sha", r.E, *[]],
                           capture_output=True, text=True, env=_env())
        assert p.returncode == 0, p.stdout + p.stderr
        return json.loads(out.read_text())
    v_old = run(old / "p133_reduce.py")
    assert v_old["verdict"] == "VOID" and v_old["reasons"], v_old
    assert all(": graph_status {" in x for x in v_old["reasons"]), v_old["reasons"][:3]
    v_new = run(LANE / "p133_reduce.py")
    assert v_new["verdict"] == "LICENSED", v_new["reasons"]
    q = json.loads((rec / "quality_off.json").read_text())
    del q["engagement"][r.TEXTS[0]]["R"][0]["graph_status"]["8"]
    (rec / "quality_off.json").write_text(json.dumps(q))
    v_mut = run(LANE / "p133_reduce.py")
    assert v_mut["verdict"] == "VOID" and any(": graph_status {" in x for x in v_mut["reasons"]), v_mut["reasons"]
