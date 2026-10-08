"""The p115 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P115; e4b#1313).

`bench/p115/p115_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p115/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's, the quality box's and the reducer's self-tests and pins the
lane's shape:
- P109's box (whose prompts, passes and slope P115's box imports), P110's, P108's and P97's boxes (whose paged pass,
  scoring and wikitext windows the quality box imports) at their registered bytes, and P39's bake and calibration at
  SC1's;
- the order: refusals, install and tripwire, the self-tests, the premise on the card, then the fetch, the bake, the
  prompts, the four speed arms, the two quality phases (skipped when a speed arm failed) and the reducer;
- the subject: the default graph server, with only the four fusion knobs differing between arms;
- the rule's constants, its count tables, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p115"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p115_run.sh": LANE / "p115_run.sh",
    "p115_box.py": LANE / "p115_box.py",
    "p115_reduce.py": LANE / "p115_reduce.py",
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
}
RUN = (LANE / "p115_run.sh").read_text()
REDUCE = (LANE / "p115_reduce.py").read_text()
BOX = (LANE / "p115_box.py").read_text()
QUAL = (LANE / "p115_quality.py").read_text()
PREREG = (LANE / "PREREG-p115.md").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p115 launch refuses ON A RENTED BOX"


def test_imported_and_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p109 = dict((n, w) for w, n in _entries(REPO / "bench" / "p109" / "staged.sha256"))
    p110 = dict((n, w) for w, n in _entries(REPO / "bench" / "p110" / "staged.sha256"))
    p111 = dict((n, w) for w, n in _entries(REPO / "bench" / "p111" / "staged.sha256"))
    assert mine["p109_box.py"] == p109["p109_box.py"]
    for name in ("p110_box.py", "p108_box.py", "p97_box.py"):
        assert mine[name] == p110[name], name
    assert mine["k8_bake.py"] == p109["k8_bake.py"] and mine["calib.json"] == p109["calib.json"]
    assert mine["test_decode_graph_buckets.py"] == p109["test_decode_graph_buckets.py"]
    assert mine["test_kv_step_select.py"] == p111["test_kv_step_select.py"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p115_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p115_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver


def test_the_self_tests_pass():
    out = subprocess.run([sys.executable, str(LANE / "p115_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (27 cases)" in out.stdout, out.stdout + out.stderr
    sep = ";" if sys.platform.startswith("win") else ":"
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}   # Windows needs it
    env = {"PYTHONPATH": str(REPO / "bench" / "p109"), **base}
    out = subprocess.run([sys.executable, str(LANE / "p115_box.py"), "--self-test"], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "p115_box self-test OK (12/12 cases)" in out.stdout, out.stdout + out.stderr
    env = {"PYTHONPATH": sep.join(str(REPO / "bench" / d) for d in ("p110", "p108", "p97")), **base}
    out = subprocess.run([sys.executable, str(LANE / "p115_quality.py"), "--self-test"], capture_output=True, text=True,
                         env=env)
    assert out.returncode == 0 and "p115_quality self-test OK (11/11 cases)" in out.stdout, out.stdout + out.stderr


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("F0a", "F1a", "F1b", "F0b")' in REDUCE
    assert "SELF_LO, SELF_HI = 0.96, 1.04" in REDUCE and "GAIN_MIN_W1, GAIN_MIN_W16 = 1.10, 1.00" in REDUCE
    assert "TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005" in REDUCE and "K8_BUDGET = 0.05" in REDUCE
    assert 'K8_GATED = ("wikitext",)' in REDUCE and 'FLOORS = ("half", "chunk")' in REDUCE
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == "b4f93f1c62d1e3436ed45bec8ccd608c90433737"
    # Registered against e4b CI's grouped-nf4-gemm pin of that day (v0.42.0). CI's pin moves with each release (0.49.0
    # pins v0.43.0); the lane's box installs its registered GNF4_SHA itself, so CI's pin is not part of the rule.
    revs = dict(re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE))
    for model, rev in revs.items():
        assert f"MODEL={model}; REV={rev}" in RUN, model
    from experts4bit_qlora import k8_gate
    assert k8_gate.BUDGET == 0.05, "the wikitext K8 bar is the package's registered budget"


def test_the_count_tables_are_the_registered_tables():
    sys.path.insert(0, str(LANE))
    try:
        import p115_reduce as r
    finally:
        sys.path.remove(str(LANE))
    assert r.census_for("qwen3", 48, True) == {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48],
                                               "fuse_router_epilogue_n": 48}
    assert r.census_for("granite", 32, False) == {"fuse_qkv_n": 0, "fuse_t1_glue_n": 65, "fuse_t1_glue_r2_n": [32, 32],
                                                  "fuse_router_epilogue_n": 32}
    assert r.per_step("qwen3", 48) == {"rmsnorm_rows": 49, "rmsnorm_resid_rows": 48, "rope_norm_heads": 96,
                                       "router_epilogue": 48}
    for text in ("48 / 193 / [48, 48] / 48", "49 / 48 / 96 / 48"):
        assert text in PREREG, text


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p115_reduce.py --self-test", "p115_box.py --self-test", "p115_quality.py --self-test",
             "python -m pytest test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py",
             'echo "premise ok"', 'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "p115_box.py --prompts-only",
             "for TAG in F0a F1a F1b F0b; do", 'if [ "$ARMS_OK" = 1 ]; then', "for PH in off on; do",
             "python $W/p115_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "14 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'serve_paged._graphs_env("", "cuda", "all-vram") is True' in trip
    assert "serve_paged.PagedServeConfig().fuse_qkv is False" in trip and 'md.version("grouped-nf4-gemm") == "0.42.0"' in trip
    assert "STOP-5" in RUN[RUN.index('if [ "$ARMS_OK" = 1 ]; then'):]


def test_the_subject_is_the_default_graph_server_and_only_the_knobs_differ():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(': > summary.txt')]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_KV_STEP_SELECT", "E4B_PAGED_MAX_SEQS", "E4B_NF4_GROUPED_SMALLM",
                 "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI", "GNF4_PDL"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert 'ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"' in RUN
    assert 'FUSE="E4B_PAGED_FUSE_QKV=1 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"' in RUN
    assert '[ "$PROVE" = 1 ] && FUSE="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"' in RUN
    assert 'ARM=${TAG:0:2}; KN=""; [ "$ARM" = F1 ] && KN=$FUSE' in RUN
    assert 'KN=""; [ "$PH" = on ] && KN=$FUSE' in RUN and "E4B_PAGED_GRAPHS=0 P115_PROVE=$PROVE" in RUN
    assert "if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, \"all-vram\", (1, 2, 4, 8, 16)):" in BOX
    assert "if cfg.graphs or cfg.placement != \"all-vram\":" in QUAL
    assert "counters = KernelCounters().install()            # BEFORE build_engine" in QUAL
    assert QUAL.index("KernelCounters().install()") < QUAL.index("parts = build_engine(cfg)")
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; WINDOWS_DEF=48; CONT_DEF=128" in RUN
    assert "SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; WINDOWS_DEF=12; CONT_DEF=32" in RUN


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 0.75 h" in PREREG and "guard 1.5 h" in PREREG
    keys = {"FETCH", "BAKE", "ARM", "QOFF", "QON"}
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|QOFF|QON)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|QOFF|QON)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == keys
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 1.5 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM", "$NEED"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    # 13 for the disk floor and 18 for the CUDA host floor (a GPU the image's torch cannot use, TC1 amendment 61's class), and
    # nothing else that excludes a machine: 14 and 17 are the launcher's. 18 appears once, in the no-cuda branch, so no other
    # lane failure can name the machine (p115-5090-1 read HARNESS_ERROR on a CUDA-unusable host at rc 10).
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {16, 25, 27} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p115-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P115_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p115_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p115"), out.stdout + out.stderr
