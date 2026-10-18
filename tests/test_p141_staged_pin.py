"""The p141 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P141 from test_p134_staged_pin.py).

`bench/p141/p141_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p141/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's, the reducer's and phase 1's self-tests and pins the lane's shape:
- P133's dry fixture and P116's borrowed pieces at P133's (and P134's) registered bytes;
- the pins (e4b = the launch commit, grouped-nf4-gemm 0.46.1), the family and its checkpoint, the declared run ids;
- the order: refusals, install and tripwire, the self-tests, the premise (the 5090's counts), phase 1, then the fetch,
  bake, prompts, fit check, four speed arms, two quality phases and the reducer;
- P136's server size, B1's plan string from phase 1, the rule's constants, every time-left check inside its guard, and
  the exit codes.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p141"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p141_run.sh": LANE / "p141_run.sh",
    "p141_box.py": LANE / "p141_box.py",
    "p141_reduce.py": LANE / "p141_reduce.py",
    "p141_select.py": LANE / "p141_select.py",
    "p133_tiny.py": REPO / "bench" / "p133" / "p133_tiny.py",
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
RUN = (LANE / "p141_run.sh").read_text()
DRIVE = (LANE / "p141_drive.sh").read_text()
REDUCE = (LANE / "p141_reduce.py").read_text()
BOX = (LANE / "p141_box.py").read_text()
SELECT = (LANE / "p141_select.py").read_text()
PREREG = (LANE / "PREREG-p141.md").read_text(encoding="utf-8")
GNF4_PIN = "70bc03cb7a61b03c4e801e7423c58c103a1b6d7d"
MIX_REV = "eba92302a2861cdc0098cc54bc9f17cb2c47eb61"
K33_PLANS = [(16, 256, 4, 1), (16, 512, 4, 1), (16, 1024, 4, 1), (16, 256, 8, 1), (16, 512, 8, 1),
             (32, 256, 4, 1), (32, 512, 4, 1), (32, 512, 8, 1), (32, 1024, 8, 1), (16, 256, 4, 2), (16, 512, 4, 2),
             (32, 256, 4, 2)]                                   # grouped-nf4-gemm kernel/k33_bench.py PLANS


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _env():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}   # Windows needs it
    sep = ";" if sys.platform.startswith("win") else ":"
    return {"PYTHONPATH": sep.join(str(REPO / "bench" / d) for d in ("p109", "p115", "p110")), **base}


def _drive(tmp_path, run_id, **extra):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "P141_DRIVE_DRYRUN": "1", "E4B_SHA": "a" * 40, **extra}
    return subprocess.run(["bash", str(LANE / "p141_drive.sh")], capture_output=True, text=True, env=env)


def _load(name, *dirs):
    sys.path[:0] = [str(LANE)] + [str(REPO / "bench" / d) for d in dirs]
    try:
        import importlib
        return importlib.import_module(name)
    finally:
        del sys.path[:1 + len(dirs)]


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p141 launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_p133s_and_p134s_bytes():
    mine = dict((n, w) for w, n in _entries())
    for lane in ("p133", "p134"):
        other = dict((n, w) for w, n in _entries(REPO / "bench" / lane / "staged.sha256"))
        for name in SOURCES:
            if not name.startswith("p141_"):
                assert mine[name] == other[name], f"{name} is not at {lane}'s bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p141_run.sh":
            assert name in staged, name
    assert "--include 'work/' --include 'work/*/' --include 'work/*/bake.json' --exclude 'work/*/*'" in DRIVE
    assert "--exclude 'gnf4src' --exclude 'fam'" in DRIVE


def test_the_self_tests_pass():
    for script, line in (("p141_reduce.py", "p141_reduce self-test OK (41 cases)"),
                         ("p141_box.py", "p141_box self-test OK (33/33 cases)"),
                         ("p141_select.py", "p141_select self-test OK (17/17 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True,
                             env=_env())
        assert out.returncode == 0 and line in out.stdout, out.stdout + out.stderr


def test_the_pins_are_exact_and_agree():
    assert f"GNF4_SHA={GNF4_PIN}" in RUN and f'GNF4_SHA = "{GNF4_PIN}"' in REDUCE and GNF4_PIN in PREREG
    assert "E4B_PIN" not in RUN and "E4B_PIN" not in DRIVE, "e4b is the launch commit, never a literal"
    assert '[[ $E4B_SHA =~ ^[0-9a-f]{40}$ ]]' in RUN and "TORCH_WANT=2.8.0+cu128" in RUN
    assert 'E4B_SHA=$(git -C "$REPO" rev-parse HEAD' in DRIVE and "status --porcelain" in DRIVE
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    for pin in ('md.version("grouped-nf4-gemm") == "0.46.1"', 'md.version("bitsandbytes") == "0.50.2"',
                'transformers.__version__ == "5.17.0"', 'torch.__version__ == os.environ["WANT_TORCH"]',
                'ng._BW_SHAPES == frozenset({(1536, 2048), (2048, 768)}) and ng._bw() == "auto"',
                "assert cc == (12, 0) and sm >= 160, (cc, sm)",
                "FAMILY_SHAPES = ((28672, 4096), (4096, 14336), (512, 512), (512, 256))",
                'assert not any(ng._bw_engages(n, k, dev) for n, k in FAMILY_SHAPES)',
                'assert not any(s in ng._DOTPAD_CONFIGS for s in FAMILY_SHAPES)',
                'assert not any(s in ng._BW_PLANS for s in FAMILY_SHAPES)',
                'ng._bw_plan(28672, 4096) == (16, 1024, 4, 1) and ng._bw_plan(4096, 14336) == (16, 256, 4, 2)',
                '"gather_div" in inspect.signature(ng.gemm_4bit_grouped).parameters'):
        assert pin in trip, pin
    assert f"MIX=mistralai/Mixtral-8x7B-Instruct-v0.1; MIX_REV={MIX_REV}" in RUN
    assert f'REVS = {{MIXTRAL: "{MIX_REV}", TINY: ""}}' in REDUCE and MIX_REV in PREREG


def test_the_declared_run_ids_and_nothing_else(tmp_path):
    ok = [("p141-5090-1", {}), ("p141-5090-2", {}), ("p141-dry-1", {"P141_DRY": "1", "P141_BOX_EXEC": "/bin/false"})]
    for run_id, extra in ok:
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> "), (run_id, out.stdout + out.stderr)
        assert "E4B_SHA=" + "a" * 40 in out.stdout, out.stdout
    refused = [("p141-5090-3", {}), ("p141-5090-prove-1", {}), ("p141-5090-1", {"P141_DRY": "1", "P141_BOX_EXEC": "/bin/false"}),
               ("p141-dry-1", {}), ("p134-5090-1", {}), ("p141-5090-1", {"E4B_SHA": "xyz"}),
               ("p141-5090-1", {"E4B_SHA": "a" * 39})]
    for run_id, extra in refused:
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 78 and "refusing" in out.stdout, (run_id, extra, out.stdout + out.stderr)
    assert "p141-5090-1" in PREREG and "p141-5090-2" in PREREG and "prove" not in DRIVE.lower().replace("approves", "")


def test_phase_one_is_k33s_rule_at_mixtrals_served_calls():
    s = _load("p141_select")
    assert s.PLANS == K33_PLANS and (s.WARM, s.SELECT_ROUNDS) == (20, 40) and (s.TOL_X, s.TOL_ABS) == (1.05, 2.0 ** -9)
    assert s.FAMILIES["mixtral"] == {"layers": 32, "top_k": 2, "gate_up": (28672, 4096), "down": (4096, 14336)}
    assert s.GATHER == {"gate_up": True, "down": False}, "the served W1 forms P141's call-site trace recorded"
    assert s.B1_ENV["GNF4_GEMV_BW"] == "1" and all(s.B1_ENV[k] is None for k in ("GNF4_GEMV_BW_DECODE", "GNF4_PDL"))
    assert "reported_never_ruled" in SELECT and '"plan_ms", "incumbent_ms", "floor_ms"' in SELECT
    r = _load("p141_reduce", "p115", "p110")
    sel = r._fake_select()
    assert r.selection_rule(sel) == {"gate_up": "16,1024,4,1", "down": "16,256,4,1"}
    assert s.select({p: 1.0 for p in K33_PLANS}, K33_PLANS) == K33_PLANS[0], "ties go to the earlier plan"
    for pin in ("16 x 768", "(16, 256, 4, 1)", "lowest median", "40 rounds"):
        assert pin in PREREG, pin


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("B0a", "B1a", "B1b", "B0b")' in REDUCE
    assert "SELF_LO, SELF_HI = 0.96, 1.04" in REDUCE and "GAIN_MIN_W1, GAIN_MIN_WCTL = 1.03, 0.99" in REDUCE
    assert "SELF_W1_LO, SELF_W1_HI = 0.985, 1.015" in REDUCE
    assert 'FLOORS = ("chunk",)' in REDUCE and 'OFF_ARMS = ("R", "rep", "chunk", "mutant_scale")' in REDUCE
    assert "TOL, SPREAD_X, SPREAD_MIN, K8_BUDGET, K8_GATED = pb.TOL, pb.SPREAD_X, pb.SPREAD_MIN, pb.K8_BUDGET, pb.K8_GATED" in REDUCE
    assert "INSTRUMENT_BUCKETS = _instrument_buckets()" in REDUCE
    r = _load("p141_reduce", "p115", "p110")
    assert r.INSTRUMENT_BUCKETS == (1, 2, 4, 8, 16) and r.SERVER_SIZE == (16, 768)
    assert r.INCUMBENT == {r.MIXTRAL: "scalar", r.TINY: "scalar"} and r.LAYERS[r.MIXTRAL] == 32
    assert r.WINDOWS == {"reading": 24, "dry": 4}
    for mode, w in (("DRY", 4), ("reading", 24)):
        assert f"WINDOWS_DEF={w};" in RUN, mode
    assert "1.03" in PREREG and "0.99" in PREREG and "[0.985, 1.015]" in PREREG and "[0.96, 1.04]" in PREREG


def test_the_premise_is_the_5090s_exact_counts():
    b = _load("p141_box", "p109")
    assert b.PREMISE["kernel"] == (35, {}) and b.PREMISE["served"] == (18, {})
    assert b.PREMISE["kernel_dry"][0] == 33 and set(b.PREMISE["kernel_dry"][1].values()) == {b.PDL}
    assert b.PREMISE["served_dry"][0] == 13 and set(b.PREMISE["served_dry"][1].values()) == {b.FP8}
    assert 'expect = f"{a.expect}_dry" if dry() else a.expect' in BOX
    assert "**35 passed, none skipped**" in PREREG and "**18 passed, none skipped**" in PREREG
    assert "--junitxml=$W/logs/premise_kernel.xml" in RUN and "--expect kernel" in RUN and "--expect served" in RUN


def test_the_order_puts_refusals_premise_and_phase_one_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: the image's torch", "REFUSED: card is", "the lane registers sm_120",
             "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4src", "python - <<'PYT'",
             "p141_reduce.py --self-test", "p141_box.py --self-test", "p141_select.py --self-test",
             "p115_quality.py --self-test", "python -m pytest test_nf4_gemv_bw.py", "--expect kernel", "--expect served",
             'echo "premise ok"', 'can_run $NEED_SELECT "phase 1"', "python $W/p141_select.py --family $SELFAM",
             "finish 23", 'if [ "$SEL_STATUS" != ok ]; then', "export P141_SELECT=$W/select.json",
             "for FAM in $FAMS; do\n  MODEL=$(model_of $FAM)", "python $W/p133_tiny.py", 'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py",
             "p141_box.py --prompts-only", "p141_box.py --mode fit", "FIT_VOID=1", "for TAG in B0a B1a B1b B0b; do",
             'if [ "$ARMS_OK" = 1 ]; then', "for PH in off on; do",
             'say "reduce $FAM"; python $W/p141_reduce.py --dir $W --family $FAM', "finish 31"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert "STOP-5 ($FAM)" in RUN[RUN.index('if [ "$ARMS_OK" = 1 ]; then'):]
    assert 'FAMS="mixtral"' in RUN and 'FAMS="tiny"' in RUN and "PROVE" not in RUN


def test_the_subject_is_the_default_server_at_p136s_size_and_only_the_switch_differs():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_KV_STEP_SELECT", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_MAX_TOKENS_PER_SEQ",
                 "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "GNF4_PDL", "GNF4_GEMV_DOTPAD", "GNF4_GEMV_BW",
                 "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "TRITON_INTERPRET"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert 'SIZE_ENV="E4B_PAGED_MAX_SEQS=16 E4B_PAGED_MAX_TOKENS_PER_SEQ=768"' in RUN
    assert 'ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$FW/nf4.arena E4B_PAGED_CALIB=$W/calib.json $SIZE_ENV"' in RUN
    assert 'BW="GNF4_GEMV_BW=1 GNF4_GEMV_BW_PLAN=$PLAN_ENV"' in RUN
    assert "PLAN_ENV=$(python -c \"import json; print(json.load(open('$W/select.json'))['env'])\")" in RUN
    assert 'ARM=${TAG:0:2}; KN=""; [ "$ARM" = B1 ] && KN=$BW' in RUN
    assert 'NEED=$NEED_QOFF; ARM=B0; KN=""; [ "$PH" = on ] && { NEED=$NEED_QON; ARM=B1; KN=$BW; }' in RUN
    assert "E4B_PAGED_GRAPHS=0 P141_ARM=$ARM" in RUN and "--windows $WINDOWS --cont $CONT --group 1" in RUN
    assert "ignore_patterns=['original/*', 'metal/*', 'consolidated*']" in RUN
    b = _load("p141_box", "p109")
    assert b.SERVER_SIZE == (16, 768) and "_size_ok(cfg)" in BOX
    assert 'if not dry() and (not cfg.graphs or cfg.placement != "all-vram"):' in BOX
    assert 'if cfg.graphs or cfg.placement != "all-vram":' in BOX and "if a.group != 1:" in BOX
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; WINDOWS_DEF=24; CONT_DEF=128" in RUN


def test_every_time_left_check_fits_the_guard():
    assert "guard 2.0 h" in PREREG
    keys = {"SELECT", "FETCH", "BAKE", "FIT", "ARM", "QOFF", "QON"}
    reading = dict(re.findall(r"NEED_(SELECT|FETCH|BAKE|FIT|ARM|QOFF|QON)=(\d+)",
                              RUN[RUN.index("else\n  FAMS=\"mixtral\""):RUN.index("fi\nmodel_of")]))
    assert set(reading) == keys
    for need in reading.values():
        assert int(need) + 600 <= 2.0 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_SELECT", "$NEED_FETCH", "$NEED_BAKE", "$NEED_FIT", "$NEED_ARM",
                                                 "$NEED"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    # 13 the disk floor and 18 the CUDA host floor; 14 and 17 are the launcher's. 18 appears once.
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 15, 16, 21, 22, 23, 25, 28, 31, 40} <= codes and 27 not in codes
    for c in ("rc 23", "rc 25", "rc 31"):
        assert c in PREREG, c
