"""The p116 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P116; e4b#1313).

`bench/p116/p116_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p116/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- P115's reducer and quality box, P109's, P110's, P108's and P97's boxes, P39's bake and calibration and the graph
  premise tests at P115's registered bytes;
- the order: refusals, install and tripwire, the self-tests, the premise on the card (the kernel's contract, 27; the served
  route, 18), then the fetch, the bake, the prompts, the four speed arms, the two quality phases (skipped when a speed arm
  failed) and the reducer;
- the subject: the default graph server, with only grouped-nf4-gemm's decode GEMV switch differing between arms, at
  K33's selected plans; the quality read at one window per pass (T == 1);
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p116"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p116_run.sh": LANE / "p116_run.sh",
    "p116_box.py": LANE / "p116_box.py",
    "p116_reduce.py": LANE / "p116_reduce.py",
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
PREMISE = ("test_decode_graph_buckets.py", "test_kv_step_select.py", "test_gemv_bw_served_gpu.py")
RUN = (LANE / "p116_run.sh").read_text()
DRIVE = (LANE / "p116_drive.sh").read_text()
REDUCE = (LANE / "p116_reduce.py").read_text()
BOX = (LANE / "p116_box.py").read_text()
PREREG = (LANE / "PREREG-p116.md").read_text(encoding="utf-8")


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


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p116 launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p115 = dict((n, w) for w, n in _entries(REPO / "bench" / "p115" / "staged.sha256"))
    for name in ("p115_reduce.py", "p115_quality.py", "p109_box.py", "p110_box.py", "p108_box.py", "p97_box.py",
                 "k8_bake.py", "calib.json", "test_decode_graph_buckets.py", "test_kv_step_select.py"):
        assert mine[name] == p115[name], f"{name} is not at P115's bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p116_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in DRIVE and "--exclude 'gnf4src'" in DRIVE


def test_the_self_tests_pass():
    out = subprocess.run([sys.executable, str(LANE / "p116_reduce.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p116_reduce self-test OK (26 cases)" in out.stdout, out.stdout + out.stderr
    out = subprocess.run([sys.executable, str(LANE / "p116_box.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "p116_box self-test OK (14/14 cases)" in out.stdout, out.stdout + out.stderr


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("B0a", "B1a", "B1b", "B0b")' in REDUCE
    assert "SELF_LO, SELF_HI = 0.96, 1.04" in REDUCE and "GAIN_MIN_W1, GAIN_MIN_W16 = 1.03, 0.99" in REDUCE
    assert "SELF_W1_LO, SELF_W1_HI = 0.985, 1.015" in REDUCE and "[0.985, 1.015] at W1" in PREREG
    assert 'FLOORS = ("chunk",)' in REDUCE and 'OFF_ARMS = ("R", "rep", "chunk", "mutant_scale")' in REDUCE
    assert "TOL, SPREAD_X, SPREAD_MIN, K8_BUDGET, K8_GATED = pb.TOL, pb.SPREAD_X, pb.SPREAD_MIN, pb.K8_BUDGET, pb.K8_GATED" in REDUCE
    pb_src = (REPO / "bench" / "p115" / "p115_reduce.py").read_text()
    assert "TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005" in pb_src and "K8_BUDGET = 0.05" in pb_src
    assert 'INCUMBENT = {QWEN: "dotpad", GRAN: "scalar"}' in REDUCE
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == "5a60c37dbd0756040052b603c9b0ee680f06d444", "K33's measured cut"
    for model, rev in re.findall(r'(QWEN|GRAN): "([0-9a-f]{40})"', REDUCE):
        assert rev in RUN, model
    assert "1.03" in PREREG and "0.99" in PREREG


def test_the_reducer_expects_the_windows_the_runner_runs():
    """Amendment 1: the reducer's per-text window count must be the runner's registered default, reading and proof.
    p116-5090-1 VOIDed because the reducer still carried P115's 48 while the runner (and the PREREG) ran 24."""
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p115")]
    try:
        import p116_reduce as r
    finally:
        del sys.path[:2]
    prove = RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]
    reading = RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]
    assert f"WINDOWS_DEF={r.WINDOWS[r.QWEN]};" in reading and r.WINDOWS[r.QWEN] == 24, r.WINDOWS
    assert f"WINDOWS_DEF={r.WINDOWS[r.GRAN]};" in prove and r.WINDOWS[r.GRAN] == 12, r.WINDOWS
    assert "**24 windows each**" in PREREG


def test_the_plans_are_k33s_selected_plans():
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p109")]
    try:
        import p116_box as b
    finally:
        del sys.path[:2]
    assert b.BW_PLANS == {(1536, 2048): (16, 1024, 4, 1), (2048, 768): (16, 256, 4, 1),
                          (1024, 1536): (16, 512, 8, 1), (1536, 512): (16, 256, 8, 1),
                          (2048, 2048): (16, 1024, 4, 1), (2048, 1024): (16, 256, 4, 1)}
    k33 = REPO.parent / "grouped-nf4-gemm" / "kernel" / "receipts-k33" / "5090" / "k33.json"
    if k33.exists():                                             # the sibling checkout, when it carries K33's read
        plans = json.loads(k33.read_text())["plan"]
        fam = {"qwen3": ((1536, 2048), (2048, 768)), "granite": ((1024, 1536), (1536, 512)),
               "olmoe": ((2048, 2048), (2048, 1024))}
        for f, (gu, dn) in fam.items():
            assert b.BW_PLANS[gu] == tuple(plans[f"{f}/gate_up"]) and b.BW_PLANS[dn] == tuple(plans[f"{f}/down"]), f
    assert "GNF4_GEMV_BW_PLAN=$BW_PLAN" in RUN and "BW_PLAN=$(cd $W && python -c \"import p116_box; print(p116_box.BW_PLAN)\")" in RUN


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4src", "python - <<'PYT'",
             "p116_reduce.py --self-test", "p116_box.py --self-test", "p115_quality.py --self-test",
             "python -m pytest test_nf4_gemv_bw.py", "python -m pytest " + " ".join(PREMISE), 'echo "premise ok"',
             'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py", "p116_box.py --prompts-only",
             "for TAG in B0a B1a B1b B0b; do", 'if [ "$ARMS_OK" = 1 ]; then', "for PH in off on; do",
             "python $W/p116_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'echo "$LASTK" | grep -q "27 passed"' in RUN and 'echo "$LASTL" | grep -q "18 passed"' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert "hr._collapsed_grouping(1, None) == (True, False) and hr._collapsed_grouping(4, None) == (False, True)" in trip
    assert 'ng._BW_SHAPES == frozenset() and ng._bw() == "0"' in trip and '== "prmt32"' in trip
    assert 'md.version("grouped-nf4-gemm") == "0.42.0"' in trip and 'transformers.__version__ == "5.17.0"' in trip
    assert "STOP-5" in RUN[RUN.index('if [ "$ARMS_OK" = 1 ]; then'):]


def test_the_premise_files_collect_18_cases():
    # test_gemv_bw_served_gpu.py skips at collection without grouped-nf4-gemm (its module-level importorskip), so the
    # count is only meaningful where the kernel package is installed: CI and the box (the maintainer's Mac has neither)
    pytest.importorskip("nf4_grouped", reason="the premise count needs grouped-nf4-gemm installed")
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          *[str(REPO / "tests" / f) for f in PREMISE]], capture_output=True, text=True, cwd=REPO)
    assert re.search(r"\b18 tests? collected\b", out.stdout), out.stdout[-600:] + out.stderr[-600:]


def test_the_subject_is_the_default_graph_server_and_only_the_switch_differs():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_KV_STEP_SELECT", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE",
                 "GNF4_PDL", "GNF4_GEMV_DOTPAD", "GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "TRITON_INTERPRET"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert 'ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"' in RUN
    assert 'BW="GNF4_GEMV_BW=1 GNF4_GEMV_BW_PLAN=$BW_PLAN"' in RUN
    assert 'ARM=${TAG:0:2}; KN=""; [ "$ARM" = B1 ] && KN=$BW' in RUN
    assert 'NEED=$NEED_QOFF; ARM=B0; KN=""; [ "$PH" = on ] && { NEED=$NEED_QON; ARM=B1; KN=$BW; }' in RUN
    assert "E4B_PAGED_GRAPHS=0 P116_ARM=$ARM" in RUN and "--windows $WINDOWS --cont $CONT --group 1" in RUN
    assert "if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, \"all-vram\", (1, 2, 4, 8, 16)):" in BOX
    assert "if cfg.graphs or cfg.placement != \"all-vram\":" in BOX and "if a.group != 1:" in BOX
    assert BOX.index("q.KernelCounters().install()") < BOX.index("parts = build_engine(cfg)\n    model = parts.runner.model")
    assert "SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; WINDOWS_DEF=24; CONT_DEF=128" in RUN
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
    # 13 the disk floor and 18 the CUDA host floor; 14 and 17 are the launcher's. 18 appears once.
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 15, 16, 21, 22, 25, 27, 40} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p116-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P116_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p116_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p116"), out.stdout + out.stderr
