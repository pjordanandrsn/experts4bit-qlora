"""Lane P115 Phase C's staged-file pin must match the repo (the e4b#642 check, mirrored for Phase C; e4b#1313).

`bench/p115/p115c_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p115/staged-c.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the amendment's shape
(bench/p115/PREREG-p115.md, Amendment 2):
- Phase B's reducer and quality box, P109's, P110's, P108's and P97's boxes, P39's bake and calibration, and the GPU
  premise tests at Phase A/B's registered bytes; P98's bake at P98's;
- the order: refusals, install and tripwire, the self-tests, the premise on the card (48 cases), then per model, Granite
  first: fetch, bake, prompts, serve off / on / explicit, then SANE off / on (gpt-oss, Qwen3.6) or Phase B's quality off /
  on at Phase B's size (Granite); then the reducer;
- the subject: the default server, gpt-oss on SC2g's e4b path, only the four fusion knobs differing between arms, SANE
  and quality built eager;
- the rule's constants and predictions as the pre-registration states them, the explicit control's refusal pattern
  against the code's own refusals, every time-left check inside its guard, and the exit codes.
"""
import hashlib
import importlib.util
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p115"
PIN = LANE / "staged-c.sha256"
SOURCES = {
    "p115c_run.sh": LANE / "p115c_run.sh",
    "p115c_box.py": LANE / "p115c_box.py",
    "p115c_reduce.py": LANE / "p115c_reduce.py",
    "p115_reduce.py": LANE / "p115_reduce.py",
    "p115_quality.py": LANE / "p115_quality.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "p110_box.py": REPO / "bench" / "p110" / "p110_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "p98_bake.py": REPO / "bench" / "p98" / "p98_bake.py",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_kv_step_select.py": REPO / "tests" / "test_kv_step_select.py",
    "test_fused_glue_decode_graphs_gpu.py": REPO / "tests" / "test_fused_glue_decode_graphs_gpu.py",
    "test_fusion_modes.py": REPO / "tests" / "test_fusion_modes.py",
}
PREMISE = ("test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py",
           "test_fusion_modes.py")
RUN = (LANE / "p115c_run.sh").read_text()
DRIVE = (LANE / "p115c_drive.sh").read_text()
REDUCE = (LANE / "p115c_reduce.py").read_text()
BOX = (LANE / "p115c_box.py").read_text()
PREREG = (LANE / "PREREG-p115.md").read_text(encoding="utf-8")
AMEND = PREREG[PREREG.index("## Amendment 2"):] if "## Amendment 2" in PREREG else ""


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p115c launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    ab = dict((n, w) for w, n in _entries(LANE / "staged.sha256"))
    p98 = dict((n, w) for w, n in _entries(REPO / "bench" / "p98" / "staged.sha256"))
    for name in ("p115_reduce.py", "p115_quality.py", "p109_box.py", "p110_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json",
                 "test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py"):
        assert mine[name] == ab[name], f"{name} is not at Phase A/B's bytes"
    assert mine["p98_bake.py"] == p98["p98_bake.py"], "p98_bake.py is not at P98's bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged-c.sha256"]:
        if name != "p115c_run.sh":
            assert name in staged, name
    assert "--include 'work_*/bake.json' --exclude 'work_*/*'" in DRIVE, "arenas, snapshots and SANE references stay on the box"


def test_the_self_tests_pass():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}   # Windows needs it
    out = subprocess.run([sys.executable, str(LANE / "p115c_reduce.py"), "--self-test"], capture_output=True, text=True,
                         env=base)
    assert out.returncode == 0 and "p115c_reduce self-test OK (27 cases)" in out.stdout, out.stdout + out.stderr
    env = {"PYTHONPATH": str(REPO / "bench" / "p109"), **base}
    out = subprocess.run([sys.executable, str(LANE / "p115c_box.py"), "--self-test"], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "p115c_box self-test OK (9/9 cases)" in out.stdout, out.stdout + out.stderr


def test_the_rule_is_the_registered_rule():
    r = _load(LANE / "p115c_reduce.py", "p115c_reduce_pin")
    assert (r.SANE_WINDOWS, r.SANE_BIAS, r.SANE_ARGMAX) == (12, 0.02, 0.95)
    assert "12 wikitext windows" in PREREG and "0.02 nats" in PREREG and "≥ 0.95" in PREREG
    assert r.READING == ("granite", "gptoss", "qw36") and r.FLIP_MODELS == ("gptoss", "qw36") and r.PROOF == ("granite",)
    assert r.SANE_MODELS == {False: ("gptoss", "qw36"), True: ("granite",)} and r.QUALITY_MODELS == ("granite",)
    assert r.QUALITY_WINDOWS == {False: 48, True: 12} and r.pb.K8_GATED == ("wikitext",) and r.pb.K8_BUDGET == 0.05
    assert (r.pb.TOL, r.pb.SPREAD_X, r.pb.SPREAD_MIN) == (0.01, 2.0, 0.005), "Granite's read is Phase B's rule unchanged"

    def text(c):
        return f"{c['fuse_qkv_n']} / {c['fuse_t1_glue_n']} / [{c['fuse_t1_glue_r2_n'][0]}, {c['fuse_t1_glue_r2_n'][1]}] / " \
               f"{c['fuse_router_epilogue_n']}"
    assert text(r.PREDICTED["gptoss"]) == "0 / 49 / [24, 0] / 24" and text(r.PREDICTED["qw36"]) == "0 / 0 / [0, 0] / 40"
    assert text(r.PREDICTED["granite"]) == "0 / 65 / [32, 32] / 32"
    for m in r.PREDICTED:
        assert text(r.PREDICTED[m]) in PREREG, m
    assert r.GNF4_SHA == "b4f93f1c62d1e3436ed45bec8ccd608c90433737" and f"GNF4_SHA={r.GNF4_SHA}" in RUN
    for tag, (model, rev) in r.MODELS.items():
        assert f'{tag}) echo "{model} {rev}"' in RUN, tag
    assert r.MODELS["gptoss"][1].startswith("6cee5e81") and r.MODELS["qw36"][1].startswith("995ad96e")


def _kernels():
    fm = _load(REPO / "tests" / "test_fusion_modes.py", "test_fusion_modes_for_p115c")
    return fm


@pytest.mark.parametrize("family", ["gpt_oss", "qwen3_5_moe", "granite"])
def test_the_explicit_control_reads_the_codes_own_refusal(monkeypatch, family):
    """All four knobs at 1 must refuse on every Phase C family, and the refusal must be one EXPLICIT_RE accepts:
    gpt-oss and Granite refuse the q/k/v fusion; the Qwen3.5 hybrid's glue round 1 refuses first (Amendment 2's
    correction)."""
    torch = pytest.importorskip("torch")  # noqa: F841
    fm = _kernels()
    r = _load(LANE / "p115c_reduce.py", "p115c_reduce_pin2")
    from experts4bit_qlora.serve_paged import FUSION_KNOBS, PagedServeConfig, _apply_fusions
    monkeypatch.setitem(sys.modules, "int4_b32", fm._kernels())
    try:
        model = {"gpt_oss": fm._gpt_oss, "qwen3_5_moe": lambda: fm._qwen3_5_moe(4), "granite": lambda: fm._granite(2)}[family]()
    except pytest.skip.Exception:
        raise
    except Exception as e:                                   # as test_fusion_modes: a config this transformers cannot build
        pytest.skip(f"tiny {family} not constructible here: {type(e).__name__}: {e}")
    with pytest.raises(RuntimeError) as ei:
        _apply_fusions(model, PagedServeConfig(fuse_qkv=True, fusion_modes={k: "1" for k in FUSION_KNOBS}))
    assert r.EXPLICIT_RE.search(str(ei.value)), str(ei.value)
    want = "E4B_FUSE_T1_GLUE=1" if family == "qwen3_5_moe" else "E4B_PAGED_FUSE_QKV=1"
    assert str(ei.value).startswith(want), str(ei.value)


def test_the_premise_is_the_four_files_and_48_cases():
    run = RUN[RUN.index("# ---- the premise"):RUN.index("model_of(){")]
    assert "python -m pytest " + " ".join(PREMISE) + " -q -rs -p no:cacheprovider" in run
    assert 'grep -q "48 passed" && ! echo "$LASTL" | grep -q skipped' in run and "finish 25" in run
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          *[str(REPO / "tests" / f) for f in PREMISE]], capture_output=True, text=True, cwd=REPO)
    assert re.search(r"\b48 tests? collected\b", out.stdout), out.stdout[-600:] + out.stderr[-600:]


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "python - <<'PYT'", "p115c_reduce.py --self-test", "p115c_box.py --self-test", "p115_quality.py --self-test",
             "python -m pytest test_decode_graph_buckets.py", 'echo "premise ok"', "for M in $TAGS; do",
             'say "fetch $M $MODEL @ $REV"', "python $W/p98_bake.py", "python $W/k8_bake.py",
             "p115c_box.py --prompts-only", 'STEPS="serve:off serve:on serve:explicit sane:off sane:on"',
             'STEPS="serve:off serve:on serve:explicit quality:off quality:on"', "for STEP in $STEPS; do",
             "python $W/p115c_reduce.py --dir $W --out $W/verdict_c.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'TAGS="granite gptoss qw36"' in RUN and 'TAGS="granite"' in RUN, "cheapest first; a later failure leaves earlier reads"
    assert ('[ "$M" = granite ] && [ "$PROVE" = 1 ] && STEPS="serve:off serve:on serve:explicit sane:off sane:on quality:off '
            'quality:on"') in RUN, "the proof runs every process kind"
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'serve_paged._fusion_env("E4B_FUSE_T1_GLUE", "") == "0"' in trip, "the knobs are opt-in at the launch commit"
    assert 'serve_paged._graphs_env("", "cuda", "all-vram") is True' in trip
    assert "all(os.environ.get(k) is None for k in serve_paged.FUSION_KNOBS)" in trip
    assert 'md.version("grouped-nf4-gemm") == "0.42.0"' in trip and 'transformers.__version__ == "5.17.0"' in trip


def test_the_subject_is_the_default_server_and_only_the_knobs_differ():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_KV_STEP_SELECT", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE",
                 "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI", "E4B_SERVE_EXP_INT4", "GNF4_PDL", "GNF4_TRITON_PREBIND"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert 'KN="E4B_PAGED_FUSE_QKV=$VAL E4B_FUSE_T1_GLUE=$VAL E4B_FUSE_T1_GLUE_R2=$VAL E4B_FUSE_ROUTER_EPI=$VAL"' in RUN
    assert 'VAL=0; [ "$ARM" = on ] && VAL=auto; [ "$ARM" = explicit ] && VAL=1' in RUN
    assert 'GR=""; [ "$MODE" != serve ] && GR="E4B_PAGED_GRAPHS=0"' in RUN
    assert 'WIN=$WINDOWS; REF=$W/work_$M/ref; [ "$MODE" = quality ] && { WIN=$QWINDOWS; REF=$W/work_$M/qref; }' in RUN
    sc2g = re.search(r'SC2G_E4B_ENV="([^"]+)"', (REPO / "bench" / "sc2" / "sc2g_box_g.sh").read_text()).group(1)
    assert f'gptoss) echo "{sc2g}";;' in RUN, "gpt-oss is served on SC2g's e4b path"
    assert ("ENGINE_ENV=\"E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work_$M/nf4.arena "
            "E4B_PAGED_CALIB=$W/calib.json $(model_env $M)\"") in RUN
    assert 'ARM_VALUE = {"off": "0", "on": "auto", "explicit": "1"}' in BOX
    assert "if cfg.graphs or cfg.placement != \"all-vram\":" in BOX, "SANE is the default server built eager"
    assert "counters = q.KernelCounters().install()" in BOX
    assert BOX.index("q.KernelCounters().install()") < BOX.index("parts = build_engine(cfg)\n    model = parts.runner.model")
    assert "NEW_DEF=32; WINDOWS_DEF=12; QWINDOWS_DEF=48; CONT_DEF=128" in RUN
    assert "NEW_DEF=8; WINDOWS_DEF=12; QWINDOWS_DEF=12; CONT_DEF=32" in RUN
    # Granite's quality read runs at Phase B's size: the box's defaults are the quality box's
    q = (LANE / "p115_quality.py").read_text()
    for arg, default in (("--group", "12"), ("--prompt", "512"), ("--chunk", "512"), ("--floor-chunk", "256")):
        assert f'ap.add_argument("{arg}", type=int, default={default})' in q, arg
        assert f'p.add_argument("{arg}", type=int, default={default})' in BOX, arg
    assert "rec = q.measure_phase(model, windows, phase=arm, prompt=a.prompt, cont=a.cont, chunk=a.chunk,\n" \
           "                          floor_chunk=a.floor_chunk, group=a.group" in BOX
    assert "windows = {t: q.LOADERS[t](parts.tokenizer, a.windows, a.prompt, a.cont) for t in q.TEXTS}" in BOX


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 0.75 h" in AMEND and "guard 2.0 h" in AMEND
    keys = {"FETCH", "BAKE", "PROC"}
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|PROC)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|PROC)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == keys
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 2.0 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_PROC"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    # 13 the disk floor and 18 the CUDA host floor (Amendment 1); 14 and 17 are the launcher's. 18 appears once.
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 15, 16, 21, 22, 25, 27} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p115c-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P115C_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p115c_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p115c"), out.stdout + out.stderr
