"""The p124 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P124 from P122's).

`bench/p124/p124_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p124/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test and pins the lane's shape:
- P119's, P117's, P108's and P97's boxes, P39's bake and the bucket test at their registered bytes; grouped-nf4-gemm's
  K16 test at the bytes of the pinned commit; the route's e4b tests as this tree has them;
- the order: refusals, install and tripwire, the self-test, the three premise tests, then (and only then) the fetch,
  the bake, the box and the reducer;
- the subject: SC2e's int4 stack built eager with one slot and the route enabled, the box setting the route per arm,
  and the arms the box and the reducer register;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import importlib.util
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p124"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p124_run.sh": LANE / "p124_run.sh",
    "p124_box.py": LANE / "p124_box.py",
    "p124_reduce.py": LANE / "p124_reduce.py",
    "p119_box.py": REPO / "bench" / "p119" / "p119_box.py",
    "p117_box.py": REPO / "bench" / "p117" / "p117_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_int4_smallm_interp.py": LANE / "test_int4_smallm_interp.py",
    "test_int4_attn.py": REPO / "tests" / "test_int4_attn.py",
    "test_int4_attn_wide.py": REPO / "tests" / "test_int4_attn_wide.py",
}
# grouped-nf4-gemm's kernel/test_int4_smallm_interp.py at the pinned commit (#522), byte for byte
K16_TEST_SHA256 = "3d3b834aa0d60361f4b553067e478641ecacd873d00b7bac143e02e305e37b50"
GNF4 = "4ed26d962ff03e664785c20413db3360a0f4d648"
RUN = (LANE / "p124_run.sh").read_text()
REDUCE = (LANE / "p124_reduce.py").read_text()
BOX = (LANE / "p124_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p124 launch refuses ON A RENTED BOX"


def test_imported_boxes_and_the_premise_tests_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p119 = dict((n, w) for w, n in _entries(REPO / "bench" / "p119" / "staged.sha256"))
    for name in ("p119_box.py", "p117_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json",
                 "test_decode_graph_buckets.py"):
        assert mine[name] == p119[name], name
    assert mine["test_int4_smallm_interp.py"] == K16_TEST_SHA256
    k16 = (LANE / "test_int4_smallm_interp.py").read_text()
    assert 'os.environ.setdefault("TRITON_INTERPRET", "1")' in k16                 # the runner sets 0 explicitly
    assert "def test_at_most_16_rows_is_k16s_launch_bit_for_bit" in k16 and "def test_up_to_64_rows_matches" in k16
    wide = (REPO / "tests" / "test_int4_attn_wide.py").read_text()
    assert "def test_two_modules_of_one_width_in_one_graph_replay_to_their_references" in wide
    assert "def test_two_graphs_on_the_route_replay_in_either_order" in wide
    assert 'pytest.importorskip("test_int4_attn")' in wide                       # why test_int4_attn.py is staged


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p124_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p124_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver           # the arena stays on the box


def test_the_reducer_self_tests_and_pins_its_constants():
    out = subprocess.run([sys.executable, str(LANE / "p124_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (42 cases)" in out.stdout, out.stdout + out.stderr
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == GNF4
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model
    assert "math.fsum" in REDUCE
    for const in ("PREMISE_SHARE = 0.05", "NOISE = 0.015", "BAR = 0.98",
                  "WARM, STEPS, BUSY, PROFILED, PROFILE_WARM = 5, 256, 32, 8, 3",
                  "BIAS_SLACK, SPREAD_FLOOR = 0.01, 0.005"):
        assert const in REDUCE, const
    prereg = (LANE / "PREREG-p124.md").read_text(encoding="utf-8")
    for words in ("under **5 %**", "more than **1.5 %**", "at most **0.98**", "**256 timed steps**", "42 cases",
                  "exactly 293 times (5 + 256 + 32)", "mean d_X ≤ B_floor + 0.01 nats", "2 × max(S_floor, 0.005)"):
        assert words in prereg, words


def _mods():
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p119"), str(REPO / "bench" / "p117"), str(REPO / "bench" / "p108"),
                    str(REPO / "bench" / "p97")]
    try:
        mods = {}
        for name in ("p124_box", "p124_reduce"):
            sp = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
            m = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(m)
            mods[name] = m
    finally:
        del sys.path[:5]
    return mods["p124_box"], mods["p124_reduce"]


def test_the_box_and_the_reducer_register_the_same_arms():
    box, red = _mods()
    assert box.BLOCKS == red.BLOCKS == ("a", "b") and box.SETTINGS == red.SETTINGS == ("OFF", "ON")
    assert box.DEPTHS == red.DEPTHS == ("64", "32")
    assert box.QUALITY == red.QUALITY and box.FLOORS == red.FLOORS == ("half", "chunk")
    assert box.SUBJECTS == red.SUBJECTS == ("ON64", "ON32") and box.MUTANTS == red.MUTANTS
    assert {a: tuple(k["buckets"]) for a, k in box.Q_KW.items()} == {a: tuple(b) for a, b in red.Q_BUCKETS.items()}
    assert {a: k["wide"] for a, k in box.Q_KW.items()} == red.Q_WIDE
    assert box.WARM == red.WARM and box.BUSY == red.BUSY and tuple(box.B64) == red.B64 and tuple(box.B32) == red.B32
    assert "warm=3, steps=8" in BOX and red.PROFILED == 8 and red.PROFILE_WARM == 3
    assert '"--steps", type=int, default=256' in BOX and red.STEPS == 256
    assert [q[0] for q in red.PREDICTIONS] == [f"Q{i}" for i in range(1, 10)]


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p124_reduce.py --self-test", "python -m pytest test_decode_graph_buckets.py",
             "TRITON_INTERPRET=0 perl -e 'alarm 1200; exec @ARGV' python -m pytest test_int4_smallm_interp.py",
             "python -m pytest test_int4_attn_wide.py",
             'echo "premise ok"', 'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py",
             "python $W/p124_box.py --out $W/box.json --rows $WINDOWS --cont $CONT",
             "python $W/p124_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "7 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    assert 'grep -q "25 passed" && ! echo "$LASTK" | grep -q skipped' in RUN
    assert 'grep -q "11 passed" && ! echo "$LASTW" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert '"block_m" in inspect.signature(int4_smallm.gemm_int4_b32_smallm).parameters' in trip
    assert "int4_smallm.SMALLM_ROWS_MAX == 64" in trip
    assert "int4_attn.Int4Linear.SMALLM_ROWS_MAX == 16 and int4_attn.Int4Linear.WIDE_ROWS_MAX == 64" in trip
    assert "int4_attn._wide_supported(int4_smallm.gemm_int4_b32_smallm)" in trip
    assert "from experts4bit_qlora.engines.step_trace import StepTrace" in trip
    assert '_caps = hot_residency._wide_tiles_caps(int4_b32.build_group_tiles_fused)' in trip
    assert 'hot_residency._wide_tiles_mode_env() == "auto" and all(_caps)' in trip
    assert 'md.version("grouped-nf4-gemm") == "0.43.0"' in trip
    assert "default_buckets(64) == (1, 2, 4, 8, 16, 32, 64)" in trip and '"capture" in inspect.signature' in trip
    assert 'hasattr(PagedModelRunner, "disable_decode_graphs")' in trip
    assert "import p119_box" in trip and "import p117_box" in trip and "import p108_box" in trip
    assert '"mutant" in inspect.signature(p117_box.paged_pass).parameters' in trip
    assert "VOID|NO_READING) say \"PROVE:" in RUN


def test_the_subject_is_sc2es_int4_stack_built_eager_with_one_slot_and_the_route_enabled():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(': > summary.txt')]
    for knob in ("E4B_ATTN_INT4_WIDE", "E4B_ATTN_INT4_SMALLM", "E4B_INT4_WIDE_TILES", "E4B_PAGED_GRAPHS",
                 "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS", "E4B_PAGED_PREFILL_GRAPH", "E4B_PAGED_BULK_KV",
                 "E4B_SERVE_EXP_INT4", "E4B_PAGED_FUSE_QKV", "E4B_INT4_PREFILL"):
        assert knob in unset, knob
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
    for var in ("FOLDS", "SPEEDENV", "ROUTEENV"):                      # SC2e's stack, byte for byte
        line = next(x for x in sc1.splitlines() if x.startswith(f"{var}="))
        assert line.split("   #")[0] in RUN, var
    assert 'else LEVERS="$ROUTEENV $SPEEDENV E4B_PAGED_FUSE_QKV=1"; fi' in RUN
    assert 'LEVERS="$ROUTEENV E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0"' in RUN         # the proof has the route
    assert 'LEVERS="$LEVERS E4B_ATTN_INT4_WIDE=1"' in RUN
    assert 'ENGINE_KNOBS="E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0"' in RUN
    assert "$ENGINE_KNOBS $LEVERS\"" in RUN
    assert 'cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1' in BOX and "model = parts.runner.model" in BOX
    assert "runner.enable_decode_graphs(B64, capture=True, verbose=False)" in BOX
    assert "with Route(model, st == \"ON\") as route:" in BOX and "bulk_kv=cfg.bulk_kv" in BOX
    assert "for st in first:                             # strict alternation, one step of each per pair" in BOX
    assert "m._wide = self.wide" in BOX and '"attn_int4_wide_env": os.environ.get("E4B_ATTN_INT4_WIDE")' in BOX
    for flag in ('"--rows", type=int, default=64', '"--prompt", type=int, default=512', '"--cont", type=int, default=128',
                 '"--stride", type=int, default=3072'):
        assert flag in BOX, flag
    assert "WINDOWS_DEF=64; CONT_DEF=128\n" in RUN and "WINDOWS_DEF=40; CONT_DEF=32\n" in RUN


def test_every_time_left_check_fits_its_own_guard():
    prereg = (LANE / "PREREG-p124.md").read_text(encoding="utf-8")
    assert "guard 0.75 h" in prereg and "guard 2.0 h" in prereg
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|BOX)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|BOX)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == {"FETCH", "BAKE", "BOX"}
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 2.0 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_BOX"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes
    assert {16, 25, 26, 27} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p124-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P124_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p124_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p124"), out.stdout + out.stderr
