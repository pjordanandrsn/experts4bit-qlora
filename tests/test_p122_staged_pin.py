"""The p122 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P122 from P120's).

`bench/p122/p122_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p122/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test and pins the lane's shape:
- P120's box at P120's registered bytes (the instrument is P120's, unchanged), P119's box and everything it stages at
  P119's; grouped-nf4-gemm's chunked cumsum test at the bytes of #519's merge commit;
- the order: refusals, install and tripwire, the self-test, both premise tests, then (and only then) the fetch, the
  bake, the box and the reducer;
- the subject: SC2e's int4 stack built eager with one slot, the knob unset until the box sets it per arm, and the arms
  the box and the reducer register;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import importlib.util
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p122"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p122_run.sh": LANE / "p122_run.sh",
    "p120_box.py": REPO / "bench" / "p120" / "p120_box.py",
    "p122_reduce.py": LANE / "p122_reduce.py",
    "p119_box.py": REPO / "bench" / "p119" / "p119_box.py",
    "p117_box.py": REPO / "bench" / "p117" / "p117_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_tile_table_cumsum_chunked_interp.py": LANE / "test_tile_table_cumsum_chunked_interp.py",
}
# grouped-nf4-gemm's kernel/test_tile_table_cumsum_chunked_interp.py at b155f1c1 (#519's merge commit), byte for byte
CHUNK_TEST_SHA256 = "da5fa63cebc2b488b3f515098844861ca9d16e8d3fbe251f2708b2316edce2c4"
GNF4 = "b155f1c120843b639581ac5d5242cb1ec37d71e1"
RUN = (LANE / "p122_run.sh").read_text()
REDUCE = (LANE / "p122_reduce.py").read_text()
BOX = (REPO / "bench" / "p120" / "p120_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p122 launch refuses ON A RENTED BOX"


def test_imported_boxes_and_the_premise_tests_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p119 = dict((n, w) for w, n in _entries(REPO / "bench" / "p119" / "staged.sha256"))
    for name in ("p119_box.py", "p117_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json",
                 "test_decode_graph_buckets.py"):
        assert mine[name] == p119[name], name
    p120 = dict((n, w) for w, n in _entries(REPO / "bench" / "p120" / "staged.sha256"))
    assert mine["p120_box.py"] == p120["p120_box.py"], "P120's box, at P120's registered bytes"
    assert mine["test_tile_table_cumsum_chunked_interp.py"] == CHUNK_TEST_SHA256
    rank = (LANE / "test_tile_table_cumsum_chunked_interp.py").read_text()
    assert 'os.environ.setdefault("TRITON_INTERPRET", "1")' in rank                 # the runner sets 0 explicitly


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p122_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p122_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver           # the arena stays on the box


def test_the_reducer_self_tests_and_pins_its_constants():
    out = subprocess.run([sys.executable, str(LANE / "p122_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (30 cases)" in out.stdout, out.stdout + out.stderr
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"GNF4_SHA={gnf4}" in RUN and gnf4 == GNF4
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model
    assert "math.fsum" in REDUCE
    for const in ("PREMISE_SHARE = 0.05", "NOISE = 0.015", "BAR = 0.98", "WARM, STEPS, PROFILED = 5, 256, 8"):
        assert const in REDUCE, const
    prereg = (LANE / "PREREG-p122.md").read_text(encoding="utf-8")
    for words in ("under **5 %**", "more than **1.5 %**", "both **≤ 0.98**", "**256 timed steps**", "30 cases"):
        assert words in prereg, words


def _mods():
    sys.path[:0] = [str(LANE), str(REPO / "bench" / "p119"), str(REPO / "bench" / "p117"), str(REPO / "bench" / "p108"),
                    str(REPO / "bench" / "p97")]
    try:
        mods = {}
        for name, where in (("p120_box", REPO / "bench" / "p120"), ("p122_reduce", LANE)):
            sp = importlib.util.spec_from_file_location(name, where / f"{name}.py")
            m = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(m)
            mods[name] = m
    finally:
        del sys.path[:5]
    return mods["p120_box"], mods["p122_reduce"]


def test_the_box_and_the_reducer_register_the_same_arms_and_kernels():
    box, red = _mods()
    assert box.SERVED == red.SERVED == ("OFF_a", "ON_a", "ON_b", "OFF_b")
    assert box.BUILDER == red.BUILDER and box.TABLE == red.TABLE == "_tile_table_r1"
    assert box.WARM == red.WARM and tuple(box.B64) == red.B64
    assert "warm=3, steps=8" in BOX and red.PROFILED == 8
    assert '"--steps", type=int, default=256' in BOX and red.STEPS == 256
    assert [q[0] for q in red.PREDICTIONS] == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p122_reduce.py --self-test", "python -m pytest test_decode_graph_buckets.py",
             "TRITON_INTERPRET=0 perl -e 'alarm 1200; exec @ARGV' python -m pytest test_tile_table_cumsum_chunked_interp.py",
             'echo "premise ok"', 'say "fetch $MODEL @ $REV"', "python $W/k8_bake.py",
             "python $W/p120_box.py --out $W/box.json --rows $WINDOWS", "python $W/p122_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "7 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    assert 'grep -q "117 passed" && ! echo "$LASTR" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'md.version("grouped-nf4-gemm") == "0.43.0"' in trip
    assert '"rank" in inspect.signature(int4_b32.build_group_tiles_fused).parameters' in trip
    assert '"rchunk" in inspect.signature(int4_b32.build_group_tiles_fused).parameters' in trip
    assert "int4_b32._cumsum_rchunk(128, 512) == 64 and int4_b32.CUMSUM_TILE_ELEMS == 8192" in trip
    assert "hot_residency._WIDE_TILES_MAX == 1024 and callable(hot_residency._wide_tiles_mode_env)" in trip
    assert "default_buckets(64) == (1, 2, 4, 8, 16, 32, 64)" in trip and '"capture" in inspect.signature' in trip
    assert 'hasattr(PagedModelRunner, "disable_decode_graphs")' in trip
    assert "import p119_box" in trip and "import p117_box" in trip and "import p108_box" in trip
    assert "VOID|NO_READING|TOKENS_DIFFER) say \"PROVE:" in RUN


def test_the_subject_is_sc2es_int4_stack_built_eager_with_one_slot():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(': > summary.txt')]
    for knob in ("E4B_INT4_WIDE_TILES", "E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS",
                 "E4B_PAGED_PREFILL_GRAPH", "E4B_PAGED_BULK_KV", "E4B_SERVE_EXP_INT4", "E4B_PAGED_FUSE_QKV",
                 "E4B_INT4_PREFILL"):
        assert knob in unset, knob
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
    for var in ("FOLDS", "SPEEDENV", "ROUTEENV"):                      # SC2e's stack, byte for byte
        line = next(x for x in sc1.splitlines() if x.startswith(f"{var}="))
        assert line.split("   #")[0] in RUN, var
    assert 'else LEVERS="$ROUTEENV $SPEEDENV E4B_PAGED_FUSE_QKV=1"; fi' in RUN
    assert 'ENGINE_KNOBS="E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0"' in RUN
    assert "$ENGINE_KNOBS $LEVERS\"" in RUN
    assert 'cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1' in BOX and "model = parts.runner.model" in BOX
    assert "runner.enable_decode_graphs(B64, capture=True, verbose=False)" in BOX
    assert "with p119_box._Grouped(), torch.no_grad():" in BOX and "bulk_kv=cfg.bulk_kv" in BOX
    assert 'os.environ["E4B_INT4_WIDE_TILES"] = "1" if on else "0"' in BOX
    assert 'os.environ.pop("E4B_INT4_WIDE_TILES", None)' in BOX
    for flag in ('"--rows", type=int, default=64', '"--prompt", type=int, default=512', '"--stride", type=int, default=3072'):
        assert flag in BOX, flag
    assert "WINDOWS_DEF=64\n" in RUN and "WINDOWS_DEF=40\n" in RUN


def test_every_time_left_check_fits_its_own_guard():
    prereg = (LANE / "PREREG-p122.md").read_text(encoding="utf-8")
    assert "guard 0.75 h" in prereg and "guard 1.5 h" in prereg
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|BOX)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|BOX)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == {"FETCH", "BAKE", "BOX"}
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 1.5 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_BOX"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes
    assert {16, 25, 26, 27} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p122-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P122_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p122_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p122"), out.stdout + out.stderr
