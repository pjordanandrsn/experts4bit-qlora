"""The p131 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P131 from P130's).

`bench/p131/p131_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p131/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test and pins the lane's shape:
- P130's box (whose counters, digests, timers, runner and scoring P131's box imports) at P130's registered bytes, P117's,
  P108's and P97's boxes (which it and P130's box import) at theirs, P39's bake and calibration at SC1's, the
  prefill-graph premise test and the fetch watchdog at the bytes pinned here;
- the order: refusals, install and tripwire, the self-tests, the premise, then (and only then) the fetch, the bake, the
  processes and the reducer;
- the subject: SC2e's int4 stack built eager with one slot at the release's prefill defaults, every process alike, P1,
  P2 and the knob under test unset before it and never set by the runner (the box sets the knob per capture);
- the rule's constants, the sizes the box, the runner and the reducer share, every time-left check inside its own guard,
  and the exit codes.

`bench/p131/staged.sha256` names each file as the BOX sees it (flat in /root/p131). Regenerate it after editing any
staged file, from the repository root (bash; the three comment lines first, as P130's pin carries them):

    for f in bench/p131/p131_run.sh bench/p131/p131_box.py bench/p131/p131_reduce.py bench/p130/p130_box.py \\
             bench/p117/p117_box.py bench/p108/p108_box.py bench/p97/p97_box.py bench/p39/k8_bake.py \\
             bench/p39/calib.json tests/test_prefill_graph_gpu.py bench/common/hf_fetch_watchdog.py; do
      echo "$(sha256sum < "$f" | cut -d' ' -f1)  $(basename "$f")"; done
"""
import ast
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p131"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p131_run.sh": LANE / "p131_run.sh",
    "p131_box.py": LANE / "p131_box.py",
    "p131_reduce.py": LANE / "p131_reduce.py",
    "p130_box.py": REPO / "bench" / "p130" / "p130_box.py",
    "p117_box.py": REPO / "bench" / "p117" / "p117_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_prefill_graph_gpu.py": REPO / "tests" / "test_prefill_graph_gpu.py",
    "hf_fetch_watchdog.py": REPO / "bench" / "common" / "hf_fetch_watchdog.py",
}
RUN = (LANE / "p131_run.sh").read_text(encoding="utf-8")
REDUCE = (LANE / "p131_reduce.py").read_text(encoding="utf-8")
BOX = (LANE / "p131_box.py").read_text(encoding="utf-8")
DRIVE = (LANE / "p131_drive.sh").read_text(encoding="utf-8")
P130_BOX = (REPO / "bench" / "p130" / "p130_box.py").read_text(encoding="utf-8")


def _entries(pin=PIN):
    for line in pin.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, (f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p131 launch refuses "
                             "ON A RENTED BOX -- regenerate bench/p131/staged.sha256 (this module's docstring)")


def test_imported_boxes_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p130 = dict((n, w) for w, n in _entries(REPO / "bench" / "p130" / "staged.sha256"))
    p117 = dict((n, w) for w, n in _entries(REPO / "bench" / "p117" / "staged.sha256"))
    sc1 = dict((n, w) for w, n in _entries(REPO / "bench" / "sc1" / "staged.sha256"))
    assert mine["p130_box.py"] == p130["p130_box.py"]                    # P130's box, as P130 ran it
    assert mine["p117_box.py"] == p117["p117_box.py"] == p130["p117_box.py"]
    assert mine["p108_box.py"] == p117["p108_box.py"] and mine["p97_box.py"] == p117["p97_box.py"]
    assert mine["k8_bake.py"] == sc1["k8_bake.py"] and mine["calib.json"] == sc1["calib.json"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p131_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in DRIVE           # the arena stays on the box


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p131_reduce.py"), "--self-test"], capture_output=True, text=True)
    m = re.search(r"P131_REDUCE self-test: (\d+)/(\d+)", out.stdout)
    assert out.returncode == 0 and m and m.group(1) == m.group(2) and int(m.group(2)) >= 44, out.stdout + out.stderr
    for line in ("TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005", "MIN_GAIN, NOISY_SPAN = 1.02, 1.02",
                 "CAPTURE_FORWARDS = 5", "DECODE_TOKENS = 64", 'FLOORS = ("half", "chunk", "rev")',
                 'PHASE_B = {0: "all"}', "WARMUP = 0", 'GLUE_DEFAULT = {"mode": "1", "source": "default-allowlisted"}'):
        assert line in REDUCE, line
    # grouped-nf4-gemm#542 is not e4b CI's pin (#1625's knob is opt-in): the runner, the reducer and the fetch gate pin
    # the one commit
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    gate = (LANE / "p131_fetch_gate.py").read_text(encoding="utf-8")
    assert f"GNF4_SHA={gnf4}" in RUN and f'GNF4_SHA = "{gnf4}"' in gate, "the lane pins one grouped-nf4-gemm"
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model


def test_the_box_the_runner_and_the_reducer_share_the_registered_shape():
    assert "ORDER" not in RUN and "--p1" not in RUN and "--ref-dir" not in RUN    # every process alike, R held in memory
    # Amendment 2: the warm-up process 0 runs Phase B and a speed pass reported only; 1..PROCS are the scored processes
    assert 'for n in $(seq 0 "$PROCS"); do' in RUN and 'PB=none; [ "$n" = 0 ] && PB=all' in RUN
    assert 'PHASE_B = {0: "all"}' in REDUCE and "WARMUP = 0" in REDUCE and "procs = range(WARMUP, N + 1)" in REDUCE
    assert 'if a.phase_b == "all" and a.proc != 0:' in BOX and "if a.proc < 0:" in BOX
    assert 'rec["gpu_state"] = {"speed_start": gs0, "speed_end": gpu_state()}' in BOX and "|GPU)" in RUN
    for line in ('ARMS = ("m0", "m32", "m64")', 'TALL = ("m32", "m64")', 'FLOORS = ("half", "chunk", "rev")',
                 'OFF_ARMS = ("R", "rep") + FLOORS + ("mutant_scale",)'):
        assert line in BOX and line in REDUCE, line
    assert 'KNOB = {"m0": "0", "m32": "32", "m64": "64"}' in BOX
    assert {arm: s for s, arm in _literal(BOX, "SUBJECTS").items()} == _literal(REDUCE, "SUBJECTS")   # S32 m32, S64 m64
    for arm, kw in (("R", '{"buckets": B16}'), ("half", '{"buckets": B8}'), ("chunk", '{"buckets": B16}'),
                    ("rev", '{"buckets": B16, "reverse": True}'), ("mutant_scale", '{"buckets": B16, "mutant": "scale"}'),
                    ("S32", '{"buckets": B16}'), ("S64", '{"buckets": B16}')):
        assert f'"{arm}": {kw}' in BOX, arm
    assert ('BUCKETS = {"R": B16, "rep": B16, "half": B8, "chunk": B16, "rev": B16, "mutant_scale": B16, "S32": B16, '
            '"S64": B16}') in REDUCE
    assert 'return {"norm": layers + 1, "layer": layers, "attention": layers}' in P130_BOX     # the folds P1 takes
    assert 'return {"norm": layers + 1, "layer": layers, "attention": layers}' in REDUCE
    assert '"--decode-tokens", type=int, default=64' in BOX
    modes = dict(re.findall(r'"(reading|proof)": \{([^}]*)\}', REDUCE))
    for mode, run_line in (("proof", "PROCS=4; WINDOWS=16; CONT=32; SPEED_WINDOWS=4; ROUNDS=3; WARM=1"),
                           ("reading", "PROCS=8; WINDOWS=64; CONT=128; SPEED_WINDOWS=16; ROUNDS=12; WARM=2")):
        assert run_line in RUN, mode
        got = {k: int(v) for k, v in re.findall(r'"(\w+)": (\d+)', modes[mode])}
        want = {k.lower(): int(v) for k, v in re.findall(r"(\w+)=(\d+)", run_line)}
        assert got == {**want, "prompt": 512}, (mode, got, want)
    for flag in ('"--windows", type=int, default=64', '"--speed-windows", type=int, default=16',
                 '"--rounds", type=int, default=12', '"--warm", type=int, default=2', '"--prompt", type=int, default=512',
                 '"--cont", type=int, default=128', '"--chunk", type=int, default=512', '"--floor-chunk", type=int, default=256',
                 '"--stride", type=int, default=3072'):
        assert flag in BOX, flag


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ['finish 18;;', "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "python - <<'PYT'", "p131_reduce.py --self-test", "hf_fetch_watchdog.py --self-test",
             "python -m pytest test_prefill_graph_gpu.py", 'echo "premise ok"', 'say "fetch $MODEL @ $REV"',
             "python $W/k8_bake.py", "python $W/p131_box.py --out $W/proc_$n.json",
             "python $W/p131_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "9 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    for needle in ("from nf4_grouped import build_group_tiles_split",
                   "from int4_smallm import gemm_int4_b32_grouped_split, GROUPED_BLOCK_M",
                   "assert GROUPED_BLOCK_M == (16, 32, 64)", 'transformers.__version__ == "5.17.0"',
                   'glue_fuse.rows_cap("1") is None and glue_fuse.rows_cap("0") == 64',
                   'hasattr(serve_paged, "resolve_prefill_glue")',
                   'hot_residency.PREFILL_LEAN_DISPATCH_ENV == "E4B_PREFILL_LEAN_DISPATCH"',
                   'hot_residency.K19_PREFILL_BIG_M_ENV == "E4B_K19_PREFILL_BIG_M" and hot_residency.K19_SPLIT_ROWS_MIN == 512',
                   "callable(hot_residency._k19_prefill_big_m_env)", '"scatter" in k19 and "gather_div" in k19',
                   "import p108_box, p117_box, p130_box", "PagedModelRunner._PG_KEY == -1"):
        assert needle in trip, needle
    assert 'md.version("grouped-nf4-gemm") ==' not in trip       # #542 by capability: its merge may bump the version


def test_the_subject_is_sc2es_int4_stack_at_the_release_defaults_with_the_knob_set_by_the_box():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index("# SC2e's served stack")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS", "E4B_PAGED_PREFILL_GRAPH", "E4B_PAGED_BULK_KV",
                 "E4B_PAGED_LAST_LOGITS", "E4B_SERVE_EXP_INT4", "E4B_PAGED_FUSE_QKV", "E4B_INT4_PREFILL",
                 "E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH", "E4B_K19_PREFILL_BIG_M"):
        assert knob in unset, knob
    for knob in ("E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH", "E4B_K19_PREFILL_BIG_M"):
        assert f"{knob}=" not in RUN, knob          # never set by the runner: the box refuses any of the three in its env
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text(encoding="utf-8")
    for var in ("FOLDS", "SPEEDENV", "ROUTEENV"):                      # SC2e's stack, byte for byte
        line = next(x for x in sc1.splitlines() if x.startswith(f"{var}="))
        assert line.split("   #")[0] in RUN, var
    assert 'LEVERS="$ROUTEENV $SPEEDENV E4B_PAGED_FUSE_QKV=1"' in RUN
    assert 'ENGINE_KNOBS="E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0"' in RUN
    assert 'ENV = "E4B_K19_PREFILL_BIG_M"' in BOX
    assert 'DEFAULT_ENVS = ("E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH")' in BOX
    assert "for k in (ENV,) + DEFAULT_ENVS if" in BOX and "if set_now:" in BOX          # the box's refusal
    assert 'cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1' in BOX and "model = parts.runner.model" in BOX
    assert "bulk_kv=cfg.bulk_kv" in BOX and "last_logits=cfg.last_logits" in BOX       # the server's runner, as built
    assert "hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = True, False" in BOX
    assert "st = runner.enable_prefill_graph(T)" in BOX


def test_every_time_left_check_fits_its_own_guard():
    prereg = (LANE / "PREREG-p131.md").read_text(encoding="utf-8")
    assert "guard 1.25 h" in prereg and "guard 2.0 h" in prereg
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|PROC)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|PROC)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == {"FETCH", "BAKE", "PROC"}
    for need in prove.values():
        assert int(need) + 600 <= 1.25 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 2.0 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_PROC"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes          # disk, and P115's registered host floor
    assert {10, 16, 21, 25, 27} <= codes


def _drive(tmp_path, gate=None, sha="0" * 40):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p131-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P131_DRIVE_DRYRUN": "1"}
    if gate is not None:
        report = {"schema": "p131-fetch-gate/1", "passed": True, "refusals": [], "e4b_sha": sha,
                  "generated_at": int(time.time()) - 60, **gate}
        (tmp_path / "gate.json").write_text(json.dumps(report))
        env["P131_FETCH_GATE"] = str(tmp_path / "gate.json")
    return subprocess.run(["bash", str(LANE / "p131_drive.sh")], capture_output=True, text=True, env=env)


def test_the_driver_runs_to_its_dry_run(tmp_path):
    out = _drive(tmp_path, gate={})
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p131"), out.stdout + out.stderr
    assert json.loads((tmp_path / "p131_fetch_gate.json").read_text())["passed"] is True   # the report travels
    assert "P131_GATE_CORPUS" not in out.stdout


def test_the_gates_corpus_commit_travels_to_the_box(tmp_path):
    c = "b08601e04326c79dfdd32d625aee71d232d685c3"
    out = _drive(tmp_path, gate={"corpus": {"main": c}})
    assert out.returncode == 0 and f"P131_GATE_CORPUS={c}" in out.stdout, out.stdout + out.stderr
    assert '--gate-corpus "$GATE_CORPUS"' in RUN and 'echo "GATE_CORPUS ${GATE_CORPUS:-none}"' in RUN
    assert '"corpus": p130_box.corpus_commits()' in BOX and "def corpus_report(recs: dict, gate_corpus" in REDUCE


def test_the_driver_refuses_without_a_passing_fresh_gate_for_the_launch_commit(tmp_path):
    for gate, sha in ((None, "0" * 40), ({}, "1" * 40), ({"passed": False, "refusals": ["model x: 403"]}, "0" * 40),
                      ({"generated_at": int(time.time()) - 25 * 3600}, "0" * 40)):
        out = _drive(tmp_path, gate=gate, sha=sha)
        assert out.returncode == 78 and "refusing" in out.stdout and "DRYRUN" not in out.stdout, (gate, out.stdout)


def _literal(src, name):
    """The module-level literal bound to ``name``, also from a tuple assignment (``MODEL, REV = ...``)."""
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if getattr(t, "id", None) == name:
                return ast.literal_eval(node.value)
            names = [getattr(e, "id", None) for e in getattr(t, "elts", [])]
            if name in names:
                return ast.literal_eval(node.value)[names.index(name)]
    raise AssertionError(f"{name} is not a literal assignment")


def test_the_fetch_gate_resolves_what_the_box_fetches():
    out = subprocess.run([sys.executable, str(LANE / "p131_fetch_gate.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (20 cases)" in out.stdout, out.stdout + out.stderr
    gate = (LANE / "p131_fetch_gate.py").read_text(encoding="utf-8")
    fetch = RUN[RUN.index("hf_fetch_watchdog.py --repo"):RUN.index("> logs/fetch.log")]
    assert tuple(re.findall(r"--allow '?([^'\s]+)'?", fetch)) == _literal(gate, "ALLOW")          # what the download takes
    pip = RUN[RUN.index('"git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA"'):RUN.index("|| { tail -4 logs/pip_e4b.log")]
    assert tuple(shlex.split(pip.replace("\\\n", " "))[1:]) == _literal(gate, "PYPI")           # what pip installs
    assert _literal(gate, "GNF4_SHA") == re.search(r"GNF4_SHA=([0-9a-f]{40})", RUN).group(1)
    assert f"MODEL={_literal(gate, 'MODEL')}; REV={_literal(gate, 'REV')}" in RUN
    assert _literal(gate, "HUB_RANGE") == ((1, 31), (2, 0)) and "\nMAX_AGE_S = 24 * 3600\n" in gate
    assert "p117_box.windows" in gate and 'load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")' in (
        REPO / "bench" / "p117" / "p117_box.py").read_text(encoding="utf-8")
