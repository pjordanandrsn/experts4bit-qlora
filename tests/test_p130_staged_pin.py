"""The p130 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P130 from P117's).

`bench/p130/p130_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p130/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test and pins the lane's shape:
- P117's, P108's and P97's boxes (whose helpers P130's box imports) at their registered bytes, P39's bake and calibration
  at SC1's, the prefill-graph premise test and the fetch watchdog at the bytes pinned here;
- the order: refusals, install and tripwire, the self-tests, the premise, then (and only then) the fetch, the bake, the
  processes and the reducer;
- the subject: SC2e's int4 stack built eager with one slot, P1 set per process in the registered order, both knobs unset
  before it;
- the rule's constants, the sizes the box, the runner and the reducer share, every time-left check inside its own guard,
  and the exit codes.
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
LANE = REPO / "bench" / "p130"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p130_run.sh": LANE / "p130_run.sh",
    "p130_box.py": LANE / "p130_box.py",
    "p130_reduce.py": LANE / "p130_reduce.py",
    "p117_box.py": REPO / "bench" / "p117" / "p117_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_prefill_graph_gpu.py": REPO / "tests" / "test_prefill_graph_gpu.py",
    "hf_fetch_watchdog.py": REPO / "bench" / "common" / "hf_fetch_watchdog.py",
}
RUN = (LANE / "p130_run.sh").read_text()
REDUCE = (LANE / "p130_reduce.py").read_text()
BOX = (LANE / "p130_box.py").read_text()
DRIVE = (LANE / "p130_drive.sh").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p130 launch refuses ON A RENTED BOX"


def test_imported_boxes_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    p117 = dict((n, w) for w, n in _entries(REPO / "bench" / "p117" / "staged.sha256"))
    sc1 = dict((n, w) for w, n in _entries(REPO / "bench" / "sc1" / "staged.sha256"))
    assert mine["p117_box.py"] == p117["p117_box.py"]
    assert mine["p108_box.py"] == p117["p108_box.py"] and mine["p97_box.py"] == p117["p97_box.py"]
    assert mine["k8_bake.py"] == sc1["k8_bake.py"] and mine["calib.json"] == sc1["calib.json"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p130_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in DRIVE           # the arena and R's log-probs stay on the box


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p130_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (40 cases)" in out.stdout, out.stdout + out.stderr
    for line in ("TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005", "P2_MARGIN, P1_MIN_GAIN, NOISY_SPAN = 1.01, 1.03, 1.02",
                 "CAPTURE_FORWARDS = 5", 'FLOORS = ("half", "chunk", "rev")', 'SUBJECT = "P1"',
                 "PAIRS = ((1, 2), (4, 3), (5, 6), (8, 7))", 'PHASE_B = {1: "off", 2: "on"}',
                 'CEILING_MS = {"p1": 5.47, "p2": 1.34, "both": 6.80}'):
        assert line in REDUCE, line
    gnf4 = re.search(r'GNF4_SHA = "([0-9a-f]{40})"', REDUCE).group(1)
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert f"GNF4_SHA={gnf4}" in RUN and f"grouped-nf4-gemm.git@{gnf4}" in ci, "the lane pins e4b CI's grouped-nf4-gemm"
    for model, rev in re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', REDUCE):
        assert f"MODEL={model}; REV={rev}" in RUN, model


def test_the_box_the_runner_and_the_reducer_share_the_registered_shape():
    assert 'ORDER="0 1 1 0 0 1 1 0"' in RUN and "ORDER = (0, 1, 1, 0, 0, 1, 1, 0)" in REDUCE
    for line in ('FLOORS = ("half", "chunk", "rev")', 'OFF_ARMS = ("R", "rep") + FLOORS + ("mutant_scale",)', 'SUBJECT = "P1"',
                 'GRAPHS = ("p2_off", "p2_on")'):
        assert line in BOX and line in REDUCE, line
    for arm, kw in (("R", '{"buckets": B16}'), ("half", '{"buckets": B8}'), ("chunk", '{"buckets": B16}'),
                    ("rev", '{"buckets": B16, "reverse": True}'), ("mutant_scale", '{"buckets": B16, "mutant": "scale"}'),
                    ("P1", '{"buckets": B16}')):
        assert f'"{arm}": {kw}' in BOX, arm
    assert 'BUCKETS = {"R": B16, "rep": B16, "half": B8, "chunk": B16, "rev": B16, "mutant_scale": B16, "P1": B16}' in REDUCE
    assert 'return {"norm": layers + 1, "layer": layers, "attention": layers}' in BOX
    assert 'return {"norm": layers + 1, "layer": layers, "attention": layers}' in REDUCE
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
             "python - <<'PYT'", "p130_reduce.py --self-test", "hf_fetch_watchdog.py --self-test",
             "python -m pytest test_prefill_graph_gpu.py", 'echo "premise ok"', 'say "fetch $MODEL @ $REV"',
             "python $W/k8_bake.py", "python $W/p130_box.py --out $W/proc_$n.json",
             "python $W/p130_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "9 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    for needle in ('md.version("grouped-nf4-gemm") == "0.45.0"', 'transformers.__version__ == "5.17.0"',
                   'glue_fuse.rows_cap("1") is None and glue_fuse.rows_cap("0") == 64',
                   'hot_residency.PREFILL_LEAN_DISPATCH_ENV == "E4B_PREFILL_LEAN_DISPATCH"',
                   '"scatter" in k19 and "gather_div" in k19', "import p108_box, p117_box", "PagedModelRunner._PG_KEY == -1"):
        assert needle in trip, needle


def test_the_subject_is_sc2es_int4_stack_with_p1_set_per_process():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index("# SC2e's served stack")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS", "E4B_PAGED_PREFILL_GRAPH", "E4B_PAGED_BULK_KV",
                 "E4B_PAGED_LAST_LOGITS", "E4B_SERVE_EXP_INT4", "E4B_PAGED_FUSE_QKV", "E4B_INT4_PREFILL",
                 "E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH"):
        assert knob in unset, knob
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
    for var in ("FOLDS", "SPEEDENV", "ROUTEENV"):                      # SC2e's stack, byte for byte
        line = next(x for x in sc1.splitlines() if x.startswith(f"{var}="))
        assert line.split("   #")[0] in RUN, var
    assert 'LEVERS="$ROUTEENV $SPEEDENV E4B_PAGED_FUSE_QKV=1"' in RUN
    assert 'ENGINE_KNOBS="E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0"' in RUN
    assert 'P1ENV=""; [ "$P1" = 1 ] && P1ENV="E4B_FUSE_PREFILL_GLUE=1"' in RUN
    assert 'PB=none; [ "$n" = 1 ] && PB=off; [ "$n" = 2 ] && PB=on' in RUN
    assert 'cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1' in BOX and "model = parts.runner.model" in BOX
    assert "bulk_kv=cfg.bulk_kv" in BOX and "last_logits=cfg.last_logits" in BOX       # the server's runner, as built
    assert "hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = True, False" in BOX
    assert "st = runner.enable_prefill_graph(T)" in BOX and "p2_env != \"0\"" in BOX


def test_every_time_left_check_fits_its_own_guard():
    prereg = (LANE / "PREREG-p130.md").read_text()
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
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p130-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P130_DRIVE_DRYRUN": "1"}
    if gate is not None:
        report = {"schema": "p130-fetch-gate/1", "passed": True, "refusals": [], "e4b_sha": sha,
                  "generated_at": int(time.time()) - 60, **gate}
        (tmp_path / "gate.json").write_text(json.dumps(report))
        env["P130_FETCH_GATE"] = str(tmp_path / "gate.json")
    return subprocess.run(["bash", str(LANE / "p130_drive.sh")], capture_output=True, text=True, env=env)


def test_the_driver_runs_to_its_dry_run(tmp_path):
    out = _drive(tmp_path, gate={})
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p130"), out.stdout + out.stderr
    assert json.loads((tmp_path / "p130_fetch_gate.json").read_text())["passed"] is True   # the report travels


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
    out = subprocess.run([sys.executable, str(LANE / "p130_fetch_gate.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (20 cases)" in out.stdout, out.stdout + out.stderr
    gate = (LANE / "p130_fetch_gate.py").read_text()
    fetch = RUN[RUN.index("hf_fetch_watchdog.py --repo"):RUN.index("> logs/fetch.log")]
    assert tuple(re.findall(r"--allow '?([^'\s]+)'?", fetch)) == _literal(gate, "ALLOW")          # what the download takes
    pip = RUN[RUN.index('"git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA"'):RUN.index("|| { tail -4 logs/pip_e4b.log")]
    assert tuple(shlex.split(pip.replace("\\\n", " "))[1:]) == _literal(gate, "PYPI")           # what pip installs
    assert _literal(gate, "GNF4_SHA") == re.search(r"GNF4_SHA=([0-9a-f]{40})", RUN).group(1)
    assert f"MODEL={_literal(gate, 'MODEL')}; REV={_literal(gate, 'REV')}" in RUN
    assert _literal(gate, "HUB_RANGE") == ((1, 31), (2, 0)) and "\nMAX_AGE_S = 24 * 3600\n" in gate
    assert "p117_box.windows" in gate and 'load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")' in (
        REPO / "bench" / "p117" / "p117_box.py").read_text()
