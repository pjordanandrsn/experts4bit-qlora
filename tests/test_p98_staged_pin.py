"""The p98 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P98).

`bench/p98/p98_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p98/staged.sha256`; this test runs
the same comparison in CI. It also runs the reducer's self-test (19 cases) and pins the registration: the kernel pin
(e4b CI's grouped-nf4-gemm), transformers 5.17.0, the model at its revision in the runner, the bake and the reducer,
the shape the reducer reads against the harness's defaults, the premise (three GPU test files, four tests, none
skipped) and the proving run before the fetch, the order (fetch, bake, arms g e p, reduce), rehearsal flags never in
the runner, and the exit codes.
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p98"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p98_run.sh": LANE / "p98_run.sh",
    "p98_box.py": LANE / "p98_box.py",
    "p98_bake.py": LANE / "p98_bake.py",
    "p98_reduce.py": LANE / "p98_reduce.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_linear_state_gpu.py": REPO / "tests" / "test_linear_state_gpu.py",
    "test_hybrid_decode_graphs_gpu.py": REPO / "tests" / "test_hybrid_decode_graphs_gpu.py",
    "test_linear_state_graph_gpu.py": REPO / "tests" / "test_linear_state_graph_gpu.py",
}
RUN = (LANE / "p98_run.sh").read_text()
REDUCE = (LANE / "p98_reduce.py").read_text()
BOX = (LANE / "p98_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _const(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p98 launch refuses ON A RENTED BOX"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p98_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    assert 'for v in P98_PROVE; do' in driver
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p98_run.sh":
            assert name in staged, name
    assert "--include 'work/bake.json' --exclude 'work/*'" in driver          # the arena stays on the box


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p98_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (19 cases)" in out.stdout, out.stdout + out.stderr


def test_the_stack_is_e4b_cis_kernel_pin_on_transformers_5_17_with_the_graph_path():
    pin = re.search(r"^GNF4_SHA=([0-9a-f]{40})\b", RUN, re.M).group(1)
    # CI's kernel pin when the lane registered: v0.34.1. CI moved to v0.35.0 in 0.42.0; a frozen lane keeps its own.
    assert pin == "34da93d6fe8d2a401b7001705658ce00b2b18213"
    assert '"transformers==5.17.0"' in RUN and 'assert transformers.__version__ == "5.17.0"' in RUN
    assert 'hasattr(linear_state, "_bucket_selector")' in RUN and '"_index"' in RUN       # #907 and #908 installed
    assert 'hasattr(fp8_kv, "fp8_kv_append_bt1")' in RUN                                  # the buckets' fused append
    assert "E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_PLACEMENT" in RUN                # levers start unset


def test_the_model_revision_and_shape_agree_across_runner_bake_harness_and_reducer():
    subj, shape = _const(REDUCE, "SUBJECT"), _const(REDUCE, "SHAPE")
    assert f"MODEL={subj['model']}; REV={subj['revision']}" in RUN
    p97 = (REPO / "bench" / "p97" / "p97_run.sh").read_text()
    assert f"SUBJ={subj['model']}; SUBJ_REV={subj['revision']}" in p97        # the checkpoint P97 read
    assert (subj["pool"], subj["attn"], subj["linear"]) == (10, 10, 30)
    assert shape == {"n": 16, "prompt": 256, "new_base": 48, "new_step": 4, "b1_new": 96, "warm_steps": 3,
                     "buckets": [1, 2, 4, 8, 16]}
    for flag, key in (("--n", "n"), ("--prompt", "prompt"), ("--new-base", "new_base"), ("--new-step", "new_step"),
                      ("--b1-new", "b1_new"), ("--warm-steps", "warm_steps")):
        assert f'ap.add_argument("{flag}", type=int, default={shape[key]})' in BOX, flag
    assert 'ap.add_argument("--buckets", default="1,2,4,8,16")' in BOX
    assert 'ap.add_argument("--placement", default="all-vram"' in BOX
    assert '--revision "$REV" --work $W/work $BAKE_EXTRA' in RUN                # the bake is pinned to the revision


def test_rehearsal_flags_never_reach_the_registered_run():
    for flag in ("--offload", "--stand-in-attention", "--placement solver"):
        assert flag not in RUN, flag
    rehearsal = RUN[RUN.index('if [ "$REHEARSAL" != 0 ]'):RUN.index("# ---- staged pieces")]
    for knob in ('[ -n "$MODEL_DIR" ]', '[ -n "$BOX_EXTRA" ]', '[ -n "$BAKE_EXTRA" ]', '[ "$ALLOW_SKIP" != 0 ]',
                 '[ "$GPU_CLASS" != 5090 ]', '[ "$MIN_DISK_GB" != 170 ]', '[ "$ARMS" != "g e p" ]'):
        assert knob in rehearsal, knob


def test_the_premise_and_the_proof_precede_the_fetch_and_the_order_is_bake_then_g_e_p():
    install = RUN.index('say "install e4b @')
    assert RUN.index("finish 15;") < install and RUN.index("finish 13;") < install
    premise = RUN.index("python -m pytest test_linear_state_gpu.py test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py")
    prove = RUN.index('if [ "$PROVE" = 1 ]; then')
    fetch = RUN.index('say "fetch $MODEL @ $REV"')
    assert install < premise < prove < fetch
    assert RUN.index(": > PROVED; finish 0", prove) < fetch
    assert 'grep -q "4 passed" && ! echo "$LASTL" | grep -q skipped' in RUN   # four tests, none skipped
    assert fetch < RUN.index('say "bake the NF4 arena"') < RUN.index("for A in $ARMS; do arm $A") \
        < RUN.index("python $W/p98_reduce.py --dir $W --out $W/verdict.json")
    assert 'ARMS=${P98_ARMS:-g e p}' in RUN


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)} | {26}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p98-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P98_DRIVE_DRYRUN": "1", "P98_PROVE": "1"}
    out = subprocess.run(["bash", str(LANE / "p98_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p98"), out.stdout + out.stderr
    assert "P98_PROVE=1" in out.stdout
