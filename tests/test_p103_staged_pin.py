"""The p103 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P103).

`bench/p103/p103_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p103/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test (20 cases) and pins the lane's shape:
- P98's measurement, bake and reducer at P98's registered bytes, and the three hybrid GPU premise tests;
- the kernel pins (flash-linear-attention 0.5.2, causal-conv1d 1.7.0) with torch held at the image's version;
- the phases t, f, fc in that order, each with an engagement probe, phase t asserted kernel-free;
- the premise before any fetch, and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p103"
P98 = REPO / "bench" / "p98"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p103_run.sh": LANE / "p103_run.sh",
    "p103_reduce.py": LANE / "p103_reduce.py",
    "p98_box.py": P98 / "p98_box.py",
    "p98_bake.py": P98 / "p98_bake.py",
    "p98_reduce.py": P98 / "p98_reduce.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_linear_state_gpu.py": REPO / "tests" / "test_linear_state_gpu.py",
    "test_hybrid_decode_graphs_gpu.py": REPO / "tests" / "test_hybrid_decode_graphs_gpu.py",
    "test_linear_state_graph_gpu.py": REPO / "tests" / "test_linear_state_graph_gpu.py",
}
RUN = (LANE / "p103_run.sh").read_text()
REDUCE = (LANE / "p103_reduce.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p103 launch refuses ON A RENTED BOX"


def test_p98s_measurement_runs_at_p98s_registered_bytes():
    p98 = dict((n, w) for w, n in _entries(P98 / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    for name in ("p98_box.py", "p98_bake.py", "p98_reduce.py", "calib.json", "test_linear_state_gpu.py",
                 "test_hybrid_decode_graphs_gpu.py", "test_linear_state_graph_gpu.py"):
        assert mine[name] == p98[name], name


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p103_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p103_run.sh":
            assert name in staged, name
    assert "--include 'work/' --include 'work/bake.json' --exclude 'work/*'" in driver   # the arena stays on the box


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p103_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (20 cases)" in out.stdout, out.stdout + out.stderr


def test_the_kernels_are_pinned_and_torch_is_held():
    assert 'FLA_PIN="flash-linear-attention==0.5.2"; CC_PIN="causal-conv1d==1.7.0"' in RUN
    assert 'PINS = {"flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0"}' in REDUCE
    assert 'echo "torch==$TORCH_PIN" > $W/constraints.txt' in RUN
    assert RUN.count("-c $W/constraints.txt") == 2                     # both pip invocations in pipx
    assert "assert not any(mods.values())" in RUN                      # phase t is transformers' torch path
    assert "USE_HUB_KERNELS" in RUN                                     # unset with the serving levers
    assert "RECOMMEND_MIN = 1.05" in REDUCE


def test_the_phases_run_in_the_registered_order():
    assert 'PHASES=${P103_PHASES:-t f fc}' in RUN
    loop = RUN[RUN.index("for P in $PHASES; do"):RUN.index('echo "kernel phases:$KSTATE"')]
    assert 'f) kernel_phase f "$FLA_PIN" || continue;;' in loop
    assert 'case "$KSTATE" in *f:ok*) ;;' in loop and 'kernel_phase fc "$CC_PIN" || continue;;' in loop
    assert "for A in g e p; do arm $A $P; rec $?; done" in loop
    assert "python $W/p98_box.py --arm $A" in RUN and "--out $W/arm_${A}_$P.json" in RUN
    kp = RUN[RUN.index("kernel_phase(){"):RUN.index('if [ "$PROVE" = 1 ]; then')]
    assert kp.index("install_kernels $P") < kp.index("probe $P") < kp.index("premise $P")


def test_the_premise_precedes_the_fetch_and_the_bake_precedes_the_arms():
    assert RUN.index('say "install e4b @') < RUN.index("probe t >/dev/null; premise t") < RUN.index('say "fetch $MODEL @ $REV"') \
        < RUN.index('say "bake the NF4 arena"') < RUN.index("for P in $PHASES; do") \
        < RUN.index("python $W/p103_reduce.py --dir $W --out $W/verdict.json")
    assert 'grep -q "4 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    prove = RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("rc_any=0")]
    assert 'kernel_phase f "$FLA_PIN"; kernel_phase fc "$CC_PIN"' in prove and ": > PROVED; finish 0" in prove


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p103-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P103_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p103_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p103"), out.stdout + out.stderr
