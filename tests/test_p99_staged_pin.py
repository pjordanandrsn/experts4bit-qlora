"""The p99 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P99).

`bench/p99/p99_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p99/staged.sha256`; this test runs
the same comparison in CI. It also runs the reducer's self-test (11 cases) and pins the diagnosis: P98's measurement
and bake staged at P98's registered bytes, the seven arms' configurations (P98's arm g repeated; OLMoE; K25 off; one
bucket) and their order, the two models at their revisions, the premise before any fetch, and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p99"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p99_run.sh": LANE / "p99_run.sh",
    "p99_box.py": LANE / "p99_box.py",
    "p99_reduce.py": LANE / "p99_reduce.py",
    "p98_box.py": REPO / "bench" / "p98" / "p98_box.py",
    "p98_bake.py": REPO / "bench" / "p98" / "p98_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_linear_state_gpu.py": REPO / "tests" / "test_linear_state_gpu.py",
    "test_hybrid_decode_graphs_gpu.py": REPO / "tests" / "test_hybrid_decode_graphs_gpu.py",
    "test_linear_state_graph_gpu.py": REPO / "tests" / "test_linear_state_graph_gpu.py",
}
RUN = (LANE / "p99_run.sh").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p99 launch refuses ON A RENTED BOX"


def test_p98s_measurement_runs_at_p98s_registered_bytes():
    p98 = dict((n, w) for w, n in _entries(REPO / "bench" / "p98" / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    for name in ("p98_box.py", "p98_bake.py", "calib.json", "test_linear_state_gpu.py",
                 "test_hybrid_decode_graphs_gpu.py", "test_linear_state_graph_gpu.py"):
        assert mine[name] == p98[name], name


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p99_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p99_run.sh":
            assert name in staged, name
    for tag in ("work_qwen", "work_olmoe"):
        assert f"--include '{tag}/bake.json' --exclude '{tag}/*'" in driver   # the arenas stay on the box


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p99_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (11 cases)" in out.stdout, out.stdout + out.stderr


def test_the_arms_are_the_registered_diagnosis_in_order():
    assert 'ARMS=${P99_ARMS:-d0g d1g d1e d2g d2e d3g d3e}' in RUN
    arm = RUN[RUN.index("arm(){"):RUN.index("for N in $ARMS; do arm $N")]
    assert "d0g) A=g;;" in arm                                                         # P98's arm g, unchanged
    assert 'd1g|d1e) A=${NAME:2:1}; MID=$OLMOE; R=$OREV; ARENA=$W/work_olmoe/nf4.arena;;' in arm
    assert 'd2g|d2e) A=${NAME:2:1}; ENVS="E4B_NF4_GROUPED_SMALLM=0";;' in arm
    assert 'd3g|d3e) A=${NAME:2:1}; XB="--buckets 16";;' in arm
    assert "python $W/p99_box.py --arm $A" in arm
    assert "MODEL=Qwen/Qwen3.6-35B-A3B; REV=995ad96eacd98c81ed38be0c5b274b04031597b0" in RUN
    assert "OLMOE=allenai/OLMoE-1B-7B-0924-Instruct; OREV=7f1c97f440f06ce36705e4f2b843edb5925f4498" in RUN
    assert "E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE" in RUN                         # levers start unset


def test_the_premise_precedes_the_fetch_and_the_bakes_precede_the_arms():
    premise = RUN.index("python -m pytest test_linear_state_gpu.py test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py")
    assert RUN.index('say "install e4b @') < premise < RUN.index('fetch qwen "$MODEL" "$REV"') \
        < RUN.index('fetch olmoe "$OLMOE" "$OREV"') < RUN.index('bake qwen "$MODEL" "$REV"') \
        < RUN.index('bake olmoe "$OLMOE" "$OREV"') < RUN.index("for N in $ARMS; do arm $N") \
        < RUN.index("python $W/p99_reduce.py --dir $W --out $W/verdict.json")
    assert 'grep -q "4 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    assert "PROVE" not in RUN                                                          # the guard is 1 h: no proving run


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p99-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P99_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p99_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p99"), out.stdout + out.stderr
