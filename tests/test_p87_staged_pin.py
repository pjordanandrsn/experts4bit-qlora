"""The p87 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P87).

`bench/p87/p87_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p87/staged.sha256`. That guard
runs on the CONTROLLER after a box is rented; this test runs the same comparison in CI, where it costs nothing.

It mirrors the driver's staging `case`: P58's harness pieces (bench/p39, bench/p42's hook), P42's census parser and
the premise test (tests/test_k19_row_exact_gpu.py), all referenced unchanged. It also:
- runs the reducer's self-test (17 cases);
- pins the parts of the runner the registration depends on: the gnf4 pin, the refusals and the premise before any
  fetch, the proof, and the arms with their order and settings.
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p87"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p87_run.sh": LANE / "p87_run.sh",
    "p87_reduce.py": LANE / "p87_reduce.py",
    "p42_reduce.py": REPO / "bench" / "p42" / "p42_reduce.py",
    "test_k19_row_exact_gpu.py": REPO / "tests" / "test_k19_row_exact_gpu.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "hook/usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
}


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v and len(v) < 40 else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = SOURCES[name]
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p87 launch refuses ON A RENTED BOX."
    )


def test_every_pinned_name_is_staged_by_the_driver_and_resolves_the_same_way():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p87_drive.sh").read_text()
    for name, src in SOURCES.items():
        rel = src.relative_to(REPO)
        if rel.parts[0] == "tests":
            assert f"{name}) src=\"$TESTS/$name\"" in driver and "$TESTS/test_k19_row_exact_gpu.py" in driver, name
            continue
        lane_dir = rel.parts[1]
        if lane_dir == "p87":
            assert f"{name}|" in driver or f"|{name})" in driver, name
        else:
            var = {"p39": "$P39", "p42": "$P42"}[lane_dir]
            assert f"{var}/" in driver, name


def test_the_harness_is_p86s_pinned_bytes():
    """The speed arms run P86's (and P58's) harness bytes, so their OFF steps sit beside P86's."""
    p86 = {name: want for want, name in _entries(REPO / "bench" / "p86" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    for name in ("step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py", "p42_reduce.py"):
        assert mine[name] == p86[name], name


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p87_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (17 cases)" in out.stdout


def test_the_pins_the_refusals_and_the_premise_come_before_any_fetch():
    run = (LANE / "p87_run.sh").read_text()
    assert "GNF4_SHA=3351c9d70e54a8ace33fa29ee7754f616332189f" in run
    assert "MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    install = run.index('say "install e4b @')
    for refusal in ("finish 15;", "finish 13;"):                                  # class, disk
        assert run.index(refusal) < install, refusal
    fetch = run.index('say "fetch $MID @ $REV"')
    premise = run.index("python -m pytest test_k19_row_exact_gpu.py")
    assert install < premise < run.index('if [ "${P87_PROVE:-0}" = 1 ]; then') < fetch
    assert "finish 25" in run[premise:fetch]
    trip = run[install:premise]
    for must in ("gemm_int4_b32_grouped_smallm", '"E4B_INT4_GROUPED_SMALLM" in inspect.getsource(hr)',
                 'hasattr(hr, "_collapsed_grouping")', "gemv_fused_reduce_default() is False", "dump_artifact_dir"):
        assert must in trip, must


def test_the_proof_compiles_k19s_contract_on_the_card_and_fetches_no_model():
    run = (LANE / "p87_run.sh").read_text()
    prove = run.index('if [ "${P87_PROVE:-0}" = 1 ]; then')
    block = run[prove:run.index('say "fetch $MID @ $REV"')]
    assert "git -C $W/gnf4 checkout -q $GNF4_SHA" in block
    assert "TRITON_INTERPRET=0" in block and "test_int4_grouped_smallm_interp.py test_int4_smallm_interp.py" in block
    assert "finish 23" in block and ": > PROVED; finish 0" in block
    assert "for v in P87_PROVE; do" in (LANE / "p87_drive.sh").read_text()


def test_the_arms_run_in_the_registered_order_with_their_settings():
    run = (LANE / "p87_run.sh").read_text()
    steps = ('can_run 900 b${B}_off && { speed 0 $B "" 1;',
             'can_run 900 b${B}_on && { speed 1 $B "" 1;',
             "k8 build watch E4B_CALIB_LAYERS_PER_PASS=10 E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact E4B_INT4_GROUPED_SMALLM=0;",
             "k8 off nowatch E4B_INT4_ARTIFACT_DIR=$W/artifact E4B_INT4_EXPECTED_FINGERPRINT=$FP E4B_INT4_GROUPED_SMALLM=0;",
             "k8 on nowatch E4B_INT4_ARTIFACT_DIR=$W/artifact E4B_INT4_EXPECTED_FINGERPRINT=$FP E4B_INT4_GROUPED_SMALLM=1;",
             'can_run 900 b${B}_on_r2 && { speed 1 $B _r2 "";',
             'can_run 900 b${B}_off_r2 && { speed 0 $B _r2 "";',
             "python $W/p87_reduce.py --dir $W --out $W/verdict.json")
    assert all(run.count(s) == 1 for s in steps), [s for s in steps if run.count(s) != 1]
    order = [run.index(s) for s in steps]
    assert order == sorted(order), order
    assert run.count("for B in 16 1; do") == 2
    assert 'SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"' in run
    assert ('LICENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 '
            'E4B_CALIB_SOURCE=c4 E4B_CALIB_NSEQ=$NSEQ $FOLDS"') in run
    assert 'FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"' in run
    assert ('K8ARGS="--placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 '
            '--ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext"') in run
    arm = run[run.index("speed(){"):run.index("first_chunk_watchdog(){")]
    assert "--b1d-loop graph --b1d-timed --fuse-qkv" in arm and "--replay-profile-out" in arm
    assert "E4B_INT4_GROUPED_SMALLM=$K" in arm
    assert "unset E4B_SERVE_EXP_INT4" in run and "E4B_INT4_GROUPED_SMALLM\n" in run   # every lever unset up front


def test_the_driver_runs_to_its_dry_run(tmp_path):
    """The whole controller script parses and reaches its dry run (P83's apostrophe lesson)."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p87-dry",
           "E4B_RENT_DEADLINE_EPOCH": "1", "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P87_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p87_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("DRYRUN stage -> root@h:/root/p87"), out.stdout
    assert not out.stderr.strip(), out.stderr
