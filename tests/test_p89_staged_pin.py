"""The p89 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P89).

`bench/p89/p89_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p89/staged.sha256`. That guard
runs on the CONTROLLER after a box is rented; this test runs the same comparison in CI, where it costs nothing.

It mirrors the driver's staging `case`: P58's harness pieces (bench/p39, bench/p42's hook), P42's census parser and
the two premise tests (tests/test_k19_row_exact_gpu.py, tests/test_k23_lean_glue_gpu.py), all referenced unchanged.
It also:
- runs the reducer's self-test (14 cases);
- pins the parts of the runner the registration depends on: the gnf4 pin (a real 40-char sha, K23's merge), the
  refusals and the premise before any fetch, the proof, and the arms with their order and settings.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p89"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p89_run.sh": LANE / "p89_run.sh",
    "p89_reduce.py": LANE / "p89_reduce.py",
    "p42_reduce.py": REPO / "bench" / "p42" / "p42_reduce.py",
    "test_k19_row_exact_gpu.py": REPO / "tests" / "test_k19_row_exact_gpu.py",
    "test_k23_lean_glue_gpu.py": REPO / "tests" / "test_k23_lean_glue_gpu.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "hook/usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
}
RUN = (LANE / "p89_run.sh").read_text()


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
        f"Re-pin it, or the next p89 launch refuses ON A RENTED BOX."
    )


def test_every_pinned_name_is_staged_by_the_driver_and_resolves_the_same_way():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p89_drive.sh").read_text()
    assert 'test_k19_row_exact_gpu.py|test_k23_lean_glue_gpu.py) src="$TESTS/$name";;' in driver
    for name, src in SOURCES.items():
        rel = src.relative_to(REPO)
        if rel.parts[0] == "tests":
            assert f"$TESTS/{name}" in driver, name
            continue
        lane_dir = rel.parts[1]
        if lane_dir == "p89":
            assert f"{name}|" in driver or f"|{name})" in driver, name
        else:
            var = {"p39": "$P39", "p42": "$P42"}[lane_dir]
            assert f"{var}/" in driver, name


def test_the_harness_is_p88s_pinned_bytes():
    """The arms run P88's (P86's, P58's) harness bytes, so OFF's step sits beside P88's ON (both are K19's route)."""
    p88 = {name: want for want, name in _entries(REPO / "bench" / "p88" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    for name in ("step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py", "p42_reduce.py"):
        assert mine[name] == p88[name], name


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p89_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (14 cases)" in out.stdout


def test_the_gnf4_pin_is_a_real_sha():
    m = re.search(r"^GNF4_SHA=(\S+)", RUN, re.M)
    assert m and re.fullmatch(r"[0-9a-f]{40}", m.group(1)), f"GNF4_SHA is not a 40-char sha: {m and m.group(1)}"


def test_the_pins_the_refusals_and_the_premise_come_before_any_fetch():
    assert "MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN
    install = RUN.index('say "install e4b @')
    for refusal in ("finish 15;", "finish 13;"):                                  # class, disk
        assert RUN.index(refusal) < install, refusal
    assert "finish 18" not in RUN                                                 # no CPU floor: no calibration build
    fetch = RUN.index('say "fetch $MID @ $REV"')
    premise = RUN.index("python -m pytest test_k19_row_exact_gpu.py test_k23_lean_glue_gpu.py")
    assert install < premise < RUN.index('if [ "${P89_PROVE:-0}" = 1 ]; then') < fetch
    assert "finish 25" in RUN[premise:fetch]
    trip = RUN[install:premise]
    for must in ('(_p["block_n"].default, _p["kc"].default) == (32, 256)', '{"scatter", "gather_div"} <= set(_p)',
                 '{"lean", "sorted_ids"} <= set(inspect.signature(build_group_tiles_fused).parameters)',
                 '"x_tokens" in inspect.signature(hr._fused_over_stack).parameters',
                 'hr._k19_mode_env() == "auto"', "hr._lean_glue_env() is False", "gemv_fused_reduce_default() is False"):
        assert must in trip, must


def test_the_proof_compiles_the_contracts_on_the_card_and_fetches_no_model():
    prove = RUN.index('if [ "${P89_PROVE:-0}" = 1 ]; then')
    block = RUN[prove:RUN.index('say "fetch $MID @ $REV"')]
    assert "git -C $W/gnf4 checkout -q $GNF4_SHA" in block
    assert "TRITON_INTERPRET=0" in block and "test_int4_grouped_smallm_interp.py test_int4_smallm_interp.py" in block
    assert 'test_int4_b32.py -k "lean or fused_tile"' in block
    assert block.count("finish 23") == 2 and ": > PROVED; finish 0" in block
    assert "for v in P89_PROVE; do" in (LANE / "p89_drive.sh").read_text()


def test_the_arms_run_in_the_registered_order_with_their_settings():
    steps = ('can_run 900 b16_off && { speed 0 "" 1;', 'can_run 900 b16_on && { speed 1 "" 1;',
             'can_run 900 b16_on_r2 && { speed 1 _r2 "";', 'can_run 900 b16_off_r2 && { speed 0 _r2 "";',
             "python $W/p89_reduce.py --dir $W --out $W/verdict.json")
    assert all(RUN.count(s) == 1 for s in steps), [s for s in steps if RUN.count(s) != 1]
    order = [RUN.index(s) for s in steps]
    assert order == sorted(order), order
    assert 'SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"' in RUN
    assert 'FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"' in RUN
    arm = RUN[RUN.index("speed(){"):RUN.index("rc_any=0;")]
    assert "--batch 16 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --fuse-qkv" in arm
    assert "--replay-profile-out" in arm and "E4B_INT4_LEAN_GLUE=$L" in arm
    assert "E4B_INT4_GROUPED_SMALLM" not in arm                                    # K19 at its licensed default (auto)
    assert "k8 " not in RUN and "K8ARGS" not in RUN                                # bit-identical: no K8 arm
    assert "unset E4B_SERVE_EXP_INT4" in RUN and "E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE\n" in RUN


def test_the_driver_runs_to_its_dry_run(tmp_path):
    """The whole controller script parses and reaches its dry run (P83's apostrophe lesson)."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p89-dry",
           "E4B_RENT_DEADLINE_EPOCH": "1", "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P89_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p89_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("DRYRUN stage -> root@h:/root/p89"), out.stdout
    assert not out.stderr.strip(), out.stderr
