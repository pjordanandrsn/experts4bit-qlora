"""The p94 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P94).

`bench/p94/p94_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p94/staged.sha256`; this test runs
the same comparison in CI. It also runs the reducer's self-test (10 cases) and pins the runner's registration: the gnf4
pin, the families at their revisions with their licensed envs, the three arms (g the scalar GEMV, m the served M-tile
through the instrument, t K25), the premise before the fetch, the order, and the exit codes.
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p94"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p94_run.sh": LANE / "p94_run.sh",
    "p94_reduce.py": LANE / "p94_reduce.py",
    "p42_reduce.py": REPO / "bench" / "p42" / "p42_reduce.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "hook/usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
    "test_k25_row_exact_gpu.py": REPO / "tests" / "test_k25_row_exact_gpu.py",
    "test_nf4_t1_device_grouping_gpu.py": REPO / "tests" / "test_nf4_t1_device_grouping_gpu.py",
}
RUN = (LANE / "p94_run.sh").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v and len(v) < 40 else "sha")
def test_staged_file_matches_its_pin(want, name):
    got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
    assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p94 launch refuses ON A RENTED BOX"


def test_every_pinned_name_is_staged_by_the_driver():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p94_drive.sh").read_text()
    for name in ("p94_run.sh", "p94_reduce.py", "p42_reduce.py", "test_k25_row_exact_gpu.py", "test_nf4_t1_device_grouping_gpu.py"):
        assert name in driver, name


def test_the_harness_is_p93s_pinned_bytes():
    p93 = {name: want for want, name in _entries(REPO / "bench" / "p93" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    for name in ("step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py", "p42_reduce.py"):
        assert mine[name] == p93[name], name


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p94_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (10 cases)" in out.stdout, out.stdout + out.stderr


def test_the_families_run_their_licensed_envs_at_the_registered_revisions():
    sys.path.insert(0, str(REPO / "bench" / "p44"))
    try:
        from serve_stack import MODELS, arm_env
    finally:
        sys.path.pop(0)
    for fam, arm, var in (("granite", "r12epi", "GR"), ("olmoe", "nf4", "OL")):
        mid, rev = MODELS[fam]
        assert f"{var}={mid}; {var}_REV={rev}" in RUN, fam
        env = arm_env(fam, arm)
        want = " ".join(f"{k}={env[k]}" for k in ("E4B_SERVE_EXP_INT4", "E4B_SERVE_ATTN_INT4_CALIB", "E4B_CALIB_SOURCE",
                                                  "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI"))
        assert f'{var}_ENV="{want}"' in RUN, (fam, want)


def test_the_kernel_pin_the_arms_and_the_route_plan_are_registered():
    assert re.search(r"^GNF4_SHA=[0-9a-f]{40}\b", RUN, re.M)
    assert 'ARM_G="E4B_NF4_GROUPED_SMALLM=0 E4B_NF4_T1_DEVICE_GROUPING=0"' in RUN
    assert 'ARM_M="E4B_NF4_GROUPED_SMALLM=0 E4B_NF4_T1_DEVICE_GROUPING=1"' in RUN
    assert 'ARM_T="E4B_NF4_GROUPED_SMALLM=1 E4B_NF4_T1_DEVICE_GROUPING=0"' in RUN
    assert "E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE" in RUN                  # the instrument starts unset
    plan = ast.literal_eval(re.search(r"assert _K25_PLAN == (\{[^}]*\})", RUN).group(1))
    assert plan == {"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}, plan


def test_the_premise_precedes_the_fetch_and_the_arms_are_in_the_registered_order():
    install = RUN.index('say "install e4b @')
    assert RUN.index("finish 15;") < install and RUN.index("finish 13;") < install
    premise = RUN.index("python -m pytest test_k25_row_exact_gpu.py test_nf4_t1_device_grouping_gpu.py")
    contract = RUN.index("python -m pytest test_nf4_grouped_smallm_interp.py")
    first_fetch = RUN.index('family granite "$GR"')
    assert install < premise < contract < first_fetch
    fam = RUN[RUN.index("family(){"):RUN.index("# the registered order: Granite")]
    assert fam.index('speed $TAG "$MID" "$QA" "$ENVS" m 1 1 1') < fam.index("for SRC in wikitext c4val1; do for ARM in g m t; do")
    assert "--batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv" in RUN
    assert RUN.index('family granite') < RUN.index('family olmoe') < RUN.index("python $W/p94_reduce.py --dir $W --out $W/verdict.json")


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p94-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P94_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p94_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p94"), out.stdout + out.stderr
