"""The k24 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for K24).

`bench/k24/k24_drive.sh` refuses to run when a staged file's sha256 differs from `bench/k24/staged.sha256`. That guard
runs on the CONTROLLER after a box is rented; this test runs the same comparison in CI, where it costs nothing.

It also pins the runner's shape:
- the refusals come before the install;
- the proof runs K21's contracts and fetches no model;
- the phases run in order (census, bench), the routing being K22's recording from the gnf4 clone;
- the bench's instrument is the census's own `_gemm_nf4_grouped` row;
- the lane's failure codes avoid the launcher's machine-exclusion codes.
"""
import hashlib
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "k24"
PIN = LANE / "staged.sha256"
SOURCES = {
    "k24_run.sh": LANE / "k24_run.sh",
    "p42_reduce.py": REPO / "bench" / "p42" / "p42_reduce.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "hook/usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
}
RUN = (LANE / "k24_run.sh").read_text()


def _entries():
    for line in PIN.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v and len(v) < 40 else "sha")
def test_staged_file_matches_its_pin(want, name):
    got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
    assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next k24 launch refuses ON A RENTED BOX"


def test_every_pinned_name_is_staged_by_the_driver():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "k24_drive.sh").read_text()
    for name in ("k24_run.sh", "p42_reduce.py"):
        assert f"{name}" in driver, name
    for var in ("$P39/", "$P42/"):
        assert var in driver, var


def test_the_harness_is_p86s_pinned_bytes():
    p86 = {n: w for w, n in (ln.split(None, 1) for ln in (REPO / "bench" / "p86" / "staged.sha256").read_text().splitlines()
                              if ln.strip() and not ln.startswith("#"))}
    mine = dict((n, w) for w, n in _entries())
    for name in ("step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py", "p42_reduce.py"):
        assert mine[name] == p86[name.strip()], name


def test_refusals_precede_install_and_the_proof_fetches_no_model():
    install = RUN.index('say "install e4b @')
    assert RUN.index("finish 15;") < install and RUN.index("finish 13;") < install
    prove = RUN.index('if [ "${K24_PROVE:-0}" = 1 ]; then')
    fetch = RUN.index('say "fetch $MID @ $REV"')
    assert install < prove < fetch
    block = RUN[prove:fetch]
    assert "test_mxfp4_grouped_smallm_interp.py test_int4_smallm_interp.py" in block and "TRITON_INTERPRET=0" in block
    assert "finish 23" in block and ": > PROVED; finish 0" in block
    assert "MID=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee" in RUN


def test_the_phases_run_in_order_with_bo7s_store_r12_env():
    order = [RUN.index(s) for s in ("cp $W/src/kernel/receipts-k22/5090/eids_b16.pt $W/eids_b16.pt",
                                    'say "phase 1: census B=16 (store_r12)"', 'say "phase 2: k24_bench')]
    assert order == sorted(order), order
    assert "record_eids" not in RUN
    assert 'STOREENV="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=0"' in RUN
    census = RUN[RUN.index('say "phase 1'):RUN.index('say "phase 2')]
    assert "--batch 16" in census and "--b1d-loop graph --b1d-timed --no-fuse-qkv" in census and "--replay-profile-out" in census
    assert 'r["name"].strip() == "_gemm_nf4_grouped"' in census                 # the instrument is the census's own row
    assert '--census-nf4-ms "$NF4MS"' in RUN


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"finish (\d+)", RUN)}
    host = {13, 14, 17, 18}
    assert codes & host == {13}, codes & host                                   # only the disk floor is a host refusal


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "k24-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "GNF4_SHA": "1" * 40, "K24_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "k24_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/k24"), out.stdout + out.stderr
