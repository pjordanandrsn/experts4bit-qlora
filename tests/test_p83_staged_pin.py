"""The p83 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P83).

`bench/p83/p83_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p83/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's staging `case`: stack O's harness is P70's (bench/p39/step_decomp.py, bench/p42/hook), stack N's is
P82's (bench/p81/step_decomp.py, bench/p81/hook), all referenced unchanged. It also runs the reducer's self-test (13
cases: SAME, DIFFERENT, and each VOID) and pins the parts of the runner the registration depends on.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p83"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p83_run.sh": LANE / "p83_run.sh",
    "p83_reduce.py": LANE / "p83_reduce.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "o/step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "o/hook/usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
    "n/step_decomp.py": REPO / "bench" / "p81" / "step_decomp.py",
    "n/hook/usercustomize.py": REPO / "bench" / "p81" / "hook" / "usercustomize.py",
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
        f"Re-pin it, or the next p83 launch refuses ON A RENTED BOX."
    )


def test_every_pinned_name_is_staged_by_the_driver_and_resolves_the_same_way():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p83_drive.sh").read_text()
    for name, src in SOURCES.items():
        rel = src.relative_to(REPO / "bench")
        lane_dir, base = rel.parts[0], "/".join(rel.parts[1:])
        # the driver's `case` maps the name to that source; the lane's own two files resolve through $HERE
        if lane_dir == "p83":
            assert f'{name}|' in driver or f'|{name})' in driver, name
        else:
            var = {"p39": "$P39", "p42": "$P42", "p81": "$P81"}[lane_dir]
            assert f'src="{var}/{base}"' in driver or f"{var}/{base}" in driver, name


def test_stack_os_harness_is_p70s_pinned_bytes():
    """Stack O is P70's build as P70 ran it: its step_decomp and hook are the bytes bench/p70 pins."""
    p70 = {name: want for want, name in _entries(REPO / "bench" / "p70" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    assert mine["o/step_decomp.py"] == p70["step_decomp.py"]
    assert mine["o/hook/usercustomize.py"] == p70["hook/usercustomize.py"]
    assert mine["k8_bake.py"] == p70["k8_bake.py"] and mine["calib.json"] == p70["calib.json"]


def test_stack_ns_harness_is_p82s_pinned_bytes():
    p82 = {name: want for want, name in _entries(REPO / "bench" / "p82" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    assert mine["n/step_decomp.py"] == p82["step_decomp.py"]
    assert mine["n/hook/usercustomize.py"] == p82["hook/usercustomize.py"]


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p83_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (13 cases)" in out.stdout


def test_stack_os_pins_are_p70s_launch_and_kernel_cuts():
    run = (LANE / "p83_run.sh").read_text()
    assert "O_E4B=c77aab6dafb5f08d98a7387d1e986d141b20a2d3" in run
    assert "O_GNF4=5ca1897585f9f456f99ea504b2a1be0ea91db496" in run
    # P70 registered the same kernel cut; its drive script names it
    assert "5ca1897585f9f456f99ea504b2a1be0ea91db496" in (REPO / "bench" / "p70" / "p70_drive.sh").read_text()


def test_the_router_is_fp32_everywhere_and_the_gate_reads_the_module():
    run = (LANE / "p83_run.sh").read_text()
    unset_at = run.index("unset E4B_SERVE_EXP_INT4")
    export_at = run.index("export E4B_ROUTER_EPI_CAST=0")
    assert unset_at < export_at < run.index("install_stack O ")
    assert "rte.CAST_WEIGHTS[0] is False" in run and 'os.environ.get("E4B_ROUTER_EPI_CAST") == "0"' in run
    assert "-u E4B_ROUTER_EPI_CAST" not in run            # no step removes it


def test_the_steps_run_in_the_registered_order():
    run = (LANE / "p83_run.sh").read_text()
    # column-0 lines only: the proof's own (indented) install of N and the bake function's text must not match
    steps = ("\ninstall_stack O ", "\nbake O\n", "\nrun_k8 O_build O watch", "\nrun_k8 O_rep O nowatch",
             "\ninstall_stack N ", "\nbake N\n", "\nrun_k8 N_build N watch", "\nrun_k8 N_rep N nowatch",
             "\npython $W/p83_reduce.py --dir")
    assert all(run.count(s) == 1 for s in steps), [s for s in steps if run.count(s) != 1]
    order = [run.index(s) for s in steps]
    assert order == sorted(order), order
    assert run.index("\ninstall_stack N ") > run.index('if [ "${P83_PROVE:-0}" = 1 ]; then')


def test_each_step_runs_its_own_stacks_harness_and_hook():
    run = (LANE / "p83_run.sh").read_text()
    body = run[run.index("run_k8(){"):run.index("# ---- stack O")]
    assert "env PYTHONPATH=$W/$s/hook" in body and "python $W/$s/step_decomp.py" in body
    assert re.search(r"--arena \$W/work_\$s/nf4\.arena", body)
    # P70's env for O (its recompile limits and dump) and P82's for N (both packs dumped, then both loaded)
    o_build = run[run.index("run_k8 O_build"):run.index("verifies $W/artifact_o")]
    assert "E4B_RECOMPILE_LIMIT=64" in o_build and "E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact_o" in o_build
    n_build = run[run.index("run_k8 N_build"):run.index("verifies $W/artifact_n")]
    assert "E4B_SERVE_ATTN_INT4_DUMP=$W/attn_n" in n_build and "E4B_RECOMPILE_LIMIT" not in n_build
    n_rep = run[run.index("run_k8 N_rep"):run.index("printf '{\"O_expert\"")]
    for k in ("E4B_INT4_EXPECTED_FINGERPRINT=$FP_N", "E4B_SERVE_ATTN_INT4_FINGERPRINT=$FP_NA"):
        assert k in n_rep, k


def test_the_driver_runs_to_its_dry_run(tmp_path):
    """The whole controller script parses and reaches its dry run (an apostrophe inside ${VAR:?...} once opened a quote
    that swallowed the rest of the file: `bash -n` passed and only a run found it)."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p83-dry",
           "E4B_RENT_DEADLINE_EPOCH": "1", "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "GNF4_SHA": "9" * 40,
           "P83_PROVE": "1", "P83_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p83_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("DRYRUN stage -> root@h:/root/p83") and "P83_PROVE=1" in out.stdout, out.stdout
    assert not out.stderr.strip(), out.stderr
