"""The p84 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P84).

`bench/p84/p84_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p84/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's staging `case`: harness o/ is P70's (bench/p39/step_decomp.py, bench/p42/hook) and harness n/ is
P82's (bench/p81/step_decomp.py, bench/p81/hook), all referenced unchanged. It also runs the reducer's self-test (15
cases) and pins the parts of the runner the registration depends on: the stacks, the known floats, the order, the
control gate, and each build's harness and env.
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p84"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p84_run.sh": LANE / "p84_run.sh",
    "p84_reduce.py": LANE / "p84_reduce.py",
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
        f"Re-pin it, or the next p84 launch refuses ON A RENTED BOX."
    )


def test_every_pinned_name_is_staged_by_the_driver_and_resolves_the_same_way():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p84_drive.sh").read_text()
    for name, src in SOURCES.items():
        rel = src.relative_to(REPO / "bench")
        lane_dir, base = rel.parts[0], "/".join(rel.parts[1:])
        # the driver's `case` maps the name to that source; the lane's own two files resolve through $HERE
        if lane_dir == "p84":
            assert f'{name}|' in driver or f'|{name})' in driver, name
        else:
            var = {"p39": "$P39", "p42": "$P42", "p81": "$P81"}[lane_dir]
            assert f'src="{var}/{base}"' in driver or f"{var}/{base}" in driver, name


def test_harness_o_is_p70s_pinned_bytes():
    """Harness o/ is P70's build as P70 ran it: its step_decomp and hook are the bytes bench/p70 pins."""
    p70 = {name: want for want, name in _entries(REPO / "bench" / "p70" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    assert mine["o/step_decomp.py"] == p70["step_decomp.py"]
    assert mine["o/hook/usercustomize.py"] == p70["hook/usercustomize.py"]
    assert mine["k8_bake.py"] == p70["k8_bake.py"] and mine["calib.json"] == p70["calib.json"]


def test_harness_n_is_p82s_pinned_bytes():
    p82 = {name: want for want, name in _entries(REPO / "bench" / "p82" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    assert mine["n/step_decomp.py"] == p82["step_decomp.py"]
    assert mine["n/hook/usercustomize.py"] == p82["hook/usercustomize.py"]


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p84_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (15 cases)" in out.stdout


def test_stack_a_is_p70s_e4b_on_the_same_kernel_cut_as_b_and_the_known_floats_are_p83s():
    import json
    run = (LANE / "p84_run.sh").read_text()
    assert "A_E4B=c77aab6dafb5f08d98a7387d1e986d141b20a2d3; A_GNF4=$GNF4_SHA" in run
    assert "B_E4B=$E4B_SHA; B_GNF4=$GNF4_SHA" in run
    red = (LANE / "p84_reduce.py").read_text()
    for name, key in (("O_KNOWN", "k8_O_build.json"), ("N_KNOWN", "k8_N_build.json")):
        want = json.loads((REPO / "bench" / "p83" / "receipts" / "p83-5090-1" / key).read_text())["mean_nll"]
        assert f"{name} = {want!r}" in red, (name, want)
    # the control gate accepts P83's N reading and refuses its O reading
    for key, rc in (("k8_N_build.json", 0), ("k8_O_build.json", 1)):
        out = subprocess.run([sys.executable, str(LANE / "p84_reduce.py"), "--control-ok",
                              str(REPO / "bench" / "p83" / "receipts" / "p83-5090-1" / key)], capture_output=True, text=True)
        assert out.returncode == rc, (key, out.stdout)


def test_the_router_is_fp32_everywhere_and_the_gate_reads_the_module():
    run = (LANE / "p84_run.sh").read_text()
    unset_at = run.index("unset E4B_SERVE_EXP_INT4")
    export_at = run.index("export E4B_ROUTER_EPI_CAST=0")
    assert unset_at < export_at < run.index("install_stack B ")
    assert "rte.CAST_WEIGHTS[0] is False" in run and 'os.environ.get("E4B_ROUTER_EPI_CAST") == "0"' in run
    assert "-u E4B_ROUTER_EPI_CAST" not in run


def test_the_steps_run_in_the_registered_order_and_the_control_gates_the_rest():
    run = (LANE / "p84_run.sh").read_text()
    steps = ("\ninstall_stack B ", "\nbake work_b\n", "\nrun_k8 C_build n work_b watch", "\nrun_k8 C_rep n work_b nowatch",
             "\nif ! python $W/p84_reduce.py --control-ok", "\np70_build H1_build work_b ", "\ninstall_stack A ",
             "\nbake work_a\n", "\np70_build H2_build work_a ")
    assert all(run.count(s) == 1 for s in steps), [s for s in steps if run.count(s) != 1]
    order = [run.index(s) for s in steps]
    assert order == sorted(order), order
    assert run.index("\ninstall_stack A ") > run.index('if [ "${P84_PROVE:-0}" = 1 ]; then')
    gate = run[run.index("\nif ! python $W/p84_reduce.py --control-ok"):run.index("\np70_build H1_build")]
    assert "finish 0" in gate and "p84_reduce.py --dir $W --out $W/verdict.json" in gate


def test_each_build_runs_its_lanes_harness_and_env():
    run = (LANE / "p84_run.sh").read_text()
    body = run[run.index("run_k8(){"):run.index("# ---- stack B")]
    assert "env PYTHONPATH=$W/$H/hook" in body and "python $W/$H/step_decomp.py" in body
    p70 = run[run.index("p70_build(){"):run.index("# ---- stack B")]
    assert "run_k8 $NAME o $WK watch E4B_RECOMPILE_LIMIT=64" in p70 and "E4B_INT4_DUMP_ARTIFACT_DIR=$ART" in p70
    assert "E4B_INT4_EXPECTED_FINGERPRINT=$fp" in p70
    c = run[run.index("run_k8 C_build"):run.index("verifies $W/artifact_c")]
    assert c.startswith("run_k8 C_build n work_b watch") and "E4B_SERVE_ATTN_INT4_DUMP=$W/attn_c" in c
    assert "E4B_RECOMPILE_LIMIT" not in c
    assert run.index("cp $ART/manifest.json $W/manifests/") < run.index("rm -rf $W/work_b $W/artifact_h1")
    assert run.index("manifests/C_build_experts.json") < run.index("rm -rf $W/artifact_c $W/attn_c   #")
    assert "budget=${P84_FIRST_CHUNK_S:-900}" in run


def test_the_driver_runs_to_its_dry_run(tmp_path):
    """The whole controller script parses and reaches its dry run (an apostrophe inside ${VAR:?...} once opened a quote
    that swallowed the rest of the file: `bash -n` passed and only a run found it)."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p84-dry",
           "E4B_RENT_DEADLINE_EPOCH": "1", "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "GNF4_SHA": "9" * 40,
           "P84_PROVE": "1", "P84_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p84_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("DRYRUN stage -> root@h:/root/p84") and "P84_PROVE=1" in out.stdout, out.stdout
    assert not out.stderr.strip(), out.stderr
