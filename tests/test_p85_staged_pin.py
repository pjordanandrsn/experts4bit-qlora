"""The p85 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P85).

`bench/p85/p85_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p85/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's staging `case`: harness o/ is P70's (bench/p39/step_decomp.py, bench/p42/hook), referenced
unchanged. It also runs the reducer's self-test (20 cases) and pins the parts of the runner the registration depends
on: the two stacks, the known floats, the order, the control gate, each reading's env, and the append stamp.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p85"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p85_run.sh": LANE / "p85_run.sh",
    "p85_reduce.py": LANE / "p85_reduce.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "o/step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "o/hook/usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
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
        f"Re-pin it, or the next p85 launch refuses ON A RENTED BOX."
    )


def test_every_pinned_name_is_staged_by_the_driver_and_resolves_the_same_way():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p85_drive.sh").read_text()
    for name, src in SOURCES.items():
        rel = src.relative_to(REPO / "bench")
        lane_dir, base = rel.parts[0], "/".join(rel.parts[1:])
        if lane_dir == "p85":
            assert f'{name}|' in driver or f'|{name})' in driver, name
        else:
            var = {"p39": "$P39", "p42": "$P42"}[lane_dir]
            assert f'src="{var}/{base}"' in driver or f"{var}/{base}" in driver, name
    assert "/n/" not in driver and "$P81" not in driver, "P85 stages P70's harness only"


def test_harness_o_is_p70s_pinned_bytes():
    """Harness o/ is P70's build as P70 ran it (and as P84 ran it): its step_decomp and hook are the bytes bench/p70 pins."""
    p70 = {name: want for want, name in _entries(REPO / "bench" / "p70" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    assert mine["o/step_decomp.py"] == p70["step_decomp.py"]
    assert mine["o/hook/usercustomize.py"] == p70["hook/usercustomize.py"]
    assert mine["k8_bake.py"] == p70["k8_bake.py"] and mine["calib.json"] == p70["calib.json"]


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p85_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (20 cases)" in out.stdout


def test_the_stacks_are_registered_constants_and_the_known_floats_are_p83s():
    run = (LANE / "p85_run.sh").read_text()
    assert "O_E4B=c77aab6dafb5f08d98a7387d1e986d141b20a2d3; O_GNF4=5ca1897585f9f456f99ea504b2a1be0ea91db496" in run
    assert "S_E4B=$O_E4B; S_GNF4=f879761501fedcac027de81f4083fab924194b8d" in run
    assert 'want = {"O": ("0.37.4", "0.33.0"), "S": ("0.37.4", "0.33.6")}[S]' in run
    red = (LANE / "p85_reduce.py").read_text()
    o = json.loads((REPO / "bench" / "p83" / "receipts" / "p83-5090-1" / "k8_O_build.json").read_text())["mean_nll"]
    n = json.loads((REPO / "bench" / "p83" / "receipts" / "p83-5090-1" / "k8_N_build.json").read_text())["mean_nll"]
    assert f"O_KNOWN = {o!r}" in red and f"N_KNOWN = {n!r}" in red
    # the control gate accepts P83's O reading and refuses its N reading
    for f, rc in ((REPO / "bench" / "p83" / "receipts" / "p83-5090-1" / "k8_O_build.json", 0),
                  (REPO / "bench" / "p83" / "receipts" / "p83-5090-1" / "k8_N_build.json", 1)):
        out = subprocess.run([sys.executable, str(LANE / "p85_reduce.py"), "--control-ok", str(f)], capture_output=True, text=True)
        assert out.returncode == rc, (f.name, out.stdout)


def test_the_router_is_fp32_everywhere_and_the_append_knob_starts_unset():
    run = (LANE / "p85_run.sh").read_text()
    unset_at = run.index("unset E4B_SERVE_EXP_INT4")
    export_at = run.index("export E4B_ROUTER_EPI_CAST=0")
    assert unset_at < export_at < run.index("\ninstall_stack O ")
    unset_block = run[unset_at:export_at]
    assert "E4B_FUSED_KV_APPEND" in unset_block
    assert "rte.CAST_WEIGHTS[0] is False" in run and 'os.environ.get("E4B_ROUTER_EPI_CAST") == "0"' in run
    assert "-u E4B_ROUTER_EPI_CAST" not in run


def test_the_tripwire_and_the_stamp_read_the_append_and_the_pre_413_kernel():
    run = (LANE / "p85_run.sh").read_text()
    assert 'assert not hasattr(fp8_kv, "_quantize_kv_fp32")' in run
    assert 'res(None, "cuda", lambda: True) is True and res("0", "cuda", lambda: True) is False' in run
    stamp = run[run.index("\nstamp(){"):run.index("\ninstall_stack O ")]
    for key in ('"fused_kv_append_env"', '"fused_kv_append_resolved": _resolve_fused_append(', '"fp8_kv_has_413"'):
        assert key in stamp, key
    # the proof exercises the stamp both ways under each stack
    assert "stamp $W/stamp_${S}_append_off.json PYTHONPATH=$W/o/hook E4B_FUSED_KV_APPEND=0" in run


def test_the_steps_run_in_the_registered_order_and_the_control_gates_the_rest():
    run = (LANE / "p85_run.sh").read_text()
    steps = ("\ninstall_stack O ", "\nbake work_o\n", "\nrun_k8 O_build o work_o watch",
             "\nif ! python $W/p85_reduce.py --control-ok $W/k8_O_build.json", "\nlic O_rep work_o\n",
             "\nlic F_1 work_o E4B_FUSED_KV_APPEND=0\n", "\nlic F_2 work_o E4B_FUSED_KV_APPEND=0\n",
             "\ninstall_stack S ", "\nbake work_s\n", "\nlic S_1 work_s\n", "\nlic S_2 work_s\n",
             "\npython $W/p85_reduce.py --dir $W --out $W/verdict.json")
    assert all(run.count(s) == 1 for s in steps), [s for s in steps if run.count(s) != 1]
    order = [run.index(s) for s in steps]
    assert order == sorted(order), order
    assert run.index("\ninstall_stack O ") < run.index('if [ "${P85_PROVE:-0}" = 1 ]; then') < run.index("\ninstall_stack S ")
    gate = run[run.index("\nif ! python $W/p85_reduce.py --control-ok"):run.index("\nlic O_rep")]
    assert "finish 0" in gate and "p85_reduce.py --dir $W --out $W/verdict.json" in gate
    # a build that crashed or left no pack is BUILD FAILED (rc 20), never a failed control: the pack is verified first
    assert run.index("\nrun_k8 O_build") < run.index("\nverifies $W/artifact_o || {") < run.index("\nif ! python $W/p85_reduce.py --control-ok")


def test_each_reading_runs_p70s_harness_and_env():
    run = (LANE / "p85_run.sh").read_text()
    body = run[run.index("run_k8(){"):run.index("# ---- stack O: the control")]
    assert "env PYTHONPATH=$W/$H/hook" in body and "python $W/$H/step_decomp.py" in body
    build = run[run.index("\nrun_k8 O_build"):run.index("\n# the control gates the rest")]
    assert "E4B_RECOMPILE_LIMIT=64" in build and "E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact_o" in build
    assert "E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0" in build and "E4B_FUSED_KV_APPEND" not in build
    lic = run[run.index("lic(){"):run.index("# ---- stack O: the control")]
    assert "run_k8 $NAME o $WK nowatch E4B_RECOMPILE_LIMIT=64" in lic
    assert "E4B_INT4_EXPECTED_FINGERPRINT=$FP_O" in lic and lic.rstrip().endswith('$FOLDS "$@"; }')
    assert run.index("cp $W/artifact_o/manifest.json $W/manifests/O_build_experts.json") < run.index("rm -rf $W/artifact_o $W/work_s")
    assert "budget=${P85_FIRST_CHUNK_S:-1500}" in run   # amendment 1 (was 900)


def test_a_host_cpu_other_than_the_registered_vendor_is_refused_before_any_fetch():
    run = (LANE / "p85_run.sh").read_text()
    assert "CPU_VENDOR=${P85_CPU_VENDOR:-AuthenticAMD}" in run
    assert '|| [ "$CPU_VENDOR" != AuthenticAMD ]' in run
    assert run.index("finish 16; }") < run.index("\ninstall_stack O ") < run.index('say "fetch $MID @ $REV"')


def test_the_driver_runs_to_its_dry_run(tmp_path):
    """The whole controller script parses and reaches its dry run (P83's apostrophe lesson)."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p85-dry",
           "E4B_RENT_DEADLINE_EPOCH": "1", "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40,
           "P85_PROVE": "1", "P85_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p85_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("DRYRUN stage -> root@h:/root/p85") and "P85_PROVE=1" in out.stdout, out.stdout
    assert "GNF4_SHA" not in out.stdout
    assert not out.stderr.strip(), out.stderr
