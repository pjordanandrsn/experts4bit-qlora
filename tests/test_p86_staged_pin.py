"""The p86 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P86).

`bench/p86/p86_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p86/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's staging `case`: P58's harness pieces (bench/p39, bench/p42's hook), P37's vLLM timing arm and
P42's census parser, all referenced unchanged. It also runs the reducer's self-test (13 cases) and pins the parts of
the runner the registration depends on: the comparator, the refusals, the arms and their order.
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p86"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p86_run.sh": LANE / "p86_run.sh",
    "p86_reduce.py": LANE / "p86_reduce.py",
    "p86_vllm_census.py": LANE / "p86_vllm_census.py",
    "p42_reduce.py": REPO / "bench" / "p42" / "p42_reduce.py",
    "p37_vllm.py": REPO / "bench" / "h2h-20260905" / "p37" / "p37_vllm.py",
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
        f"Re-pin it, or the next p86 launch refuses ON A RENTED BOX."
    )


def test_every_pinned_name_is_staged_by_the_driver_and_resolves_the_same_way():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p86_drive.sh").read_text()
    for name, src in SOURCES.items():
        rel = src.relative_to(REPO / "bench")
        lane_dir = rel.parts[0]
        if lane_dir == "p86":
            assert f"{name}|" in driver or f"|{name})" in driver, name
        else:
            var = {"p39": "$P39", "p42": "$P42", "h2h-20260905": "$P37"}[lane_dir]
            assert f"{var}/" in driver, name


def test_the_harness_and_comparator_are_p58s_pinned_bytes():
    """The e4b harness, the hook and P37's vLLM arm are the bytes P58 staged: P86 explains P58's ratio."""
    p58 = {name: want for want, name in _entries(REPO / "bench" / "p58" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    for name in ("step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py", "p37_vllm.py"):
        assert mine[name] == p58[name], name


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p86_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (13 cases)" in out.stdout


def test_the_comparator_and_the_refusals_are_registered_before_any_install():
    run = (LANE / "p86_run.sh").read_text()
    assert "VLLM_VER=${P86_VLLM:-0.30.0}" in run and 'VLLM_VERSION = "0.30.0"' in (LANE / "p86_reduce.py").read_text()
    assert "GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4; GPTQ_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441" in run
    assert "GNF4_SHA=9407d499a4d1e0fe8c22b050878a9f869b385b45" in run
    install = run.index('say "install e4b @')
    for refusal in ("finish 15;", "finish 18;", "finish 13;"):        # class, driver, disk
        assert run.index(refusal) < install, refusal
    assert "MIN_DRIVER=${P86_MIN_DRIVER:-580}" in run


def test_the_proof_runs_the_census_on_a_small_model_before_any_30b_fetch():
    run = (LANE / "p86_run.sh").read_text()
    prove = run.index('if [ "${P86_PROVE:-0}" = 1 ]; then')
    assert run.index('say "install vllm==') < prove < run.index('say "fetch $MID @ $REV"')
    block = run[prove:run.index('say "fetch $MID @ $REV"')]
    assert "P86_MODEL=Qwen/Qwen3-0.6B" in block and "P86_SHORT=8 P86_LONG=24" in block and "p86_vllm_census.py" in block
    assert "finish 23" in block and ": > PROVED; finish 0" in block
    assert "for v in P86_PROVE; do" in (LANE / "p86_drive.sh").read_text()


def test_the_arms_run_in_the_registered_order_with_their_settings():
    run = (LANE / "p86_run.sh").read_text()
    steps = ('for B in 16 1; do can_run 900 e4b_b$B && { e4b_arm "" $B 1;',
             'for B in 16 1; do can_run 900 vllm_graph_b$B && { vllm_time $B "";',
             "for B in 16 1; do can_run 900 vllm_census_b$B && { vllm_census $B;",
             'for B in 16 1; do can_run 900 e4b_b${B}_r2 && { e4b_arm _r2 $B "";',
             "for B in 16 1; do can_run 900 vllm_graph_b${B}_r2 && { vllm_time $B _r2;",
             "python $W/p86_reduce.py --dir $W --out $W/verdict.json")
    assert all(run.count(s) == 1 for s in steps), [s for s in steps if run.count(s) != 1]
    order = [run.index(s) for s in steps]
    assert order == sorted(order), order
    arm = run[run.index("e4b_arm(){"):run.index("vllm_time(){")]
    assert "--b1d-loop graph --b1d-timed --fuse-qkv" in arm and "--replay-profile-out" in arm
    assert "E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0" in arm
    assert "P37_ARM=graph_r1" in run[run.index("vllm_time(){"):run.index("vllm_census(){")]


def test_the_census_runs_the_engine_in_process_with_p37s_settings():
    c = (LANE / "p86_vllm_census.py").read_text()
    assert c.index('os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")') < c.index("import vllm")
    p37 = (REPO / "bench" / "h2h-20260905" / "p37" / "p37_vllm.py").read_text()
    for setting in ("max_model_len", "enable_prefix_caching=False", "seed=0", "tensor_parallel_size=1"):
        assert setting in c and setting in p37, setting
    assert 'SHORT, LONG = int(os.environ.get("P86_SHORT", "32")), int(os.environ.get("P86_LONG", "128"))' in c


def test_the_driver_runs_to_its_dry_run(tmp_path):
    """The whole controller script parses and reaches its dry run (P83's apostrophe lesson)."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p86-dry",
           "E4B_RENT_DEADLINE_EPOCH": "1", "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P86_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p86_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("DRYRUN stage -> root@h:/root/p86"), out.stdout
    assert not out.stderr.strip(), out.stderr
