"""The p90 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P90).

`bench/p90/p90_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p90/staged.sha256`. That guard
runs on the CONTROLLER after a box is rented; this test runs the same comparison in CI, where it costs nothing.

It mirrors the driver's staging `case`: P58's harness pieces (bench/p39, bench/p42's hook), P42's census parser, the
premise test (tests/test_k21_row_exact_gpu.py) and P44's KL instrument (bench/p44, bench/kl_*.py), all referenced
unchanged. It also:
- runs the reducer's self-test (18 cases);
- pins the parts of the runner the registration depends on: the gnf4 pin (a real sha), the refusals, the premise and the
  K0 controls before any fetch, the proof, and the arms with their order and settings.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p90"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p90_run.sh": LANE / "p90_run.sh",
    "p90_reduce.py": LANE / "p90_reduce.py",
    "p42_reduce.py": REPO / "bench" / "p42" / "p42_reduce.py",
    "test_k21_row_exact_gpu.py": REPO / "tests" / "test_k21_row_exact_gpu.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "serve_stack.py": REPO / "bench" / "p44" / "serve_stack.py",
    "kl_serve.py": REPO / "bench" / "p44" / "kl_serve.py",
    "kl_fidelity.py": REPO / "bench" / "kl_fidelity.py",
    "kl_paths.py": REPO / "bench" / "kl_paths.py",
    "kl_prompts.py": REPO / "bench" / "kl_prompts.py",
    "hook/usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
}
RUN = (LANE / "p90_run.sh").read_text()


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
        f"Re-pin it, or the next p90 launch refuses ON A RENTED BOX."
    )


def test_every_pinned_name_is_staged_by_the_driver_and_resolves_the_same_way():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p90_drive.sh").read_text()
    assert 'test_k21_row_exact_gpu.py) src="$TESTS/$name";;' in driver
    assert 'serve_stack.py|kl_serve.py) src="$P44/$name";;' in driver
    assert 'kl_fidelity.py|kl_paths.py|kl_prompts.py) src="$KL/$name";;' in driver
    for name, src in SOURCES.items():
        rel = src.relative_to(REPO)
        if rel.parts[0] == "tests":
            assert f"$TESTS/{name}" in driver, name
        elif rel.parts[1] == "p90":
            assert f"{name}|" in driver or f"|{name})" in driver, name
        elif rel.parts[1] in ("p39", "p42", "p44"):
            assert {"p39": "$P39/", "p42": "$P42/", "p44": "$P44/"}[rel.parts[1]] in driver, name
        else:
            assert f"$KL/{name}" in driver, name


def test_the_harness_is_p88s_pinned_bytes_and_the_instrument_p44s():
    """The speed arms run P88's (P86's, P58's) harness bytes; the KL instrument's shared modules are P44-b's."""
    p88 = {name: want for want, name in _entries(REPO / "bench" / "p88" / "staged.sha256")}
    p44b = {name: want for want, name in _entries(REPO / "bench" / "p44" / "staged-b.sha256")}
    mine = {name: want for want, name in _entries()}
    for name in ("step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py", "p42_reduce.py"):
        assert mine[name] == p88[name], name
    for name in ("kl_paths.py", "kl_prompts.py"):
        assert mine[name] == p44b[name], name


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p90_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (18 cases)" in out.stdout


def test_the_gnf4_pin_is_a_real_sha():
    m = re.search(r"^GNF4_SHA=(\S+)", RUN, re.M)
    assert m and re.fullmatch(r"[0-9a-f]{40}", m.group(1)), f"GNF4_SHA is not a 40-char sha: {m and m.group(1)}"


def test_the_refusals_the_premise_and_k0_come_before_any_fetch():
    assert "MID=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee" in RUN
    install = RUN.index('say "install e4b @')
    for refusal in ("finish 15;", "finish 13;"):                                  # class, disk
        assert RUN.index(refusal) < install, refusal
    fetch = RUN.index('say "fetch $MID @ $REV"')
    premise = RUN.index("python -m pytest test_k21_row_exact_gpu.py")
    k0 = RUN.index("python $W/kl_fidelity.py --controls --out $W/k0.json")
    assert install < premise < k0 < RUN.index('if [ "${P90_PROVE:-0}" = 1 ]; then') < fetch
    assert "finish 25" in RUN[premise:k0] and "finish 26" in RUN[k0:fetch]
    trip = RUN[install:premise]
    for must in ("hr._k21_has_masked_tail(mxfp4_grouped)", 'hr._K21_PLAN == {"block_n": 32, "kc": 128, "warps": 4, "stages": 3}',
                 "hr._k21_mode_env() is False", "hr._MXFP4_GEMV_ROWS == 16", "from transformers import Mxfp4Config"):
        assert must in trip, must


def test_the_proof_compiles_k21s_contract_on_the_card_and_fetches_no_model():
    prove = RUN.index('if [ "${P90_PROVE:-0}" = 1 ]; then')
    block = RUN[prove:RUN.index('say "fetch $MID @ $REV"')]
    assert "git -C $W/gnf4 checkout -q $GNF4_SHA" in block
    assert "TRITON_INTERPRET=0" in block and "test_mxfp4_grouped_smallm_interp.py test_int4_smallm_interp.py" in block
    assert "finish 23" in block and ": > PROVED; finish 0" in block
    assert "for v in P90_PROVE; do" in (LANE / "p90_drive.sh").read_text()


def test_the_arms_run_in_the_registered_order_with_their_settings():
    steps = ('can_run 900 b${B}_off && { speed 0 $B "" 1;', 'can_run 900 b${B}_on && { speed 1 $B "" 1;',
             "can_run 3000 kl_off && { kl 0;", "can_run 1800 kl_on && { kl 1;",
             'can_run 900 b${B}_on_r2 && { speed 1 $B _r2 "";', 'can_run 900 b${B}_off_r2 && { speed 0 $B _r2 "";',
             "python $W/p90_reduce.py --dir $W --out $W/verdict.json")
    assert all(RUN.count(s) == 1 for s in steps), [s for s in steps if RUN.count(s) != 1]
    order = [RUN.index(s) for s in steps]
    assert order == sorted(order), order
    assert RUN.count("for B in 16 1; do") == 2
    assert ('STOREENV="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 '
            'E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=0"') in RUN
    arm = RUN[RUN.index("speed(){"):RUN.index("kl(){")]
    assert "--b1d-loop graph --b1d-timed --no-fuse-qkv" in arm and "--replay-profile-out" in arm
    assert "E4B_MXFP4_GROUPED_SMALLM=$K" in arm
    kl = RUN[RUN.index("kl(){"):RUN.index("rc_any=0;")]
    assert "E4B_MXFP4_GROUPED_SMALLM=$K" in kl and "--family gptoss --arms store_r12" in kl
    assert '--scorer $([ "$K" = 1 ] && echo decode || echo auto)' in kl and "--k0-receipt $W/k0.json" in kl
    assert "unset E4B_SERVE_EXP_INT4" in RUN and "E4B_MXFP4_GROUPED_SMALLM E4B_CALIB_SOURCE" in RUN


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"finish (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes                               # only the disk floor names the host


def test_the_driver_runs_to_its_dry_run(tmp_path):
    """The whole controller script parses and reaches its dry run (P83's apostrophe lesson)."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p90-dry",
           "E4B_RENT_DEADLINE_EPOCH": "1", "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P90_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p90_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("DRYRUN stage -> root@h:/root/p90"), out.stdout
    assert not out.stderr.strip(), out.stderr
