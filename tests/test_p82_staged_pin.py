"""The p82 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P82).

`bench/p82/p82_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p82/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's own staging `case`: step_decomp.py and the hook are bench/p81's, referenced, never copied. It also
runs the lane reducer's self-test, whose 21 synthetic cases pin the registered decision rule (P81's, with the eager
runner's identity to its bucket step made decisive and the router gate added) and the K8 table reported for e4b#674.
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p82"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p82_drive.sh's `case`, verbatim."""
    if name in ("p82_run.sh", "p82_reduce.py"):
        return LANE / name
    if name in ("step_decomp.py", "hook/usercustomize.py"):
        return REPO / "bench" / "p81" / name
    return REPO / "bench" / "p39" / name


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v and len(v) < 40 else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = _resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p82 launch refuses ON A RENTED BOX."
    )


def test_every_staged_piece_the_driver_names_is_pinned():
    driver = (LANE / "p82_drive.sh").read_text()
    named = set()
    for key in ("STAGE=", "HOOK="):
        line = next(ln for ln in driver.splitlines() if ln.startswith(key))
        for p in line.split("=", 1)[1].split():
            p = p.strip('"')
            named.add("hook/" + p.rsplit("/", 1)[-1] if "/hook/" in p else p.rsplit("/", 1)[-1])
    named.discard("staged.sha256")
    pinned = {name for _w, name in _entries()}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"


def test_the_harness_and_hook_are_p81s_pinned_bytes():
    """P82 re-measures P81's stage: the step_decomp and hook it stages are the bytes bench/p81 pins."""
    p81 = dict((name, want) for want, name in _entries(REPO / "bench" / "p81" / "staged.sha256"))
    mine = dict((name, want) for want, name in _entries())
    for name in ("step_decomp.py", "hook/usercustomize.py", "k8_bake.py", "calib.json"):
        assert mine[name] == p81[name], name


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p82_reduce.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (21 cases)" in out.stdout


def test_every_process_runs_the_fp32_router_but_k16():
    """E4B_ROUTER_EPI_CAST=0 is exported after the unset list, so the build and every arm inherit it; only K16 removes it."""
    run = (LANE / "p82_run.sh").read_text()
    unset_at = run.index("unset E4B_SERVE_EXP_INT4")
    export_at = run.index("export E4B_ROUTER_EPI_CAST=0")
    assert unset_at < export_at < run.index("step_decomp.py --model")
    assert run.count("-u E4B_ROUTER_EPI_CAST") == 1 and '[ "$NAME" = K16 ] && U="-u E4B_ROUTER_EPI_CAST"' in run
    assert "rte.CAST_WEIGHTS[0] is False" in run      # the tripwire reads the module, not just the env


def test_the_router_stamp_is_sound():
    """The runner stamps CAST_WEIGHTS from a separate process under the arm's env. That reads what the arm read only
    while the module takes it from the env at import and nothing in the staged harness or hook sets either."""
    for f in (REPO / "bench" / "p81" / "step_decomp.py", REPO / "bench" / "p81" / "hook" / "usercustomize.py"):
        src = f.read_text()
        assert "CAST_WEIGHTS" not in src and "E4B_ROUTER_EPI_CAST" not in src, f
    mod = (REPO / "experts4bit_qlora" / "engines" / "router_epilogue.py").read_text()
    assert "CAST_WEIGHTS = [_cast_default()]" in mod
    assert 'os.environ.get("E4B_ROUTER_EPI_CAST", "")' in mod


def test_the_tripwire_refuses_a_cut_without_the_two_fixes():
    run = (LANE / "p82_run.sh").read_text()
    assert '"_g_sel", None) is not None\' in inspect.getsource(pa.paged_attention_forward)' in run      # e4b#777
    assert '"_e4m3_group" in inspect.getsource(fp8_kv)' in run                                           # gnf4#413
    # and the markers are real: #777's routing is in the shim this repo ships
    pa = (REPO / "experts4bit_qlora" / "engines" / "paged_attention.py").read_text()
    assert 'getattr(ctx.kv, "_g_sel", None) is not None' in pa


def test_the_decisive_reduction_runs_before_the_k8_arms():
    """A K8 arm that runs out of guard must not cost the verdict: the reducer runs after the five arms and again after."""
    run = (LANE / "p82_run.sh").read_text()
    first = run.index("python $W/p82_reduce.py --dir")
    assert run.index("arm P padded") < first < run.index("k8arm K32\n") < run.index("k8arm K16\n")
    assert run.count("python $W/p82_reduce.py --dir") == 2


def test_every_arm_loads_both_packs_by_fingerprint():
    run = (LANE / "p82_run.sh").read_text()
    load = next(ln for ln in run.splitlines() if ln.startswith("LOAD="))
    for k in ("E4B_INT4_ARTIFACT_DIR=$W/artifact1", "E4B_INT4_EXPECTED_FINGERPRINT=$FP_E",
              "E4B_SERVE_ATTN_INT4_ARTIFACT=$W/attn1", "E4B_SERVE_ATTN_INT4_FINGERPRINT=$FP_A", "E4B_SERVE_ATTN_INT4_CALIB=1"):
        assert k in load, k
    assert "E4B_SERVE_ATTN_INT4_DUMP" not in load and "E4B_INT4_DUMP_ARTIFACT_DIR" not in load
    assert "--dynb-grouping device" in run
