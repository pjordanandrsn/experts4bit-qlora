"""The p81 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P81).

`bench/p81/p81_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p81/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's own staging `case`. It also runs the lane reducer's self-test, whose thirteen synthetic cases pin
the registered decision rule (P80's nine, plus the pack gate and the eager control's grouping).
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p81"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p81_drive.sh's `case`, verbatim."""
    if name in ("p81_run.sh", "p81_reduce.py", "step_decomp.py", "hook/usercustomize.py"):
        return LANE / name
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
        f"Re-pin it, or the next p81 launch refuses ON A RENTED BOX."
    )


def test_every_staged_piece_the_driver_names_is_pinned():
    driver = (LANE / "p81_drive.sh").read_text()
    named = set()
    for key in ("STAGE=", "HOOK="):
        line = next(ln for ln in driver.splitlines() if ln.startswith(key))
        for p in line.split("=", 1)[1].split():
            p = p.strip('"')
            named.add("hook/" + p.rsplit("/", 1)[-1] if "/hook/" in p else p.rsplit("/", 1)[-1])
    named.discard("staged.sha256")
    pinned = {name for _w, name in _entries()}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p81_reduce.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (13 cases)" in out.stdout


def _removed(a: pathlib.Path, b: pathlib.Path):
    import difflib
    return [ln for ln in difflib.unified_diff(a.read_text().splitlines(), b.read_text().splitlines(), lineterm="", n=0)
            if ln.startswith("-") and not ln.startswith("---")]


def test_the_lane_copy_is_p80s_plus_the_p81_additions_only():
    """P81 measures P80's stage on another stack; a copy that changed the stage would measure something else."""
    removed = _removed(REPO / "bench" / "p80" / "step_decomp.py", LANE / "step_decomp.py")
    # the P80 lines the copy rewrites: its docstring's lane line, the grouping condition (widened to the eager control
    # under --dynb-grouping device) and its comment, the lane label in the refusal, the receipt and the banner
    allowed = ('"""Lane P80 (bench/p80/PREREG-p80.md)', 'if a.dynb_mode in ("graph", "padded"):',
               "# runs the same grouping so it computes the captured function", 'P80 TRACE MISMATCH',
               'rep = {"lane": "P80"', '"capture_s": cap_s, "device_grouping"', 'print(f"P80_DYNB arm=')
    assert all(any(s in ln for s in allowed) for ln in removed), removed
    assert len(removed) == len(allowed), removed


def test_the_p80_copy_still_pins_to_the_live_harness():
    """P81's copy is derived from P80's, which test_p80_staged_pin ties to bench/hybrid-g9; both stay true together."""
    removed = _removed(REPO / "bench" / "hybrid-g9" / "step_decomp.py", REPO / "bench" / "p80" / "step_decomp.py")
    assert all("max(a.gen_tokens, a.ppl_steps)" in ln or ln.strip() == '-                    device="cuda")'
               for ln in removed), removed


def test_the_eager_control_takes_the_device_grouping():
    """With the int4 store, grouping off sends T > 1 decode to the prefill branch; every P81 arm passes device."""
    run = (LANE / "p81_run.sh").read_text()
    assert "--dynb-grouping device" in run
    src = (LANE / "step_decomp.py").read_text()
    assert 'if a.dynb_mode in ("graph", "padded") or a.dynb_grouping == "device":' in src


def test_every_arm_loads_both_packs_by_fingerprint():
    run = (LANE / "p81_run.sh").read_text()
    load = next(ln for ln in run.splitlines() if ln.startswith("LOAD="))
    for k in ("E4B_INT4_ARTIFACT_DIR=$W/artifact1", "E4B_INT4_EXPECTED_FINGERPRINT=$FP_E",
              "E4B_SERVE_ATTN_INT4_ARTIFACT=$W/attn1", "E4B_SERVE_ATTN_INT4_FINGERPRINT=$FP_A", "E4B_SERVE_ATTN_INT4_CALIB=1"):
        assert k in load, k
    assert "E4B_SERVE_ATTN_INT4_DUMP" not in load and "E4B_INT4_DUMP_ARTIFACT_DIR" not in load


def test_the_trace_is_the_registered_one():
    """step_decomp's _dynb_plan at the registered batch and segment: 16 rows, the set halving every 32 steps."""
    import ast
    src = (LANE / "step_decomp.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_dynb_plan")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "step_decomp._dynb_plan", "exec"), ns)
    max_new, trace = ns["_dynb_plan"](16, 32)
    assert trace == [16] * 32 + [8] * 32 + [4] * 32 + [2] * 32 + [1] * 32
    assert sum(trace) == 992
