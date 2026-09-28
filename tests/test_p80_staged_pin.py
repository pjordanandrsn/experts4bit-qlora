"""The p80 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P80).

`bench/p80/p80_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p80/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's own staging `case`. It also runs the lane reducer's self-test, whose nine synthetic cases pin the
registered decision rule (confirm, refute, void, the oracle identity, fallbacks, OOM).
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p80"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p80_drive.sh's `case`, verbatim."""
    if name in ("p80_run.sh", "p80_reduce.py", "step_decomp.py"):
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
        f"Re-pin it, or the next p80 launch refuses ON A RENTED BOX."
    )


def test_every_staged_piece_the_driver_names_is_pinned():
    driver = (LANE / "p80_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split("=", 1)[1].split()}
    named.discard("staged.sha256")
    pinned = {name for _w, name in _entries()}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p80_reduce.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK" in out.stdout


def test_the_lane_copy_is_the_live_harness_plus_the_p80_stage_only():
    """A lane copy that silently diverged from the harness would measure a different setup than the p37 control's."""
    import difflib
    live = (REPO / "bench" / "hybrid-g9" / "step_decomp.py").read_text().splitlines()
    lane = (LANE / "step_decomp.py").read_text().splitlines()
    removed = [ln for ln in difflib.unified_diff(live, lane, lineterm="", n=0)
               if ln.startswith("-") and not ln.startswith("---")]
    # the only live lines the copy replaces are the Fp8PagedKV call it extends with scratch_slots
    assert all("max(a.gen_tokens, a.ppl_steps)" in ln or ln.strip() in ("-                    device=\"cuda\")",)
               for ln in removed), removed


def test_the_trace_is_the_registered_one():
    """step_decomp's _dynb_plan at the registered batch and segment: 16 rows, the set halving every 32 steps."""
    import ast
    src = (LANE / "step_decomp.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_dynb_plan")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "step_decomp._dynb_plan", "exec"), ns)
    max_new, trace = ns["_dynb_plan"](16, 32)
    assert trace == [16] * 32 + [8] * 32 + [4] * 32 + [2] * 32 + [1] * 32
    assert sorted(max_new) == [33] * 8 + [65] * 4 + [97] * 2 + [129, 161]
    assert sum(trace) == sum(m - 1 for m in max_new) == 992
    with pytest.raises(SystemExit):
        ns["_dynb_plan"](12, 32)
