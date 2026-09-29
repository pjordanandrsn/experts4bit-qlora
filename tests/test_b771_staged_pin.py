"""Lane B771's staged-file pin must match the repo (the e4b#642 check, mirrored for B771).

`bench/b771/b771_drive.sh` refuses to run when a staged file's sha256 differs from `bench/b771/staged.sha256`; that
guard runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI. It mirrors the driver's
own staging `case`, runs the reducer's self-test (the registered rule on nine synthetic cases), and checks the arms
the runner registers.
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "b771"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """b771_drive.sh's `case`, verbatim."""
    if name in ("b771_run.sh", "b771_reduce.py", "b771_bytes.py"):
        return LANE / name
    if name == "step_decomp.py":
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
        f"Re-pin it, or the next b771 launch refuses ON A RENTED BOX."
    )


def test_every_staged_piece_the_driver_names_is_pinned():
    driver = (LANE / "b771_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split("=", 1)[1].split()}
    named.discard("staged.sha256")
    pinned = {name for _w, name in _entries()}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "b771_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (9 cases)" in out.stdout


def test_the_runner_registers_four_arms_across_the_swap_all_with_the_device_grouping():
    run = (LANE / "b771_run.sh").read_text()
    calls = [ln.strip() for ln in run.splitlines() if ln.startswith(("arm ", "bytes_stage ", "say \"swap"))]
    assert calls == ["bytes_stage old", "arm A_old eager old", "arm P_old padded old",
                     'say "swap grouped-nf4-gemm to the new cut @$GNF4_NEW_SHA"',
                     "bytes_stage new", "arm P_new padded new", "arm A_new eager new"], calls
    assert "--dynb-grouping device" in run
