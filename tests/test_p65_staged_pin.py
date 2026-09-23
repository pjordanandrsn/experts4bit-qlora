"""The p65 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P65).

`bench/p65/p65_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p65/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's own `case` statement rather than searching for files by name (a search finds the wrong
`calib.json`: there are several under bench/): the lane's own pieces resolve to `bench/p65/`, P44's census row, served-
model builder and tail statistic to `bench/p44/`, P39's bake and placement calibration to `bench/p39/`.
"""
import hashlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p65"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p65_drive.sh's staging `case`, verbatim."""
    if name.startswith("p65_") or name == "expert_entropy.py":
        return LANE / name
    if name in ("expert_residuals.py", "serve_stack.py", "p44_reduce.py"):
        return REPO / "bench" / "p44" / name
    return REPO / "bench" / "p39" / name


def _entries():
    if not PIN.exists():                     # parametrize runs at collection, before skipif can apply
        return
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.skipif(not PIN.exists(), reason="p65 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = _resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p65 launch refuses ON A RENTED BOX."
    )


@pytest.mark.skipif(not PIN.exists(), reason="p65 lane not present")
def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded."""
    driver = (LANE / "p65_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split() if p.strip('"').endswith((".py", ".json", ".sh"))}
    named.discard("staged.sha256")           # the pin file itself rides along; it cannot pin itself
    pinned = {name for _w, name in _entries()}
    missing = {n for n in named if n not in pinned}
    assert not missing, f"staged by the driver but not pinned: {sorted(missing)}"


@pytest.mark.skipif(not PIN.exists(), reason="p65 lane not present")
def test_the_runner_checks_every_piece_it_needs():
    """p65_run.sh's own STAGE MISSING list and the driver's STAGE line name the same files."""
    driver = (LANE / "p65_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    staged = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split("=", 1)[1].split()}
    runner = (LANE / "p65_run.sh").read_text()
    need_line = next(ln for ln in runner.splitlines() if ln.startswith("for f in p65_run.sh"))
    need = set(need_line.split(" in ", 1)[1].split(";", 1)[0].split())
    assert staged == need, (sorted(staged ^ need))
