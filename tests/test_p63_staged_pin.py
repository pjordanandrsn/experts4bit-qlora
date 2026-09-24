"""The p63 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P63).

`bench/p63/p63_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p63/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's own staging `case` rather than searching for files by name (several `calib.json` exist under
bench/; a search could re-pin a digest against a file the box never stages).
"""
import hashlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p63"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p63_drive.sh's `case`, verbatim: the lane's own pieces from bench/p63, serve_stack.py from bench/p44, everything
    else (k8_bake.py, calib.json) from bench/p39."""
    if name in ("p63_run.sh", "p63_probe.py", "p63_compare.py", "p63_reduce.py", "fixture.txt"):
        return LANE / name
    if name == "serve_stack.py":
        return REPO / "bench" / "p44" / name
    return REPO / "bench" / "p39" / name


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.skipif(not PIN.exists(), reason="p63 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v and len(v) < 40 else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = _resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p63 launch refuses ON A RENTED BOX."
    )


@pytest.mark.skipif(not PIN.exists(), reason="p63 lane not present")
def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded; one the pin names but the driver does not stage makes
    the box's `sha256sum -c` fail on a missing file."""
    driver = (LANE / "p63_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split("=", 1)[1].split()}
    named.discard("staged.sha256")
    pinned = {name for _w, name in _entries()}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"


@pytest.mark.skipif(not PIN.exists(), reason="p63 lane not present")
def test_the_runner_checks_every_staged_piece_it_uses():
    """p63_run.sh's STAGE MISSING loop names the pieces it runs; each must be pinned (the runner's sha256sum -c covers
    exactly the pin, so an unpinned piece it uses would run unverified)."""
    runner = (LANE / "p63_run.sh").read_text()
    loop = next(ln for ln in runner.splitlines() if ln.strip().startswith("for f in p63_probe.py"))
    used = set(loop.split(" in ", 1)[1].split(";")[0].split())
    used.discard("staged.sha256")
    pinned = {name for _w, name in _entries()}
    assert used <= pinned, f"used on the box but not pinned: {sorted(used - pinned)}"
