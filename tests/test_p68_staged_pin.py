"""The p68 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P68).

`bench/p68/p68_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p68/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's own staging `case` rather than searching for files by name.
"""
import hashlib
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p68"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p68_drive.sh's `case`, verbatim."""
    if name in ("p68_run.sh", "p68_probe.py", "p68_reduce.py"):
        return LANE / name
    if name in ("p63_probe.py", "p63_compare.py", "fixture.txt"):
        return REPO / "bench" / "p63" / name
    if name == "prompts_wikitext.json":
        return REPO / "bench" / "p64" / "receipts" / name
    if name == "serve_stack.py":
        return REPO / "bench" / "p44" / name
    return REPO / "bench" / "p39" / name


def _entries():
    for line in PIN.read_text().splitlines():
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
        f"Re-pin it, or the next p68 launch refuses ON A RENTED BOX."
    )


def test_every_staged_piece_the_driver_names_is_pinned():
    driver = (LANE / "p68_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split("=", 1)[1].split()}
    named.discard("staged.sha256")
    pinned = {name for _w, name in _entries()}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"


def test_the_runner_checks_every_piece_it_stages():
    runner = (LANE / "p68_run.sh").read_text()
    line = next(ln for ln in runner.splitlines() if ln.startswith("for f in "))
    checked = set(line.split(" in ", 1)[1].split(";")[0].split())
    checked.discard("staged.sha256")
    pinned = {name for _w, name in _entries()} - {"p68_run.sh"}
    assert checked == pinned, (sorted(checked - pinned), sorted(pinned - checked))


def test_the_egress_probe_runs_the_way_the_fetch_runs():
    """P65 Amendment 2's lesson: a single-stream probe under-read a four-worker fetch 2.3x. The probe and the fetch
    read one worker count; the reading refuses below the floor with rc 13 (host-limited), before any install."""
    runner = (LANE / "p68_run.sh").read_text()
    assert "FETCH_WORKERS=4" in runner and runner.count("max_workers=$FETCH_WORKERS))") == 1
    assert not re.search(r"max_workers=\d", runner)
    assert 'int(os.environ["P68_FETCH_WORKERS"])' in runner and "urllib.request" in runner
    assert runner.index("egress pre-flight") < runner.index('say "install e4b')
    assert re.search(r"REFUSAL; finish 13; fi", runner)


def test_the_size_rows_are_the_committed_p64_rows():
    """The size reading's text is P64's committed wikitext rows (P59's window, digest f67e7e4d): nothing is fetched."""
    import json
    d = json.loads((REPO / "bench" / "p64" / "receipts" / "prompts_wikitext.json").read_text())
    assert d["prompts_sha256"].startswith("f67e7e4d") and len(d["prompts"]) >= 4
    assert all(len(r) >= 512 for r in d["prompts"][:4])
