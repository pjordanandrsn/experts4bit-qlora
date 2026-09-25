"""The p70 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P70).

`bench/p70/p70_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p70/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. It
mirrors the driver's own staging `case` rather than searching for files by name. P70 imports P64's scorer and reducer
unchanged, so their pins must also equal P64's own: a drift in either lane is caught here.
"""
import hashlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p70"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p70_drive.sh's `case`, verbatim."""
    if name in ("p70_run.sh", "kl_router.py", "p70_reduce.py"):
        return LANE / name
    if name in ("kl_a16.py", "p64_reduce.py"):
        return REPO / "bench" / "p64" / name
    if name in ("kl_b16.py", "p59_reduce.py"):
        return REPO / "bench" / "p59" / name
    if name == "serve_stack.py":
        return REPO / "bench" / "p44" / name
    if name == "kl_fidelity.py":
        return REPO / "bench" / name
    if name.startswith("hook/"):
        return REPO / "bench" / "p42" / name
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
        f"Re-pin it, or the next p70 launch refuses ON A RENTED BOX."
    )


def test_every_staged_piece_the_driver_names_is_pinned():
    driver = (LANE / "p70_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split("=", 1)[1].split()}
    named.discard("staged.sha256")
    pinned = {name for _w, name in _entries()} - {"hook/usercustomize.py"}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"


def test_the_runner_checks_every_piece_it_stages():
    runner = (LANE / "p70_run.sh").read_text()
    line = next(ln for ln in runner.splitlines() if ln.startswith("for f in kl_router.py"))
    checked = set(line.split(" in ", 1)[1].split(";")[0].split())
    # the runner does not test its own presence (it is running); `sha256sum -c staged.sha256` covers its bytes
    pinned = ({name for _w, name in _entries()} | {"staged.sha256"}) - {"p70_run.sh"}
    assert checked == pinned, (sorted(checked - pinned), sorted(pinned - checked))
    assert "sha256sum -c staged.sha256" in runner


def test_the_pieces_shared_with_p64_carry_p64s_own_pins():
    """kl_a16.py and p64_reduce.py (and P64's own reused pieces) are the SAME bytes P64 registered and ran."""
    p64 = dict((n, w) for w, n in _entries(REPO / "bench" / "p64" / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    shared = set(p64) & set(mine)
    assert {"kl_a16.py", "p64_reduce.py", "kl_b16.py", "serve_stack.py", "step_decomp.py"} <= shared
    assert all(p64[n] == mine[n] for n in shared), {n: (p64[n][:12], mine[n][:12]) for n in shared if p64[n] != mine[n]}


def test_the_rental_forwards_no_knob_that_changes_what_is_measured():
    driver = (LANE / "p70_drive.sh").read_text()
    line = next(ln for ln in driver.splitlines() if ln.startswith("for v in P70_"))
    forwarded = set(line.split(" in ", 1)[1].split(";")[0].split())
    assert forwarded == {"P70_FIRST_CHUNK_S", "P70_PROVE"}, forwarded
