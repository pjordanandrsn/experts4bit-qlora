"""The p47 lane's staged-file pin must match the repo (e4b#642).

`bench/p47/p47_drive.sh` refuses to run when a staged file's sha256 differs
from `bench/p47/staged.sha256`. That guard is right and it works -- but it runs
on the CONTROLLER at launch time, after a box has been rented. On 2026-09-20
`p53-gemma4calib` added a family to `serve_stack.py` without re-pinning, and
the refusal arrived on a rented instance ($0.23 and a burned run id) for a
mismatch that is entirely determinable from the repository.

This test moves that check to CI, where it costs nothing. It deliberately
mirrors the driver's own `case` statement rather than searching for files by
name: a search picks the wrong `calib.json` (there are several in bench/) and
would re-pin a digest against a file the box never stages -- which would leave
the guard green while guarding nothing.
"""
import hashlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PIN = REPO / "bench" / "p47" / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p47_drive.sh:19, verbatim."""
    if name.startswith("hook/"):
        return REPO / "bench" / "p42" / name
    if name == "p47_run.sh":
        return REPO / "bench" / "p47" / name
    if name in ("serve_stack.py", "kl_serve.py"):
        return REPO / "bench" / "p44" / name
    if name == "act_probe.py":
        return REPO / "bench" / "p49" / name
    if name.startswith("kl_") and name.endswith(".py"):
        return REPO / "bench" / name
    return REPO / "bench" / "p39" / name


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.skipif(not PIN.exists(), reason="p47 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "/" in v or (isinstance(v, str) and v.endswith((".py", ".json"))) else "")
def test_staged_file_matches_its_pin(want, name):
    src = _resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p47 launch refuses ON A RENTED BOX."
    )


@pytest.mark.skipif(not PIN.exists(), reason="p47 lane not present")
def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded."""
    driver = (REPO / "bench" / "p47" / "p47_drive.sh").read_text()
    stage_line = next(l for l in driver.splitlines() if l.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1] for p in stage_line.split() if p.endswith((".py", ".json"))}
    named.add("hook/usercustomize.py")           # $HOOK, referenced not copied
    pinned = {name for _w, name in _entries()}
    missing = {n for n in named if n not in pinned and f"hook/{n}" not in pinned}
    assert not missing, f"staged by the driver but not pinned: {sorted(missing)}"
