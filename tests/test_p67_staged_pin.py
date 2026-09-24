"""The p67 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P67).

`bench/p67/p67_drive.sh` refuses to launch when a staged file's sha256 differs from `bench/p67/staged.sha256`, and
`bench/p67/p67_run.sh` refuses again ON THE BOX -- both after a box is rented. This test runs the same comparison in
CI, where it costs nothing. It mirrors the driver's staging rule: the lane's guard and knobs resolve to `bench/p67/`,
tp4's harness to `bench/tp4/`, the two dataset helpers to `bench/flagship-matrix/`. Once the registered draw has run
and its results are committed, these pins are the record of what it ran, and a later change to tp4's harness must
re-pin (bench/p67/pin.sh) or say in the lane that the draw is done.
"""
import hashlib
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p67"
PIN = LANE / "staged.sha256"
SRC = {"p67_run.sh": LANE, "registered.knobs": LANE, "n9_datasets.py": REPO / "bench" / "flagship-matrix" / "drivers",
       "ds_manifest.json": REPO / "bench" / "flagship-matrix"}


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _src(name):
    return SRC.get(name, REPO / "bench" / "tp4") / name


@pytest.mark.skipif(not PIN.exists(), reason="p67 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and not re.fullmatch(r"[0-9a-f]{64}", v) else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = _src(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
                         f"Re-pin (bench/p67/pin.sh), or the next p67 launch refuses ON A RENTED BOX.")


@pytest.mark.skipif(not PIN.exists(), reason="p67 lane not present")
def test_every_file_the_box_is_staged_is_pinned():
    """A staged file the pin omits is unguarded. The box is staged tp4_drive.sh's STAGE plus p67_drive.sh's extras."""
    drive4 = (REPO / "bench" / "tp4" / "tp4_drive.sh").read_text()
    stage_line = next(ln for ln in drive4.splitlines() if ln.startswith("STAGE=\"$HERE/tp4_run.sh"))
    named = {p.rsplit("/", 1)[-1].rstrip('"') for p in stage_line.split() if p.rstrip('"').endswith((".py", ".json", ".sh"))}
    drive67 = (LANE / "p67_drive.sh").read_text()
    extra = re.search(r'TP4_EXTRA_STAGE="([^"]+)"', drive67).group(1)
    named |= {p.rsplit("/", 1)[-1] for p in extra.split()}
    named.discard("staged.sha256")           # the pin file rides along; it cannot pin itself
    pinned = {name for _w, name in _entries()}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"
