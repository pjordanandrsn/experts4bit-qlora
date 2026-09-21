"""The p55x lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P55x).

`bench/p55x/p55x_drive.sh` refuses to run when a staged file's sha256 differs from
`bench/p55x/staged.sha256`. That guard is right and it works -- but it runs on the CONTROLLER at
launch time, after a box has been rented. On 2026-09-20 `p53-gemma4calib` added a family to
`serve_stack.py` without re-pinning, and the refusal arrived on a rented instance ($0.23 and a
burned run id) for a mismatch that is entirely determinable from the repository.

This test moves that check to CI, where it costs nothing. It deliberately mirrors the driver's own
`case` statement rather than searching for files by name: a search picks the wrong `calib.json`
(there are several in bench/) and would re-pin a digest against a file the box never stages --
which would leave the guard green while guarding nothing.
"""
import hashlib
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p55x"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p55x_drive.sh's staging `case`, verbatim: the lane's own runner from bench/p55x, the hook from
    bench/p42, everything else (step_decomp.py, k8_bake.py, calib.json) from bench/p39."""
    if name.startswith("hook/"):
        return REPO / "bench" / "p42" / name
    if name == "p55x_run.sh":
        return LANE / name
    return REPO / "bench" / "p39" / name


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


pytestmark = pytest.mark.skipif(not PIN.exists(), reason="p55x lane not present")


@pytest.mark.parametrize("want,name", list(_entries()) if PIN.exists() else [])
def test_staged_file_matches_its_pin(want, name):
    src = _resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p55x launch refuses ON A RENTED BOX."
    )


def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded."""
    driver = (LANE / "p55x_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1] for p in stage_line.split() if p.endswith((".py", ".json"))}
    named.add("hook/usercustomize.py")   # $HOOK, referenced not copied
    named.add("p55x_run.sh")             # $HERE/p55x_run.sh, the lane's own runner
    pinned = {name for _w, name in _entries()}
    missing = {n for n in named if n not in pinned and f"hook/{n}" not in pinned}
    assert not missing, f"staged by the driver but not pinned: {sorted(missing)}"


def test_the_shared_pieces_are_pinned_to_the_same_bytes_as_p54():
    """P55x stages P39's harness and P42's hook by reference, exactly as P54 does.

    If the two lanes' pins for the SAME source file ever disagree, one of them was re-pinned against
    a file the other does not stage, and at most one of the two launches would be running the harness
    its receipts claim. This asserts they agree rather than merely that each is internally consistent.
    """
    def pins(p: pathlib.Path) -> dict:
        out = {}
        for line in p.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            want, name = line.split(None, 1)
            out[name.strip()] = want
        return out

    p54 = REPO / "bench" / "p54" / "staged.sha256"
    if not p54.exists():
        pytest.skip("p54 lane not present")
    mine, theirs = pins(PIN), pins(p54)
    shared = set(mine) & set(theirs) - {"p54_run.sh", "p55x_run.sh"}
    assert shared, "the two lanes share no staged pieces -- has the staging changed?"
    disagree = {n: (mine[n], theirs[n]) for n in shared if mine[n] != theirs[n]}
    assert not disagree, f"p55x and p54 pin different bytes for the same staged files: {disagree}"


def test_the_prereg_and_the_runner_agree_on_the_recipe():
    """The recipe knobs the pre-registration states must be the ones the box actually exports.

    A pre-registration that names `E4B_CALIB_NSEQ=128` while the runner exports 32 is not a
    pre-registration of the run that happens; the receipt would carry the truth and the registered
    document the fiction, and nothing mechanical would notice.
    """
    run = (LANE / "p55x_run.sh").read_text()
    prereg = (LANE / "P55X-PREREG.md").read_text()
    cal = next(ln for ln in run.splitlines() if ln.startswith("CAL="))
    for knob in ("E4B_SERVE_EXP_INT4_CALIB=1", "E4B_CALIB_NSEQ=128", "E4B_CALIB_LAYERS_PER_PASS=10"):
        assert knob in cal, f"the runner's CAL line does not set {knob}"
        assert knob.split("=")[0] in prereg, f"the pre-registration never names {knob.split('=')[0]}"
    assert "E4B_INT4_HESSIAN_BUDGET_GB=24" in run
    # the gate arms must load the artifact BY FINGERPRINT -- a recipe rebuild there would score bytes
    # that are not the ones being published
    load = next(ln for ln in run.splitlines() if ln.startswith("LOAD="))
    assert "E4B_INT4_ARTIFACT_DIR=" in load and "E4B_INT4_EXPECTED_FINGERPRINT=" in load
    # and the min_rows / damping / budget the lane must NOT move
    assert not re.search(r"E4B_INT4_GPTQ_DAMP=", run), "this lane does not move damping"
    assert not re.search(r"\bmin_rows\s*=", run), "this lane does not move min_rows"
