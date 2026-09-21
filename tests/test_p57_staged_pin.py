"""The p57 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P57).

`bench/p57/p57_drive.sh` refuses to run when a staged file's sha256 differs
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
PIN = REPO / "bench" / "p57" / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p57_drive.sh's staging `case`, verbatim: the runner, the instrument and the P57 hook from bench/p57;
    everything else (step_decomp.py, k8_bake.py, calib.json) from bench/p39."""
    if name.startswith("hook/") or name in ("p57_run.sh", "distinct_experts.py"):
        return REPO / "bench" / "p57" / name
    return REPO / "bench" / "p39" / name


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.skipif(not PIN.exists(), reason="p57 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "/" in v or (isinstance(v, str) and v.endswith((".py", ".json"))) else "")
def test_staged_file_matches_its_pin(want, name):
    src = _resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p57 launch refuses ON A RENTED BOX."
    )


@pytest.mark.skipif(not PIN.exists(), reason="p57 lane not present")
def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded."""
    driver = (REPO / "bench" / "p57" / "p57_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1] for p in stage_line.split() if p.endswith((".py", ".json", ".sh"))}
    named.add("hook/usercustomize.py")           # $HOOK, referenced not copied
    named.add("p57_run.sh")                      # $HERE/p57_run.sh, the lane's own runner
    named.discard("staged.sha256")               # rides along; cannot pin itself
    pinned = {name for _w, name in _entries()}
    missing = {n for n in named if n not in pinned and f"hook/{n}" not in pinned}
    assert not missing, f"staged by the driver but not pinned: {sorted(missing)}"


def test_every_serve_stack_family_names_its_reference():
    """e4b#647: a family in serve_stack that kl_serve does not know is unrunnable.

    p53-gemma4calib-21 reached the box, passed K0, fetched 52 GB of checkpoint
    over 24 minutes, baked, and then died in 19 seconds on:

        kl_serve.py: error: argument --family: invalid choice: 'gemma4calib'

    #640 added the family to serve_stack's ARMS and MODELS and tested both, but
    `--family` takes its choices from kl_serve's REFERENCE map, which is a
    SEPARATE declaration. Two places had to agree and only one was updated, so
    the defect was invisible until a rented box had done all the expensive work.

    REFERENCE is not merely a choices list -- it is where each family states
    what its KL is measured against, which this file's own rule requires of
    every row. So the missing entry was a missing disclosure as well as a bug,
    and deriving the choices from ARMS would have papered over that by letting
    a family run without naming its reference. The right invariant is that the
    two agree, which is what this asserts.
    """
    import importlib.util
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "bench" / "p44"
    spec = importlib.util.spec_from_file_location("serve_stack_mod", root / "serve_stack.py")
    stack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stack)

    # kl_serve imports torch at module scope, so read REFERENCE's keys textually.
    src = (root / "kl_serve.py").read_text()
    block = src[src.index("REFERENCE = {"):]
    block = block[:block.index("\n}")]
    referenced = set(re.findall(r'^\s{4}"([A-Za-z0-9_]+)":', block, re.M))

    # PRE-EXISTING, and deliberately not "fixed" here. These three are in ARMS but
    # have never had a REFERENCE entry, so `--family granite` would refuse exactly
    # as gemma4calib did. I have NOT verified what their KL reference ought to be,
    # and inventing prose to turn the test green would be worse than the gap: this
    # file's rule is that every row names its reference, and a fabricated one is a
    # false disclosure rather than a missing one. Tracked in e4b#647.
    KNOWN_UNSCORED = {"granite", "mixtral", "olmoe"}

    missing = sorted(set(stack.ARMS) - referenced - KNOWN_UNSCORED)
    assert not missing, (
        f"serve_stack offers {missing} but kl_serve's REFERENCE does not name them: "
        "--family would refuse AFTER the box has fetched the checkpoint. Add a REFERENCE "
        "entry saying what that family's KL is measured against -- not to satisfy this test, "
        "but because every row must name its reference."
    )
    # and the known gap must not silently grow into the referenced set either
    assert KNOWN_UNSCORED.isdisjoint(referenced), (
        "a family listed as known-unscored now has a REFERENCE entry -- remove it from "
        "KNOWN_UNSCORED so the invariant covers it"
    )
