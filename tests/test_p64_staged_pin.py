"""The p64 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P64 (e4b#709)).

`bench/p64/p64_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p64/staged.sha256`. That
guard runs on the CONTROLLER after a box is rented; this test runs the same comparison in CI, where it costs
nothing. It mirrors the driver's own `case` statement rather than searching for files by name: a search picks the
wrong `calib.json` (there are several in bench/) and would re-pin a digest against a file the box never stages.
"""
import hashlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p64"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """p64_drive.sh's staging `case`, verbatim: the runner, the scorer and the reducer from bench/p64; P59's scorer
    and KL reader from bench/p59; serve_stack.py from bench/p44; kl_fidelity.py from bench/; the hook from
    bench/p42; everything else (step_decomp.py, k8_bake.py, calib.json) from bench/p39."""
    if name in ("p64_run.sh", "kl_a16.py", "p64_reduce.py"):
        return LANE / name
    if name in ("kl_b16.py", "p59_reduce.py"):
        return REPO / "bench" / "p59" / name
    if name == "serve_stack.py":
        return REPO / "bench" / "p44" / name
    if name == "kl_fidelity.py":
        return REPO / "bench" / name
    if name.startswith("hook/"):
        return REPO / "bench" / "p42" / name
    return REPO / "bench" / "p39" / name


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.skipif(not PIN.exists(), reason="p64 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and ("/" in v or v.endswith((".py", ".json", ".sh"))) else "")
def test_staged_file_matches_its_pin(want, name):
    src = _resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next p64 launch refuses ON A RENTED BOX."
    )


@pytest.mark.skipif(not PIN.exists(), reason="p64 lane not present")
def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded."""
    driver = (LANE / "p64_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].rstrip('"') for p in stage_line.split() if p.rstrip('"').endswith((".py", ".json", ".sh"))}
    named.add("hook/usercustomize.py")           # $HOOK, referenced not copied
    named.discard("staged.sha256")               # rides along; cannot pin itself
    pinned = {name for _w, name in _entries()}
    missing = {n for n in named if n not in pinned}
    assert not missing, f"staged by the driver but not pinned: {sorted(missing)}"


@pytest.mark.skipif(not PIN.exists(), reason="p64 lane not present")
def test_the_runner_checks_every_piece_it_needs():
    """The box-side runner's own presence check names every pinned file (a pinned file the runner never
    checks for would reach `sha256sum -c` only by luck of staging)."""
    runner = (LANE / "p64_run.sh").read_text()
    line = next(ln for ln in runner.splitlines() if ln.startswith("for f in kl_a16.py"))
    checked = set(line.split(" in ", 1)[1].split(";")[0].split())
    pinned = {name for _w, name in _entries()} | {"staged.sha256"}
    assert pinned - {"p64_run.sh"} <= checked, f"pinned but never checked by the runner: {sorted(pinned - {'p64_run.sh'} - checked)}"
