"""Lane B771b's staged-file pin must match the repo (the e4b#642 check, mirrored for B771b).

`bench/b771b/b771b_drive.sh` refuses to run when a staged file's sha256 differs from `bench/b771b/staged.sha256`; that
guard runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI. It mirrors the driver's
own staging `case`, runs the reducer's self-test (the registered rule on eleven synthetic cases), checks the runner's
arms, and applies the registered mutation to a copy of the fixed shim to prove it still finds its target.
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "b771b"
PIN = LANE / "staged.sha256"


def _resolve(name: str) -> pathlib.Path:
    """b771b_drive.sh's `case`, verbatim."""
    if name in ("b771b_run.sh", "b771b_reduce.py", "mut_b771b.py"):
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
        f"Re-pin it, or the next b771b launch refuses ON A RENTED BOX."
    )


def test_every_staged_piece_the_driver_names_is_pinned():
    driver = (LANE / "b771b_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].strip('"') for p in stage_line.split("=", 1)[1].split()}
    named.discard("staged.sha256")
    pinned = {name for _w, name in _entries()}
    assert named == pinned, f"staged but not pinned: {sorted(named - pinned)}; pinned but not staged: {sorted(pinned - named)}"


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "b771b_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test OK (11 cases)" in out.stdout


def test_the_runner_registers_six_arms_with_their_groupings():
    run = (LANE / "b771b_run.sh").read_text()
    arms = [ln.strip() for ln in run.splitlines() if ln.startswith("arm ")]
    assert arms == ["arm A1 eager default", "arm B1 graph default", "arm B2 graph default",
                    "arm A2 eager default", "arm P padded default", "arm Ad eager device"], arms


def test_the_mutation_finds_its_target_in_the_fixed_shim(tmp_path):
    """mut_b771b.py must apply to the shim as shipped; a refactor that moved the condition would make arm M a no-op."""
    shim = REPO / "experts4bit_qlora" / "engines" / "paged_attention.py"
    dst = tmp_path / "experts4bit_qlora" / "engines"
    dst.mkdir(parents=True)
    (dst / "paged_attention.py").write_text(shim.read_text())
    out = subprocess.run([sys.executable, str(LANE / "mut_b771b.py")], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert '"_g_sel", None) is not None' not in (dst / "paged_attention.py").read_text()
