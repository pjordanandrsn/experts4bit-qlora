"""The K30 lane's staged-file pin must match the repo, and its runner must gate the sweep on correctness.

`bench/k30/k30_drive.sh` refuses to run when a staged file's sha256 differs from `bench/k30/staged.sha256`; that guard
runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it costs nothing. The
harness (`sk_sweep.py`, `k30_sk_check.py`, `k30_reduce.py`) is cloned ON THE BOX from grouped-nf4-gemm at GNF4_SHA
(kernel/PREREG-k30-splitk-r-term-l4.md), so only the runner is staged.
"""
import hashlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "k30"
PIN = LANE / "staged.sha256"


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.skipif(not PIN.exists(), reason="k30 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and v.endswith(".sh") else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = LANE / name
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next k30 launch refuses ON A RENTED BOX."
    )


@pytest.mark.skipif(not PIN.exists(), reason="k30 lane not present")
def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded."""
    driver = (LANE / "k30_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1].rstrip('"') for p in stage_line.split() if p.rstrip('"').endswith((".py", ".json", ".sh"))}
    named.discard("staged.sha256")           # the pin file itself rides along; it cannot pin itself
    pinned = {name for _w, name in _entries()}
    missing = {n for n in named if n not in pinned}
    assert not missing, f"staged by the driver but not pinned: {sorted(missing)}"


@pytest.mark.skipif(not PIN.exists(), reason="k30 lane not present")
def test_correctness_gates_run_before_any_timing_and_prove_exits_first():
    """No timing without correctness (STOP-1): the compiled suite (rc 21), the per-sk check (rc 22), the rule's
    self-test and the plan cross-check all run before the first sweep; the proving run exits before any install."""
    run = (LANE / "k30_run.sh").read_text()
    card = run.index('[ "$GPU" = "$WANT_CARD" ]')
    prove = run.index('[ "${K30_PROVE:-0}" = 1 ]')
    install = run.index("python -m pip install -q --no-input --force-reinstall --no-deps")
    compiled = run.index("python -m pytest src/kernel/test_int4_b32.py")
    skcheck = run.index("python $W/k30_sk_check.py")
    selftest = run.index("python $W/k30_reduce.py --self-test")
    crosscheck = run.index("k30_reduce.check_installed_plan()")
    sweep = run.index("python $W/sk_sweep.py")
    reduce = run.index("python $W/k30_reduce.py $W/rows $W/k30_verdict.json --check-installed-plan")
    assert card < prove < install < compiled < skcheck < selftest < crosscheck < sweep < reduce
    assert "finish 21; }" in run[compiled:skcheck] and "finish 22; }" in run[skcheck:selftest]
    assert "OUT_PREFIX=$pass" in run and "for pass in k30p1 k30p2" in run


@pytest.mark.skipif(not PIN.exists(), reason="k30 lane not present")
def test_the_card_check_is_exact_not_a_substring():
    """"NVIDIA L40S" contains "L4": a substring test would accept the wrong class. The token maps to the two
    registered cards' exact names and refuses anything else."""
    run = (LANE / "k30_run.sh").read_text()
    assert '[ "$GPU" = "$WANT_CARD" ]' in run and "*L4*" not in run
    assert 'L4) WANT_CARD="NVIDIA L4";;' in run and 'A4000) WANT_CARD="NVIDIA RTX A4000";;' in run
    assert "is not a registered card (L4 | A4000)\"; finish 78;;" in run
    assert "K30_CARD=${K30_CARD:-L4}" in (LANE / "k30_drive.sh").read_text()
