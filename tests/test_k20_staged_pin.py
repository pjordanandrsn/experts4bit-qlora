"""The k20 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for K20).

`bench/k20/k20_drive.sh` refuses to run when a staged file's sha256 differs from `bench/k20/staged.sha256`;
that guard runs on the CONTROLLER after a box is rented. This test runs the same comparison in CI, where it
costs nothing. It mirrors the driver's own staging rule: the lane's runner resolves to `bench/k20/` and P60's recorded
routing (`eids_b16.int16.{bin,json}`) to `bench/p60/receipts/`; the harness is cloned ON THE BOX from
grouped-nf4-gemm at GNF4_SHA.
"""
import hashlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "k20"
PIN = LANE / "staged.sha256"


def _entries():
    for line in PIN.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


@pytest.mark.skipif(not PIN.exists(), reason="k20 lane not present")
@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and v.endswith(".sh") else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = (REPO / "bench" / "p60" / "receipts" / name) if name.startswith("eids_b16.int16.") else LANE / name
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (
        f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}. "
        f"Re-pin it, or the next k20 launch refuses ON A RENTED BOX."
    )


@pytest.mark.skipif(not PIN.exists(), reason="k20 lane not present")
def test_every_staged_piece_the_driver_names_is_pinned():
    """A file the driver stages but the pin omits is unguarded."""
    driver = (LANE / "k20_drive.sh").read_text()
    stage_line = next(ln for ln in driver.splitlines() if ln.startswith("STAGE="))
    named = {p.rsplit("/", 1)[-1] for p in stage_line.split() if p.endswith((".py", ".json", ".sh", ".bin"))}
    named.discard("staged.sha256")           # the pin file itself rides along; it cannot pin itself
    pinned = {name for _w, name in _entries()}
    missing = {n for n in named if n not in pinned}
    assert not missing, f"staged by the driver but not pinned: {sorted(missing)}"


@pytest.mark.skipif(not PIN.exists(), reason="k20 lane not present")
def test_the_contracts_run_before_the_bench_and_the_rule_self_tests():
    """K17's order: K19's interpreter and compiled contracts gate the bench (rc 21 / 22), and the rule's self-test runs
    before any timing."""
    run = (LANE / "k20_run.sh").read_text()
    interp = run.index("TRITON_INTERPRET=1 perl -e 'alarm 900; exec @ARGV' python -m pytest src/kernel/test_int4_grouped_smallm_interp.py")
    compiled = run.index("TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest src/kernel/test_int4_grouped_smallm_interp.py src/kernel/test_int4_smallm_interp.py")
    selftest = run.index("python $W/k20_bench.py --self-test")
    bench = run.index("python $W/k20_bench.py $W/eids_b16.int16.bin $W/eids_b16.int16.json $W/k20_rows.json")
    assert interp < compiled < selftest < bench
    assert "finish 21; }" in run[interp:compiled] and "finish 22; }" in run[compiled:selftest]
    assert '"gemm_int4_b32_grouped_smallm"' in run and '"build_group_tiles_fused"' in run
