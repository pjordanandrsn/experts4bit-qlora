"""Lane K29 (grouped-nf4-gemm#71): the staged-file pin, the box script's contract, and the reducer's rule.

`bench/k29/k29_drive.sh` refuses on the controller, after a box is rented, when a staged file differs from
`bench/k29/staged.sha256`; this runs that comparison in CI. The reducer's self-test exercises every branch of the
registered rule (grouped-nf4-gemm kernel/PREREG-k29-pinned-charge-cgroup-v2.md).
"""
import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "k29"
RUN = (LANE / "k29_run.sh").read_text()
DRIVE = (LANE / "k29_drive.sh").read_text()


def _entries():
    for line in (LANE / "staged.sha256").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256((LANE / name).read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning; the launch refuses ON A RENTED BOX"


def test_every_pinned_file_is_staged_by_the_driver():
    staged = {n for _w, n in _entries()}
    assert staged == {"k29_run.sh", "pinned_charge_probe.py"}
    for name in staged:
        assert f"$HERE/{name}" in DRIVE


def test_the_box_script_checks_v2_and_runs_the_registered_sizes_with_no_install():
    assert 'grep -qE " cgroup2 " /proc/mounts && [ -r /sys/fs/cgroup/memory.current ]' in RUN
    assert "SIZES=${K29_SIZES:-0,340,512,700,1024,1359,2048,2100,3000,4096}" in RUN and "REPS=${K29_REPS:-2}" in RUN
    assert "pip install" not in RUN, "the probe needs only the image's torch"
    assert 'perl -e "alarm $PROBE_S; exec @ARGV" python3 pinned_charge_probe.py' in RUN


def test_the_reducer_selftest_covers_every_branch():
    p = subprocess.run([sys.executable, str(LANE / "k29_reduce.py"), "--selftest"], capture_output=True, text=True)
    assert p.returncode == 0 and "selftest: 10/10 ok" in p.stdout, p.stdout + p.stderr
    for v in ("CONFIRMED", "PREMIUM", "VOID", "MIXED"):
        assert f"got {v}" in p.stdout


def test_the_rule_constants_are_the_registered_ones():
    src = (LANE / "k29_reduce.py").read_text()
    assert "R_LO, R_HI = 0.97, 1.06" in src and "CONTROL_LO, CONTROL_HI, CONTROL_R2 = 0.95, 1.10, 0.99" in src


def test_the_driver_retries_the_fetch_before_giving_up_a_finished_run():
    """k29-5090-1 finished on the box and lost its data to ONE rsync that hit a closed connection. The fetch retries,
    then falls back to tar over ssh, before exiting 22."""
    assert "for attempt in 1 2 3 4; do" in DRIVE and 'falling back to tar over ssh' in DRIVE
    assert DRIVE.index("for attempt in 1 2 3 4; do") < DRIVE.index('say "fetch failed: rsync x4 and tar"; exit 22')
