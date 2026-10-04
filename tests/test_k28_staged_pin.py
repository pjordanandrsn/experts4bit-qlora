"""The k28 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for K28).

`bench/k28/k28_drive.sh` refuses to run when the runner's sha256 differs from `bench/k28/staged.sha256`. That guard runs
on the CONTROLLER after a box is rented; this test runs the same comparison in CI, where it costs nothing.

It also pins the runner's shape: the refusals come before the install; the tripwire proves the installed gnf4 carries
the GNF4_PDL switch and reads it as active on the card; the premise (the switch's contract, test_pdl.py compiled: 23
passed, none skipped) comes before the bench; the bench is the gnf4 clone's at GNF4_SHA; no model is fetched; and the
lane's failure codes avoid the launcher's machine-exclusion codes (the disk floor's 13 aside).
"""
import hashlib
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "k28"
RUN = (LANE / "k28_run.sh").read_text()


def _entries():
    for line in (LANE / "staged.sha256").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


def test_the_runner_matches_its_pin():
    entries = list(_entries())
    assert [n for _w, n in entries] == ["k28_run.sh"]
    got = hashlib.sha256((LANE / "k28_run.sh").read_bytes()).hexdigest()
    assert got == entries[0][0], f"k28_run.sh changed without re-pinning ({got[:12]}); the next k28 launch refuses ON A RENTED BOX"


def test_the_runner_shape():
    install = RUN.index('say "install gnf4 @')
    assert RUN.index("finish 15;") < install and RUN.index("finish 13;") < install
    trip = RUN.index("python - <<'PYT'")
    premise = RUN.index("python -m pytest test_pdl.py")
    bench = RUN.index("python $W/k28_bench.py $W/k28.json")
    assert install < trip < premise < bench
    assert "finish 23; }" in RUN[premise:bench]
    assert 'grep -q "23 passed" && ! echo "$LASTL" | grep -q skipped' in RUN[premise:bench]
    body = RUN[trip:premise]
    assert 'int4_b32.pdl_refresh() is True and int4_b32.pdl_active("cuda") is True' in body and '"launch_pdl" in CUDAOptions.__dataclass_fields__' in body
    assert "cp $W/src/kernel/k28_bench.py $W/" in RUN
    assert "snapshot_download" not in RUN and "step_decomp" not in RUN                 # no model


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "k28-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "GNF4_SHA": "1" * 40, "K28_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "k28_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/k28"), out.stdout + out.stderr
