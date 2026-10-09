"""The k34 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for K34).

`bench/k34/k34_drive.sh` refuses to run when the runner's sha256 differs from `bench/k34/staged.sha256`. That guard runs
on the CONTROLLER after a box is rented; this test runs the same comparison in CI, where it costs nothing.

It also pins the runner's shape: the refusals come before the install; the tripwire proves the installed gnf4 carries
#522's 32- and 64-row tiles and that k34_bench's SHIPPED plan is the one experts4bit-qlora serves (``plan_smallm`` at both
shapes, the signature's warps and stages); the premise (the small-M GEMM's contract, test_int4_smallm_interp.py compiled:
25 passed, none skipped) comes before the bench; the bench is the gnf4 clone's at GNF4_SHA; every GNF4 decode knob starts unset; no
model is fetched; and the lane's failure codes avoid the launcher's machine-exclusion codes (the disk floor's 13
aside).
"""
import hashlib
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "k34"
RUN = (LANE / "k34_run.sh").read_text()


def _entries():
    for line in (LANE / "staged.sha256").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


def test_the_runner_matches_its_pin():
    entries = list(_entries())
    assert [n for _w, n in entries] == ["k34_run.sh"]
    got = hashlib.sha256((LANE / "k34_run.sh").read_bytes()).hexdigest()
    assert got == entries[0][0], f"k34_run.sh changed without re-pinning ({got[:12]}); the next k34 launch refuses ON A RENTED BOX"


def test_the_runner_shape():
    install = RUN.index('say "install gnf4 @')
    assert RUN.index("finish 15;") < install and RUN.index("finish 13;") < install
    trip = RUN.index("python - <<'PYT'")
    premise = RUN.index("python -m pytest test_int4_smallm_interp.py")
    bench = RUN.index("python $W/k34_bench.py $W/k34.json")
    assert install < trip < premise < bench
    assert "finish 23; }" in RUN[premise:bench]
    assert 'grep -q "25 passed" && ! echo "$LASTL" | grep -q skipped' in RUN[premise:bench]
    body = RUN[trip:premise]
    assert 'tuple(sm._SUPPORTED_BLOCK_M) == (16, 32, 64)' in body and 'commit_id") == os.environ["WANT_GNF4"]' in body
    assert 'sm.plan_smallm(N, K) + (p["warps"].default, p["stages"].default)' in body
    assert "v == tuple(kb.SHIPPED) for v in served.values()" in body and "len(kb.PLANS) == 48" in body
    assert RUN.index("cp $W/src/kernel/k34_bench.py $W/") < trip                       # the tripwire imports the copy
    unset = RUN[RUN.index("unset GNF4_PDL"):RUN.index(": > summary.txt")]
    for knob in ("GNF4_PDL", "GNF4_PDL_MAX_ROWS", "GNF4_GEMV_FUSED_REDUCE", "GNF4_TILE_PROGRAMS", "TRITON_INTERPRET"):
        assert knob in unset, knob
    assert "cp $W/src/kernel/k34_bench.py $W/" in RUN
    assert "snapshot_download" not in RUN and "step_decomp" not in RUN                 # no model


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    # 13 for the disk floor and 18 for the CUDA host floor (a GPU the image's torch cannot use, TC1 amendment 61's class), and
    # nothing else that excludes a machine: 14 and 17 are the launcher's. 18 appears once, in the no-cuda branch, so no other
    # lane failure can name the machine (p115-5090-1 read HARNESS_ERROR on a CUDA-unusable host at rc 10).
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "k34-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "GNF4_SHA": "1" * 40, "K34_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "k34_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/k34"), out.stdout + out.stderr
