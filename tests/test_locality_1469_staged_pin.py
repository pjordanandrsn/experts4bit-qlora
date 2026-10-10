"""The #1469 locality census's staged-file pin must match the repo (the e4b#642 check, mirrored from K34's).

`bench/locality-1469/locality_drive.sh` refuses to run when a staged file's sha256 differs from
`bench/locality-1469/staged.sha256`. That guard runs on the CONTROLLER after a box is rented; this test runs the same
comparison in CI, where it costs nothing.

It also pins the runner's shape: the host refusals (class, disk, RAM) come before the install; gnf4 is installed from
its clone at GNF4_SHA, whose capture_routing.py the census reuses; the tripwire imports every API the census calls;
the model is fetched at its pinned revision; both self-tests run before the model is loaded; calibrate comes before the
census, and the census before the summaries; every residency knob starts unset; and the failure codes avoid the
launcher's machine-exclusion codes (the disk floor's 13 and the CUDA host floor's 18 aside).
"""
import hashlib
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "locality-1469"
RUN = (LANE / "locality_run.sh").read_text()
STAGED = ["locality_run.sh", "locality_capture.py", "locality_summary.py"]


def _entries():
    for line in (LANE / "staged.sha256").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


def test_the_staged_files_match_their_pin():
    entries = list(_entries())
    assert [n for _w, n in entries] == STAGED
    for want, name in entries:
        got = hashlib.sha256((LANE / name).read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]}); the next census launch refuses ON A RENTED BOX"


def test_the_runner_shape():
    install = RUN.index('say "install experts4bit-qlora[train] @')
    assert '"experts4bit-qlora[train] @ git+' in RUN and '"transformers==5.16.1"' in RUN   # loc-a2000-4: rc 9 without it
    for refusal in ("finish 15;", "finish 13;", "finish 16;"):
        assert RUN.index(refusal) < install, refusal
    clone = RUN.index("git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4")
    gnf4 = RUN.index("--force-reinstall --no-deps $W/gnf4")
    trip = RUN.index("python - <<'PYT'")
    fetch = RUN.index("snapshot_download('$MODEL', revision='$REV')")
    selftest = RUN.index("python $W/locality_capture.py --self-test")
    calib = RUN.index("python $W/locality_capture.py --phase calibrate")
    census = RUN.index("python $W/locality_capture.py --phase census")
    summary = RUN.index("python $W/locality_summary.py --npz $W/out/decode.npz")
    assert install < clone < gnf4 < trip < fetch < selftest < calib < census < summary
    body = RUN[trip:fetch]
    for api in ("attach, flush, hot_sets_from_profile", "enable_pipelined_residency", "load_moe_4bit_streaming"):
        assert api in body, api
    assert "--trace-dir $W/gnf4/bench/cold-engine/routing-trace" in RUN[census:summary]
    assert "REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN
    unset = RUN[RUN.index("unset E4B_EXPERT_PROFILE"):RUN.index(": > summary.txt")]
    for knob in ("E4B_EXPERT_PROFILE", "E4B_RESIDENCY", "E4B_HOT_PROFILE", "E4B_HOT_PER_LAYER", "TRITON_INTERPRET"):
        assert knob in unset, knob


def test_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "loc-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "GNF4_SHA": "1" * 40, "LOC_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "locality_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/loc"), out.stdout + out.stderr
