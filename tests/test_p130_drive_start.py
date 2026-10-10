"""P130's driver after a dropped start (`bench/p130/p130_drive.sh`; p130-prove-1, 2026-10-10).

p130-prove-1's box proxy closed the lane-start connection after its banner, so the driver could not tell whether the
child had started. The driver now never resends the start: fresh, read-only connections probe for THIS run's nonce, a
child that bound it is adopted, one that bound it and already exited leaves its files to fetch, and anything else is the
old failure (21). These cases run the real driver against fake `ssh`, `scp` and `rsync` that play the box, and count
the starts it sent: always exactly one.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVE = REPO / "bench" / "p130" / "p130_drive.sh"

FAKE_SSH = r"""#!/bin/bash
LOG="$FAKE_LOG"
in=$(cat)
case "$in" in
  *"nohup env"*)
    echo x >> "$LOG/starts"
    printf '%s' "$in" | sed -n 's/.*P130_RUN_NONCE=\([0-9a-f]*\).*/\1/p' | head -1 > "$LOG/nonce"
    case "$(cat "$LOG/start_mode")" in
      ok) printf 'started:4242\nidentity:1 77 - 500\n'; exit 0;;
      early) echo child-exited-early:rc=9 >&2; exit 125;;
      *) echo "Connection closed by 203.0.113.9 port 19364" >&2; exit 255;;
    esac;;
  *"no-nonce"*)
    echo x >> "$LOG/probes"
    case "$(cat "$LOG/probe_mode")" in
      adopt) printf 'started:4242\nidentity:1 77 - 500\n';;
      gone) echo nonce-bound-gone;;
      *) echo no-nonce;;
    esac
    exit 0;;
esac
case "$*" in *"test -f"*) exit 0;; esac
exit 0
"""
FAKE_RSYNC = r"""#!/bin/bash
for last; do :; done
n=$(cat "$FAKE_LOG/nonce")
mkdir -p "$last"
printf '%s\n' "$n" > "$last/P130_RUN_NONCE"; : > "$last/TP_DONE.$n"; echo 0 > "$last/P130_EXIT_CODE.$n"; : > "$last/P130_SUCCESS.$n"
exit 0
"""


def _run(tmp_path, start_mode, probe_mode="none"):
    bin_, log, run = tmp_path / "bin", tmp_path / "log", tmp_path / "run"
    for d in (bin_, log, run):
        d.mkdir(exist_ok=True)
    (bin_ / "ssh").write_text(FAKE_SSH)
    (bin_ / "scp").write_text("#!/bin/bash\nexit 0\n")
    (bin_ / "rsync").write_text(FAKE_RSYNC)
    for f in ("ssh", "scp", "rsync"):
        (bin_ / f).chmod(0o755)
    (log / "start_mode").write_text(start_mode)
    (log / "probe_mode").write_text(probe_mode)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"schema": "p130-fetch-gate/1", "passed": True, "refusals": [], "e4b_sha": "0" * 40,
                                "generated_at": int(time.time()) - 60}))
    env = {"PATH": f"{bin_}{os.pathsep}/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "FAKE_LOG": str(log), "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1",
           "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts", "E4B_RENT_RUN_DIR": str(run),
           "E4B_RENT_RUN_ID": "p130-start", "E4B_RENT_DEADLINE_EPOCH": str(int(time.time()) + 3600),
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P130_FETCH_GATE": str(gate),
           "P130_START_PROBE_WAIT_S": "0", "P130_POLL_S": "0"}
    out = subprocess.run(["bash", str(DRIVE)], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL,
                         timeout=120)
    count = lambda f: len((log / f).read_text().splitlines()) if (log / f).exists() else 0  # noqa: E731
    return out, count("starts"), count("probes")


@pytest.fixture(autouse=True)
def _posix():
    if os.name == "nt":
        pytest.skip("the driver and its fakes are bash on a POSIX controller")


def test_a_dropped_start_whose_child_bound_the_nonce_is_adopted(tmp_path):
    out, starts, probes = _run(tmp_path, "drop", "adopt")
    assert out.returncode == 0 and "adopting it" in out.stdout and "lane complete rc=0" in out.stdout, out.stdout + out.stderr
    assert (starts, probes) == (1, 1) and "the start is not resent" in out.stdout


def test_a_dropped_start_whose_child_already_exited_is_fetched(tmp_path):
    out, starts, probes = _run(tmp_path, "drop", "gone")
    assert out.returncode == 0 and "fetching what exists" in out.stdout, out.stdout + out.stderr
    assert (starts, probes) == (1, 1) and "lane started; polling" in out.stdout and "TP_DONE seen" not in out.stdout


def test_a_dropped_start_with_no_child_fails_as_before(tmp_path):
    out, starts, probes = _run(tmp_path, "drop", "none")
    assert out.returncode == 21 and "no child bound the nonce (probe: no-nonce)" in out.stdout, out.stdout + out.stderr
    assert (starts, probes) == (1, 1)


def test_a_clean_start_never_probes(tmp_path):
    out, starts, probes = _run(tmp_path, "ok")
    assert out.returncode == 0 and "lane identity: pid=4242" in out.stdout and "probing" not in out.stdout, out.stdout
    assert (starts, probes) == (1, 0)


def test_a_child_that_exited_early_is_not_probed(tmp_path):
    out, starts, probes = _run(tmp_path, "early")
    assert out.returncode == 21 and "start failed (rc=125)" in out.stdout and "probing" not in out.stdout, out.stdout
    assert (starts, probes) == (1, 0)
