"""The drivers start the run DETACHED: the remote start command returns as soon as the run is in the background.

``pgx_drive.sh`` and ``p238_drive.sh`` start the box's runner over ssh and then poll its TP_DONE marker with a heartbeat.
Before this fix the start read ``cd $W && VARS nohup bash X_run.sh > outer.log 2>&1 < /dev/null & echo started``. ``&&``
binds tighter than ``&``, so bash on the box backgrounded the whole and-list in a subshell. That subshell kept the ssh
session's stdout open while it waited for the runner, so the ssh (and the driver) did not return until the run ended. The
poll loop, its heartbeats and its "deadline near: fetch what exists" branch never ran: no launch log of either lane has a
heartbeat line. ``cd $W || exit 20;`` first makes the backgrounded job the nohup command alone, with its own redirections.

Each test runs the driver's OWN start command, read from its file, under ``bash -c`` with stdout on a pipe, as sshd does.
``subprocess.run`` returns only at EOF on that pipe, so its wall time is the ssh's. The runner is a stand-in that sleeps
and then writes a file. The command must return at once and the stand-in must still finish; the old form, built from the
same command, must hold the pipe for the stand-in's whole run (the test's own power check).
"""

import re
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = {"pgx": ROOT / "bench" / "pgx" / "pgx_drive.sh", "p238": ROOT / "bench" / "p238" / "p238_drive.sh"}
HOLD_S = 3


def _start_command(text):
    """The remote start command, as bash on the box receives it: the double-quoted $SSH argument of the line that ends
    'echo started" || { say "start failed"; exit 21; }', with its backslash-newline continuations joined."""
    m = re.findall(
        r'^\$SSH "(cd \$W [^"]*?& echo started)" \|\| \{ say "start failed"; exit 21; \}$', text, re.M | re.S
    )
    assert len(m) == 1, "the driver's start line is not where the test expects it"
    return m[0].replace("\\\n", "")


def _run(cmd, work):
    """The command with $W -> work and every other $NAME -> a stand-in value, under bash -c with stdout piped."""
    cmd = cmd.replace("$W", str(work))
    cmd = re.sub(r"\$[A-Z0-9_]+", "x", cmd)
    t0 = time.monotonic()
    r = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    return r, time.monotonic() - t0


def _stand_in(work, lane):
    work.mkdir()
    (work / f"{lane}_run.sh").write_text(f"sleep {HOLD_S}\necho done > ran\n", encoding="utf-8")


@pytest.mark.parametrize("lane", sorted(DRIVERS))
def test_the_start_returns_at_once_and_the_run_goes_on(tmp_path, lane):
    cmd = _start_command(DRIVERS[lane].read_text(encoding="utf-8"))
    assert cmd.startswith("cd $W || exit 20; "), cmd[:60]
    work = tmp_path / "w"
    _stand_in(work, lane)
    r, dt = _run(cmd, work)
    assert r.returncode == 0 and r.stdout == "started\n", (r.returncode, r.stdout, r.stderr)
    assert dt < HOLD_S - 1, f"the start held the pipe {dt:.1f} s: the run is not detached"
    for _ in range(40):
        if (work / "ran").exists():
            break
        time.sleep(0.25)
    assert (work / "ran").read_text() == "done\n", "the run did not go on after the start returned"


@pytest.mark.parametrize("lane", sorted(DRIVERS))
def test_the_old_and_list_form_holds_the_pipe(tmp_path, lane):
    """The power check: the same command with 'cd $W &&' (the form before this fix) holds the pipe until the run ends."""
    cmd = _start_command(DRIVERS[lane].read_text(encoding="utf-8")).replace("cd $W || exit 20; ", "cd $W && ", 1)
    work = tmp_path / "w"
    _stand_in(work, lane)
    r, dt = _run(cmd, work)
    assert r.stdout == "started\n" and dt >= HOLD_S - 0.5, f"the old form returned in {dt:.1f} s"


@pytest.mark.parametrize("lane", sorted(DRIVERS))
def test_a_missing_work_dir_starts_nothing(tmp_path, lane):
    """No work dir: the command exits 20 before any runner starts, so the driver says "start failed" (21)."""
    r, _ = _run(_start_command(DRIVERS[lane].read_text(encoding="utf-8")), tmp_path / "absent")
    assert r.returncode == 20 and r.stdout == "", (r.returncode, r.stdout)
