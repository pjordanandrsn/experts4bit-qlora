"""Execute the shared Linux probe and each driver's poll loop without a rental."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bench/common/lane_liveness.sh"
DRIVERS = [ROOT / f"bench/{folder}/{script}_drive.sh" for folder, script in
           [("tc1", "tc1"), ("p127", "p127"), ("fam", "fam"), ("locality-1469", "locality")]]
BOOT_A, BOOT_B = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
INITIAL = f"1 123 {BOOT_A} 10000.00"


def bash(code, *, env=None):
    return subprocess.run(["bash", "-c", f'source "{HELPER}"\n' + code],
                          env=env, text=True, capture_output=True, timeout=10)


def fake_proc(tmp_path, *, state="S", ticks=123, uptime=10000.0, boot=BOOT_A, present=True):
    root = tmp_path / "proc"
    (root / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
    (root / "uptime").write_text(f"{uptime} 0.0\n")
    (root / "sys/kernel/random/boot_id").write_text(boot + "\n")
    if present:
        (root / "4242").mkdir(exist_ok=True)
        # Kernel fields 3..21 then field22, with a deliberately awkward comm.
        (root / "4242/stat").write_text(f"4242 (bash odd ) name) {state} " + "0 " * 18 + f"{ticks} 0\n")
    return root


def snapshot(proc, env=None):
    result = subprocess.run(["bash", str(HELPER), "--probe", "4242", str(proc)],
                            env=env, text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def verdict(current, initial=INITIAL, age=3600):
    result = bash(f"lane_snapshot_verdict '{current}' '{initial}' '{age}'")
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_fake_pgrep_self_matches_but_probe_reports_missing(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pgrep = bin_dir / "pgrep"
    pgrep.write_text('#!/bin/sh\necho 111\necho 222\n')
    pgrep.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    old = subprocess.run(["bash", "-c", "pgrep -f 'bash p127_run.sh' | wc -l"],
                         env=env, capture_output=True, text=True, check=True)
    assert int(old.stdout) == 2  # reproduces the old driver's misleading 'live 2'
    current = snapshot(fake_proc(tmp_path, present=False), env)
    assert current.startswith("0 - ") and verdict(current) == "0"
    assert bash('lane_two_missing 0 0').stdout.strip() == "dead"


@pytest.mark.parametrize("state", ["S", "R", "D"])
def test_healthy_long_run_and_exec_keep_the_launch_identity(tmp_path, state):
    # D (uninterruptible I/O) is alive; an idle/fetching run is never killed as a stall.
    current = snapshot(fake_proc(tmp_path, state=state, uptime=50000))
    assert verdict(current, age=40000) == "1"


def test_pid_reuse_is_missing_even_though_process_exists(tmp_path):
    assert verdict(snapshot(fake_proc(tmp_path, ticks=999))) == "0"


@pytest.mark.parametrize("state", ["Z", "X"])
def test_zombie_and_dead_states_are_missing(tmp_path, state):
    assert verdict(snapshot(fake_proc(tmp_path, state=state))) == "0"


def test_boot_id_change_detects_reboot_even_with_same_pid_ticks(tmp_path):
    assert verdict(snapshot(fake_proc(tmp_path, boot=BOOT_B, uptime=20000))) == "reboot"


def test_incident_uptime_below_lane_age_detects_reboot_without_boot_id():
    # p127-prove-2: started20:40:29, sampled21:41:37 (3668s); uptime3124.65s.
    assert verdict("1 123 - 3124.65", "1 123 - 100.00", age=3668) == "reboot"


def test_uptime_reset_detects_reboot_before_it_falls_below_lane_age():
    assert verdict("1 123 - 9000.00", "1 123 - 10000.00", age=10) == "reboot"


@pytest.mark.parametrize("corrupt", ["", "ssh: connection lost", "0 - bad 10000", "1 123 - nope",
                                    "0 - - 10000 extra", "999 123 - 10000"])
def test_malformed_probe_is_unknown_and_resets_missing_streak(corrupt):
    assert verdict(corrupt) == ""
    # A definite zero on either side of unknown must not combine into death.
    assert bash('lane_two_missing "" 0; lane_two_missing 0 ""; true').stdout == ""


def test_unreadable_existing_proc_is_unknown(tmp_path):
    proc = fake_proc(tmp_path)
    (proc / "4242/stat").unlink()
    assert snapshot(proc).startswith("unknown - ")
    assert verdict(snapshot(proc)) == ""


def test_failed_probe_read_is_not_a_definite_zero(tmp_path):
    proc = fake_proc(tmp_path)
    (proc / "uptime").write_text("bad data\n")
    result = subprocess.run(["bash", str(HELPER), "--probe", "4242", str(proc)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and not result.stdout


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda p: p.parent.name)
@pytest.mark.parametrize("sequence,expected,polls", [
    (["0 - - 10000", "0 - - 10001"], "dead", 2),
    ([f"1 999 {BOOT_A} 10000", f"1 999 {BOOT_A} 10001"], "dead", 2),
    ([f"1 123 {BOOT_B} 20000"], "reboot", 1),
    (["1 123 - 3124.65"], "reboot", 1),
    (["0 - - 10000", None, "0 - - 10001", INITIAL], "complete", 4),
    ([INITIAL, INITIAL, INITIAL, INITIAL], "complete", 4),
    (["0 - - 10000", "PARTIAL_FAIL:0 - - 10000", "0 - - 10001", INITIAL], "complete", 4),
    (["0 - - 10000 extra", "ssh failed", INITIAL, INITIAL], "complete", 4),
])
def test_actual_driver_poll_loop(driver, sequence, expected, polls, tmp_path):
    """Run actual adoption/decision code: SSH failure never ends a healthy lane."""
    text = driver.read_text()
    start = text.index('LAST=""; LAST_CHANGE=')
    end = text.index('\ndone\n', start) + len('\ndone\n')
    loop = text[start:end]
    # Functions come from the driver's actual code, with its progress knobs set below.
    funcs = re.search(r"^(?:tc1_)?progress_verdict\(\).*?(?=^say \"lane started|^# e4b#835)",
                      text, re.MULTILINE | re.DOTALL).group(0)
    values = tmp_path / "values"
    values.write_text("\n".join("TRANSPORT_FAIL" if x is None else x for x in sequence) + "\n")
    count = tmp_path / "count"
    count.write_text("0")
    # Each SSH shell gets its own process, so persist poll count in a file.
    prelude = f'''
source "{HELPER}"
LANE_HELPER="{HELPER}"; LANE_PID=4242; LANE_INITIAL="{INITIAL}"; LANE_STARTED_AT=0
DEADLINE=999999; POLL=0; STALL_S=900; TC1_MIN_PROGRESS_MB=16; P127_MIN_PROGRESS_MB=16; FAM_MIN_PROGRESS_MB=16; LOC_MIN_PROGRESS_MB=16
W=/fake; NONCE=nonce; RUN_DIR=/unused
say(){{ printf '%s\\n' "$*"; }}
date(){{ echo 4000; }}
sleep(){{ :; }}
tc1_partial_fetch(){{ return 0; }}
fake_ssh(){{
  case "$*" in
    *"test -f"*) [ "$(cat '{count}')" -ge {len(sequence)} ];;
    *"--probe"*)
      cat >/dev/null
      n=$(cat '{count}'); n=$((n+1)); echo "$n" > '{count}'
      row=$(sed -n "${{n}}p" '{values}')
      [ "$row" != TRANSPORT_FAIL ] || return 255
      case "$row" in PARTIAL_FAIL:*) printf '%s\\n' "${{row#PARTIAL_FAIL:}}"; return 255;; esac
      printf '%s\\n' "$row";;
    *) echo 'working | gpu 0,0 | du 0M | disk 10G | dfk 10000';;
  esac
}}
SSH=fake_ssh
'''
    result = subprocess.run(["bash", "-uc", prelude + funcs + loop + '\nprintf "END:%s\\n" "$LANE_DEAD"'],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert int(count.read_text()) == polls
    assert ("END:0" if expected == "complete" else "END:1") in result.stdout
    if expected == "reboot":
        assert "host rebooted" in result.stdout
    assert "pgrep" not in loop and "lane_snapshot_verdict" in loop


def test_all_four_launches_capture_pid_and_snapshot_without_staging_helper():
    for driver in DRIVERS:
        text = driver.read_text()
        assert 'identity=\\$(lane_proc_snapshot \\$child)' in text
        assert 'echo started:\\$child; echo identity:\\$identity' in text
        assert 'source "$LANE_HELPER"' in text and 'cat "$LANE_HELPER"' in text
        assert 'LANE_HELPER' not in next(x for x in text.splitlines() if x.startswith('STAGE='))
        assert "pgrep -f" not in text
        subprocess.run(["bash", "-n", str(driver)], check=True)


@pytest.mark.skipif(not Path('/proc/self/stat').is_file(), reason='real launch identity needs Linux /proc')
@pytest.mark.parametrize('driver', DRIVERS, ids=lambda p: p.parent.name)
def test_actual_launch_handshake_captures_linux_child_identity(driver, tmp_path):
    """Exercise the real streamed launch snippet, including a TC1 guard exec."""
    text = driver.read_text()
    start = text.index('LANE_STARTED_AT=')
    end = text.index('\nsay "lane identity:', start)
    launch = text[start:end]
    runner = driver.name.replace('_drive.sh', '_run.sh')
    prefix = {'tc1': 'TC1', 'p127': 'P127', 'fam': 'FAM', 'locality-1469': 'LOC'}[driver.parent.name]
    (tmp_path / runner).write_text(f'#!/bin/bash\nprintf nonce > {prefix}_RUN_NONCE\nsleep 2\n')
    (tmp_path / 'guard.sh').write_text(f'#!/bin/bash\nexec bash {runner}\n')
    setup = f'''source "{HELPER}"
LANE_HELPER="{HELPER}"; W="{tmp_path}"; PASS=""; NONCE=nonce; RUNNER=guard.sh
say(){{ echo "$*"; }}
local_ssh(){{ "$@"; }}
SSH=local_ssh
'''
    # No rental, SSH network, or driver staging; only this temp runner is launched.
    result = subprocess.run(['bash', '-uc', setup + launch + '\nprintf "PID:%s SNAP:%s\\n" "$LANE_PID" "$LANE_INITIAL"'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, (result.stdout, result.stderr)
    row = re.search(r'PID:(\d+) SNAP:(.*)', result.stdout)
    assert row and row.group(2).startswith('1 '), result.stdout
    fields = row.group(2).split()
    assert len(fields) == 4 and fields[1].isdigit() and fields[2] != '-'
    assert verdict(row.group(2), row.group(2), age=0) == '1'
