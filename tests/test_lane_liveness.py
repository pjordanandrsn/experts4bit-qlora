"""Execute the shared Linux probe and each driver's poll loop without a rental."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bench/common/lane_liveness.sh"
DRIVERS = [ROOT / f"bench/{folder}/{script}_drive.sh" for folder, script in
           [("tc1", "tc1"), ("p127", "p127"), ("fam", "fam"), ("locality-1469", "locality"), ("sc1", "sc1"),
            ("p137", "p137"), ("p144", "p144"), ("p146", "p146"), ("p143", "p143a3"), ("p148", "p148"), ("p150", "p150"), ("p151", "p151"), ("p152", "p152"), ("p153", "p153"), ("p154", "p154")]]
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


def test_lxcfs_uptime_reset_with_same_boot_id_is_rebooted():
    # lxcfs can reset its exposed uptime while the boot UUID/PID/ticks stay equal.
    # The registered driver contract treats that backwards clock as a reboot.
    assert verdict(f"1 123 {BOOT_A} 9000.00", INITIAL, age=10) == "reboot"


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
    ([f"1 123 {BOOT_A} 9000.00"], "reboot", 1),
    (["0 - - 10000", None, "0 - - 10001", INITIAL], "complete", 4),
    ([INITIAL, INITIAL, INITIAL, INITIAL], "complete", 4),
    ([INITIAL, None, None, INITIAL], "complete", 4),
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
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Each SSH shell gets its own process, so persist poll count in a file.
    prelude = f'''
source "{HELPER}"
LANE_HELPER="{HELPER}"; LANE_PID=4242; LANE_INITIAL="{INITIAL}"; LANE_STARTED_AT=0
DEADLINE=999999; POLL=0; STALL_S=900; TC1_MIN_PROGRESS_MB=16; P127_MIN_PROGRESS_MB=16; FAM_MIN_PROGRESS_MB=16; LOC_MIN_PROGRESS_MB=16; SC1_MIN_PROGRESS_MB=16
P137_MIN_PROGRESS_MB=16; P144_MIN_PROGRESS_MB=16; P146_MIN_PROGRESS_MB=16; P143A3_MIN_PROGRESS_MB=16; P148_MIN_PROGRESS_MB=16; P150_MIN_PROGRESS_MB=16
W=/fake; NONCE=nonce; RUN_DIR=/unused; E4B_RENT_RUN_DIR="{run_dir}"
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
box(){{ $SSH "$1"; }}   # the P137-family drivers reach the box through box(), one command plus stdin
'''
    result = subprocess.run(["bash", "-uc", prelude + funcs + loop + '\nprintf "END:%s\\n" "$LANE_DEAD"'],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert int(count.read_text()) == polls
    assert ("END:0" if expected == "complete" else "END:1") in result.stdout
    marker = run_dir / "host-fault.json"
    if expected == "reboot":
        assert "host rebooted" in result.stdout and "host-fault.json written (kind reboot)" in result.stdout
        fault = json.loads(marker.read_text())
        assert fault["kind"] == "reboot" and fault["at"].endswith("Z")
        assert fault["evidence"].startswith(f"boot_id {BOOT_A} -> ") and "at lane age 4000s" in fault["evidence"]
    else:
        assert not marker.exists()
    if sequence == [INITIAL, None, None, INITIAL]:
        assert re.findall(r"unknown_probes (\d+)", result.stdout) == ["0", "1", "2", "0"]
    assert "pgrep" not in loop and "lane_snapshot_verdict" in loop


def test_every_launch_captures_pid_and_snapshot_without_staging_helper():
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
    prefix = {'tc1': 'TC1', 'p127': 'P127', 'fam': 'FAM', 'locality-1469': 'LOC', 'sc1': 'SC1',
              'p137': 'P137', 'p144': 'P144', 'p146': 'P146', 'p143': 'P143A3', 'p148': 'P148', 'p150': 'P150', 'p151': 'P151', 'p152': 'P152', 'p153': 'P153', 'p154': 'P154'}[driver.parent.name]
    (tmp_path / runner).write_text(f'#!/bin/bash\nprintf nonce > {prefix}_RUN_NONCE\nsleep 2\n')
    (tmp_path / 'guard.sh').write_text(f'#!/bin/bash\nexec bash {runner}\n')
    setup = f'''source "{HELPER}"
LANE_HELPER="{HELPER}"; W="{tmp_path}"; PASS=""; NONCE=nonce; RUNNER=guard.sh
say(){{ echo "$*"; }}
local_ssh(){{ "$@"; }}
SSH=local_ssh
box(){{ bash -c "$1"; }}
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


def test_host_fault_marker_is_well_formed_json_the_launcher_accepts(tmp_path):
    """adertha-agents#204's contract: kind in reboot / unreachable / gpu-lost and non-empty evidence text; quotes,
    backslashes and control characters cannot break the JSON."""
    out = bash(f'lane_write_host_fault "{tmp_path}" reboot "$(printf \'boot_id a -> b; "quoted" \\\\ x\\tz\')"')
    assert out.returncode == 0, out.stderr
    fault = json.loads((tmp_path / "host-fault.json").read_text())
    assert fault["kind"] == "reboot" and fault["evidence"] == 'boot_id a -> b; "quoted" \\ xz'
    assert not list(tmp_path.glob(".host-fault.json.*"))


@pytest.mark.parametrize("args", ['"{d}" power "evidence"', '"{d}" reboot ""', '"" reboot "evidence"',
                                  '"{d}/absent" reboot "evidence"'])
def test_host_fault_marker_refuses_what_the_launcher_would_not_accept(tmp_path, args):
    assert bash("lane_write_host_fault " + args.format(d=tmp_path)).returncode != 0
    assert not (tmp_path / "host-fault.json").exists()


def test_reboot_evidence_names_both_boot_ids_and_the_uptimes():
    out = bash(f'lane_reboot_evidence "1 123 {BOOT_B} 20.5" "{INITIAL}" 3668').stdout.strip()
    assert out == f"boot_id {BOOT_A} -> {BOOT_B}; uptime 20.5s at lane age 3668s (10000.00s at launch)"


def test_command_error_line_carries_the_rc_the_drive_and_the_box(tmp_path):
    """adertha-agents#209: one line for the receipt's notes; control characters dropped (the launcher cleans it again)."""
    out = bash(f'lane_write_command_error "{tmp_path}" 3 "lane exit rc=3 without success marker" "$(printf \'OOM\\033[0m here\')"')
    assert out.returncode == 0, out.stderr
    line = (tmp_path / "command-error.txt").read_text()
    assert line == "rc=3; drive: lane exit rc=3 without success marker; box: OOM[0m here\n", repr(line)
    assert bash(f'lane_write_command_error "{tmp_path / "absent"}" 3 x y').returncode != 0


def test_box_error_line_reads_outer_log_first_then_the_newest_log(tmp_path):
    box = tmp_path / "tc1"
    (box / "logs").mkdir(parents=True)
    assert bash(f'printf "[%s]" "$(lane_box_error_line "{box}")"').stdout == "[]"
    old, new = box / "logs/a.log", box / "logs/b.log"
    old.write_text("torch.OutOfMemoryError: old\n")
    new.write_text("step 3\nRuntimeError: newest\nstep 4\n")
    os.utime(old, (1, 1))
    assert bash(f'lane_box_error_line "{box}"').stdout == "RuntimeError: newest"
    (box / "outer.log").write_text("start\ntc1/boxA: arm refused: no wheel\ndone\n")
    assert bash(f'lane_box_error_line "{box}"').stdout == "tc1/boxA: arm refused: no wheel"


def test_tc1_drive_names_its_refusal_in_the_run_dir(tmp_path):
    """The drive's EXIT trap writes the error line on a refusal and leaves the exit code alone."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E4B_RENT_", "TC1_"))}
    env["E4B_RENT_RUN_DIR"] = str(tmp_path)
    r = subprocess.run(["bash", str(ROOT / "bench/tc1/tc1_drive.sh")], env=env, text=True, capture_output=True, timeout=30)
    assert r.returncode == 78, (r.returncode, r.stdout, r.stderr)
    line = (tmp_path / "command-error.txt").read_text().strip()
    assert line.startswith("rc=78; drive: refusing: E4B_RENT_SSH_HOST is not set"), line


# Legacy drivers that still decide LANE DEAD (or ALIVE) from `pgrep -f '<runner>'` run inside the ssh command string. That
# command line contains the pattern, so pgrep counts its own shell and a dead lane reads "live" until the deadline. They
# are frozen as registered: none is to be launched again without first moving to bench/common/lane_liveness.sh (and out
# of this list). The list may only shrink -- a NEW driver that makes a lane-death decision is held to the rule below.
LEGACY_PGREP_DRIVERS = frozenset([
    "b771/b771", "b771b/b771b", "g4census/g4census", "gpu-contract/gc", "i8dense/i8", "k22/k22", "k24/k24",
    "k26/k26", "k27/k27", "k28/k28", "k29/k29", "k33/k33", "k34/k34", "p47/p47", "p54/p54", "p55/p55", "p55x/p55x",
    "p57/p57", "p58/p58", "p59/p59", "p60/p60", "p63/p63", "p64/p64", "p65/p65", "p66/p66", "p68/p68", "p70/p70",
    "p80/p80", "p81/p81", "p82/p82", "p83/p83", "p84/p84", "p85/p85", "p86/p86", "p87/p87", "p88/p88", "p89/p89",
    "p90/p90", "p91/p91", "p92/p92", "p93/p93", "p94/p94", "p95/p95", "p96/p96", "p97/p97", "p98/p98", "p99/p99",
    "p100/p100", "p102/p102", "p103/p103", "p104/p104", "p105/p105", "p106/p106", "p107/p107", "p108/p108",
    "p109/p109", "p110/p110", "p111/p111", "p112/p112", "p113/p113", "p114/p114", "p115/p115", "p115/p115c",
    "p115/p115d", "p116/p116", "p117/p117", "p118/p118", "p119/p119", "p120/p120", "p121/p121", "p122/p122",
    "p123/p123", "p124/p124", "p125/p125", "p126/p126", "p133/p133", "p134/p134", "p136/p136", "p140/p140",
    "p141/p141", "p143/p143", "p143/p143a2", "p145/p145", "p319/p319", "p319b/p319b", "pername/pername", "tp4/tp4",
])


def _lane_death_drivers():
    """Every driver that decides liveness: it can declare LANE DEAD, or it runs `pgrep -f` at all (p319/p319b read ALIVE
    from one without the words), since a driver's pgrep runs inside an ssh command line that contains its pattern."""
    return sorted(p for p in ROOT.glob("bench/*/*_drive.sh") if "LANE DEAD" in (t := p.read_text()) or "pgrep -f" in t)


def _uses_launch_identity(text):
    return ('LANE_HELPER="$REPO/bench/common/lane_liveness.sh"' in text and 'source "$LANE_HELPER"' in text
            and "lane_snapshot_verdict" in text and "--probe $LANE_PID" in text and "pgrep -f" not in text)


def test_every_lane_death_decision_uses_the_launch_identity_not_pgrep():
    """Case from 2026-10-18: P137/P144/P146 read 'live' from a pgrep that matched its own ssh command line, so a dead lane
    waited out its whole deadline. Any driver that can declare LANE DEAD must source the shared helper and decide from
    the launch PID's identity probe; only the frozen legacy drivers above are exempt."""
    drivers = _lane_death_drivers()
    assert len(drivers) > len(LEGACY_PGREP_DRIVERS)  # the glob found the bench tree
    names = {f"{p.parent.name}/{p.name.removesuffix('_drive.sh')}" for p in drivers}
    assert LEGACY_PGREP_DRIVERS <= names, sorted(LEGACY_PGREP_DRIVERS - names)  # no stale entries
    offenders = [n for p, n in ((p, f"{p.parent.name}/{p.name.removesuffix('_drive.sh')}") for p in drivers)
                 if n not in LEGACY_PGREP_DRIVERS and not _uses_launch_identity(p.read_text())]
    assert not offenders, f"lane drivers deciding LANE DEAD without lane_liveness.sh: {offenders}"
    migrated = [n for p, n in ((p, f"{p.parent.name}/{p.name.removesuffix('_drive.sh')}") for p in drivers)
                if n in LEGACY_PGREP_DRIVERS and _uses_launch_identity(p.read_text())]
    assert not migrated, f"migrated drivers still on the legacy list (remove them): {migrated}"
    for driver in DRIVERS:  # every driver whose poll loop is exercised above is held to the rule too
        assert _uses_launch_identity(driver.read_text()), driver
