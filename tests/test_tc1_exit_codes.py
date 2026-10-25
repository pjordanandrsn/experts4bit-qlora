"""TC1 amendment 82: tc1_run.sh never exits a host code except where it means one.

The launcher excludes a machine when the DRIVER exits a host code: 13 (disk), 14 (egress), 17 (power cap), 18 (host floor). It reads
13 as free space on /root under the checkpoint floor. Before this amendment, tc1_run.sh also finished 13 on a dataset fetch failure or
sha mismatch, so a pinned-upstream fault excluded every host in turn. Those now finish 21, a TC1 lane code.

The rules, read off the script's own text:
- every literal ``finish N`` is either a host code on a line that states a host-floor reason, or one of TC1's lane codes;
- 13 appears exactly once, at the disk floor (``BOX_REFUSED disk=``);
- tc1_drive.sh's ``drive_finish`` lane list is exactly the script's non-zero lane codes, so no code the box writes turns into 26.
Each rule has a mutant here.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "bench" / "tc1" / "tc1_run.sh"
DRIVE = REPO / "bench" / "tc1" / "tc1_drive.sh"
HOST = {13, 14, 17, 18}                                       # the launcher's host-excluding codes (bench/common/drive_tail.sh)
LANE = {0, 9, 12, 15, 19, 21, 78, 130}                        # TC1's own codes, each a refusal or an end the box records
HOST_REASON = re.compile(r"BOX_REFUSED|host floor|host-limited")
FINISH = re.compile(r"\bfinish (\d+)\b")
DRIVE_FINISH = re.compile(r'drive_finish "\$RUN_DIR/tc1" "\$NONCE" TC1 "([0-9 ]+)"')


def findings(run_text: str, drive_text: str) -> list[str]:
    out, sites13, lane_seen = [], [], set()
    for n, line in enumerate(run_text.splitlines(), 1):
        for m in FINISH.finditer(line):
            code = int(m.group(1))
            if code in HOST:
                if not HOST_REASON.search(line):
                    out.append(f"tc1_run.sh:{n}: finish {code} is a host code on a line with no host-floor reason")
                if code == 13:
                    sites13.append(n)
                    if "BOX_REFUSED disk=" not in line:
                        out.append(f"tc1_run.sh:{n}: finish 13 outside the disk floor (the launcher reads 13 as disk)")
            elif code in LANE:
                lane_seen.add(code)
            else:
                out.append(f"tc1_run.sh:{n}: finish {code} is neither a host code nor a registered TC1 code")
    if len(sites13) != 1:
        out.append(f"finish 13 at {len(sites13)} sites, not exactly the disk floor's one")
    m = DRIVE_FINISH.search(drive_text)
    if not m:
        out.append("tc1_drive.sh does not hand the box code to drive_finish")
    elif set(map(int, m.group(1).split())) != lane_seen - {0}:
        out.append(f"tc1_drive.sh's lane list {sorted(map(int, m.group(1).split()))} is not tc1_run.sh's codes {sorted(lane_seen - {0})}")
    return out


def test_tc1_run_exits_a_host_code_only_where_it_means_one():
    assert findings(RUN.read_text(), DRIVE.read_text()) == []


def test_the_dataset_refusals_are_lane_code_21():
    text = RUN.read_text()
    for what in ("DATASET FAIL (alpaca)", "DATASET MISMATCH alpaca", "DATASET MISMATCH clinical"):
        line = next(ln for ln in text.splitlines() if what in ln)
        assert FINISH.findall(line) == ["21"], (what, FINISH.findall(line))


def test_mutant_the_old_dataset_code_is_caught():
    text = RUN.read_text().replace('say "DATASET MISMATCH alpaca: $GOT != $DS_ALPACA_SHA"; finish 21;',
                                   'say "DATASET MISMATCH alpaca: $GOT != $DS_ALPACA_SHA"; finish 13;')
    assert text != RUN.read_text(), "the mutant's anchor is stale"
    assert any("outside the disk floor" in f for f in findings(text, DRIVE.read_text()))


def test_mutant_a_host_code_without_a_reason_is_caught():
    text = RUN.read_text() + '\n[ -n "$X" ] || { say "something"; finish 18; }\n'
    assert any("no host-floor reason" in f for f in findings(text, DRIVE.read_text()))


def test_mutant_an_unregistered_code_is_caught():
    text = RUN.read_text() + '\n[ -n "$X" ] || { say "something"; finish 33; }\n'
    assert any("neither a host code" in f for f in findings(text, DRIVE.read_text()))


def test_mutant_21_missing_from_the_drivers_lane_list_is_caught():
    drive = DRIVE.read_text().replace('TC1 "9 12 15 19 21 78 130"', 'TC1 "9 12 15 19 78 130"')
    assert drive != DRIVE.read_text(), "the mutant's anchor is stale"
    assert any("is not tc1_run.sh's codes" in f for f in findings(RUN.read_text(), drive))
