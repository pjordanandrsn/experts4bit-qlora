"""PGX2 at $0 (bench/pgx/PREREG-pgx2.md): the successor lane's ids and its one new rule, a host fault after STARTED.

- the record first: pgx-4090-1-2-1, the reading box that ended PGX, is committed byte-for-byte (SHA256SUMS) and its fetch
  log is sha-pinned here;
- under its own PGX id it is still a STRIKE: PGX's verdict does not change;
- relabelled to a PGX2 id it is a HOST_FAULT (exit 14), on the real log;
- the rule fails closed: a network-shaped line not on the list, a truncated, empty, undecodable or unreadable log, a log
  with no traceback, a missing or out-of-order summary, a later tag's log, every checkpoint fetched, another exit code, a
  caller who read another code, and an rc 11 without STARTED are each NOT a host fault;
- the listed lines are the page's, the ids are accepted by the schedule, the runner and the driver, and the page's budget
  is the schedule's.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PGX = REPO / "bench" / "pgx"
REC = PGX / "receipts" / "pgx-4090-1-2-1"
PAGE = PGX / "PREREG-pgx2.md"
sys.path.insert(0, str(PGX))

import pgx_box_end as be  # noqa: E402
import pgx_schedule as sch  # noqa: E402

NONCE = "d910131296e70565"
FETCH_INT4_SHA = "9e8842ad18989aab57d5f93315b56cc88375ab422a07ec784455bde26651cede"
LAST = "httpx.ConnectError: [Errno 101] Network is unreachable"
RUN = (PGX / "pgx_run.sh").read_text(encoding="utf-8")
DRIVE = (PGX / "pgx_drive.sh").read_text(encoding="utf-8")


def _page() -> str:
    return " ".join(PAGE.read_text(encoding="utf-8").split())        # the page wraps its lines


# ---------------------------------------------------------------- the record
def test_the_record_is_the_boxs_bytes():
    rows = [ln.split(None, 1) for ln in (REC / "SHA256SUMS").read_text().splitlines()]
    files = sorted(str(p.relative_to(REC)) for p in REC.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    assert sorted(name for _, name in rows) == files
    for sha, name in rows:
        assert hashlib.sha256((REC / name).read_bytes()).hexdigest() == sha, name
    assert hashlib.sha256((REC / "logs" / "fetch_int4.log").read_bytes()).hexdigest() == FETCH_INT4_SHA
    assert (REC / "RUN_ID").read_text().strip() == "pgx-4090-1-2-1"
    assert (REC / f"PGX_EXIT_CODE.{NONCE}").read_text().strip() == "11"
    assert (REC / "logs" / "fetch_int4.log").read_text(encoding="utf-8").rstrip().splitlines()[-1] == LAST


def test_under_its_own_pgx_id_it_is_still_a_strike():
    """PGX's verdict stands: the new rule reads PGX2 ids only."""
    res = be.classify(str(REC), "pgx-4090-1-2-1", 11)
    assert res["class"] == be.STRIKE and be.exit_code(res) == 12
    assert res["why"] == ["rc 11: only 0 and 27 can end a reading box without a strike"]


def _relabel(tmp_path, rid="pgx2-4090-1-2-1"):
    d = tmp_path / "pgx"
    shutil.copytree(REC, d)
    (d / "RUN_ID").write_text(rid + "\n")
    return d


def _cls(d, rid="pgx2-4090-1-2-1", rc=11):
    return be.classify(str(d), rid, rc)


def test_relabelled_to_pgx2_it_is_a_host_fault(tmp_path, capsys):
    d = _relabel(tmp_path)
    res = _cls(d)
    assert res["class"] == be.HOST_FAULT and be.exit_code(res) == 14
    assert "the int4 fetch ended on a listed network fault" in res["why"][0] and LAST in res["why"][0]
    assert be.main([str(d), "--run-id", "pgx2-4090-1-2-1", "--rc", "11"]) == 14
    assert capsys.readouterr().out.splitlines()[0].startswith("BOX_END HOST_FAULT pgx2-4090-1-2-1 rc 11")


def test_a_retry_id_reads_the_same(tmp_path):
    d = _relabel(tmp_path, "pgx2-4090-r1-2-1")
    assert _cls(d, "pgx2-4090-r1-2-1")["class"] == be.HOST_FAULT


# ---------------------------------------------------------------- it fails closed
def _set_log(d, text, tag="int4"):
    (d / "logs" / f"fetch_{tag}.log").write_bytes(text if isinstance(text, bytes) else text.encode())


def _with_last(line):
    body = (REC / "logs" / "fetch_int4.log").read_text(encoding="utf-8")
    assert body.endswith(LAST + "\n")
    return body[: -len(LAST) - 1] + line + "\n"


@pytest.mark.parametrize("line", [
    "httpx.ConnectError: [Errno 111] Connection refused",            # network-shaped, not on the list
    "httpx.ConnectError: [Errno 104] Connection reset by peer",      # network-shaped, not on the list
    "httpx.ReadTimeout: The read operation timed out",               # network-shaped, not on the list
    "httpcore.ConnectError: [Errno 101] Network is unreachable",     # the listed text, another class
    LAST + " (retry 3)",                                             # a listed line with a tail: fullmatch only
    "x " + LAST,                                                     # a listed line with a head
    "OSError: [Errno 28] No space left on device",                   # a disk fault is the box's, not the network's
])
def test_a_line_not_on_the_list_is_a_strike(tmp_path, line):
    d = _relabel(tmp_path)
    _set_log(d, _with_last(line))
    res = _cls(d)
    assert res["class"] == be.STRIKE and "not a listed network fault" in res["why"][0], res


LISTED = [LAST, "httpx.ConnectError: [Errno -3] Temporary failure in name resolution",
          "httpx.ConnectTimeout: _ssl.c:999: The handshake operation timed out"]


@pytest.mark.parametrize("line", LISTED)
def test_each_listed_line_is_a_host_fault(tmp_path, line):
    d = _relabel(tmp_path)
    _set_log(d, _with_last(line))
    assert _cls(d)["class"] == be.HOST_FAULT


def test_a_truncated_log_is_a_strike(tmp_path):
    body = (REC / "logs" / "fetch_int4.log").read_text(encoding="utf-8")
    for cut, why in ((body[: -len(LAST) - 1], "not a listed network fault"),                 # the final line lost
                     (body[: body.index("Traceback (most recent call last):")], "no traceback"),   # cut before it
                     ("", "no traceback")):
        d = _relabel(tmp_path / str(len(cut)))
        _set_log(d, cut)
        res = _cls(d)
        assert res["class"] == be.STRIKE and why in res["why"][0], (len(cut), res)


def test_the_listed_line_alone_without_a_traceback_is_a_strike(tmp_path):
    d = _relabel(tmp_path)
    _set_log(d, LAST + "\n")
    res = _cls(d)
    assert res["class"] == be.STRIKE and "no traceback" in res["why"][0]


def test_an_undecodable_log_is_a_strike(tmp_path):
    d = _relabel(tmp_path)
    _set_log(d, _with_last(LAST).encode() + b"\xff\n")
    res = _cls(d)
    assert res["class"] == be.STRIKE and "unreadable (UnicodeDecodeError)" in res["why"][0]


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root posix user")
def test_an_unreadable_log_is_a_strike_never_an_exception(tmp_path):
    d = _relabel(tmp_path)
    p = d / "logs" / "fetch_int4.log"
    p.chmod(0)
    try:
        res = _cls(d)
    finally:
        p.chmod(0o644)
    assert res["class"] == be.STRIKE and "unreadable (PermissionError)" in res["why"][0]


def test_the_failed_tag_is_read_from_the_summary(tmp_path):
    """The failed fetch is the first tag without a FETCH line; its log, not another, is read."""
    d = _relabel(tmp_path)
    s = (d / "summary.txt").read_text()
    # the summary says only bf16 fetched: fp8 is the failed tag, and its log ends in a snapshot path, not a fault
    (d / "summary.txt").write_text("\n".join(ln for ln in s.splitlines() if not ln.startswith("FETCH fp8")) + "\n")
    res = _cls(d)
    assert res["class"] == be.STRIKE and "fetch_int4.log exists after the failed fp8 fetch" in res["why"][0]
    (d / "logs" / "fetch_int4.log").unlink()
    res = _cls(d)
    assert res["class"] == be.STRIKE and "fetch_fp8.log has no traceback" in res["why"][0]


def test_a_bf16_host_fault_reads_the_bf16_log(tmp_path):
    d = _relabel(tmp_path)
    s = (d / "summary.txt").read_text()
    (d / "summary.txt").write_text("\n".join(ln for ln in s.splitlines() if not ln.startswith("FETCH ")) + "\n")
    os.replace(d / "logs" / "fetch_int4.log", d / "logs" / "fetch_bf16.log")
    (d / "logs" / "fetch_fp8.log").unlink()
    res = _cls(d)
    assert res["class"] == be.HOST_FAULT and "the bf16 fetch" in res["why"][0]


def test_the_summary_must_be_readable_and_in_order(tmp_path):
    d = _relabel(tmp_path / "missing")
    (d / "summary.txt").unlink()
    assert "summary.txt unreadable (FileNotFoundError)" in _cls(d)["why"][0]
    d = _relabel(tmp_path / "order")
    s = (d / "summary.txt").read_text()
    (d / "summary.txt").write_text(s.replace("FETCH bf16 ", "FETCH xx ").replace("FETCH fp8 ", "FETCH bf16 "))
    assert "are not the runner's order" in _cls(d)["why"][0]
    d = _relabel(tmp_path / "all")
    (d / "summary.txt").write_text(s + "FETCH int4 x@y /z\n")
    res = _cls(d)
    assert res["class"] == be.STRIKE and "every checkpoint fetched" in res["why"][0]


def test_another_exit_code_or_a_disagreeing_caller_is_a_strike(tmp_path):
    d = _relabel(tmp_path)
    (d / f"PGX_EXIT_CODE.{NONCE}").write_text("9\n")
    res = _cls(d, rc=9)
    assert res["class"] == be.STRIKE and res["why"] == ["rc 9: only 0 and 27 can end a reading box without a strike"]
    (d / f"PGX_EXIT_CODE.{NONCE}").write_text("11\n")
    res = _cls(d, rc=0)
    assert res["class"] == be.STRIKE and "not the code read by the caller 0" in res["why"][0]


def test_an_rc_11_without_started_never_started(tmp_path):
    """No STARTED: the registered pre-STARTED rules apply; the host-fault rule never reads it."""
    d = _relabel(tmp_path)
    (d / "STARTED").unlink()
    assert _cls(d)["class"] == be.NOT_STARTED


# ---------------------------------------------------------------- the ids
_CANDIDATES = [f"{lane}-4090-{a}{s}" for lane in ("pgx", "pgx2", "pgx3", "pgx22") for a in ("1", "2", "r1", "r3", "r4")
               for s in ("", "-2-1", "-2-2", "-2-3", "-3-1")] + ["pgx2-shake-1", "pgx2-4090-1-2-1x", "xpgx2-4090-1-2-1"]


def _case_accepts(text: str, rid: str) -> bool:
    pats = re.findall(r"\n\s*((?:pgx2?-4090-1-2-\[12\])\|(?:pgx2?-4090-r\[1-3\]-2-\[12\]))\)", text)
    assert len(pats) == 2, pats                                   # PGX's line and PGX2's
    r = subprocess.run(["bash", "-c", f'case "$1" in {pats[0]}|{pats[1]}) exit 0;; *) exit 1;; esac', "_", rid])
    return r.returncode == 0


def test_the_runner_and_driver_accept_exactly_both_lanes_plan_ids():
    def registered(rid):
        try:
            k, _i, _n = sch.parse_run_id(rid)
        except ValueError:
            return False
        return k == sch.PLAN_K
    for rid in _CANDIDATES:
        assert _case_accepts(RUN, rid) == registered(rid) == _case_accepts(DRIVE, rid), rid
    assert "  pgx2-4090-1-2-[12]|pgx2-4090-r[1-3]-2-[12]) WANT_BOX=read; BOX_I=${PGX_RUN_ID##*-}; BOX_SPEC=2-$BOX_I;;" in RUN
    assert "  pgx2-4090-1-2-[12]|pgx2-4090-r[1-3]-2-[12]) BOX=read;;" in DRIVE


def test_the_schedule_names_the_lane():
    assert sch.box_ids(2, "pgx2") == ["pgx2-4090-1-2-1", "pgx2-4090-1-2-2"]
    assert sch.box_ids(2) == ["pgx-4090-1-2-1", "pgx-4090-1-2-2"]                  # PGX's, unchanged
    assert sch.retry_id(3, 2, 1, "pgx2") == "pgx2-4090-r3-2-1" and sch.retry_id(2, 2, 2) == "pgx-4090-r2-2-2"
    assert sch.run_lane("pgx2-4090-r2-2-2") == "pgx2" and sch.run_lane("pgx-4090-1-2-1") == "pgx"
    assert sch.parse_run_id("pgx2-4090-r2-2-2") == sch.parse_run_id("pgx-4090-r2-2-2") == (2, 2, 2)
    for bad in ("pgx3-4090-1-2-1", "pgx2-4090-r4-2-1", "pgx2-4090-1-2-3"):
        with pytest.raises(ValueError):
            sch.run_lane(bad)
    with pytest.raises(ValueError):
        sch.box_ids(2, "pgx3")
    assert sch.main(["--run-id", "pgx2-4090-1-2-2"]) == 0 and sch.main(["--run-id", "pgx2-4090-1-3-1"]) == 2


# ---------------------------------------------------------------- the page
def test_the_page_lists_the_classifiers_lines_exactly():
    page = PAGE.read_text(encoding="utf-8")
    listed = re.findall(r"^\d\. `(httpx\.[^`]+)`", page, re.M)
    assert [re.fullmatch(p, ln) is not None for (p, _), ln in zip(be.HOST_FAULT_LINES, listed)] == [True] * 3
    assert len(listed) == len(be.HOST_FAULT_LINES) == 3


def test_the_page_states_the_schedules_budget():
    import math
    p = sch.plan()
    boxes = {s["k"]: s for s in p["splits"]}[sch.PLAN_K]["boxes"]
    a5 = _page()
    total = 0.0
    for b in boxes:
        assert f"{b['guard_s']:,} s = {b['guard_h']} h" in a5, b                     # Amendment 3's fit, unchanged
        guard_h = math.ceil(b["guard_s"] / 36) / 100                                  # rounded UP to 0.01 h
        worst = round(guard_h * sch.USD_PER_H + sch.DOWNLOAD_USD, 2)
        assert f"guard {guard_h:.2f} h, USD {worst:.2f}" in a5, (b, guard_h, worst)
        total += worst
    assert f"USD {total:.2f}" in a5
    lane = total + 5.00 + 3 * 0.25
    assert f"**USD {lane:.2f}**, under the USD 35" in a5 and lane < 35
    for want in ("`pgx2-4090-1-2-1`", "`pgx2-4090-1-2-2`", "Strikes start at 0", "USD 3.49", "USD 35",
                 "PGX's verdict is not changed", "at most two", "USD 5.00"):
        assert want in a5, want
