"""bench/common/controller_fault.py: the controller-side fault (PGX2 Amendment 1, P238 Amendment 7), from the real record first.

The record is ``pgx2-4090-r1-2-1``'s guard files, redacted (tests/fixtures/ctlfault_pgx2_r1/README.md) and sha256-pinned. It is a
controller fault for either lane's directory name. Then each of the five conditions is broken once, and each break is NOT a
controller fault; so is every unreadable or malformed record. The pages state the rule as the classifier reads it."""
import hashlib
import json
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "common"))
import controller_fault as cf  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "ctlfault_pgx2_r1"
PINS = {"guard-firing.json": "3d983ded9dc8d68024beb605fc6582459c755faae6dd26618a76d33c0d5e73f8",
        "teardown-proof.guard.json": "a4185031f258813e14c80e5a04793aec39fe864048c039001e8619a6b0c609c3",
        "receipt_status.json": "0d32edca40cc1fa055dbd95b7178af320ea1aae783599f40adfbb68282199c33",
        "teardown-proof.json": "0bc65eae8da66c495e9559a67026a6301ac44f7ad6490809dee4e2652b1f8896"}


def _run_dir(tmp_path, lane_dir=None):
    """The record as a receipt directory (receipt.json from its status field), plus a fetched lane directory if named."""
    d = tmp_path / "pgx2-4090-r1-2-1"
    d.mkdir()
    for name in ("guard-firing.json", "teardown-proof.guard.json", "teardown-proof.json"):
        shutil.copy(FIX / name, d / name)
    shutil.copy(FIX / "receipt_status.json", d / "receipt.json")
    if lane_dir:
        (d / lane_dir).mkdir()
    return d


def test_the_fixtures_are_the_redacted_record():
    for name, sha in PINS.items():
        b = (FIX / name).read_bytes()
        assert hashlib.sha256(b).hexdigest() == sha, name
        assert b"\r" not in b, name
    assert json.loads((FIX / "guard-firing.json").read_text(encoding="utf-8"))["instance_id"] == "INSTANCE"
    for name in ("teardown-proof.guard.json", "teardown-proof.json"):
        assert (FIX / name).read_text(encoding="utf-8").count("INSTANCE") == 2, name


@pytest.mark.parametrize("lane", ["pgx", "p238"])
def test_the_real_record_is_a_controller_fault(tmp_path, lane):
    ok, why = cf.controller_fault(str(_run_dir(tmp_path)), lane)
    assert ok, why
    assert why == ("guard fired heartbeat-loss at 2026-11-07T12:02:32Z; its listing after the destroy did not answer (proof at "
                   "2026-11-07T12:05:02Z); instance absent; receipt ALARM; nothing fetched"), why


def _edit_json(d, name, fn):
    p = d / name
    v = json.loads(p.read_text(encoding="utf-8"))
    fn(v)
    p.write_text(json.dumps(v), encoding="utf-8")


def _evidence(v, **kw):
    e = json.loads(v["evidence"])
    e.update(kw)
    v["evidence"] = json.dumps(e)


BREAKS = {
    "the guard fired on another reason": (lambda d: _edit_json(d, "guard-firing.json", lambda v: v.update(reason="deadline")),
                                         "did not fire on heartbeat-loss"),
    "the guard firing has no reason": (lambda d: _edit_json(d, "guard-firing.json", lambda v: v.pop("reason")),
                                      "did not fire on heartbeat-loss"),
    "the guard firing is not an object": (lambda d: (d / "guard-firing.json").write_text("[]", encoding="utf-8"),
                                         "did not fire on heartbeat-loss"),
    # condition 2, the guard's own proof at the firing. A stalled or crashed controller (the provider answered) leaves a list:
    "the provider answered after the destroy (a stalled controller)": (
        lambda d: _edit_json(d, "teardown-proof.guard.json", lambda v: _evidence(v, list_after=[])), "provider unreachable"),
    "the guard's proof names another reason": (
        lambda d: _edit_json(d, "teardown-proof.guard.json", lambda v: v.update(reason="wallclock")), "provider unreachable"),
    "the guard's proof has no list_after": (
        lambda d: _edit_json(d, "teardown-proof.guard.json",
                             lambda v: v.update(evidence=json.dumps({k: x for k, x in json.loads(v["evidence"]).items()
                                                                     if k != "list_after"}))), "provider unreachable"),
    "the guard's proof evidence does not parse": (
        lambda d: _edit_json(d, "teardown-proof.guard.json", lambda v: v.update(evidence="{not json")), "provider unreachable"),
    "the guard's proof evidence is not an object": (
        lambda d: _edit_json(d, "teardown-proof.guard.json", lambda v: v.update(evidence="[]")), "provider unreachable"),
    "the final proof evidence is not an object": (
        lambda d: _edit_json(d, "teardown-proof.json", lambda v: v.update(evidence="[]")), "instance absent"),
    "the guard's proof is not an object": (
        lambda d: (d / "teardown-proof.guard.json").write_text("[]", encoding="utf-8"), "provider unreachable"),
    "the final teardown proof is not complete": (lambda d: _edit_json(d, "teardown-proof.json", lambda v: v.update(complete=False)),
                                                "instance absent"),
    "the instance is not shown absent": (lambda d: _edit_json(d, "teardown-proof.json", lambda v: _evidence(v, instance_absent=False)),
                                        "instance absent"),
    "the teardown evidence does not parse": (lambda d: _edit_json(d, "teardown-proof.json", lambda v: v.update(evidence="{not json")),
                                            "instance absent"),
    "the teardown proof has no evidence": (lambda d: _edit_json(d, "teardown-proof.json", lambda v: v.pop("evidence")),
                                          "instance absent"),
    "the receipt status is not ALARM": (lambda d: _edit_json(d, "receipt.json", lambda v: v.update(status="OK")), "not ALARM"),
}


@pytest.mark.parametrize("case", sorted(BREAKS))
def test_each_condition_broken_is_not_a_controller_fault(tmp_path, case):
    d = _run_dir(tmp_path)
    breaker, said = BREAKS[case]
    breaker(d)
    ok, why = cf.controller_fault(str(d), "pgx")
    assert not ok and said in why, (case, why)


@pytest.mark.parametrize("lane", ["pgx", "p238"])
def test_a_fetched_lane_directory_is_read_by_the_lane(tmp_path, lane):
    ok, why = cf.controller_fault(str(_run_dir(tmp_path, lane_dir=lane)), lane)
    assert not ok and f"a lane directory ({lane}/) was fetched" in why


def test_an_uncheckable_lane_directory_is_not_a_controller_fault(tmp_path, monkeypatch):
    d = _run_dir(tmp_path)

    def lstat(path, *a, **k):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(cf.os, "lstat", lstat)
    ok, why = cf.controller_fault(str(d), "pgx")
    assert not ok and "the lane directory cannot be checked (PermissionError)" in why


@pytest.mark.parametrize("missing", ["guard-firing.json", "teardown-proof.guard.json", "teardown-proof.json", "receipt.json"])
def test_a_missing_file_is_not_a_controller_fault(tmp_path, missing):
    d = _run_dir(tmp_path)
    (d / missing).unlink()
    ok, why = cf.controller_fault(str(d), "pgx")
    assert not ok and why == "the record is unreadable (FileNotFoundError)"


def test_a_malformed_file_is_not_a_controller_fault(tmp_path):
    d = _run_dir(tmp_path)
    (d / "receipt.json").write_text("{", encoding="utf-8")
    ok, why = cf.controller_fault(str(d), "pgx")
    assert not ok and why == "the record is unreadable (JSONDecodeError)"


def test_the_command_line(tmp_path, capsys):
    d = _run_dir(tmp_path)
    assert cf.main([str(d), "pgx"]) == 0 and capsys.readouterr().out.startswith("CONTROLLER_FAULT guard fired")
    assert cf.main([str(d), "pgx", "x"]) == 2 and capsys.readouterr().out.startswith("A controller-side fault")
    (d / "pgx").mkdir()
    assert cf.main([str(d), "pgx"]) == 1 and capsys.readouterr().out.startswith("NOT_CONTROLLER_FAULT: a lane directory")


@pytest.mark.parametrize("page,heading,lane,retry", [
    ("bench/pgx/PREREG-pgx2.md", "## Amendment 1 (2026-11-07): a controller-side fault, from `pgx2-4090-r1-2-1`", "", "pgx2-4090-r2-2-1"),
    ("bench/p238/PREREG-p238.md", "## Amendment 7 (2026-11-07): a controller-side fault, registered before `p238-shake-r9`",
     "`p238/`", "r10 or r11 may follow a controller-fault box"),
])
def test_each_page_registers_the_rule_as_the_classifier_reads_it(page, heading, lane, retry):
    text = (ROOT / page).read_text(encoding="utf-8")
    assert text.count(heading) == 1
    amend = " ".join(text[text.index(heading):].split())
    for term in ("`reason` exactly `heartbeat-loss`", "its `reason` is `heartbeat-loss` and its evidence reads `list_after` "
                 "`\"UNKNOWN\"`", "a stalled or crashed controller process", "The guard's log is not read",
                 "`complete` is true and its evidence reads `instance_absent` true", "`status` is `ALARM`",
                 "is not a strike and not a read", "is not a STARTED box", "**at most once**",
                 "A second controller fault ends the lane", "before its \"no readable exit code\" check", retry, lane):
        assert term in amend, (page, term)
