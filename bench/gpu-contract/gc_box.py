# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/gpu-contract: the box-side checks the runner calls (no torch import; standard library only).

    python gc_box.py --card "ok torch=2.8.0+cu128 cc=8.9 sms=128" --declared 8.9/128   (exit 15 outside the class)
    python gc_box.py --junit report.xml --suite gnf4                                    (one JSON line; exit 30 on a failure)
    python gc_box.py --require "mod::test ..." --junits a.xml b.xml                     (exit 30 unless every one PASSED)
    python gc_box.py --self-test

A declared card class is ``MAJOR.MINOR/SMS``, where SMS is an exact count (``128``) or a floor (``>=160``). The class is the
device's compute capability and SM count as torch reports them, never its marketing name.
"""
from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET

CLASS = re.compile(r"^(\d+)\.(\d+)/(>=)?(\d+)$")
TEST_ID = re.compile(r"^[A-Za-z0-9_.]+::[A-Za-z0-9_.\[\]-]+$")
PROBE = re.compile(r"\bcc=(\d+)\.(\d+) sms=(\d+)\b")


def parse_class(declared: str) -> tuple:
    m = CLASS.match(declared or "")
    if not m:
        raise ValueError(f"card class {declared!r}: expected MAJOR.MINOR/SMS or MAJOR.MINOR/>=SMS (e.g. 8.9/128)")
    return (int(m.group(1)), int(m.group(2))), bool(m.group(3)), int(m.group(4))


def card_ok(probe: str, declared: str) -> tuple:
    """(ok, why) for a CUDA probe line against a declared class."""
    cc, floor, sms = parse_class(declared)
    m = PROBE.search(probe or "")
    if not m:
        return False, f"no cc/sms in the probe {probe!r}"
    got_cc, got_sms = (int(m.group(1)), int(m.group(2))), int(m.group(3))
    if got_cc != cc:
        return False, f"compute capability {got_cc[0]}.{got_cc[1]}, the class is {cc[0]}.{cc[1]}"
    if (got_sms < sms) if floor else (got_sms != sms):
        return False, f"{got_sms} SMs, the class is {'>=' if floor else ''}{sms}"
    return True, f"cc {got_cc[0]}.{got_cc[1]}, {got_sms} SMs: inside {declared}"


def junit_summary(path: str, suite: str) -> dict:
    """Pass, fail, error and skip counts from a pytest junit report, with every skipped test's id and reason."""
    root = ET.parse(path).getroot()
    out = {"suite": suite, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "skips": [], "failures": []}
    for tc in root.iter("testcase"):
        tid = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        kinds = {child.tag for child in tc}
        if "failure" in kinds:
            out["failed"] += 1
            out["failures"].append(tid)
        elif "error" in kinds:
            out["errors"] += 1
            out["failures"].append(tid)
        elif "skipped" in kinds:
            out["skipped"] += 1
            sk = next(child for child in tc if child.tag == "skipped")
            out["skips"].append({"id": tid, "reason": (sk.get("message") or "")[:300]})
        else:
            out["passed"] += 1
    out["ok"] = out["failed"] == 0 and out["errors"] == 0 and out["passed"] > 0
    return out


def parse_required(ids: str) -> list:
    """GC_REQUIRE: test ids in junit form (``<classname>::<name>``, e.g.
    ``kernel.test_dispatch_counts::test_on_a_real_rtx_4090_auto_engages_unmocked``), space-separated."""
    out = (ids or "").split()
    bad = [x for x in out if not TEST_ID.match(x)]
    if bad:
        raise ValueError(f"GC_REQUIRE ids {bad}: expected <classname>::<name>")
    return out


def required_status(paths: list, ids: list) -> dict:
    """Each required id's outcome across the junit reports: PASSED only if it ran and passed; skipped, failed, errored or
    absent is not a pass. ``ok`` iff every required id PASSED."""
    seen = {}
    for path in paths:
        for tc in ET.parse(path).getroot().iter("testcase"):
            kinds = {child.tag for child in tc}
            tid = f"{tc.get('classname', '')}::{tc.get('name', '')}"
            got = "FAILED" if "failure" in kinds else "ERROR" if "error" in kinds else "SKIPPED" if "skipped" in kinds else "PASSED"
            # worst of: pytest can write one id twice (a pass, then a teardown error), and any non-PASSED outcome must win
            if seen.get(tid, "PASSED") == "PASSED":
                seen[tid] = got
    status = {i: seen.get(i, "ABSENT") for i in ids}
    return {"required": status, "ok": bool(ids) and all(v == "PASSED" for v in status.values())}


def required_status_text(xml: str, ids: list) -> dict:
    """:func:`required_status` over a report given as text (the self-test's)."""
    import tempfile
    fd, p = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    try:
        return required_status([p], ids)
    finally:
        os.unlink(p)


def self_test() -> int:
    good = "ok torch=2.8.0+cu128 cc=8.9 sms=128"
    assert card_ok(good, "8.9/128")[0]
    assert not card_ok(good, "8.9/>=160")[0] and not card_ok(good, "12.0/>=160")[0] and not card_ok(good, "8.9/76")[0]
    assert card_ok("ok torch=2.8.0+cu128 cc=12.0 sms=170", "12.0/>=160")[0]
    assert not card_ok("ok torch=2.8.0+cu128 cc=8.6 sms=128", "8.9/128")[0]
    assert not card_ok("no-cuda torch 2.8.0", "8.9/128")[0]
    for bad in ("", "8.9", "4090", "8.9/>160", "8.9/128x"):
        try:
            parse_class(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    import os
    import tempfile
    xml = ('<testsuites><testsuite name="pytest"><testcase classname="test_a" name="t1"/>'
           '<testcase classname="test_a" name="t2"><skipped message="needs sm_90+"/></testcase>'
           '<testcase classname="test_b" name="t3"><failure message="x"/></testcase>'
           '<testcase classname="test_b" name="t4"><error message="y"/></testcase></testsuite></testsuites>')
    fd, p = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    s = junit_summary(p, "x")
    os.unlink(p)
    assert (s["passed"], s["failed"], s["errors"], s["skipped"], s["ok"]) == (1, 1, 1, 1, False), s
    assert s["skips"] == [{"id": "test_a::t2", "reason": "needs sm_90+"}] and s["failures"] == ["test_b::t3", "test_b::t4"]
    fd, p = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    want = ["test_a::t1", "test_a::t2", "test_b::t3", "test_c::t9"]
    r = required_status([p], want)
    os.unlink(p)
    assert r["required"] == {"test_a::t1": "PASSED", "test_a::t2": "SKIPPED", "test_b::t3": "FAILED", "test_c::t9": "ABSENT"}, r
    assert not r["ok"] and required_status([], [])["ok"] is False
    only_skipped = required_status_text(xml, ["test_a::t2"])
    assert only_skipped == {"required": {"test_a::t2": "SKIPPED"}, "ok": False}, only_skipped   # a skip is never a pass
    twice = ('<testsuites><testsuite><testcase classname="m" name="t"><error message="setup"/></testcase>'
             '<testcase classname="m" name="t"/></testsuite></testsuites>')            # an error, then a pass: last-wins would read PASSED
    assert required_status_text(twice, ["m::t"])["required"] == {"m::t": "ERROR"}, "worst of, not last"
    assert parse_required("a.b::c d::e[x-1]") == ["a.b::c", "d::e[x-1]"]
    for bad in ("nocolons", "a::b;rm", "a::b c"):            # "c" has no classname
        try:
            parse_required(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    print("gc_box self-test OK")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--card")
    g.add_argument("--junit")
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--require")
    ap.add_argument("--junits", nargs="*", default=[])
    ap.add_argument("--declared")
    ap.add_argument("--suite", default="suite")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.require is not None:
        try:
            ids = parse_required(a.require)
        except ValueError as e:
            print(f"GC_REQUIRE refused: {e}")
            return 78
        r = required_status([p for p in a.junits if os.path.exists(p)], ids)
        print("GC_REQUIRE " + json.dumps(r))
        return 0 if r["ok"] else 30
    if a.card is not None:
        try:
            ok, why = card_ok(a.card, a.declared)
        except ValueError as e:
            print(f"GC_CARD refused: {e}")
            return 78
        print(f"GC_CARD {'ok' if ok else 'REFUSED'}: {why}")
        return 0 if ok else 15
    s = junit_summary(a.junit, a.suite)
    print("GC_SUITE " + json.dumps(s))
    return 0 if s["ok"] else 30


if __name__ == "__main__":
    sys.exit(main())
