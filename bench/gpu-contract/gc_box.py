# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/gpu-contract: the box-side checks the runner calls (no torch import; standard library only).

    python gc_box.py --card "ok torch=2.8.0+cu128 cc=8.9 sms=128" --declared 8.9/128   (exit 15 outside the class)
    python gc_box.py --junit report.xml --suite gnf4                                    (one JSON line; exit 30 on a failure)
    python gc_box.py --self-test

A declared card class is ``MAJOR.MINOR/SMS``, where SMS is an exact count (``128``) or a floor (``>=160``). The class is the
device's compute capability and SM count as torch reports them, never its marketing name.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET

CLASS = re.compile(r"^(\d+)\.(\d+)/(>=)?(\d+)$")
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
    print("gc_box self-test OK")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--card")
    g.add_argument("--junit")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--declared")
    ap.add_argument("--suite", default="suite")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
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
