# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""D0's result re-derives from the public record (bench/d0/REDERIVE.md).

The reducer that ran is in the receipt. Over each fixture's stored pass records it must reproduce that fixture's
summary.json exactly, and the receipt's files must match its SHA256SUMS. CPU only, standard library only.
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO, "bench", "d0", "receipts", "d0-p224s3")
REDUCE = os.path.join(RUN, "tools", "d0_reduce.py")


def test_the_receipt_matches_its_sums():
    bad = []
    with open(os.path.join(RUN, "SHA256SUMS")) as f:
        for line in f:
            digest, name = line.rstrip("\n").split("  ", 1)
            with open(os.path.join(RUN, name), "rb") as g:
                if hashlib.sha256(g.read()).hexdigest() != digest:
                    bad.append(name)
    assert not bad, bad


@pytest.mark.parametrize("fixture", ["dq", "nodq"])
def test_the_reducer_reproduces_the_summary(tmp_path, fixture):
    out = tmp_path / f"{fixture}.json"
    subprocess.run([sys.executable, REDUCE, "--pass-dir", os.path.join(RUN, fixture, "pass"), "--out", str(out)],
                   check=True, capture_output=True, text=True, timeout=300)
    with open(os.path.join(RUN, fixture, "summary.json")) as f:
        stored = json.load(f)
    assert json.loads(out.read_text()) == stored


@pytest.mark.parametrize("fixture", ["dq", "nodq"])
def test_the_stated_result(fixture):
    """RESULTS-d0.md's headline, read from the stored summary: RESOLVED, and every cell RETAINED."""
    with open(os.path.join(RUN, fixture, "summary.json")) as f:
        s = json.load(f)
    assert s["step2"]["resolved"] is True
    assert s["step2"]["improvement"] > 3 * s["step2"]["improvement_se"]
    assert set(s["labels"].values()) == {"RETAINED"} and len(s["labels"]) == 6
