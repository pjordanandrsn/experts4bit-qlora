"""Golden test for the paired-transfer summarizer (review order O-2).

The external review's E3 statistics (docs/MODE_DECOUPLED_ADAPTERS.md) were computed from
runs/query_jobs; the first run of scripts/summarize_paired_transfer.py on that same tree is
committed as tests/fixtures/paired_transfer_golden.md. This test regenerates it and compares
byte-for-byte — the review's numbers stay reproducible from the shipped data forever.
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_paired_transfer_reproduces_review_numbers(tmp_path):
    out_md = tmp_path / "paired_transfer.md"
    out_csv = tmp_path / "paired_transfer.csv"
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "summarize_paired_transfer.py"),
         "--query-jobs", os.path.join(REPO, "runs", "query_jobs"),
         "--out-md", str(out_md), "--out-csv", str(out_csv)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    golden = open(os.path.join(REPO, "tests", "fixtures", "paired_transfer_golden.md"), encoding="utf-8").read()
    assert out_md.read_text(encoding="utf-8") == golden

    # The review's E3 anchors, verbatim in the output (sign convention: script emits
    # L(int8-resident) - L(nf4-resident), the negation of the review's L(nf4)-L(int8)).
    assert "| int8-resident | L(→int8-resident) − L(→nf4-resident) | -0.0069 / -0.0106 / -0.0050 | -0.0075 | 0.0028 | -4.55 | 0/3 |" in golden
    assert "| nf4-resident | int8-resident − nf4-resident | +0.0054 / +0.0048 / +0.0137 | +0.0079 | 0.0050 | 2.77 | 3/3 |" in golden
