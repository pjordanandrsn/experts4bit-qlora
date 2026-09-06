"""CI wrapper for ``bench/tp3/tp3_arm.py --selftest`` (#434 follow-up, CEO/Warden review on PR #442).

The T10 detector dry-runs (115 on a 30-layer K-as-V stack, 4 x L on a plain stack, REFUSED on a
missing k_proj) plus the whole tp2-derived bookkeeping suite live in the harness's ``--selftest``
mode, which CI did not run — a green suite that never executes is not a gate (the int4 tripwire's
own lesson, ci.yml). This test runs it end to end on CPU and asserts the receipt-level outcomes.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_tp3_arm_selftest():
    p = subprocess.run(
        [sys.executable, str(REPO / "bench" / "tp3" / "tp3_arm.py"), "--selftest"],
        capture_output=True, text=True, timeout=600, cwd=REPO,
    )
    tail = (p.stdout + p.stderr)[-3000:]
    assert p.returncode == 0, tail
    assert "SELFTEST OK" in p.stdout
    # the three #434 dry-runs against the REAL structural detector, and the --prereg wiring arm
    assert "'tiny_keqv30': 115" in p.stdout
    assert "'tiny_plain4': 16" in p.stdout
    assert "'tiny_missing_k': ['layers.1']" in p.stdout
    assert '"prereg": "p41/P41-PREREG.md"' in p.stdout
