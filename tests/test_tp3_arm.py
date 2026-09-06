"""CI wrapper for ``bench/tp3/tp3_arm.py --selftest`` (#434 follow-up, CEO/Warden review on PR #442/#447).

The T10 detector dry-runs (115 on a 30-layer K-as-V stack, 4 x L on a plain stack, REFUSED on a
missing k_proj) plus the whole tp2-derived bookkeeping suite live in the harness's ``--selftest``
mode, which CI did not run — a green suite that never executes is not a gate (the int4 tripwire's
own lesson, ci.yml). These tests run it end to end on CPU and assert the receipt-level outcomes,
including that a real run without ``--prereg`` is refused before any receipt exists.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "bench" / "tp3" / "tp3_arm.py"


def _run(*args):
    return subprocess.run([sys.executable, str(ARM), *args], capture_output=True, text=True, timeout=600, cwd=REPO)


def test_tp3_arm_selftest():
    p = _run("--selftest")
    tail = (p.stdout + p.stderr)[-3000:]
    assert p.returncode == 0, tail
    assert "SELFTEST OK" in p.stdout
    # the three #434 dry-runs against the REAL structural detector
    assert "'tiny_keqv30': 115" in p.stdout
    assert "'tiny_plain4': 16" in p.stdout
    assert "'tiny_missing_k': ['layers.1']" in p.stdout
    # the --prereg wiring arm: assert on the written receipt, not the print format (CEO LOW on #447)
    d = Path(re.search(r"SELFTEST OK dir=(\S+)", p.stdout).group(1))
    rec = json.loads((d / "tiny_unsloth_ckpt_unsloth_prereg.json").read_text())
    assert rec["prereg"] == "p41/P41-PREREG.md", rec["prereg"]
    assert json.loads((d / "tiny_e4b_reference_attn4.json").read_text())["prereg"] == "tp2/P40-PREREG.md"


def test_real_run_without_prereg_refuses():
    p = _run("--framework", "e4b", "--arm", "fused")   # no --selftest, no --prereg: refuse before any cell or stub
    assert p.returncode == 2, (p.returncode, (p.stdout + p.stderr)[-2000:])
    assert "--prereg is required" in p.stderr
