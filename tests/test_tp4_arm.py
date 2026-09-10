"""CI wrapper for ``bench/tp4/tp4_arm.py --selftest`` (lane tp4, TP4-PREREG.md).

The selftest drives all three framework branches (e4b / Unsloth / plain HF+PEFT+bnb) through the one ``run_arm``
on CPU with mocked kernels, plus the T12/T13/T14 additions (Alpaca template, micro-batches of 2 with padding, the
linear-with-warmup schedule) and the tp3 T10 detector dry-runs. A green suite that never executes is not a gate, so
CI runs it end to end and asserts the receipt-level outcomes -- including that a real run without ``--prereg`` is
refused before any receipt exists, and that the pinned Alpaca subset builder reproduces the registered sha when the
source file is present (skipped without it: the test never downloads).
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "bench" / "tp4" / "tp4_arm.py"
ALPACA = REPO / "bench" / "tp4" / "tp4_alpaca.py"
REGISTERED_DS_SHA = "5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb"   # TP4-PREREG.md "Fixture"


def _run(*args, script=ARM):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=900, cwd=REPO)


def test_tp4_arm_selftest():
    p = _run("--selftest")
    tail = (p.stdout + p.stderr)[-3000:]
    assert p.returncode == 0, tail
    assert "SELFTEST OK" in p.stdout
    # the tp3 T10 dry-runs against the REAL structural detector still pass through this copy
    assert "'tiny_keqv30': 115" in p.stdout and "'tiny_plain4': 16" in p.stdout and "'tiny_missing_k': ['layers.1']" in p.stdout
    d = Path(re.search(r"SELFTEST OK dir=(\S+)", p.stdout).group(1))
    # receipts name THIS lane's pre-registration
    assert json.loads((d / "tiny_e4b_reference_attn4.json").read_text())["prereg"] == "tp4/TP4-PREREG.md"
    # T11: the hf arm wrote a receipt with the module-level counter and the field regime recorded by the census
    hf = json.loads((d / "tiny_hf_hf_peft.json").read_text())
    assert hf["status"] == "ok" and hf["kernel_counter_key"] == "experts_forward" and hf["hf_targets"]["n_target_parameters"] == 4
    # T12/T13/T14: micro-batch 2 on the alpaca template with padding and a linear schedule, every framework
    for fw, tag in (("e4b", "fused_attn4"), ("unsloth", "ckpt_unsloth"), ("hf", "hf_peft")):
        r = json.loads((d / f"tinya_{fw}_{tag}.json").read_text())
        assert r["status"] == "ok" and r["micro_batch"] == 2 and r["template"] == "alpaca" and r["tokens_padded_total"] > 0, (fw, r["status"])
        assert r["lr_per_step"][0] == 0.0 and r["optimizer"].endswith("schedule=linear warmup_steps=3"), (fw, r["optimizer"])


def test_real_run_without_prereg_refuses():
    p = _run("--framework", "hf", "--arm", "hf")   # no --selftest, no --prereg: refuse before any cell or stub
    assert p.returncode == 2, (p.returncode, (p.stdout + p.stderr)[-2000:])
    assert "--prereg is required" in p.stderr


def test_alpaca_builder_refuses_a_wrong_source(tmp_path):
    bad = tmp_path / "alpaca_data_cleaned.json"
    bad.write_text("[]")
    p = _run("--src", str(bad), "--out", str(tmp_path / "ds.json"), script=ALPACA)
    assert p.returncode == 13 and "SOURCE MISMATCH" in p.stdout, (p.returncode, p.stdout[-500:])


@pytest.mark.skipif(not os.environ.get("TP4_ALPACA_SRC"), reason="set TP4_ALPACA_SRC to a local copy of the pinned alpaca_data_cleaned.json; the test never downloads")
def test_alpaca_builder_reproduces_the_registered_sha(tmp_path):
    out = tmp_path / "ds_alpaca.json"
    p = _run("--src", os.environ["TP4_ALPACA_SRC"], "--out", str(out), script=ALPACA)
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-500:]
    assert hashlib.sha256(out.read_bytes()).hexdigest() == REGISTERED_DS_SHA
