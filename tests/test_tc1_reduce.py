"""CI wrapper for ``bench/tc1/tc1_reduce.py --selftest`` (lane TC1): the reducer's readings on hand-built receipts.

The selftest builds a complete OK receipt per registered arm and drives every reading the registration names: the VERDICT
column (exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN), the matched-set
validity predicates (one receipt per predicate that MUST read VOID), the two-draw stability (one UNSTABLE), the equivalence
bands (one DIVERGENT, one COMPARABLE), the frozen-base readings and the P1-P10 scoring. A reducer whose VOID cannot fire is
a reducer that quotes everything, so the negative cases are the point.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REDUCE = REPO / "bench" / "tc1" / "tc1_reduce.py"


def _mod():
    spec = importlib.util.spec_from_file_location("tc1_reduce_under_test", REDUCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_tc1_reduce_selftest():
    p = subprocess.run([sys.executable, str(REDUCE), "--selftest"], capture_output=True, text=True, timeout=300, cwd=REPO)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    m = re.search(r"REDUCE SELFTEST OK cases=(\d+)", p.stdout)
    assert m and int(m.group(1)) >= 16, p.stdout[-500:]


def test_verdict_vocabulary_and_mapping():
    R = _mod()
    assert set(R.VERDICTS) == {"VALID", "VOID", "QUALITY_FAIL", "OOM", "UNSUPPORTED", "HARNESS_ERROR", "ALARM", "NOT_RUN"}
    assert R.verdict_of("OK", "VALID", 0.01) == "VALID" and R.verdict_of("OK", "VALID", -0.051) == "QUALITY_FAIL" and R.verdict_of("OK", "VOID", 0.0) == "VOID"
    assert R.verdict_of("OK", "VALID", None) == "VALID"           # no anchor to read quality against: the predicates alone
    assert R.verdict_of("OOM", "—", None) == "OOM"
    for st in ("REFUSED", "INSTALL_FAILED", "LOAD_FAULT"):
        assert R.verdict_of(st, "—", None) == "UNSUPPORTED"
    for st in ("HARNESS_ERROR", "ALARM", "NOT_RUN"):
        assert R.verdict_of(st, "—", None) == st
    assert R.STATUS_MAP["phase_alarm"] == "ALARM" and R.STATUS_MAP["c1_failed"] == "OK"


def test_registered_arm_order_and_matched_set():
    R = _mod()
    assert R.EXPECTED["qwen3"] == [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "reference_attn4_m"), ("e4b", "fused_attn4_m_d2"),
                                   ("unsloth", "ckpt_unsloth_m_d2"), ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"),
                                   ("e4b", "fused_attn4_m_prof"), ("unsloth", "ckpt_unsloth_prof")]
    assert R.EXPECTED["qwen3native"] == [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_best"), ("unsloth", "ckpt_unsloth_t28"), ("unsloth", "ckpt_unsloth_triton"),
                                         ("e4b", "fused_attn4_shipped"), ("e4b", "fused_attn4_m_nodgrad"), ("e4b", "fused_attn4_m_t212"),
                                         ("axolotl", "ckpt_axolotl_best"), ("hf", "hf_peft_m_mb1_t214")]
    assert set(R.LABELLED) == {("unsloth", "ckpt_unsloth_t28"), ("unsloth", "ckpt_unsloth_triton"), ("unsloth", "ckpt_unsloth_best"), ("axolotl", "ckpt_axolotl_best"),
                               ("e4b", "fused_attn4_shipped"), ("e4b", "fused_attn4_m_nodgrad"), ("e4b", "fused_attn4_m_t212"), ("hf", "hf_peft_m_mb1_t214")}
    assert R.PRIMARY["unsloth"] == "ckpt_unsloth_m" and R.P8_BUSY_MIN == 0.5 and R.TP4_T28_S_PER_STEP == 29.05
    assert R.EQUIV_ANCHOR == ("e4b", "fused_attn4_m") and R.REFERENCE == ("e4b", "reference_attn4_m")
    assert "fused_attn4_shipped" not in R.MATCHED and "ckpt_unsloth_best" not in R.MATCHED and "reference_attn4_m" in R.MATCHED and "fused_attn4_m_t212" in R.MATCHED
    assert R.STABILITY == {"e4b": 0.05} and R.STABILITY_OTHER == 0.05 and R.COMPARABLE == 0.05 and R.STEP0_SAME == 0.01 and R.STEP0_VOID == 0.05 and R.EQUIV_FLOOR == 0.005 and R.GMM_FACTOR == 6
    assert R.step0_class(0.01) == "SAME-BYTES-CLASS" and R.step0_class(0.03) == "NEAR" and R.step0_class(0.06) == "VOID"


def test_receipt_filenames_with_an_axolotl_framework_are_read():
    R = _mod()
    import json
    import tempfile
    d = tempfile.mkdtemp()
    json.dump(R._stub("axolotl", "ckpt_axolotl_m", "axolotl", "install_failed", "pip rc=1"), open(f"{d}/qwen3_axolotl_ckpt_axolotl_m.json", "w"))
    recs = R.load(d)
    assert ("axolotl", "ckpt_axolotl_m") in recs["qwen3"]
    F = R.reduce_family("qwen3", recs["qwen3"], {}, 20)
    assert F["verdicts"][("axolotl", "ckpt_axolotl_m")] == "UNSUPPORTED"
