"""CI wrapper for ``bench/tc1/tc1_reduce.py --selftest`` (lane TC1): the reducer's readings on hand-built receipts.

The selftest builds a complete OK receipt per registered arm and drives every reading the registration names: the VERDICT
column (exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN), the matched-set
validity predicates (one receipt per predicate that MUST read VOID), the two-draw stability (one UNSTABLE), the equivalence
bands (one DIVERGENT, one COMPARABLE), the frozen-base readings and the P1-P10 scoring. A reducer whose VOID cannot fire is
a reducer that quotes everything, so the negative cases are the point. Lane TC1b (the `qwen3curve` token) adds the curve table and
reading, the plateau test, time to target, the 11..200 vs 11..20 speed check, the anchor pair and the two scaling points, with a
DIVERGENT curve, a refuted plateau, a failing anchor, a VOID r64 pair and a DOES-NOT-TRAVEL row among the selftest's cases.
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


# ----------------------------------------------------------------------------- lane TC1b (the qwen3curve token)
def test_tc1b_selftest_cases_and_failing_lines():
    p = subprocess.run([sys.executable, str(REDUCE), "--selftest"], capture_output=True, text=True, timeout=300, cwd=REPO)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    assert int(re.search(r"REDUCE SELFTEST OK cases=(\d+)", p.stdout).group(1)) >= 44, p.stdout[-300:]
    for needle in ("FAILING-CASE TC1b-P1 (reducer): curve DIVERGENT -- first paired |delta| > 0.02 at step 120", "FAILING-CASE TC1b-P2 (reducer): NOT-P38-SHAPE",
                   "FAILING-CASE TC1b-P4 (reducer): anchor unsloth/e4b 1.850 -> FINDING", "FAILING-CASE TC1b-r64 (reducer): trainable 2570059777 != e4b's 2570059776",
                   "FAILING-CASE TC1b-P3 (reducer): unsloth 11..200 median 3.500 vs TC1's 11..20 3.030 (+15.5%) -> DOES-NOT-TRAVEL"):
        assert needle in p.stdout, needle


def test_tc1b_registration_constants_and_arm_order():
    R = _mod()
    assert R.CURVE_FAM == "qwen3curve" and "qwen3curve" in R.FAMS and R.N_LAYERS["qwen3curve"] == 48
    assert R.EXPECTED["qwen3curve"] == [("e4b", "fused_attn4_m_200"), ("unsloth", "ckpt_unsloth_m_200"), ("e4b", "fused_attn4_shipped_200"),
                                        ("e4b", "fused_attn4_p38"), ("unsloth", "ckpt_unsloth_p38"), ("unsloth", "ckpt_unsloth_p38_t28"),
                                        ("e4b", "fused_attn4_m_t1"), ("unsloth", "ckpt_unsloth_m_t1"), ("e4b", "fused_attn4_m_r64"), ("unsloth", "ckpt_unsloth_m_r64")]
    assert R.CURVE_ARMS == R.EXPECTED["qwen3curve"][:3]
    for t in ("fused_attn4_m_200", "ckpt_unsloth_m_200", "fused_attn4_m_t1", "ckpt_unsloth_m_t1", "fused_attn4_m_r64", "ckpt_unsloth_m_r64"):
        assert t in R.MATCHED, t
    for t in ("fused_attn4_shipped_200", "fused_attn4_p38", "ckpt_unsloth_p38", "ckpt_unsloth_p38_t28"):
        assert t not in R.MATCHED, t                     # native rows: no R3
    # TC1's registration is untouched by the addition
    assert "fused_attn4_shipped" not in R.MATCHED and R.EXPECTED["qwen3"][0] == ("e4b", "fused_attn4_m") and R.EQUIV_ANCHOR == ("e4b", "fused_attn4_m")
    assert R.CURVE_BAND == 0.02 and "draft" in R.CURVE_BAND_NOTE and R.PLATEAU_GAP == 0.01 and R.PLATEAU_EARLY_STEP == 40 and R.TARGET_MARGIN == 0.02
    assert R.TRAVEL_TOL == 0.10 and R.ANCHOR_TOL == 0.10 and R.TP2_ANCHOR == 1.457 and R.P38_ANCHOR == 1.413 and R.TP4_ANCHOR_RATIO is None   # tp4's own anchor row is not in this tree
    assert R.CURVE_N == {"fused_attn4_m_200": 200, "ckpt_unsloth_m_200": 200, "fused_attn4_shipped_200": 200, "fused_attn4_p38": 60, "ckpt_unsloth_p38": 60, "ckpt_unsloth_p38_t28": 60,
                         "fused_attn4_m_t1": 20, "ckpt_unsloth_m_t1": 20, "fused_attn4_m_r64": 20, "ckpt_unsloth_m_r64": 20}
    assert R.CURVE_TRAINABLE == {"200": 642514944, "p38": 321257472} and set(R.CURVE_GROUPS) == {"200", "p38", "t1", "r64"}
    assert R.CURVE_GROUPS["r64"] == ("e4b", "fused_attn4_m_r64") and R.CURVE_SCALING.keys() == {"t1", "r64"} and all("position" not in v for v in R.CURVE_SCALING.values())


def test_tc1b_readings_on_hand_built_receipts():
    R = _mod()
    C = R.reduce_curve_family("qwen3curve", R._curve_set(), {})
    assert C["curve"]["reading"] == "EQUIVALENT-AT-EVERY-EVAL" and C["plateau"]["reading"] == "REPRODUCES-P38"
    assert C["anchor"]["grouped_mm"]["reading"] == "AGREES" and C["anchor"]["t28"]["quoted"] and C["scaling"]["t1"]["quoted"] and C["scaling"]["r64"]["quoted"]
    assert C["scaling"]["t1"]["kind"] == "scaling point" and [n for n, _, _ in C["scaling"]["t1"]["predicates"]] == [
        "both arms VALID", "same trainable count within the pair", "lora_path_loop == 0 on every e4b step",
        "Unsloth backend counters: torch._grouped_mm >= 6*L*A per step, manual fallback 0"]
    assert {p: v for p, _, v, _ in R.score_curve_predictions({"qwen3curve": C})} == {"P1": "HELD", "P2": "HELD", "P3": "UNTESTED", "P4": "HELD"}
    # a DIVERGENT curve names the first step and the sign at 200
    S = R._curve_set()
    S[("unsloth", "ckpt_unsloth_m_200")] = R._curve_receipt("unsloth", "ckpt_unsloth_m_200", "unsloth", s=3.0, heldouts=[2.0, 1.90, 1.85, 1.85, 1.83, 1.82])
    c = R.reduce_curve_family("qwen3curve", S, {})["curve"]
    assert c["reading"] == "DIVERGENT" and c["first_divergent_step"] == 120 and c["sign_at_N"] == "+" and abs(c["max_abs"] - 0.03) < 1e-9
    # the receipt file names of the curve token parse to the right (fam, fw, tag), the t28 variant included
    import json
    import tempfile
    d = tempfile.mkdtemp()
    for (fw, tag), r in R._curve_set().items():
        json.dump(r, open(f"{d}/qwen3curve_{fw}_{tag}.json", "w"))
    recs = R.load(d)
    assert set(recs) == {"qwen3curve"} and set(recs["qwen3curve"]) == set(R.EXPECTED["qwen3curve"])
    F = R.reduce_dir(d, 20)
    text = R.render(F, d)
    assert "## TC1b predictions P1–P4" in text and "## Predictions P1–P10" not in text and "SCALING POINT `r64` (never a position)" in text
