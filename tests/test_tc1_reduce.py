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


# ----------------------------------------------------------------------------- lane TC2 (the tc2small / tc2big tokens)
def test_tc2_registration_constants_and_arm_order():
    R = _mod()
    assert R.TC2_FAMS == ["granite", "olmoe", "gptoss", "qwen3_5", "mixtral"] and all(f in R.FAMS for f in R.TC2_FAMS)
    assert R.TC2_TOKENS == {"tc2small": ["granite", "olmoe", "gptoss"], "tc2big": ["qwen3_5", "mixtral"]}
    assert {f: R.N_LAYERS[f] for f in R.TC2_FAMS} == {"granite": 32, "olmoe": 16, "gptoss": 24, "qwen3_5": 40, "mixtral": 32}
    assert {f: R.ATTN_CENSUS[f] for f in R.TC2_FAMS} == {"granite": 128, "olmoe": 64, "gptoss": None, "qwen3_5": None, "mixtral": 128} and R.ATTN_CENSUS["qwen3"] == 192
    assert R.TC2_MODELS == {"granite": ("ibm-granite/granite-3.1-3b-a800m-instruct", "a02780686e08a03fe0d2679a293b5c74a90efa89", 32),
                            "olmoe": ("allenai/OLMoE-1B-7B-0924-Instruct", "7f1c97f440f06ce36705e4f2b843edb5925f4498", 16),
                            "gptoss": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", 24),
                            "qwen3_5": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0", 40),
                            "mixtral": ("mistralai/Mixtral-8x7B-Instruct-v0.1", "eba92302a2861cdc0098cc54bc9f17cb2c47eb61", 32)}
    assert R.EXPECTED["granite"] == [("e4b", "fused_attn4_m"), ("hf", "hf_peft_m"), ("e4b", "reference_attn4_m"), ("e4b", "fused_attn4_m_d2"), ("hf", "hf_peft_m_d2"),
                                     ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_experts"), ("hf", "hf_peft_m_t214"),
                                     ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_best"), ("e4b", "fused_attn4_shipped")]
    assert R.EXPECTED["olmoe"] == [k for k in R.EXPECTED["granite"] if k != ("unsloth", "ckpt_unsloth_m_experts")]
    assert R.EXPECTED["gptoss"] == [("e4b", "fused_attn4_m"), ("e4b", "attn_only_m"), ("e4b", "attn_only_m_d2"), ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_mxfp4"),
                                    ("unsloth", "ckpt_unsloth_mxfp4_d2"), ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"), ("e4b", "reference_attn4_m")]
    assert R.EXPECTED["qwen3_5"] == [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "fused_attn4_m_d2"), ("unsloth", "ckpt_unsloth_m_d2"), ("unsloth", "ckpt_unsloth_m_experts"),
                                     ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_best"), ("e4b", "fused_attn4_shipped"), ("e4b", "reference_attn4_m")]
    assert R.EXPECTED["mixtral"] == [k for k in R.EXPECTED["qwen3_5"] if k != ("unsloth", "ckpt_unsloth_m_experts")]
    assert R.TC2_ANCHOR == {"gptoss": ("e4b", "attn_only_m")} and R.anchor_of("gptoss") == ("e4b", "attn_only_m") and R.anchor_of("granite") == ("e4b", "fused_attn4_m")
    assert set(R.NO_COMMON_SET) == {"gptoss"} and set(R.FOOTPRINT_FAMS) == {"mixtral"}
    for t in ("attn_only_m", "attn_only_m_d2", "ckpt_unsloth_m_experts", "ckpt_unsloth_mxfp4", "ckpt_unsloth_mxfp4_d2", "hf_peft_m_d2", "hf_peft_m_t214"):
        assert t in R.MATCHED, t
    assert R.DRAW2[("e4b", "attn_only_m")] == ("e4b", "attn_only_m_d2") and R.DRAW2[("hf", "hf_peft_m")] == ("hf", "hf_peft_m_d2") and R.DRAW2[("unsloth", "ckpt_unsloth_mxfp4")] == ("unsloth", "ckpt_unsloth_mxfp4_d2")
    assert R.registered_draw2("granite", ("hf", "hf_peft_m")) == ("hf", "hf_peft_m_d2") and R.registered_draw2("qwen3", ("hf", "hf_peft_m")) is None
    assert R.registered_draw2("qwen3", ("unsloth", "ckpt_unsloth_m")) == ("unsloth", "ckpt_unsloth_m_d2") and R.registered_draw2("qwen3native", ("e4b", "fused_attn4_m")) is None
    # TC1's registration is untouched: its LABELLED set, its qwen3 order and anchors; TC2's labelled rows live in their own dict
    assert set(R.LABELLED) == {("unsloth", "ckpt_unsloth_t28"), ("unsloth", "ckpt_unsloth_triton"), ("unsloth", "ckpt_unsloth_best"), ("axolotl", "ckpt_axolotl_best"),
                               ("e4b", "fused_attn4_shipped"), ("e4b", "fused_attn4_m_nodgrad"), ("e4b", "fused_attn4_m_t212"), ("hf", "hf_peft_m_mb1_t214")}
    assert set(R.TC2_LABELLED) == {("unsloth", "ckpt_unsloth_m_experts"), ("hf", "hf_peft_m_t214"), ("unsloth", "ckpt_unsloth_mxfp4")} and set(R.ALL_LABELLED) == set(R.LABELLED) | set(R.TC2_LABELLED)
    assert R.EXPECTED["qwen3"][0] == ("e4b", "fused_attn4_m") and R.QUALITY_ANCHOR == ("e4b", "fused_attn4_m") and R.EQUIV_ANCHOR == ("e4b", "fused_attn4_m")
    assert R.TC2_P1_HF_BAND == (1.1, 1.6) and R.TC2_P2_UNS_BAND == (1.2, 3.0) and R.TC2_P2_HF_BAND == (1.5, 2.5) and R.TC2_P4_UNS_BAND == (2.0, 6.0)
    assert R.TP4_QWEN3_5_E4B_S_PER_STEP == 6.3344 and R.TC2_P4_E4B_TOL == 0.15 and R.TC2_P5_BAND == (0.3, 0.5) and R.TC2_P5_PEAK_X == 8.0
    assert R.TP2_MIXTRAL == {"ratio_unsloth_over_e4b": 0.361, "peak_unsloth_gb": 29.16, "peak_e4b_gb": 3.22}
    assert R.UNSLOTH_BANNER_PER_EXPERT == "Detected MoE model with per-expert Linear experts" and R.UNSLOTH_BANNER == "Enabling LoRA on MoE parameters"


def test_tc2_selftest_cases_and_failing_lines():
    p = subprocess.run([sys.executable, str(REDUCE), "--selftest"], capture_output=True, text=True, timeout=300, cwd=REPO)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    assert int(re.search(r"REDUCE SELFTEST OK cases=(\d+)", p.stdout).group(1)) >= 53, p.stdout[-300:]
    for needle in ("FAILING-CASE TC2-attention-only (reducer): attention-only: trainable 5200000 != e4b's 99600000 (unsloth adapted no expert parameter",
                   "FAILING-CASE TC2-no-common-set (reducer): a ratio on gpt-oss is never quoted -- - **NO COMMON ADAPTER SET (unsloth/ckpt_unsloth_mxfp4 vs e4b/attn_only_m)",
                   "FAILING-CASE TC2-packed (reducer): packed expert parameters 0 < 2*24 (expert parameter classes {'Parameter': 48}): the experts are not packed",
                   "FAILING-CASE TC2-footprint (reducer): Unsloth OOM -> - **FOOTPRINT (e4b under expert offload (--offload 1, tp2 / tp4's arm) vs Unsloth resident): not readable**",
                   "FAILING-CASE TC2-dispatch (reducer): experts_implementation requested 'grouped_mm', accepted True, config 'grouped_mm'; torch grouped_mm calls/step min 0 (F.grouped_mm 0): dispatch did NOT reach grouped_mm (recorded, not VOID)"):
        assert needle in p.stdout, needle


def test_tc2_readings_on_hand_built_receipts():
    R = _mod()
    G = R.reduce_family("granite", R._tc2_set("granite"), {}, None)
    assert G["positions"]["hf"]["label"] == "HF (bf16 experts)" and G["positions"]["hf"]["quoted"] and G["verdicts"][("unsloth", "ckpt_unsloth_m")] == "VOID"
    assert next(x for x in G["rows"] if x["tag"] == "ckpt_unsloth_m")["why"].startswith("attention-only:") and not G["positions"]["unsloth"]["quoted"]
    GO = R.reduce_family("gptoss", R._tc2_set("gptoss"), {}, None)
    assert all(pz["no_common_set"] and not pz["quoted"] for pz in GO["positions"].values()) and GO["verdicts"][("unsloth", "ckpt_unsloth_mxfp4")] == "VALID" and GO["verdicts"][("e4b", "attn_only_m")] == "VALID"
    assert "NO COMMON ADAPTER SET" in R.pos_lines(GO["positions"]["unsloth_mxfp4"], 60)[0] and "5000000 trainable" in R.pos_lines(GO["positions"]["unsloth_mxfp4"], 60)[0]
    MX = R.reduce_family("mixtral", R._tc2_set("mixtral"), {}, None)
    assert MX["footprint"]["readable"] and R.family_block(MX)[1].startswith("- **FOOTPRINT (e4b under expert offload") and abs(MX["footprint"]["peak_ratio"] - 29.18 / 3.235) < 1e-9
    F = {fam: R.reduce_family(fam, R._tc2_set(fam), {}, None) for fam in R.TC2_FAMS}
    assert {p: v for p, _, v, _ in R.score_tc2_predictions(F)} == {f"P{i}": "HELD" for i in range(1, 8)}
    # a t214 arm whose dispatch did not reach grouped_mm is VALID with the note recorded
    S = R._tc2_set("granite")
    S[("hf", "hf_peft_m_t214")]["hf_experts_dispatch"].update({"torch_grouped_mm_calls_per_step_min": 0, "reached_grouped_mm": False})
    x = next(x for x in R.reduce_family("granite", S, {}, None)["rows"] if x["tag"] == "hf_peft_m_t214")
    assert x["verdict"] == "VALID" and "did NOT reach grouped_mm" in x["dispatch"]
    # the receipt file names parse (qwen3_5's underscore included) and the printer renders the TC2 table without TC1's
    import json
    import tempfile
    d = tempfile.mkdtemp()
    for fam in R.TC2_FAMS:
        for (fw, tag), r in R._tc2_set(fam).items():
            json.dump(r, open(f"{d}/{fam}_{fw}_{tag}.json", "w"))
    recs = R.load(d)
    assert set(recs) == set(R.TC2_FAMS) and set(recs["qwen3_5"]) == set(R.EXPECTED["qwen3_5"]) and set(recs["gptoss"]) == set(R.EXPECTED["gptoss"])
    text = R.render(R.reduce_dir(d, None), d)
    assert "## TC2 predictions P1–P7" in text and "## Predictions P1–P10" not in text and "MATCHED POSITION: s/step ratio HF (bf16 experts) / e4b = 1.292**" in text


# ----------------------------------------------------------------------------- TC1-PREREG amendment 3 (the qwen3axolotl token)
def test_amendment_3_profile_sidecars_are_not_receipts_and_the_axolotl_token_is_registered():
    R = _mod()
    import json
    import tempfile
    d = tempfile.mkdtemp()
    json.dump(R._receipt("e4b", "fused_attn4_m", "fused"), open(f"{d}/qwen3_e4b_fused_attn4_m.json", "w"))
    json.dump({"top_kernels": []}, open(f"{d}/qwen3_e4b_fused_attn4_m_prof_profile.json", "w"))      # box -16 read this sidecar as a HARNESS_ERROR row
    assert set(R.load(d)["qwen3"]) == {("e4b", "fused_attn4_m")}
    assert R.AX_FAM == "qwen3axolotl" and R.AX_FAM in R.FAMS and R.N_LAYERS[R.AX_FAM] == 48 and R.ATTN_CENSUS[R.AX_FAM] == 192
    assert R.EXPECTED[R.AX_FAM] == [("e4b", "fused_attn4_m"), ("axolotl", "ckpt_axolotl_m"), ("e4b", "fused_attn4_m_d2"), ("axolotl", "ckpt_axolotl_m_d2"),
                                    ("axolotl", "ckpt_axolotl_best"), ("hf", "hf_peft_m_mb1_t214")]
    assert R.DRAW2[("axolotl", "ckpt_axolotl_m")] == ("axolotl", "ckpt_axolotl_m_d2") and "ckpt_axolotl_m_d2" in R.MATCHED
    assert R.registered_draw2(R.AX_FAM, ("axolotl", "ckpt_axolotl_m")) == ("axolotl", "ckpt_axolotl_m_d2")
    assert R.registered_draw2("qwen3", ("axolotl", "ckpt_axolotl_m")) is None       # the judged box registers one axolotl draw, as before
    AXR = R.reduce_family(R.AX_FAM, R._ax_set(), {}, 20)
    assert AXR["positions"]["axolotl"]["quoted"] and abs(AXR["positions"]["axolotl"]["ratio"] - 1.990) < 1e-3
    assert {p: v for p, _, v, _ in R.score_predictions({R.AX_FAM: AXR})}["P6"] == "HELD"
# ----------------------------------------------------------------------------- lane TC3 (the frontier tokens `qwen3frontier` / `qwen3frontier12`)
def test_tc3_registration_constants_and_arm_order():
    R = _mod()
    assert R.FRONTIER_FAM == "qwen3frontier" and R.FRONTIER12_FAM == "qwen3frontier12" and set(R.FRONTIER_FAMS) <= set(R.FAMS)
    assert R.EXPECTED["qwen3frontier"] == [("e4b", "fused_attn4_m"), ("e4b", "fused_attn4_m_offload"), ("e4b", "fused_attn4_m_mb1"), ("e4b", "fused_attn4_shipped"), ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_mb1"),
                                           ("hf", "hf_peft_m"), ("hf", "hf_peft_m_offload"), ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_m_layeroffload"),
                                           ("axolotl", "ckpt_axolotl_m_zero3"), ("e4b", "reference_attn4_m_offload")]
    assert R.EXPECTED["qwen3frontier12"] == [("e4b", "fused_attn4_m_offload"), ("e4b", "fused_attn4_m_offload_d2"), ("e4b", "reference_attn4_m_offload"), ("e4b", "fused_attn4_m"),
                                             ("unsloth", "ckpt_unsloth_m_mb1"), ("hf", "hf_peft_m_mb1"), ("axolotl", "ckpt_axolotl_m")]
    assert R.FRONTIER_ANCHOR == ("e4b", "fused_attn4_m_offload") and R.FRONTIER_REF == ("e4b", "reference_attn4_m_offload") and R.RESIDENT_BAND == 0.02 and R.RESIDENT_KEY == ("qwen3", ("e4b", "fused_attn4_m"))
    assert R.DRAW2[("e4b", "fused_attn4_m_offload")] == ("e4b", "fused_attn4_m_offload_d2") and R.N_LAYERS["qwen3frontier"] == R.N_LAYERS["qwen3frontier12"] == 48
    for t in ("fused_attn4_m_offload", "fused_attn4_m_offload_d2", "reference_attn4_m_offload", "hf_peft_m_offload", "ckpt_axolotl_m_layeroffload", "ckpt_axolotl_m_zero3"):
        assert t in R.MATCHED, t
    assert R.P1_HF_SLOWER == 20.0 and R.P1_Z3_SLOWER == 5.0 and R.P1_Z3_HOST_GB == 60.0 and R.P4_FACTOR == 2.0
    assert set(R.FRONTIER_LEVER) >= {t for _, t in R.EXPECTED["qwen3frontier"]} | {t for _, t in R.EXPECTED["qwen3frontier12"]}
    # TC1 and TC1b untouched by the addition
    assert R.EXPECTED["qwen3"][0] == ("e4b", "fused_attn4_m") and R.EQUIV_ANCHOR == ("e4b", "fused_attn4_m") and R.CURVE_ARMS == R.EXPECTED["qwen3curve"][:3] and "fused_attn4_shipped" not in R.MATCHED


def test_tc3_selftest_cases_and_failing_lines():
    p = subprocess.run([sys.executable, str(REDUCE), "--selftest"], capture_output=True, text=True, timeout=300, cwd=REPO)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    assert int(re.search(r"REDUCE SELFTEST OK cases=(\d+)", p.stdout).group(1)) >= 50, p.stdout[-300:]
    for needle in ("FAILING-CASE TC3-fit (reducer): unsloth -> NO ARM COMPLETED on this box: `ckpt_unsloth_m` OOM, `ckpt_unsloth_m_mb1` OOM",
                   "FAILING-CASE TC3-hf-offload (reducer): UNSUPPORTED -- hf_offload: NotImplementedError at step 1: Cannot copy out of meta tensor",
                   "FAILING-CASE TC3-zero3 (reducer): UNSUPPORTED -- axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer",
                   "FAILING-CASE TC3-equiv (reducer): offload anchor VOID -> N-A", "FAILING-CASE TC3-P3 (reducer): median per-step |delta| vs TC1's resident 0.0300 > 0.02 -> DIVERGENT-FROM-RESIDENT",
                   "FAILING-CASE TC3-P2 (reducer):"):
        assert needle in p.stdout, needle


def test_tc3_readings_on_hand_built_receipts():
    import json
    import tempfile
    R = _mod()
    F = R.reduce_frontier_family(R.FRONTIER_FAM, R._frontier_set(), {}, 20)
    assert F["fit"]["unsloth"]["fits"] is False and F["fit"]["e4b"]["completed"] == ["fused_attn4_m_offload", "fused_attn4_m_mb1", "fused_attn4_shipped", "reference_attn4_m_offload"] and F["fit"]["hf"]["fits"] is False
    assert F["equivalence"][R.FRONTIER_REF]["reading"] == "EQUIVALENT" and F["parity"]["verdict"] == "PASS" and F["resident"]["reading"] == "UNTESTED"
    assert F["verdicts"][("hf", "hf_peft_m_offload")] == "UNSUPPORTED" and F["verdicts"][("axolotl", "ckpt_axolotl_m_zero3")] == "UNSUPPORTED"
    assert {p: v for p, _, v, _ in R.score_frontier_predictions({R.FRONTIER_FAM: F})} == {"P1": "HELD", "P3": "UNTESTED", "P4": "UNTESTED"}
    # with a TC1 dir: the resident reading and P4 as two measurements
    tc1 = {"qwen3": R.reduce_family("qwen3", R._good_set(), {}, 20)}
    F = R.reduce_frontier_family(R.FRONTIER_FAM, R._frontier_set(), {}, 20, tc1=tc1)
    assert F["resident"]["reading"] == "EQUIVALENT-TO-RESIDENT" and abs(F["resident"]["tc1_s"] - 1.01) < 1e-9
    sc = {p: (v, ev) for p, _, v, ev in R.score_frontier_predictions({R.FRONTIER_FAM: F})}
    assert sc["P3"][0] == "HELD" and sc["P4"][0] == "HELD" and "no cross-box ratio formed" in sc["P4"][1]
    # the 12 GB token: P2 reads frameworks; the receipt file names of both tokens parse and render their own table only
    F12 = R.reduce_frontier_family(R.FRONTIER12_FAM, R._frontier12_set(), {}, 20)
    assert F12["draws"][R.FRONTIER_ANCHOR]["verdict"] == "STABLE" and {p: v for p, _, v, _ in R.score_frontier_predictions({R.FRONTIER12_FAM: F12})} == {"P2": "HELD", "P3": "UNTESTED"}
    d = tempfile.mkdtemp()
    for fam, S in ((R.FRONTIER_FAM, R._frontier_set()), (R.FRONTIER12_FAM, R._frontier12_set())):
        for (fw, tag), r in S.items():
            json.dump(r, open(f"{d}/{fam}_{fw}_{tag}.json", "w"))
    recs = R.load(d)
    assert set(recs) == set(R.FRONTIER_FAMS) and set(recs["qwen3frontier"]) == set(R.EXPECTED["qwen3frontier"])
    text = R.render(R.reduce_dir(d, 20), d)
    assert "## TC3 predictions P1–P4" in text and "**(a) FIT TABLE**" in text and "## Predictions P1–P10" not in text and "MATCHED POSITION" not in text


def test_amendment_7_registers_the_native_best_token():
    R = _mod()
    assert R.NB_FAM == "qwen3nativebest" and R.NB_FAM in R.FAMS and R.N_LAYERS[R.NB_FAM] == 48 and R.ATTN_CENSUS[R.NB_FAM] == 192
    assert R.EXPECTED[R.NB_FAM] == [("e4b", "fused_attn4_shipped"), ("axolotl", "ckpt_axolotl_best"), ("unsloth", "ckpt_unsloth_best"),
                                    ("e4b", "fused_attn4_shipped_d2"), ("axolotl", "ckpt_axolotl_best_d2"), ("unsloth", "ckpt_unsloth_best_d2"),
                                    ("e4b", "fused_attn4_m")]
    for fw in ("e4b", "axolotl", "unsloth"):
        k = (fw, R.NATIVE[fw])
        assert R.registered_draw2(R.NB_FAM, k) == (fw, R.NATIVE[fw] + "_d2") and R.registered_draw2("qwen3native", k) is None
    NB = R.reduce_family(R.NB_FAM, R._nb_set(e4b_s=(4.306, 4.578), ax_s=(5.128, 5.20), un_s=(7.938, 7.897)), {}, 20)
    assert NB["verdicts"][("e4b", "fused_attn4_m")] == "VALID"          # tc1-5090-33 read it VOID: no registered n_layers for the token
    assert {p: v for p, _, v, _ in R.score_p13({R.NB_FAM: NB})}["P13"] == "UNTESTED"   # e4b shipped's 6.1 % pair, as on tc1-5090-33


def test_amendment_8_registers_the_200_step_native_best_token():
    R = _mod()
    assert R.NB200_FAM == "qwen3nativebest200" and R.NB200_FAM in R.FAMS and R.N_LAYERS[R.NB200_FAM] == 48 and R.ATTN_CENSUS[R.NB200_FAM] == 192
    assert R.EXPECTED[R.NB200_FAM] == [("e4b", "fused_attn4_shipped_200"), ("axolotl", "ckpt_axolotl_best_200"), ("e4b", "fused_attn4_shipped_200_d2"),
                                       ("axolotl", "ckpt_axolotl_best_200_d2"), ("e4b", "fused_attn4_m_200")]
    assert R.anchor_of(R.NB200_FAM) == ("e4b", "fused_attn4_m_200") and R.LATE_FROM == 101 and R.P14_BAND == (0.90, 1.10)
    F = {R.NB200_FAM: R.reduce_family(R.NB200_FAM, R._nb200_set(), {}, None)}
    assert [(p, v) for p, _, v, _ in R.score_p14(F)] == [("P14", "HELD")]
    assert R.score_p14({}) == [] and R.NATIVE["e4b"] == "fused_attn4_shipped"          # the 20-step tokens keep their native arms




def test_amendment_10_registers_the_sync_ab_token():
    R = _mod()
    assert R.SYNC_FAM == "qwen3syncab" and R.SYNC_FAM in R.FAMS and R.N_LAYERS[R.SYNC_FAM] == 48 and R.SYNC_BAND == (0.75, 0.95)
    assert R.anchor_of(R.SYNC_FAM) == ("e4b", "fused_attn4_m_legacy") and "fused_attn4_m_sync1_d2" in R.MATCHED and "fused_attn4_shipped_sync1" not in R.MATCHED
    assert R.registered_draw2(R.SYNC_FAM, ("e4b", "fused_attn4_shipped_sync1")) == ("e4b", "fused_attn4_shipped_sync1_d2")
    F = {R.SYNC_FAM: R.reduce_family(R.SYNC_FAM, R._sync_set(), {}, 20)}
    assert [(p, v) for p, _, v, _ in R.score_syncab(F)] == [("P16", "HELD"), ("P17", "HELD")]
    assert R.sync_ab_why("fused_attn4_m_legacy", {}).startswith("no sync_ab record")


def test_amendment_13_registers_the_lean_delta_token():
    R = _mod()
    assert R.LEAN_FAM == "qwen3leanab" and R.LEAN_FAM in R.FAMS and R.N_LAYERS[R.LEAN_FAM] == 48 and R.LEAN_BAND == (0.90, 0.99)
    assert R.LEAN_REVERT_ABOVE == 1.01 and R.anchor_of(R.LEAN_FAM) == ("e4b", "fused_attn4_m_lean0")
    assert "fused_attn4_m_lean1_d2" in R.MATCHED and "fused_attn4_shipped_lean1" not in R.MATCHED
    assert R.registered_draw2(R.LEAN_FAM, ("e4b", "fused_attn4_shipped_lean1")) == ("e4b", "fused_attn4_shipped_lean1_d2")
    F = {R.LEAN_FAM: R.reduce_family(R.LEAN_FAM, R._lean_set(), {}, 20)}
    assert [(p, v) for p, _, v, _ in R.score_leanab(F)] == [("P20", "HELD"), ("P21", "HELD")]
    assert R.lean_ab_why("fused_attn4_m_lean0", {}).startswith("no lean_ab record")


def test_amendment_14_registers_the_tile_rule_token():
    R = _mod()
    assert R.TILE_FAM == "qwen3tileab" and R.TILE_FAM in R.FAMS and R.N_LAYERS[R.TILE_FAM] == 48
    assert R.TILE_BANDS == {"P22": (0.85, 0.97), "P23": (0.88, 0.98)} and R.TILE_FLIP_AT_OR_BELOW == 0.99 and R.TILE_KEEP_ABOVE == 1.01
    assert R.anchor_of(R.TILE_FAM) == ("e4b", "fused_attn4_m_tilemax") and "fused_attn4_m_tilecost_d2" in R.MATCHED
    assert R.registered_draw2(R.TILE_FAM, ("e4b", "fused_attn4_shipped_tilecost")) == ("e4b", "fused_attn4_shipped_tilecost_d2")
    F = {R.TILE_FAM: R.reduce_family(R.TILE_FAM, R._tile_set(), {}, 20)}
    assert [(p, v) for p, _, v, _ in R.score_tileab(F)] == [("P22", "HELD"), ("P23", "HELD")]
    assert R.tile_ab_why("fused_attn4_m_tilemax", {}).startswith("no tile_ab record")


def test_amendment_12_registers_the_profile_token():
    R = _mod()
    assert R.PROF945_FAM == "qwen3prof945" and R.PROF945_FAM in R.FAMS and R.EXPECTED[R.PROF945_FAM] == list(R.PROF945_ARMS)
    assert R.anchor_of(R.PROF945_FAM) == ("e4b", "fused_attn4_m_prof_legacy") and R.P19_MIN_GAIN == 0.05
    F = {R.PROF945_FAM: R.reduce_family(R.PROF945_FAM, R._prof945_set(), {}, 20)}
    assert [(p, v) for p, _, v, _ in R.score_prof945(F)] == [("P19", "HELD")]

