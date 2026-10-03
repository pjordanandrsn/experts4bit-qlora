#!/usr/bin/env python3
"""tc1_reduce.py -- the per-family support / validity / VERDICT / draws / position / equivalence / frozen-base / prediction
table for lane TC1 in the pre-registered form (TC1-PREREG.md).

Base: bench/tp4/tp4_reduce.py @ e4b main 10ce711d (a committed lane's reducer, never edited): its inputs (<fam>_<fw>_<tag>.json,
summary.txt, versions.txt, box.json), its status vocabulary, its validity predicates, its parity rule and its printer shape are
kept. What TC1 adds, named so the files can be diffed:

  R1 DRAWS: an arm registered with a second draw (`_d2`, a fresh process, same everything) is read as ONE arm with two
     draws: every quoted metric is the median over both draws; STABILITY = |d1 - d2| / mean <= 0.05 for e4b and <= 0.10 for
     every other framework, else the arm is UNSTABLE (reported, never quoted). A missing or non-VALID second draw leaves the
     stability UNMEASURED and the position unquoted (a single draw cannot show the agreement the registration asks for).
  R2 VERDICT per attempt, exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN:
     OK + validity VALID + quality COMPARABLE (held-out |delta| at N <= 0.05 nats against e4b/fused_attn4_m) -> VALID;
     OK + a validity predicate failed -> VOID; OK + VALID + |delta| > 0.05 -> QUALITY_FAIL; OOM -> OOM;
     REFUSED / INSTALL_FAILED / LOAD_FAULT -> UNSUPPORTED; HARNESS_ERROR / ALARM / NOT_RUN as named.
  R3 MATCHED-SET validity (on top of tp4's predicates): a registered matched arm must have run `--lora-init matched:<seed>`
     with the anchor's seed, `matched_init.complete` true, `adapter_dtypes_after` == {fp32}, and a step-0 held-out loss
     within 0.005 nats of e4b/reference_attn4_m (the same base and the same init by construction) -- else VOID.
  R4 EQUIVALENCE for each matched arm against e4b/reference_attn4_m (both VALID by predicates): median per-step
     |delta train loss| <= 0.02 AND |delta held-out at N| <= 0.02 reads EQUIVALENT; both <= 0.05 reads COMPARABLE; else
     DIVERGENT (a finding that stops the position on that pair).
  R5 FROZEN BASE across arms: per registered slot (gate_up, down, q_proj) each arm's sha against e4b/fused_attn4_m's:
     SAME-BYTES / DIFFERENT when the regimes are equal, N-A when they differ or a slot is missing; each arm's own byte-flip
     control shown.
  R6 PREDICTIONS P1-P10 (TC1-PREREG "Predictions") scored HELD / FALSIFIED / UNTESTED, mechanically.
  R7 `--selftest`: hand-built receipts (a complete OK receipt per arm with every field this file reads), >= 16 cases,
     including one receipt per validity predicate that MUST read VOID, one QUALITY_FAIL, one UNSTABLE, one DIVERGENT.
  R8 (phase 2) LABELLED rows: `ckpt_unsloth_t28` (tp4's torch-2.8 venv, the loader's default backend), `ckpt_unsloth_triton`,
     `ckpt_unsloth_best` and `ckpt_axolotl_best` get positions against e4b/fused_attn4_m under their own label, never the
     quoted matched position. Unsloth engagement validity: an arm that REQUESTED a backend must show that backend's counter
     >= L*A per step, and a grouped_mm request must show zero per-expert-loop calls, else VOID with the reason; axolotl with
     quantize_moe_experts must show n_bnb4bit_unwrapped >= L. P1b: the t28 row within 15 % of 29.05 s/step; P8 is now
     "device busy fraction >= 0.5 on the profiled grouped_mm arm".
  R9 (phase 3, harness-phase3.md): A an e4b fused arm is VOID if any step's grouped-nf4-gemm LoRA path was the per-expert loop
     (`lora_path_loop_steps`, named) or the receipt carries no path counters; B a registered-matched arm is VOID unless its
     name-free `matched_init_sha` equals the anchor's (e4b/fused_attn4_m, else reference_attn4_m), and |delta loss at step 2|
     is printed; C the step-0 band: <= 0.01 SAME-BYTES-CLASS, <= 0.05 NEAR (recorded), > 0.05 VOID -- never "same init";
     D C1 is clean only with a named `C1_control_tensor` and a computed control; E a grouped_mm Unsloth arm is VOID unless
     torch._grouped_mm ran >= 6*L*A per step and the manual fallback 0 times; F a recompile between step 10 and N marks the
     draw UNSTABLE; G per-row held-out paired mean +- SE per matched pair, the draw-noise floor, and "inside the draw noise"
     for a reading narrower than it; H the position's interval (min, max) over the four cross-draw ratios, 5 % stability for
     every framework, EQUIVALENT iff both deltas <= max(0.005, 3 x the in-draw fused-vs-reference |delta|) -- COMPARABLE or
     DIVERGENT only when the reference did not run; I equivalence across frameworks is against e4b/fused_attn4_m, the
     fused-vs-reference pair is the e4b-side control, the labelled rows live in the `qwen3native` family token.
  R10 (lane TC1b, the `qwen3curve` family token; TC1B-PREREG.md, drafted in TC1b-PREREG-draft.md): the same model and tokens on one
     box, read per SUB-FIXTURE -- each of {200, p38, t1, r64} has its OWN e4b arm as the trainable / tokens-sha / step-0 / matched-sha
     reference (as tp4's anchor pair had), N per arm (200 / 60 / 20 / 20), the registered trainable counts on arms 1-3 and the anchor
     arms. Readings: (a) the CURVE TABLE -- at every eval step the paired held-out mean +- SE over the 16 rows for the matched pair and
     the as-shipped e4b arm and the paired |delta| (2 vs 1, 3 vs 1); the curve reading = the largest |delta(1,2)| over the evals and its
     sign at step 200, EQUIVALENT-AT-EVERY-EVAL iff every paired |delta| <= 0.02 (the DRAFT's band; the registration may tie it to the
     TC1 floor -- said in the output), else DIVERGENT with the first divergent step; (b) the PLATEAU test, held-out(3) - held-out(1) at
     200 and at 40, REPRODUCES-P38 iff >= +0.01 at 200 and <= 0 at 40; (c) TIME TO TARGET = the matched pair's step-200 held-out + 0.02,
     per arm from its own eval grid (the first eval at or below); (d) s/step medians over steps 11..200 beside TC1's 11..20 medians
     when `--tc1-dir` names a qwen3 receipt dir (within 10 % = TRAVELS), with the in-receipt 11..20 window as a same-draw check;
     (e) the anchor pair's ratio vs tp2 1.457 / P38 1.413 (+-10 %), the t28 variant (tp4's arm byte-for-byte) read separately;
     (f) the t1 and r64 pairs as SCALING POINTS (never a position) under the matched set's predicates, listed. P1-P4 of the draft
     scored HELD / FALSIFIED / UNTESTED; the selftest adds a DIVERGENT curve, a REPRODUCES-P38 plateau and its refutations, a
     failing anchor, a VOID r64 pair, a DOES-NOT-TRAVEL speed row and a target never reached.

  R11 (lane TC2, the `tc2small` / `tc2big` tokens; TC2-PREREG.md, drafted in TC2-PREREG-draft.md): the other families -- granite, olmoe,
     gptoss (box A, N 60, 48 rows every 20) and qwen3_5, mixtral (box B, N 20, 8 rows) -- read per family with TC1's support / validity /
     VERDICT / positions (point + cross-draw interval where two draws exist) / equivalence / frozen-base readings, plus: the registered
     n_layers and structural attention census per family (a receipt that disagrees is VOID); the HF position quoted as "HF (bf16
     experts) / e4b" (or 4-bit) with its regime; an Unsloth arm whose trainable count differs from e4b's is VOID with the reason
     "attention-only" when it adapted no expert parameter, as tp4; on gpt-oss (e4b adapts attention only: `attn_only_m` is the anchor)
     NO ratio is quoted across different adapter sets -- a "no common adapter set" line carries both s/step values and both trainable
     counts, the trainable / sha / quality predicates against e4b are not applied to the other frameworks' arms there; the Unsloth
     regimes beyond bnb stacks: `ckpt_unsloth_mxfp4` (load_in_4bit=False) is VALID only with >= 2L packed expert parameters of a
     recorded class, grouped_mm selected and the MXFP4 grouped GEMM counted >= L*A per step, and gpt-oss's bnb-4bit arm (per-expert
     Linear4bit, no stacks) with >= 2L Params4bit under its experts and either MoE banner; mixtral's FOOTPRINT line (e4b under expert
     offload vs Unsloth resident: peak VRAM and s/step) leads its block; an HF t214 arm whose dispatch did not reach grouped_mm is
     RECORDED (the row's note), never VOID. P1-P7 of the draft scored HELD / FALSIFIED / UNTESTED; the selftest adds a VOID attention-only
     Unsloth arm, the gpt-oss no-common-set line, the mixtral footprint line, a t214 arm that did not reach grouped_mm, the packed-arm
     VOIDs (unpacked, wrong backend, uncounted GEMM), the census / pin VOIDs and every prediction's failing leg.

It licenses nothing and quotes no cross-box number. stdlib only.  Usage: tc1_reduce.py <dir> [--md out.md] [--steps N] | --selftest
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

PREREG = "tc1/TC1-PREREG.md"
READ = 0.05                       # quality: held-out |delta| at N vs e4b/fused_attn4_m <= 0.05 nats reads COMPARABLE, else QUALITY_FAIL
BAND = 0.05                       # tp1's parity rule (B2/C2): |delta final train loss| <= 0.05 AND median step-wise |delta| <= 0.05
EQUIV, COMPARABLE = 0.02, 0.05    # R4: equivalence bands on BOTH the median per-step |delta train| and |delta held-out at N|
STEP0_TOL = 0.005                 # R3: the matched set agrees on step-0 held-out loss to 0.005 nats, by construction
STABILITY = {"e4b": 0.05}         # R1/H: |d1 - d2| / mean <= 5 % for EVERY framework (phase 3 tightened the others from 10 %)
STABILITY_OTHER = 0.05
STEP0_SAME, STEP0_VOID = 0.01, 0.05   # C: SAME-BYTES-CLASS / NEAR / VOID bands on |delta step-0 held-out| vs the anchor
EQUIV_FLOOR = 0.005               # H: EQUIVALENT band = max(EQUIV_FLOOR, 3 x the in-draw fused-vs-reference |delta|)
GMM_FACTOR = 6                    # E: base + 2 LoRA GEMMs x 2 projections per expert-module forward -> torch._grouped_mm calls >= 6*L*A per step
TP4_T28_S_PER_STEP = 29.05        # P1b: the Unsloth s/step the phase-2 instruction quotes for this family on the torch-2.8 image; UNVERIFIED by me against the tp4 receipt
P8_BUSY_MIN = 0.5                 # P8 (phase 2): device busy fraction >= 0.5 on the profiled grouped_mm arm
UNSLOTH_BACKEND_KEYS = {"grouped_mm": "unsloth_grouped_mm", "unsloth_triton": "unsloth_triton", "native_torch": "unsloth_loop"}   # tc1_arm.py's counter keys

AX_FAM = "qwen3axolotl"           # TC1-PREREG amendment 3 (2026-10-02): the axolotl rows re-asked on their own box (two draws of the matched pair)
FAMS = ["qwen3", "qwen3native", AX_FAM]
NAMES = {"qwen3": "Qwen3-30B-A3B", "qwen3native": "Qwen3-30B-A3B (the labelled / native-best box)", AX_FAM: "Qwen3-30B-A3B (amendment 3: the axolotl box)"}
N_LAYERS = {"qwen3": 48, "qwen3native": 48, AX_FAM: 48}
FW = ("e4b", "unsloth", "hf", "axolotl")
QUALITY_ANCHOR = ("e4b", "fused_attn4_m")        # the quality reading's anchor and the matched position's e4b side
EQUIV_ANCHOR = ("e4b", "fused_attn4_m")          # I: equivalence across frameworks is against fused_m; fused-vs-reference is the e4b-side control
REFERENCE = ("e4b", "reference_attn4_m")
PRIMARY = {"e4b": "fused_attn4_m", "unsloth": "ckpt_unsloth_m", "hf": "hf_peft_m", "axolotl": "ckpt_axolotl_m"}
SECONDARY = {fw: PRIMARY[fw] + "_mb1" for fw in PRIMARY}
DRAW2 = {("e4b", "fused_attn4_m"): ("e4b", "fused_attn4_m_d2"), ("unsloth", "ckpt_unsloth_m"): ("unsloth", "ckpt_unsloth_m_d2"),
         ("axolotl", "ckpt_axolotl_m"): ("axolotl", "ckpt_axolotl_m_d2")}      # the axolotl second draw is registered on the amendment-3 box only
NATIVE = {"e4b": "fused_attn4_shipped", "unsloth": "ckpt_unsloth_best", "axolotl": "ckpt_axolotl_best"}
PROF = ("unsloth", "ckpt_unsloth_prof")
# R8: labelled rows -- a position against e4b/fused_attn4_m under this label, never the quoted matched position
LABELLED = {("unsloth", "ckpt_unsloth_t28"): "unsloth t28 (field image: tp4's torch-2.8 venv, loader-default backend)",
            ("unsloth", "ckpt_unsloth_triton"): "unsloth triton backend",
            ("unsloth", "ckpt_unsloth_best"): "unsloth native-best (grouped_mm + speed tilt, native init)",
            ("axolotl", "ckpt_axolotl_best"): "axolotl native-best (KernelsPlugin scattermoe)",
            ("e4b", "fused_attn4_shipped"): "e4b shipped (bf16 expert adapters, N(0,1/r) init)",
            ("e4b", "fused_attn4_m_nodgrad"): "e4b fused dgrad=False (enable_fast_train's default)",
            ("e4b", "fused_attn4_m_t212"): "e4b fused on torch 2.12.1+cu130 (venv-unsloth)",
            ("hf", "hf_peft_m_mb1_t214"): "hf mb1 on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm"}
# I: the judged family (one box) in its order, and the labelled / native-best family token on its own box (with its own e4b fused_m)
EXPECTED = {"qwen3": [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "reference_attn4_m"), ("e4b", "fused_attn4_m_d2"),
                      ("unsloth", "ckpt_unsloth_m_d2"), ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"),
                      ("e4b", "fused_attn4_m_prof"), ("unsloth", "ckpt_unsloth_prof")],
            "qwen3native": [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_best"), ("unsloth", "ckpt_unsloth_t28"), ("unsloth", "ckpt_unsloth_triton"),
                            ("e4b", "fused_attn4_shipped"), ("e4b", "fused_attn4_m_nodgrad"), ("e4b", "fused_attn4_m_t212"),
                            ("axolotl", "ckpt_axolotl_best"), ("hf", "hf_peft_m_mb1_t214")],
            AX_FAM: [("e4b", "fused_attn4_m"), ("axolotl", "ckpt_axolotl_m"), ("e4b", "fused_attn4_m_d2"), ("axolotl", "ckpt_axolotl_m_d2"),
                     ("axolotl", "ckpt_axolotl_best"), ("hf", "hf_peft_m_mb1_t214")]}
# the arms registered MATCHED (fp32 adapters, --lora-init matched:<seed>): R3 applies to these, native rows carry no R3
MATCHED = {"fused_attn4_m", "fused_attn4_m_d2", "ckpt_unsloth_m", "ckpt_unsloth_m_d2", "hf_peft_m", "ckpt_axolotl_m",
           "reference_attn4_m", "ckpt_unsloth_prof", "fused_attn4_m_prof", "ckpt_unsloth_t28", "ckpt_unsloth_triton",
           "fused_attn4_m_nodgrad", "fused_attn4_m_t212", "hf_peft_m_mb1_t214",
           "fused_attn4_m_mb1", "ckpt_unsloth_m_mb1", "hf_peft_m_mb1", "ckpt_axolotl_m_mb1", "ckpt_axolotl_m_d2"}
VOCAB = ("OK", "REFUSED", "OOM", "INSTALL_FAILED", "LOAD_FAULT", "HARNESS_ERROR", "ALARM", "NOT_RUN")
VERDICTS = ("VALID", "VOID", "QUALITY_FAIL", "OOM", "UNSUPPORTED", "HARNESS_ERROR", "ALARM", "NOT_RUN")
STATUS_MAP = {"ok": "OK", "c1_failed": "OK", "refused": "REFUSED", "oom": "OOM", "install_failed": "INSTALL_FAILED",
              "load_fault": "LOAD_FAULT", "verify_failed": "LOAD_FAULT", "alarm": "ALARM", "not_run": "NOT_RUN",
              "phase_alarm": "ALARM",             # e4b#548: the same registered word as the SIGALRM it replaces; the row can name the phase
              "tokens_mismatch": "HARNESS_ERROR", "void_trainable": "HARNESS_ERROR", "void_attn4": "HARNESS_ERROR",
              "harness_error": "HARNESS_ERROR", "unreadable": "HARNESS_ERROR"}
FW_RE = "|".join(FW)

# ----------------------------------------------------------------------------- TC1 amendments 5 and 7: native-best against native-best (the qwen3nativebest token)
NB_FAM = "qwen3nativebest"        # e4b shipped, axolotl native-best, Unsloth native-best, two interleaved draws each, then e4b fused_m as the box's anchor
NB_HALVES = (("axolotl", "axolotl"), ("unsloth", "Unsloth"))     # P13's two halves: <framework> native-best / e4b shipped
FAMS.append(NB_FAM)
NAMES[NB_FAM] = "Qwen3-30B-A3B (amendment 5: native-best against native-best, two draws each)"
N_LAYERS[NB_FAM] = 48
EXPECTED[NB_FAM] = [("e4b", NATIVE["e4b"]), ("axolotl", NATIVE["axolotl"]), ("unsloth", NATIVE["unsloth"]),
                    ("e4b", NATIVE["e4b"] + "_d2"), ("axolotl", NATIVE["axolotl"] + "_d2"), ("unsloth", NATIVE["unsloth"] + "_d2"), ("e4b", "fused_attn4_m")]
DRAW2.update({("e4b", NATIVE["e4b"]): ("e4b", NATIVE["e4b"] + "_d2"), ("axolotl", NATIVE["axolotl"]): ("axolotl", NATIVE["axolotl"] + "_d2"),
              ("unsloth", NATIVE["unsloth"]): ("unsloth", NATIVE["unsloth"] + "_d2")})   # registered (in EXPECTED) on this token only: qwen3native reads one draw

# ----------------------------------------------------------------------------- R10: lane TC1b (the qwen3curve token)
CURVE_FAM = "qwen3curve"
CURVE_BAND = 0.02         # (a): paired held-out |delta| <= 0.02 at every eval -- the DRAFT's band (TC1b-PREREG-draft "Readings"); the registration may tie it to the TC1 floor
CURVE_BAND_NOTE = "the draft's band, kept fixed by the registration (0.02 nats at every eval: this lane has no in-draw reference arm); TC1's floor is quoted beside it"
PLATEAU_GAP, PLATEAU_EARLY_STEP = 0.01, 40   # (b): as-shipped minus matched held-out >= +0.01 at step 200 AND <= 0 at step 40 reproduces P38's shape
TARGET_MARGIN = 0.02      # (c): time to the matched pair's step-200 held-out + 0.02, read from each arm's eval grid
TRAVEL_TOL = 0.10         # (d) / P3: TC1's 11..20 median and TC1b's 11..200 median within 10 % per arm = TRAVELS
ANCHOR_TOL = 0.10         # (e) / P4: the p38 pair's ratio within +-10 % of tp2's 1.457 / P38's 1.413 (TP4-PREREG "Cross-lane anchor")
TP2_ANCHOR, P38_ANCHOR = 1.457, 1.413   # tp2 (2026-09-06) and P38 (2026-09-05), RTX 5090, the same fixture (bench/tp4/tp4_reduce.py, TP4-PREREG)
TP4_ANCHOR_RATIO = None   # UNVERIFIED: tp4's own anchor row is NOT in this tree (bench/p67/reread-existing/reread.json names its qwen3_e4b_fused_attn4_p38.json
#                           receipts by sha only; no RESULTS file carries the pair) -- the draft's P4 names it. Set it when the PI supplies the number;
#                           until then P4 reads against tp2's 1.457 (the number tp4 itself registered against) and SAYS so in its evidence.
CURVE_N = {"fused_attn4_m_200": 200, "ckpt_unsloth_m_200": 200, "fused_attn4_shipped_200": 200,
           "fused_attn4_p38": 60, "ckpt_unsloth_p38": 60, "ckpt_unsloth_p38_t28": 60,
           "fused_attn4_m_t1": 20, "ckpt_unsloth_m_t1": 20, "fused_attn4_m_r64": 20, "ckpt_unsloth_m_r64": 20}
CURVE_GROUP = {"fused_attn4_m_200": "200", "ckpt_unsloth_m_200": "200", "fused_attn4_shipped_200": "200",
               "fused_attn4_p38": "p38", "ckpt_unsloth_p38": "p38", "ckpt_unsloth_p38_t28": "p38",
               "fused_attn4_m_t1": "t1", "ckpt_unsloth_m_t1": "t1", "fused_attn4_m_r64": "r64", "ckpt_unsloth_m_r64": "r64"}
CURVE_GROUPS = {"200": ("e4b", "fused_attn4_m_200"), "p38": ("e4b", "fused_attn4_p38"), "t1": ("e4b", "fused_attn4_m_t1"), "r64": ("e4b", "fused_attn4_m_r64")}   # each sub-fixture's own e4b arm
CURVE_TRAINABLE = {"200": 642514944, "p38": 321257472}   # TC1b-PREREG-draft "Validity": arms 1-3 (the field recipe, r 16) and the anchor arms (r 8); t1 inherits the field count, r64 is asserted equal BETWEEN its two arms only
CURVE_ARMS = [("e4b", "fused_attn4_m_200"), ("unsloth", "ckpt_unsloth_m_200"), ("e4b", "fused_attn4_shipped_200")]   # arms 1, 2, 3 of the registration
CURVE_ANCHOR = {"e4b": ("e4b", "fused_attn4_p38"), "unsloth": ("unsloth", "ckpt_unsloth_p38"), "t28": ("unsloth", "ckpt_unsloth_p38_t28")}
CURVE_SCALING = {"t1": "tokens-per-step scaling point (seq 2048, micro-batch 1 x accum 1, N 20)", "r64": "rank scaling point (r 64 / alpha 64, N 20)"}
CURVE_TC1_COUNTERPART = {("e4b", "fused_attn4_m_200"): ("qwen3", ("e4b", "fused_attn4_m")), ("unsloth", "ckpt_unsloth_m_200"): ("qwen3", ("unsloth", "ckpt_unsloth_m")),
                         ("e4b", "fused_attn4_shipped_200"): ("qwen3native", ("e4b", "fused_attn4_shipped"))}   # (d): TC1's 11..20 median for each curve arm
EXPECTED[CURVE_FAM] = [("e4b", "fused_attn4_m_200"), ("unsloth", "ckpt_unsloth_m_200"), ("e4b", "fused_attn4_shipped_200"),
                       ("e4b", "fused_attn4_p38"), ("unsloth", "ckpt_unsloth_p38"), ("unsloth", "ckpt_unsloth_p38_t28"),
                       ("e4b", "fused_attn4_m_t1"), ("unsloth", "ckpt_unsloth_m_t1"), ("e4b", "fused_attn4_m_r64"), ("unsloth", "ckpt_unsloth_m_r64")]
MATCHED |= {"fused_attn4_m_200", "ckpt_unsloth_m_200", "fused_attn4_m_t1", "ckpt_unsloth_m_t1", "fused_attn4_m_r64", "ckpt_unsloth_m_r64"}   # the shipped and p38 arms are native rows
FAMS.append(CURVE_FAM)
NAMES[CURVE_FAM] = "Qwen3-30B-A3B (lane TC1b: the 200-step curve box, the anchor pair, the t1 and r64 scaling pairs)"
N_LAYERS[CURVE_FAM] = 48

# ----------------------------------------------------------------------------- R11: lane TC2 (the tc2small / tc2big tokens; TC2-PREREG-draft, registered by the PI)
TC2_FAMS = ["granite", "olmoe", "gptoss", "qwen3_5", "mixtral"]
TC2_TOKENS = {"tc2small": ["granite", "olmoe", "gptoss"], "tc2big": ["qwen3_5", "mixtral"]}
# (model id, revision, n_layers): bench/tp4/tp4_run.sh's family table / TC2-PREREG-draft; a receipt whose model or revision differs from the pin is VOID
TC2_MODELS = {"granite": ("ibm-granite/granite-3.1-3b-a800m-instruct", "a02780686e08a03fe0d2679a293b5c74a90efa89", 32),
              "olmoe": ("allenai/OLMoE-1B-7B-0924-Instruct", "7f1c97f440f06ce36705e4f2b843edb5925f4498", 16),
              "gptoss": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", 24),
              "qwen3_5": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0", 40),
              "mixtral": ("mistralai/Mixtral-8x7B-Instruct-v0.1", "eba92302a2861cdc0098cc54bc9f17cb2c47eb61", 32)}
NAMES.update({"granite": "Granite-3.1-3B-A800M-instruct (lane TC2, box A)", "olmoe": "OLMoE-1B-7B-0924-Instruct (lane TC2, box A)",
              "gptoss": "gpt-oss-20b (lane TC2, box A; e4b attention-only -- no common adapter set)",
              "qwen3_5": "Qwen3.6-35B-A3B (lane TC2, box B)", "mixtral": "Mixtral-8x7B-Instruct-v0.1 (lane TC2, box B; e4b under expert offload, Unsloth resident)"})
N_LAYERS.update({fam: v[2] for fam, v in TC2_MODELS.items()})
# the structural attention census per family (tc1_arm.py's T10 docstring, #434: "granite 128, olmoe 64, qwen3 192, mixtral 128 (exactly 4 x n_layers)";
# gpt-oss is REFUSED on the bias rule -- its attention is never converted; Gemma-4 is out of scope). None = not registered here: the receipt's own
# structural_expected_n_attn4 governs (qwen3_5: linear-attention layers, no committed census in this tree -- UNVERIFIED). A receipt whose census
# disagrees with a registered value is VOID (R11).
ATTN_CENSUS = {"qwen3": 192, "qwen3native": 192, AX_FAM: 192, NB_FAM: 192, CURVE_FAM: 192, "granite": 128, "olmoe": 64, "mixtral": 128, "gptoss": None, "qwen3_5": None}
TC2_ANCHOR = {"gptoss": ("e4b", "attn_only_m")}      # the family's e4b anchor arm (quality, draws, step-0, sha): attention-only on gpt-oss (bare experts, tp1/tp2)
NO_COMMON_SET = {"gptoss": "e4b adapts attention only on this family (experts built bare, no ExpertsLoRA: tp1/tp2 cited) while Unsloth / HF / axolotl adapt the experts -- "
                           "no ratio is quoted across different adapter sets (TC2-PREREG-draft 'Arms per family')"}
FOOTPRINT_FAMS = {"mixtral": "e4b under expert offload (--offload 1, tp2 / tp4's arm) vs Unsloth resident"}
TC2_LABELLED = {("unsloth", "ckpt_unsloth_m_experts"): "unsloth with the family's own expert names as targets (tp4 amendment 4's second arm)",
                ("hf", "hf_peft_m_t214"): "hf on torch 2.14 (venv-axolotl) with experts_implementation=grouped_mm (what dispatched is recorded)",
                ("unsloth", "ckpt_unsloth_mxfp4"): "unsloth 16-bit load (load_in_4bit=False): the MXFP4 experts kept packed, grouped_mm"}
ALL_LABELLED = {**LABELLED, **TC2_LABELLED}            # LABELLED stays TC1's registered set; reduce_family iterates the union

# ----------------------------------------------------------------------------- TC1 amendment 8: e4b shipped vs axolotl scattermoe over 200 steps (the qwen3nativebest200 token)
NB200_FAM = "qwen3nativebest200"  # tc1-5090-34: the scattermoe arm's per-process warm-up spikes fall inside 11..20, so P13's axolotl half is re-asked on a late window
LATE_FROM = 101                   # P14's window: the median s/step over steps 101..200 (each draw's own step_ms), its stability per side within 5 %
P14_BAND = (0.90, 1.10)           # P14: axolotl scattermoe native-best / e4b shipped over the late window, within 10 % of parity
FAMS.append(NB200_FAM)
NAMES[NB200_FAM] = "Qwen3-30B-A3B (amendment 8: e4b shipped vs axolotl scattermoe over 200 steps, two draws each)"
N_LAYERS[NB200_FAM] = 48
ATTN_CENSUS[NB200_FAM] = 192
FAM_ANCHOR = {NB200_FAM: ("e4b", "fused_attn4_m_200")}   # the box's matched anchor runs the same 200 steps (validity, trainable count, quality)
NATIVE_BY_FAM = {NB200_FAM: {"e4b": "fused_attn4_shipped_200", "axolotl": "ckpt_axolotl_best_200"}}
EXPECTED[NB200_FAM] = [("e4b", "fused_attn4_shipped_200"), ("axolotl", "ckpt_axolotl_best_200"), ("e4b", "fused_attn4_shipped_200_d2"),
                       ("axolotl", "ckpt_axolotl_best_200_d2"), ("e4b", "fused_attn4_m_200")]
DRAW2.update({("e4b", "fused_attn4_shipped_200"): ("e4b", "fused_attn4_shipped_200_d2"), ("axolotl", "ckpt_axolotl_best_200"): ("axolotl", "ckpt_axolotl_best_200_d2")})

# ----------------------------------------------------------------------------- TC1 amendment 10 (#945): e4b's grouping + index-transfer syncs, A/B on one card
SYNC_FAM = "qwen3syncab"          # legacy grouping + pageable copies (13 host syncs per MoE layer pass) vs single-read grouping + pinned ring (1)
SYNC_PAIRS = (("P16", "shipped", "fused_attn4_shipped"), ("P17", "matched", "fused_attn4_m"))   # each: <tag>_legacy vs <tag>_sync1, two draws a side
SYNC_BAND = (0.75, 0.95)          # P16 / P17: sync1 / legacy s/step on stable pairs -- at least 5 % faster, at most 25 %
FAMS.append(SYNC_FAM)
NAMES[SYNC_FAM] = "Qwen3-30B-A3B (amendment 10: e4b legacy grouping + pageable copies vs single-read grouping + pinned ring, #945)"
N_LAYERS[SYNC_FAM] = 48
ATTN_CENSUS[SYNC_FAM] = 192
FAM_ANCHOR[SYNC_FAM] = ("e4b", "fused_attn4_m_legacy")
EXPECTED[SYNC_FAM] = [("e4b", "fused_attn4_shipped_legacy"), ("e4b", "fused_attn4_shipped_sync1"), ("e4b", "fused_attn4_m_legacy"), ("e4b", "fused_attn4_m_sync1"),
                      ("e4b", "fused_attn4_m_sync1_d2"), ("e4b", "fused_attn4_m_legacy_d2"), ("e4b", "fused_attn4_shipped_sync1_d2"), ("e4b", "fused_attn4_shipped_legacy_d2")]
MATCHED |= {"fused_attn4_m_legacy", "fused_attn4_m_sync1", "fused_attn4_m_legacy_d2", "fused_attn4_m_sync1_d2"}
for _p, _k, _t in SYNC_PAIRS:
    for _side in ("legacy", "sync1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")


# ----------------------------------------------------------------------------- TC1 amendment 12 (#945): the profile after the syncs are gone
PROF945_FAM = "qwen3prof945"      # the TC1 profile instrument on e4b shipped + matched (new path) and matched (legacy path) on one card
PROF945_ARMS = (("e4b", "fused_attn4_shipped_prof"), ("e4b", "fused_attn4_m_prof"), ("e4b", "fused_attn4_m_prof_legacy"))
P19_MIN_GAIN = 0.05               # P19: matched new-path device busy fraction >= the legacy arm's + 0.05
FAMS.append(PROF945_FAM)
NAMES[PROF945_FAM] = "Qwen3-30B-A3B (amendment 12: where e4b's step goes after #945, profiled; the legacy path as the before-picture)"
N_LAYERS[PROF945_FAM] = 48
ATTN_CENSUS[PROF945_FAM] = 192
FAM_ANCHOR[PROF945_FAM] = ("e4b", "fused_attn4_m_prof_legacy")
EXPECTED[PROF945_FAM] = list(PROF945_ARMS)
MATCHED |= {"fused_attn4_m_prof_legacy"}


# ----------------------------------------------------------------------------- TC1 amendment 13 (#945): gnf4's padded LoRA delta, previous body vs trimmed
LEAN_FAM = "qwen3leanab"          # NF4_QLORA_LEAN_DELTA=0 (gnf4's previous padded delta body) vs =1 (gnf4#440's trimmed body), both on the post-#945 sync path
LEAN_PAIRS = (("P20", "shipped", "fused_attn4_shipped"), ("P21", "matched", "fused_attn4_m"))   # each: <tag>_lean0 vs <tag>_lean1, two draws a side
LEAN_BAND = (0.90, 0.99)          # P20 / P21: lean1 / lean0 s/step on stable pairs -- at least 1 % faster, at most 10 %
LEAN_REVERT_ABOVE = 1.01          # amendment 13's decision rule: a stable ratio above this on either arm reverts gnf4's default
FAMS.append(LEAN_FAM)
NAMES[LEAN_FAM] = "Qwen3-30B-A3B (amendment 13: gnf4's previous padded LoRA delta vs its trimmed body, both on the post-#945 sync path)"
N_LAYERS[LEAN_FAM] = 48
ATTN_CENSUS[LEAN_FAM] = 192
FAM_ANCHOR[LEAN_FAM] = ("e4b", "fused_attn4_m_lean0")
EXPECTED[LEAN_FAM] = [("e4b", "fused_attn4_shipped_lean0"), ("e4b", "fused_attn4_shipped_lean1"), ("e4b", "fused_attn4_m_lean0"), ("e4b", "fused_attn4_m_lean1"),
                      ("e4b", "fused_attn4_m_lean1_d2"), ("e4b", "fused_attn4_m_lean0_d2"), ("e4b", "fused_attn4_shipped_lean1_d2"), ("e4b", "fused_attn4_shipped_lean0_d2")]
MATCHED |= {"fused_attn4_m_lean0", "fused_attn4_m_lean1", "fused_attn4_m_lean0_d2", "fused_attn4_m_lean1_d2"}
for _p, _k, _t in LEAN_PAIRS:
    for _side in ("lean0", "lean1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")


# ----------------------------------------------------------------------------- TC1 amendment 14 (#945): gnf4's prefill M-tile rule, max vs cost
TILE_FAM = "qwen3tileab"          # GNF4_PREFILL_TILE_RULE=max (the M-tile height keyed on the largest group) vs =cost (gnf4#441), lean delta + post-#945 sync path
TILE_PAIRS = (("P22", "shipped", "fused_attn4_shipped"), ("P23", "matched", "fused_attn4_m"))   # each: <tag>_tilemax vs <tag>_tilecost, two draws a side
TILE_BANDS = {"P22": (0.85, 0.97), "P23": (0.88, 0.98)}   # tilecost / tilemax s/step on stable pairs
TILE_FLIP_AT_OR_BELOW = 0.99      # amendment 14's decision rule: both stable ratios at or below this flip gnf4's default to cost
TILE_KEEP_ABOVE = 1.01            # ... a stable ratio above this on either arm keeps max
FAMS.append(TILE_FAM)
NAMES[TILE_FAM] = "Qwen3-30B-A3B (amendment 14: gnf4's max-keyed prefill M-tile vs the cost rule, on the trimmed delta and the post-#945 sync path)"
N_LAYERS[TILE_FAM] = 48
ATTN_CENSUS[TILE_FAM] = 192
FAM_ANCHOR[TILE_FAM] = ("e4b", "fused_attn4_m_tilemax")
EXPECTED[TILE_FAM] = [("e4b", "fused_attn4_shipped_tilemax"), ("e4b", "fused_attn4_shipped_tilecost"), ("e4b", "fused_attn4_m_tilemax"), ("e4b", "fused_attn4_m_tilecost"),
                      ("e4b", "fused_attn4_m_tilecost_d2"), ("e4b", "fused_attn4_m_tilemax_d2"), ("e4b", "fused_attn4_shipped_tilecost_d2"), ("e4b", "fused_attn4_shipped_tilemax_d2")]
MATCHED |= {"fused_attn4_m_tilemax", "fused_attn4_m_tilecost", "fused_attn4_m_tilemax_d2", "fused_attn4_m_tilecost_d2"}
for _p, _k, _t in TILE_PAIRS:
    for _side in ("tilemax", "tilecost"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")


def tile_ab_why(tag, r):
    """Amendment 14's engagement predicate: the arm ran the tile rule its tag names, by its own record, on the trimmed LoRA delta; a cost
    arm launched at least one tile shorter than 128 (else the rule changed nothing on this box). Empty string = engaged."""
    ta, la = (r or {}).get("tile_ab"), (r or {}).get("lean_ab") or {}
    if not isinstance(ta, dict):
        return "no tile_ab record on the receipt: the tile rule this arm ran cannot be verified"
    want = "max" if "_tilemax" in tag else "cost"
    bm = ta.get("prefill_bm_launches") or {}
    short = sum(int(v) for k, v in bm.items() if str(k) != "128")
    bad = [k for k, ok in ((f"gnf4_tile_rule {want}", ta.get("gnf4_tile_rule") == want), ("gnf4_has_tile_rule", ta.get("gnf4_has_tile_rule") is True),
                           ("gnf4_lean_delta 1", la.get("gnf4_lean_delta") == "1"),
                           ("a tile shorter than 128 launched", want == "max" or short > 0)) if not ok]
    return "" if not bad else f"tile-rule A/B not engaged ({', '.join(bad)}; record {ta})"


def lean_ab_why(tag, r):
    """Amendment 13's engagement predicate: the arm ran the delta body its tag names, by its own record, and the padded path -- the only
    one the switch touches -- served the delta. Empty string = engaged."""
    la = (r or {}).get("lean_ab")
    if not isinstance(la, dict):
        return "no lean_ab record on the receipt: the delta body this arm ran cannot be verified"
    want = "0" if "_lean0" in tag else "1"
    calls = la.get("lora_path_calls") or {}
    bad = [k for k, ok in ((f"gnf4_lean_delta {want}", la.get("gnf4_lean_delta") == want), ("gnf4_has_lean_delta", la.get("gnf4_has_lean_delta") is True),
                           ("padded calls > 0", (calls.get("padded") or 0) > 0)) if not ok]
    return "" if not bad else f"lean-delta A/B body not engaged ({', '.join(bad)}; record {la})"


def sync_ab_why(tag, r):
    """Amendment 10's engagement predicate: the arm ran the path its tag names, by its own record. Empty string = engaged."""
    sa = (r or {}).get("sync_ab")
    if not isinstance(sa, dict):
        return "no sync_ab record on the receipt: the path this arm ran cannot be verified"
    if "_legacy" in tag:
        bad = [k for k, ok in (("e4b_grouping legacy", sa.get("e4b_grouping") == "legacy"), ("gnf4_pinned_ring 0", sa.get("gnf4_pinned_ring") == "0"),
                               ("ring_staged 0", sa.get("ring_staged") == 0)) if not ok]
    else:
        bad = [k for k, ok in (("e4b_grouping single", sa.get("e4b_grouping") in ("single", "default")), ("gnf4_pinned_ring 1", sa.get("gnf4_pinned_ring") == "1"),
                               ("ring_staged > 0", (sa.get("ring_staged") or 0) > 0), ("e4b_has_group_by_expert", sa.get("e4b_has_group_by_expert") is True),
                               ("gnf4_has_ring", sa.get("gnf4_has_ring") is True)) if not ok]
    return "" if not bad else f"sync A/B path not engaged ({', '.join(bad)}; record {sa})"
EXPECTED["granite"] = [("e4b", "fused_attn4_m"), ("hf", "hf_peft_m"), ("e4b", "reference_attn4_m"), ("e4b", "fused_attn4_m_d2"), ("hf", "hf_peft_m_d2"),
                       ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_experts"), ("hf", "hf_peft_m_t214"),
                       ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_best"), ("e4b", "fused_attn4_shipped")]
EXPECTED["olmoe"] = [k for k in EXPECTED["granite"] if k != ("unsloth", "ckpt_unsloth_m_experts")]
EXPECTED["gptoss"] = [("e4b", "fused_attn4_m"), ("e4b", "attn_only_m"), ("e4b", "attn_only_m_d2"), ("unsloth", "ckpt_unsloth_m"),
                      ("unsloth", "ckpt_unsloth_mxfp4"), ("unsloth", "ckpt_unsloth_mxfp4_d2"), ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"),
                      ("e4b", "reference_attn4_m")]
EXPECTED["qwen3_5"] = [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "fused_attn4_m_d2"), ("unsloth", "ckpt_unsloth_m_d2"),
                       ("unsloth", "ckpt_unsloth_m_experts"), ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_best"),
                       ("e4b", "fused_attn4_shipped"), ("e4b", "reference_attn4_m")]
EXPECTED["mixtral"] = [k for k in EXPECTED["qwen3_5"] if k != ("unsloth", "ckpt_unsloth_m_experts")]
MATCHED |= {"attn_only_m", "attn_only_m_d2", "ckpt_unsloth_m_experts", "ckpt_unsloth_mxfp4", "ckpt_unsloth_mxfp4_d2", "hf_peft_m_d2", "hf_peft_m_t214"}
DRAW2.update({("e4b", "attn_only_m"): ("e4b", "attn_only_m_d2"), ("hf", "hf_peft_m"): ("hf", "hf_peft_m_d2"),
              ("unsloth", "ckpt_unsloth_mxfp4"): ("unsloth", "ckpt_unsloth_mxfp4_d2")})   # registered per family through EXPECTED (registered_draw2)
FAMS += TC2_FAMS
UNSLOTH_BANNER = "Enabling LoRA on MoE parameters"
UNSLOTH_BANNER_PER_EXPERT = "Detected MoE model with per-expert Linear experts"   # unsloth llama.py:3888-3896 (UPSTREAM-NOTES): gpt-oss's bnb-4bit class, per-expert Linear4bit
# the draft's predictions (TC2-PREREG-draft "Predictions") and the lineage numbers they read against
TC2_P1_HF_BAND = (1.1, 1.6)            # granite HF/e4b (tp4: 1.292, RESULTS-tp4-p46cut.md)
TC2_P2_UNS_BAND = (1.2, 3.0)           # olmoe Unsloth/e4b if its arm engages
TC2_P2_HF_BAND = (1.5, 2.5)            # olmoe HF/e4b (tp4: 1.945, quality FLAGGED 0.0550 there)
TC2_P4_UNS_BAND = (2.0, 6.0)           # qwen3_5 Unsloth (explicit expert targets)/e4b if it engages the routed experts
TP4_QWEN3_5_E4B_S_PER_STEP = 6.3344    # bench/tp4/RESULTS-tp4-p46cut.md (e4b fused_attn4 on Qwen3.6-35B-A3B, N 20; the draft rounds it to 6.33)
TC2_P4_E4B_TOL = 0.15                  # P4's e4b leg: fused_m within 15 % of that
TC2_P5_BAND = (0.3, 0.5)               # mixtral: the pair's s/step ratio in the lane's other/e4b convention, Unsloth(resident)/e4b(offload) -- tp2's 0.361 = 0.858 / 2.377 s
TC2_P5_PEAK_X = 8.0                    # ... at a >= 8x lower e4b peak
TP2_MIXTRAL = {"ratio_unsloth_over_e4b": 0.361, "peak_unsloth_gb": 29.16, "peak_e4b_gb": 3.22}   # bench/h2h-20260906/tp2/RESULTS-tp2.md:84 (the draft's 3.2 / 29.2 GB)


def anchor_of(fam):
    """R11: the family's e4b anchor arm -- fused_attn4_m everywhere but gpt-oss, where e4b trains attention only (attn_only_m)."""
    return TC2_ANCHOR.get(fam) or FAM_ANCHOR.get(fam, QUALITY_ANCHOR)


def registered_draw2(fam, key):
    """R11: the second-draw key of `key` when the family REGISTERS it (in EXPECTED), else None -- DRAW2 is global, the registration per family."""
    k2 = DRAW2.get(key)
    return k2 if (k2 is not None and k2 in EXPECTED.get(fam, EXPECTED["qwen3"])) else None


def unsloth_regime(r):
    """R11: which Unsloth expert regime a receipt describes -- `packed` (load_in_4bit=False: the checkpoint's packed expert format kept,
    gpt-oss MXFP4), `per-expert-linear4bit` (no Params4bit stacks but Params4bit under the experts container: gpt-oss's bnb-4bit
    GptOssExpertsBnb4bit), else `bnb-stacks` (TC1's regime: Params4bit on the fused 3-D stacks)."""
    c = r.get("census") or {}
    if r.get("unsloth_load_in_4bit") is False:
        return "packed"
    if (c.get("Params4bit_expert_stacks") or 0) == 0 and (c.get("Params4bit_expert_linears") or 0) > 0:
        return "per-expert-linear4bit"
    return "bnb-stacks"


def packed_expert_params(r):
    """R11: (count of expert parameters in a class other than a plain Parameter, {class: count}) from the census -- the packed stacks."""
    epc = (r.get("census") or {}).get("expert_param_classes") or {}
    return sum(v for k, v in epc.items() if k != "Parameter"), epc


def hf_dispatch_note(r):
    """R11: what the HF arm's experts implementation DISPATCHED (recorded, never a validity predicate): config._experts_implementation and
    the torch grouped_mm counters per step; None when the receipt carries no dispatch record (TC1's HF rows)."""
    d = r.get("hf_experts_dispatch")
    if not isinstance(d, dict):
        return None
    reached = d.get("reached_grouped_mm")
    return (f"experts_implementation requested {d.get('requested')!r}, accepted {d.get('accepted')!r}, config {d.get('config')!r}; torch grouped_mm calls/step min "
            f"{d.get('torch_grouped_mm_calls_per_step_min')} (F.grouped_mm {d.get('torch_F_grouped_mm_calls_per_step_min')}): dispatch "
            + ("REACHED grouped_mm" if reached else "did NOT reach grouped_mm (recorded, not VOID)"))


def hf_label(fam, r):
    """R11: the HF position's label -- "HF (bf16 experts)" or "HF (4-bit experts)" from the arm's own regime."""
    reg = regime_of(fam, r) or ""
    return "HF (bf16 experts)" if "bf16 experts" in reg else ("HF (4-bit experts)" if "4-bit expert" in reg else "HF")
# ----------------------------------------------------------------------------- R11: lane TC3 (the memory frontier; TC3-PREREG-draft, registered as bench/tc1/TC3-PREREG.md by the PI)
# Two family tokens, each ONE box: `qwen3frontier` (a 24 GB RTX 4090, every framework with its own memory lever) and `qwen3frontier12` (the owned 12 GB RTX A2000).
# The box's anchor is the e4b OFFLOAD arm (the resident e4b arm is expected to OOM there); the in-box control is the reference path under offload. Readings:
# (a) the FIT TABLE per framework; (b) the in-box equivalence fused_offload vs reference_offload under TC1's R4 bands, and, with --tc1-dir, the matched
# trajectory against TC1's RESIDENT e4b/fused_attn4_m (same tokens / init / precision asserted first; median per-step |delta| <= RESIDENT_BAND reads
# EQUIVALENT-TO-RESIDENT); (c) no cross-box ratio anywhere (P1's ratios are within the 24 GB box; P4 is two measurements); (d) P1-P4 of the draft scored.
FRONTIER_FAM, FRONTIER12_FAM = "qwen3frontier", "qwen3frontier12"
FRONTIER_FAMS = (FRONTIER_FAM, FRONTIER12_FAM)
FRONTIER_ANCHOR = ("e4b", "fused_attn4_m_offload")      # the quality / equivalence anchor on a frontier box
FRONTIER_REF = ("e4b", "reference_attn4_m_offload")     # the in-box control (parity under offload)
RESIDENT_BAND = 0.02                                    # (b): median per-step |delta| vs TC1's resident e4b/fused_attn4_m <= 0.02 reads EQUIVALENT-TO-RESIDENT (the draft's band)
RESIDENT_KEY = ("qwen3", ("e4b", "fused_attn4_m"))      # TC1's resident arm (same tokens, init, precision), read from --tc1-dir
P1_HF_SLOWER, P1_Z3_SLOWER, P1_Z3_HOST_GB = 20.0, 5.0, 60.0   # P1's within-box clauses (the draft): HF offload REFUSED or > 20x e4b offload; ZeRO-3, if it runs, > 5x at >= 60 GB host RAM
P4_FACTOR = 2.0                                          # P4: e4b offload s/step on the 24 GB card within 2x of TC1's resident e4b -- two measurements, never a ratio
FRONTIER_LEVER = {"fused_attn4_m": "resident", "fused_attn4_m_offload": "e4b expert offload (--offload 1)", "fused_attn4_m_offload_d2": "e4b expert offload (--offload 1), draw 2",
                  "fused_attn4_m_mb1": "resident, micro-batch 1 x accum 8", "fused_attn4_shipped": "resident, as shipped (bf16 expert adapters, N(0,1/r) init): a fit row, never a position",
                  "ckpt_unsloth_m": "resident (Unsloth's own: use_gradient_checkpointing=unsloth)",
                  "ckpt_unsloth_m_mb1": "micro-batch 1 x accum 8", "hf_peft_m": "resident", "hf_peft_m_offload": "accelerate device_map=auto + max_memory (--hf-offload 1)",
                  "hf_peft_m_mb1": "micro-batch 1 x accum 8", "ckpt_axolotl_m": "resident (quantize_moe_experts)", "ckpt_axolotl_m_layeroffload": "layer_offloading (--axolotl-layer-offload 1)",
                  "ckpt_axolotl_m_zero3": "DeepSpeed ZeRO-3 parameter offload, bf16 experts (--axolotl-zero3 1)", "reference_attn4_m_offload": "e4b expert offload, the reference path",
                  "fused_attn4_m_offload_mb1": "e4b expert offload, micro-batch 1 x accum 8 (the 12 GB secondary)", "reference_attn4_m_offload_mb1": "e4b expert offload, the reference path, micro-batch 1 x accum 8",
                  "fused_attn4_shipped_offload": "e4b expert offload, as shipped (bf16 expert adapters, N(0,1/r) init): a fit row, never a position"}
FRONTIER12_SECONDARY = (("e4b", "fused_attn4_m_offload_mb1"), ("e4b", "reference_attn4_m_offload_mb1"))   # run only when the field-recipe offload arm OOMed (one draw each)
EXPECTED[FRONTIER_FAM] = [("e4b", "fused_attn4_m"), ("e4b", "fused_attn4_m_offload"), ("e4b", "fused_attn4_m_mb1"), ("e4b", "fused_attn4_shipped"), ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_mb1"),
                          ("hf", "hf_peft_m"), ("hf", "hf_peft_m_offload"), ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_m_layeroffload"),
                          ("axolotl", "ckpt_axolotl_m_zero3"), ("e4b", "reference_attn4_m_offload")]
EXPECTED[FRONTIER12_FAM] = [("e4b", "fused_attn4_m_offload"), ("e4b", "fused_attn4_m_offload_d2"), ("e4b", "reference_attn4_m_offload"), ("e4b", "fused_attn4_m"),
                            ("unsloth", "ckpt_unsloth_m_mb1"), ("hf", "hf_peft_m_mb1"), ("axolotl", "ckpt_axolotl_m")]
MATCHED |= {"fused_attn4_m_offload", "fused_attn4_m_offload_d2", "reference_attn4_m_offload", "hf_peft_m_offload", "ckpt_axolotl_m_layeroffload", "ckpt_axolotl_m_zero3",
            "fused_attn4_m_offload_mb1", "reference_attn4_m_offload_mb1"}
DRAW2[("e4b", "fused_attn4_m_offload")] = ("e4b", "fused_attn4_m_offload_d2")   # registered on the 12 GB token only (draws_of reads the family's `expected`)
FAMS += list(FRONTIER_FAMS)
NAMES[FRONTIER_FAM] = "Qwen3-30B-A3B (lane TC3: the 24 GB RTX 4090 memory frontier, every framework with its own lever)"
NAMES[FRONTIER12_FAM] = "Qwen3-30B-A3B (lane TC3: the owned 12 GB RTX A2000, e4b's offload territory)"
N_LAYERS[FRONTIER_FAM] = N_LAYERS[FRONTIER12_FAM] = 48


# ----------------------------------------------------------------------------- inputs
def load(d):
    """{fam: {(fw, tag): receipt}} from <fam>_<fw>_<tag>.json (tokens_*.json, box.json and other files are skipped)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_*_*.json"))):
        base = os.path.basename(p)[:-5]
        if base.endswith("_profile"):         # the profiled arm's kernel-table sidecar (<arm>_prof_profile.json), not a receipt (amendment 3)
            continue
        m = re.match(rf"^([a-z0-9_]+?)_({FW_RE})_(.+)$", base)
        if not m:
            continue
        fam, fw, tag = m.groups()
        try:
            r = json.load(open(p))
        except Exception as e:
            r = {"framework": fw, "fam": fam, "tag": tag, "status": "unreadable", "reason": str(e)}
        out.setdefault(fam, {})[(fw, tag)] = r
    return out


def load_summary(d):
    """{(fam, fw, tag): [rc, ...]} from summary.txt's `fam/fw/tag rc=N` lines (every attempt)."""
    out = {}
    p = os.path.join(d, "summary.txt")
    if not os.path.exists(p):
        return out
    for ln in open(p, errors="replace"):
        m = re.match(rf"^([a-z0-9_]+)/({FW_RE})/(\S+) rc=(\d+)", ln)
        if m:
            out.setdefault((m.group(1), m.group(2), m.group(3)), []).append(int(m.group(4)))
    return out


def f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ratio(num, den):
    try:
        return num / den if den else None
    except Exception:
        return None


def is_ok(r):
    return r is not None and STATUS_MAP.get(r.get("status")) == "OK"


# ----------------------------------------------------------------------------- 1. status, 2. validity, R2 verdict
def status_of(r, rcs):
    if r is None:
        if rcs:
            return "HARNESS_ERROR", f"rc={rcs[-1]} and no receipt (the process died before its first write; attempts {rcs})"
        return "NOT_RUN", "no receipt and no attempt line"
    st = STATUS_MAP.get(r.get("status"), "HARNESS_ERROR")
    assert st in VOCAB
    reason = r.get("reason", "") or ""
    if r.get("status") == "c1_failed":
        reason = "C1 FAILED: frozen bytes changed -- the arm trained but is VOID"
    if r.get("cited") and r.get("probe_reason"):
        reason = f"{reason} | probe on this box: {r['probe_reason']}"
    if r.get("loader_fallback_reason"):
        reason = (reason + " | " if reason else "") + f"loader fallback: {r.get('loader_used')}"
    return st, reason[:260]


def prologue_lines(rows, top=4):
    """e4b#548: what the arm spent before step 1, from the receipt's own `phase_seconds`; the residual printed ALWAYS."""
    out = []
    for x in rows:
        r = x["r"] or {}
        ph, tot = r.get("phase_seconds"), r.get("prologue_s")
        if not ph or tot is None:
            continue
        big = sorted(ph.items(), key=lambda kv: -kv[1])[:top]
        share = f"{100 * tot / (tot + (r.get('window_wall_s') or 0)):.0f}% of the arm" if r.get("window_wall_s") else "—"
        out.append(f"- prologue `{x['fw']}/{x['tag']}` **{tot:.1f} s** before step 1 ({share}): "
                   + ", ".join(f"{k} {v:.1f}" for k, v in big)
                   + f"; unattributed {r.get('prologue_unattributed_s', '—')}"
                   + (f"; budget {r['phase_budget_s']}" if r.get("phase_budget_s") else ""))
    return out


def c1_ok(r):
    """D: clean only with bytes hashed, no empties, bit-exact, AND a computed control on a NAMED tensor (a receipt without
    `C1_control_tensor` carries no real control, whatever its boolean says)."""
    return bool(r.get("C1_bit_exact")) and r.get("C1_bytes_hashed", 0) > 0 and r.get("C1_empties_skipped", 1) == 0 \
        and bool(r.get("C1_control_detects_flipped_byte")) and bool(r.get("C1_control_tensor"))


def regime_of(fam, r):
    """What is 4-bit in this arm, from the census: the recorded regime label (never a VOID)."""
    L = N_LAYERS.get(fam) or r.get("n_layers") or 0
    c = r.get("census", {}) or {}
    if r.get("framework") == "e4b":
        return "4-bit experts (e4b NF4) + " + ("NF4 attention" if r.get("attn_4bit") else "bf16 attention") + (" (attention-only adapters)" if r.get("arm") == "attn_only" else "")
    stacks = c.get("Params4bit_expert_stacks", 0) or 0
    attn4 = c.get("Linear4bit", 0) or 0
    if r.get("framework") == "unsloth":                                   # R11: the regimes beyond bnb stacks
        reg = unsloth_regime(r)
        if reg == "packed":
            n, epc = packed_expert_params(r)
            return (f"packed experts in the checkpoint's own format ({', '.join(f'{k} x{v}' for k, v in epc.items()) or 'no expert parameter class recorded'}; load_in_4bit=False) + "
                    f"{'bnb-4bit' if attn4 else 'bf16'} attention; backend selected {r.get('moe_backend_selected')}")
        if reg == "per-expert-linear4bit":
            return (f"per-expert Linear4bit experts ({c.get('Params4bit_expert_linears')} Params4bit under the experts container, no fused stack: the per-expert LoRA loop) + "
                    f"{'bnb-4bit' if attn4 else 'bf16'} attention (Linear4bit {attn4})")
    if r.get("framework") == "axolotl":
        # Corrected 2026-10-02: axolotl's quantize_moe_experts stores each expert stack as a torch parametrization whose original
        # is the 4-bit tensor, so the census's Params4bit_expert_stacks reads 0 and this label said "bf16 experts" for arms whose
        # experts were NF4 (Granite: 64 stacks; Qwen3-30B-A3B: 96, at a 26.9 GB peak bf16 experts could not fit). The arm's own
        # record of what axolotl quantized is the evidence, with the frozen-base probe's dequantisation regime beside it.
        ab = r.get("axolotl_bnb4bit_modules") or {}
        n_q = int(ab.get("quantized_moe_experts_n") or 0)
        if L and n_q >= 2 * L:
            probe = ((r.get("frozen_base_probe") or {}).get("slots") or {})
            reg = next((s.get("regime") for s in probe.values() if isinstance(s, dict) and s.get("regime")), None)
            return (f"4-bit expert stacks (axolotl quantize_moe_experts: {n_q} parametrized stacks{', ' + reg if reg else ''}) + "
                    f"{'bnb-4bit' if attn4 else 'bf16'} attention (Linear4bit {attn4})")
    exp = "4-bit expert stacks" if L and stacks >= 2 * L else ("PARTIAL 4-bit experts" if stacks else "bf16 experts (NOT the 4-bit MoE regime)")
    out = f"{exp} + {'bnb-4bit' if attn4 else 'bf16'} attention (Params4bit stacks {stacks}, Linear4bit {attn4})"
    d = r.get("hf_experts_dispatch")
    if isinstance(d, dict):                                                # R11: the HF arm's dispatch rides on its regime label
        out += f"; experts_implementation {d.get('config')!r}, torch grouped_mm {d.get('torch_grouped_mm_calls_per_step_min')}/step ({'reached' if d.get('reached_grouped_mm') else 'NOT reached'})"
    return out


def matched_seed_of(r):
    m = re.fullmatch(r"matched:(\d+)", str((r or {}).get("lora_init") or ""))
    return int(m.group(1)) if m else None


def step0_class(d):
    """C: |delta step-0 held-out| vs the anchor -> SAME-BYTES-CLASS (<= 0.01) / NEAR (<= 0.05) / VOID; never 'same init'."""
    if d is None:
        return None
    return "SAME-BYTES-CLASS" if d <= STEP0_SAME else ("NEAR" if d <= STEP0_VOID else "VOID")


def validity(fam, r, tokens_sha, e4b_trainable, n_steps, matched=False, ref_step0=None, anchor_seed=None, is_ref=False, anchor_sha=None, common_set=True):
    """tp4's validity rules for an OK row (+ R3 for a registered matched arm, + R9, + R11) -> (VALID|VOID, why). The n_layers table is
    the registered one; accum-aware (a kernel call per micro-batch per layer). `common_set=False` (gpt-oss): the trainable-count and
    harness-mismatch predicates against e4b do not apply -- the family has no common adapter set by registration."""
    if r is None or not is_ok(r):
        return "—", ""
    L = N_LAYERS.get(fam)
    A = int(r.get("accum") or 1)
    fw = r.get("framework")
    why = []
    if L is None:
        why.append(f"no registered n_layers for family {fam}")
        L = r.get("n_layers") or 0
    elif r.get("n_layers") not in (None, L):
        why.append(f"config n_layers {r.get('n_layers')} != registered {L}")
    want_census = ATTN_CENSUS.get(fam)                                   # R11: the registered structural attention census
    if want_census is not None and r.get("structural_expected_n_attn4") not in (None, want_census):
        why.append(f"structural attention census {r.get('structural_expected_n_attn4')} != registered {want_census}")
    if fam in TC2_MODELS:                                                # R11: the pin
        mid, rev, _ = TC2_MODELS[fam]
        if r.get("model") not in (None, mid) or r.get("revision") not in (None, rev):
            why.append(f"model/revision {r.get('model')} @ {str(r.get('revision'))[:12]} != the registered pin {mid} @ {rev[:12]}")
    if not c1_ok(r) or r.get("status") == "c1_failed":
        why.append("C1 not clean" if r.get("status") != "c1_failed" else "C1 FAILED (the arm's own status)")
        if not r.get("C1_control_tensor"):
            why[-1] += " (no C1_control_tensor: the positive control was not run on real storage)"
    if len(r.get("losses", [])) != r.get("steps") or (n_steps and r.get("steps") != n_steps):
        why.append(f"step count {len(r.get('losses', []))}/{r.get('steps')} != N={n_steps or r.get('steps')}")
    if tokens_sha and r.get("tokens", {}).get("sha256") != tokens_sha:
        why.append("tokens sha differs from the family's")
    if not common_set:
        pass                                                             # R11: no common adapter set with e4b on this family (recorded on the row, never a VOID)
    elif e4b_trainable is not None and r.get("trainable_params") != e4b_trainable:
        g = r.get("trainable_by_group") or {}
        if fw != "e4b" and g.get("experts") == 0:                        # R11: the arm adapted no expert parameter -- attention only, as tp4 read Granite / Qwen3.6
            why.append(f"attention-only: trainable {r.get('trainable_params')} != e4b's {e4b_trainable} ({fw} adapted no expert parameter; by group: {g})")
        else:
            why.append(f"trainable {r.get('trainable_params')} != e4b's {e4b_trainable} (by group: {g})")
    elif r.get("trainable_mismatch"):
        tm = r["trainable_mismatch"]
        why.append(f"harness recorded a trainable mismatch: expected {tm.get('expected')} got {tm.get('got')} (by group: {tm.get('by_group')})")
    if fw == "e4b":
        if r.get("arm") == "fused":
            if r.get("n_patched") != L:
                why.append(f"n_patched {r.get('n_patched')} != {L}")
            if r.get("kernel_calls_per_step_min", 0) < 2 * L * A:
                why.append(f"kernel calls/step min {r.get('kernel_calls_per_step_min')} < 2*{L}*accum {A}")
            if not r.get("lora_path_present"):                     # A [F1]
                why.append("no grouped-nf4-gemm LoRA-path counters on the receipt (lora_path_present false): the path cannot be verified")
            elif r.get("lora_path_loop_steps"):
                st = r["lora_path_loop_steps"]
                why.append(f"the per-expert LoRA loop ran on step(s) {st[:6]}{'...' if len(st) > 6 else ''} (lora_loop_share max {max(v for v in (r.get('lora_loop_share') or [0]) if v is not None):.3f})")
        if r.get("attn_4bit"):
            want = r.get("structural_expected_n_attn4")
            if want is None:
                want = 4 * L
            if r.get("n_attn4") != want:
                why.append(f"n_attn4 {r.get('n_attn4')} != {want} ({'structural census' if r.get('structural_expected_n_attn4') is not None else '4*L'})")
    elif fw == "unsloth" and unsloth_regime(r) == "packed":              # R11: the 16-bit load, experts kept in the checkpoint's packed format (gpt-oss MXFP4)
        n_packed, epc = packed_expert_params(r)
        if n_packed < 2 * L:
            why.append(f"packed expert parameters {n_packed} < 2*{L} (expert parameter classes {epc}): the experts are not packed")
        if r.get("experts_forward_calls_per_step_min", 0) < L * A:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < {L}*accum {A}")
        if not any(UNSLOTH_BANNER in s for s in r.get("engagement_banners", []) or []):
            why.append(f"no '{UNSLOTH_BANNER}' banner")
        if r.get("moe_backend_selected") != "grouped_mm":
            why.append(f"moe_backend_selected {r.get('moe_backend_selected')!r} != grouped_mm (the packed MXFP4 path's own condition, mxfp4.py:141-164 per UPSTREAM-NOTES)")
        key, pk = "unsloth_mxfp4_grouped_mm", r.get("unsloth_packed_calls_per_step_min")
        absent = [s for s in (r.get("unsloth_backend_absent") or []) if "Mxfp4GroupedMM" in s]
        if absent or not isinstance(pk, dict) or key not in pk:
            why.append(f"the MXFP4 grouped GEMM entry point (unsloth_zoo.mxfp4_gemm.Mxfp4GroupedMM.apply) was not counted on this box ({absent or 'no packed counters on the receipt'}): the packed path cannot be verified")
        elif pk.get(key, 0) < L * A:
            why.append(f"MXFP4 grouped GEMM engaged {pk.get(key, 0)} < {L}*accum {A} per step")
    elif fw == "unsloth" and unsloth_regime(r) == "per-expert-linear4bit":   # R11: gpt-oss's bnb-4bit load -- per-expert Linear4bit, the per-expert LoRA loop (gpt_oss.py:1214-1300)
        c = r.get("census", {}) or {}
        if c.get("Params4bit_expert_linears", 0) < 2 * L:
            why.append(f"Params4bit under the experts container {c.get('Params4bit_expert_linears')} < 2*{L}")
        if r.get("experts_forward_calls_per_step_min", 0) < L * A:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < {L}*accum {A}")
        if not any((UNSLOTH_BANNER in s) or (UNSLOTH_BANNER_PER_EXPERT in s) for s in r.get("engagement_banners", []) or []):
            why.append(f"neither the '{UNSLOTH_BANNER}' nor the '{UNSLOTH_BANNER_PER_EXPERT}' banner")
        # the backend request is recorded, not required: LoRA-wrapped 4-bit experts take the per-expert loop on this class (UPSTREAM-NOTES)
    elif fw == "unsloth":
        c = r.get("census", {}) or {}
        if c.get("Params4bit_expert_stacks", 0) < 2 * L:
            why.append(f"Params4bit expert stacks {c.get('Params4bit_expert_stacks')} < 2*{L}")
        if r.get("experts_forward_calls_per_step_min", 0) < L * A:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < {L}*accum {A}")
        b = r.get("unsloth_bnb4bit_modules")
        nb = b.get("n_bnb4bit_unwrapped") if isinstance(b, dict) else None
        if nb is None or nb < L:
            why.append(f"bnb4bit expert modules (innermost) {nb} < {L} (silent fallback?)")
        if not any("Enabling LoRA on MoE parameters" in s for s in r.get("engagement_banners", []) or []):
            why.append("no 'Enabling LoRA on MoE parameters' banner")
        kn = r.get("unsloth_knobs") or {}                # R8: the REQUESTED backend must be the one that ran
        req = kn.get("moe_backend_requested")
        if req and req != "default":
            mins, maxs = r.get("unsloth_backend_calls_per_step_min"), r.get("unsloth_backend_calls_per_step_max") or {}
            key = UNSLOTH_BACKEND_KEYS.get(req)
            if not isinstance(mins, dict) or key is None:
                why.append(f"backend {req} requested but the receipt carries no backend counters")
            else:
                if mins.get(key, 0) < L * A:
                    why.append(f"requested backend {req}: {key} engaged {mins.get(key, 0)} < {L}*accum {A} per step (moe_backend_selected {r.get('moe_backend_selected')})")
                if req == "grouped_mm" and (mins.get("unsloth_loop", 0) or maxs.get("unsloth_loop", 0)):
                    why.append(f"grouped_mm requested but the per-expert loop ran (unsloth_loop min {mins.get('unsloth_loop')} max {maxs.get('unsloth_loop')})")
            if req == "grouped_mm":                                 # E [F15]: the torch op itself, and the manual fallback
                g, m = r.get("unsloth_grouped_mm_calls_per_step_min"), r.get("unsloth_manual_grouped_mm_calls_per_step_max")
                if g is None or m is None:
                    why.append("grouped_mm requested but no torch._grouped_mm / manual-fallback counters on the receipt")
                else:
                    if g < GMM_FACTOR * L * A:
                        why.append(f"torch._grouped_mm {g} calls/step < {GMM_FACTOR}*{L}*accum {A} = {GMM_FACTOR * L * A}")
                    if m:
                        why.append(f"the manual grouped-mm fallback ran ({m} calls/step max)")
    elif fw in ("hf", "axolotl"):
        if r.get("experts_forward_calls_per_step_min", 0) < L * A:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < {L}*accum {A}")
        key = "hf_targets" if fw == "hf" else "axolotl_targets"
        if not (r.get(key) or r.get("hf_targets") or {}).get("n_target_parameters"):
            why.append("adapted no expert parameter (target_parameters empty) -- attention-only, not the registered adapter set")
        if fw == "axolotl" and ((r.get("axolotl") or {}).get("config") or {}).get("quantize_moe_experts"):   # R8: the 4-bit expert path engaged
            nb = (r.get("axolotl_bnb4bit_modules") or {}).get("n_bnb4bit_unwrapped")
            if nb is None or nb < L:
                why.append(f"quantize_moe_experts set but bnb-parametrized experts modules (innermost) {nb} < {L}")
    if fam in (SYNC_FAM, PROF945_FAM, LEAN_FAM, TILE_FAM) and fw == "e4b":   # amendments 10 / 12 / 13 / 14: the arm ran the sync path its tag names (13, 14: the new one)
        w = sync_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == LEAN_FAM and fw == "e4b":                 # amendment 13: ... and the padded LoRA-delta body its tag names
        w = lean_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == TILE_FAM and fw == "e4b":                 # amendment 14: ... and the prefill M-tile rule its tag names, on the trimmed delta
        w = tile_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if matched:                                   # R3: the matched set's own predicates (TC1-PREREG "Validity", new in this lane)
        seed = matched_seed_of(r)
        if seed is None:
            why.append(f"registered matched but ran lora_init={r.get('lora_init')!r}")
        elif anchor_seed is not None and seed != anchor_seed:
            why.append(f"matched seed {seed} != the anchor's {anchor_seed}")
        mi = r.get("matched_init") or {}
        if not mi.get("complete"):
            why.append(f"matched_init.complete is {mi.get('complete')} (set {mi.get('n_slots_set')} of {mi.get('n_slots_expected')}; unmapped {str((mi.get('unmapped') or [])[:2])[:160]})")
        dts = list((r.get("adapter_dtypes_after") or {}).keys())
        if dts != ["torch.float32"]:
            why.append(f"adapter_dtypes_after {r.get('adapter_dtypes_after')} is not fp32-only")
        if anchor_sha is not None:                                   # B [F3]: the name-free slot sha must equal the anchor's
            if not r.get("matched_init_sha"):
                why.append("no matched_init_sha on the receipt")
            elif r["matched_init_sha"] != anchor_sha and not is_ref:
                why.append(f"matched_init_sha {r['matched_init_sha'][:12]} != the anchor's {anchor_sha[:12]} (the slot tensors differ: the override did not take, or the slot map differs)")
        if not is_ref and ref_step0 is not None and r.get("eval_loss_step0") is not None:
            d0 = abs(r["eval_loss_step0"] - ref_step0)
            if step0_class(d0) == "VOID":                            # C: > 0.05 VOIDs; <= 0.05 is NEAR, <= 0.01 SAME-BYTES-CLASS (recorded, never 'same init')
                why.append(f"step-0 held-out {r['eval_loss_step0']:.4f} differs from the anchor's {ref_step0:.4f} by {d0:.4f} > {STEP0_VOID}")
    return ("VOID" if why else "VALID"), "; ".join(why)


def verdict_of(st, v, qdelta):
    """R2: the one-word verdict. `qdelta` is the held-out delta at N against the quality anchor (None = no anchor to read)."""
    if st == "OK":
        if v == "VOID":
            return "VOID"
        if qdelta is not None and abs(qdelta) > READ:
            return "QUALITY_FAIL"
        return "VALID"
    if st == "OOM":
        return "OOM"
    if st in ("REFUSED", "INSTALL_FAILED", "LOAD_FAULT"):
        return "UNSUPPORTED"
    assert st in ("HARNESS_ERROR", "ALARM", "NOT_RUN"), st
    return st


# ----------------------------------------------------------------------------- 4. e4b internal parity (tp1's rule)
def parity(ref, arm):
    if arm is None or not is_ok(arm):
        return "NO-ARM", None, None, ""
    if ref is None or not is_ok(ref):
        return "NO-REF", None, None, "reference arm missing or not OK"
    why = []
    if arm.get("init_sha") != ref.get("init_sha"):
        why.append("init_sha differs from reference (arms did not start identical)")
    for tag, r in (("ref", ref), ("arm", arm)):
        if not c1_ok(r):
            why.append(f"C1 not clean on {tag}")
    if arm.get("n_patched", 0) == 0:
        why.append("n_patched == 0")
    A = int(arm.get("accum") or 1)
    if arm.get("kernel_calls_per_step_min", 0) < 2 * arm.get("n_patched", 0) * A:
        why.append(f"kernel calls/step min {arm.get('kernel_calls_per_step_min')} < 2*n_patched*accum={2 * arm.get('n_patched', 0) * A}")
    if len(arm.get("losses", [])) != len(ref.get("losses", [])) or not arm.get("losses"):
        why.append("step counts differ")
    if why:
        return "VOID", None, None, "; ".join(why)
    d_final = abs(arm["loss_last"] - ref["loss_last"])
    med = statistics.median(abs(x - y) for x, y in zip(arm["losses"], ref["losses"]))
    return ("PASS" if d_final <= BAND and med <= BAND else "FAIL"), d_final, med, ""


def curve_at(r, step):
    for c in r.get("eval_curve", []) or []:
        if c["step"] == step:
            return c["heldout_loss"]
    return None


def heldout_at_N(r):
    return curve_at(r, r.get("steps")) if r else None


# ----------------------------------------------------------------------------- R1: draws
def draws_of(recs, verdicts, key, expected=None):
    """One arm, one or two draws -> {usable, s, peak, j, tok, draws, stability, threshold, verdict, why}. Medians over the
    usable draws; a second draw REGISTERED for this family (in `expected`) that is missing or not VALID leaves the stability
    UNMEASURED (not usable); a family that registers no second draw (qwen3native) reads the arm as a single draw."""
    fw, tag = key
    r1, k2 = recs.get(key), DRAW2.get(key)
    if k2 is not None and expected is not None and k2 not in expected:
        k2 = None
    v1 = verdicts.get(key)
    out = {"key": key, "usable": False, "draws": 0, "stability": None, "threshold": STABILITY.get(fw, STABILITY_OTHER), "verdict": "—", "why": ""}
    if v1 != "VALID":
        out["why"] = f"{fw}/{tag} is {v1 or 'missing'}"
        return out
    rs = [r1]
    if k2 is not None:
        r2, v2 = recs.get(k2), verdicts.get(k2)
        if v2 != "VALID":
            out.update({"draws": 1, "verdict": "UNMEASURED", "why": f"second draw {k2[1]} is {v2 or 'missing'}: stability unmeasured, position not quoted"})
            return out
        s1, s2 = r1["s_per_step_median_11plus"], r2["s_per_step_median_11plus"]
        stab = abs(s1 - s2) / statistics.mean([s1, s2]) if statistics.mean([s1, s2]) else None
        out.update({"draws": 2, "stability": stab, "s1": s1, "s2": s2, "verdict": "STABLE" if (stab is not None and stab <= out["threshold"]) else "UNSTABLE"})
        rs.append(r2)
        if out["verdict"] != "STABLE":
            out["why"] = f"draws {s1:.3f} / {s2:.3f} s/step differ by {100 * (stab or 0):.1f}% > {100 * out['threshold']:.0f}% (UNSTABLE: reported, not quoted)"
            return out
    else:
        out.update({"draws": 1, "verdict": "SINGLE", "why": "single draw (no second draw registered)"})
    rec_steps = [recompiles_in_window(r) for r in rs]                 # F [F16]: a recompile inside steps 11..N is an unstable draw
    if any(v for v in rec_steps):
        out.update({"usable": False, "verdict": "UNSTABLE", "recompiles": rec_steps,
                    "why": f"torch._dynamo recompiled inside steps 11..N ({rec_steps} between the step-10 and step-N snapshots): UNSTABLE, not quoted"})
        return out
    out.update({"usable": True, "s": statistics.median([r["s_per_step_median_11plus"] for r in rs]), "s_list": [r["s_per_step_median_11plus"] for r in rs],
                "heldout_list": [heldout_at_N(r) for r in rs],
                "peak": statistics.median([r["peak_vram_gb"] for r in rs]),
                "j": (statistics.median([r["joules_per_step"] for r in rs]) if all(r.get("joules_per_step") is not None for r in rs) else None),
                "tok": (statistics.median([r["tokens_per_s"] for r in rs]) if all(r.get("tokens_per_s") is not None for r in rs) else None),
                "heldout": heldout_at_N(r1), "step0": r1.get("eval_loss_step0"), "regime": None})
    return out


def recompiles_in_window(r):
    """F: recompiles between the step-10 and step-N dynamo snapshots (None when a snapshot is missing)."""
    d = r.get("dynamo_counters") or {}
    a, b = d.get("step10") or {}, d.get(f"step{r.get('steps')}") or {}
    if a.get("recompiles_total") is None or b.get("recompiles_total") is None:
        return None
    return int(b["recompiles_total"]) - int(a["recompiles_total"])


def position(e, o, label):
    """The pair (e4b draws e, other draws o): quoted only when both are usable (VALID verdict, STABLE or single draw).
    H: the point ratio (medians) AND the interval (min, max) over the cross-draw ratios."""
    pos = {"label": label, "quoted": False, "why": ""}
    if e.get("usable") and o.get("usable"):
        cross = [ratio(os_, es_) for os_ in o.get("s_list", [o["s"]]) for es_ in e.get("s_list", [e["s"]])]
        pos.update({"quoted": True, "ratio": ratio(o["s"], e["s"]), "ratio_min": min(cross), "ratio_max": max(cross), "n_cross": len(cross),
                    "e4b_s": e["s"], "other_s": o["s"], "e4b_draws": e["draws"], "other_draws": o["draws"],
                    "e4b_stability": e.get("stability"), "other_stability": o.get("stability"),
                    "peak_e4b": e["peak"], "peak_other": o["peak"], "peak_delta": o["peak"] - e["peak"],
                    "j_e4b": e.get("j"), "j_other": o.get("j"), "j_ratio": ratio(o.get("j"), e.get("j")),
                    "tok_e4b": e.get("tok"), "tok_other": o.get("tok"), "other_regime": None,
                    "heldout_e4b": e.get("heldout"), "heldout_other": o.get("heldout"),
                    "heldout_delta": (o["heldout"] - e["heldout"]) if (e.get("heldout") is not None and o.get("heldout") is not None) else None,
                    "step0_delta": (o["step0"] - e["step0"]) if (e.get("step0") is not None and o.get("step0") is not None) else None})
        pos["quality"] = "COMPARABLE" if pos["heldout_delta"] is not None and abs(pos["heldout_delta"]) <= READ else "FLAGGED"
    else:
        pos["why"] = f"not quoted: e4b {e.get('why') or e.get('verdict')} / {label} {o.get('why') or o.get('verdict')}"
    return pos


# ----------------------------------------------------------------------------- R4: equivalence, R5: frozen base
def paired_rows(ref, r):
    """G [F13]: the per-row held-out differences (arm - anchor) at N: mean +- SE and the rows favouring each side."""
    ra = next((e for e in (r.get("eval_rows") or []) if e.get("step") == r.get("steps")), None)
    rb = next((e for e in (ref.get("eval_rows") or []) if e.get("step") == ref.get("steps")), None)
    if not ra or not rb or not ra.get("losses") or not rb.get("losses") or len(ra["losses"]) != len(rb["losses"]):
        return None
    d = [x - y for x, y in zip(ra["losses"], rb["losses"])]
    n = len(d)
    se = (statistics.stdev(d) / (n ** 0.5)) if n > 1 else None
    return {"n": n, "mean": statistics.mean(d), "se": se, "rows_favouring_arm": sum(1 for v in d if v < 0), "rows_favouring_anchor": sum(1 for v in d if v > 0)}


def equivalence(ref, r, vref, vr, band=None, noise_floor=None):
    """{reading, med_train, d_heldout, d_step0, step0_class, band, paired, why} for one matched arm against the anchor
    (e4b/fused_attn4_m). H: EQUIVALENT iff median per-step |delta train| and |delta held-out at N| are both <= the band
    (max(0.005, 3 x the in-draw fused-vs-reference |delta|)); without a reference arm the band is None and the reading can
    only be COMPARABLE (<= 0.05) or DIVERGENT. G: a reading narrower than the draw-noise floor reads INSIDE-DRAW-NOISE."""
    if r is None or not is_ok(r):
        return {"reading": "—", "why": "no OK receipt"}
    if ref is None or not is_ok(ref):
        return {"reading": "N-A", "why": "the anchor e4b/fused_attn4_m is missing or not OK"}
    if vref != "VALID" or vr != "VALID":
        return {"reading": "N-A", "why": f"validity: anchor {vref}, arm {vr} (VOID never enters an equivalence reading)"}
    if len(r.get("losses", [])) != len(ref.get("losses", [])) or not r.get("losses"):
        return {"reading": "N-A", "why": "step counts differ"}
    med = statistics.median(abs(x - y) for x, y in zip(r["losses"], ref["losses"]))
    hr, hf_ = heldout_at_N(r), heldout_at_N(ref)
    if hr is None or hf_ is None:
        return {"reading": "N-A", "why": "no held-out loss at N on one side", "med_train": med}
    dh = abs(hr - hf_)
    d0 = abs((r.get("eval_loss_step0") or 0) - (ref.get("eval_loss_step0") or 0)) if (r.get("eval_loss_step0") is not None and ref.get("eval_loss_step0") is not None) else None
    d2 = abs(r["loss_step2"] - ref["loss_step2"]) if (r.get("loss_step2") is not None and ref.get("loss_step2") is not None) else None
    why = ""
    if band is not None and med <= band["train"] and dh <= band["heldout"]:
        reading = "EQUIVALENT"
        if noise_floor is not None and dh <= noise_floor and med <= noise_floor:
            reading, why = "INSIDE-DRAW-NOISE", f"|delta| {dh:.4f} / {med:.4f} are narrower than the draw-noise floor {noise_floor:.4f}: inside the draw noise, not a precision statement"
    elif med <= COMPARABLE and dh <= COMPARABLE:
        reading = "COMPARABLE"
        if band is None:
            why = "no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling"
    else:
        reading = "DIVERGENT"
    return {"reading": reading, "med_train": med, "d_heldout": dh, "d_step0": d0, "step0_class": step0_class(d0), "d_loss_step2": d2,
            "band": band, "noise_floor": noise_floor, "paired": paired_rows(ref, r), "why": why}


def frozen_base(anchor, r):
    """Per registered slot: SAME-BYTES / DIFFERENT when both arms carry the slot in the SAME regime, N-A otherwise;
    plus each side's own control. Nothing is compared across regimes."""
    out = {}
    pa = ((anchor or {}).get("frozen_base_probe") or {}).get("slots") or {}
    pr = ((r or {}).get("frozen_base_probe") or {}).get("slots") or {}
    for slot in ("gate_up", "down", "q_proj"):
        a, b = pa.get(slot), pr.get(slot)
        if not a or not b:
            out[slot] = {"reading": "N-A", "why": "slot missing on " + ("both" if not a and not b else ("the anchor" if not a else "this arm")), "regime": (b or {}).get("regime")}
        elif a.get("regime") != b.get("regime"):
            out[slot] = {"reading": "N-A", "why": f"regime differs: anchor {a.get('regime')} vs {b.get('regime')}", "regime": b.get("regime")}
        else:
            out[slot] = {"reading": "SAME-BYTES" if a.get("sha") == b.get("sha") else "DIFFERENT", "regime": b.get("regime"), "why": ""}
        out[slot]["control"] = bool((b or {}).get("control_detects_flip"))
    return out


# ----------------------------------------------------------------------------- R11: the no-common-set line (gpt-oss) and the footprint line (mixtral)
def no_common_set_reading(fam, e_d, o_d, e_rec, o_rec, key):
    """R11: the line that REPLACES a position on a family with no common adapter set: both s/step values (quoted draws only), both peaks,
    both trainable counts and each side's state -- never a ratio."""
    def _state(d):
        return f"{d.get('draws')} draw(s), {d.get('verdict')}" if d.get("usable") else (d.get("why") or d.get("verdict") or "no receipt")
    return {"quoted": False, "no_common_set": True, "label": f"{key[0]}/{key[1]}", "e4b_label": f"e4b/{anchor_of(fam)[1]}",
            "e4b_s": e_d.get("s") if e_d.get("usable") else None, "other_s": o_d.get("s") if o_d.get("usable") else None,
            "e4b_peak": e_d.get("peak") if e_d.get("usable") else None, "other_peak": o_d.get("peak") if o_d.get("usable") else None,
            "e4b_trainable": (e_rec or {}).get("trainable_params") if is_ok(e_rec) else None,
            "other_trainable": (o_rec or {}).get("trainable_params") if is_ok(o_rec) else None,
            "e4b_state": _state(e_d), "other_state": _state(o_d), "why": NO_COMMON_SET[fam]}


def footprint(fam, recs, draws, verdicts):
    """R11: mixtral's footprint reading -- e4b under expert offload vs Unsloth resident: peak VRAM (GB) and s/step from the quoted draws when
    usable, else from the single OK receipt with its verdict named (a measurement, never a position)."""
    out = {"fam": fam, "label": FOOTPRINT_FAMS.get(fam, "e4b (offload) vs Unsloth (resident)"), "sides": {}}
    for side, key in (("e4b", ("e4b", PRIMARY["e4b"])), ("unsloth", ("unsloth", PRIMARY["unsloth"]))):
        d, r = draws.get(key, {}), recs.get(key)
        if d.get("usable"):
            out["sides"][side] = {"peak": d["peak"], "s": d["s"], "draws": d["draws"], "basis": f"quoted draws ({d['verdict']})", "offload": bool((r or {}).get("offload"))}
        elif is_ok(r) and r.get("peak_vram_gb") is not None:
            out["sides"][side] = {"peak": r.get("peak_vram_gb"), "s": r.get("s_per_step_median_11plus"), "draws": 1,
                                  "basis": f"single receipt, verdict {verdicts.get(key)} (not a quoted draw)", "offload": bool(r.get("offload"))}
        else:
            out["sides"][side] = {"peak": None, "s": None, "draws": 0, "basis": f"{verdicts.get(key, 'missing')}: no peak to read", "offload": None}
    e, u = out["sides"]["e4b"], out["sides"]["unsloth"]
    out["peak_ratio"] = ratio(u.get("peak"), e.get("peak")) if (e.get("peak") and u.get("peak")) else None
    out["s_ratio_e4b_over_unsloth"] = ratio(e.get("s"), u.get("s")) if (e.get("s") and u.get("s")) else None
    out["readable"] = out["peak_ratio"] is not None
    return out


def footprint_line(fp):
    e, u = fp["sides"]["e4b"], fp["sides"]["unsloth"]
    if fp["readable"]:
        return (f"- **FOOTPRINT ({fp['label']}): e4b `fused_attn4_m` peak VRAM {e['peak']:.2f} GB{' under expert offload' if e.get('offload') else ''} vs Unsloth `ckpt_unsloth_m` "
                f"{u['peak']:.2f} GB resident (×{fp['peak_ratio']:.2f} lower on e4b); s/step e4b {f(e['s'])} vs Unsloth {f(u['s'])} (e4b/Unsloth {f(fp['s_ratio_e4b_over_unsloth'])})** "
                f"— e4b: {e['basis']}; Unsloth: {u['basis']}")
    return f"- **FOOTPRINT ({fp['label']}): not readable** — e4b: {e['basis']}; Unsloth: {u['basis']}"


# ----------------------------------------------------------------------------- the per-family reduction (one dict; the printer and the tests read it)
def reduce_family(fam, recs, rcs_all, n_steps=None):
    exp = list(EXPECTED.get(fam, EXPECTED["qwen3"]))
    keys = exp + [k for k in recs if k not in exp]
    qa = anchor_of(fam)                              # R11: the family's e4b anchor arm (attn_only_m on gpt-oss)
    common = fam not in NO_COMMON_SET                # R11: gpt-oss -- no framework shares e4b's adapter set
    e_anchor = recs.get(qa)
    e4b_trainable = e_anchor.get("trainable_params") if is_ok(e_anchor) else None
    if e4b_trainable is None:                        # fall back to the e4b reference arm (same recipe, same adapter set)
        rr = recs.get(EQUIV_ANCHOR)
        if is_ok(rr):
            e4b_trainable = rr.get("trainable_params")

    def trainable_ref(tag):                          # the mb1 pair has its OWN e4b arm as the trainable reference
        if tag.endswith("_mb1"):
            r = recs.get(("e4b", SECONDARY["e4b"]))
            return r.get("trainable_params") if is_ok(r) else None
        return e4b_trainable
    tokens_sha = next((r["tokens"]["sha256"] for k in keys if (r := recs.get(k)) and r.get("tokens", {}).get("sha256")), None)
    N = n_steps or next((r.get("steps") for r in recs.values() if is_ok(r) and r.get("steps")), None) \
        or next((r.get("steps") for r in recs.values() if r and r.get("steps")), None)      # R11: an OK receipt's N before a stub's
    ref = recs.get(REFERENCE)
    anchor = e_anchor if is_ok(e_anchor) else (ref if is_ok(ref) else None)          # I: fused_m, else the reference as a stand-in
    anchor_key = qa if is_ok(e_anchor) else (REFERENCE if is_ok(ref) else None)
    ref_step0 = anchor.get("eval_loss_step0") if anchor else None
    anchor_seed = matched_seed_of(anchor) if anchor else None
    anchor_sha = (anchor.get("matched_init_sha") or None) if (anchor and matched_seed_of(anchor) is not None) else None   # B
    rows, V = [], {}
    for fw, tag in keys:
        r = recs.get((fw, tag))
        st, reason = status_of(r, rcs_all.get((fam, fw, tag)))
        own_set = common or fw == "e4b"              # R11: on gpt-oss only e4b's own arms are read against the e4b anchor's count / sha
        v, why = validity(fam, r, tokens_sha, trainable_ref(tag) if (fw != "e4b" and common) else None, N,
                          matched=(tag in MATCHED), ref_step0=ref_step0, anchor_seed=anchor_seed, is_ref=((fw, tag) == anchor_key),
                          anchor_sha=(anchor_sha if own_set else None), common_set=own_set)
        d0 = abs(r["eval_loss_step0"] - ref_step0) if (is_ok(r) and tag in MATCHED and ref_step0 is not None and r.get("eval_loss_step0") is not None) else None
        rows.append({"fw": fw, "tag": tag, "status": st, "reason": reason, "validity": v, "why": why, "r": r,
                     "regime": regime_of(fam, r) if (r and st == "OK") else None, "matched": tag in MATCHED,
                     "step0_delta": d0, "step0_class": step0_class(d0),
                     "dispatch": hf_dispatch_note(r) if (fw == "hf" and is_ok(r)) else None,                        # R11: recorded, never VOID
                     "no_common_set": (f"no common adapter set with e4b: trainable {r.get('trainable_params')} vs e4b's {e4b_trainable} -- {NO_COMMON_SET[fam]}"
                                       if (not own_set and is_ok(r)) else None)})
        V[(fw, tag)] = v
    # R2: quality against the anchor (read only when the anchor is VALID by predicates), then the verdict
    anchor_ok = is_ok(e_anchor) and V.get(qa) == "VALID"
    ha = heldout_at_N(e_anchor) if anchor_ok else None
    verdicts = {}
    for x in rows:
        r = x["r"]
        q = None
        if x["status"] == "OK" and ha is not None and heldout_at_N(r) is not None and (common or x["fw"] == "e4b"):
            q = heldout_at_N(r) - ha
        x["quality_delta"] = q
        if x["status"] == "OK" and not common and x["fw"] != "e4b":
            x["quality"] = "N-A (no common adapter set)"                 # R11: a different adapter set's held-out is not a quality reading against e4b
        else:
            x["quality"] = ("COMPARABLE" if abs(q) <= READ else "FLAGGED") if q is not None else ("N-A (anchor " + ("VOID" if is_ok(e_anchor) else "missing") + ")" if x["status"] == "OK" else None)
        x["verdict"] = verdict_of(x["status"], x["validity"], q)
        assert x["verdict"] in VERDICTS
        verdicts[(x["fw"], x["tag"])] = x["verdict"]
    # tp1's parity (fused_m vs reference_m): the e4b-side control, and H's band for EQUIVALENT
    fu = recs.get(QUALITY_ANCHOR)
    pv, d_final, med, pwhy = parity(ref, fu)
    par = {"verdict": pv, "d_final": d_final, "median": med, "why": pwhy}
    band = None
    if pv in ("PASS", "FAIL"):
        par["speed_x"] = ratio(ref.get("s_per_step_median_11plus"), fu.get("s_per_step_median_11plus"))
        par["peak_x"] = ratio(fu.get("peak_vram_gb"), ref.get("peak_vram_gb"))
        hr_, hf2 = heldout_at_N(ref), heldout_at_N(fu)
        par["d_heldout"] = abs(hr_ - hf2) if (hr_ is not None and hf2 is not None) else None
        if par["d_heldout"] is not None:
            band = {"train": max(EQUIV_FLOOR, 3 * med), "heldout": max(EQUIV_FLOOR, 3 * par["d_heldout"])}
    par["equiv_band"] = band
    # R1 draws per arm, then the positions
    draws = {k: draws_of(recs, verdicts, k, expected=exp) for k in keys if k not in DRAW2.values()}
    e_d = draws.get(qa, {"usable": False, "why": f"no e4b/{qa[1]}"})
    positions = {}
    if common:
        for other in ("unsloth", "hf", "axolotl"):
            k = (other, PRIMARY[other])
            label = other
            if other == "hf" and fam in TC2_FAMS and is_ok(recs.get(k)):   # R11: "HF (bf16 experts) / e4b" (or 4-bit), the regime beside it
                label = hf_label(fam, recs[k])
            positions[other] = position(e_d, draws.get(k, {"usable": False, "why": "no receipt"}), label)
            if positions[other].get("quoted"):
                positions[other]["other_regime"] = regime_of(fam, recs[k])
    else:                                                                   # R11: gpt-oss -- the no-common-set line replaces every position
        for name, k in (("unsloth", ("unsloth", PRIMARY["unsloth"])), ("unsloth_mxfp4", ("unsloth", "ckpt_unsloth_mxfp4")),
                        ("hf", ("hf", PRIMARY["hf"])), ("axolotl", ("axolotl", PRIMARY["axolotl"]))):
            positions[name] = no_common_set_reading(fam, e_d, draws.get(k, {"usable": False, "why": "no receipt"}), e_anchor, recs.get(k), k)
    secondary = {}
    e2 = draws.get(("e4b", SECONDARY["e4b"]))
    for other in ("unsloth", "hf", "axolotl"):
        k2 = (other, SECONDARY[other])
        if e2 or recs.get(k2):
            secondary[other] = position(e2 or {"usable": False, "why": "no receipt"}, draws.get(k2, {"usable": False, "why": "no receipt"}), other + " (mb1)")
    labelled = {}                                       # R8: labelled rows against e4b/fused_attn4_m (R11: none on a family with no common adapter set)
    for k, label in ALL_LABELLED.items():
        if recs.get(k) is not None and common:
            labelled[k] = position(e_d, draws.get(k, {"usable": False, "why": "no receipt"}), label)
            if labelled[k].get("quoted"):
                labelled[k]["other_regime"] = regime_of(fam, recs[k])
    native = {}
    nat = NATIVE_BY_FAM.get(fam, NATIVE)                # amendment 8: the 200-step token names its own native arms
    sh = draws.get(("e4b", nat["e4b"]), {"usable": False, "why": "no receipt"})
    for other in ("unsloth", "axolotl"):
        if other not in nat:
            continue
        k = (other, nat[other])
        if recs.get(k) or recs.get(("e4b", nat["e4b"])):
            native[other] = position(sh, draws.get(k, {"usable": False, "why": "no receipt"}), f"{other} native-best vs e4b shipped")
            if native[other].get("quoted"):
                native[other]["other_regime"] = regime_of(fam, recs[k])
    # G: the draw-noise floor = max over arms with two usable draws of |held-out(d1) - held-out(d2)| at N
    floors = [abs(d["heldout_list"][0] - d["heldout_list"][1]) for d in draws.values()
              if d.get("usable") and len(d.get("heldout_list") or []) == 2 and None not in d["heldout_list"]]
    noise_floor = max(floors) if floors else None
    # R4/I: equivalence of every matched arm against e4b/fused_attn4_m (the fused-vs-reference pair rides along as the control); R5: frozen base
    equiv = {}
    for fw, tag in keys:
        if tag in MATCHED and (fw, tag) != qa and recs.get((fw, tag)) is not None:
            if not common and fw != "e4b":                                  # R11: a different adapter set has no equivalence reading against e4b
                equiv[(fw, tag)] = {"reading": "N-A", "why": "no common adapter set with e4b (gpt-oss: e4b attention-only vs this arm's experts); not an equivalence reading"}
            else:
                equiv[(fw, tag)] = equivalence(e_anchor, recs[(fw, tag)], V.get(qa), V.get((fw, tag)), band=band, noise_floor=noise_floor)
    frozen = {}
    for fw, tag in keys:
        r = recs.get((fw, tag))
        if is_ok(r) and (fw, tag) != qa:
            frozen[(fw, tag)] = frozen_base(e_anchor if is_ok(e_anchor) else None, r)
    anchor_probe = ((e_anchor or {}).get("frozen_base_probe") or {}) if is_ok(e_anchor) else {}
    prof = recs.get(PROF)
    profile = (prof.get("profile") or None) if is_ok(prof) else None
    fp = footprint(fam, recs, draws, verdicts) if (fam in FOOTPRINT_FAMS or (is_ok(e_anchor) and e_anchor.get("offload"))) else None   # R11
    return {"fam": fam, "rows": rows, "V": V, "verdicts": verdicts, "parity": par, "draws": draws, "positions": positions, "secondary": secondary,
            "native": native, "labelled": labelled, "equivalence": equiv, "frozen": frozen, "anchor_probe": anchor_probe, "profile": profile,
            "noise_floor": noise_floor, "equiv_band": band, "anchor_sha": anchor_sha, "footprint": fp, "common_set": common, "anchor_key": qa,
            "prof_knobs": (prof.get("unsloth_knobs") or {}) if is_ok(prof) else {},
            "tokens_sha": tokens_sha, "e4b_trainable": e4b_trainable, "N": N, "e": e_anchor, "ref": ref,
            "u": recs.get(("unsloth", PRIMARY["unsloth"])), "h": recs.get(("hf", PRIMARY["hf"])), "ax": recs.get(("axolotl", PRIMARY["axolotl"]))}


def _score_p6(fam, vd, pos, nvd):
    """P6 axolotl: INSTALL_FAILED/UNSUPPORTED/OOM, or trains with axolotl/e4b in [1.5, 6] -- read on `fam`: the judged box, or amendment 3's
    axolotl box (AX_FAM) when it ran, since the judged boxes' INSTALL_FAILED was the harness's own uv index strategy, not a reading on axolotl."""
    av = vd.get(("axolotl", PRIMARY["axolotl"]), "missing")
    pa = pos.get("axolotl", {})
    if av == "missing" and nvd.get(("axolotl", NATIVE["axolotl"]), "missing") != "missing":     # only the native-best axolotl row ran
        av = nvd.get(("axolotl", NATIVE["axolotl"]))
    if av in ("UNSUPPORTED", "OOM"):
        return ("P6", fam, "HELD", f"axolotl {av}")
    if av in ("VALID", "QUALITY_FAIL", "VOID"):
        if pa.get("quoted"):
            return ("P6", fam, "HELD" if 1.5 <= pa["ratio"] <= 6 else "FALSIFIED", f"axolotl trained; axolotl/e4b {pa['ratio']:.3f} vs [1.5, 6]")
        return ("P6", fam, "UNTESTED", f"axolotl {av} but no quoted position: {pa.get('why')}")
    return ("P6", fam, "UNTESTED", f"axolotl {av} (NOT_RUN / HARNESS_ERROR / ALARM is not a reading)")


def score_p13(F):
    """TC1-PREREG amendment 5's P13, on the qwen3nativebest box: e4b as shipped is the fastest native configuration of the three. Each half
    (axolotl native-best / e4b shipped, Unsloth native-best / e4b shipped) is HELD when its point ratio over two STABLE draws a side is above
    1.0 and FALSIFIED at or below it; an unstable, unmeasured or missing pair leaves that half UNTESTED. P13 is FALSIFIED when either half
    is, HELD when both are, UNTESTED otherwise. Held-out losses ride along, never judged across different inits."""
    R = F.get(NB_FAM)
    if not R:
        return []
    out = []
    for fw, name in NB_HALVES:
        p = R["native"].get(fw) or {"quoted": False, "why": f"no {fw}/{NATIVE[fw]} receipt"}
        if not p.get("quoted"):
            out.append((f"P13 ({name} half)", NB_FAM, "UNTESTED", p.get("why") or "not quoted"))
            continue
        if p["e4b_draws"] != 2 or p["other_draws"] != 2:
            out.append((f"P13 ({name} half)", NB_FAM, "UNTESTED", f"draws e4b {p['e4b_draws']} / {name} {p['other_draws']}: the registration asks for two stable draws a side"))
            continue
        ho = (f"; held-out at N e4b shipped {p['heldout_e4b']:.4f} / {name} {p['heldout_other']:.4f} (reported, not judged: each framework's own init)"
              if p.get("heldout_e4b") is not None and p.get("heldout_other") is not None else "")
        out.append((f"P13 ({name} half)", NB_FAM, "HELD" if p["ratio"] > 1.0 else "FALSIFIED",
                    f"{name} native-best / e4b shipped {p['ratio']:.3f} [{p['ratio_min']:.3f}, {p['ratio_max']:.3f} over {p['n_cross']} cross-draw ratios] (> 1.0 predicted); "
                    f"s/step e4b {p['e4b_s']:.3f} (draws within {100 * p['e4b_stability']:.1f}%) vs {name} {p['other_s']:.3f} (within {100 * p['other_stability']:.1f}%){ho}"))
    vs = [v for _, _, v, _ in out]
    overall = "FALSIFIED" if "FALSIFIED" in vs else ("HELD" if all(v == "HELD" for v in vs) else "UNTESTED")
    out.append(("P13", NB_FAM, overall, "; ".join(f"{pid.split('(')[1].rstrip(')')} {v}" for pid, _, v, _ in out)))
    return out


def late_median(r, lo=LATE_FROM):
    """Amendment 8: the median s/step over steps lo..N from the arm's own step_ms (None when the arm did not record steps lo..N)."""
    ms = (r or {}).get("step_ms") or []
    return statistics.median(ms[lo - 1:]) / 1e3 if len(ms) >= lo else None


def score_p14(F):
    """TC1-PREREG amendment 8's P14, on the qwen3nativebest200 box: over steps 101..200, axolotl scattermoe native-best / e4b shipped lies
    within P14_BAND. Each side needs two VALID draws whose late medians agree within 5 %; the ratio is of the per-side medians of the late
    medians, with the interval over the four cross-draw ratios. Outside the band FALSIFIED, an unstable / missing / non-VALID side UNTESTED.
    The ordering reading rides along: the whole interval above 1.0 = e4b shipped faster, below 1.0 = axolotl faster, else no ordering."""
    R = F.get(NB200_FAM)
    if not R:
        return []
    nat = NATIVE_BY_FAM[NB200_FAM]
    recs = {(x["fw"], x["tag"]): x["r"] for x in R["rows"]}
    sides = {}
    for fw in ("e4b", "axolotl"):
        k = (fw, nat[fw])
        k2 = DRAW2[k]
        vs = [R["verdicts"].get(k, "missing"), R["verdicts"].get(k2, "missing")]
        if vs != ["VALID", "VALID"]:
            return [("P14", NB200_FAM, "UNTESTED", f"{fw} draws {k[1]} {vs[0]} / {k2[1]} {vs[1]}: two VALID draws a side are registered")]
        late = [late_median(recs[k]), late_median(recs[k2])]
        if None in late:
            return [("P14", NB200_FAM, "UNTESTED", f"{fw} draws carry no step_ms over steps {LATE_FROM}..N")]
        stab = abs(late[0] - late[1]) / statistics.mean(late)
        sides[fw] = {"late": late, "stab": stab, "s": statistics.median(late)}
        if stab > STABILITY.get(fw, STABILITY_OTHER):
            return [("P14", NB200_FAM, "UNTESTED", f"{fw} late-window draws {late[0]:.3f} / {late[1]:.3f} s/step differ by {100 * stab:.1f}% > 5% (UNSTABLE)")]
    e, a = sides["e4b"], sides["axolotl"]
    r = a["s"] / e["s"]
    cross = [x / y for x in a["late"] for y in e["late"]]
    lo, hi = min(cross), max(cross)
    order = "e4b shipped faster" if lo > 1.0 else ("axolotl scattermoe faster" if hi < 1.0 else "no ordering at this resolution (the interval spans 1.0)")
    held = P14_BAND[0] <= r <= P14_BAND[1]
    return [("P14", NB200_FAM, "HELD" if held else "FALSIFIED",
             f"axolotl scattermoe / e4b shipped over steps {LATE_FROM}..200 {r:.3f} [{lo:.3f}, {hi:.3f} over 4 cross-draw ratios] vs {list(P14_BAND)}; "
             f"late medians e4b {e['late'][0]:.3f} / {e['late'][1]:.3f} (within {100 * e['stab']:.1f}%), axolotl {a['late'][0]:.3f} / {a['late'][1]:.3f} "
             f"(within {100 * a['stab']:.1f}%); ordering: {order}")]


def score_syncab(F):
    """TC1-PREREG amendment 10 (#945), on the qwen3syncab box: P16 (shipped arm) and P17 (matched arm) -- the single-read grouping with the
    pinned ring steps at sync1 / legacy within SYNC_BAND, the median over two VALID draws a side with each side's draws within 5 %. Outside
    the band FALSIFIED; an unstable, missing or non-VALID side UNTESTED. The interval is over the four cross-draw ratios."""
    R = F.get(SYNC_FAM)
    if not R:
        return []
    out = []
    for pid, name, t in SYNC_PAIRS:
        L, N = R["draws"].get(("e4b", f"{t}_legacy"), {}), R["draws"].get(("e4b", f"{t}_sync1"), {})
        if not (L.get("usable") and N.get("usable") and L.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("legacy", L), ("sync1", N)))
            out.append((pid, SYNC_FAM, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            continue
        ratio_ = N["s"] / L["s"]
        cross = [n / l for n in N["s_list"] for l in L["s_list"]]
        held = SYNC_BAND[0] <= ratio_ <= SYNC_BAND[1]
        out.append((pid, SYNC_FAM, "HELD" if held else "FALSIFIED",
                    f"{name}: sync1 / legacy {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {list(SYNC_BAND)}; "
                    f"s/step legacy {L['s_list'][0]:.3f} / {L['s_list'][1]:.3f} (within {100 * L['stability']:.1f}%), sync1 {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} "
                    f"(within {100 * N['stability']:.1f}%); held-out at N legacy {f(L.get('heldout'), 4)} / sync1 {f(N.get('heldout'), 4)}"))
    return out


def score_leanab(F):
    """TC1-PREREG amendment 13 (#945), on the qwen3leanab box: P20 (shipped arm) and P21 (matched arm) -- gnf4's trimmed padded LoRA
    delta steps at lean1 / lean0 within LEAN_BAND, the median over two VALID draws a side with each side's draws within 5 %. Outside the
    band FALSIFIED (the evidence names a stable ratio above LEAN_REVERT_ABOVE, which the decision rule acts on); an unstable, missing or
    non-VALID side UNTESTED. The interval is over the four cross-draw ratios."""
    R = F.get(LEAN_FAM)
    if not R:
        return []
    out = []
    for pid, name, t in LEAN_PAIRS:
        O, N = R["draws"].get(("e4b", f"{t}_lean0"), {}), R["draws"].get(("e4b", f"{t}_lean1"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("lean0", O), ("lean1", N)))
            out.append((pid, LEAN_FAM, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        held = LEAN_BAND[0] <= ratio_ <= LEAN_BAND[1]
        r1 = next((x["r"] for x in R["rows"] if (x["fw"], x["tag"]) == ("e4b", f"{t}_lean1")), None) or {}
        calls = (r1.get("lean_ab") or {}).get("lora_path_calls") or {}
        out.append((pid, LEAN_FAM, "HELD" if held else "FALSIFIED",
                    f"{name}: lean1 / lean0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {list(LEAN_BAND)}"
                    f"{'; ABOVE ' + str(LEAN_REVERT_ABOVE) + ' -- the decision rule reverts the default' if ratio_ > LEAN_REVERT_ABOVE else ''}; "
                    f"s/step lean0 {O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), lean1 {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} "
                    f"(within {100 * N['stability']:.1f}%); held-out at N lean0 {f(O.get('heldout'), 4)} / lean1 {f(N.get('heldout'), 4)}; "
                    f"delta paths (lean1, process) {json.dumps(calls, sort_keys=True)}"))
    return out


def score_tileab(F):
    """TC1-PREREG amendment 14 (#945), on the qwen3tileab box: P22 (shipped arm) and P23 (matched arm) -- gnf4's cost tile rule steps at
    tilecost / tilemax within TILE_BANDS[pid], the median over two VALID draws a side with each side's draws within 5 %. Outside the band
    FALSIFIED; an unstable, missing or non-VALID side UNTESTED. The evidence names the decision rule's reading (flip / keep / no effect)
    and each side's launched tile heights."""
    R = F.get(TILE_FAM)
    if not R:
        return []
    out = []
    for pid, name, t in TILE_PAIRS:
        O, N = R["draws"].get(("e4b", f"{t}_tilemax"), {}), R["draws"].get(("e4b", f"{t}_tilecost"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("tilemax", O), ("tilecost", N)))
            out.append((pid, TILE_FAM, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        lo, hi = TILE_BANDS[pid]
        rule = ("flip-eligible" if ratio_ <= TILE_FLIP_AT_OR_BELOW else "KEEP max" if ratio_ > TILE_KEEP_ABOVE else "no measurable effect")
        bms = {}
        for side in ("tilemax", "tilecost"):
            r1 = next((x["r"] for x in R["rows"] if (x["fw"], x["tag"]) == ("e4b", f"{t}_{side}")), None) or {}
            bms[side] = (r1.get("tile_ab") or {}).get("prefill_bm_launches") or {}
        out.append((pid, TILE_FAM, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name}: tilecost / tilemax {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}; decision reading: {rule}; "
                    f"s/step tilemax {O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), tilecost {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} "
                    f"(within {100 * N['stability']:.1f}%); held-out at N tilemax {f(O.get('heldout'), 4)} / tilecost {f(N.get('heldout'), 4)}; "
                    f"tile heights launched (process) tilemax {json.dumps(bms['tilemax'], sort_keys=True)} tilecost {json.dumps(bms['tilecost'], sort_keys=True)}"))
    return out


def prof945_table(F):
    """Amendment 12, descriptive: each profiled arm's summary (device busy fraction, device events and CPU ops per step, CPU self by family)."""
    R = F.get(PROF945_FAM)
    if not R:
        return []
    out = []
    for k in PROF945_ARMS:
        r = next((x["r"] for x in R["rows"] if (x["fw"], x["tag"]) == k), None) or {}
        pr = r.get("profile") or {}
        out.append({"arm": k[1], "verdict": R["verdicts"].get(k, "missing"), "s_per_step_profiled_run": r.get("s_per_step_median_11plus"),
                    "device_busy_fraction": pr.get("device_busy_fraction"), "device_events_per_step": pr.get("device_events_per_step"),
                    "cpu_ops_per_step": pr.get("cpu_ops_per_step"), "cpu_self_by_family_fraction": pr.get("cpu_self_by_family_fraction")})
    return out


def score_prof945(F):
    """P19: the matched arm's device busy fraction with the single-read grouping + pinned ring is at least the legacy arm's + P19_MIN_GAIN,
    both arms VALID with a profile summary. Otherwise FALSIFIED; a missing / non-VALID / unprofiled arm UNTESTED."""
    rows = {t["arm"]: t for t in prof945_table(F)}
    if not rows:
        return []
    new, old = rows.get("fused_attn4_m_prof") or {}, rows.get("fused_attn4_m_prof_legacy") or {}
    if new.get("verdict") != "VALID" or old.get("verdict") != "VALID" or new.get("device_busy_fraction") is None or old.get("device_busy_fraction") is None:
        return [("P19", PROF945_FAM, "UNTESTED", f"matched new {new.get('verdict')} busy {new.get('device_busy_fraction')}; legacy {old.get('verdict')} busy {old.get('device_busy_fraction')}")]
    gain = new["device_busy_fraction"] - old["device_busy_fraction"]
    return [("P19", PROF945_FAM, "HELD" if gain >= P19_MIN_GAIN else "FALSIFIED",
             f"device busy fraction matched new {new['device_busy_fraction']:.3f} vs legacy {old['device_busy_fraction']:.3f} ({gain:+.3f}; >= +{P19_MIN_GAIN} predicted); "
             f"device events/step {new.get('device_events_per_step')} vs {old.get('device_events_per_step')}; CPU ops/step {new.get('cpu_ops_per_step')} vs {old.get('cpu_ops_per_step')}")]


# ----------------------------------------------------------------------------- R6: predictions (TC1-PREREG.md, scored mechanically)
def score_predictions(F):
    R = F.get("qwen3")
    AX = F.get(AX_FAM)                               # amendment 3: P6 reads the axolotl box when it ran
    out = []
    if not R:
        base = [(f"P{i}", "qwen3", "UNTESTED", "no receipts") for i in range(1, 11)]
        if AX:
            base[5] = _score_p6(AX_FAM, AX["verdicts"], AX["positions"], AX["verdicts"])
        return base
    vd = R["verdicts"]
    pos = R["positions"]
    dr = R["draws"]
    NAT = F.get("qwen3native") or R                 # I: the labelled / native rows live on their own box when it ran
    ndr, nvd = NAT["draws"], NAT["verdicts"]

    def vof(fw, tag):
        return vd.get((fw, tag), "missing")

    def nvof(fw, tag):
        return nvd.get((fw, tag), "missing")
    # P1 matched position Unsloth/e4b in [2.0, 5.0]; below 1.5 refutes the standing position
    pu = pos.get("unsloth", {})
    if pu.get("quoted"):
        r = pu["ratio"]
        held = 2.0 <= r <= 5.0
        out.append(("P1", "qwen3", "HELD" if held else "FALSIFIED",
                    f"matched unsloth/e4b {r:.3f} vs [2.0, 5.0]" + ("" if held else ("; BELOW 1.5: the standing position is refuted and superseded" if r < 1.5 else "; outside the band"))))
    else:
        out.append(("P1", "qwen3", "UNTESTED", "no quoted matched position: " + pu.get("why", "no pair")))
    # P1b (phase 2): the field-image row (tp4's torch-2.8 venv, loader-default backend) within 15 % of the quoted tp4 s/step
    t = ndr.get(("unsloth", "ckpt_unsloth_t28"), {})
    if t.get("usable"):
        dev = t["s"] / TP4_T28_S_PER_STEP - 1
        t28 = next((x["r"] for x in NAT["rows"] if (x["fw"], x["tag"]) == ("unsloth", "ckpt_unsloth_t28")), None) or {}
        out.append(("P1b", NAT["fam"], "HELD" if abs(dev) <= 0.15 else "FALSIFIED", f"t28 {t['s']:.3f} s/step vs {TP4_T28_S_PER_STEP} ({dev:+.1%}; within 15 % predicted); backend selected {t28.get('moe_backend_selected')}"))
    else:
        out.append(("P1b", NAT["fam"], "UNTESTED", f"ckpt_unsloth_t28 {nvof('unsloth', 'ckpt_unsloth_t28')}: " + t.get("why", "no receipt")))
    # P2 draws agree within 5 % (e4b) / 10 % (Unsloth)
    ev = []
    for k in (QUALITY_ANCHOR, ("unsloth", PRIMARY["unsloth"])):
        d = dr.get(k, {})
        if d.get("draws") == 2 and d.get("stability") is not None:
            ev.append((k[0], d["stability"], d["threshold"], d["verdict"]))
    if not ev:
        out.append(("P2", "qwen3", "UNTESTED", "no arm with two VALID draws: " + "; ".join(f"{k[0]} {dr.get(k, {}).get('why', 'no receipt')}" for k in (QUALITY_ANCHOR, ("unsloth", PRIMARY["unsloth"])))))
    else:
        bad = [e for e in ev if e[3] != "STABLE"]
        out.append(("P2", "qwen3", "HELD" if not bad else "FALSIFIED", "; ".join(f"{fw} |d1-d2|/mean {100 * s:.1f}% vs {100 * t:.0f}% -> {v}" for fw, s, t, v in ev)))
    # P3 the matched set is EQUIVALENT: e4b reference_m and unsloth_m against e4b fused_m (I); INSIDE-DRAW-NOISE is as equivalent as the box can tell (G)
    eq = {k: R["equivalence"].get(k, {}).get("reading", "—") for k in (REFERENCE, ("unsloth", PRIMARY["unsloth"]))}
    ok_read = ("EQUIVALENT", "INSIDE-DRAW-NOISE")
    if any(v in ("—", "N-A") for v in eq.values()):
        out.append(("P3", "qwen3", "UNTESTED", "; ".join(f"{k[0]}/{k[1]} {v} ({R['equivalence'].get(k, {}).get('why', 'no receipt')})" for k, v in eq.items())))
    else:
        held = all(v in ok_read for v in eq.values())
        out.append(("P3", "qwen3", "HELD" if held else "FALSIFIED",
                    "; ".join(f"{k[0]}/{k[1]} {v} (median step |Δ| {f(R['equivalence'][k].get('med_train'), 4)}, |Δ held-out at N| {f(R['equivalence'][k].get('d_heldout'), 4)}, band {R['equivalence'][k].get('band')})" for k, v in eq.items())
                    + (f"; draw-noise floor {f(R.get('noise_floor'), 4)}" if R.get("noise_floor") is not None else "")
                    + ("" if held else ("; DIVERGENT: no position is quoted from this lane (decision rule)" if any(v == "DIVERGENT" for v in eq.values()) else "; COMPARABLE, not EQUIVALENT, and not DIVERGENT: the decision rule does not fire"))))
    # P4 shipped is 1.05-1.3x faster per step than fused_m and its held-out at N is lower by >= 0.01 (within the native box)
    sh, fm = ndr.get(("e4b", NATIVE["e4b"]), {}), ndr.get(QUALITY_ANCHOR, {})
    if sh.get("usable") and fm.get("usable") and sh.get("heldout") is not None and fm.get("heldout") is not None:
        x = fm["s"] / sh["s"] if sh["s"] else None
        gain = fm["heldout"] - sh["heldout"]
        held = x is not None and 1.05 <= x <= 1.3 and gain >= 0.01
        out.append(("P4", NAT["fam"], "HELD" if held else "FALSIFIED", f"fused_m/shipped s/step {f(x)} vs [1.05, 1.3]; held-out matched {fm['heldout']:.4f} - shipped {sh['heldout']:.4f} = {gain:+.4f} (>= 0.01 predicted)"))
    else:
        out.append(("P4", NAT["fam"], "UNTESTED", f"shipped {nvof('e4b', NATIVE['e4b'])} / fused_m {nvof(*QUALITY_ANCHOR)}: both must be VALID on the same box"))
    # P5 HF OOMs at the primary recipe and at mb1
    hp, hm = vof("hf", PRIMARY["hf"]), vof("hf", SECONDARY["hf"])
    if hp == "missing" and hm == "missing":
        out.append(("P5", "qwen3", "UNTESTED", "no hf receipts"))
    elif hp == "OOM" and hm == "OOM":
        out.append(("P5", "qwen3", "HELD", "hf_peft_m OOM and hf_peft_m_mb1 OOM"))
    elif hp in ("VALID", "QUALITY_FAIL", "VOID") or hm in ("VALID", "QUALITY_FAIL", "VOID"):
        out.append(("P5", "qwen3", "FALSIFIED", f"hf_peft_m {hp}, hf_peft_m_mb1 {hm}: HF trained at least one recipe"))
    else:
        out.append(("P5", "qwen3", "UNTESTED", f"hf_peft_m {hp}, hf_peft_m_mb1 {hm}: neither an OOM pair nor a trained arm"))
    # P6 axolotl (amendment 3: read on the axolotl box when it ran; the judged box otherwise)
    out.append(_score_p6(AX_FAM, AX["verdicts"], AX["positions"], AX["verdicts"]) if AX else _score_p6("qwen3", vd, pos, nvd))
    # P7 unsloth_best at most 1.5x faster than unsloth_m, and e4b_m >= 2x faster than unsloth_best (best and e4b_m from the native box; unsloth_m from the judged box)
    ub, um, em = ndr.get(("unsloth", NATIVE["unsloth"]), {}), dr.get(("unsloth", PRIMARY["unsloth"]), {}), ndr.get(QUALITY_ANCHOR, {})
    if ub.get("usable") and um.get("usable") and em.get("usable"):
        gain_x, e_x = um["s"] / ub["s"], ub["s"] / em["s"]
        out.append(("P7", NAT["fam"], "HELD" if (gain_x <= 1.5 and e_x >= 2.0) else "FALSIFIED", f"unsloth_m/unsloth_best {gain_x:.3f} (<= 1.5 predicted; across boxes when the native family ran apart); unsloth_best/e4b_m {e_x:.3f} (>= 2 predicted, within its box)"))
    else:
        out.append(("P7", NAT["fam"], "UNTESTED", f"unsloth_best {nvof('unsloth', NATIVE['unsloth'])} / unsloth_m {vof('unsloth', PRIMARY['unsloth'])} / e4b_m {nvof(*QUALITY_ANCHOR)}: all three must be VALID"))
    # P8 (phase 2): device busy fraction >= 0.5 on the profiled grouped_mm arm
    pr = R.get("profile")
    req = (R.get("prof_knobs") or {}).get("moe_backend_requested")
    if pr and pr.get("device_busy_fraction") is not None and req == "grouped_mm":
        out.append(("P8", "qwen3", "HELD" if pr["device_busy_fraction"] >= P8_BUSY_MIN else "FALSIFIED", f"device_busy_fraction {pr['device_busy_fraction']:.3f} (>= {P8_BUSY_MIN} predicted on the grouped_mm arm); device events/step {pr.get('device_events_per_step')}"))
    elif pr and pr.get("device_busy_fraction") is not None:
        out.append(("P8", "qwen3", "UNTESTED", f"the profiled arm requested backend {req!r}, not grouped_mm (device_busy_fraction {pr['device_busy_fraction']:.3f} recorded)"))
    else:
        out.append(("P8", "qwen3", "UNTESTED", f"ckpt_unsloth_prof {vof(*PROF)} carries no profile summary"))
    # P9 e4b internal parity PASS
    pv = R["parity"]["verdict"]
    out.append(("P9", "qwen3", {"PASS": "HELD", "FAIL": "FALSIFIED"}.get(pv, "UNTESTED"), f"fused_m vs reference_m {pv}" + (f" (Δfinal {R['parity']['d_final']:.5f}, median {R['parity']['median']:.5f})" if pv in ("PASS", "FAIL") else f" {R['parity']['why']}")))
    # P10 C1 bit-exact in every OK arm; expert slots SAME-BYTES e4b vs Unsloth; attention slot not SAME-BYTES
    oks = [x for x in R["rows"] if x["status"] == "OK"]
    c1 = [f"{x['fw']}/{x['tag']}" for x in oks if not c1_ok(x["r"])]
    fb = R["frozen"].get(("unsloth", PRIMARY["unsloth"]))
    if not oks:
        out.append(("P10", "qwen3", "UNTESTED", "no OK arm"))
    elif c1:
        out.append(("P10", "qwen3", "FALSIFIED", "C1 not bit-exact on " + ", ".join(c1)))
    elif not fb:
        out.append(("P10", "qwen3", "UNTESTED", "C1 bit-exact on every OK arm; no unsloth_m probe to read the slots against e4b"))
    else:
        ex = [fb[s]["reading"] for s in ("gate_up", "down")]
        at = fb["q_proj"]["reading"]
        if "N-A" in ex:
            out.append(("P10", "qwen3", "UNTESTED", f"C1 bit-exact; expert slots {ex} ({fb['gate_up'].get('why') or fb['down'].get('why')}); attention {at}"))
        else:
            held = all(v == "SAME-BYTES" for v in ex) and at != "SAME-BYTES"
            out.append(("P10", "qwen3", "HELD" if held else "FALSIFIED", f"C1 bit-exact on every OK arm; expert slots e4b vs unsloth {ex}; attention {at} ({fb['q_proj'].get('why') or fb['q_proj'].get('regime')})"))
    return out



# ----------------------------------------------------------------------------- R11: the TC2 predictions (TC2-PREREG-draft "Predictions"), scored mechanically
def _in(x, band):
    return x is not None and band[0] <= x <= band[1]


def _row(R, fw, tag):
    return next((x for x in R["rows"] if (x["fw"], x["tag"]) == (fw, tag)), None)


def _attention_only(R, fw, tag):
    """An Unsloth arm that adapted attention only: VOID with the attention-only reason, or OK with experts 0 in its trainable groups."""
    x = _row(R, fw, tag)
    if x is None:
        return False
    g = ((x["r"] or {}).get("trainable_by_group") or {})
    return (x["verdict"] == "VOID" and "attention-only" in (x["why"] or "")) or (x["status"] == "OK" and g.get("experts") == 0)


def score_tc2_predictions(F):
    out = []
    G, OL, GO, Q, MX = (F.get(fam) for fam in TC2_FAMS)

    def vof(R, fw, tag):
        return R["verdicts"].get((fw, tag), "missing") if R else "missing"
    # P1 granite: Unsloth adapts attention only on both target lists -> UNSUPPORTED for the 4-bit MoE regime; HF/e4b in [1.1, 1.6]; axolotl ~ HF or UNSUPPORTED (no band in the draft: printed, not scored)
    if not G:
        out.append(("P1", "granite", "UNTESTED", "no receipts"))
    else:
        legs, bad, untested = [], [], []
        for tag in ("ckpt_unsloth_m", "ckpt_unsloth_m_experts"):
            v = vof(G, "unsloth", tag)
            if v in ("UNSUPPORTED", "OOM") or _attention_only(G, "unsloth", tag):
                legs.append(f"unsloth/{tag} {v}" + (" attention-only" if _attention_only(G, "unsloth", tag) else ""))
            elif v in ("VALID", "QUALITY_FAIL", "VOID"):
                bad.append(f"unsloth/{tag} {v} and NOT attention-only (it adapted expert parameters)")
            else:
                untested.append(f"unsloth/{tag} {v}")
        ph = G["positions"].get("hf", {})
        if ph.get("quoted"):
            legs.append(f"HF/e4b {ph['ratio']:.3f} vs {TC2_P1_HF_BAND}")
            if not _in(ph["ratio"], TC2_P1_HF_BAND):
                bad.append(f"HF/e4b {ph['ratio']:.3f} outside {TC2_P1_HF_BAND}")
        else:
            untested.append("HF position not quoted: " + ph.get("why", "no pair"))
        pa = G["positions"].get("axolotl", {})
        ax = f"axolotl/e4b {pa['ratio']:.3f} (vs HF, unscored: the draft gives no band for ~)" if pa.get("quoted") else f"axolotl {vof(G, 'axolotl', 'ckpt_axolotl_m')}"
        out.append(("P1", "granite", "FALSIFIED" if bad else ("UNTESTED" if untested else "HELD"), "; ".join(legs + bad + untested) + "; " + ax))
    # P2 olmoe: Unsloth engages (ratio in [1.2, 3]) or dies again (HARNESS_ERROR / UNSUPPORTED); HF/e4b in [1.5, 2.5] and COMPARABLE
    if not OL:
        out.append(("P2", "olmoe", "UNTESTED", "no receipts"))
    else:
        legs, bad, untested = [], [], []
        pu, vu = OL["positions"].get("unsloth", {}), vof(OL, "unsloth", "ckpt_unsloth_m")
        if pu.get("quoted"):
            legs.append(f"Unsloth engaged: unsloth/e4b {pu['ratio']:.3f} vs {TC2_P2_UNS_BAND}")
            if not _in(pu["ratio"], TC2_P2_UNS_BAND):
                bad.append(f"unsloth/e4b {pu['ratio']:.3f} outside {TC2_P2_UNS_BAND}")
        elif vu in ("UNSUPPORTED", "HARNESS_ERROR"):
            legs.append(f"Unsloth died again: {vu}")
        elif vu in ("VALID", "QUALITY_FAIL", "VOID", "OOM"):
            bad.append(f"unsloth/ckpt_unsloth_m {vu} but no quoted position: " + pu.get("why", ""))
        else:
            untested.append(f"unsloth/ckpt_unsloth_m {vu}")
        ph = OL["positions"].get("hf", {})
        if ph.get("quoted"):
            legs.append(f"HF/e4b {ph['ratio']:.3f} vs {TC2_P2_HF_BAND}, quality {ph.get('quality')}")
            if not _in(ph["ratio"], TC2_P2_HF_BAND) or ph.get("quality") != "COMPARABLE":
                bad.append(f"HF leg: ratio {ph['ratio']:.3f} / quality {ph.get('quality')} (COMPARABLE within {TC2_P2_HF_BAND} predicted)")
        else:
            untested.append("HF position not quoted: " + ph.get("why", "no pair"))
        out.append(("P2", "olmoe", "FALSIFIED" if bad else ("UNTESTED" if untested else "HELD"), "; ".join(legs + bad + untested)))
    # P3 gptoss: Unsloth's packed-MXFP4 arm trains the experts (VALID, packed class recorded); e4b attn-only VALID; both s/step reported, never a ratio
    if not GO:
        out.append(("P3", "gptoss", "UNTESTED", "no receipts"))
    else:
        vm, va = vof(GO, "unsloth", "ckpt_unsloth_mxfp4"), vof(GO, "e4b", "attn_only_m")
        xm = _row(GO, "unsloth", "ckpt_unsloth_mxfp4")
        n_packed, epc = packed_expert_params((xm or {}).get("r") or {}) if xm else (0, {})
        line = GO["positions"].get("unsloth_mxfp4", {})
        if vm == "VALID" and va == "VALID":
            out.append(("P3", "gptoss", "HELD", f"unsloth/ckpt_unsloth_mxfp4 VALID with packed expert parameters {epc}; e4b/attn_only_m VALID; "
                        f"reported without a ratio: e4b {f(line.get('e4b_s'))} s/step ({line.get('e4b_trainable')} trainable) vs Unsloth packed {f(line.get('other_s'))} s/step ({line.get('other_trainable')} trainable)"))
        elif vm in ("UNSUPPORTED", "OOM", "VOID", "QUALITY_FAIL"):
            out.append(("P3", "gptoss", "FALSIFIED", f"unsloth/ckpt_unsloth_mxfp4 {vm}: " + ((xm["why"] or xm["reason"]) if xm else "") + f"; e4b/attn_only_m {va}"))
        else:
            out.append(("P3", "gptoss", "UNTESTED", f"unsloth/ckpt_unsloth_mxfp4 {vm}, e4b/attn_only_m {va}"))
    # P4 qwen3_5: Unsloth with explicit expert names engages the routed experts (ratio in [2, 6]) or VOIDs on trainable count again; HF OOM; e4b fused_m within 15 % of tp4's 6.3344
    if not Q:
        out.append(("P4", "qwen3_5", "UNTESTED", "no receipts"))
    else:
        legs, bad, untested = [], [], []
        k = ("unsloth", "ckpt_unsloth_m_experts")
        lp, vx = Q["labelled"].get(k, {}), vof(Q, *k)
        if lp.get("quoted"):
            legs.append(f"Unsloth (explicit expert targets) engaged: unsloth/e4b {lp['ratio']:.3f} vs {TC2_P4_UNS_BAND}")
            if not _in(lp["ratio"], TC2_P4_UNS_BAND):
                bad.append(f"ratio {lp['ratio']:.3f} outside {TC2_P4_UNS_BAND}")
        elif vx == "VOID" and "trainable" in (_row(Q, *k) or {}).get("why", ""):
            legs.append(f"unsloth/ckpt_unsloth_m_experts VOID on trainable count again: {_row(Q, *k)['why'][:120]}")
        elif vx in ("VALID", "QUALITY_FAIL", "UNSUPPORTED", "OOM", "VOID"):
            bad.append(f"unsloth/ckpt_unsloth_m_experts {vx} (neither an engaged ratio nor a trainable-count VOID): " + (lp.get("why") or (_row(Q, *k) or {}).get("why") or ""))
        else:
            untested.append(f"unsloth/ckpt_unsloth_m_experts {vx}")
        vh = vof(Q, "hf", "hf_peft_m")
        if vh == "OOM":
            legs.append("HF OOM")
        elif vh in ("VALID", "QUALITY_FAIL", "VOID", "UNSUPPORTED"):
            bad.append(f"hf/hf_peft_m {vh} (OOM predicted)")
        else:
            untested.append(f"hf/hf_peft_m {vh}")
        de = Q["draws"].get(QUALITY_ANCHOR, {})
        if de.get("usable"):
            dev = de["s"] / TP4_QWEN3_5_E4B_S_PER_STEP - 1
            legs.append(f"e4b fused_m {de['s']:.3f} s/step vs tp4's {TP4_QWEN3_5_E4B_S_PER_STEP} ({dev:+.1%}; within {TC2_P4_E4B_TOL:.0%} predicted)")
            if abs(dev) > TC2_P4_E4B_TOL:
                bad.append(f"e4b fused_m {dev:+.1%} from tp4's {TP4_QWEN3_5_E4B_S_PER_STEP}")
        else:
            untested.append("e4b fused_m not usable: " + de.get("why", "no receipt"))
        out.append(("P4", "qwen3_5", "FALSIFIED" if bad else ("UNTESTED" if untested else "HELD"), "; ".join(legs + bad + untested)))
    # P5 mixtral: e4b (offload) / Unsloth (resident) in [0.3, 0.5] (tp2 0.361) at a >= 8x lower e4b peak; HF and axolotl OOM resident
    if not MX:
        out.append(("P5", "mixtral", "UNTESTED", "no receipts"))
    else:
        legs, bad, untested = [], [], []
        pu, fp = MX["positions"].get("unsloth", {}), MX.get("footprint") or {}
        if pu.get("quoted"):
            px = ratio(pu.get("peak_other"), pu.get("peak_e4b"))
            legs.append(f"Unsloth(resident)/e4b(offload) s/step {pu['ratio']:.3f} vs {TC2_P5_BAND} (the lane's other/e4b convention; tp2 {TP2_MIXTRAL['ratio_unsloth_over_e4b']} = 0.858 / 2.377); "
                        f"Unsloth peak / e4b peak {f(px, 2)} (>= {TC2_P5_PEAK_X:.0f} predicted; tp2 {TP2_MIXTRAL['peak_unsloth_gb']} / {TP2_MIXTRAL['peak_e4b_gb']} GB)")
            if not _in(pu["ratio"], TC2_P5_BAND):
                bad.append(f"Unsloth/e4b {pu['ratio']:.3f} outside {TC2_P5_BAND}")
            if px is None or px < TC2_P5_PEAK_X:
                bad.append(f"peak ratio {f(px, 2)} < {TC2_P5_PEAK_X:.0f}")
        else:
            untested.append("mixtral position not quoted: " + pu.get("why", "no pair") + (f" (footprint: {footprint_line(fp)[2:120]})" if fp else ""))
        for fw, tag in (("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m")):
            v = vof(MX, fw, tag)
            if v == "OOM":
                legs.append(f"{fw} OOM")
            elif v in ("VALID", "QUALITY_FAIL", "VOID", "UNSUPPORTED"):
                bad.append(f"{fw}/{tag} {v} (OOM predicted)")
            else:
                untested.append(f"{fw}/{tag} {v}")
        out.append(("P5", "mixtral", "FALSIFIED" if bad else ("UNTESTED" if untested else "HELD"), "; ".join(legs + bad + untested)))
    # P6 e4b internal parity PASS on every family with both arms
    ev, fails = [], []
    for fam in TC2_FAMS:
        R = F.get(fam)
        if not R:
            continue
        pv = R["parity"]["verdict"]
        if pv in ("PASS", "FAIL"):
            ev.append(f"{fam} {pv} (Δfinal {R['parity']['d_final']:.5f})")
            if pv == "FAIL":
                fails.append(fam)
    if not ev:
        out.append(("P6", "tc2", "UNTESTED", "no family with both e4b arms OK"))
    else:
        out.append(("P6", "tc2", "FALSIFIED" if fails else "HELD", "; ".join(ev)))
    # P7 matched sets EQUIVALENT where two frameworks train the SAME adapter set (VALID, trainable == e4b's, against a VALID e4b anchor)
    ev, bad = [], []
    for fam in TC2_FAMS:
        R = F.get(fam)
        if not R or not R.get("common_set", True):
            continue
        for k, e in R["equivalence"].items():
            if k[0] == "e4b" or R["verdicts"].get(k) != "VALID" or R["verdicts"].get(R["anchor_key"]) != "VALID":
                continue
            ev.append(f"{fam} {k[0]}/{k[1]} {e['reading']}")
            if e["reading"] not in ("EQUIVALENT", "INSIDE-DRAW-NOISE"):
                bad.append(f"{fam} {k[0]}/{k[1]} {e['reading']}")
    if not ev:
        out.append(("P7", "tc2", "UNTESTED", "no VALID matched pair sharing e4b's adapter set"))
    else:
        out.append(("P7", "tc2", "FALSIFIED" if bad else "HELD", "; ".join(ev)))
    return out


# ----------------------------------------------------------------------------- R10: the TC1b readings (the qwen3curve family)
def rows_at(r, step):
    """The per-row held-out losses an arm recorded at `step` (eval_rows), or None."""
    e = next((e for e in ((r or {}).get("eval_rows") or []) if e.get("step") == step), None)
    return list(e["losses"]) if e and e.get("losses") else None


def mean_se(v):
    n = len(v)
    return statistics.mean(v), ((statistics.stdev(v) / (n ** 0.5)) if n > 1 else None)


def paired_delta(a, b):
    """Paired per-row differences a - b: {n, mean, se, abs, favouring_a, favouring_b}; None unless both sides carry the same row count."""
    if not a or not b or len(a) != len(b):
        return None
    d = [x - y for x, y in zip(a, b)]
    m, se = mean_se(d)
    return {"n": len(d), "mean": m, "se": se, "abs": abs(m), "favouring_a": sum(1 for v in d if v < 0), "favouring_b": sum(1 for v in d if v > 0)}


def curve_table(recs, V):
    """(a): at every eval step of the first VALID curve arm's grid, mean +- SE over the rows for each VALID arm 1-3 and the paired
    deltas (2 - 1) and (3 - 1). A non-VALID arm's cells are None (VOID never enters a reading)."""
    k1, k2, k3 = CURVE_ARMS
    ok = {k: V.get(k) == "VALID" for k in CURVE_ARMS}
    src = next((recs[k] for k in CURVE_ARMS if ok[k]), None)
    if src is None:
        return []
    out = []
    for step in [c["step"] for c in src.get("eval_curve") or []]:
        rows = {k: (rows_at(recs.get(k), step) if ok[k] else None) for k in CURVE_ARMS}
        row = {"step": step, "arms": {k: (mean_se(rows[k]) if rows[k] else None) for k in CURVE_ARMS},
               "d12": paired_delta(rows[k2], rows[k1]), "d31": paired_delta(rows[k3], rows[k1])}
        out.append(row)
    return out


def curve_reading(table, V, tc1=None):
    """The curve reading for the matched pair: the largest paired |delta(1,2)| over the evals and its sign at step N;
    EQUIVALENT-AT-EVERY-EVAL iff every |delta| <= CURVE_BAND, else DIVERGENT with the first divergent step. Sign: + = Unsloth's held-out ABOVE e4b's.
    TC1's floor (EQUIV_FLOOR, and TC1's measured held-out band when `tc1` is a reduced TC1 dir) rides beside the fixed band, never replaces it."""
    k1, k2, _ = CURVE_ARMS
    tb = ((tc1 or {}).get("qwen3") or {}).get("equiv_band") if tc1 else None
    floor = {"equiv_floor": EQUIV_FLOOR, "tc1_heldout_band": (tb or {}).get("heldout") if tb else None,
             "text": f"TC1's floor max({EQUIV_FLOOR}, 3 × fused-vs-reference |Δ|)" + (f" = {tb['heldout']:.4f} held-out from --tc1-dir" if tb else (" (no --tc1-dir: the measured band is not to hand)" if tc1 is None else " (the TC1 dir carries no reference arm: band unset)"))}
    v1, v2 = V.get(k1), V.get(k2)
    if v1 != "VALID" or v2 != "VALID":
        return {"reading": "N-A", "why": f"e4b/{k1[1]} {v1 or 'missing'} / unsloth/{k2[1]} {v2 or 'missing'}: both must be VALID"}
    ds = [(row["step"], row["d12"]) for row in table]
    if not ds:
        return {"reading": "N-A", "why": "no eval grid"}
    missing = [st for st, d in ds if d is None]
    if missing:
        return {"reading": "N-A", "why": f"paired rows missing or unequal in count at step(s) {missing[:4]}"}
    if ds[-1][0] != CURVE_N[k1[1]]:
        return {"reading": "N-A", "why": f"the eval grid ends at step {ds[-1][0]}, not {CURVE_N[k1[1]]}"}
    mx_step, mx = max(ds, key=lambda t: t[1]["abs"])
    at_n = ds[-1][1]["mean"]
    first = next((st for st, d in ds if d["abs"] > CURVE_BAND), None)
    return {"reading": "EQUIVALENT-AT-EVERY-EVAL" if first is None else "DIVERGENT", "max_abs": mx["abs"], "max_step": mx_step,
            "delta_at_N": at_n, "sign_at_N": "+" if at_n > 0 else ("-" if at_n < 0 else "0"), "first_divergent_step": first,
            "n_evals": len(ds), "band": CURVE_BAND, "band_note": CURVE_BAND_NOTE, "tc1_floor": floor, "why": ""}


def plateau_test(recs, V):
    """(b): held-out(as-shipped) - held-out(matched e4b) at step 200 and at step 40; REPRODUCES-P38 iff >= +0.01 at 200 and <= 0 at 40."""
    k1, _, k3 = CURVE_ARMS
    v1, v3 = V.get(k1), V.get(k3)
    if v1 != "VALID" or v3 != "VALID":
        return {"reading": "N-A", "why": f"e4b/{k1[1]} {v1 or 'missing'} / e4b/{k3[1]} {v3 or 'missing'}: both must be VALID"}
    N = CURVE_N[k1[1]]
    r1, r3 = recs[k1], recs[k3]
    hN, h40 = (curve_at(r3, N), curve_at(r1, N)), (curve_at(r3, PLATEAU_EARLY_STEP), curve_at(r1, PLATEAU_EARLY_STEP))
    if None in hN or None in h40:
        return {"reading": "N-A", "why": f"no held-out at step {N} or {PLATEAU_EARLY_STEP} on one side"}
    gN, g40 = hN[0] - hN[1], h40[0] - h40[1]
    ok = gN >= PLATEAU_GAP and g40 <= 0
    why = "" if ok else (f"as-shipped ends {gN:+.4f} vs the matched arm at {N}: within {PLATEAU_GAP} or below (no plateau above)" if gN < PLATEAU_GAP
                         else f"as-shipped is {g40:+.4f} ABOVE the matched arm at step {PLATEAU_EARLY_STEP} too: no early lead, not P38's shape")
    return {"reading": "REPRODUCES-P38" if ok else "NOT-P38-SHAPE", "gap_N": gN, "gap_early": g40, "N": N, "early": PLATEAU_EARLY_STEP,
            "paired_N": paired_delta(rows_at(r3, N), rows_at(r1, N)), "paired_early": paired_delta(rows_at(r3, PLATEAU_EARLY_STEP), rows_at(r1, PLATEAU_EARLY_STEP)), "why": why}


def time_to_target(recs, V):
    """(c): the target = the matched pair's step-200 held-out (mean over the VALID matched arms) + TARGET_MARGIN; per arm 1-3 the first
    eval at or below it (step, train_wall_s), or not reached."""
    k1, k2, _ = CURVE_ARMS
    have = [(k, heldout_at_N(recs[k])) for k in (k1, k2) if V.get(k) == "VALID" and heldout_at_N(recs.get(k)) is not None]
    if not have:
        return {"target": None, "why": "no VALID matched arm with a held-out at step 200", "arms": {}}
    target = statistics.mean([h for _, h in have]) + TARGET_MARGIN
    basis = ("mean of the matched pair's step-200 held-out" if len(have) == 2 else f"the one VALID matched arm's step-200 held-out ({have[0][0][0]}/{have[0][0][1]})") + f" + {TARGET_MARGIN}"
    arms = {}
    for k in CURVE_ARMS:
        if V.get(k) != "VALID":
            arms[k] = {"reached": None, "why": f"{V.get(k) or 'missing'}"}
            continue
        hit = next((c for c in (recs[k].get("eval_curve") or []) if c.get("heldout_loss") is not None and c["heldout_loss"] <= target), None)
        arms[k] = {"reached": hit is not None, "step": hit["step"] if hit else None, "train_wall_s": hit.get("train_wall_s") if hit else None,
                   "heldout": hit["heldout_loss"] if hit else None,
                   "why": "" if hit else f"not reached by step {recs[k].get('steps')} (final held-out {f(heldout_at_N(recs[k]), 4)})"}
    return {"target": target, "basis": basis, "arms": arms}


def speed_travel(recs, V, draws, tc1=None):
    """(d): each curve arm's 11..200 median (the receipt's s_per_step_median_11plus: steady = step_ms[10:]) beside the in-receipt 11..20
    window and, when `tc1` (a reduced TC1 dir) is given, TC1's quoted 11..20 median for its counterpart; TRAVELS iff within TRAVEL_TOL."""
    out = {}
    for k in CURVE_ARMS:
        r, d = recs.get(k), draws.get(k, {})
        e = {"verdict": V.get(k), "usable": bool(d.get("usable")), "s_200": r.get("s_per_step_median_11plus") if is_ok(r) else None, "reading": "UNTESTED"}
        sm = (r.get("step_ms") if is_ok(r) else None) or []
        e["s_11_20_in_receipt"] = (statistics.median(sm[10:20]) / 1e3) if len(sm) >= 20 else None
        if e["s_11_20_in_receipt"] and e["s_200"]:
            e["in_receipt_dev"] = e["s_200"] / e["s_11_20_in_receipt"] - 1
        fam, key = CURVE_TC1_COUNTERPART[k]
        if tc1 is None:
            e["tc1_why"] = "no --tc1-dir given"
        else:
            TF = tc1.get(fam)
            td = (TF or {}).get("draws", {}).get(key, {})
            if td.get("usable"):
                e.update({"tc1_s": td["s"], "tc1_draws": td.get("draws"), "tc1_key": (fam, key)})
            else:
                e["tc1_why"] = f"{fam} {key[0]}/{key[1]}: " + ((td.get("why") or td.get("verdict") or "no receipt") if TF else f"no {fam} receipts in --tc1-dir")
        if e["usable"] and e.get("tc1_s"):
            e["dev"] = e["s_200"] / e["tc1_s"] - 1
            e["reading"] = "TRAVELS" if abs(e["dev"]) <= TRAVEL_TOL else "DOES-NOT-TRAVEL"
        elif not e["usable"]:
            e["why"] = d.get("why") or f"{V.get(k) or 'missing'}"
        out[k] = e
    return out


def anchor_reading(draws):
    """(e): the p38 pair's s/step ratio unsloth/e4b vs tp2's 1.457 and P38's 1.413 (+-ANCHOR_TOL), the cu130/grouped_mm arm and the t28
    arm (tp4's, byte-for-byte) read separately against the same e4b p38 arm; the tp4 leg only when TP4_ANCHOR_RATIO is set."""
    e = draws.get(CURVE_ANCHOR["e4b"], {"usable": False, "why": "no e4b/fused_attn4_p38 receipt"})

    def one(key, label):
        pos = position(e, draws.get(key, {"usable": False, "why": "no receipt"}), label)
        pos["kind"] = "anchor"
        if pos.get("quoted"):
            r = pos["ratio"]
            pos.update({"tp2": TP2_ANCHOR, "p38": P38_ANCHOR, "dev_tp2": r / TP2_ANCHOR - 1, "dev_p38": r / P38_ANCHOR - 1})
            pos["agrees_tp2"], pos["agrees_p38"] = abs(pos["dev_tp2"]) <= ANCHOR_TOL, abs(pos["dev_p38"]) <= ANCHOR_TOL
            if TP4_ANCHOR_RATIO:
                pos["tp4"], pos["dev_tp4"] = TP4_ANCHOR_RATIO, r / TP4_ANCHOR_RATIO - 1
                pos["agrees_tp4"] = abs(pos["dev_tp4"]) <= ANCHOR_TOL
            pos["reading"] = "AGREES" if pos["agrees_tp2"] else ("AGREES-P38-ONLY" if pos["agrees_p38"] else "FINDING")
        return pos
    return {"grouped_mm": one(CURVE_ANCHOR["unsloth"], "unsloth p38 (venv-unsloth cu130 torch 2.12.1, grouped_mm: tp4's arm EXCEPT the venv)"),
            "t28": one(CURVE_ANCHOR["t28"], "unsloth p38_t28 (venv-unsloth-t28, loader-default backend: tp4's arm byte-for-byte)")}


def scaling_point(group, recs, V, draws):
    """(f): the t1 / r64 pair's s/step ratio under the matched set's predicates, every one listed: both arms VALID, the same trainable count
    WITHIN the pair, lora_path_loop == 0 on every e4b step, the Unsloth backend counters. Labelled a scaling point, never a position."""
    ke, ku = CURVE_GROUPS[group], ("unsloth", f"ckpt_unsloth_m_{group}")
    pos = position(draws.get(ke, {"usable": False, "why": "no receipt"}), draws.get(ku, {"usable": False, "why": "no receipt"}), CURVE_SCALING[group])
    pos["kind"] = "scaling point"
    re_, ru = recs.get(ke) or {}, recs.get(ku) or {}
    L, A = N_LAYERS[CURVE_FAM], int(ru.get("accum") or 1)
    te, tu = re_.get("trainable_params"), ru.get("trainable_params")
    g, m = ru.get("unsloth_grouped_mm_calls_per_step_min"), ru.get("unsloth_manual_grouped_mm_calls_per_step_max")
    preds = [("both arms VALID", V.get(ke) == "VALID" and V.get(ku) == "VALID", f"e4b {V.get(ke) or 'missing'} / unsloth {V.get(ku) or 'missing'}"),
             ("same trainable count within the pair", te is not None and te == tu, f"{te} / {tu}"),
             ("lora_path_loop == 0 on every e4b step", bool(re_.get("lora_path_present")) and not re_.get("lora_path_loop_steps"), f"loop steps {re_.get('lora_path_loop_steps')}"),
             (f"Unsloth backend counters: torch._grouped_mm >= {GMM_FACTOR}*L*A per step, manual fallback 0",
              g is not None and m is not None and g >= GMM_FACTOR * L * A and m == 0, f"grouped_mm min {g} (>= {GMM_FACTOR * L * A}), manual max {m}")]
    pos["predicates"] = preds
    failed = [name for name, ok, _ in preds if not ok]
    if failed and pos.get("quoted"):
        pos["quoted"], pos["why"] = False, "not quoted: " + "; ".join(f"{n} FAILS ({ev})" for n, ok, ev in preds if not ok)
    elif failed:
        pos["why"] = (pos.get("why") or "not quoted") + "; predicates failing: " + ", ".join(failed)
    return pos


def reduce_curve_family(fam, recs, rcs_all, tc1=None):
    """R10: the qwen3curve family, validity per sub-fixture, then readings (a)-(f)."""
    exp = list(EXPECTED[CURVE_FAM])
    keys = exp + [k for k in recs if k not in exp]
    rows, V, groups = [], {}, {}
    for g, ak in CURVE_GROUPS.items():
        ga = recs.get(ak)
        groups[g] = {"anchor": ak, "anchor_ok": is_ok(ga), "tokens_sha": (ga.get("tokens") or {}).get("sha256") if is_ok(ga) else None,
                     "trainable": ga.get("trainable_params") if is_ok(ga) else None, "step0": ga.get("eval_loss_step0") if is_ok(ga) else None,
                     "seed": matched_seed_of(ga) if is_ok(ga) else None,
                     "sha": (ga.get("matched_init_sha") or None) if (is_ok(ga) and matched_seed_of(ga) is not None) else None}
    for fw, tag in keys:
        r = recs.get((fw, tag))
        g = CURVE_GROUP.get(tag)
        G = groups.get(g) or {}
        matched = tag in MATCHED
        st, reason = status_of(r, rcs_all.get((fam, fw, tag)))
        v, why = validity(fam, r, G.get("tokens_sha"), (G.get("trainable") if fw != "e4b" else None), CURVE_N.get(tag),
                          matched=matched, ref_step0=(G.get("step0") if matched else None), anchor_seed=G.get("seed"),
                          is_ref=((fw, tag) == G.get("anchor")), anchor_sha=G.get("sha"))
        if is_ok(r) and g in ("200", "p38", "t1"):                    # the registered counts (draft "Validity"); r64 only within its pair
            want = CURVE_TRAINABLE["200" if g == "t1" else g]
            if r.get("trainable_params") != want:
                v, why = "VOID", (why + "; " if why else "") + f"trainable {r.get('trainable_params')} != the registered {want} for the {g} sub-fixture (TC1B draft 'Validity')"
        d0 = abs(r["eval_loss_step0"] - G["step0"]) if (is_ok(r) and matched and G.get("step0") is not None and r.get("eval_loss_step0") is not None) else None
        rows.append({"fw": fw, "tag": tag, "status": st, "reason": reason, "validity": v, "why": why, "r": r, "group": g,
                     "regime": regime_of(fam, r) if (r and st == "OK") else None, "matched": matched, "step0_delta": d0, "step0_class": step0_class(d0)})
        V[(fw, tag)] = v
    verdicts = {}
    for x in rows:                                     # quality against the sub-fixture's own e4b arm (VALID by predicates), then the verdict
        r, G = x["r"], groups.get(x["group"]) or {}
        ak = G.get("anchor")
        ha = heldout_at_N(recs.get(ak)) if (ak and V.get(ak) == "VALID") else None
        q = (heldout_at_N(r) - ha) if (x["status"] == "OK" and ha is not None and heldout_at_N(r) is not None and (x["fw"], x["tag"]) != ak) else None
        x["quality_delta"] = q
        x["quality"] = ("COMPARABLE" if abs(q) <= READ else "FLAGGED") if q is not None else (("anchor of its sub-fixture" if (x["fw"], x["tag"]) == ak else "N-A (its e4b arm " + ("VOID" if (ak and is_ok(recs.get(ak))) else "missing") + ")") if x["status"] == "OK" else None)
        x["verdict"] = verdict_of(x["status"], x["validity"], q)
        assert x["verdict"] in VERDICTS
        verdicts[(x["fw"], x["tag"])] = x["verdict"]
    draws = {k: draws_of(recs, verdicts, k, expected=exp) for k in keys}
    table = curve_table(recs, V)
    equiv = {}                                         # TC1's R4 reading at N on each matched pair, informational (no reference arm here: COMPARABLE is the ceiling)
    for g in ("200", "t1", "r64"):
        ke, ku = CURVE_GROUPS[g], ("unsloth", f"ckpt_unsloth_m_{g}")
        if recs.get(ku) is not None:
            equiv[ku] = equivalence(recs.get(ke), recs[ku], V.get(ke), V.get(ku), band=None, noise_floor=None)
    src = next((recs[k] for k in keys if is_ok(recs.get(k))), None) or next((recs[k] for k in keys if recs.get(k)), None)
    return {"fam": fam, "rows": rows, "V": V, "verdicts": verdicts, "draws": draws, "groups": groups, "curve_table": table,
            "curve": curve_reading(table, V, tc1=tc1), "plateau": plateau_test(recs, V), "ttt": time_to_target(recs, V),
            "speed": speed_travel(recs, V, draws, tc1=tc1), "anchor": anchor_reading(draws),
            "scaling": {g: scaling_point(g, recs, V, draws) for g in ("t1", "r64")}, "equivalence": equiv,
            "tc1_given": tc1 is not None, "src": src, "N": None, "e4b_trainable": groups["200"].get("trainable"), "tokens_sha": groups["200"].get("tokens_sha")}


def score_curve_predictions(F):
    """TC1b-PREREG-draft "Predictions" P1-P4, scored mechanically on the qwen3curve family."""
    R = F.get(CURVE_FAM)
    if not R:
        return [(f"P{i}", CURVE_FAM, "UNTESTED", "no qwen3curve receipts") for i in range(1, 5)]
    out = []
    c = R["curve"]
    if c["reading"] == "EQUIVALENT-AT-EVERY-EVAL":
        out.append(("P1", CURVE_FAM, "HELD", f"max paired |Δ(1,2)| {c['max_abs']:.4f} at step {c['max_step']} over {c['n_evals']} evals (<= {CURVE_BAND}: {CURVE_BAND_NOTE}); Δ at 200 {c['delta_at_N']:+.4f} (sign {c['sign_at_N']}, + = Unsloth above)"))
    elif c["reading"] == "DIVERGENT":
        out.append(("P1", CURVE_FAM, "FALSIFIED", f"DIVERGENT: first paired |Δ(1,2)| > {CURVE_BAND} at step {c['first_divergent_step']}; max {c['max_abs']:.4f} at step {c['max_step']}; sign at 200 {c['sign_at_N']} (Δ {c['delta_at_N']:+.4f}, + = Unsloth above); no training position on this family until TC4 explains it (decision rule)"))
    else:
        out.append(("P1", CURVE_FAM, "UNTESTED", c.get("why", "")))
    pl = R["plateau"]
    if pl["reading"] == "REPRODUCES-P38":
        out.append(("P2", CURVE_FAM, "HELD", f"as-shipped − matched held-out {pl['gap_N']:+.4f} at {pl['N']} (>= {PLATEAU_GAP}) and {pl['gap_early']:+.4f} at {pl['early']} (<= 0): P38's shape, with init scale / bf16 adapters now isolated on e4b's side; load_moe_4bit_streaming's adapter defaults become an open item (decision rule)"))
    elif pl["reading"] == "NOT-P38-SHAPE":
        out.append(("P2", CURVE_FAM, "FALSIFIED", f"gap at {pl['N']} {pl['gap_N']:+.4f}, at {pl['early']} {pl['gap_early']:+.4f}: {pl['why']}"))
    else:
        out.append(("P2", CURVE_FAM, "UNTESTED", pl.get("why", "")))
    sp = R["speed"]
    tested = {k: e for k, e in sp.items() if e["reading"] in ("TRAVELS", "DOES-NOT-TRAVEL")}
    if not tested:
        out.append(("P3", CURVE_FAM, "UNTESTED", "; ".join(f"{k[0]}/{k[1]}: {e.get('tc1_why') or e.get('why') or e['reading']}" for k, e in sp.items())))
    else:
        bad = [k for k, e in tested.items() if e["reading"] == "DOES-NOT-TRAVEL"]
        ev = "; ".join(f"{k[0]}/{k[1]} 11..200 {e['s_200']:.3f} vs TC1 11..20 {e['tc1_s']:.3f} ({e['dev']:+.1%}) {e['reading']}" for k, e in tested.items())
        un = [k for k in sp if k not in tested]
        out.append(("P3", CURVE_FAM, "FALSIFIED" if bad else "HELD", ev + (f"; untested: {', '.join(k[1] for k in un)}" if un else "")))
    an = R["anchor"]["grouped_mm"]
    if an.get("quoted"):
        if TP4_ANCHOR_RATIO:
            out.append(("P4", CURVE_FAM, "HELD" if an["agrees_tp4"] else "FALSIFIED", f"p38 unsloth/e4b {an['ratio']:.3f} vs tp4's anchor row {TP4_ANCHOR_RATIO} ({an['dev_tp4']:+.1%}; +-{ANCHOR_TOL:.0%}); tp2 {TP2_ANCHOR} ({an['dev_tp2']:+.1%}), P38 {P38_ANCHOR} ({an['dev_p38']:+.1%})"))
        else:
            out.append(("P4", CURVE_FAM, "HELD" if an["agrees_tp2"] else "FALSIFIED",
                        f"p38 unsloth/e4b {an['ratio']:.3f} vs tp2 {TP2_ANCHOR} ({an['dev_tp2']:+.1%}) and P38 {P38_ANCHOR} ({an['dev_p38']:+.1%}), +-{ANCHOR_TOL:.0%}; tp4's own anchor row is NOT in this tree (TP4_ANCHOR_RATIO unset, UNVERIFIED) -- scored against tp2's 1.457, the number tp4 registered against"
                        + (f"; t28 variant {R['anchor']['t28']['ratio']:.3f} ({R['anchor']['t28']['dev_tp2']:+.1%} vs tp2)" if R["anchor"]["t28"].get("quoted") else "; t28 variant not quoted")))
    else:
        out.append(("P4", CURVE_FAM, "UNTESTED", an.get("why", "no anchor pair")))
    return out


# ----------------------------------------------------------------------------- R11: lane TC3 (the frontier tokens)
def fit_table(rows):
    """(a) the FIT TABLE: per framework every arm with its lever, VERDICT, peak VRAM, host-RAM high-water, s/step, J/step and regime; `fits` = at least one arm
    COMPLETED on this box (status OK: VALID, VOID or QUALITY_FAIL -- validity is its own column), the completed arms named; an OOM-only column reads so.
    The OOM rows carry the peak VRAM and host RAM they reached."""
    out = {}
    for fw in FW:
        arms = [x for x in rows if x["fw"] == fw]
        if not arms:
            continue
        done = [x for x in arms if x["status"] == "OK"]
        out[fw] = {"arms": [{"tag": x["tag"], "lever": x["lever"], "verdict": x["verdict"], "status": x["status"], "validity": x["validity"],
                             "peak_vram_gb": (x["r"] or {}).get("peak_vram_gb"), "host_ram_high_water_gb": (x["r"] or {}).get("host_ram_high_water_gb"),
                             "host_ram_total_gb": (x["r"] or {}).get("host_ram_total_gb"), "s": (x["r"] or {}).get("s_per_step_median_11plus"),
                             "j": (x["r"] or {}).get("joules_per_step"), "regime": x["regime"], "note": x["reason"] or x["why"]} for x in arms],
                   "fits": bool(done), "completed": [x["tag"] for x in done],
                   "reading": ("FITS on this box: " + ", ".join(f"`{x['tag']}` [{x['lever']}] {x['verdict']}" for x in done)) if done
                   else ("NO ARM COMPLETED on this box: " + ", ".join(f"`{x['tag']}` {x['verdict']}" for x in arms))}
    return out


def resident_comparison(anchor, v_anchor, tc1):
    """(b) with --tc1-dir: the frontier box's e4b offload arm against TC1's RESIDENT e4b/fused_attn4_m -- the same tokens (sha), the same init (the name-free
    matched sha), the same precision (fp32-only adapters) and the same N asserted FIRST, then the median per-step |delta train| <= RESIDENT_BAND reads
    EQUIVALENT-TO-RESIDENT, else DIVERGENT-FROM-RESIDENT; |delta held-out at N| and the step-0 class beside. TC1's resident s/step (its quoted draws when
    usable) and peak VRAM are carried as MEASUREMENTS for P4; no ratio is formed across the two boxes."""
    if tc1 is None:
        return {"reading": "UNTESTED", "why": "no --tc1-dir given"}
    fam_, key = RESIDENT_KEY
    T = (tc1 or {}).get(fam_)
    res = T.get("e") if T else None
    if not is_ok(res):
        return {"reading": "UNTESTED", "why": f"no OK {fam_} {key[0]}/{key[1]} receipt in the TC1 dir"}
    dq = (T.get("draws") or {}).get(key) or {}
    tc1_s = dq.get("s") if dq.get("usable") else res.get("s_per_step_median_11plus")
    carry = {"tc1_s": tc1_s, "tc1_draws": dq.get("draws"), "tc1_peak": res.get("peak_vram_gb"), "tc1_box": (res.get("env") or {}).get("box_class")}
    if not is_ok(anchor):
        return {"reading": "N-A", "why": "the box's offload anchor e4b/fused_attn4_m_offload is missing or not OK", **carry}
    if v_anchor != "VALID":
        return {"reading": "N-A", "why": f"the box's offload anchor is {v_anchor} (VOID never enters an equivalence reading)", **carry}
    vres = (T.get("V") or {}).get(key)
    if vres != "VALID":
        return {"reading": "N-A", "why": f"TC1's resident arm is {vres}", **carry}
    pre = []
    if (anchor.get("tokens") or {}).get("sha256") != (res.get("tokens") or {}).get("sha256"):
        pre.append("tokens sha differs")
    if not anchor.get("matched_init_sha") or anchor.get("matched_init_sha") != res.get("matched_init_sha"):
        pre.append("matched_init_sha differs (not the same init)")
    for nm, r in (("offload", anchor), ("resident", res)):
        if list((r.get("adapter_dtypes_after") or {}).keys()) != ["torch.float32"]:
            pre.append(f"{nm} adapters are not fp32-only")
    if len(anchor.get("losses") or []) != len(res.get("losses") or []) or not anchor.get("losses"):
        pre.append("step counts differ")
    if pre:
        return {"reading": "N-A", "why": "not the matched trajectory: " + "; ".join(pre), **carry}
    med = statistics.median(abs(x - y) for x, y in zip(anchor["losses"], res["losses"]))
    hr, hs = heldout_at_N(anchor), heldout_at_N(res)
    dh = abs(hr - hs) if (hr is not None and hs is not None) else None
    d0 = abs(anchor["eval_loss_step0"] - res["eval_loss_step0"]) if (anchor.get("eval_loss_step0") is not None and res.get("eval_loss_step0") is not None) else None
    return {"reading": "EQUIVALENT-TO-RESIDENT" if med <= RESIDENT_BAND else "DIVERGENT-FROM-RESIDENT", "med_train": med, "d_heldout": dh, "d_step0": d0,
            "step0_class": step0_class(d0), "band": RESIDENT_BAND, "offload_s": anchor.get("s_per_step_median_11plus"), "offload_peak": anchor.get("peak_vram_gb"), "why": "", **carry}


def reduce_frontier_family(fam, recs, rcs_all, n_steps=None, tc1=None):
    """R11: one frontier token. Validity per arm as TC1's (tp4's predicates + the matched-set predicates against the box's OWN e4b offload anchor, since the
    resident e4b arm is expected to OOM here; the reference-under-offload arm stands in when the fused offload arm did not complete), the VERDICT (quality
    against that anchor), the draws (the 12 GB token registers a second offload draw), then (a) the FIT TABLE, (b) the in-box equivalence of every matched
    arm against the anchor under the fixed R4 bands with tp1's parity on the fused/reference pair, and the resident comparison when --tc1-dir is given."""
    exp = list(EXPECTED[fam])
    keys = exp + [k for k in recs if k not in exp]
    anchor, ref = recs.get(FRONTIER_ANCHOR), recs.get(FRONTIER_REF)
    anchor_key = next((k for k in (FRONTIER_ANCHOR, FRONTIER_REF) + FRONTIER12_SECONDARY if is_ok(recs.get(k))), None)   # the mb1 secondary stands in on the 12 GB token
    A = recs.get(anchor_key) if anchor_key else None
    e4b_trainable = A.get("trainable_params") if A else None
    tokens_sha = ((A.get("tokens") or {}).get("sha256") if A else None) or next((r["tokens"]["sha256"] for k in keys if (r := recs.get(k)) and (r.get("tokens") or {}).get("sha256")), None)
    N = n_steps or next((r.get("steps") for r in recs.values() if r and r.get("steps")), None)
    ref_step0 = A.get("eval_loss_step0") if A else None
    anchor_seed = matched_seed_of(A) if A else None
    anchor_sha = (A.get("matched_init_sha") or None) if (A and anchor_seed is not None) else None
    rows, V = [], {}
    for fw, tag in keys:
        r = recs.get((fw, tag))
        st, reason = status_of(r, rcs_all.get((fam, fw, tag)))
        v, why = validity(fam, r, tokens_sha, (e4b_trainable if fw != "e4b" else None), N, matched=(tag in MATCHED), ref_step0=ref_step0,
                          anchor_seed=anchor_seed, is_ref=((fw, tag) == anchor_key), anchor_sha=anchor_sha)
        d0 = abs(r["eval_loss_step0"] - ref_step0) if (is_ok(r) and tag in MATCHED and ref_step0 is not None and r.get("eval_loss_step0") is not None) else None
        rows.append({"fw": fw, "tag": tag, "status": st, "reason": reason, "validity": v, "why": why, "r": r, "regime": regime_of(fam, r) if (r and st == "OK") else None,
                     "matched": tag in MATCHED, "step0_delta": d0, "step0_class": step0_class(d0),
                     "lever": FRONTIER_LEVER.get(tag) or (r or {}).get("memory_lever") or "—"})
        V[(fw, tag)] = v
    ha = heldout_at_N(A) if (A and V.get(anchor_key) == "VALID") else None
    verdicts = {}
    for x in rows:
        r, q = x["r"], None
        if x["status"] == "OK" and ha is not None and heldout_at_N(r) is not None and (x["fw"], x["tag"]) != anchor_key:
            q = heldout_at_N(r) - ha
        x["quality_delta"] = q
        x["quality"] = ("COMPARABLE" if abs(q) <= READ else "FLAGGED") if q is not None else \
            (("the box's anchor" if (x["fw"], x["tag"]) == anchor_key else "N-A (anchor " + ("VOID" if A else "missing") + ")") if x["status"] == "OK" else None)
        x["verdict"] = verdict_of(x["status"], x["validity"], q)
        assert x["verdict"] in VERDICTS
        verdicts[(x["fw"], x["tag"])] = x["verdict"]
    draws = {k: draws_of(recs, verdicts, k, expected=exp) for k in keys if k not in DRAW2.values()}
    fit = fit_table(rows)
    pv, d_final, med, pwhy = parity(ref, anchor)                   # tp1's rule on the in-box pair (fused offload vs reference offload)
    par = {"verdict": pv, "d_final": d_final, "median": med, "why": pwhy}
    if pv in ("PASS", "FAIL"):
        par["speed_x"] = ratio(ref.get("s_per_step_median_11plus"), anchor.get("s_per_step_median_11plus"))
        par["peak_x"] = ratio(anchor.get("peak_vram_gb"), ref.get("peak_vram_gb"))
    band = {"train": EQUIV, "heldout": EQUIV}                      # (b): TC1's R4 bands, FIXED on a frontier box (EQUIVALENT <= 0.02 on both, COMPARABLE <= 0.05)
    equiv = {}
    for fw, tag in keys:
        if tag in MATCHED and (fw, tag) != FRONTIER_ANCHOR and recs.get((fw, tag)) is not None:
            e = equivalence(anchor, recs[(fw, tag)], V.get(FRONTIER_ANCHOR), V.get((fw, tag)), band=band, noise_floor=None)
            if e["reading"] == "N-A" and not is_ok(anchor):
                e["why"] = "the box's offload anchor e4b/fused_attn4_m_offload is missing or not OK"
            equiv[(fw, tag)] = e
    resident = resident_comparison(anchor, V.get(FRONTIER_ANCHOR), tc1)
    src = next((recs[k] for k in keys if is_ok(recs.get(k))), None) or next((recs[k] for k in keys if recs.get(k)), None)
    return {"fam": fam, "rows": rows, "V": V, "verdicts": verdicts, "draws": draws, "fit": fit, "parity": par, "equivalence": equiv, "equiv_band": band,
            "resident": resident, "tc1_given": tc1 is not None, "recs": recs, "src": src, "N": N, "e4b_trainable": e4b_trainable, "tokens_sha": tokens_sha,
            "anchor_key": anchor_key, "anchor_sha": anchor_sha, "e": anchor, "ref": ref}


def _completed(v):
    return v in ("VALID", "VOID", "QUALITY_FAIL")


def _untested(v):
    return v in (None, "NOT_RUN", "HARNESS_ERROR", "ALARM")


def score_frontier_predictions(F):
    """TC3-PREREG-draft "Predictions" P1-P4, scored mechanically. 'Completed' = the arm trained to N (status OK: VALID / VOID / QUALITY_FAIL -- validity is read
    separately); OOM and UNSUPPORTED are the two ways not to; NOT_RUN / HARNESS_ERROR / ALARM leave a clause UNTESTED. P1's ratios are WITHIN the 24 GB box; P2 is
    about FRAMEWORKS (an e4b resident arm that also completed is noted, never a refutation); P3 is read per token; P4 is two measurements with the factor applied."""
    out = []
    R = F.get(FRONTIER_FAM)
    if R:
        V, rec = R["verdicts"], R["recs"]
        s_e = (rec.get(FRONTIER_ANCHOR) or {}).get("s_per_step_median_11plus") if _completed(V.get(FRONTIER_ANCHOR)) else None
        cl = []
        v = V.get(FRONTIER_ANCHOR)                                              # (i) e4b fits under offload (mb1 resident informational)
        cl.append(("i", "HELD" if _completed(v) else ("UNTESTED" if _untested(v) else "FALSIFIED"),
                   f"e4b offload {v}" + (f" at {s_e:.3f} s/step" if s_e else "") + f"; mb1 resident {V.get(('e4b', 'fused_attn4_m_mb1'))} (informational)"))
        vu, vu2 = V.get(("unsloth", "ckpt_unsloth_m")), V.get(("unsloth", "ckpt_unsloth_m_mb1"))   # (ii) Unsloth OOMs at both recipes
        if _completed(vu) or _completed(vu2):
            cl.append(("ii", "FALSIFIED", f"Unsloth completed: m {vu}, mb1 {vu2}"))
        elif vu == "OOM" and vu2 == "OOM":
            cl.append(("ii", "HELD", "Unsloth OOM at both recipes"))
        else:
            cl.append(("ii", "UNTESTED", f"Unsloth m {vu}, mb1 {vu2} (not two OOM readings)"))
        vh, vho = V.get(("hf", "hf_peft_m")), V.get(("hf", "hf_peft_m_offload"))   # (iii) HF OOMs resident; the offload arm REFUSED or > 20x slower than e4b offload
        if _completed(vh):
            cl.append(("iii-a", "FALSIFIED", f"HF resident completed ({vh})"))
        elif vh == "OOM":
            cl.append(("iii-a", "HELD", "HF resident OOM"))
        else:
            cl.append(("iii-a", "UNTESTED", f"HF resident {vh}"))
        rho = rec.get(("hf", "hf_peft_m_offload")) or {}
        if vho == "UNSUPPORTED":
            cl.append(("iii-b", "HELD", f"HF offload REFUSED ({(rho.get('reason') or '')[:90]})"))
        elif _completed(vho):
            s_h = rho.get("s_per_step_median_11plus")
            if s_h is None or s_e is None:
                cl.append(("iii-b", "UNTESTED", f"HF offload {vho} but no s/step on both sides (hf {s_h}, e4b offload {s_e})"))
            else:
                xr = s_h / s_e
                cl.append(("iii-b", "HELD" if xr > P1_HF_SLOWER else "FALSIFIED", f"HF offload RAN: {s_h:.3f} vs e4b offload {s_e:.3f} s/step = {xr:.1f}x within this box ({'>' if xr > P1_HF_SLOWER else '<='} {P1_HF_SLOWER:.0f}x)"))
        elif vho == "OOM":
            cl.append(("iii-b", "FALSIFIED", "HF offload OOM: neither REFUSED nor a measured slowdown"))
        else:
            cl.append(("iii-b", "UNTESTED", f"HF offload {vho}"))
        vz = V.get(("axolotl", "ckpt_axolotl_m_zero3"))                        # (iv) ZeRO-3, if it runs: > 5x slower than e4b offload at >= 60 GB host RAM
        rz = rec.get(("axolotl", "ckpt_axolotl_m_zero3")) or {}
        if _completed(vz):
            s_z, h_z = rz.get("s_per_step_median_11plus"), rz.get("host_ram_high_water_gb")
            if s_z is None or s_e is None or h_z is None:
                cl.append(("iv", "UNTESTED", f"axolotl zero3 {vz} but s/step or host RAM missing (s {s_z}, e4b offload {s_e}, host {h_z})"))
            else:
                xr = s_z / s_e
                ok = xr > P1_Z3_SLOWER and h_z >= P1_Z3_HOST_GB
                cl.append(("iv", "HELD" if ok else "FALSIFIED", f"axolotl zero3 RAN: {s_z:.3f} vs e4b offload {s_e:.3f} s/step = {xr:.1f}x within this box, host RAM high-water {h_z:.1f} GB "
                                                                f"(wants > {P1_Z3_SLOWER:.0f}x and >= {P1_Z3_HOST_GB:.0f} GB)"))
        elif _untested(vz):
            cl.append(("iv", "UNTESTED", f"axolotl zero3 {vz}"))
        else:
            cl.append(("iv", "HELD", f"axolotl zero3 did not run ({vz}: {(rz.get('reason') or '')[:80]}) -- a conditional clause, vacuously held"))
        vs = [c for _, c, _ in cl]
        out.append(("P1", FRONTIER_FAM, "FALSIFIED" if "FALSIFIED" in vs else ("UNTESTED" if "UNTESTED" in vs else "HELD"), "; ".join(f"({i}) {e} -> {c}" for i, c, e in cl)))
    R12 = F.get(FRONTIER12_FAM)
    if R12:
        V, rec = R12["verdicts"], R12["recs"]
        e4b_off = [("e4b", "fused_attn4_m_offload"), ("e4b", "fused_attn4_m_offload_d2"), ("e4b", "reference_attn4_m_offload")]
        sec = [k for k in FRONTIER12_SECONDARY if k in V]                       # present only when the field-recipe offload arm OOMed
        others = [k for k in EXPECTED[FRONTIER12_FAM] if k[0] != "e4b"]
        ev = ["e4b offload arms " + ", ".join(f"{k[1]} {V.get(k)}" for k in e4b_off) + (" ; mb1 secondary " + ", ".join(f"{k[1]} {V.get(k)}" for k in sec) if sec else ""),
              "other frameworks " + ", ".join(f"{k[0]}/{k[1]} {V.get(k)}" + (f" (not an OOM reading: {((rec.get(k) or {}).get('reason') or '')[:60]})" if V.get(k) == "UNSUPPORTED" else "") for k in others)]
        vres = V.get(("e4b", "fused_attn4_m"))
        ev.append(f"e4b resident {vres}" + (" (completed too: noted, P2 is about frameworks)" if _completed(vres) else ""))
        vsh = V.get(("e4b", "fused_attn4_shipped_offload"))
        if vsh is not None:
            ev.append(f"e4b as-shipped offload {vsh} (a fit row, outside P2)")
        # P2 (TC3-PREREG): HELD iff the e4b FUSED offload arm completed -- at the field recipe, or at its mb1 secondary after a field-recipe OOM -- and every
        # non-e4b arm is OOM / UNSUPPORTED; UNTESTED when a non-e4b arm, or the deciding e4b arm, is NOT_RUN / HARNESS_ERROR / ALARM; the second draw and the
        # reference-under-offload arm are reported in the evidence, never scored (a stability and a parity reading, not a fit reading)
        fused_done = _completed(V.get(("e4b", "fused_attn4_m_offload"))) or _completed(V.get(("e4b", "fused_attn4_m_offload_mb1")))
        if any(_completed(V.get(k)) for k in others):
            verdict = "FALSIFIED"
        elif any(_untested(V.get(k)) for k in others):
            verdict = "UNTESTED"
        elif fused_done:
            verdict = "HELD"
            ev.append("the e4b fused offload arm completed at " + ("the field recipe" if _completed(V.get(("e4b", "fused_attn4_m_offload"))) else "its mb1 secondary (the field recipe OOMed)"))
        elif _untested(V.get(("e4b", "fused_attn4_m_offload"))) or any(_untested(V.get(k)) for k in sec):
            verdict = "UNTESTED"
        else:
            verdict = "FALSIFIED"
            ev.append("no e4b fused offload arm completed at either recipe")
        out.append(("P2", FRONTIER12_FAM, verdict, "; ".join(ev)))
    for fam in FRONTIER_FAMS:
        Rf = F.get(fam)
        if not Rf:
            continue
        res = Rf["resident"]
        ctrl = Rf["equivalence"].get(FRONTIER_REF) or {}
        ctrl_txt = f"in-box control fused_offload vs reference_offload: {ctrl.get('reading', '—')}" + (
            f" (median step |Δ| {f(ctrl.get('med_train'), 4)}, |Δ held-out at N| {f(ctrl.get('d_heldout'), 4)})" if ctrl.get("med_train") is not None else (f" ({ctrl.get('why')})" if ctrl.get("why") else ""))
        if res["reading"] == "EQUIVALENT-TO-RESIDENT":
            out.append(("P3", fam, "HELD", f"median per-step |Δ| vs TC1's resident e4b/fused_attn4_m {res['med_train']:.4f} <= {RESIDENT_BAND} (|Δ held-out at N| {f(res.get('d_heldout'), 4)}, "
                                           f"step-0 {f(res.get('d_step0'), 4)} {res.get('step0_class') or ''}); {ctrl_txt}"))
        elif res["reading"] == "DIVERGENT-FROM-RESIDENT":
            out.append(("P3", fam, "FALSIFIED", f"median per-step |Δ| vs TC1's resident e4b/fused_attn4_m {res['med_train']:.4f} > {RESIDENT_BAND}; {ctrl_txt}"))
        else:
            out.append(("P3", fam, "UNTESTED", f"resident comparison {res['reading']}: {res.get('why', '')}; {ctrl_txt}"))
    if R:
        res = R["resident"]
        s_e = res.get("offload_s")
        if res["reading"] in ("EQUIVALENT-TO-RESIDENT", "DIVERGENT-FROM-RESIDENT") and res.get("tc1_s") is not None and s_e is not None:
            lim = P4_FACTOR * res["tc1_s"]
            box = ((R["e"] or {}).get("env") or {}).get("box_class")
            out.append(("P4", FRONTIER_FAM, "HELD" if s_e <= lim else "FALSIFIED",
                        f"e4b offload on this box ({box}) {s_e:.3f} s/step; TC1's resident e4b/fused_attn4_m on {res.get('tc1_box')} {res['tc1_s']:.3f} s/step over {res.get('tc1_draws') or 1} draw(s); "
                        f"{P4_FACTOR:.0f} x {res['tc1_s']:.3f} = {lim:.3f} -- two measurements, no cross-box ratio formed"))
        else:
            out.append(("P4", FRONTIER_FAM, "UNTESTED", f"resident comparison {res['reading']}: {res.get('why', '')}" if res["reading"] not in ("EQUIVALENT-TO-RESIDENT", "DIVERGENT-FROM-RESIDENT")
                        else "TC1's resident s/step or this box's offload s/step unavailable"))
    return out


# ----------------------------------------------------------------------------- the printer
def pos_lines(pos, N, prefix="POSITION"):
    if pos.get("no_common_set"):                                            # R11: gpt-oss -- both values, both counts, never a ratio
        return [f"- **NO COMMON ADAPTER SET ({pos['label']} vs {pos['e4b_label']}) — no ratio is quoted**: e4b attention-only {f(pos.get('e4b_s'))} s/step "
                f"({pos.get('e4b_trainable') if pos.get('e4b_trainable') is not None else '—'} trainable; {pos['e4b_state']}) vs {pos['label']} {f(pos.get('other_s'))} s/step "
                f"({pos.get('other_trainable') if pos.get('other_trainable') is not None else '—'} trainable; {pos['other_state']}); peak VRAM e4b {f(pos.get('e4b_peak'), 2)} / "
                f"{pos['label']} {f(pos.get('other_peak'), 2)} GB — {pos['why']}"]
    if not pos.get("quoted"):
        return [f"- **NO {prefix} QUOTED ({pos['label']})** — {pos['why']}"]
    flag = "" if pos["quality"] == "COMPARABLE" else " **[QUALITY FLAGGED: not a clean position]**"
    who = pos["label"]
    sep = " / " if "(" in who else "/"                                      # R11: "HF (bf16 experts) / e4b"
    lines = [f"- **{prefix}: s/step ratio {who}{sep}e4b = {pos['ratio']:.3f}** [{f(pos.get('ratio_min'))}, {f(pos.get('ratio_max'))} over {pos.get('n_cross')} cross-draw ratios]{flag} ({pos['other_s']:.3f} vs {pos['e4b_s']:.3f} s, medians over {pos['other_draws']}/{pos['e4b_draws']} draws; "
             f"{'e4b faster' if pos['ratio'] > 1 else who + ' faster'} per step); "
             f"peak VRAM {who} {pos['peak_other']:.2f} vs e4b {pos['peak_e4b']:.2f} GB (Δ {pos['peak_delta']:+.2f}); J/step {who} {f(pos['j_other'], 1)} vs e4b {f(pos['j_e4b'], 1)} (×{f(pos['j_ratio'], 3)}); "
             f"tok/s {who} {f(pos['tok_other'], 1)} vs e4b {f(pos['tok_e4b'], 1)}" + (f"; {who} regime: {pos['other_regime']}" if pos.get("other_regime") else "")]
    lines.append(f"- quality reading at N={N} ({who}): held-out e4b {f(pos['heldout_e4b'], 4)} / {who} {f(pos['heldout_other'], 4)} (Δ {f(pos['heldout_delta'], 4)}) → **{pos['quality']}** "
                 f"(|Δ| ≤ {READ} reads COMPARABLE)" + (f"; step-0 Δ {pos['step0_delta']:+.4f}" if pos.get("step0_delta") is not None else ""))
    return lines


def support_table(rows):
    """The per-attempt support table (one shape for every family token)."""
    lines = ["| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |",
             "|" + "---|" * 20]
    for x in rows:
        r = x["r"] or {}
        b = r.get("unsloth_bnb4bit_modules")
        if x["fw"] == "e4b":
            eng = f"patched {r.get('n_patched', '—')} / kcalls {r.get('kernel_calls_per_step_min', '—')}"
        elif x["fw"] == "unsloth":
            eng = f"stacks {(r.get('census') or {}).get('Params4bit_expert_stacks', '—')} / fwd {r.get('experts_forward_calls_per_step_min', '—')} / u8 {b.get('n_bnb4bit_unwrapped') if isinstance(b, dict) else '—'}"
        else:
            ht = r.get("hf_targets") or r.get("axolotl_targets") or {}
            eng = f"peft mods {ht.get('n_target_modules', '—')} / params {ht.get('n_target_parameters', '—')} / fwd {r.get('experts_forward_calls_per_step_min', '—')}"
        mi = r.get("matched_init") or {}
        init = f"{r.get('lora_init', '—')}" + (f" ({'complete' if mi.get('complete') else 'INCOMPLETE'} {mi.get('n_slots_set')}/{mi.get('n_slots_expected')})" if mi else "") \
            + f" / {','.join(sorted(k.replace('torch.', '') for k in (r.get('adapter_dtypes_after') or {}))) or '—'}"
        note = " — ".join(s for s in (x["reason"], x["why"], x.get("dispatch"), x.get("no_common_set")) if s)      # R11: dispatch / no-common-set notes
        s0 = f"{x['step0_class']} ({x['step0_delta']:.4f})" if x.get("step0_class") else "—"
        lines.append(f"| {x['fw']} | {x['tag']} | **{x['status']}** | {x['validity']} | **{x['verdict']}** | {'yes' if x['matched'] else 'native'} | {init} | {s0} | {r.get('steps', '—')} | "
                     f"{f(r.get('s_per_step_median_11plus'))} | {f(r.get('tokens_per_s'), 1)} | {f(r.get('peak_vram_gb'))} | {f(r.get('joules_per_step'), 1)} | "
                     f"{f(r.get('loss_first'), 4)}→{f(r.get('loss_last'), 4)} | {f(r.get('eval_loss_step0'), 4)}→{f(r.get('eval_loss_final'), 4)} | "
                     f"{f(x.get('quality_delta'), 4) if x.get('quality_delta') is not None else (x.get('quality') or '—')} | {eng} | {r.get('trainable_params', '—')} | {x['regime'] or '—'} | {note} |")
    return lines


def _ms(t):
    return f"{t[0]:.4f} ± {f(t[1], 4)}" if t else "—"


def _pd(d):
    return f"{d['mean']:+.4f} ± {f(d['se'], 4)} (|Δ| {d['abs']:.4f}; rows favouring {d['favouring_a']}/{d['favouring_b']})" if d else "—"


def curve_block(R):
    """R10: the TC1b family's block -- the support table, then readings (a)-(f) in the registration's words."""
    fam = R["fam"]
    k1, k2, k3 = CURVE_ARMS
    lines = [f"\n### {NAMES.get(fam, fam)} (`{fam}`, registered n_layers {N_LAYERS.get(fam, '?')})"]
    src = R.get("src")
    if src:
        env = src.get("env", {}) or {}
        lines.append(f"- model `{src.get('model')}` @ `{str(src.get('revision', ''))[:12]}`; sub-fixtures (each with its OWN e4b arm as the trainable / tokens / step-0 / sha reference): "
                     + "; ".join(f"`{g}` anchor `{G['anchor'][0]}/{G['anchor'][1]}` {'OK' if G['anchor_ok'] else 'not OK'} tokens `{str(G.get('tokens_sha') or '')[:12]}` trainable {G.get('trainable')}" for g, G in R["groups"].items())
                     + f"; box_class {env.get('box_class')} gpu {env.get('gpu')}")
    lines += support_table(R["rows"])
    for line in prologue_lines(R["rows"]):
        lines.append(line)
    # (a) the curve table and reading
    lines.append(f"- **(a) curve table** (paired held-out mean ± SE over the rows at every eval; Δ(2−1) = `{k2[1]}` − `{k1[1]}`, Δ(3−1) = `{k3[1]}` − `{k1[1]}`; a non-VALID arm's column reads —):")
    lines.append(f"| eval step | e4b `{k1[1]}` | unsloth `{k2[1]}` | e4b `{k3[1]}` | paired Δ(2−1) ± SE | paired Δ(3−1) ± SE |")
    lines.append("|---|---|---|---|---|---|")
    for row in R["curve_table"]:
        lines.append(f"| {row['step']} | {_ms(row['arms'][k1])} | {_ms(row['arms'][k2])} | {_ms(row['arms'][k3])} | {_pd(row['d12'])} | {_pd(row['d31'])} |")
    c = R["curve"]
    if c["reading"] in ("EQUIVALENT-AT-EVERY-EVAL", "DIVERGENT"):
        lines.append(f"- **CURVE READING (arms 1 vs 2): {c['reading']}** — largest paired |Δ(1,2)| {c['max_abs']:.4f} at step {c['max_step']} over {c['n_evals']} evals; Δ at step 200 {c['delta_at_N']:+.4f} (sign {c['sign_at_N']}; + = Unsloth's held-out above e4b's)"
                     + (f"; first divergent step {c['first_divergent_step']}" if c["first_divergent_step"] is not None else "") + f"; band {c['band']} = {c['band_note']}: {c['tc1_floor']['text']}")
    else:
        lines.append(f"- **CURVE READING: {c['reading']}** — {c.get('why', '')}")
    # (b) plateau
    pl = R["plateau"]
    if pl["reading"] in ("REPRODUCES-P38", "NOT-P38-SHAPE"):
        lines.append(f"- **(b) PLATEAU TEST (arm 3 − arm 1): {pl['reading']}** — held-out gap at {pl['N']} {pl['gap_N']:+.4f} (paired {_pd(pl['paired_N'])}), at {pl['early']} {pl['gap_early']:+.4f} (paired {_pd(pl['paired_early'])}); "
                     f"REPRODUCES-P38 iff >= +{PLATEAU_GAP} at {pl['N']} and <= 0 at {pl['early']}" + (f" — {pl['why']}" if pl.get("why") else ""))
    else:
        lines.append(f"- **(b) PLATEAU TEST: {pl['reading']}** — {pl.get('why', '')}")
    # (c) time to target
    t = R["ttt"]
    if t.get("target") is not None:
        lines.append(f"- **(c) time to target** (target {t['target']:.4f} = {t['basis']}; the first eval at or below it, informational): "
                     + "; ".join(f"`{k[0]}/{k[1]}` " + (f"step {a['step']} at {a['train_wall_s']} s train wall (held-out {f(a['heldout'], 4)})" if a.get("reached") else (a.get("why") or "—")) for k, a in t["arms"].items()))
    else:
        lines.append(f"- **(c) time to target**: {t.get('why', '')}")
    # (d) speed
    lines.append(f"- **(d) s/step medians over steps 11..200** (not a position: TC1 owns positions; TRAVELS iff within {TRAVEL_TOL:.0%} of TC1's 11..20 median when `--tc1-dir` is given"
                 + (", given" if R.get("tc1_given") else ", NOT given: UNTESTED") + "; the in-receipt 11..20 window is a same-draw check): "
                 + "; ".join(f"`{k[0]}/{k[1]}` {f(e.get('s_200'))} s/step (in-receipt 11..20 {f(e.get('s_11_20_in_receipt'))}" + (f", {e['in_receipt_dev']:+.1%}" if e.get("in_receipt_dev") is not None else "") + ")"
                             + (f" vs TC1 {e['tc1_s']:.3f} over {e.get('tc1_draws')} draw(s) ({e['dev']:+.1%}) **{e['reading']}**" if e.get("tc1_s") and e["reading"] != "UNTESTED" else f" **{e['reading']}** ({e.get('tc1_why') or e.get('why') or ''})")
                             for k, e in R["speed"].items()))
    # (e) anchor
    for key, an in R["anchor"].items():
        if an.get("quoted"):
            lines.append(f"- **(e) ANCHOR {key}: s/step ratio unsloth/e4b = {an['ratio']:.3f} → {an['reading']}** ({an['label']}; {an['other_s']:.3f} vs {an['e4b_s']:.3f} s; tp2 {TP2_ANCHOR} {an['dev_tp2']:+.1%} {'AGREES' if an['agrees_tp2'] else 'outside'}, P38 {P38_ANCHOR} {an['dev_p38']:+.1%} {'AGREES' if an['agrees_p38'] else 'outside'}"
                         + (f", tp4 {an['tp4']} {an['dev_tp4']:+.1%} {'AGREES' if an['agrees_tp4'] else 'outside'}" if an.get("tp4") else ", tp4's anchor row not in this tree (UNVERIFIED)")
                         + f"; ±{ANCHOR_TOL:.0%}); quality Δ {f(an.get('heldout_delta'), 4)} {an.get('quality')}; peak Δ {an['peak_delta']:+.2f} GB")
        else:
            lines.append(f"- **(e) ANCHOR {key}: NOT QUOTED** ({an['label']}) — {an['why']}")
    # (f) scaling points
    for g, spt in R["scaling"].items():
        preds = "; ".join(f"{n} {'ok' if ok else 'FAILS'} ({ev})" for n, ok, ev in spt["predicates"])
        if spt.get("quoted"):
            lines.append(f"- **(f) SCALING POINT `{g}` (never a position): s/step ratio unsloth/e4b = {spt['ratio']:.3f}** ({spt['label']}; {spt['other_s']:.3f} vs {spt['e4b_s']:.3f} s, single draws; "
                         f"peak Δ {spt['peak_delta']:+.2f} GB; held-out Δ {f(spt.get('heldout_delta'), 4)} {spt.get('quality')}); predicates: {preds}")
        else:
            lines.append(f"- **(f) SCALING POINT `{g}`: NOT QUOTED** — {spt['why']}; predicates: {preds}")
    if R["equivalence"]:
        lines.append("- TC1's equivalence reading at N per matched pair (informational; no reference arm in this family, so COMPARABLE is the ceiling): "
                     + "; ".join(f"`{k[0]}/{k[1]}` **{e['reading']}**" + (f" (median step |Δ| {f(e.get('med_train'), 4)}, |Δ held-out at N| {f(e.get('d_heldout'), 4)}, step-0 {f(e.get('d_step0'), 4)} {e.get('step0_class') or ''})" if e.get("med_train") is not None else f" ({e.get('why', '')})")
                                 for k, e in R["equivalence"].items()))
    shas = [f"`{g}` anchor `{str(G['sha'])[:16]}`: " + ", ".join(f"{x['fw']}/{x['tag']} {'same' if (x['r'] or {}).get('matched_init_sha') == G['sha'] else 'DIFFERS'}" for x in R["rows"] if x["group"] == g and x["matched"] and x["status"] == "OK" and (x["fw"], x["tag"]) != G["anchor"])
            for g, G in R["groups"].items() if G.get("sha")]
    if shas:
        lines.append("- matched_init_sha per sub-fixture (B, name-free): " + "; ".join(shas))
    return lines


def family_block(R):
    fam = R["fam"]
    lines = [f"\n### {NAMES.get(fam, fam)} (`{fam}`, registered n_layers {N_LAYERS.get(fam, '?')}" + (f", attention census {ATTN_CENSUS[fam]}" if ATTN_CENSUS.get(fam) else "") + ")"]
    if R.get("footprint"):
        lines.append(footprint_line(R["footprint"]))                                  # R11: mixtral's footprint line leads its block

    def _ran(*cands):
        for c in cands:
            if c and is_ok(c):
                return c
        return None
    src = (_ran(R["e"], R["ref"], R["u"], R["h"], R["ax"], *[x["r"] for x in R["rows"]])
           or R["e"] or R["ref"] or R["u"] or R["h"] or next((x["r"] for x in R["rows"] if x["r"]), None))
    if src:
        env = src.get("env", {}) or {}
        lines.append(f"- model `{src.get('model')}` @ `{str(src.get('revision', ''))[:12]}`; tokens sha `{str(R['tokens_sha'] or '')[:12]}`; N={R['N']}; "
                     f"fixture template {src.get('template')} seq {src.get('seq')} micro-batch {src.get('micro_batch')} × accum {src.get('accum')} lr {src.get('lr')} r {src.get('r')} α {src.get('alpha')} "
                     f"optimizer {src.get('optimizer')} autocast {src.get('autocast')}; e4b trainable {R['e4b_trainable']}; box_class {env.get('box_class')} gpu {env.get('gpu')}")
    lines += support_table(R["rows"])
    for line in prologue_lines(R["rows"]):
        lines.append(line)
    lines.append("- draws (R1): " + "; ".join(
        f"`{k[0]}/{k[1]}` {d['verdict']}" + (f" ({d['s1']:.3f}/{d['s2']:.3f} s, |Δ|/mean {100 * d['stability']:.1f}% vs {100 * d['threshold']:.0f}%)" if d.get("draws") == 2 and d.get("stability") is not None else (f" ({d['why']})" if d.get("why") else ""))
        for k, d in R["draws"].items() if registered_draw2(fam, k) is not None or d.get("draws")))
    p = R["parity"]
    if p["verdict"] in ("PASS", "FAIL"):
        lines.append(f"- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal {p['d_final']:.5f}, median step |Δ| {p['median']:.5f} → **{p['verdict']}** "
                     f"(band {BAND}/{BAND}); ×{f(p.get('speed_x'), 2)} faster per step, peak ×{f(p.get('peak_x'), 3)}")
    else:
        lines.append(f"- e4b internal parity: {p['verdict']}{(' — ' + p['why']) if p['why'] else ''}")
    for other, pz in R["positions"].items():
        lines += pos_lines(pz, R["N"], prefix="MATCHED POSITION")
    for other, sp in R["secondary"].items():
        lines += pos_lines(sp, R["N"], prefix="SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed)")
    for other, np_ in R["native"].items():
        lines += pos_lines(np_, R["N"], prefix="NATIVE-BEST (reported beside, never instead of, the matched position)")
    for k, lp in R["labelled"].items():
        lines += pos_lines(lp, R["N"], prefix=f"LABELLED ROW {k[1]} vs e4b/fused_attn4_m (never the quoted position)")
    if R["equivalence"]:
        bd = R.get("equiv_band")
        lines.append(f"- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max({EQUIV_FLOOR}, 3 × fused-vs-reference |Δ|) = "
                     f"{('train ' + f(bd['train'], 4) + ' / held-out ' + f(bd['heldout'], 4)) if bd else 'none (no reference arm: COMPARABLE is the ceiling)'}; COMPARABLE ≤ {COMPARABLE}; "
                     f"draw-noise floor {f(R.get('noise_floor'), 4)}): "
                     + "; ".join(f"`{k[0]}/{k[1]}` **{e['reading']}**" + (f" (median step |Δ| {f(e.get('med_train'), 4)}, |Δ held-out at N| {f(e.get('d_heldout'), 4)}, step-0 {f(e.get('d_step0'), 4)} {e.get('step0_class') or ''}, |Δ loss at step 2| {f(e.get('d_loss_step2'), 4)}"
                                                                          + (f", paired rows mean {e['paired']['mean']:+.4f} ± {f(e['paired']['se'], 4)} SE over {e['paired']['n']}, favouring arm {e['paired']['rows_favouring_arm']} / anchor {e['paired']['rows_favouring_anchor']}" if e.get("paired") else "") + ")" if e.get("med_train") is not None else "")
                                 + (f" — {e['why']}" if e.get("why") else "") for k, e in R["equivalence"].items()))
        if R.get("anchor_sha"):
            lines.append(f"- matched_init_sha (B, name-free, canonical slot order): anchor `{R['anchor_sha'][:16]}`; " + "; ".join(f"`{x['fw']}/{x['tag']}` {'same' if (x['r'] or {}).get('matched_init_sha') == R['anchor_sha'] else 'DIFFERS: ' + str((x['r'] or {}).get('matched_init_sha'))[:12]}" for x in R["rows"] if x["matched"] and x["status"] == "OK"))
    if R["anchor_probe"]:
        ap = R["anchor_probe"].get("slots") or {}
        lines.append("- frozen base (R5), anchor `e4b/fused_attn4_m`: " + ", ".join(f"{s} {v.get('regime')} sha {str(v.get('sha'))[:12]} control {'detects' if v.get('control_detects_flip') else 'FAILS'}" for s, v in ap.items())
                     + ("" if R["anchor_probe"].get("control_detects_flip") else " **[anchor control did not detect its flip]**"))
    for k, fbz in R["frozen"].items():
        lines.append(f"- frozen base `{k[0]}/{k[1]}`: " + ", ".join(f"{s} **{v['reading']}**" + (f" ({v['why']})" if v.get("why") else f" ({v.get('regime')})") + f" control {'detects' if v['control'] else 'FAILS'}" for s, v in fbz.items()))
    if R["profile"]:
        pr = R["profile"]
        lines.append(f"- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction {f(pr.get('device_busy_fraction'), 3)}, device events/step {pr.get('device_events_per_step')}, "
                     f"CPU ops/step {pr.get('cpu_ops_per_step')}, CPU self by family {pr.get('cpu_self_by_family_fraction')}")
    return lines


def frontier_block(R):
    """R11: a frontier token's block -- the support table, the FIT TABLE, the draws, the in-box parity and equivalence, the resident comparison, what each lever recorded."""
    fam = R["fam"]
    lines = [f"\n### {NAMES.get(fam, fam)} (`{fam}`, registered n_layers {N_LAYERS.get(fam, '?')})"]
    src = R.get("src")
    if src:
        env, hr = src.get("env", {}) or {}, src.get("host_ram") or {}
        ak = R.get("anchor_key")
        lines.append(f"- model `{src.get('model')}` @ `{str(src.get('revision', ''))[:12]}`; tokens sha `{str(R['tokens_sha'] or '')[:12]}`; N={R['N']}; fixture template {src.get('template')} seq {src.get('seq')} "
                     f"micro-batch {src.get('micro_batch')} × accum {src.get('accum')} r {src.get('r')} α {src.get('alpha')}; e4b trainable {R['e4b_trainable']}; box_class {env.get('box_class')} gpu {env.get('gpu')}; "
                     f"host RAM total {f(src.get('host_ram_total_gb'))} GB (cgroup limit {f(hr.get('cgroup_limit_gb'))}); "
                     + (f"the box's anchor `{ak[0]}/{ak[1]}`" if ak else "NO ANCHOR: no e4b offload arm completed"))
    lines += support_table(R["rows"])
    for line in prologue_lines(R["rows"]):
        lines.append(line)
    lines.append("- **(a) FIT TABLE** (per framework: did any arm complete on this box, with its lever; peak VRAM = torch max_memory_allocated over the window (an OOM row: at the OOM); "
                 "host RAM high-water = max over the arm of the process peak RSS and the cgroup peak when it rose during the arm; s/step = the median over steps 11..N; J/step = net of idle):")
    lines.append("| framework | arm | lever | **VERDICT** | peak VRAM GB | host RAM high-water GB | s/step | J/step | regime | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for fw, Fw in R["fit"].items():
        for ar in Fw["arms"]:
            lines.append(f"| {fw} | {ar['tag']} | {ar['lever']} | **{ar['verdict']}** | {f(ar['peak_vram_gb'])} | {f(ar['host_ram_high_water_gb'])} | {f(ar['s'])} | {f(ar['j'], 1)} | {ar['regime'] or '—'} | {(ar['note'] or '')[:140]} |")
    for fw, Fw in R["fit"].items():
        lines.append(f"- fit `{fw}`: **{Fw['reading']}**")
    lines.append("- draws (R1): " + "; ".join(
        f"`{k[0]}/{k[1]}` {d['verdict']}" + (f" ({d['s1']:.3f}/{d['s2']:.3f} s, |Δ|/mean {100 * d['stability']:.1f}% vs {100 * d['threshold']:.0f}%)" if d.get("draws") == 2 and d.get("stability") is not None else (f" ({d['why']})" if d.get("why") else ""))
        for k, d in R["draws"].items() if k in DRAW2 or d.get("draws")))
    p = R["parity"]
    if p["verdict"] in ("PASS", "FAIL"):
        lines.append(f"- e4b internal parity under offload (tp1's rule): fused_attn4_m_offload vs reference_attn4_m_offload Δfinal {p['d_final']:.5f}, median step |Δ| {p['median']:.5f} → **{p['verdict']}** "
                     f"(band {BAND}/{BAND}); ×{f(p.get('speed_x'), 2)} faster per step, peak ×{f(p.get('peak_x'), 3)}")
    else:
        lines.append(f"- e4b internal parity under offload: {p['verdict']}{(' — ' + p['why']) if p['why'] else ''}")
    bd = R["equiv_band"]
    lines.append(f"- **(b) in-box equivalence** vs the box's anchor `e4b/fused_attn4_m_offload` (TC1's R4 bands, fixed: EQUIVALENT iff median step |Δ train| ≤ {bd['train']} and |Δ held-out at N| ≤ {bd['heldout']}; COMPARABLE ≤ {COMPARABLE}; "
                 f"the fused/reference pair is the control): "
                 + ("; ".join(f"`{k[0]}/{k[1]}` **{e['reading']}**" + (f" (median step |Δ| {f(e.get('med_train'), 4)}, |Δ held-out at N| {f(e.get('d_heldout'), 4)}, step-0 {f(e.get('d_step0'), 4)} {e.get('step0_class') or ''}, |Δ loss at step 2| {f(e.get('d_loss_step2'), 4)})" if e.get("med_train") is not None else "")
                              + (f" — {e['why']}" if e.get("why") else "") for k, e in R["equivalence"].items()) or "no matched arm beside the anchor"))
    rs = R["resident"]
    if rs["reading"] in ("EQUIVALENT-TO-RESIDENT", "DIVERGENT-FROM-RESIDENT"):
        lines.append(f"- **RESIDENT COMPARISON: **{rs['reading']}**** — the offload anchor vs TC1's resident `e4b/fused_attn4_m` (same tokens, init, precision, N asserted): median per-step |Δ| {rs['med_train']:.4f} vs band {rs['band']}, "
                     f"|Δ held-out at N| {f(rs.get('d_heldout'), 4)}, step-0 {f(rs.get('d_step0'), 4)} {rs.get('step0_class') or ''}; measurements beside each other, never a ratio: this box's offload {f(rs.get('offload_s'))} s/step "
                     f"at peak {f(rs.get('offload_peak'))} GB; TC1's resident on {rs.get('tc1_box')} {f(rs.get('tc1_s'))} s/step over {rs.get('tc1_draws') or 1} draw(s) at peak {f(rs.get('tc1_peak'))} GB")
    else:
        lines.append(f"- RESIDENT COMPARISON: {rs['reading']} — {rs.get('why', '')}")
    if R.get("anchor_sha"):
        lines.append(f"- matched_init_sha (B, name-free): anchor `{R['anchor_sha'][:16]}`; " + "; ".join(f"`{x['fw']}/{x['tag']}` {'same' if (x['r'] or {}).get('matched_init_sha') == R['anchor_sha'] else 'DIFFERS: ' + str((x['r'] or {}).get('matched_init_sha'))[:12]}" for x in R["rows"] if x["matched"] and x["status"] == "OK"))
    for x in R["rows"]:                                    # what each lever recorded, on the row that carries it (OK or not)
        r = x["r"] or {}
        if r.get("hf_offload"):
            ho = r["hf_offload"]
            dm = ho.get("device_map_summary") or {}
            lines.append(f"- lever `hf/{x['tag']}`: max_memory {ho.get('max_memory')}, llm_int8_enable_fp32_cpu_offload {ho.get('llm_int8_enable_fp32_cpu_offload')}; hf_device_map by device {dm.get('by_device')}, "
                         f"expert entries by device {dm.get('experts_entries_by_device')}, CPU/disk sample {dm.get('cpu_or_disk_sample')}; trained: {'yes (status OK)' if x['status'] == 'OK' else 'NO -- ' + (x['reason'] or '')[:160]}")
        if r.get("axolotl_layer_offload"):
            lo = r["axolotl_layer_offload"]
            lines.append(f"- lever `axolotl/{x['tag']}`: layer_offloading engaged {lo.get('engaged')} over {lo.get('n_layers')} layers ({lo.get('n_frozen_params_managed')} frozen params, {lo.get('n_hooks')} hooks; "
                         f"VRAM allocated {f(lo.get('vram_allocated_before_gb'))} → {f(lo.get('vram_allocated_after_gb'))} GB at setup); driven by {lo.get('driven_by')}")
        if r.get("axolotl_zero3"):
            z = r["axolotl_zero3"]
            lines.append(f"- lever `axolotl/{x['tag']}` (ZeRO-3): {x['status']} — {(x['reason'] or '')[:400]} (deepspeed {z.get('deepspeed_version')}; the config it would have used is on the receipt)")
    return lines


def render(F, d):
    out = [f"# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision ({os.path.basename(os.path.abspath(d))})",
           f"Rule ({PREREG}): status per attempt in the vocabulary {' / '.join(VOCAB)}; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, "
           f"fp32 adapters, step-0 held-out within {STEP0_TOL} of e4b/reference_attn4_m); VERDICT exactly one of {' / '.join(VERDICTS)} (QUALITY_FAIL = held-out |Δ| at N > {READ} "
           f"vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only "
           f"when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ {EQUIV} / COMPARABLE ≤ {COMPARABLE} / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; "
           f"predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these."]
    for fn in ("versions.txt", "box.json"):
        vp = os.path.join(d, fn)
        if os.path.exists(vp):
            out.append(f"`{fn}`\n```\n" + open(vp).read().strip() + "\n```")
    if CURVE_FAM in F:
        out.append(f"Lane TC1b (`{CURVE_FAM}`, TC1B-PREREG.md): validity per sub-fixture against its own e4b arm; the curve reading EQUIVALENT-AT-EVERY-EVAL iff every paired held-out |Δ| <= {CURVE_BAND} "
                   f"({CURVE_BAND_NOTE}), else DIVERGENT with the first divergent step; plateau REPRODUCES-P38 iff >= +{PLATEAU_GAP} at 200 and <= 0 at {PLATEAU_EARLY_STEP}; time to target = the matched pair's step-200 held-out + {TARGET_MARGIN}; "
                   f"s/step 11..200 vs TC1's 11..20 within {TRAVEL_TOL:.0%} = TRAVELS; anchor within ±{ANCHOR_TOL:.0%} of tp2 {TP2_ANCHOR} / P38 {P38_ANCHOR}; the t1 and r64 pairs are SCALING POINTS, never positions; P1–P4 of the draft scored HELD / FALSIFIED / UNTESTED.")
    if any(fam in F for fam in TC2_FAMS):
        out.append(f"Lane TC2 (`tc2small` / `tc2big` tokens, TC2-PREREG.md, drafted in TC2-PREREG-draft): TC1's readings per family with the registered n_layers "
                   f"{ {fam: N_LAYERS[fam] for fam in TC2_FAMS} } and attention census { {fam: ATTN_CENSUS.get(fam) for fam in TC2_FAMS} } (None = the receipt's own structural census governs); "
                   f"the HF position is quoted as HF (bf16 experts) / e4b (or 4-bit) with its regime; an Unsloth arm whose trainable count differs from e4b's is VOID (attention-only when it "
                   f"adapted no expert parameter), as tp4; gpt-oss carries a NO COMMON ADAPTER SET line (both s/step values, both trainable counts), never a ratio; mixtral's FOOTPRINT line "
                   f"(e4b under expert offload vs Unsloth resident: peak VRAM and s/step) leads its block; `ckpt_unsloth_mxfp4` (load_in_4bit=False) is VALID only with >= 2L packed expert "
                   f"parameters of a recorded class, grouped_mm selected and the MXFP4 grouped GEMM counted >= L*A per step; gpt-oss's bnb-4bit Unsloth arm reads the per-expert Linear4bit regime; "
                   f"an HF t214 arm whose dispatch did not reach grouped_mm is recorded, never VOID; P1–P7 of the draft scored HELD / FALSIFIED / UNTESTED "
                   f"(P1 HF band {TC2_P1_HF_BAND}; P2 Unsloth {TC2_P2_UNS_BAND} / HF {TC2_P2_HF_BAND}; P4 Unsloth {TC2_P4_UNS_BAND}, e4b within {TC2_P4_E4B_TOL:.0%} of tp4's "
                   f"{TP4_QWEN3_5_E4B_S_PER_STEP} s/step; P5 e4b/Unsloth {TC2_P5_BAND} at a >= {TC2_P5_PEAK_X:.0f}x lower e4b peak, tp2 {TP2_MIXTRAL['ratio_unsloth_over_e4b']}).")
    if any(fam in F for fam in FRONTIER_FAMS):
        out.append(f"Lane TC3 (`{FRONTIER_FAM}` / `{FRONTIER12_FAM}`, TC3-PREREG.md): one box per token; the box's anchor is `e4b/fused_attn4_m_offload` (the resident e4b arm is expected to OOM); "
                   f"(a) the FIT TABLE per framework; (b) in-box equivalence under TC1's R4 bands (EQUIVALENT ≤ {EQUIV} / COMPARABLE ≤ {COMPARABLE}) with the fused/reference offload pair as the control, and with `--tc1-dir` "
                   f"the matched trajectory against TC1's resident `e4b/fused_attn4_m` (median per-step |Δ| ≤ {RESIDENT_BAND} reads EQUIVALENT-TO-RESIDENT); (c) no cross-box ratio: P1's ratios are within the 24 GB box, "
                   f"P4 is two measurements; (d) P1–P4 of the draft scored HELD / FALSIFIED / UNTESTED.")
    for fam in list(FAMS) + sorted(set(F) - set(FAMS)):
        if fam in F:
            out += curve_block(F[fam]) if fam == CURVE_FAM else (frontier_block(F[fam]) if fam in FRONTIER_FAMS else family_block(F[fam]))
    out += ["\n## Verdicts", "| family | arm | VERDICT | validity | quality | note |", "|---|---|---|---|---|---|"]
    for fam in list(FAMS) + sorted(set(F) - set(FAMS)):
        R = F.get(fam)
        if not R:
            continue
        for x in R["rows"]:
            out.append(f"| {fam} | {x['fw']}/{x['tag']} | **{x['verdict']}** | {x['validity']} | {f(x.get('quality_delta'), 4) if x.get('quality_delta') is not None else (x.get('quality') or '—')} | {(x['why'] or x['reason'])[:160]} |")
    if any(fam in F for fam in ("qwen3", "qwen3native", AX_FAM)):
        out += ["\n## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)", "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_predictions(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if PROF945_FAM in F:
        out += ["\n## Amendment 12 (#945): the profile after the syncs are gone (descriptive) and P19",
                "| arm | verdict | s/step (profiled run) | device busy | device events/step | CPU ops/step | CPU self by family |", "|---|---|---|---|---|---|---|"]
        for t in prof945_table(F):
            out.append(f"| {t['arm']} | **{t['verdict']}** | {f(t['s_per_step_profiled_run'])} | {f(t['device_busy_fraction'], 3)} | {t['device_events_per_step']} | "
                       f"{t['cpu_ops_per_step']} | {json.dumps(t['cpu_self_by_family_fraction'], sort_keys=True) if t['cpu_self_by_family_fraction'] else '—'} |")
        out += ["", "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_prof945(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if SYNC_FAM in F:
        out += ["\n## Predictions P16 / P17 (TC1-PREREG amendment 10, #945: single-read grouping + pinned ring vs legacy, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_syncab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if LEAN_FAM in F:
        out += ["\n## Predictions P20 / P21 (TC1-PREREG amendment 13, #945: gnf4's trimmed padded LoRA delta vs its previous body, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_leanab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if TILE_FAM in F:
        out += ["\n## Predictions P22 / P23 (TC1-PREREG amendment 14, #945: gnf4's cost tile rule vs the max-keyed M-tile, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_tileab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if NB200_FAM in F:
        out += [f"\n## Prediction P14 (TC1-PREREG amendment 8: e4b shipped vs axolotl scattermoe over steps {LATE_FROM}..200, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_p14(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if NB_FAM in F:
        out += ["\n## Prediction P13 (TC1-PREREG amendment 5: native-best against native-best, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_p13(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if CURVE_FAM in F:
        out += ["\n## TC1b predictions P1–P4 (TC1b-PREREG-draft, scored mechanically)", "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_curve_predictions(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if any(fam in F for fam in TC2_FAMS):
        out += ["\n## TC2 predictions P1–P7 (TC2-PREREG-draft, scored mechanically)", "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_tc2_predictions(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if any(fam in F for fam in FRONTIER_FAMS):
        out += ["\n## TC3 predictions P1–P4 (TC3-PREREG-draft, scored mechanically; P1 on the 24 GB token, P2 on the 12 GB token, P3 per token, P4 on the 24 GB token with --tc1-dir)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_frontier_predictions(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    return "\n".join(out)


def reduce_dir(d, n_steps=None, tc1_dir=None):
    """Every family token in the dir; the qwen3curve token through R10 (its N is per arm, `--steps` does not apply to it) and the frontier tokens through R11,
    with TC1's receipts reduced first when `tc1_dir` names them (TC1b's (d) speed reading and TC3's resident comparison read them)."""
    recs, rcs = load(d), load_summary(d)
    tc1 = reduce_dir(tc1_dir, None) if tc1_dir else None
    return {fam: (reduce_curve_family(fam, recs[fam], rcs, tc1=tc1) if fam == CURVE_FAM
                  else (reduce_frontier_family(fam, recs[fam], rcs, n_steps, tc1=tc1) if fam in FRONTIER_FAMS else reduce_family(fam, recs[fam], rcs, n_steps)))
            for fam in list(FAMS) + sorted(set(recs) - set(FAMS)) if fam in recs}


# ----------------------------------------------------------------------------- R7: the selftest (hand-built receipts)
def _receipt(fw, tag, arm, steps=20, s=1.0, heldout0=2.0000, heldout_n=1.8000, losses=None, matched=True, seed=3407, backend="grouped_mm", sha=None, **over):
    """A complete OK receipt carrying every field this reducer reads, valid by every predicate for qwen3 (L=48, accum 4)."""
    L, A = 48, 4
    bkey = UNSLOTH_BACKEND_KEYS.get(backend, "unsloth_grouped_mm")
    ub = {"unsloth_grouped_mm": 0, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": L * A}
    ub[bkey] = L * A
    losses = losses if losses is not None else [round(2.0 - 0.01 * i, 5) for i in range(steps)]
    sha = sha or (("m" * 64) if matched else ("n" * 60 + fw[:4]).ljust(64, "n"))
    rows_n = [round(heldout_n + 0.001 * (i - 4), 5) for i in range(8)]
    r = {"framework": fw, "fam": "qwen3", "arm": arm, "tag": tag, "status": "ok", "steps": steps, "seq": 2048, "accum": A, "micro_batch": 2,
         "model": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "n_layers": L, "r": 16, "alpha": 16, "lr": 2e-4, "autocast": False,
         "template": "alpaca", "optimizer": "adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5", "env": {"box_class": "RTX 5090", "gpu": "NVIDIA GeForce RTX 5090"},
         "tokens": {"sha256": "t" * 64}, "trainable_params": 642514944, "trainable_mismatch": None, "trainable_by_group": {"attention": 1, "experts": 2, "other": 0},
         "losses": losses, "loss_first": losses[0], "loss_last": losses[-1], "loss_step2": losses[2],
         "eval_curve": [{"step": 0, "heldout_loss": heldout0}, {"step": steps, "heldout_loss": heldout_n}], "eval_loss_step0": heldout0, "eval_loss_final": heldout_n,
         "eval_rows": [{"step": 0, "losses": [heldout0] * 8}, {"step": steps, "losses": rows_n}],
         "s_per_step_median_11plus": s, "peak_vram_gb": 20.0, "joules_per_step": 100.0, "tokens_per_s": 1000.0,
         "C1_bit_exact": True, "C1_bytes_hashed": 10, "C1_empties_skipped": 0, "C1_control_detects_flipped_byte": True, "C1_control_tensor": "model.layers.0.mlp.experts.base.down_absmax",
         "C1_regime_by_tensor": {"nf4/64": 2}, "init_sha": "i" * 64,
         "attn_4bit": fw == "e4b", "n_attn4": 4 * L if fw == "e4b" else 0, "structural_expected_n_attn4": 4 * L if fw == "e4b" else None,
         "n_patched": L if arm == "fused" else 0, "kernel_calls_per_step_min": 2 * L * A if arm == "fused" else 0,
         "lora_path_present": (arm == "fused"), "lora_loop_share": ([0.0] * steps if arm == "fused" else None), "lora_path_loop_steps": ([] if arm == "fused" else None),
         "census": {"Params4bit_expert_stacks": 2 * L if fw in ("unsloth", "axolotl") else 0, "Linear4bit": 4 * L if fw != "hf" else 0},
         "experts_forward_calls_per_step_min": L * A if fw != "e4b" else 0,
         "unsloth_bnb4bit_modules": {"n_bnb4bit_unwrapped": L} if fw == "unsloth" else None,
         "engagement_banners": ["Unsloth: Enabling LoRA on MoE parameters"] if fw == "unsloth" else [],
         "hf_targets": {"n_target_modules": 4 * L, "n_target_parameters": 2 * L} if fw in ("hf", "axolotl") else None,
         "adapter_dtype": "fp32" if matched else "native", "adapter_dtypes_after": {"torch.float32": 576} if matched else {"torch.bfloat16": 192, "torch.float32": 384},
         "lora_init": f"matched:{seed}" if matched else "native",
         "matched_init": {"seed": seed, "complete": True, "n_slots_set": 192 + 2 * L * 128, "n_slots_expected": 192 + 2 * L * 128, "unmapped": []} if matched else None,
         "matched_init_sha": sha, "matched_init_sha_slots": 192 + 2 * L * 128, "matched_init_sha_unmapped": 0,
         "dynamo_counters": {"step10": {"recompiles_total": 0}, f"step{steps}": {"recompiles_total": 0}},
         "frozen_base_probe": {"slots": {"gate_up": {"sha": "g" * 64, "regime": "nf4/64", "control_detects_flip": True},
                                         "down": {"sha": "d" * 64, "regime": "nf4/64", "control_detects_flip": True},
                                         "q_proj": {"sha": ("q" if fw == "e4b" else "u") * 64, "regime": "nf4/64+dq" if fw == "e4b" else "nf4/64", "control_detects_flip": True}},
                               "control_detects_flip": True, "errors": []},
         "adapter": {"bytes": 1, "dtypes": ["torch.float32"]}, "profile": None,
         "unsloth_knobs": ({"moe_backend_requested": backend, "speed_tilt": False, "double_quant_requested": "off", "env_set": ({} if backend == "default" else {"UNSLOTH_MOE_BACKEND": backend})} if fw == "unsloth" else None),
         "moe_backend_selected": (backend if backend != "default" else "grouped_mm") if fw == "unsloth" else None,
         "unsloth_backend_calls_per_step_min": dict(ub) if fw == "unsloth" else None, "unsloth_backend_calls_per_step_max": dict(ub) if fw == "unsloth" else None,
         "unsloth_grouped_mm_calls_per_step_min": (GMM_FACTOR * L * A if backend == "grouped_mm" else 0) if fw == "unsloth" else None,
         "unsloth_grouped_mm_calls_per_step_max": (GMM_FACTOR * L * A if backend == "grouped_mm" else 0) if fw == "unsloth" else None,
         "unsloth_manual_grouped_mm_calls_per_step_max": 0 if fw == "unsloth" else None,
         "unsloth_double_quant": {"requested": False, "how": "from_pretrained(bnb_4bit_use_double_quant=False)", "loaded_nested": False} if fw == "unsloth" else None,
         "axolotl": ({"version": "0.20.0", "config": {"quantize_moe_experts": True, "plugins": (["axolotl.integrations.kernels.KernelsPlugin"] if tag.endswith("_best") else None)}, "census": {"n_bnb4bit_unwrapped": L}} if fw == "axolotl" else None),
         "axolotl_bnb4bit_modules": {"n_bnb4bit_unwrapped": L, "quantized_moe_experts_n": 2 * L} if fw == "axolotl" else None}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(r.get(k), dict):
            r[k] = {**r[k], **v}
        else:
            r[k] = v
    return r


def _stub(fw, tag, arm, status, reason="x"):
    return {"framework": fw, "fam": "qwen3", "arm": arm, "tag": tag, "status": status, "reason": reason, "steps": 20}


def _good_set():
    """The judged family, baseline: everything that can be VALID is, Unsloth/e4b 3.0 on both draws, HF OOM twice, axolotl INSTALL_FAILED."""
    R = {}
    R[("e4b", "fused_attn4_m")] = _receipt("e4b", "fused_attn4_m", "fused", s=1.00)
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.00, heldout_n=1.8100)
    R[("e4b", "reference_attn4_m")] = _receipt("e4b", "reference_attn4_m", "reference", s=2.00, heldout_n=1.8050)
    R[("e4b", "fused_attn4_m_d2")] = _receipt("e4b", "fused_attn4_m_d2", "fused", s=1.02)
    R[("unsloth", "ckpt_unsloth_m_d2")] = _receipt("unsloth", "ckpt_unsloth_m_d2", "unsloth", s=3.06, heldout_n=1.8100)
    R[("hf", "hf_peft_m")] = _stub("hf", "hf_peft_m", "hf", "oom", "OOM at step 1")
    R[("hf", "hf_peft_m_mb1")] = _stub("hf", "hf_peft_m_mb1", "hf", "oom", "OOM at step 1")
    R[("axolotl", "ckpt_axolotl_m")] = _stub("axolotl", "ckpt_axolotl_m", "axolotl", "install_failed", "pip rc=1")
    R[("e4b", "fused_attn4_m_prof")] = _receipt("e4b", "fused_attn4_m_prof", "fused", s=1.05, profile={"device_busy_fraction": 0.8, "device_events_per_step": 900, "cpu_ops_per_step": 5000, "cpu_self_by_family_fraction": {"fused_kernel": 0.5}})
    R[("unsloth", "ckpt_unsloth_prof")] = _receipt("unsloth", "ckpt_unsloth_prof", "unsloth", s=3.30, heldout_n=1.8100, profile={"device_busy_fraction": 0.55, "device_events_per_step": 5000, "cpu_ops_per_step": 90000, "cpu_self_by_family_fraction": {"routing": 0.4}})
    return R


def _native_set():
    """The labelled / native-best box: its own e4b fused_m, then every labelled row."""
    R = {}
    R[("e4b", "fused_attn4_m")] = _receipt("e4b", "fused_attn4_m", "fused", s=1.00)
    R[("unsloth", "ckpt_unsloth_best")] = _receipt("unsloth", "ckpt_unsloth_best", "unsloth", s=2.40, heldout_n=1.8300, matched=False)
    R[("unsloth", "ckpt_unsloth_t28")] = _receipt("unsloth", "ckpt_unsloth_t28", "unsloth", s=29.0, heldout_n=1.8100, backend="default")
    R[("unsloth", "ckpt_unsloth_triton")] = _receipt("unsloth", "ckpt_unsloth_triton", "unsloth", s=4.0, heldout_n=1.8100, backend="unsloth_triton")
    R[("e4b", "fused_attn4_shipped")] = _receipt("e4b", "fused_attn4_shipped", "fused", s=0.85, heldout_n=1.7800, matched=False)
    R[("e4b", "fused_attn4_m_nodgrad")] = _receipt("e4b", "fused_attn4_m_nodgrad", "fused", s=1.10, heldout_n=1.8000)
    R[("e4b", "fused_attn4_m_t212")] = _receipt("e4b", "fused_attn4_m_t212", "fused", s=0.98, heldout_n=1.8000)
    R[("axolotl", "ckpt_axolotl_best")] = _stub("axolotl", "ckpt_axolotl_best", "axolotl", "install_failed", "pip rc=1")
    R[("hf", "hf_peft_m_mb1_t214")] = _stub("hf", "hf_peft_m_mb1_t214", "hf", "not_run", "runs only when hf_peft_m OOMed on this box")
    for r in R.values():
        r["fam"] = "qwen3native"
    return R


def _ax_set(ax_s=(2.00, 2.02)):
    """Amendment 3: the axolotl box -- its own e4b fused_m x2, the matched axolotl arm x2 (quoted at ax_s / 1.01), scattermoe native-best, the HF t214 mb1 row."""
    R = {}
    R[("e4b", "fused_attn4_m")] = _receipt("e4b", "fused_attn4_m", "fused", s=1.00)
    R[("axolotl", "ckpt_axolotl_m")] = _receipt("axolotl", "ckpt_axolotl_m", "axolotl", s=ax_s[0], heldout_n=1.8020)
    R[("e4b", "fused_attn4_m_d2")] = _receipt("e4b", "fused_attn4_m_d2", "fused", s=1.02)
    R[("axolotl", "ckpt_axolotl_m_d2")] = _receipt("axolotl", "ckpt_axolotl_m_d2", "axolotl", s=ax_s[1], heldout_n=1.8030)
    R[("axolotl", "ckpt_axolotl_best")] = _receipt("axolotl", "ckpt_axolotl_best", "axolotl", s=1.50, heldout_n=1.8100, matched=False)
    R[("hf", "hf_peft_m_mb1_t214")] = _receipt("hf", "hf_peft_m_mb1_t214", "hf", s=3.00, heldout_n=1.8050, accum=8, micro_batch=1, experts_forward_calls_per_step_min=48 * 8)
    for r in R.values():
        r["fam"] = AX_FAM
    return R


def _nb_set(e4b_s=(1.00, 1.02), ax_s=(1.20, 1.21), un_s=(1.80, 1.82)):
    """Amendments 5 and 7: the native-best box -- e4b shipped, axolotl scattermoe and Unsloth native-best, two draws each, then e4b fused_m."""
    R = {}
    for i, sfx in enumerate(("", "_d2")):
        R[("e4b", "fused_attn4_shipped" + sfx)] = _receipt("e4b", "fused_attn4_shipped" + sfx, "fused", s=e4b_s[i], heldout_n=1.7800, matched=False)
        R[("axolotl", "ckpt_axolotl_best" + sfx)] = _receipt("axolotl", "ckpt_axolotl_best" + sfx, "axolotl", s=ax_s[i], heldout_n=1.7950, matched=False)
        R[("unsloth", "ckpt_unsloth_best" + sfx)] = _receipt("unsloth", "ckpt_unsloth_best" + sfx, "unsloth", s=un_s[i], heldout_n=1.8000, matched=False)
    R[("e4b", "fused_attn4_m")] = _receipt("e4b", "fused_attn4_m", "fused", s=1.25)
    for r in R.values():
        r["fam"] = NB_FAM
    return R


def _nb200_set(e4b_late=(3.00, 3.03), ax_late=(3.10, 3.12)):
    """Amendment 8: the 200-step native-best box -- e4b shipped and axolotl scattermoe x2 each, then e4b fused_m_200. axolotl's steps 1..30
    carry tc1-5090-34's warm-up shape (a 216 s first step, spikes to ~80 s), its later steps sit at its late value; e4b is flat."""
    R = {}
    for i, sfx in enumerate(("", "_d2")):
        e_ms = [5000.0] + [e4b_late[i] * 1e3] * 199
        a_ms = [216000.0, 32000.0, 61000.0, 4500.0, 11000.0, 81000.0] + [7000.0 if j % 3 == 0 else 3300.0 for j in range(24)] + [ax_late[i] * 1e3] * 170
        R[("e4b", "fused_attn4_shipped_200" + sfx)] = _receipt("e4b", "fused_attn4_shipped_200" + sfx, "fused", steps=200, s=statistics.median(e_ms[10:]) / 1e3,
                                                               heldout_n=0.7945, matched=False, step_ms=e_ms)
        R[("axolotl", "ckpt_axolotl_best_200" + sfx)] = _receipt("axolotl", "ckpt_axolotl_best_200" + sfx, "axolotl", steps=200, s=statistics.median(a_ms[10:]) / 1e3,
                                                                 heldout_n=0.7800, matched=False, step_ms=a_ms)
    R[("e4b", "fused_attn4_m_200")] = _receipt("e4b", "fused_attn4_m_200", "fused", steps=200, s=4.0, heldout_n=0.7687, step_ms=[4000.0] * 200)
    for r in R.values():
        r["fam"] = NB200_FAM
    return R


def _sync_set(ship=((5.00, 5.05), (4.30, 4.32)), match=((5.30, 5.33), (4.60, 4.62)), staged=(0, 9216)):
    """Amendment 10: e4b against itself -- each pair as (legacy draws, sync1 draws); `staged` = the ring's staged count on (legacy, sync1) arms."""
    R = {}
    for t, (leg, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for side, ss, st, mode, ring in (("legacy", leg, staged[0], "legacy", "0"), ("sync1", new, staged[1], "single", "1")):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                R[("e4b", tag)] = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000), matched=matched,
                                           sync_ab={"e4b_grouping": mode, "gnf4_pinned_ring": ring, "e4b_has_group_by_expert": True,
                                                    "gnf4_has_ring": True, "ring_staged": st, "ring_waits": 0})
    for r in R.values():
        r["fam"] = SYNC_FAM
    return R


def _prof945_set(busy_new=0.62, busy_old=0.48):
    """Amendment 12: three profiled e4b arms, each recording the path it ran and a profile summary."""
    R = {}
    for tag, matched, mode, ring, staged, busy in (("fused_attn4_shipped_prof", False, "single", "1", 9216, 0.66), ("fused_attn4_m_prof", True, "single", "1", 9216, busy_new),
                                                   ("fused_attn4_m_prof_legacy", True, "legacy", "0", 0, busy_old)):
        R[("e4b", tag)] = _receipt("e4b", tag, "fused", s=4.0, matched=matched, profile={"device_busy_fraction": busy, "device_events_per_step": 140000, "cpu_ops_per_step": 700000,
                                   "cpu_self_by_family_fraction": {"other": 0.4}},
                                   sync_ab={"e4b_grouping": mode, "gnf4_pinned_ring": ring, "e4b_has_group_by_expert": True, "gnf4_has_ring": True, "ring_staged": staged, "ring_waits": 0})
    for r in R.values():
        r["fam"] = PROF945_FAM
    return R


def _lean_set(ship=((5.00, 5.05), (4.70, 4.72)), match=((5.30, 5.33), (4.95, 4.97)), padded=(9216, 9216), lean=("0", "1")):
    """Amendment 13: e4b against itself on the post-#945 sync path -- each pair as (lean0 draws, lean1 draws); `padded` = the padded
    path's call count on (lean0, lean1) arms; `lean` = the body each side's record says it ran."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for side, ss, pc, body in (("lean0", old, padded[0], lean[0]), ("lean1", new, padded[1], lean[1])):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                R[("e4b", tag)] = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000), matched=matched,
                                           sync_ab={"e4b_grouping": "default", "gnf4_pinned_ring": "1", "e4b_has_group_by_expert": True,
                                                    "gnf4_has_ring": True, "ring_staged": 9216, "ring_waits": 0},
                                           lean_ab={"gnf4_lean_delta": body, "gnf4_lean_delta_env": body, "gnf4_has_lean_delta": True,
                                                    "lora_path_calls": {"loop": 0, "padded": pc, "grouped_mm": 0}})
    for r in R.values():
        r["fam"] = LEAN_FAM
    return R


def _tile_set(ship=((5.00, 5.05), (4.60, 4.62)), match=((5.30, 5.33), (5.00, 5.02)), rules=("max", "cost"), short=96, lean="1"):
    """Amendment 14: e4b against itself on the trimmed delta and the post-#945 sync path -- each pair as (tilemax draws, tilecost draws);
    `rules` = the rule each side's record says it ran; `short` = the cost side's launches of tiles shorter than 128."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for side, ss, rule in (("tilemax", old, rules[0]), ("tilecost", new, rules[1])):
            bm = {"16": 0, "32": 0, "64": 0, "128": 384} if side == "tilemax" else {"16": 0, "32": short // 2, "64": short - short // 2, "128": 384 - short}
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                R[("e4b", tag)] = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000), matched=matched,
                                           sync_ab={"e4b_grouping": "default", "gnf4_pinned_ring": "1", "e4b_has_group_by_expert": True,
                                                    "gnf4_has_ring": True, "ring_staged": 9216, "ring_waits": 0},
                                           lean_ab={"gnf4_lean_delta": lean, "gnf4_lean_delta_env": None, "gnf4_has_lean_delta": True,
                                                    "lora_path_calls": {"loop": 0, "padded": 9216, "grouped_mm": 0}},
                                           tile_ab={"gnf4_tile_rule": rule, "gnf4_tile_rule_env": rule, "gnf4_tile_d_env": None, "gnf4_has_tile_rule": True,
                                                    "prefill_bm_launches": bm})
    for r in R.values():
        r["fam"] = TILE_FAM
    return R


def _curve_receipt(fw, tag, arm, s=1.0, heldouts=None, row_shift=0.0, native=False, backend="grouped_mm", **over):
    """R10: a complete OK receipt for one qwen3curve arm from `_receipt`, with the sub-fixture's N / accum / r / tokens / trainable count,
    an eval grid with `train_wall_s`, `eval_rows` with 16 (the 200 group) or 8 rows, per-step lists of length N and counters consistent
    with its accum. `heldouts` = the held-out means at the grid's steps; each row i = mean + 0.001*(i - (n-1)/2) + row_shift."""
    g, N = CURVE_GROUP[tag], CURVE_N[tag]
    L = 48
    accum = 1 if g in ("p38", "t1") else 4
    mb = 1 if g in ("p38", "t1") else 2
    r_, alpha, seq, template = (8, 16, 512, "clinical") if g == "p38" else ((64, 64, 2048, "alpaca") if g == "r64" else (16, 16, 2048, "alpaca"))
    trainable = {"200": 642514944, "t1": 642514944, "p38": 321257472, "r64": 4 * 642514944}[g]
    every = {"200": 40, "p38": 20, "t1": 20, "r64": 20}[g]
    grid = list(range(0, N + 1, every))
    if heldouts is None:
        heldouts = [round(2.0 - 0.21 * (1 - (0.5 ** (st / max(every, 1)))), 5) for st in grid]
    assert len(heldouts) == len(grid), (tag, len(heldouts), grid)
    n_rows = 16 if g == "200" else 8
    curve = [{"step": st, "heldout_loss": h, "train_wall_s": round(st * s, 2)} for st, h in zip(grid, heldouts)]
    rows = [{"step": st, "losses": [round(h + 0.001 * (i - (n_rows - 1) / 2) + row_shift, 5) for i in range(n_rows)]} for st, h in zip(grid, heldouts)]
    losses = [round(2.0 - 0.002 * i, 5) for i in range(N)]
    ub = {"unsloth_grouped_mm": 0, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": L * accum}
    ub[UNSLOTH_BACKEND_KEYS.get(backend, "unsloth_grouped_mm")] = L * accum
    base = dict(steps=N, s=s, heldout0=heldouts[0], heldout_n=heldouts[-1], losses=losses, matched=not native, backend=backend,
                sha=("s" * 60 + g).ljust(64, "s") if not native else None)
    r = _receipt(fw, tag, arm, **base)
    r.update({"fam": CURVE_FAM, "accum": accum, "micro_batch": mb, "r": r_, "alpha": alpha, "seq": seq, "template": template,
              "tokens": {"sha256": (("c" if g == "p38" else "t") * 64)}, "trainable_params": trainable,
              "eval_curve": curve, "eval_rows": rows, "eval_loss_step0": heldouts[0], "eval_loss_final": heldouts[-1],
              "step_ms": [round(s * 1e3, 1)] * N, "lora_loop_share": ([0.0] * N if arm == "fused" else None), "lora_path_loop_steps": ([] if arm == "fused" else None),
              "dynamo_counters": {"step10": {"recompiles_total": 0}, f"step{N}": {"recompiles_total": 0}},
              "kernel_calls_per_step_min": 2 * L * accum if arm == "fused" else 0, "experts_forward_calls_per_step_min": L * accum if fw != "e4b" else 0,
              "optimizer": ("adamw_torch(lr=0.0001, weight_decay=0.01) schedule=constant warmup_steps=0" if g == "p38" else "adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5")})
    if fw == "unsloth":
        r.update({"unsloth_backend_calls_per_step_min": dict(ub), "unsloth_backend_calls_per_step_max": dict(ub),
                  "unsloth_grouped_mm_calls_per_step_min": GMM_FACTOR * L * accum if backend == "grouped_mm" else 0,
                  "unsloth_grouped_mm_calls_per_step_max": GMM_FACTOR * L * accum if backend == "grouped_mm" else 0})
    if native and fw == "unsloth":                     # tp4's anchor arm: fp32 adapters by tp4's cast, native init, the loader's double-quant
        r.update({"adapter_dtype": "fp32", "adapter_dtypes_after": {"torch.float32": 576}, "lora_init": "native", "matched_init": None, "matched_init_sha": "n" * 64,
                  "unsloth_double_quant": {"requested": True, "how": "loader default", "loaded_nested": True}})
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(r.get(k), dict):
            r[k] = {**r[k], **v}
        else:
            r[k] = v
    return r


def _curve_set():
    """The TC1b family, baseline: the matched pair EQUIVALENT at every eval (rows +0.005), the as-shipped arm below at 40 and +0.015 above
    at 200 (P38's shape), the anchor pair at 1.450 (tp2's 1.457 within 10 %), the t28 variant at 1.425, the t1 pair at 2.0, the r64 pair at 3.25."""
    H1 = [2.0, 1.90, 1.85, 1.82, 1.80, 1.79]
    H3 = [2.0, 1.89, 1.84, 1.815, 1.808, 1.805]
    R = {}
    R[("e4b", "fused_attn4_m_200")] = _curve_receipt("e4b", "fused_attn4_m_200", "fused", s=1.00, heldouts=H1)
    R[("unsloth", "ckpt_unsloth_m_200")] = _curve_receipt("unsloth", "ckpt_unsloth_m_200", "unsloth", s=3.00, heldouts=[round(h + 0.005, 5) for h in H1], row_shift=0.0)
    R[("e4b", "fused_attn4_shipped_200")] = _curve_receipt("e4b", "fused_attn4_shipped_200", "fused", s=0.85, heldouts=H3, native=True)
    R[("e4b", "fused_attn4_p38")] = _curve_receipt("e4b", "fused_attn4_p38", "fused", s=4.00, native=True)
    R[("unsloth", "ckpt_unsloth_p38")] = _curve_receipt("unsloth", "ckpt_unsloth_p38", "unsloth", s=5.80, native=True)
    R[("unsloth", "ckpt_unsloth_p38_t28")] = _curve_receipt("unsloth", "ckpt_unsloth_p38_t28", "unsloth", s=5.70, native=True, backend="default")
    R[("e4b", "fused_attn4_m_t1")] = _curve_receipt("e4b", "fused_attn4_m_t1", "fused", s=0.50)
    R[("unsloth", "ckpt_unsloth_m_t1")] = _curve_receipt("unsloth", "ckpt_unsloth_m_t1", "unsloth", s=1.00)
    R[("e4b", "fused_attn4_m_r64")] = _curve_receipt("e4b", "fused_attn4_m_r64", "fused", s=1.20)
    R[("unsloth", "ckpt_unsloth_m_r64")] = _curve_receipt("unsloth", "ckpt_unsloth_m_r64", "unsloth", s=3.90)
    # the native-init Unsloth p38 arms carry their own matched_init_sha (not the matched set's); the shipped arm its loader's
    R[("e4b", "fused_attn4_shipped_200")]["matched_init_sha"] = "e" * 64
    return R


# R11: TC2 hand-built receipts. The trainable counts are selftest STAND-INS (the orders of magnitude tp4 recorded: Granite 99.6 M vs Unsloth's
# 5.2 M, Qwen3.6 926 M vs 3.4 M -- RESULTS-tp4-p46cut.md), never registered numbers; every other field is consistent with the family's L / accum.
TC2_SELFTEST_TRAINABLE = {"granite": 99600000, "olmoe": 70000000, "gptoss": 90000000, "qwen3_5": 926000000, "mixtral": 100000000}


def _merge(r, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(r.get(k), dict):
            r[k] = {**r[k], **v}
        else:
            r[k] = v
    return r


def _tc2_receipt(fam, fw, tag, arm, s=1.0, heldout_n=1.8, trainable=None, matched=True, **over):
    """A complete OK receipt for one TC2 family arm, from `_receipt` with the family's registered pin / L / census and counters consistent
    with its accum (A 4) and instrument (small: N 60, 48 rows; big: N 20, 8 rows); `trainable` defaults to the family's stand-in."""
    L, A = N_LAYERS[fam], 4
    mid, rev, _ = TC2_MODELS[fam]
    small = fam in TC2_TOKENS["tc2small"]
    steps, rows = (60, 48) if small else (20, 8)
    n_attn = ATTN_CENSUS.get(fam) or 4 * L
    tr = trainable if trainable is not None else TC2_SELFTEST_TRAINABLE[fam]
    e4b_attn4 = (fw == "e4b" and arm != "attn_only")
    r = _receipt(fw, tag, arm, steps=steps, s=s, heldout_n=heldout_n, matched=matched)
    ub = {"unsloth_grouped_mm": L * A, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": L * A}
    r.update({"fam": fam, "model": mid, "revision": rev, "n_layers": L, "trainable_params": tr,
              "eval_rows": [{"step": 0, "losses": [r["eval_loss_step0"]] * rows}, {"step": steps, "losses": [round(heldout_n + 0.001 * (i - rows // 2), 5) for i in range(rows)]}],
              "attn_4bit": e4b_attn4, "n_attn4": n_attn if e4b_attn4 else 0, "structural_expected_n_attn4": n_attn if e4b_attn4 else None,
              "n_patched": L if arm == "fused" else 0, "kernel_calls_per_step_min": 2 * L * A if arm == "fused" else 0,
              "lora_path_present": (arm == "fused"), "lora_loop_share": ([0.0] * steps if arm == "fused" else None), "lora_path_loop_steps": ([] if arm == "fused" else None),
              "census": {"Params4bit_expert_stacks": 2 * L if fw in ("unsloth", "axolotl") else 0, "Linear4bit": 4 * L if fw != "hf" else 0,
                         "expert_param_classes": ({"Params4bit": 2 * L} if fw == "unsloth" else ({"Parameter": 2 * L} if fw in ("hf", "axolotl") else {}))},
              "experts_forward_calls_per_step_min": L * A if fw != "e4b" else 0,
              "unsloth_bnb4bit_modules": {"n_bnb4bit_unwrapped": L} if fw == "unsloth" else None,
              "hf_targets": {"n_target_modules": n_attn, "n_target_parameters": 2 * L} if fw in ("hf", "axolotl") else None,
              "unsloth_grouped_mm_calls_per_step_min": (GMM_FACTOR * L * A) if fw == "unsloth" else None,
              "unsloth_grouped_mm_calls_per_step_max": (GMM_FACTOR * L * A) if fw == "unsloth" else None,
              "unsloth_backend_calls_per_step_min": dict(ub) if fw == "unsloth" else None, "unsloth_backend_calls_per_step_max": dict(ub) if fw == "unsloth" else None,
              "axolotl_bnb4bit_modules": {"n_bnb4bit_unwrapped": L, "quantized_moe_experts_n": 2 * L} if fw == "axolotl" else None,
              "trainable_by_group": {"attention": 1, "experts": 0, "other": 0} if arm == "attn_only" else {"attention": 1, "experts": 2, "other": 0}})
    if matched:
        n_slots = n_attn if arm == "attn_only" else n_attn + 2 * L * 8
        r["matched_init"] = {"seed": 3407, "complete": True, "n_slots_set": n_slots, "n_slots_expected": n_slots, "unmapped": [],
                             "expected_parts": ({"n_attention_projections": n_attn, "experts": "excluded: attn_only arm (expert adapters frozen or absent; trainable slots only)"} if arm == "attn_only"
                                                else {"n_attention_projections": n_attn, "n_layers": L, "n_experts": 8})}
        r["matched_init_sha_slots"] = n_slots
    return _merge(r, over)


def _tc2_set(fam):
    """R11: the baseline receipt set per TC2 family, in the registered order -- every arm that can be VALID is, the attention-only Unsloth
    arms VOID (granite, qwen3_5's UT4 arm), gpt-oss's packed arm VALID beside e4b's attention-only arm, mixtral under offload."""
    def T(fw, tag, arm, **kw):
        return _tc2_receipt(fam, fw, tag, arm, **kw)

    def S(fw, tag, arm, st, why="x", **extra):
        return {**_stub(fw, tag, arm, st, why), "fam": fam, "steps": 60 if fam in TC2_TOKENS["tc2small"] else 20, **extra}
    R = {}
    zero = {"unsloth_grouped_mm": 0, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": 0}
    reached = {"requested": "grouped_mm", "accepted": True, "config": "grouped_mm", "torch_grouped_mm_calls_per_step_min": 2 * N_LAYERS[fam] * 4, "torch_F_grouped_mm_calls_per_step_min": 0, "reached_grouped_mm": True}
    if fam in ("granite", "olmoe"):
        e_s, e2, h_s, h2, r_s, u_s, a_s = {"granite": (2.366, 2.39, 3.057, 3.09, 18.59, 2.445, 3.3), "olmoe": (1.395, 1.40, 2.713, 2.73, 14.883, 2.8, 3.0)}[fam]
        R[("e4b", "fused_attn4_m")] = T("e4b", "fused_attn4_m", "fused", s=e_s)
        R[("hf", "hf_peft_m")] = T("hf", "hf_peft_m", "hf", s=h_s, heldout_n=1.8004)
        R[("e4b", "reference_attn4_m")] = T("e4b", "reference_attn4_m", "reference", s=r_s, heldout_n=1.8005)
        R[("e4b", "fused_attn4_m_d2")] = T("e4b", "fused_attn4_m_d2", "fused", s=e2)
        R[("hf", "hf_peft_m_d2")] = T("hf", "hf_peft_m_d2", "hf", s=h2, heldout_n=1.8004)
        if fam == "granite":          # Unsloth never discovers Granite's ParallelExperts: attention only, experts bf16, no banner (UPSTREAM-NOTES)
            att = dict(trainable=5200000, heldout_n=1.85, trainable_by_group={"attention": 1, "experts": 0, "other": 0}, unsloth_bnb4bit_modules={"n_bnb4bit_unwrapped": 0},
                       census={"Params4bit_expert_stacks": 0, "expert_param_classes": {"Parameter": 64}},
                       engagement_banners=["Unsloth: get_moe_target_parameters resolved no expert parameters for the requested targets. The expert weights will NOT be trained."])
            R[("unsloth", "ckpt_unsloth_m")] = T("unsloth", "ckpt_unsloth_m", "unsloth", s=u_s, **att)
            R[("unsloth", "ckpt_unsloth_m_experts")] = T("unsloth", "ckpt_unsloth_m_experts", "unsloth", s=u_s + 0.01, **att)
        else:
            R[("unsloth", "ckpt_unsloth_m")] = T("unsloth", "ckpt_unsloth_m", "unsloth", s=u_s, heldout_n=1.8004)
        R[("hf", "hf_peft_m_t214")] = T("hf", "hf_peft_m_t214", "hf", s=round(h_s * 0.95, 4), heldout_n=1.8004, hf_experts_dispatch=dict(reached))
        R[("axolotl", "ckpt_axolotl_m")] = T("axolotl", "ckpt_axolotl_m", "axolotl", s=a_s, heldout_n=1.8004)
        R[("axolotl", "ckpt_axolotl_best")] = T("axolotl", "ckpt_axolotl_best", "axolotl", s=round(a_s * 0.9, 4), heldout_n=1.83, matched=False)
        R[("e4b", "fused_attn4_shipped")] = T("e4b", "fused_attn4_shipped", "fused", s=round(e_s * 0.9, 4), heldout_n=1.78, matched=False)
    elif fam == "gptoss":
        R[("e4b", "fused_attn4_m")] = S("e4b", "fused_attn4_m", "fused", "refused", "SKIPPED as REFUSED: tp1 (P36) + tp2 (P40) rows cited -- enable_fast_train(dgrad=True) patched 0 modules on gpt-oss",
                                        cited="tp1,tp2", probed_by="attn_only", probe_reason="enable_fast_train(dgrad=True) patched 0 modules on this box: 0 ExpertsLoRA", n_patched=0)
        R[("e4b", "attn_only_m")] = T("e4b", "attn_only_m", "attn_only", s=3.1, trainable=5000000)
        R[("e4b", "attn_only_m_d2")] = T("e4b", "attn_only_m_d2", "attn_only", s=3.15, trainable=5000000)
        L = N_LAYERS["gptoss"]
        R[("unsloth", "ckpt_unsloth_m")] = T("unsloth", "ckpt_unsloth_m", "unsloth", s=9.0, heldout_n=1.75, unsloth_load_in_4bit=True,
                                             census={"Params4bit_expert_stacks": 0, "Params4bit_expert_linears": 2 * L * 32, "expert_param_classes": {"Params4bit": 2 * L * 32}},
                                             unsloth_bnb4bit_modules={"n_bnb4bit_unwrapped": 0}, engagement_banners=[f"Unsloth: {UNSLOTH_BANNER_PER_EXPERT}"],
                                             unsloth_backend_calls_per_step_min=dict(zero), unsloth_backend_calls_per_step_max=dict(zero),
                                             unsloth_grouped_mm_calls_per_step_min=0, unsloth_grouped_mm_calls_per_step_max=0)
        pk = {"unsloth_mxfp4_grouped_mm": L * 4}
        for tag, s_ in (("ckpt_unsloth_mxfp4", 4.0), ("ckpt_unsloth_mxfp4_d2", 4.1)):
            R[("unsloth", tag)] = T("unsloth", tag, "unsloth", s=s_, heldout_n=1.75, unsloth_load_in_4bit=False,
                                    census={"Params4bit_expert_stacks": 0, "Linear4bit": 0, "expert_param_classes": {"Mxfp4ExpertParam": 2 * L}},
                                    unsloth_bnb4bit_modules={"n_bnb4bit_unwrapped": 0}, unsloth_packed_calls_per_step_min=dict(pk), unsloth_packed_calls_per_step_max=dict(pk),
                                    unsloth_backend_calls_per_step_min=dict(zero), unsloth_backend_calls_per_step_max=dict(zero),
                                    unsloth_grouped_mm_calls_per_step_min=0, unsloth_grouped_mm_calls_per_step_max=0,
                                    unsloth_double_quant={"requested": None, "how": "not applicable: load_in_4bit=False", "loaded_nested": None})
        R[("hf", "hf_peft_m")] = S("hf", "hf_peft_m", "hf", "oom", "OOM at load")
        R[("axolotl", "ckpt_axolotl_m")] = S("axolotl", "ckpt_axolotl_m", "axolotl", "refused", "quantize_moe_experts on an is_transposed stack")
        R[("e4b", "reference_attn4_m")] = S("e4b", "reference_attn4_m", "reference", "refused", "SKIPPED as REFUSED: tp4's bias rule", cited="tp4", n_patched=0)
    elif fam == "qwen3_5":
        R[("e4b", "fused_attn4_m")] = T("e4b", "fused_attn4_m", "fused", s=6.33)
        att = dict(trainable=3400000, heldout_n=1.85, trainable_by_group={"attention": 1, "experts": 0, "other": 0}, engagement_banners=[])   # UT4: the stacks quantised, nothing on them adapted
        R[("unsloth", "ckpt_unsloth_m")] = T("unsloth", "ckpt_unsloth_m", "unsloth", s=26.98, **att)
        R[("e4b", "fused_attn4_m_d2")] = T("e4b", "fused_attn4_m_d2", "fused", s=6.35)
        R[("unsloth", "ckpt_unsloth_m_d2")] = T("unsloth", "ckpt_unsloth_m_d2", "unsloth", s=27.0, **att)
        R[("unsloth", "ckpt_unsloth_m_experts")] = T("unsloth", "ckpt_unsloth_m_experts", "unsloth", s=25.0, heldout_n=1.8004)
        R[("hf", "hf_peft_m")] = S("hf", "hf_peft_m", "hf", "oom", "OOM at load")
        R[("axolotl", "ckpt_axolotl_m")] = S("axolotl", "ckpt_axolotl_m", "axolotl", "oom", "OOM at step 1")
        R[("axolotl", "ckpt_axolotl_best")] = S("axolotl", "ckpt_axolotl_best", "axolotl", "oom", "OOM at step 1")
        R[("e4b", "fused_attn4_shipped")] = T("e4b", "fused_attn4_shipped", "fused", s=5.5, heldout_n=1.78, matched=False)
        R[("e4b", "reference_attn4_m")] = T("e4b", "reference_attn4_m", "reference", s=88.0, heldout_n=1.8005)
    elif fam == "mixtral":
        R[("e4b", "fused_attn4_m")] = T("e4b", "fused_attn4_m", "fused", s=2.377, peak_vram_gb=3.22, offload=True)
        R[("unsloth", "ckpt_unsloth_m")] = T("unsloth", "ckpt_unsloth_m", "unsloth", s=0.858, heldout_n=1.8004, peak_vram_gb=29.16)
        R[("e4b", "fused_attn4_m_d2")] = T("e4b", "fused_attn4_m_d2", "fused", s=2.40, peak_vram_gb=3.25, offload=True)
        R[("unsloth", "ckpt_unsloth_m_d2")] = T("unsloth", "ckpt_unsloth_m_d2", "unsloth", s=0.87, heldout_n=1.8004, peak_vram_gb=29.2)
        R[("hf", "hf_peft_m")] = S("hf", "hf_peft_m", "hf", "oom", "OOM at load")
        R[("axolotl", "ckpt_axolotl_m")] = S("axolotl", "ckpt_axolotl_m", "axolotl", "oom", "OOM at load")
        R[("axolotl", "ckpt_axolotl_best")] = S("axolotl", "ckpt_axolotl_best", "axolotl", "oom", "OOM at load")
        R[("e4b", "fused_attn4_shipped")] = T("e4b", "fused_attn4_shipped", "fused", s=2.2, heldout_n=1.78, matched=False, peak_vram_gb=3.2, offload=True)
        R[("e4b", "reference_attn4_m")] = T("e4b", "reference_attn4_m", "reference", s=19.0, heldout_n=1.8005, peak_vram_gb=3.3, offload=True)
    return R


def _frontier_receipt(fam, fw, tag, arm, lever=None, peak=None, host=None, total=125.0, box="RTX 4090", **over):
    """R11: a complete OK receipt for a frontier arm from `_receipt`, with the lever, the host-RAM fields and the box class the arm driver records."""
    r = _receipt(fw, tag, arm, **over)
    r.update({"fam": fam, "memory_lever": lever, "offload": lever == "e4b_offload", "host_ram_high_water_gb": host, "host_ram_total_gb": total,
              "host_ram": {"high_water_gb": host, "total_gb": total, "cgroup_limit_gb": None}, "env": {"box_class": box, "gpu": "NVIDIA " + box}})
    if peak is not None:
        r["peak_vram_gb"] = peak
    return r


def _fstub(fam, fw, tag, arm, status, reason, peak=None, host=None, total=125.0, **over):
    """R11: a stub (OOM / refused) as tc1_arm.py writes it on a frontier box: the peak VRAM and host RAM it reached travel on the row."""
    return {**_stub(fw, tag, arm, status, reason), "fam": fam, "peak_vram_gb": peak, "host_ram_high_water_gb": host, "host_ram_total_gb": total, **over}


def _frontier_set():
    """The 24 GB token, baseline: e4b resident OOM, offload OK (2.0 s/step, 9.5 GB, 40 GB host), mb1 resident OK, Unsloth OOM at both recipes, HF OOM resident and
    REFUSED under offload (accelerate's exception text), axolotl 4-bit OOM, layer-offload OK, ZeRO-3 REFUSED (the trainer's engine), reference offload OK."""
    F = FRONTIER_FAM
    R = {}
    R[("e4b", "fused_attn4_m")] = _fstub(F, "e4b", "fused_attn4_m", "fused", "oom", "OOM at step 1: CUDA out of memory", peak=23.6, host=6.0)
    R[("e4b", "fused_attn4_m_offload")] = _frontier_receipt(F, "e4b", "fused_attn4_m_offload", "fused", lever="e4b_offload", peak=9.5, host=40.0, s=2.00)
    R[("e4b", "fused_attn4_m_mb1")] = _frontier_receipt(F, "e4b", "fused_attn4_m_mb1", "fused", peak=21.0, host=6.0, s=1.60, accum=8, micro_batch=1, kernel_calls_per_step_min=2 * 48 * 8)
    R[("e4b", "fused_attn4_shipped")] = _frontier_receipt(F, "e4b", "fused_attn4_shipped", "fused", peak=22.9, host=6.0, s=1.30, heldout_n=1.79, matched=False)   # amendment 3: the as-shipped resident fit row
    R[("unsloth", "ckpt_unsloth_m")] = _fstub(F, "unsloth", "ckpt_unsloth_m", "unsloth", "oom", "OOM at step 1", peak=23.9, host=5.0)
    R[("unsloth", "ckpt_unsloth_m_mb1")] = _fstub(F, "unsloth", "ckpt_unsloth_m_mb1", "unsloth", "oom", "OOM at step 2", peak=23.8, host=5.0)
    R[("hf", "hf_peft_m")] = _fstub(F, "hf", "hf_peft_m", "hf", "oom", "OOM at load", peak=23.9, host=4.0)
    R[("hf", "hf_peft_m_offload")] = _fstub(F, "hf", "hf_peft_m_offload", "hf", "refused", "hf_offload: NotImplementedError at step 1: Cannot copy out of meta tensor; no data!", peak=20.1, host=70.0,
                                            memory_lever="hf_offload", hf_offload={"max_memory": {"0": "22GiB", "cpu": "125GiB"}, "llm_int8_enable_fp32_cpu_offload": True,
                                                                                   "device_map_summary": {"by_device": {"cuda:0": 60, "cpu": 40}, "experts_entries_by_device": {"cpu": 40}, "cpu_or_disk_sample": ["model.layers.8.mlp.experts"], "any_cpu_or_disk": True}})
    R[("axolotl", "ckpt_axolotl_m")] = _fstub(F, "axolotl", "ckpt_axolotl_m", "axolotl", "oom", "OOM at step 1", peak=23.7, host=5.0)
    R[("axolotl", "ckpt_axolotl_m_layeroffload")] = _frontier_receipt(F, "axolotl", "ckpt_axolotl_m_layeroffload", "axolotl", lever="axolotl_layer_offload", peak=14.0, host=30.0, s=6.00, heldout_n=1.8100,
                                                                      axolotl_layer_offload={"engaged": True, "n_layers": 48, "n_frozen_params_managed": 480, "n_hooks": 192, "driven_by": "the harness"})
    R[("axolotl", "ckpt_axolotl_m_zero3")] = _fstub(F, "axolotl", "ckpt_axolotl_m_zero3", "axolotl", "refused",
                                                    "axolotl ZeRO-3 parameter offload cannot be driven outside axolotl's trainer: ModelLoader.load() reads cfg.deepspeed only for zero.init() partitioning; "
                                                    "the engine is deepspeed.initialize, created by transformers' Trainer inside axolotl.train.train", host=3.0, memory_lever="axolotl_zero3",
                                                    axolotl_zero3={"deepspeed_version": "0.19.7", "deepspeed_config_not_run": {"zero_optimization": {"stage": 3}}})
    R[("e4b", "reference_attn4_m_offload")] = _frontier_receipt(F, "e4b", "reference_attn4_m_offload", "reference", lever="e4b_offload", peak=9.6, host=40.0, s=4.00, heldout_n=1.8050)
    return R


def _frontier12_set():
    """The 12 GB token, baseline: the e4b offload pair (two draws) and the reference offload OK, the e4b resident OOM, Unsloth and HF OOM at mb1, axolotl REFUSED by the driver gate."""
    F, b = FRONTIER12_FAM, "RTX A2000"
    R = {}
    R[("e4b", "fused_attn4_m_offload")] = _frontier_receipt(F, "e4b", "fused_attn4_m_offload", "fused", lever="e4b_offload", peak=7.2, host=42.0, total=64.0, box=b, s=9.00)
    R[("e4b", "fused_attn4_m_offload_d2")] = _frontier_receipt(F, "e4b", "fused_attn4_m_offload_d2", "fused", lever="e4b_offload", peak=7.2, host=42.0, total=64.0, box=b, s=9.18)
    R[("e4b", "reference_attn4_m_offload")] = _frontier_receipt(F, "e4b", "reference_attn4_m_offload", "reference", lever="e4b_offload", peak=7.3, host=42.0, total=64.0, box=b, s=18.0, heldout_n=1.8050)
    R[("e4b", "fused_attn4_m")] = _fstub(F, "e4b", "fused_attn4_m", "fused", "oom", "OOM at load", peak=11.9, host=6.0, total=64.0)
    R[("unsloth", "ckpt_unsloth_m_mb1")] = _fstub(F, "unsloth", "ckpt_unsloth_m_mb1", "unsloth", "oom", "OOM at step 1", peak=11.8, host=5.0, total=64.0)
    R[("hf", "hf_peft_m_mb1")] = _fstub(F, "hf", "hf_peft_m_mb1", "hf", "oom", "OOM at load", peak=11.9, host=4.0, total=64.0)
    R[("axolotl", "ckpt_axolotl_m")] = _fstub(F, "axolotl", "ckpt_axolotl_m", "axolotl", "refused", "cu130 wheels need driver >= 580; host has 575.57", total=64.0, venv="venv-axolotl (torch cu130)")
    return R


def selftest():
    import tempfile
    cases = 0

    def run(R, n_steps=20, fam="qwen3"):
        return reduce_family(fam, R, {}, n_steps)

    def both(R=None, NR=None):
        return {"qwen3": run(R if R is not None else _good_set()), "qwen3native": run(NR if NR is not None else _native_set(), fam="qwen3native")}

    def P(F):
        return {p: v for p, _, v, _ in score_predictions(F)}

    def row(F, fw, tag):
        return next(x for x in F["rows"] if (x["fw"], x["tag"]) == (fw, tag))

    # 1. baseline: every verdict, the quoted position with its interval, every prediction HELD (P1b/P4/P7 from the native box)
    F = run(_good_set())
    V = F["verdicts"]
    assert V[("e4b", "fused_attn4_m")] == "VALID" and V[("unsloth", "ckpt_unsloth_m")] == "VALID" and V[("e4b", "reference_attn4_m")] == "VALID", V
    assert V[("hf", "hf_peft_m")] == "OOM" and V[("axolotl", "ckpt_axolotl_m")] == "UNSUPPORTED" and V[("e4b", "fused_attn4_m_prof")] == "VALID" and V[("unsloth", "ckpt_unsloth_prof")] == "VALID"
    assert F["draws"][("e4b", "fused_attn4_m")]["verdict"] == "STABLE" and F["draws"][("unsloth", "ckpt_unsloth_m")]["verdict"] == "STABLE"
    pu = F["positions"]["unsloth"]
    assert pu["quoted"] and abs(pu["ratio"] - 3.03 / 1.01) < 1e-9 and pu["e4b_draws"] == 2 and pu["other_draws"] == 2 and pu["quality"] == "COMPARABLE", pu
    assert pu["n_cross"] == 4 and abs(pu["ratio_min"] - 3.00 / 1.02) < 1e-9 and abs(pu["ratio_max"] - 3.06 / 1.00) < 1e-9, (pu["ratio_min"], pu["ratio_max"])   # H: the interval
    eq = F["equivalence"]
    assert eq[("unsloth", "ckpt_unsloth_m")]["reading"] == "EQUIVALENT" and eq[("e4b", "reference_attn4_m")]["reading"] == "EQUIVALENT", {k: v["reading"] for k, v in eq.items()}
    assert abs(F["equiv_band"]["train"] - 0.005) < 1e-9 and abs(F["equiv_band"]["heldout"] - 0.015) < 1e-9 and F["noise_floor"] == 0.0, (F["equiv_band"], F["noise_floor"])
    pr = eq[("unsloth", "ckpt_unsloth_m")]["paired"]
    assert pr["n"] == 8 and abs(pr["mean"] - 0.01) < 1e-9 and pr["se"] is not None and pr["rows_favouring_anchor"] == 8 and pr["rows_favouring_arm"] == 0, pr   # G
    assert row(F, "unsloth", "ckpt_unsloth_m")["step0_class"] == "SAME-BYTES-CLASS" and eq[("unsloth", "ckpt_unsloth_m")]["d_loss_step2"] == 0.0
    fb = F["frozen"][("unsloth", "ckpt_unsloth_m")]
    assert fb["gate_up"]["reading"] == "SAME-BYTES" and fb["down"]["reading"] == "SAME-BYTES" and fb["q_proj"]["reading"] == "N-A" and "regime differs" in fb["q_proj"]["why"], fb
    assert F["parity"]["verdict"] == "PASS" and F["anchor_sha"] == "m" * 64
    FF = both()
    assert P(FF) == {**{f"P{i}": "HELD" for i in range(1, 11)}, "P1b": "HELD"}, P(FF)
    NF = FF["qwen3native"]
    assert set(NF["labelled"]) == set(LABELLED) and NF["labelled"][("unsloth", "ckpt_unsloth_t28")]["quoted"] and abs(NF["labelled"][("unsloth", "ckpt_unsloth_t28")]["ratio"] - 29.0) < 1e-9
    assert NF["labelled"][("e4b", "fused_attn4_m_t212")]["quoted"] and not NF["labelled"][("axolotl", "ckpt_axolotl_best")]["quoted"] and NF["native"]["unsloth"]["quoted"]
    assert not NF["positions"]["unsloth"]["quoted"]                     # the matched position is never read on the native box
    cases += 1
    # 2-6. one receipt per R3 predicate that MUST read VOID (the matched-set predicates)
    for name, over in (("incomplete", {"matched_init": {"complete": False, "n_slots_set": 100, "n_slots_expected": 192, "unmapped": ["x"]}}),
                       ("bf16", {"adapter_dtypes_after": {"torch.float32": 384, "torch.bfloat16": 192}}),
                       ("step0", {"eval_loss_step0": 2.0600, "eval_curve": [{"step": 0, "heldout_loss": 2.06}, {"step": 20, "heldout_loss": 1.81}]}),
                       ("native-on-matched", {"lora_init": "native", "matched_init": None}),
                       ("seed", {"lora_init": "matched:1", "matched_init": {"seed": 1}})):
        R = _good_set()
        R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81, **over)
        F = run(R)
        x = row(F, "unsloth", "ckpt_unsloth_m")
        assert x["verdict"] == "VOID", (name, x["verdict"], x["why"])
        assert not F["positions"]["unsloth"]["quoted"] and F["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "N-A", (name, F["positions"]["unsloth"])
        Pp = P({"qwen3": F})
        assert Pp["P1"] == "UNTESTED" and Pp["P3"] == "UNTESTED", (name, Pp)
        cases += 1
    # 7. C: the step-0 band -- 0.03 reads NEAR and stays VALID (recorded), 0.06 VOIDs
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81, eval_loss_step0=2.03)
    F = run(R)
    x = row(F, "unsloth", "ckpt_unsloth_m")
    assert x["verdict"] == "VALID" and x["step0_class"] == "NEAR" and abs(x["step0_delta"] - 0.03) < 1e-9, (x["verdict"], x["step0_class"])
    assert step0_class(0.005) == "SAME-BYTES-CLASS" and step0_class(0.05) == "NEAR" and step0_class(0.0501) == "VOID"
    print("FAILING-CASE C: step-0 |delta| 0.06 ->", row(run({**_good_set(), ("unsloth", "ckpt_unsloth_m"): _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81, eval_loss_step0=2.06)}), "unsloth", "ckpt_unsloth_m")["why"])
    cases += 1
    # 8-12. tp4's predicates still VOID: n_patched, C1, trainable count, unsloth banner/stacks, hf attention-only
    for name, key, over in (("n_patched", ("e4b", "fused_attn4_m"), {"n_patched": 47}),
                            ("c1", ("e4b", "fused_attn4_m"), {"C1_bit_exact": False}),
                            ("trainable", ("unsloth", "ckpt_unsloth_m"), {"trainable_params": 1}),
                            ("banner", ("unsloth", "ckpt_unsloth_m"), {"engagement_banners": []}),
                            ("stacks", ("unsloth", "ckpt_unsloth_m"), {"census": {"Params4bit_expert_stacks": 10}})):
        R = _good_set()
        R[key] = {**R[key], **over}
        F = run(R)
        assert row(F, *key)["verdict"] == "VOID", (name, row(F, *key)["why"])
        assert not F["positions"]["unsloth"]["quoted"], name
        cases += 1
    R = _good_set()
    R[("hf", "hf_peft_m")] = _receipt("hf", "hf_peft_m", "hf", s=5.0, hf_targets={"n_target_modules": 192, "n_target_parameters": 0})
    F = run(R)
    assert row(F, "hf", "hf_peft_m")["verdict"] == "VOID" and "attention-only" in row(F, "hf", "hf_peft_m")["why"]
    cases += 1
    # 13. QUALITY_FAIL: validity holds, held-out at N is 0.08 above the anchor -> QUALITY_FAIL, position not quoted, DIVERGENT
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.8800)
    R[("unsloth", "ckpt_unsloth_m_d2")] = _receipt("unsloth", "ckpt_unsloth_m_d2", "unsloth", s=3.0, heldout_n=1.8800)
    F = run(R)
    assert row(F, "unsloth", "ckpt_unsloth_m")["verdict"] == "QUALITY_FAIL" and row(F, "unsloth", "ckpt_unsloth_m")["validity"] == "VALID"
    assert not F["positions"]["unsloth"]["quoted"] and "QUALITY_FAIL" in F["positions"]["unsloth"]["why"]
    assert F["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "DIVERGENT"
    cases += 1
    # 14. UNSTABLE: the e4b draws differ by 12 % -> UNSTABLE, no position, P2 FALSIFIED; 5 % is the bar for every framework (H)
    R = _good_set()
    R[("e4b", "fused_attn4_m_d2")] = _receipt("e4b", "fused_attn4_m_d2", "fused", s=1.13)
    F = run(R)
    d = F["draws"][("e4b", "fused_attn4_m")]
    assert d["verdict"] == "UNSTABLE" and abs(d["stability"] - 0.13 / 1.065) < 1e-9 and not d["usable"], d
    assert not F["positions"]["unsloth"]["quoted"] and "UNSTABLE" in F["positions"]["unsloth"]["why"]
    assert P({"qwen3": F})["P2"] == "FALSIFIED" and P({"qwen3": F})["P1"] == "UNTESTED"
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m_d2")] = _receipt("unsloth", "ckpt_unsloth_m_d2", "unsloth", s=3.21, heldout_n=1.81)   # 6.8 %: unstable at the 5 % bar
    assert run(R)["draws"][("unsloth", "ckpt_unsloth_m")]["verdict"] == "UNSTABLE"
    cases += 1
    # 15. DIVERGENT: unsloth_m's train losses sit 0.10 above the anchor's -> DIVERGENT, P3 FALSIFIED
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81, losses=[round(2.1 - 0.01 * i, 5) for i in range(20)])
    F = run(R)
    e = F["equivalence"][("unsloth", "ckpt_unsloth_m")]
    assert e["reading"] == "DIVERGENT" and abs(e["med_train"] - 0.1) < 1e-9, e
    assert P({"qwen3": F})["P3"] == "FALSIFIED"
    cases += 1
    # 16. COMPARABLE (0.03 on both): above the band, below 0.05 -> P3 FALSIFIED with the decision-rule note
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.8300, losses=[round(2.03 - 0.01 * i, 5) for i in range(20)])
    F = run(R)
    assert F["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "COMPARABLE"
    pe = next(x for x in score_predictions({"qwen3": F}) if x[0] == "P3")
    assert pe[2] == "FALSIFIED" and "does not fire" in pe[3], pe
    cases += 1
    # 17. the status->verdict map, c1_failed -> VOID, a missing receipt -> NOT_RUN
    for st, want in (("oom", "OOM"), ("refused", "UNSUPPORTED"), ("install_failed", "UNSUPPORTED"), ("load_fault", "UNSUPPORTED"), ("verify_failed", "UNSUPPORTED"),
                     ("alarm", "ALARM"), ("phase_alarm", "ALARM"), ("harness_error", "HARNESS_ERROR"), ("tokens_mismatch", "HARNESS_ERROR"), ("not_run", "NOT_RUN")):
        R = _good_set()
        R[("unsloth", "ckpt_unsloth_m")] = _stub("unsloth", "ckpt_unsloth_m", "unsloth", st)
        assert row(run(R), "unsloth", "ckpt_unsloth_m")["verdict"] == want, (st, want)
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = {**_receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81), "status": "c1_failed"}
    assert row(run(R), "unsloth", "ckpt_unsloth_m")["verdict"] == "VOID"
    assert row(run({}), "unsloth", "ckpt_unsloth_m")["verdict"] == "NOT_RUN"
    cases += 1
    # 18. a missing second draw: stability UNMEASURED, position not quoted, P2 UNTESTED for that pair
    R = _good_set()
    del R[("unsloth", "ckpt_unsloth_m_d2")]
    F = run(R)
    assert F["draws"][("unsloth", "ckpt_unsloth_m")]["verdict"] == "UNMEASURED" and not F["positions"]["unsloth"]["quoted"]
    assert P({"qwen3": F})["P1"] == "UNTESTED" and P({"qwen3": F})["P2"] == "HELD"
    cases += 1
    # 19. frozen base: DIFFERENT expert bytes -> P10 FALSIFIED; a missing slot -> N-A
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")]["frozen_base_probe"]["slots"]["gate_up"]["sha"] = "z" * 64
    F = run(R)
    assert F["frozen"][("unsloth", "ckpt_unsloth_m")]["gate_up"]["reading"] == "DIFFERENT" and P({"qwen3": F})["P10"] == "FALSIFIED"
    R = _good_set()
    del R[("unsloth", "ckpt_unsloth_m")]["frozen_base_probe"]["slots"]["down"]
    F = run(R)
    assert F["frozen"][("unsloth", "ckpt_unsloth_m")]["down"]["reading"] == "N-A" and P({"qwen3": F})["P10"] == "UNTESTED"
    cases += 1
    # 20. the other predictions' FALSIFIED legs: P1 below 1.5, P4 matched as good (native box), P5 hf trains, P7 best too fast, P8 host-bound, P1b off by 20 %
    R = _good_set()
    for k in (("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_d2")):
        R[k]["s_per_step_median_11plus"] = 1.2
    pe = next(x for x in score_predictions({"qwen3": run(R)}) if x[0] == "P1")
    assert pe[2] == "FALSIFIED" and "BELOW 1.5" in pe[3], pe
    NR = _native_set()
    NR[("e4b", "fused_attn4_shipped")]["eval_curve"][1]["heldout_loss"] = 1.8000
    assert P(both(NR=NR))["P4"] == "FALSIFIED"
    R = _good_set()
    R[("hf", "hf_peft_m")] = _receipt("hf", "hf_peft_m", "hf", s=6.0, heldout_n=1.81)
    assert P({"qwen3": run(R)})["P5"] == "FALSIFIED"
    NR = _native_set()
    NR[("unsloth", "ckpt_unsloth_best")]["s_per_step_median_11plus"] = 1.5
    assert P(both(NR=NR))["P7"] == "FALSIFIED"
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_prof")]["profile"]["device_busy_fraction"] = 0.3
    assert P({"qwen3": run(R)})["P8"] == "FALSIFIED"
    NR = _native_set()
    NR[("unsloth", "ckpt_unsloth_t28")]["s_per_step_median_11plus"] = 29.05 * 1.2
    assert P(both(NR=NR))["P1b"] == "FALSIFIED"
    assert P({"qwen3": run(_good_set())})["P1b"] == "UNTESTED" and P({"qwen3": run(_good_set())})["P4"] == "UNTESTED"   # no native box ran
    cases += 1
    # 21. axolotl trains: P6 reads the ratio; the mb1 secondary pair when the primary OOMed
    R = _good_set()
    R[("axolotl", "ckpt_axolotl_m")] = _receipt("axolotl", "ckpt_axolotl_m", "axolotl", s=4.0, heldout_n=1.81)
    F = run(R)
    assert F["positions"]["axolotl"]["quoted"] and P({"qwen3": F})["P6"] == "HELD"
    R[("axolotl", "ckpt_axolotl_m")]["s_per_step_median_11plus"] = 9.0
    assert P({"qwen3": run(R)})["P6"] == "FALSIFIED"
    R = _good_set()
    R[("e4b", "fused_attn4_m_mb1")] = _receipt("e4b", "fused_attn4_m_mb1", "fused", s=1.1, accum=8, micro_batch=1, kernel_calls_per_step_min=2 * 48 * 8)
    R[("hf", "hf_peft_m_mb1")] = _receipt("hf", "hf_peft_m_mb1", "hf", s=7.7, heldout_n=1.81, accum=8, micro_batch=1, experts_forward_calls_per_step_min=48 * 8)
    F = run(R)
    assert F["secondary"]["hf"]["quoted"] and abs(F["secondary"]["hf"]["ratio"] - 7.0) < 1e-9, F["secondary"]["hf"]
    cases += 1
    # 22. end to end through the files: both family tokens, the printer renders every section
    d = tempfile.mkdtemp(prefix="tc1_reduce_selftest_")
    for (fw, tag), r in _good_set().items():
        json.dump(r, open(os.path.join(d, f"qwen3_{fw}_{tag}.json"), "w"))
    for (fw, tag), r in _native_set().items():
        json.dump(r, open(os.path.join(d, f"qwen3native_{fw}_{tag}.json"), "w"))
    open(os.path.join(d, "summary.txt"), "w").write("qwen3/hf/hf_peft_m rc=5 CELL OOM\n")
    F = reduce_dir(d, 20)
    assert set(F) == {"qwen3", "qwen3native"}
    text = render(F, d)
    for needle in ("**VALID**", "**OOM**", "**UNSUPPORTED**", "MATCHED POSITION: s/step ratio unsloth/e4b = 3.000** [2.941, 3.060 over 4 cross-draw ratios]", "EQUIVALENT", "SAME-BYTES",
                   "NATIVE-BEST", "LABELLED ROW ckpt_unsloth_t28", "| P10 | qwen3 | **HELD** |", "| P1b | qwen3native | **HELD** |", "draws (R1)", "matched_init_sha (B", "paired rows mean", "draw-noise floor"):
        assert needle in text, needle
    cases += 1
    # 23. the registered arm order is the support table's row order; unexpected receipts follow, never vanish
    R = _good_set()
    R[("hf", "hf_peft_m_mb1")] = _stub("hf", "hf_peft_m_mb1", "hf", "oom")
    order = [(x["fw"], x["tag"]) for x in run(R)["rows"]]
    assert order[:len(EXPECTED["qwen3"])] == EXPECTED["qwen3"] and order[len(EXPECTED["qwen3"]):] == [("hf", "hf_peft_m_mb1")], order
    assert [(x["fw"], x["tag"]) for x in run(_native_set(), fam="qwen3native")["rows"]] == EXPECTED["qwen3native"]
    assert set(VERDICTS) == {"VALID", "VOID", "QUALITY_FAIL", "OOM", "UNSUPPORTED", "HARNESS_ERROR", "ALARM", "NOT_RUN"}
    cases += 1
    # 24. R8: a grouped_mm request whose per-expert loop ran, or whose counter is short, or no counters at all -> VOID
    for name, over in (("loop", {"unsloth_backend_calls_per_step_max": {"unsloth_grouped_mm": 192, "unsloth_triton": 0, "unsloth_loop": 1, "moe_bnb4bit_backend": 192}}),
                       ("short", {"unsloth_backend_calls_per_step_min": {"unsloth_grouped_mm": 100, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": 192}}),
                       ("nocounters", {"unsloth_backend_calls_per_step_min": None})):
        R = _good_set()
        R[("unsloth", "ckpt_unsloth_m")] = {**R[("unsloth", "ckpt_unsloth_m")], **over}
        F = run(R)
        x = row(F, "unsloth", "ckpt_unsloth_m")
        assert x["verdict"] == "VOID" and ("grouped_mm" in x["why"] or "counters" in x["why"]), (name, x["why"])
    NR = _native_set()
    NR[("unsloth", "ckpt_unsloth_triton")] = _receipt("unsloth", "ckpt_unsloth_triton", "unsloth", s=4.0, heldout_n=1.81, backend="unsloth_triton",
                                                       unsloth_backend_calls_per_step_min={"unsloth_grouped_mm": 192, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": 192})
    NF = run(NR, fam="qwen3native")
    assert row(NF, "unsloth", "ckpt_unsloth_triton")["verdict"] == "VOID" and "unsloth_triton engaged 0" in row(NF, "unsloth", "ckpt_unsloth_triton")["why"]
    assert row(run(_native_set(), fam="qwen3native"), "unsloth", "ckpt_unsloth_t28")["verdict"] == "VALID"
    cases += 1
    # 25. R8: axolotl with quantize_moe_experts but fewer bnb-parametrized experts modules than layers -> VOID
    R = _good_set()
    R[("axolotl", "ckpt_axolotl_m")] = _receipt("axolotl", "ckpt_axolotl_m", "axolotl", s=4.0, heldout_n=1.81, axolotl_bnb4bit_modules={"n_bnb4bit_unwrapped": 10})
    assert row(run(R), "axolotl", "ckpt_axolotl_m")["verdict"] == "VOID" and "bnb-parametrized" in row(run(R), "axolotl", "ckpt_axolotl_m")["why"]
    cases += 1
    # 26. P8 on a non-grouped_mm profiled arm -> UNTESTED
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_prof")]["unsloth_knobs"]["moe_backend_requested"] = "native_torch"
    R[("unsloth", "ckpt_unsloth_prof")]["unsloth_backend_calls_per_step_min"] = {"unsloth_grouped_mm": 0, "unsloth_triton": 0, "unsloth_loop": 192, "moe_bnb4bit_backend": 192}
    R[("unsloth", "ckpt_unsloth_prof")]["unsloth_grouped_mm_calls_per_step_min"] = 0
    assert P({"qwen3": run(R)})["P8"] == "UNTESTED"
    cases += 1
    # 27. A [F1]: a fused arm whose kernel took the per-expert loop on one step -> VOID naming the step; no path counters -> VOID
    R = _good_set()
    R[("e4b", "fused_attn4_m")] = {**R[("e4b", "fused_attn4_m")], "lora_path_loop_steps": [3], "lora_loop_share": [0.0] * 2 + [0.25] + [0.0] * 17}
    F = run(R)
    x = row(F, "e4b", "fused_attn4_m")
    assert x["verdict"] == "VOID" and "step(s) [3]" in x["why"] and not F["positions"]["unsloth"]["quoted"], x["why"]
    print("FAILING-CASE A (reducer):", x["why"])
    R = _good_set()
    R[("e4b", "fused_attn4_m")] = {**R[("e4b", "fused_attn4_m")], "lora_path_present": False, "lora_path_loop_steps": None, "lora_loop_share": None}
    assert row(run(R), "e4b", "fused_attn4_m")["verdict"] == "VOID" and "lora_path_present" in row(run(R), "e4b", "fused_attn4_m")["why"]
    cases += 1
    # 28. B [F3]: a matched arm whose name-free slot sha differs from the anchor's -> VOID (the matched set cannot be read); missing -> VOID
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81, sha="x" * 64)
    F = run(R)
    x = row(F, "unsloth", "ckpt_unsloth_m")
    assert x["verdict"] == "VOID" and "matched_init_sha" in x["why"] and F["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "N-A", x["why"]
    print("FAILING-CASE B (reducer):", x["why"])
    R = _good_set()
    R[("e4b", "reference_attn4_m")]["matched_init_sha"] = None
    assert row(run(R), "e4b", "reference_attn4_m")["verdict"] == "VOID"
    cases += 1
    # 29. D [F14]: a receipt whose control was not run on real storage (no C1_control_tensor) is not clean, whatever its boolean says
    R = _good_set()
    R[("e4b", "fused_attn4_m")] = {**R[("e4b", "fused_attn4_m")], "C1_control_tensor": None}
    x = row(run(R), "e4b", "fused_attn4_m")
    assert x["verdict"] == "VOID" and "no C1_control_tensor" in x["why"], x["why"]
    print("FAILING-CASE D (reducer):", x["why"])
    cases += 1
    # 30. E [F15]: a grouped_mm arm with too few torch._grouped_mm calls, or any manual-fallback call, or no torch-level counters -> VOID
    for name, over in (("short", {"unsloth_grouped_mm_calls_per_step_min": 6 * 48 * 4 - 1}), ("manual", {"unsloth_manual_grouped_mm_calls_per_step_max": 2}), ("none", {"unsloth_grouped_mm_calls_per_step_min": None})):
        R = _good_set()
        R[("unsloth", "ckpt_unsloth_m")] = {**R[("unsloth", "ckpt_unsloth_m")], **over}
        x = row(run(R), "unsloth", "ckpt_unsloth_m")
        assert x["verdict"] == "VOID" and ("_grouped_mm" in x["why"] or "fallback" in x["why"]), (name, x["why"])
        if name == "manual":
            print("FAILING-CASE E (reducer):", x["why"])
    cases += 1
    # 31. F [F16]: a recompile between the step-10 and step-N snapshots marks the draw UNSTABLE -> no position
    R = _good_set()
    R[("e4b", "fused_attn4_m_d2")]["dynamo_counters"]["step20"]["recompiles_total"] = 1
    F = run(R)
    dw = F["draws"][("e4b", "fused_attn4_m")]
    assert dw["verdict"] == "UNSTABLE" and dw["recompiles"] == [0, 1] and not F["positions"]["unsloth"]["quoted"], dw
    print("FAILING-CASE F (reducer):", dw["why"])
    cases += 1
    # 32. G [F13]: the draw-noise floor -- when the two Unsloth draws differ at N by more than the cross-arm |delta|, the reading is INSIDE-DRAW-NOISE, not EQUIVALENT; P3 still reads it as held
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m_d2")] = _receipt("unsloth", "ckpt_unsloth_m_d2", "unsloth", s=3.06, heldout_n=1.8300)
    F = run(R)
    e = F["equivalence"][("unsloth", "ckpt_unsloth_m")]
    assert abs(F["noise_floor"] - 0.02) < 1e-9 and e["reading"] == "INSIDE-DRAW-NOISE" and "inside the draw noise" in e["why"], (F["noise_floor"], e)
    assert P({"qwen3": F})["P3"] == "HELD"
    print("FAILING-CASE G (reducer): reading", e["reading"], "--", e["why"])
    cases += 1
    # 33. H: without a reference arm the band cannot be set -> COMPARABLE is the ceiling (never EQUIVALENT), P3 UNTESTED
    R = _good_set()
    del R[("e4b", "reference_attn4_m")]
    F = run(R)
    e = F["equivalence"][("unsloth", "ckpt_unsloth_m")]
    assert F["equiv_band"] is None and e["reading"] == "COMPARABLE" and "no reference arm" in e["why"] and P({"qwen3": F})["P3"] == "UNTESTED", (e, P({"qwen3": F})["P3"])
    print("FAILING-CASE H (reducer): no reference arm ->", e["reading"], "--", e["why"])
    cases += 1
    # ----------------------------------------------------------------------- R10: lane TC1b (the qwen3curve token)
    def crun(R, tc1=None):
        return reduce_curve_family(CURVE_FAM, R, {}, tc1=tc1)

    def CP(F):
        return {p_: v for p_, _, v, _ in score_curve_predictions(F)}
    # 34. baseline: every arm VALID in its sub-fixture, the curve EQUIVALENT-AT-EVERY-EVAL, the plateau REPRODUCES-P38, the target reached by every arm,
    #     the anchor AGREES with tp2 (and P38), the t28 variant read separately, both scaling points quoted, P1/P2/P4 HELD and P3 UNTESTED without --tc1-dir
    C = crun(_curve_set())
    assert [(x["fw"], x["tag"]) for x in C["rows"]] == EXPECTED[CURVE_FAM]
    assert all(v == "VALID" for v in C["verdicts"].values()), C["verdicts"]
    assert C["groups"]["p38"]["trainable"] == 321257472 and C["groups"]["200"]["trainable"] == 642514944 and C["groups"]["r64"]["sha"] == ("s" * 60 + "r64").ljust(64, "s")
    assert row(C, "unsloth", "ckpt_unsloth_m_r64")["step0_class"] == "SAME-BYTES-CLASS" and row(C, "unsloth", "ckpt_unsloth_p38")["step0_class"] is None   # native rows carry no R3
    assert row(C, "e4b", "fused_attn4_shipped_200")["quality_delta"] is not None and abs(row(C, "e4b", "fused_attn4_shipped_200")["quality_delta"] - 0.015) < 1e-9
    tb = C["curve_table"]
    assert [row_["step"] for row_ in tb] == [0, 40, 80, 120, 160, 200] and all(row_["d12"]["n"] == 16 and abs(row_["d12"]["mean"] - 0.005) < 1e-9 for row_ in tb), tb[1]["d12"]
    assert abs(tb[1]["arms"][("e4b", "fused_attn4_m_200")][0] - 1.90) < 1e-9 and tb[1]["arms"][("e4b", "fused_attn4_m_200")][1] is not None and tb[1]["arms"][("e4b", "fused_attn4_m_200")][1] > 0
    c = C["curve"]
    assert c["reading"] == "EQUIVALENT-AT-EVERY-EVAL" and abs(c["max_abs"] - 0.005) < 1e-9 and c["sign_at_N"] == "+" and c["first_divergent_step"] is None and c["band"] == CURVE_BAND and "draft" in c["band_note"], c
    pl = C["plateau"]
    assert pl["reading"] == "REPRODUCES-P38" and abs(pl["gap_N"] - 0.015) < 1e-9 and abs(pl["gap_early"] + 0.01) < 1e-9 and pl["paired_N"]["n"] == 16, pl
    t = C["ttt"]
    assert abs(t["target"] - (statistics.mean([1.79, 1.795]) + 0.02)) < 1e-9 and all(a["reached"] for a in t["arms"].values())
    assert t["arms"][("e4b", "fused_attn4_m_200")]["step"] == 160 and t["arms"][("e4b", "fused_attn4_m_200")]["train_wall_s"] == 160.0 and t["arms"][("e4b", "fused_attn4_shipped_200")]["step"] == 160, t["arms"]
    sp = C["speed"]
    assert all(e["reading"] == "UNTESTED" and e["tc1_why"] == "no --tc1-dir given" for e in sp.values()) and abs(sp[("unsloth", "ckpt_unsloth_m_200")]["s_11_20_in_receipt"] - 3.0) < 1e-9 and sp[("unsloth", "ckpt_unsloth_m_200")]["in_receipt_dev"] == 0.0
    an = C["anchor"]
    assert an["grouped_mm"]["quoted"] and abs(an["grouped_mm"]["ratio"] - 1.45) < 1e-9 and an["grouped_mm"]["reading"] == "AGREES" and an["grouped_mm"]["agrees_p38"] and an["grouped_mm"]["kind"] == "anchor", an["grouped_mm"]
    assert an["t28"]["quoted"] and abs(an["t28"]["ratio"] - 1.425) < 1e-9 and an["t28"]["reading"] == "AGREES" and "tp4's arm byte-for-byte" in an["t28"]["label"]
    assert "tp4" not in an["grouped_mm"] or TP4_ANCHOR_RATIO            # the tp4 leg appears only when the number is supplied
    sc = C["scaling"]
    assert sc["t1"]["quoted"] and abs(sc["t1"]["ratio"] - 2.0) < 1e-9 and sc["t1"]["kind"] == "scaling point" and all(ok for _, ok, _ in sc["t1"]["predicates"]) and len(sc["t1"]["predicates"]) == 4
    assert sc["r64"]["quoted"] and abs(sc["r64"]["ratio"] - 3.25) < 1e-9 and "position" not in sc["r64"]["label"]
    assert C["equivalence"][("unsloth", "ckpt_unsloth_m_200")]["reading"] == "COMPARABLE" and "no reference arm" in C["equivalence"][("unsloth", "ckpt_unsloth_m_200")]["why"]
    assert CP({CURVE_FAM: C}) == {"P1": "HELD", "P2": "HELD", "P3": "UNTESTED", "P4": "HELD"}, CP({CURVE_FAM: C})
    assert c["tc1_floor"]["equiv_floor"] == EQUIV_FLOOR and c["tc1_floor"]["tc1_heldout_band"] is None and "no --tc1-dir" in c["tc1_floor"]["text"]
    p4 = next(x for x in score_curve_predictions({CURVE_FAM: C}) if x[0] == "P4")
    assert "NOT in this tree" in p4[3] and "t28 variant 1.425" in p4[3], p4
    cases += 1
    # 35. a DIVERGENT curve: Unsloth's rows sit +0.03 above e4b's from step 120 on -> DIVERGENT, first divergent step 120, sign + at 200, P1 FALSIFIED
    R = _curve_set()
    H1 = [2.0, 1.90, 1.85, 1.82, 1.80, 1.79]
    R[("unsloth", "ckpt_unsloth_m_200")] = _curve_receipt("unsloth", "ckpt_unsloth_m_200", "unsloth", s=3.00, heldouts=[round(h + (0.03 if i >= 3 else 0.005), 5) for i, h in enumerate(H1)])
    C = crun(R)
    c = C["curve"]
    assert c["reading"] == "DIVERGENT" and c["first_divergent_step"] == 120 and c["sign_at_N"] == "+" and abs(c["max_abs"] - 0.03) < 1e-9 and c["max_step"] in (120, 160, 200), c
    assert CP({CURVE_FAM: C})["P1"] == "FALSIFIED" and "step 120" in next(x for x in score_curve_predictions({CURVE_FAM: C}) if x[0] == "P1")[3]
    print(f"FAILING-CASE TC1b-P1 (reducer): curve {c['reading']} -- first paired |delta| > {CURVE_BAND} at step {c['first_divergent_step']}, max {c['max_abs']:.4f} at step {c['max_step']}, sign at 200 {c['sign_at_N']}")
    R2 = _curve_set()                                   # the sign reads the other way too
    R2[("unsloth", "ckpt_unsloth_m_200")] = _curve_receipt("unsloth", "ckpt_unsloth_m_200", "unsloth", s=3.00, heldouts=[round(h - 0.025, 5) for h in H1])
    assert crun(R2)["curve"]["sign_at_N"] == "-" and crun(R2)["curve"]["first_divergent_step"] == 0
    cases += 1
    # 36. the plateau refuted both ways: as-shipped within 0.01 at 200 -> NOT-P38-SHAPE; as-shipped above at 40 as well -> NOT-P38-SHAPE (no early lead); P2 FALSIFIED
    R = _curve_set()
    R[("e4b", "fused_attn4_shipped_200")] = _curve_receipt("e4b", "fused_attn4_shipped_200", "fused", s=0.85, heldouts=[2.0, 1.89, 1.84, 1.815, 1.80, 1.795], native=True)
    C = crun(R)
    assert C["plateau"]["reading"] == "NOT-P38-SHAPE" and "within" in C["plateau"]["why"] and CP({CURVE_FAM: C})["P2"] == "FALSIFIED", C["plateau"]
    print("FAILING-CASE TC1b-P2 (reducer):", C["plateau"]["reading"], "--", C["plateau"]["why"])
    R[("e4b", "fused_attn4_shipped_200")] = _curve_receipt("e4b", "fused_attn4_shipped_200", "fused", s=0.85, heldouts=[2.0, 1.91, 1.86, 1.83, 1.81, 1.805], native=True)
    C = crun(R)
    assert C["plateau"]["reading"] == "NOT-P38-SHAPE" and "ABOVE" in C["plateau"]["why"] and CP({CURVE_FAM: C})["P2"] == "FALSIFIED", C["plateau"]
    cases += 1
    # 37. a failing anchor: the p38 pair at 1.85 (outside +-10 % of both tp2 and P38) -> FINDING, P4 FALSIFIED; the t28 variant is read on its own
    R = _curve_set()
    R[("unsloth", "ckpt_unsloth_p38")]["s_per_step_median_11plus"] = 7.4
    C = crun(R)
    an = C["anchor"]
    assert an["grouped_mm"]["quoted"] and abs(an["grouped_mm"]["ratio"] - 1.85) < 1e-9 and an["grouped_mm"]["reading"] == "FINDING" and not an["grouped_mm"]["agrees_p38"]
    assert an["t28"]["reading"] == "AGREES" and CP({CURVE_FAM: C})["P4"] == "FALSIFIED"
    print(f"FAILING-CASE TC1b-P4 (reducer): anchor unsloth/e4b {an['grouped_mm']['ratio']:.3f} -> {an['grouped_mm']['reading']} (tp2 {an['grouped_mm']['dev_tp2']:+.1%}, P38 {an['grouped_mm']['dev_p38']:+.1%}; +-{ANCHOR_TOL:.0%})")
    R = _curve_set()
    R[("e4b", "fused_attn4_p38")] = _stub("e4b", "fused_attn4_p38", "fused", "oom")
    C = crun(R)
    assert not C["anchor"]["grouped_mm"]["quoted"] and not C["anchor"]["t28"]["quoted"] and CP({CURVE_FAM: C})["P4"] == "UNTESTED" and row(C, "unsloth", "ckpt_unsloth_p38")["verdict"] == "VALID"
    cases += 1
    # 38. a VOID r64 pair: the two r64 arms' trainable counts differ -> the Unsloth arm VOID, the scaling point not quoted with the predicate named; the t1 pair unaffected
    R = _curve_set()
    R[("unsloth", "ckpt_unsloth_m_r64")]["trainable_params"] = 4 * 642514944 + 1
    C = crun(R)
    x = row(C, "unsloth", "ckpt_unsloth_m_r64")
    assert x["verdict"] == "VOID" and "trainable" in x["why"] and not C["scaling"]["r64"]["quoted"] and C["scaling"]["t1"]["quoted"], (x["why"], C["scaling"]["r64"]["why"])
    assert [ok for _, ok, _ in C["scaling"]["r64"]["predicates"]][:2] == [False, False]
    print("FAILING-CASE TC1b-r64 (reducer):", x["why"], "->", C["scaling"]["r64"]["why"])
    R = _curve_set()                                    # the e4b r64 arm's kernel took the loop on one step -> VOID, the scaling point names the predicate
    R[("e4b", "fused_attn4_m_r64")].update({"lora_path_loop_steps": [7], "lora_loop_share": [0.0] * 6 + [0.5] + [0.0] * 13})
    C = crun(R)
    assert row(C, "e4b", "fused_attn4_m_r64")["verdict"] == "VOID" and not C["scaling"]["r64"]["quoted"] and "lora_path_loop" in C["scaling"]["r64"]["why"]
    R = _curve_set()                                    # the Unsloth t1 arm's manual grouped-mm fallback ran -> VOID, the scaling point not quoted
    R[("unsloth", "ckpt_unsloth_m_t1")]["unsloth_manual_grouped_mm_calls_per_step_max"] = 3
    C = crun(R)
    assert row(C, "unsloth", "ckpt_unsloth_m_t1")["verdict"] == "VOID" and not C["scaling"]["t1"]["quoted"]
    cases += 1
    # 39. the registered trainable counts: an arm-1 count off by one VOIDs it (and the curve reading with it); the anchor arms read against r 8's count
    R = _curve_set()
    R[("e4b", "fused_attn4_m_200")]["trainable_params"] = 642514945
    C = crun(R)
    assert row(C, "e4b", "fused_attn4_m_200")["verdict"] == "VOID" and "registered 642514944" in row(C, "e4b", "fused_attn4_m_200")["why"]
    assert C["curve"]["reading"] == "N-A" and CP({CURVE_FAM: C})["P1"] == "UNTESTED" and C["plateau"]["reading"] == "N-A"
    R = _curve_set()
    R[("e4b", "fused_attn4_p38")]["trainable_params"] = 642514944
    C = crun(R)
    assert row(C, "e4b", "fused_attn4_p38")["verdict"] == "VOID" and "registered 321257472" in row(C, "e4b", "fused_attn4_p38")["why"]
    cases += 1
    # 40. (d) with --tc1-dir: the curve arms read beside TC1's quoted draws -- TRAVELS at 1.00 vs 1.01 / 3.00 vs 3.03 / 0.85 vs 0.85; a 3.5 s/step Unsloth curve arm DOES-NOT-TRAVEL, P3 FALSIFIED
    tc1 = {"qwen3": run(_good_set()), "qwen3native": run(_native_set(), fam="qwen3native")}
    C = crun(_curve_set(), tc1=tc1)
    sp = C["speed"]
    assert all(e["reading"] == "TRAVELS" for e in sp.values()) and abs(sp[("unsloth", "ckpt_unsloth_m_200")]["tc1_s"] - 3.03) < 1e-9 and sp[("unsloth", "ckpt_unsloth_m_200")]["tc1_draws"] == 2, sp
    assert abs(sp[("e4b", "fused_attn4_shipped_200")]["tc1_s"] - 0.85) < 1e-9 and sp[("e4b", "fused_attn4_shipped_200")]["tc1_key"][0] == "qwen3native"
    assert CP({CURVE_FAM: C})["P3"] == "HELD"
    assert abs(C["curve"]["tc1_floor"]["tc1_heldout_band"] - 0.015) < 1e-9 and "from --tc1-dir" in C["curve"]["tc1_floor"]["text"]   # TC1's measured band quoted beside the fixed 0.02
    R = _curve_set()
    R[("unsloth", "ckpt_unsloth_m_200")]["s_per_step_median_11plus"] = 3.5
    C = crun(R, tc1=tc1)
    assert C["speed"][("unsloth", "ckpt_unsloth_m_200")]["reading"] == "DOES-NOT-TRAVEL" and CP({CURVE_FAM: C})["P3"] == "FALSIFIED"
    print(f"FAILING-CASE TC1b-P3 (reducer): unsloth 11..200 median {C['speed'][('unsloth', 'ckpt_unsloth_m_200')]['s_200']:.3f} vs TC1's 11..20 {C['speed'][('unsloth', 'ckpt_unsloth_m_200')]['tc1_s']:.3f} ({C['speed'][('unsloth', 'ckpt_unsloth_m_200')]['dev']:+.1%}) -> DOES-NOT-TRAVEL")
    C = crun(_curve_set(), tc1={"qwen3": run(_good_set())})          # no native box in the TC1 dir: the shipped arm UNTESTED, the pair still read
    assert C["speed"][("e4b", "fused_attn4_shipped_200")]["reading"] == "UNTESTED" and "no qwen3native receipts" in C["speed"][("e4b", "fused_attn4_shipped_200")]["tc1_why"] and CP({CURVE_FAM: C})["P3"] == "HELD"
    cases += 1
    # 41. (c) a target never reached: the as-shipped arm ends above the matched pair's step-200 held-out + 0.02 -> "not reached"; a single VALID matched arm sets the target alone
    R = _curve_set()
    R[("e4b", "fused_attn4_shipped_200")] = _curve_receipt("e4b", "fused_attn4_shipped_200", "fused", s=0.85, heldouts=[2.0, 1.95, 1.90, 1.86, 1.84, 1.83], native=True)
    C = crun(R)
    a3 = C["ttt"]["arms"][("e4b", "fused_attn4_shipped_200")]
    assert a3["reached"] is False and "not reached by step 200" in a3["why"], a3
    R[("unsloth", "ckpt_unsloth_m_200")] = _stub("unsloth", "ckpt_unsloth_m_200", "unsloth", "alarm")
    C = crun(R)
    assert "one VALID matched arm" in C["ttt"]["basis"] and abs(C["ttt"]["target"] - 1.81) < 1e-9 and C["curve"]["reading"] == "N-A" and CP({CURVE_FAM: C})["P1"] == "UNTESTED"
    cases += 1
    # 42. a matched Unsloth 200 arm whose sha differs from its sub-fixture's e4b arm -> VOID; the curve N-A; a grid that stops short of 200 -> N-A
    R = _curve_set()
    R[("unsloth", "ckpt_unsloth_m_200")]["matched_init_sha"] = "x" * 64
    C = crun(R)
    assert row(C, "unsloth", "ckpt_unsloth_m_200")["verdict"] == "VOID" and C["curve"]["reading"] == "N-A"
    R = _curve_set()
    R[("unsloth", "ckpt_unsloth_m_200")]["eval_rows"] = R[("unsloth", "ckpt_unsloth_m_200")]["eval_rows"][:-1]
    C = crun(R)
    assert C["curve"]["reading"] == "N-A" and "200" in C["curve"]["why"], C["curve"]
    cases += 1
    # 43. end to end through the files: the qwen3curve token alone renders its block and the TC1b table (and NOT TC1's P1-P10 table); with a --tc1-dir the (d) row reads TRAVELS
    cd = tempfile.mkdtemp(prefix="tc1b_reduce_selftest_")
    for (fw, tag), r in _curve_set().items():
        json.dump(r, open(os.path.join(cd, f"{CURVE_FAM}_{fw}_{tag}.json"), "w"))
    F = reduce_dir(cd, 20)
    assert set(F) == {CURVE_FAM} and [(x["fw"], x["tag"]) for x in F[CURVE_FAM]["rows"]] == EXPECTED[CURVE_FAM]
    text = render(F, cd)
    for needle in ("Lane TC1b", "**(a) curve table**", "| 200 |", "CURVE READING (arms 1 vs 2): EQUIVALENT-AT-EVERY-EVAL", "the draft's band", "PLATEAU TEST (arm 3 − arm 1): REPRODUCES-P38",
                   "(c) time to target", "step 160 at 160.0 s train wall", "(d) s/step medians over steps 11..200", "NOT given: UNTESTED", "ANCHOR grouped_mm: s/step ratio unsloth/e4b = 1.450 → AGREES",
                   "ANCHOR t28: s/step ratio unsloth/e4b = 1.425 → AGREES", "SCALING POINT `t1` (never a position): s/step ratio unsloth/e4b = 2.000", "SCALING POINT `r64` (never a position): s/step ratio unsloth/e4b = 3.250",
                   "| P1 | qwen3curve | **HELD** |", "| P2 | qwen3curve | **HELD** |", "| P3 | qwen3curve | **UNTESTED** |", "| P4 | qwen3curve | **HELD** |", "matched_init_sha per sub-fixture"):
        assert needle in text, needle
    assert "## Predictions P1–P10" not in text and "MATCHED POSITION" not in text and "POSITION:" not in text
    for (fw, tag), r in _good_set().items():
        json.dump(r, open(os.path.join(d, f"qwen3_{fw}_{tag}.json"), "w"))
    F = reduce_dir(cd, 20, tc1_dir=d)
    text = render(F, cd)
    assert "**TRAVELS**" in text and "| P3 | qwen3curve | **HELD** |" in text and F[CURVE_FAM]["tc1_given"]
    for (fw, tag), r in _curve_set().items():                      # the two tokens side by side: both tables, both blocks
        json.dump(r, open(os.path.join(d, f"{CURVE_FAM}_{fw}_{tag}.json"), "w"))
    text = render(reduce_dir(d, 20), d)
    assert "## Predictions P1–P10" in text and "## TC1b predictions P1–P4" in text and "Lane TC1b" in text
    cases += 1
    # ----------------------------------------------------------------------- R11: lane TC2 (the tc2small / tc2big tokens)
    def trun(fam, R=None):
        return reduce_family(fam, R if R is not None else _tc2_set(fam), {}, None)

    def TP(F):
        return {p_: v for p_, _, v, _ in score_tc2_predictions(F)}
    T_ALL = {fam: trun(fam) for fam in TC2_FAMS}
    # 44. the baseline per family: the registered order, the verdicts, the HF position spelled "HF (bf16 experts) / e4b" with two draws, every prediction HELD
    G = T_ALL["granite"]
    for fam in TC2_FAMS:
        assert [(x["fw"], x["tag"]) for x in T_ALL[fam]["rows"]] == EXPECTED[fam], fam
    assert G["verdicts"][("e4b", "fused_attn4_m")] == "VALID" and G["verdicts"][("hf", "hf_peft_m")] == "VALID" and G["verdicts"][("hf", "hf_peft_m_d2")] == "VALID" and G["verdicts"][("e4b", "reference_attn4_m")] == "VALID"
    assert G["verdicts"][("unsloth", "ckpt_unsloth_m")] == "VOID" and G["verdicts"][("unsloth", "ckpt_unsloth_m_experts")] == "VOID"
    assert G["verdicts"][("hf", "hf_peft_m_t214")] == "VALID" and G["verdicts"][("axolotl", "ckpt_axolotl_m")] == "VALID" and G["verdicts"][("axolotl", "ckpt_axolotl_best")] == "VALID" and G["verdicts"][("e4b", "fused_attn4_shipped")] == "VALID"
    ph = G["positions"]["hf"]
    assert ph["quoted"] and ph["label"] == "HF (bf16 experts)" and ph["e4b_draws"] == 2 and ph["other_draws"] == 2 and abs(ph["ratio"] - 3.0735 / 2.378) < 1e-9 and "bf16 experts" in ph["other_regime"], ph
    assert G["draws"][("hf", "hf_peft_m")]["verdict"] == "STABLE" and G["parity"]["verdict"] == "PASS" and G["common_set"] is True and G["anchor_key"] == ("e4b", "fused_attn4_m")
    assert "HF (bf16 experts) / e4b = 1.292**" in pos_lines(ph, 60, prefix="MATCHED POSITION")[0]
    assert TP(T_ALL) == {f"P{i}": "HELD" for i in range(1, 8)}, TP(T_ALL)
    cases += 1
    # 45. a VOID attention-only Unsloth arm (granite): the reason names it first, no position, the labelled _experts arm likewise; a same-count arm stays VALID; a different count WITH experts is plain VOID
    x = row(G, "unsloth", "ckpt_unsloth_m")
    assert x["verdict"] == "VOID" and x["why"].startswith("attention-only: trainable 5200000 != e4b's 99600000 (unsloth adapted no expert parameter"), x["why"]
    assert not G["positions"]["unsloth"]["quoted"] and "VOID" in G["positions"]["unsloth"]["why"] and not G["labelled"][("unsloth", "ckpt_unsloth_m_experts")]["quoted"]
    print("FAILING-CASE TC2-attention-only (reducer):", x["why"])
    R = _tc2_set("granite")
    R[("unsloth", "ckpt_unsloth_m")] = _tc2_receipt("granite", "unsloth", "ckpt_unsloth_m", "unsloth", s=2.9, heldout_n=1.8004)
    assert row(trun("granite", R), "unsloth", "ckpt_unsloth_m")["verdict"] == "VALID"
    R[("unsloth", "ckpt_unsloth_m")]["trainable_params"] = 99600001
    y = row(trun("granite", R), "unsloth", "ckpt_unsloth_m")
    assert y["verdict"] == "VOID" and "attention-only" not in y["why"] and y["why"].startswith("trainable 99600001 != e4b's 99600000"), y["why"]
    cases += 1
    # 46. gptoss: the no-common-set line replaces every position (both s/step values, both counts, no ratio); the harness-recorded mismatch and the sha / quality predicates against e4b do not apply there
    GO = T_ALL["gptoss"]
    assert GO["common_set"] is False and GO["anchor_key"] == ("e4b", "attn_only_m") and GO["labelled"] == {}
    assert GO["verdicts"][("e4b", "attn_only_m")] == "VALID" and GO["verdicts"][("e4b", "attn_only_m_d2")] == "VALID" and GO["verdicts"][("unsloth", "ckpt_unsloth_mxfp4")] == "VALID"
    assert GO["verdicts"][("unsloth", "ckpt_unsloth_mxfp4_d2")] == "VALID" and GO["verdicts"][("unsloth", "ckpt_unsloth_m")] == "VALID"
    assert GO["verdicts"][("e4b", "fused_attn4_m")] == "UNSUPPORTED" and GO["verdicts"][("e4b", "reference_attn4_m")] == "UNSUPPORTED" and GO["verdicts"][("hf", "hf_peft_m")] == "OOM" and GO["verdicts"][("axolotl", "ckpt_axolotl_m")] == "UNSUPPORTED"
    assert GO["draws"][("e4b", "attn_only_m")]["verdict"] == "STABLE" and GO["draws"][("unsloth", "ckpt_unsloth_mxfp4")]["verdict"] == "STABLE"
    ln = GO["positions"]["unsloth_mxfp4"]
    assert ln["no_common_set"] and not ln["quoted"] and "ratio" not in ln and abs(ln["e4b_s"] - 3.125) < 1e-9 and abs(ln["other_s"] - 4.05) < 1e-9, ln
    assert ln["e4b_trainable"] == 5000000 and ln["other_trainable"] == 90000000 and all(pz.get("no_common_set") and not pz.get("quoted") for pz in GO["positions"].values())
    assert set(GO["positions"]) == {"unsloth", "unsloth_mxfp4", "hf", "axolotl"} and GO["positions"]["hf"]["other_s"] is None and "OOM" in GO["positions"]["hf"]["other_state"]
    txt = pos_lines(ln, 60)[0]
    assert txt.startswith("- **NO COMMON ADAPTER SET (unsloth/ckpt_unsloth_mxfp4 vs e4b/attn_only_m) — no ratio is quoted**") and "3.125 s/step (5000000 trainable" in txt and "4.050 s/step (90000000 trainable" in txt, txt
    assert row(GO, "unsloth", "ckpt_unsloth_m")["quality"] == "N-A (no common adapter set)" and "no common adapter set with e4b: trainable 90000000 vs e4b's 5000000" in row(GO, "unsloth", "ckpt_unsloth_m")["no_common_set"]
    assert GO["equivalence"][("unsloth", "ckpt_unsloth_mxfp4")]["reading"] == "N-A" and GO["equivalence"][("e4b", "attn_only_m_d2")]["reading"] == "COMPARABLE"
    assert "per-expert Linear4bit experts (1536 Params4bit under the experts container" in row(GO, "unsloth", "ckpt_unsloth_m")["regime"]
    assert "packed experts in the checkpoint's own format (Mxfp4ExpertParam x48; load_in_4bit=False)" in row(GO, "unsloth", "ckpt_unsloth_mxfp4")["regime"] and "attention-only adapters" in row(GO, "e4b", "attn_only_m")["regime"]
    print("FAILING-CASE TC2-no-common-set (reducer): a ratio on gpt-oss is never quoted --", txt[:160])
    R = _tc2_set("gptoss")
    R[("unsloth", "ckpt_unsloth_mxfp4")]["trainable_mismatch"] = {"expected": 5000000, "got": 90000000, "by_group": {}}   # expect_of hands the arm attn_only_m's count
    assert row(trun("gptoss", R), "unsloth", "ckpt_unsloth_mxfp4")["verdict"] == "VALID"
    cases += 1
    # 47. the packed arm's VOIDs (unpacked / wrong backend / uncounted / absent / short GEMM), and the per-expert Linear4bit arm's own floor
    for name, over, needle in (("unpacked", {"census": {"expert_param_classes": {"Parameter": 48}}}, "not packed"),
                               ("backend", {"moe_backend_selected": "native_torch"}, "!= grouped_mm"),
                               ("uncounted", {"unsloth_packed_calls_per_step_min": None}, "not counted"),
                               ("absent", {"unsloth_backend_absent": ["unsloth_zoo.mxfp4_gemm.Mxfp4GroupedMM.apply: ModuleNotFoundError"]}, "not counted"),
                               ("short", {"unsloth_packed_calls_per_step_min": {"unsloth_mxfp4_grouped_mm": 24 * 4 - 1}}, "engaged 95 < 24*accum 4")):
        R = _tc2_set("gptoss")
        _merge(R[("unsloth", "ckpt_unsloth_mxfp4")], over)
        y = row(trun("gptoss", R), "unsloth", "ckpt_unsloth_mxfp4")
        assert y["verdict"] == "VOID" and needle in y["why"], (name, y["why"])
        if name == "unpacked":
            print("FAILING-CASE TC2-packed (reducer):", y["why"])
    assert TP({**T_ALL, "gptoss": trun("gptoss", R)})["P3"] == "FALSIFIED"
    R = _tc2_set("gptoss")
    _merge(R[("unsloth", "ckpt_unsloth_m")], {"census": {"Params4bit_expert_linears": 10}})
    y = row(trun("gptoss", R), "unsloth", "ckpt_unsloth_m")
    assert y["verdict"] == "VOID" and "under the experts container 10 < 2*24" in y["why"], y["why"]
    cases += 1
    # 48. mixtral's footprint line leads its block; P5 reads the pair (Unsloth/e4b, the lane's convention: tp2 0.361) and the peak ratio; its failing legs
    MX = T_ALL["mixtral"]
    fp = MX["footprint"]
    assert fp["readable"] and abs(fp["peak_ratio"] - 29.18 / 3.235) < 1e-9 and fp["sides"]["e4b"]["offload"] is True and fp["sides"]["e4b"]["basis"].startswith("quoted draws"), fp
    blk = family_block(MX)
    assert blk[1].startswith("- **FOOTPRINT (e4b under expert offload") and "peak VRAM 3.24 GB under expert offload vs Unsloth `ckpt_unsloth_m` 29.18 GB resident (×9.02 lower on e4b); s/step e4b 2.388 vs Unsloth 0.864" in blk[1], blk[1]
    assert 1 < next(i for i, ln_ in enumerate(blk) if ln_.startswith("| framework |"))
    pu = MX["positions"]["unsloth"]
    assert pu["quoted"] and abs(pu["ratio"] - 0.864 / 2.3885) < 1e-9 and pu["quality"] == "COMPARABLE", pu
    R = _tc2_set("mixtral")
    R[("unsloth", "ckpt_unsloth_m")]["peak_vram_gb"] = R[("unsloth", "ckpt_unsloth_m_d2")]["peak_vram_gb"] = 20.0
    assert TP({**T_ALL, "mixtral": trun("mixtral", R)})["P5"] == "FALSIFIED"
    R = _tc2_set("mixtral")
    R[("unsloth", "ckpt_unsloth_m")] = {**_stub("unsloth", "ckpt_unsloth_m", "unsloth", "oom"), "fam": "mixtral", "steps": 20}
    M2 = trun("mixtral", R)
    assert M2["footprint"]["readable"] is False and "OOM: no peak to read" in M2["footprint"]["sides"]["unsloth"]["basis"] and TP({**T_ALL, "mixtral": M2})["P5"] == "UNTESTED"
    print("FAILING-CASE TC2-footprint (reducer): Unsloth OOM ->", footprint_line(M2["footprint"]))
    cases += 1
    # 49. an HF t214 arm whose dispatch did not reach grouped_mm: VALID, the note on the row and the regime (never VOID); accepted=False likewise
    R = _tc2_set("granite")
    R[("hf", "hf_peft_m_t214")]["hf_experts_dispatch"] = {"requested": "grouped_mm", "accepted": True, "config": "grouped_mm", "torch_grouped_mm_calls_per_step_min": 0, "torch_F_grouped_mm_calls_per_step_min": 0, "reached_grouped_mm": False}
    y = row(trun("granite", R), "hf", "hf_peft_m_t214")
    assert y["verdict"] == "VALID" and y["dispatch"].endswith("did NOT reach grouped_mm (recorded, not VOID)") and "NOT reached" in y["regime"], (y["verdict"], y["dispatch"])
    print("FAILING-CASE TC2-dispatch (reducer):", y["dispatch"])
    R[("hf", "hf_peft_m_t214")]["hf_experts_dispatch"].update({"accepted": False, "config": "eager"})
    y = row(trun("granite", R), "hf", "hf_peft_m_t214")
    assert y["verdict"] == "VALID" and "accepted False" in y["dispatch"]
    assert row(G, "hf", "hf_peft_m_t214")["dispatch"].endswith("REACHED grouped_mm") and row(G, "hf", "hf_peft_m")["dispatch"] is None
    cases += 1
    # 50. the registered census and pin VOID a receipt that disagrees; a family without a registered census takes the receipt's own
    R = _tc2_set("granite")
    R[("e4b", "fused_attn4_m")].update({"structural_expected_n_attn4": 127, "n_attn4": 127})
    assert "structural attention census 127 != registered 128" in row(trun("granite", R), "e4b", "fused_attn4_m")["why"]
    R = _tc2_set("olmoe")
    R[("hf", "hf_peft_m")]["revision"] = "f" * 40
    assert "the registered pin" in row(trun("olmoe", R), "hf", "hf_peft_m")["why"]
    assert row(trun("qwen3_5"), "e4b", "fused_attn4_m")["verdict"] == "VALID" and ATTN_CENSUS["qwen3_5"] is None
    cases += 1
    # 51. the predictions' other legs: P1 (Unsloth adapts experts / HF outside the band), P2 (Unsloth dies -> HELD; VOID -> FALSIFIED), P4 (VOID on count -> HELD; e4b drifts), P6 FAIL, P7 COMPARABLE, UNTESTED legs
    R = _tc2_set("granite")
    R[("unsloth", "ckpt_unsloth_m")] = _tc2_receipt("granite", "unsloth", "ckpt_unsloth_m", "unsloth", s=2.9, heldout_n=1.8004)
    assert TP({**T_ALL, "granite": trun("granite", R)})["P1"] == "FALSIFIED"
    R = _tc2_set("granite")
    R[("hf", "hf_peft_m")]["s_per_step_median_11plus"] = R[("hf", "hf_peft_m_d2")]["s_per_step_median_11plus"] = 5.0
    assert TP({**T_ALL, "granite": trun("granite", R)})["P1"] == "FALSIFIED"
    R = _tc2_set("olmoe")
    R[("unsloth", "ckpt_unsloth_m")] = {**_stub("unsloth", "ckpt_unsloth_m", "unsloth", "harness_error", "died before its first step"), "fam": "olmoe", "steps": 60}
    assert TP({**T_ALL, "olmoe": trun("olmoe", R)})["P2"] == "HELD"
    R = _tc2_set("olmoe")
    R[("unsloth", "ckpt_unsloth_m")]["trainable_params"] = 1
    assert TP({**T_ALL, "olmoe": trun("olmoe", R)})["P2"] == "FALSIFIED"
    R = _tc2_set("qwen3_5")
    R[("unsloth", "ckpt_unsloth_m_experts")].update({"trainable_params": 3400000, "trainable_by_group": {"attention": 1, "experts": 0, "other": 0}})
    Q2 = trun("qwen3_5", R)
    assert row(Q2, "unsloth", "ckpt_unsloth_m_experts")["verdict"] == "VOID" and TP({**T_ALL, "qwen3_5": Q2})["P4"] == "HELD"
    R = _tc2_set("qwen3_5")
    R[("e4b", "fused_attn4_m")]["s_per_step_median_11plus"] = R[("e4b", "fused_attn4_m_d2")]["s_per_step_median_11plus"] = 8.0
    assert TP({**T_ALL, "qwen3_5": trun("qwen3_5", R)})["P4"] == "FALSIFIED"
    R = _tc2_set("olmoe")
    R[("e4b", "reference_attn4_m")]["losses"] = [round(2.1 - 0.01 * i, 5) for i in range(60)]
    R[("e4b", "reference_attn4_m")]["loss_last"] = R[("e4b", "reference_attn4_m")]["losses"][-1]
    assert TP({**T_ALL, "olmoe": trun("olmoe", R)})["P6"] == "FALSIFIED"
    R = _tc2_set("granite")
    R[("hf", "hf_peft_m")]["eval_curve"][1]["heldout_loss"] = 1.83
    assert TP({**T_ALL, "granite": trun("granite", R)})["P7"] == "FALSIFIED"
    assert TP({"granite": trun("granite")})["P2"] == "UNTESTED" and TP({})["P6"] == "UNTESTED" and TP({})["P7"] == "UNTESTED"
    cases += 1
    # 52. end to end through the files: both tokens' families (qwen3_5's underscore parses), the printer renders the TC2 paragraph, the footprint, the no-common-set line, the HF spelling and the P1-P7 table
    td = tempfile.mkdtemp(prefix="tc2_reduce_selftest_")
    for fam in TC2_FAMS:
        for (fw, tag), r in _tc2_set(fam).items():
            json.dump(r, open(os.path.join(td, f"{fam}_{fw}_{tag}.json"), "w"))
    F = reduce_dir(td, None)
    assert set(F) == set(TC2_FAMS) and all([(x["fw"], x["tag"]) for x in F[fam]["rows"]] == EXPECTED[fam] for fam in TC2_FAMS)
    text = render(F, td)
    for needle in ("Lane TC2 (`tc2small` / `tc2big` tokens", "MATCHED POSITION: s/step ratio HF (bf16 experts) / e4b = 1.292**", "NO COMMON ADAPTER SET (unsloth/ckpt_unsloth_mxfp4 vs e4b/attn_only_m) — no ratio is quoted",
                   "- **FOOTPRINT (e4b under expert offload", "## TC2 predictions P1–P7", "| P3 | gptoss | **HELD** |", "| P5 | mixtral | **HELD** |", "| P7 | tc2 | **HELD** |",
                   "attention census 128", "LABELLED ROW hf_peft_m_t214", "attention-only: trainable 5200000", "did NOT reach grouped_mm" if False else "REACHED grouped_mm"):
        assert needle in text, needle
    assert "## Predictions P1–P10" not in text and "## TC1b predictions" not in text
    assert text.index("- **FOOTPRINT") < text.index("| framework |", text.index("### Mixtral"))
    cases += 1
    # 53. amendment 3: the axolotl box -- the matched axolotl position carries its cross-draw interval, the labelled rows read, P6 is scored
    #     from this family alone (no judged-box receipts), the profiled arm's sidecar is not a row, and a 1.2x axolotl refutes P6
    ad = tempfile.mkdtemp(prefix="tc1_axolotl_selftest_")
    for (fw, tag), r in _ax_set().items():
        json.dump(r, open(os.path.join(ad, f"{AX_FAM}_{fw}_{tag}.json"), "w"))
    json.dump({"top_kernels": []}, open(os.path.join(ad, f"{AX_FAM}_e4b_fused_attn4_m_prof_profile.json"), "w"))
    F = reduce_dir(ad, 20)
    assert set(F) == {AX_FAM} and [(x["fw"], x["tag"]) for x in F[AX_FAM]["rows"]] == EXPECTED[AX_FAM], [(x["fw"], x["tag"]) for x in F[AX_FAM]["rows"]]
    text = render(F, ad)
    for needle in ("MATCHED POSITION: s/step ratio axolotl/e4b = 1.990** [1.961, 2.020 over 4 cross-draw ratios]", "LABELLED ROW ckpt_axolotl_best",
                   "LABELLED ROW hf_peft_m_mb1_t214", "| P6 | qwen3axolotl | **HELD** | axolotl trained; axolotl/e4b 1.990 vs [1.5, 6] |", "| P1 | qwen3 | **UNTESTED** |"):
        assert needle in text, needle
    assert "prof_profile" not in text
    AXR = reduce_family(AX_FAM, _ax_set((1.20, 1.21)), {}, 20)
    assert {p: v for p, _, v, _ in score_predictions({AX_FAM: AXR})}["P6"] == "FALSIFIED"
    assert {p: v for p, _, v, _ in score_predictions({"qwen3": run(_good_set()), AX_FAM: AXR})}["P6"] == "FALSIFIED"      # the axolotl box outranks the judged box's install defect
    cases += 1
    # 54. amendments 5 and 7: the native-best box -- every arm VALID against its own anchor (tc1-5090-33 read the anchor VOID for want of this
    #     registration), two draws a side, P13 HELD on both halves; an unstable e4b pair, a refused second draw and a faster framework each read as registered
    nd = tempfile.mkdtemp(prefix="tc1_nativebest_selftest_")
    for (fw, tag), r in _nb_set().items():
        json.dump(r, open(os.path.join(nd, f"{NB_FAM}_{fw}_{tag}.json"), "w"))
    F = reduce_dir(nd, 20)
    NB = F[NB_FAM]
    assert set(F) == {NB_FAM} and [(x["fw"], x["tag"]) for x in NB["rows"]] == EXPECTED[NB_FAM], [(x["fw"], x["tag"]) for x in NB["rows"]]
    assert all(x["verdict"] == "VALID" for x in NB["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in NB["rows"]]
    assert NB["draws"][("e4b", "fused_attn4_shipped")]["verdict"] == "STABLE" and NB["draws"][("e4b", "fused_attn4_m")]["verdict"] == "SINGLE"
    P13 = {p: (v, ev) for p, _, v, ev in score_p13(F)}
    assert P13["P13"][0] == "HELD" and P13["P13 (axolotl half)"][0] == "HELD" and P13["P13 (Unsloth half)"][0] == "HELD", P13
    assert "axolotl native-best / e4b shipped 1.193 [1.176, 1.210 over 4 cross-draw ratios]" in P13["P13 (axolotl half)"][1], P13
    text = render(F, nd)
    for needle in ("## Prediction P13", "| P13 | qwen3nativebest | **HELD** |", "NATIVE-BEST (reported beside, never instead of, the matched position): s/step ratio unsloth native-best vs e4b shipped/e4b = 1.792**",
                   "draws (R1): `e4b/fused_attn4_shipped` STABLE (1.000/1.020 s"):
        assert needle in text, needle
    assert os.path.dirname(os.path.abspath(nd)) not in text, "the header names the receipt dir, never its absolute path"
    assert "## Predictions P1–P10" not in text
    def p13(R):
        return {p: v for p, _, v, _ in score_p13({NB_FAM: reduce_family(NB_FAM, R, {}, 20)})}
    assert p13(_nb_set(e4b_s=(1.00, 1.07))) == {"P13 (axolotl half)": "UNTESTED", "P13 (Unsloth half)": "UNTESTED", "P13": "UNTESTED"}   # tc1-5090-33's shape
    assert p13(_nb_set(ax_s=(0.90, 0.91))) == {"P13 (axolotl half)": "FALSIFIED", "P13 (Unsloth half)": "HELD", "P13": "FALSIFIED"}
    R = _nb_set(); R[("axolotl", "ckpt_axolotl_best_d2")] = {**_stub("axolotl", "ckpt_axolotl_best_d2", "axolotl", "refused", "offline Hub"), "fam": NB_FAM}
    assert p13(R) == {"P13 (axolotl half)": "UNTESTED", "P13 (Unsloth half)": "HELD", "P13": "UNTESTED"}
    assert registered_draw2("qwen3native", ("e4b", "fused_attn4_shipped")) is None and run(_native_set(), fam="qwen3native")["draws"][("e4b", "fused_attn4_shipped")]["verdict"] == "SINGLE"
    assert score_p13({"qwen3": run(_good_set())}) == []
    cases += 1
    # 55. amendment 8: the 200-step native-best box -- every arm VALID against the 200-step matched anchor, P14 read on steps 101..200 (the
    #     warm-up spikes of steps 1..30 move the 11..200 median, never the late window), parity HELD, a faster axolotl FALSIFIED, an unstable side UNTESTED
    bd = tempfile.mkdtemp(prefix="tc1_nativebest200_selftest_")
    for (fw, tag), r in _nb200_set().items():
        json.dump(r, open(os.path.join(bd, f"{NB200_FAM}_{fw}_{tag}.json"), "w"))
    F = reduce_dir(bd, None)
    NB2 = F[NB200_FAM]
    assert set(F) == {NB200_FAM} and [(x["fw"], x["tag"]) for x in NB2["rows"]] == EXPECTED[NB200_FAM], [(x["fw"], x["tag"]) for x in NB2["rows"]]
    assert all(x["verdict"] == "VALID" for x in NB2["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in NB2["rows"]]
    assert NB2["anchor_key"] == ("e4b", "fused_attn4_m_200") and NB2["N"] == 200 and NB2["native"]["axolotl"]["quoted"] and "unsloth" not in NB2["native"]
    P14 = score_p14(F)
    assert [(p, v) for p, _, v, _ in P14] == [("P14", "HELD")] and "1.032 [1.023, 1.040 over 4 cross-draw ratios]" in P14[0][3] and "ordering: e4b shipped faster" in P14[0][3], P14
    text = render(F, bd)
    for needle in ("## Prediction P14", "| P14 | qwen3nativebest200 | **HELD** |", "NATIVE-BEST (reported beside, never instead of, the matched position): s/step ratio axolotl native-best vs e4b shipped/e4b"):
        assert needle in text, needle
    assert "## Prediction P13" not in text and "## Predictions P1–P10" not in text

    def p14(**kw):
        return [(p, v) for p, _, v, _ in score_p14({NB200_FAM: reduce_family(NB200_FAM, _nb200_set(**kw), {}, None)})]
    assert p14(ax_late=(2.50, 2.52)) == [("P14", "FALSIFIED")]
    assert "ordering: axolotl scattermoe faster" in score_p14({NB200_FAM: reduce_family(NB200_FAM, _nb200_set(ax_late=(2.50, 2.52)), {}, None)})[0][3]
    assert p14(ax_late=(3.00, 3.40)) == [("P14", "UNTESTED")]
    assert late_median({"step_ms": [1000.0] * 50}) is None and abs(late_median({"step_ms": [9e5] * 100 + [3000.0] * 100}) - 3.0) < 1e-12
    R = _nb200_set(); R.pop(("axolotl", "ckpt_axolotl_best_200_d2"))
    assert p14() == [("P14", "HELD")] and [(p, v) for p, _, v, _ in score_p14({NB200_FAM: reduce_family(NB200_FAM, R, {}, None)})] == [("P14", "UNTESTED")]
    assert registered_draw2(NB200_FAM, ("e4b", "fused_attn4_shipped_200")) == ("e4b", "fused_attn4_shipped_200_d2") and registered_draw2(CURVE_FAM, ("e4b", "fused_attn4_shipped_200")) is None
    cases += 1
    # 56. amendment 10 (#945): e4b legacy grouping + pageable copies vs single-read grouping + pinned ring -- every arm VALID when it ran the
    #     path its tag names, P16 / P17 HELD inside the band, a sync1 arm whose ring never staged VOID (its prediction UNTESTED), parity FALSIFIED
    sd = tempfile.mkdtemp(prefix="tc1_syncab_selftest_")
    for (fw, tag), r in _sync_set().items():
        json.dump(r, open(os.path.join(sd, f"{SYNC_FAM}_{fw}_{tag}.json"), "w"))
    F = reduce_dir(sd, 20)
    SY = F[SYNC_FAM]
    assert [(x["fw"], x["tag"]) for x in SY["rows"]] == EXPECTED[SYNC_FAM] and all(x["verdict"] == "VALID" for x in SY["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in SY["rows"]]
    assert SY["anchor_key"] == ("e4b", "fused_attn4_m_legacy")
    PS = {p: (v, ev) for p, _, v, ev in score_syncab(F)}
    assert PS["P16"][0] == "HELD" and PS["P17"][0] == "HELD", PS
    assert "sync1 / legacy 0.858 [" in PS["P16"][1], PS["P16"][1]
    text = render(F, sd)
    for needle in ("## Predictions P16 / P17", "| P16 | qwen3syncab | **HELD** |", "| P17 | qwen3syncab | **HELD** |"):
        assert needle in text, needle
    def ps(R):
        return {p: v for p, _, v, _ in score_syncab({SYNC_FAM: reduce_family(SYNC_FAM, R, {}, 20)})}
    R = _sync_set(staged=(0, 0))
    RR = reduce_family(SYNC_FAM, R, {}, 20)
    assert RR["verdicts"][("e4b", "fused_attn4_shipped_sync1")] == "VOID" and "ring_staged > 0" in next(x["why"] for x in RR["rows"] if x["tag"] == "fused_attn4_shipped_sync1")
    assert ps(R) == {"P16": "UNTESTED", "P17": "UNTESTED"}
    assert ps(_sync_set(ship=((5.00, 5.05), (4.97, 5.00)))) == {"P16": "FALSIFIED", "P17": "HELD"}          # no gain: 0.99
    R = _sync_set(); R[("e4b", "fused_attn4_m_legacy")]["sync_ab"]["e4b_grouping"] = "single"                 # a legacy arm that ran the new grouping
    assert reduce_family(SYNC_FAM, R, {}, 20)["verdicts"][("e4b", "fused_attn4_m_legacy")] == "VOID"
    cases += 1
    # 57. amendment 12 (#945): the profile token -- every arm VALID, the descriptive table, P19 HELD on a +0.14 busy-fraction gain, FALSIFIED on +0.02,
    #     UNTESTED when the legacy arm ran the new path (VOID by the engagement predicate)
    def p19(R):
        return [(p, v) for p, _, v, _ in score_prof945({PROF945_FAM: reduce_family(PROF945_FAM, R, {}, 20)})]
    PF = {PROF945_FAM: reduce_family(PROF945_FAM, _prof945_set(), {}, 20)}
    assert all(x["verdict"] == "VALID" for x in PF[PROF945_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in PF[PROF945_FAM]["rows"]]
    assert [t["arm"] for t in prof945_table(PF)] == [k[1] for k in PROF945_ARMS] and p19(_prof945_set()) == [("P19", "HELD")]
    assert p19(_prof945_set(busy_new=0.50)) == [("P19", "FALSIFIED")]
    R = _prof945_set(); R[("e4b", "fused_attn4_m_prof_legacy")]["sync_ab"]["e4b_grouping"] = "single"
    assert p19(R) == [("P19", "UNTESTED")]
    text = render(PF, "x")
    assert "## Amendment 12 (#945)" in text and "| P19 | qwen3prof945 | **HELD** |" in text
    cases += 1
    # 58. amendment 13 (#945): gnf4's previous padded LoRA delta vs its trimmed body -- every arm VALID when it ran the body its tag names on
    #     the new sync path, P20 / P21 HELD inside the band, FALSIFIED with no gain and flagged above the revert line, a lean1 arm that ran the
    #     previous body or whose padded path never served VOID (its prediction UNTESTED), and a lean arm on the legacy sync path VOID
    def pl(R):
        return {p: v for p, _, v, _ in score_leanab({LEAN_FAM: reduce_family(LEAN_FAM, R, {}, 20)})}
    LF = {LEAN_FAM: reduce_family(LEAN_FAM, _lean_set(), {}, 20)}
    assert [(x["fw"], x["tag"]) for x in LF[LEAN_FAM]["rows"]] == EXPECTED[LEAN_FAM]
    assert all(x["verdict"] == "VALID" for x in LF[LEAN_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in LF[LEAN_FAM]["rows"]]
    assert LF[LEAN_FAM]["anchor_key"] == ("e4b", "fused_attn4_m_lean0")
    PL = {p: (v, ev) for p, _, v, ev in score_leanab(LF)}
    assert PL["P20"][0] == "HELD" and PL["P21"][0] == "HELD" and "lean1 / lean0 0.937 [" in PL["P20"][1], PL
    assert pl(_lean_set(ship=((5.00, 5.05), (4.99, 5.02)))) == {"P20": "FALSIFIED", "P21": "HELD"}           # no gain: 0.995
    slow = score_leanab({LEAN_FAM: reduce_family(LEAN_FAM, _lean_set(match=((5.00, 5.02), (5.20, 5.22))), {}, 20)})
    assert [(p, v) for p, _, v, _ in slow] == [("P20", "HELD"), ("P21", "FALSIFIED")] and "the decision rule reverts the default" in slow[1][3]
    RR = reduce_family(LEAN_FAM, _lean_set(lean=("0", "0")), {}, 20)
    assert RR["verdicts"][("e4b", "fused_attn4_shipped_lean1")] == "VOID" and "gnf4_lean_delta 1" in next(x["why"] for x in RR["rows"] if x["tag"] == "fused_attn4_shipped_lean1")
    assert pl(_lean_set(lean=("0", "0"))) == {"P20": "UNTESTED", "P21": "UNTESTED"}
    assert reduce_family(LEAN_FAM, _lean_set(padded=(9216, 0)), {}, 20)["verdicts"][("e4b", "fused_attn4_m_lean1")] == "VOID"
    R = _lean_set(); R[("e4b", "fused_attn4_m_lean0")]["sync_ab"]["gnf4_pinned_ring"] = "0"            # an arm on the pre-#945 sync path
    assert reduce_family(LEAN_FAM, R, {}, 20)["verdicts"][("e4b", "fused_attn4_m_lean0")] == "VOID"
    text = render(LF, "x")
    assert "## Predictions P20 / P21" in text and "| P20 | qwen3leanab | **HELD** |" in text and "| P21 | qwen3leanab | **HELD** |" in text
    cases += 1
    # 59. amendment 14 (#945): gnf4's max-keyed prefill M-tile vs the cost rule -- every arm VALID when it ran the rule its tag names on the
    #     trimmed delta, P22 / P23 HELD inside their bands with the flip reading, FALSIFIED and KEEP above 1.01, a cost arm that launched only
    #     128-row tiles or ran max VOID (UNTESTED), an arm on the previous delta body VOID
    def pt(R):
        return {p: v for p, _, v, _ in score_tileab({TILE_FAM: reduce_family(TILE_FAM, R, {}, 20)})}
    TF = {TILE_FAM: reduce_family(TILE_FAM, _tile_set(), {}, 20)}
    assert [(x["fw"], x["tag"]) for x in TF[TILE_FAM]["rows"]] == EXPECTED[TILE_FAM]
    assert all(x["verdict"] == "VALID" for x in TF[TILE_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in TF[TILE_FAM]["rows"]]
    PT = {p: (v, ev) for p, _, v, ev in score_tileab(TF)}
    assert PT["P22"][0] == "HELD" and PT["P23"][0] == "HELD" and "tilecost / tilemax 0.917 [" in PT["P22"][1] and "flip-eligible" in PT["P22"][1], PT
    slow = score_tileab({TILE_FAM: reduce_family(TILE_FAM, _tile_set(match=((5.00, 5.02), (5.20, 5.22))), {}, 20)})
    assert [(p, v) for p, _, v, _ in slow] == [("P22", "HELD"), ("P23", "FALSIFIED")] and "KEEP max" in slow[1][3]
    assert pt(_tile_set(ship=((5.00, 5.05), (4.99, 5.02)))) == {"P22": "FALSIFIED", "P23": "HELD"}         # no gain: 0.995, no measurable effect
    RR = reduce_family(TILE_FAM, _tile_set(short=0), {}, 20)
    assert RR["verdicts"][("e4b", "fused_attn4_shipped_tilecost")] == "VOID" and "a tile shorter than 128 launched" in next(x["why"] for x in RR["rows"] if x["tag"] == "fused_attn4_shipped_tilecost")
    assert pt(_tile_set(short=0)) == {"P22": "UNTESTED", "P23": "UNTESTED"} and pt(_tile_set(rules=("max", "max"))) == {"P22": "UNTESTED", "P23": "UNTESTED"}
    assert reduce_family(TILE_FAM, _tile_set(lean="0"), {}, 20)["verdicts"][("e4b", "fused_attn4_m_tilemax")] == "VOID"
    text = render(TF, "x")
    assert "## Predictions P22 / P23" in text and "| P22 | qwen3tileab | **HELD** |" in text and "| P23 | qwen3tileab | **HELD** |" in text
    cases += 1
    # ----------------------------------------------------------------------- R11: lane TC3 (the frontier tokens)

    def frun(R, fam=FRONTIER_FAM, tc1=None):
        return reduce_frontier_family(fam, R, {}, 20, tc1=tc1)

    def FP(F):
        return {p_: v for p_, _, v, _ in score_frontier_predictions(F)}
    # 44. baseline 24 GB: the verdicts, the FIT TABLE (an OOM-only Unsloth column), the HF offload refused row and the ZeRO-3 refused row UNSUPPORTED with their text,
    #     the in-box equivalence EQUIVALENT under the fixed bands with parity PASS, the resident comparison UNTESTED without --tc1-dir; P1 HELD, P3 / P4 UNTESTED
    Fr = frun(_frontier_set())
    V = Fr["verdicts"]
    assert [(x["fw"], x["tag"]) for x in Fr["rows"]] == EXPECTED[FRONTIER_FAM]
    assert V[("e4b", "fused_attn4_m")] == "OOM" and V[FRONTIER_ANCHOR] == "VALID" and V[("e4b", "fused_attn4_m_mb1")] == "VALID" and V[FRONTIER_REF] == "VALID", V
    assert V[("unsloth", "ckpt_unsloth_m")] == "OOM" and V[("unsloth", "ckpt_unsloth_m_mb1")] == "OOM" and V[("hf", "hf_peft_m")] == "OOM"
    assert V[("hf", "hf_peft_m_offload")] == "UNSUPPORTED" and V[("axolotl", "ckpt_axolotl_m_zero3")] == "UNSUPPORTED" and V[("axolotl", "ckpt_axolotl_m_layeroffload")] == "VALID", V
    ft = Fr["fit"]
    assert set(ft) == {"e4b", "unsloth", "hf", "axolotl"} and ft["e4b"]["fits"] and ft["e4b"]["completed"] == ["fused_attn4_m_offload", "fused_attn4_m_mb1", "fused_attn4_shipped", "reference_attn4_m_offload"], ft["e4b"]["completed"]
    assert not ft["unsloth"]["fits"] and ft["unsloth"]["reading"].startswith("NO ARM COMPLETED") and "`ckpt_unsloth_m_mb1` OOM" in ft["unsloth"]["reading"], ft["unsloth"]["reading"]
    assert not ft["hf"]["fits"] and ft["axolotl"]["completed"] == ["ckpt_axolotl_m_layeroffload"]
    a1 = next(x for x in ft["e4b"]["arms"] if x["tag"] == "fused_attn4_m_offload")
    assert a1["lever"].startswith("e4b expert offload") and a1["peak_vram_gb"] == 9.5 and a1["host_ram_high_water_gb"] == 40.0 and a1["s"] == 2.0, a1
    oomrow = next(x for x in ft["e4b"]["arms"] if x["tag"] == "fused_attn4_m")
    assert oomrow["peak_vram_gb"] == 23.6 and oomrow["verdict"] == "OOM" and oomrow["lever"] == "resident"
    print("FAILING-CASE TC3-fit (reducer): unsloth ->", ft["unsloth"]["reading"])
    hx = row(Fr, "hf", "hf_peft_m_offload")
    assert hx["verdict"] == "UNSUPPORTED" and "NotImplementedError" in hx["reason"] and "Cannot copy out of meta tensor" in hx["reason"], hx["reason"]
    print("FAILING-CASE TC3-hf-offload (reducer):", hx["verdict"], "--", hx["reason"][:100])
    zx = row(Fr, "axolotl", "ckpt_axolotl_m_zero3")
    assert zx["verdict"] == "UNSUPPORTED" and "trainer" in zx["reason"], zx["reason"]
    print("FAILING-CASE TC3-zero3 (reducer):", zx["verdict"], "--", zx["reason"][:100])
    eq = Fr["equivalence"]
    assert eq[FRONTIER_REF]["reading"] == "EQUIVALENT" and abs(eq[FRONTIER_REF]["d_heldout"] - 0.005) < 1e-9 and eq[FRONTIER_REF]["med_train"] == 0.0 and Fr["equiv_band"] == {"train": EQUIV, "heldout": EQUIV}, eq[FRONTIER_REF]
    assert eq[("axolotl", "ckpt_axolotl_m_layeroffload")]["reading"] == "EQUIVALENT" and eq[("e4b", "fused_attn4_m_mb1")]["reading"] == "EQUIVALENT"   # every matched arm beside the anchor, the mb1 e4b arm included
    assert eq[("hf", "hf_peft_m_offload")]["reading"] == "—" and eq[("axolotl", "ckpt_axolotl_m_zero3")]["reading"] == "—"                         # the refused rows: no OK receipt
    assert Fr["parity"]["verdict"] == "PASS" and abs(Fr["parity"]["speed_x"] - 2.0) < 1e-9, Fr["parity"]
    assert Fr["resident"]["reading"] == "UNTESTED" and "no --tc1-dir" in Fr["resident"]["why"]
    assert Fr["draws"][FRONTIER_ANCHOR]["verdict"] == "SINGLE" and Fr["anchor_key"] == FRONTIER_ANCHOR        # no second draw registered on the 24 GB token
    P_ = FP({FRONTIER_FAM: Fr})
    assert P_ == {"P1": "HELD", "P3": "UNTESTED", "P4": "UNTESTED"}, P_
    p1 = next(x for x in score_frontier_predictions({FRONTIER_FAM: Fr}) if x[0] == "P1")
    assert "(iii-b) HF offload REFUSED" in p1[3] and "(iv) axolotl zero3 did not run" in p1[3] and "vacuously" in p1[3] and "(ii) Unsloth OOM at both recipes -> HELD" in p1[3], p1[3]
    cases += 1
    # 45. a VOID offload equivalence: the fused offload anchor's kernel took the loop on a step -> VOID, the control pair's equivalence N-A, P3 UNTESTED (P1's (i) still "completed");
    #     the control itself VOID -> N-A; the in-box pair DIVERGENT (the reference's train losses 0.10 above) with parity FAIL; no e4b offload arm completed -> the reference stands in as anchor, P1 (i) FALSIFIED
    R = _frontier_set()
    R[FRONTIER_ANCHOR] = {**R[FRONTIER_ANCHOR], "lora_path_loop_steps": [2], "lora_loop_share": [0.0, 0.5] + [0.0] * 18}
    Fr = frun(R)
    assert Fr["verdicts"][FRONTIER_ANCHOR] == "VOID" and Fr["equivalence"][FRONTIER_REF]["reading"] == "N-A" and "VOID" in Fr["equivalence"][FRONTIER_REF]["why"], Fr["equivalence"][FRONTIER_REF]
    assert FP({FRONTIER_FAM: Fr})["P3"] == "UNTESTED" and FP({FRONTIER_FAM: Fr})["P1"] == "HELD"
    print("FAILING-CASE TC3-equiv (reducer): offload anchor", Fr["verdicts"][FRONTIER_ANCHOR], "->", Fr["equivalence"][FRONTIER_REF]["reading"], "--", Fr["equivalence"][FRONTIER_REF]["why"])
    R = _frontier_set()
    R[FRONTIER_REF] = {**R[FRONTIER_REF], "C1_control_tensor": None}
    assert frun(R)["equivalence"][FRONTIER_REF]["reading"] == "N-A"
    R = _frontier_set()
    R[FRONTIER_REF] = _frontier_receipt(FRONTIER_FAM, "e4b", "reference_attn4_m_offload", "reference", lever="e4b_offload", peak=9.6, host=40.0, s=4.0, heldout_n=1.805, losses=[round(2.1 - 0.01 * i, 5) for i in range(20)])
    Fr = frun(R)
    assert Fr["equivalence"][FRONTIER_REF]["reading"] == "DIVERGENT" and Fr["parity"]["verdict"] == "FAIL", (Fr["equivalence"][FRONTIER_REF], Fr["parity"])
    R = _frontier_set()
    R[FRONTIER_ANCHOR] = _fstub(FRONTIER_FAM, "e4b", "fused_attn4_m_offload", "fused", "oom", "OOM at step 3", peak=23.9, host=40.0)
    Fr = frun(R)
    assert Fr["anchor_key"] == FRONTIER_REF and FP({FRONTIER_FAM: Fr})["P1"] == "FALSIFIED" and Fr["equivalence"][FRONTIER_REF]["reading"] == "N-A" and "offload anchor" in Fr["equivalence"][FRONTIER_REF]["why"]
    cases += 1
    # 46. P1's other legs, within the box: Unsloth mb1 completes -> FALSIFIED; HF offload RUNS at 10x -> FALSIFIED, at 25x -> HELD; HF offload OOM -> FALSIFIED;
    #     ZeRO-3 runs at 6x with 70 GB host -> HELD, at 3x -> FALSIFIED, at 6x with 40 GB -> FALSIFIED; a missing Unsloth mb1 row -> UNTESTED
    R = _frontier_set()
    ub = {"unsloth_grouped_mm": 48 * 8, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": 48 * 8}
    R[("unsloth", "ckpt_unsloth_m_mb1")] = _frontier_receipt(FRONTIER_FAM, "unsloth", "ckpt_unsloth_m_mb1", "unsloth", peak=22.0, host=8.0, s=5.0, heldout_n=1.81, accum=8, micro_batch=1,
                                                             experts_forward_calls_per_step_min=48 * 8, unsloth_backend_calls_per_step_min=ub, unsloth_backend_calls_per_step_max=ub,
                                                             unsloth_grouped_mm_calls_per_step_min=GMM_FACTOR * 48 * 8, unsloth_grouped_mm_calls_per_step_max=GMM_FACTOR * 48 * 8)
    Fr = frun(R)
    assert Fr["verdicts"][("unsloth", "ckpt_unsloth_m_mb1")] == "VALID" and FP({FRONTIER_FAM: Fr})["P1"] == "FALSIFIED" and Fr["fit"]["unsloth"]["fits"], (Fr["verdicts"][("unsloth", "ckpt_unsloth_m_mb1")], row(Fr, "unsloth", "ckpt_unsloth_m_mb1")["why"])
    for s_h, want in ((20.0, "FALSIFIED"), (50.0, "HELD")):
        R = _frontier_set()
        R[("hf", "hf_peft_m_offload")] = _frontier_receipt(FRONTIER_FAM, "hf", "hf_peft_m_offload", "hf", lever="hf_offload", peak=20.0, host=90.0, s=s_h, heldout_n=1.81)
        assert FP({FRONTIER_FAM: frun(R)})["P1"] == want, (s_h, want, FP({FRONTIER_FAM: frun(R)}))
    R = _frontier_set()
    R[("hf", "hf_peft_m_offload")] = _fstub(FRONTIER_FAM, "hf", "hf_peft_m_offload", "hf", "oom", "OOM at step 1", peak=23.9, host=90.0)
    assert FP({FRONTIER_FAM: frun(R)})["P1"] == "FALSIFIED"
    for s_z, h_z, want in ((12.0, 70.0, "HELD"), (6.0, 70.0, "FALSIFIED"), (12.0, 40.0, "FALSIFIED")):
        R = _frontier_set()
        R[("axolotl", "ckpt_axolotl_m_zero3")] = _frontier_receipt(FRONTIER_FAM, "axolotl", "ckpt_axolotl_m_zero3", "axolotl", lever="axolotl_zero3", peak=12.0, host=h_z, s=s_z, heldout_n=1.81,
                                                                    axolotl={"version": "0.20.0", "config": {"quantize_moe_experts": False}}, axolotl_bnb4bit_modules={"n_bnb4bit_unwrapped": 0})
        assert FP({FRONTIER_FAM: frun(R)})["P1"] == want, (s_z, h_z, want)
    R = _frontier_set()
    del R[("unsloth", "ckpt_unsloth_m_mb1")]
    assert FP({FRONTIER_FAM: frun(R)})["P1"] == "UNTESTED"
    cases += 1
    # 47. with --tc1-dir: the offload anchor against TC1's resident e4b/fused_attn4_m (same tokens / init / precision) -> EQUIVALENT-TO-RESIDENT, P3 HELD, P4 HELD (2.000 <= 2 x 1.010);
    #     offload at 2.1 s/step -> P4 FALSIFIED; its losses +0.03 -> DIVERGENT-FROM-RESIDENT, P3 FALSIFIED; another tokens sha -> N-A naming it, P3 / P4 UNTESTED
    tc1 = {"qwen3": run(_good_set())}
    Fr = frun(_frontier_set(), tc1=tc1)
    rs = Fr["resident"]
    assert rs["reading"] == "EQUIVALENT-TO-RESIDENT" and rs["med_train"] == 0.0 and abs(rs["tc1_s"] - 1.01) < 1e-9 and rs["tc1_draws"] == 2 and rs["tc1_box"] == "RTX 5090" and rs["offload_s"] == 2.0, rs
    P_ = FP({FRONTIER_FAM: Fr})
    assert P_["P3"] == "HELD" and P_["P4"] == "HELD", P_
    p4 = next(x for x in score_frontier_predictions({FRONTIER_FAM: Fr}) if x[0] == "P4")
    assert "two measurements, no cross-box ratio formed" in p4[3] and "2.000 s/step" in p4[3] and "1.010 s/step" in p4[3] and "RTX 4090" in p4[3], p4[3]
    R = _frontier_set()
    R[FRONTIER_ANCHOR]["s_per_step_median_11plus"] = 2.1
    assert FP({FRONTIER_FAM: frun(R, tc1=tc1)})["P4"] == "FALSIFIED"
    R = _frontier_set()
    R[FRONTIER_ANCHOR]["losses"] = [round(2.03 - 0.01 * i, 5) for i in range(20)]
    Fr = frun(R, tc1=tc1)
    assert Fr["resident"]["reading"] == "DIVERGENT-FROM-RESIDENT" and abs(Fr["resident"]["med_train"] - 0.03) < 1e-9 and FP({FRONTIER_FAM: Fr})["P3"] == "FALSIFIED", Fr["resident"]
    print("FAILING-CASE TC3-P3 (reducer): median per-step |delta| vs TC1's resident", f(Fr["resident"]["med_train"], 4), ">", RESIDENT_BAND, "->", Fr["resident"]["reading"])
    R = _frontier_set()
    for k in R:
        if is_ok(R[k]):
            R[k]["tokens"] = {"sha256": "u" * 64}
    Fr = frun(R, tc1=tc1)
    assert Fr["resident"]["reading"] == "N-A" and "tokens sha differs" in Fr["resident"]["why"] and FP({FRONTIER_FAM: Fr}) == {"P1": "HELD", "P3": "UNTESTED", "P4": "UNTESTED"}, (Fr["resident"], FP({FRONTIER_FAM: Fr}))
    assert Fr["verdicts"][FRONTIER_ANCHOR] == "VALID"                        # the family's own tokens agree among themselves: nothing VOID
    cases += 1
    # 48. the 12 GB token: baseline -> P2 HELD (the axolotl row UNSUPPORTED, said as "not an OOM reading"), the offload pair STABLE (9.00 / 9.18 s), P3 UNTESTED (its own box);
    #     HF mb1 completes -> P2 FALSIFIED; the second draw missing -> UNMEASURED, P2 still HELD (a stability reading, not a fit reading); the reference offload
    #     OOM -> P2 still HELD with the OOM in the evidence (a parity control, not the fit claim); HF mb1 NOT_RUN -> UNTESTED
    F12 = frun(_frontier12_set(), fam=FRONTIER12_FAM)
    assert [(x["fw"], x["tag"]) for x in F12["rows"]] == EXPECTED[FRONTIER12_FAM]
    dw = F12["draws"][FRONTIER_ANCHOR]
    assert dw["verdict"] == "STABLE" and dw["draws"] == 2 and abs(dw["stability"] - 0.18 / 9.09) < 1e-9 and dw["usable"], dw
    assert F12["equivalence"][("e4b", "fused_attn4_m_offload_d2")]["reading"] == "EQUIVALENT" and F12["equivalence"][FRONTIER_REF]["reading"] == "EQUIVALENT"
    P_ = FP({FRONTIER12_FAM: F12})
    assert P_ == {"P2": "HELD", "P3": "UNTESTED"}, P_
    p2 = next(x for x in score_frontier_predictions({FRONTIER12_FAM: F12}) if x[0] == "P2")
    assert "axolotl/ckpt_axolotl_m UNSUPPORTED (not an OOM reading" in p2[3] and "e4b resident OOM" in p2[3], p2[3]
    R = _frontier12_set()
    R[("hf", "hf_peft_m_mb1")] = _frontier_receipt(FRONTIER12_FAM, "hf", "hf_peft_m_mb1", "hf", peak=11.5, host=8.0, total=64.0, box="RTX A2000", s=30.0, heldout_n=1.81, accum=8, micro_batch=1, experts_forward_calls_per_step_min=48 * 8)
    F12 = frun(R, fam=FRONTIER12_FAM)
    assert F12["verdicts"][("hf", "hf_peft_m_mb1")] == "VALID" and FP({FRONTIER12_FAM: F12})["P2"] == "FALSIFIED", row(F12, "hf", "hf_peft_m_mb1")["why"]
    print("FAILING-CASE TC3-P2 (reducer):", next(x for x in score_frontier_predictions({FRONTIER12_FAM: F12}) if x[0] == "P2")[3][:170])
    R = _frontier12_set()
    del R[("e4b", "fused_attn4_m_offload_d2")]
    F12 = frun(R, fam=FRONTIER12_FAM)
    assert F12["draws"][FRONTIER_ANCHOR]["verdict"] == "UNMEASURED" and FP({FRONTIER12_FAM: F12})["P2"] == "HELD"
    R = _frontier12_set()
    R[FRONTIER_REF] = _fstub(FRONTIER12_FAM, "e4b", "reference_attn4_m_offload", "reference", "oom", "OOM at step 1", peak=11.9, host=40.0, total=64.0)
    p2 = next(x for x in score_frontier_predictions({FRONTIER12_FAM: frun(R, fam=FRONTIER12_FAM)}) if x[0] == "P2")
    assert p2[2] == "HELD" and "reference_attn4_m_offload OOM" in p2[3], p2
    R = _frontier12_set()
    R[("hf", "hf_peft_m_mb1")] = _fstub(FRONTIER12_FAM, "hf", "hf_peft_m_mb1", "hf", "not_run", "deadline", total=64.0)
    assert FP({FRONTIER12_FAM: frun(R, fam=FRONTIER12_FAM)})["P2"] == "UNTESTED"
    cases += 1
    # 48b. the 12 GB token under the mb1 secondary: the field-recipe offload pair OOMs, fused_attn4_m_offload_mb1 completes (one draw) -> P2 HELD naming the
    #      secondary, the anchor falls back to it (quality / equivalence read against it); the as-shipped offload row is a FIT row outside P2; nothing completed -> FALSIFIED
    S12 = _frontier12_set()
    for t in ("fused_attn4_m_offload", "fused_attn4_m_offload_d2", "reference_attn4_m_offload"):
        S12[("e4b", t)] = _fstub(FRONTIER12_FAM, "e4b", t, "fused" if t.startswith("fused") else "reference", "oom", "OOM at step 1", peak=11.9, host=40.0, total=64.0)
    S12[("e4b", "fused_attn4_m_offload_mb1")] = _frontier_receipt(FRONTIER12_FAM, "e4b", "fused_attn4_m_offload_mb1", "fused", lever="e4b_offload", peak=9.8, host=42.0, total=64.0, box="RTX A2000",
                                                                  s=14.0, accum=8, micro_batch=1, kernel_calls_per_step_min=2 * 48 * 8)
    S12[("e4b", "reference_attn4_m_offload_mb1")] = _frontier_receipt(FRONTIER12_FAM, "e4b", "reference_attn4_m_offload_mb1", "reference", lever="e4b_offload", peak=9.9, host=42.0, total=64.0, box="RTX A2000",
                                                                      s=40.0, heldout_n=1.8050, accum=8, micro_batch=1)
    S12[("e4b", "fused_attn4_shipped_offload")] = _frontier_receipt(FRONTIER12_FAM, "e4b", "fused_attn4_shipped_offload", "fused", lever="e4b_offload", peak=8.1, host=42.0, total=64.0, box="RTX A2000",
                                                                    s=11.0, heldout_n=1.79, matched=False)
    F12b = reduce_frontier_family(FRONTIER12_FAM, S12, {}, 20)
    assert F12b["anchor_key"] == ("e4b", "fused_attn4_m_offload_mb1"), F12b["anchor_key"]
    assert F12b["verdicts"][("e4b", "fused_attn4_m_offload_mb1")] == "VALID" and F12b["verdicts"][("e4b", "fused_attn4_shipped_offload")] == "VALID", F12b["verdicts"]
    assert F12b["fit"]["e4b"]["completed"] == ["fused_attn4_m_offload_mb1", "reference_attn4_m_offload_mb1", "fused_attn4_shipped_offload"], F12b["fit"]["e4b"]["completed"]
    p2 = next(x for x in score_frontier_predictions({FRONTIER12_FAM: F12b}) if x[0] == "P2")
    assert p2[2] == "HELD" and "its mb1 secondary" in p2[3] and "a fit row, outside P2" in p2[3], p2
    S12[("e4b", "fused_attn4_m_offload_mb1")] = _fstub(FRONTIER12_FAM, "e4b", "fused_attn4_m_offload_mb1", "fused", "oom", "OOM at step 1", peak=11.9, host=40.0, total=64.0)
    S12[("e4b", "reference_attn4_m_offload_mb1")] = _fstub(FRONTIER12_FAM, "e4b", "reference_attn4_m_offload_mb1", "reference", "oom", "OOM at step 1", peak=11.9, host=40.0, total=64.0)
    p2 = next(x for x in score_frontier_predictions({FRONTIER12_FAM: reduce_frontier_family(FRONTIER12_FAM, S12, {}, 20)}) if x[0] == "P2")
    print("FAILING-CASE TC3-P2-secondary (reducer):", p2[2], "--", p2[3][-120:])
    assert p2[2] == "FALSIFIED" and "no e4b fused offload arm completed at either recipe" in p2[3], p2
    cases += 1
    # 49. end to end through the files: both frontier tokens, the printer renders the FIT TABLE and fit lines, the in-box equivalence, the resident line, the levers' records and the TC3 P1-P4 table
    #     (never TC1's P1-P10 table or a POSITION); with --tc1-dir the resident reading and P3 / P4
    fd = tempfile.mkdtemp(prefix="tc3_reduce_selftest_")
    for (fw, tag), r in _frontier_set().items():
        json.dump(r, open(os.path.join(fd, f"{FRONTIER_FAM}_{fw}_{tag}.json"), "w"))
    for (fw, tag), r in _frontier12_set().items():
        json.dump(r, open(os.path.join(fd, f"{FRONTIER12_FAM}_{fw}_{tag}.json"), "w"))
    F = reduce_dir(fd, 20)
    assert set(F) == set(FRONTIER_FAMS) and [(x["fw"], x["tag"]) for x in F[FRONTIER_FAM]["rows"]] == EXPECTED[FRONTIER_FAM]
    text = render(F, fd)
    for needle in ("Lane TC3", "**(a) FIT TABLE**", "| e4b | fused_attn4_m_offload | e4b expert offload (--offload 1) | **VALID** | 9.500 | 40.000 | 2.000 |", "fit `unsloth`: **NO ARM COMPLETED on this box",
                   "fit `e4b`: **FITS on this box", "**(b) in-box equivalence**", "`e4b/reference_attn4_m_offload` **EQUIVALENT**", "RESIDENT COMPARISON: UNTESTED", "e4b internal parity under offload",
                   "lever `hf/hf_peft_m_offload`: max_memory", "trained: NO -- hf_offload: NotImplementedError", "lever `axolotl/ckpt_axolotl_m_layeroffload`: layer_offloading engaged True over 48 layers",
                   "lever `axolotl/ckpt_axolotl_m_zero3` (ZeRO-3): REFUSED", "## TC3 predictions P1–P4", "| P1 | qwen3frontier | **HELD** |", "| P2 | qwen3frontier12 | **HELD** |",
                   "| P3 | qwen3frontier | **UNTESTED** |", "| P3 | qwen3frontier12 | **UNTESTED** |", "| P4 | qwen3frontier | **UNTESTED** |", "draws (R1): `e4b/fused_attn4_m_offload` STABLE (9.000/9.180 s"):
        assert needle in text, needle
    assert "## Predictions P1–P10" not in text and "MATCHED POSITION" not in text and "POSITION:" not in text
    F = reduce_dir(fd, 20, tc1_dir=d)                    # `d` holds TC1's qwen3 receipts (cases 22 / 43)
    text = render(F, fd)
    assert "RESIDENT COMPARISON: **EQUIVALENT-TO-RESIDENT**" in text and "| P3 | qwen3frontier | **HELD** |" in text and "| P4 | qwen3frontier | **HELD** |" in text and "no cross-box ratio formed" in text
    assert "| P3 | qwen3frontier12 | **HELD** |" in text and text.count("| P4 |") == 1   # P3 is read per token (the 12 GB anchor shares TC1's tokens and init here too); P4 only on the 24 GB token
    cases += 1
    print(f"REDUCE SELFTEST OK cases={cases} dir={d}")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--md", default=None)
    ap.add_argument("--steps", type=int, default=None, help="registered N (default: read from the receipts)")
    ap.add_argument("--selftest", action="store_true", help="R7: hand-built receipts through every reading; exit 0 iff every case reads as registered")
    ap.add_argument("--tc1-dir", default=None, help="R10 (d): a TC1 receipt dir (qwen3 / qwen3native tokens) whose quoted 11..20 s/step medians the qwen3curve arms are read beside (TRAVELS within 10 %%); "
                                                   "R11 (b): the frontier tokens' offload anchor is read against its resident e4b/fused_attn4_m (EQUIVALENT-TO-RESIDENT) and P4 is scored")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.dir:
        ap.error("dir is required (or --selftest)")
    F = reduce_dir(a.dir, a.steps, tc1_dir=a.tc1_dir)
    text = render(F, a.dir)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
