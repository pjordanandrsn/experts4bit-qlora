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

  R12 (TC1 amendment 23, the `qwen3memcensus` token): e4b fused_attn4_m_mb1 (fp32 expert absmax), e4b fused_attn4_m_mb1_dq (E4B_ABSMAX_DQ=1) and
     Unsloth ckpt_unsloth_m_mb1, one draw each at micro-batch 1 x accum 8 with tc1_arm.py's memory census on. Validity adds the pin, the recipe,
     a `mem_census` on the receipt (an errored census is recorded, never VOID) and the absmax each e4b tag names; no position is quoted on the
     token (NO_SPEED_FAMS: the census slows the step). The census tables put the three arms side by side (score_memcensus / memcensus_block),
     and P41-P43 are scored HELD / FALSIFIED / UNTESTED from the receipts (a missing or non-VALID arm, or an errored census, is UNTESTED).

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
              "qwen3_5": "Qwen3.6-35B-A3B (lane TC2, box B)", "mixtral": "Mixtral-8x7B-Instruct-v0.1 (lane TC2, box B; e4b's regime per arm, the footprint line when it ran under offload)"})
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


# ----------------------------------------------------------------------------- TC1 amendment 15 (#945): the RMSNorm composite vs e4b's fused training kernel
RMS_FAM = "qwen3rmsab"            # E4B_FUSED_RMSNORM=0 (the Hugging Face composite) vs =1 (#961's fused kernel), trimmed delta + post-#945 sync path
RMS_PAIRS = (("P24", "shipped", "fused_attn4_shipped"), ("P25", "matched", "fused_attn4_m"))   # each: <tag>_rms0 vs <tag>_rms1, two draws a side
RMS_BANDS = {"P24": (0.85, 0.97), "P25": (0.88, 0.98)}   # rms1 / rms0 s/step on stable pairs
RMS_QUALITY_MAX = 0.01            # P26: |mean held-out at N, rms1 - rms0| on each arm (the kernel is near-exact, not exact)
RMS_FLIP_AT_OR_BELOW = 0.99       # amendment 15's decision rule: both stable ratios at or below this AND P26 HELD flip the default on
FAMS.append(RMS_FAM)
NAMES[RMS_FAM] = "Qwen3-30B-A3B (amendment 15: the Hugging Face RMSNorm composite vs e4b's fused training RMSNorm, trimmed delta, post-#945 sync path)"
N_LAYERS[RMS_FAM] = 48
ATTN_CENSUS[RMS_FAM] = 192
FAM_ANCHOR[RMS_FAM] = ("e4b", "fused_attn4_m_rms0")
EXPECTED[RMS_FAM] = [("e4b", "fused_attn4_shipped_rms0"), ("e4b", "fused_attn4_shipped_rms1"), ("e4b", "fused_attn4_m_rms0"), ("e4b", "fused_attn4_m_rms1"),
                     ("e4b", "fused_attn4_m_rms1_d2"), ("e4b", "fused_attn4_m_rms0_d2"), ("e4b", "fused_attn4_shipped_rms1_d2"), ("e4b", "fused_attn4_shipped_rms0_d2")]
MATCHED |= {"fused_attn4_m_rms0", "fused_attn4_m_rms1", "fused_attn4_m_rms0_d2", "fused_attn4_m_rms1_d2"}
for _p, _k, _t in RMS_PAIRS:
    for _side in ("rms0", "rms1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")


def rms_ab_why(tag, r):
    """Amendment 15's engagement predicate: an rms1 arm requested, patched and called the fused RMSNorm; an rms0 arm patched none;
    both on the trimmed LoRA delta. Empty string = engaged."""
    ra, la = (r or {}).get("rms_ab"), (r or {}).get("lean_ab") or {}
    if not isinstance(ra, dict):
        return "no rms_ab record on the receipt: whether the fused RMSNorm ran cannot be verified"
    if "_rms1" in tag:
        checks = (("requested_env 1", ra.get("requested_env") == "1"), ("e4b_has_fused_rmsnorm", ra.get("e4b_has_fused_rmsnorm") is True),
                  ("patched > 0", (ra.get("patched") or 0) > 0), ("calls > 0", (ra.get("calls") or 0) > 0))
    else:
        checks = (("patched 0", (ra.get("patched") or 0) == 0), ("calls 0", (ra.get("calls") or 0) == 0))
    bad = [k for k, ok in checks + (("gnf4_lean_delta 1", la.get("gnf4_lean_delta") == "1"),) if not ok]
    return "" if not bad else f"fused-RMSNorm A/B not engaged ({', '.join(bad)}; record {ra})"


# ----------------------------------------------------------------------------- TC1 amendment 20 (#945): gnf4's per-pass host reuse, off vs on
REUSE_FAM = "qwen3reuseab"        # GNF4_HOST_REUSE=0 (the default) vs =1 (gnf4#444), on every current default (post-#945 sync path, trimmed delta, cost tile, fused RMSNorm)
REUSE_PAIRS = (("P33", "shipped", "fused_attn4_shipped"), ("P34", "matched", "fused_attn4_m"))   # each: <tag>_reuse0 vs <tag>_reuse1, two draws a side
REUSE_BANDS = {"P33": (0.92, 0.99), "P34": (0.93, 0.99)}   # reuse1 / reuse0 s/step on stable pairs
REUSE_FLIP_AT_OR_BELOW = 0.99     # amendment 20's decision rule: both stable ratios at or below this flip gnf4's default on
REUSE_KEEP_ABOVE = 1.01           # ... a stable ratio above this on either arm keeps it off
FAMS.append(REUSE_FAM)
NAMES[REUSE_FAM] = "Qwen3-30B-A3B (amendment 20: gnf4's per-pass host reuse off vs on, on every current default)"
N_LAYERS[REUSE_FAM] = 48
ATTN_CENSUS[REUSE_FAM] = 192
FAM_ANCHOR[REUSE_FAM] = ("e4b", "fused_attn4_m_reuse0")
EXPECTED[REUSE_FAM] = [("e4b", "fused_attn4_shipped_reuse0"), ("e4b", "fused_attn4_shipped_reuse1"), ("e4b", "fused_attn4_m_reuse0"), ("e4b", "fused_attn4_m_reuse1"),
                       ("e4b", "fused_attn4_m_reuse1_d2"), ("e4b", "fused_attn4_m_reuse0_d2"), ("e4b", "fused_attn4_shipped_reuse1_d2"), ("e4b", "fused_attn4_shipped_reuse0_d2")]
MATCHED |= {"fused_attn4_m_reuse0", "fused_attn4_m_reuse1", "fused_attn4_m_reuse0_d2", "fused_attn4_m_reuse1_d2"}
for _p, _k, _t in REUSE_PAIRS:
    for _side in ("reuse0", "reuse1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")


def reuse_ab_why(tag, r):
    """Amendment 20's engagement predicate: a reuse1 arm ran with the flag in force and the process's upload and plan memos both hit; a
    reuse0 arm ran with it off and nothing hit; both on the trimmed LoRA delta. Empty string = engaged."""
    ra, la = (r or {}).get("reuse_ab"), (r or {}).get("lean_ab") or {}
    if not isinstance(ra, dict):
        return "no reuse_ab record on the receipt: whether gnf4's host reuse ran cannot be verified"
    st = ra.get("stats") or {}
    if "_reuse1" in tag:
        checks = (("gnf4_host_reuse 1", ra.get("gnf4_host_reuse") == "1"), ("gnf4_has_host_reuse", ra.get("gnf4_has_host_reuse") is True),
                  ("upload_hits > 0", (st.get("upload_hits") or 0) > 0), ("plan_hits > 0", (st.get("plan_hits") or 0) > 0))
    else:
        checks = (("gnf4_host_reuse 0", ra.get("gnf4_host_reuse") == "0"), ("upload_hits 0", (st.get("upload_hits") or 0) == 0),
                  ("plan_hits 0", (st.get("plan_hits") or 0) == 0))
    bad = [k for k, ok in checks + (("gnf4_lean_delta 1", la.get("gnf4_lean_delta") == "1"),) if not ok]
    return "" if not bad else f"host-reuse A/B not engaged ({', '.join(bad)}; record {ra})"


# ----------------------------------------------------------------------------- TC1 amendment 21 (#945): whole-layer checkpointing vs keeping MoE activations
KEEP_FAM = "qwen3keepab"          # E4B_MOE_KEEP_LAYERS unset (whole-layer checkpointing) vs =n with NF4_QLORA_COMPACT_DELTA=1 (e4b#1007, gnf4#445)
KEEP_PAIRS = (("P35", "shipped", "fused_attn4_shipped"), ("P36", "matched", "fused_attn4_m"))   # each: <tag>_keep0 vs <tag>_keep1, two draws a side
KEEP_N = {"fused_attn4_shipped": 32, "fused_attn4_m": 16}  # the layers kept on each arm's keep1 side, sized to the 5090's headroom
KEEP_BANDS = {"P35": (0.80, 0.95), "P36": (0.86, 0.98)}    # keep1 / keep0 s/step on stable pairs
KEEP_PEAK_MAX_GB = 31.0           # P37: each keep1 side's median peak at or under this ...
KEEP_HELDOUT_MAX = 0.005          # ... and |mean held-out at N, keep1 - keep0| at or under this on each arm (gradients are identical by construction)
FAMS.append(KEEP_FAM)
NAMES[KEEP_FAM] = "Qwen3-30B-A3B (amendment 21: whole-layer gradient checkpointing vs keeping the last n layers' MoE activations)"
N_LAYERS[KEEP_FAM] = 48
ATTN_CENSUS[KEEP_FAM] = 192
FAM_ANCHOR[KEEP_FAM] = ("e4b", "fused_attn4_m_keep0")
EXPECTED[KEEP_FAM] = [("e4b", "fused_attn4_shipped_keep0"), ("e4b", "fused_attn4_shipped_keep1"), ("e4b", "fused_attn4_m_keep0"), ("e4b", "fused_attn4_m_keep1"),
                      ("e4b", "fused_attn4_m_keep1_d2"), ("e4b", "fused_attn4_m_keep0_d2"), ("e4b", "fused_attn4_shipped_keep1_d2"), ("e4b", "fused_attn4_shipped_keep0_d2")]
MATCHED |= {"fused_attn4_m_keep0", "fused_attn4_m_keep1", "fused_attn4_m_keep0_d2", "fused_attn4_m_keep1_d2"}
for _p, _k, _t in KEEP_PAIRS:
    for _side in ("keep0", "keep1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")


def keep_ab_why(tag, r):
    """Amendment 21's engagement predicate: a keep1 arm kept exactly its arm's KEEP_N layers with the compact delta in force; a keep0
    arm kept none; both on the trimmed LoRA delta. Empty string = engaged."""
    ka, la = (r or {}).get("keep_ab"), (r or {}).get("lean_ab") or {}
    if not isinstance(ka, dict):
        return "no keep_ab record on the receipt: how many layers kept their MoE activations cannot be verified"
    if "_keep1" in tag:
        want = next((n for t, n in KEEP_N.items() if tag.startswith(t + "_")), None)
        checks = (("e4b_has_moe_keep", ka.get("e4b_has_moe_keep") is True), (f"layers_kept {want}", ka.get("layers_kept") == want),
                  ("gnf4_compact_delta 1", ka.get("gnf4_compact_delta") == "1"))
    else:
        checks = (("layers_kept 0", (ka.get("layers_kept") or 0) == 0),)
    bad = [k for k, ok in checks + (("gnf4_lean_delta 1", la.get("gnf4_lean_delta") == "1"),) if not ok]
    return "" if not bad else f"MoE-keep A/B not engaged ({', '.join(bad)}; record {ka})"


# ----------------------------------------------------------------------------- TC1 amendment 22: grouped-nf4-gemm's dense route vs its fused kernels
QDENSE_FAM = "qwen3denseab"       # GNF4_TRAIN_GEMM=fused (side 0) vs =dense (side 1, gnf4#459) on the matched arm, every other default
MDENSE_FAM = "mixtraldenseab"     # the same on Mixtral-8x7B-Instruct at TC2's pin and field recipe, resident, E4B_ABSMAX_DQ=1 on both sides
DENSE_FAMS = (MDENSE_FAM, QDENSE_FAM)                    # scored in this order: P38 (Mixtral), P39 (Qwen3-30B-A3B), then P40 over both
DENSE_ARM = "fused_attn4_m"       # each family's one arm: <arm>_dense0 vs <arm>_dense1, two draws a side in ABBA order
DENSE_PREDS = {MDENSE_FAM: ("P38", "Mixtral-8x7B"), QDENSE_FAM: ("P39", "Qwen3-30B-A3B")}
DENSE_BANDS = {"P38": (0.55, 0.90), "P39": (0.85, 1.15)}   # dense1 / dense0 s/step on stable pairs
DENSE_HELDOUT_MAX = 0.01          # P40: |mean held-out at N over the dense1 draws - over the dense0 draws| on each family
DENSE_ALL_GROUPS_AT_OR_BELOW = 0.95   # amendment 22's decision rule: P39 HELD at or below this makes `auto` dense for every group count
DENSE_PINS = {QDENSE_FAM: ("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"), MDENSE_FAM: TC2_MODELS["mixtral"][:2]}   # mixtral: TC2's pin
FAMS += [QDENSE_FAM, MDENSE_FAM]
NAMES[QDENSE_FAM] = "Qwen3-30B-A3B (amendment 22: grouped-nf4-gemm's fused 4-bit kernels vs its dense route, matched arm)"
NAMES[MDENSE_FAM] = "Mixtral-8x7B-Instruct-v0.1 (amendment 22: grouped-nf4-gemm's fused 4-bit kernels vs its dense route, matched arm, resident, E4B_ABSMAX_DQ=1)"
N_LAYERS.update({QDENSE_FAM: 48, MDENSE_FAM: 32})
ATTN_CENSUS.update({QDENSE_FAM: 192, MDENSE_FAM: 128})
for _fam in DENSE_FAMS:
    FAM_ANCHOR[_fam] = ("e4b", f"{DENSE_ARM}_dense0")
    EXPECTED[_fam] = [("e4b", f"{DENSE_ARM}_dense0"), ("e4b", f"{DENSE_ARM}_dense1"), ("e4b", f"{DENSE_ARM}_dense1_d2"), ("e4b", f"{DENSE_ARM}_dense0_d2")]
MATCHED |= {f"{DENSE_ARM}_dense0", f"{DENSE_ARM}_dense1", f"{DENSE_ARM}_dense0_d2", f"{DENSE_ARM}_dense1_d2"}
for _side in ("dense0", "dense1"):
    DRAW2[("e4b", f"{DENSE_ARM}_{_side}")] = ("e4b", f"{DENSE_ARM}_{_side}_d2")


# ----------------------------------------------------------------------------- TC1 amendment 24: the RTX 5090's fp32 bmm host cost and the torch 2.12 environment
BMMAB_FAM = "qwen3bmmab"          # venv-e4b (torch 2.8.0+cu128, side tv0) vs venv-unsloth + e4b (torch 2.12.1+cu130, side tv1), matched and shipped arms
BMMAB_PAIRS = (("matched", "fused_attn4_m"), ("shipped", "fused_attn4_shipped"))   # each: <tag>_tv0 vs <tag>_tv1, two draws a side in ABBA order
BMMAB_TORCH = {"tv0": "2.8.", "tv1": "2.12."}   # the torch each side's receipt must record (env.torch prefix)
BMM_FILES = {"t28": "BMMBENCH-t28.json", "t212": "BMMBENCH-t212.json"}   # the replay's outputs (bench/tc1/bmm_bench.py), one per environment
BMM_CAP = [12, 0]                 # the registered card: sm_120 (RTX 5090)
BMM_P44_MIN_US, BMM_P44_MIN_X = 100.0, 3.0   # P44: fp32 recorded median host us per forward bmm >= 100 and >= 3x bf16's (torch 2.8)
BMM_P45_MIN_X = 2.0               # P45: fp32 cold median >= 2x its repeated-shape median (torch 2.8)
BMM_P46_MAX_X = 0.5               # P46: torch 2.12's fp32 cold median <= 0.5x torch 2.8's
BMM_P47_BAND = (0.70, 0.95)       # P47: matched tv1 / tv0 s/step on stable pairs
BMM_P48_MIN_GAP = 0.05            # P48: shipped tv1 / tv0 >= matched tv1 / tv0 + this
BMM_HELDOUT_MAX = 0.01            # P49: |mean held-out at N, tv1 - tv0| on each arm
FAMS.append(BMMAB_FAM)
NAMES[BMMAB_FAM] = "Qwen3-30B-A3B (amendment 24: venv-e4b torch 2.8.0+cu128 vs venv-unsloth torch 2.12.1+cu130, matched and shipped arms)"
N_LAYERS[BMMAB_FAM] = 48
ATTN_CENSUS[BMMAB_FAM] = 192
DENSE_PINS[BMMAB_FAM] = DENSE_PINS[QDENSE_FAM]   # amendment 24 reads the same pin through amendment 22's check
FAM_ANCHOR[BMMAB_FAM] = ("e4b", "fused_attn4_m_tv0")
EXPECTED[BMMAB_FAM] = [("e4b", f"{_t}_{_s}") for _, _t in BMMAB_PAIRS for _s in ("tv0", "tv1", "tv1_d2", "tv0_d2")]
MATCHED |= {f"fused_attn4_m_{_s}" for _s in ("tv0", "tv1", "tv0_d2", "tv1_d2")}
for _, _t in BMMAB_PAIRS:
    for _side in ("tv0", "tv1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")


def bmm_ab_why(tag, r):
    """Amendment 24's engagement predicate: the arm ran in the environment its tag names (env.torch 2.8.* for _tv0, 2.12.* for _tv1) with
    the adapter dtype its arm names (fp32 matched, native shipped). Empty string = engaged."""
    side = "tv1" if "_tv1" in tag else "tv0"
    torch_v = str(((r or {}).get("env") or {}).get("torch") or "")
    want_dt = "fp32" if tag.startswith("fused_attn4_m_") else "native"
    bad = []
    if not torch_v.startswith(BMMAB_TORCH[side]):
        bad.append(f"env.torch {torch_v or 'missing'} is not {BMMAB_TORCH[side]}*")
    if (r or {}).get("adapter_dtype") != want_dt:
        bad.append(f"adapter_dtype {(r or {}).get('adapter_dtype')!r} != {want_dt!r}")
    return "" if not bad else f"environment A/B not engaged ({'; '.join(bad)})"


# ----------------------------------------------------------------------------- TC1 amendment 25: the matched position with both frameworks on one stack
SAMESTACK_FAM = "qwen3samestack"   # e4b's anchor, second draw and reference in venv-unsloth; Unsloth as TC1; e4b's field-image arm as _t28 (two draws)
SAMESTACK_T28 = "fused_attn4_m_t28"
SAMESTACK_P50_BAND = (1.9, 2.9)    # P50: Unsloth/e4b, both on torch 2.12.1+cu130 / transformers 5.5.0, both pairs two stable draws
SAMESTACK_P51_BAND = (0.80, 0.95)  # P51: e4b matched arm venv-unsloth / venv-e4b on this host (replicates amendment 24's P47)
FAMS.append(SAMESTACK_FAM)
NAMES[SAMESTACK_FAM] = "Qwen3-30B-A3B (amendment 25: the matched set with e4b and Unsloth on one stack, torch 2.12.1+cu130 / transformers 5.5.0)"
N_LAYERS[SAMESTACK_FAM] = 48
ATTN_CENSUS[SAMESTACK_FAM] = 192
DENSE_PINS[SAMESTACK_FAM] = DENSE_PINS[QDENSE_FAM]   # the qwen3 pin, read through amendment 22's check
EXPECTED[SAMESTACK_FAM] = [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "reference_attn4_m"), ("e4b", SAMESTACK_T28),
                           ("e4b", SAMESTACK_T28 + "_d2"), ("unsloth", "ckpt_unsloth_m_d2"), ("e4b", "fused_attn4_m_d2")]
MATCHED |= {SAMESTACK_T28, SAMESTACK_T28 + "_d2"}
DRAW2[("e4b", SAMESTACK_T28)] = ("e4b", SAMESTACK_T28 + "_d2")


# TC1c amendment 9: amendment 25's family on one H100 NVL (token qwen3samestackh100), read with its own bands and a route check
SAMESTACK_H100_FAM = "qwen3samestackh100"
FAMS.append(SAMESTACK_H100_FAM)
NAMES[SAMESTACK_H100_FAM] = "Qwen3-30B-A3B on an H100 NVL (TC1c amendment 9: the matched set with e4b and Unsloth on one stack, torch 2.12.1+cu130 / transformers 5.5.0)"
N_LAYERS[SAMESTACK_H100_FAM] = 48
ATTN_CENSUS[SAMESTACK_H100_FAM] = 192
DENSE_PINS[SAMESTACK_H100_FAM] = DENSE_PINS[QDENSE_FAM]
EXPECTED[SAMESTACK_H100_FAM] = list(EXPECTED[SAMESTACK_FAM])
# per family: (position id, band, environment id, band, matched-set id or None, route id or None, the environment reading to cite)
# TC2 amendment 9: amendment 25's family on Mixtral-8x7B-Instruct at TC2's pin, resident at default settings (grouped-nf4-gemm's auto route
# is dense off sm_90 for Mixtral's calls, gnf4#463)
SAMESTACK_MIXTRAL_FAM = "mixtralsamestack"
FAMS.append(SAMESTACK_MIXTRAL_FAM)
NAMES[SAMESTACK_MIXTRAL_FAM] = "Mixtral-8x7B-Instruct-v0.1 (TC2 amendment 9: the matched set with e4b and Unsloth on one stack, resident, e4b at default settings)"
N_LAYERS[SAMESTACK_MIXTRAL_FAM] = 32
ATTN_CENSUS[SAMESTACK_MIXTRAL_FAM] = 128
DENSE_PINS[SAMESTACK_MIXTRAL_FAM] = DENSE_PINS[MDENSE_FAM]   # TC2's mixtral pin, read through amendment 22's check
EXPECTED[SAMESTACK_MIXTRAL_FAM] = list(EXPECTED[SAMESTACK_FAM])
# the route each family's fused e4b arms must show, with GNF4_TRAIN_GEMM unset: on the H100 auto resolves the process to grouped_mm (TC1c
# amendment 8's P25); on Mixtral off sm_90 auto keeps the process at `fused` and sends every call dense, so the counts say it (TC2 box M:
# 11,264 dense forward and 5,120 dense dgrad calls, no fused ones)
SAMESTACK_ROUTE = {SAMESTACK_H100_FAM: ("grouped_mm", lambda ra: ra.get("gnf4_train_gemm") == "grouped_mm"),
                   SAMESTACK_MIXTRAL_FAM: ("dense", lambda ra: ((ra.get("stats") or {}).get("dense_fwd") or 0) > 0
                                           and ((ra.get("stats") or {}).get("dense_dgrad") or 0) > 0 and ((ra.get("stats") or {}).get("fwd") or 0) == 0)}
SAMESTACK_SPECS = {SAMESTACK_FAM: ("P50", SAMESTACK_P50_BAND, "P51", SAMESTACK_P51_BAND, "P52", None, "amendment 24's P47 read 0.882"),
                   SAMESTACK_MIXTRAL_FAM: ("P29", (0.85, 1.25), "P30", (0.85, 1.02), None, "P31", "TC1 amendment 34's whole environment read 0.909 on Qwen3-30B-A3B"),
                   SAMESTACK_H100_FAM: ("P27", (1.00, 1.35), "P28", (0.80, 1.00), None, "P29", "TC1 amendment 33's P51 read 0.900 on an RTX 5090")}


def samestack_why(tag, r):
    """Amendment 25's engagement predicate for an e4b arm: _t28 arms ran torch 2.8.* (venv-e4b), every other e4b arm 2.12.* (venv-unsloth)."""
    want = "2.8." if tag.startswith(SAMESTACK_T28) else "2.12."
    torch_v = str(((r or {}).get("env") or {}).get("torch") or "")
    return "" if torch_v.startswith(want) else f"same-stack A/B not engaged (env.torch {torch_v or 'missing'} is not {want}*)"


# ----------------------------------------------------------------------------- TC1 amendment 39: the packed 4,096-token regime with both frameworks on one stack
PACKED4K_FAM = "qwen3samestack4k"   # amendment 25's arms and venvs on rows of exactly 4,096 real tokens (TC1_PACK=1), micro-batch 1 x accum 4
PACKED4K_SEQ = 4096
PACKED4K_RECIPE = (1, 4)            # (micro-batch, accum): 16,384 real tokens per optimizer step, none padded
FAMS.append(PACKED4K_FAM)
NAMES[PACKED4K_FAM] = "Qwen3-30B-A3B (amendment 39: the packed 4,096-token regime with e4b and Unsloth on one stack, torch 2.12.1+cu130 / transformers 5.5.0)"
N_LAYERS[PACKED4K_FAM] = 48
ATTN_CENSUS[PACKED4K_FAM] = 192
DENSE_PINS[PACKED4K_FAM] = DENSE_PINS[QDENSE_FAM]   # the qwen3 pin, read through amendment 22's check
EXPECTED[PACKED4K_FAM] = list(EXPECTED[SAMESTACK_FAM])
# no matched-set prediction and no route check; P86 (score_packed4k) reads whether every e4b arm that ran completed resident
# TC1 amendment 42: amendment 33's same-stack box on a second host, on the current code
SAMESTACK_HOST2_FAM = "qwen3samestackh2"
FAMS.append(SAMESTACK_HOST2_FAM)
NAMES[SAMESTACK_HOST2_FAM] = "Qwen3-30B-A3B (amendment 42: the matched set with e4b and Unsloth on one stack, a second host, the current code)"
N_LAYERS[SAMESTACK_HOST2_FAM] = 48
ATTN_CENSUS[SAMESTACK_HOST2_FAM] = 192
DENSE_PINS[SAMESTACK_HOST2_FAM] = DENSE_PINS[QDENSE_FAM]
EXPECTED[SAMESTACK_HOST2_FAM] = list(EXPECTED[SAMESTACK_FAM])
SAMESTACK_SPECS[SAMESTACK_HOST2_FAM] = ("P94", (1.9, 2.9), "P95", (0.80, 0.95), None, None, "amendment 33 read 0.900 on an EPYC 7B13")
SAMESTACK_SPECS[PACKED4K_FAM] = ("P84", (0.80, 1.60), "P85", (0.80, 1.00), None, None, "TC1 amendment 33's P51 read 0.900 at the field recipe")


# TC1 amendment 40: amendment 39's box again with e4b's chunked LM loss (E4B_CHUNKED_LM_LOSS=1 through TC1_E4B_ENV) on every e4b arm
PACKED4KCE_FAM = "qwen3samestack4kce"
FAMS.append(PACKED4KCE_FAM)
NAMES[PACKED4KCE_FAM] = "Qwen3-30B-A3B (amendment 40: the packed 4,096-token regime on one stack, e4b with its chunked LM loss)"
N_LAYERS[PACKED4KCE_FAM] = 48
ATTN_CENSUS[PACKED4KCE_FAM] = 192
DENSE_PINS[PACKED4KCE_FAM] = DENSE_PINS[QDENSE_FAM]
EXPECTED[PACKED4KCE_FAM] = list(EXPECTED[SAMESTACK_FAM])
SAMESTACK_SPECS[PACKED4KCE_FAM] = ("P87", (0.80, 1.60), "P88", (0.80, 1.00), None, None, "TC1 amendment 39: every e4b arm OOMed at step 1 without it")
# TC1 amendment 43: amendment 40's box again, the per-expert LoRA loop read as a recorded route (grouped-nf4-gemm's auto takes it for padded
# blocks over its 2 GiB limit, ~1.5 % of delta calls at 4,096 tokens on tc1-5090-91) up to LOOP_ROUTE_SHARE_MAX of a step's calls
PACKED4KCE2_FAM = "qwen3samestack4kce2"
FAMS.append(PACKED4KCE2_FAM)
NAMES[PACKED4KCE2_FAM] = "Qwen3-30B-A3B (amendment 43: the packed 4,096-token regime on one stack, e4b with its chunked LM loss, the LoRA loop a recorded route)"
N_LAYERS[PACKED4KCE2_FAM] = 48
ATTN_CENSUS[PACKED4KCE2_FAM] = 192
DENSE_PINS[PACKED4KCE2_FAM] = DENSE_PINS[QDENSE_FAM]
EXPECTED[PACKED4KCE2_FAM] = list(EXPECTED[SAMESTACK_FAM])
SAMESTACK_SPECS[PACKED4KCE2_FAM] = ("P96", (1.25, 1.65), "P97", (0.84, 0.95), None, None, "amendment 40's unquotable readings: 1.43 / 0.892")
LOOP_ROUTE_SHARE_MAX = {PACKED4KCE2_FAM: 0.05}   # a fused arm's per-expert LoRA loop is a recorded route up to this share of a step's delta calls
PACKED_FAMS = (PACKED4K_FAM, PACKED4KCE_FAM, PACKED4KCE2_FAM)
CHUNKED_FAMS = (PACKED4KCE_FAM, PACKED4KCE2_FAM)
PACKED_FIT_ID = {PACKED4K_FAM: "P86", PACKED4KCE_FAM: "P89", PACKED4KCE2_FAM: "P98"}   # every e4b arm that ran completed resident


def chunked_lm_loss_why(r):
    """Amendment 40's engagement predicate for an e4b arm: the chunked LM loss was requested (E4B_CHUNKED_LM_LOSS set, not 0), e4b has it,
    the training forwards went through it (chunked_calls > 0) and no forward fell back at run time. Empty string = engaged."""
    c = (r or {}).get("chunked_lm_loss")
    if not isinstance(c, dict):
        return "no chunked_lm_loss record on the receipt: whether the loss was chunked cannot be verified"
    bad = [k for k, ok in (("E4B_CHUNKED_LM_LOSS set", str(c.get("env") or "0").strip() not in ("", "0")),
                           ("e4b has the chunked loss", c.get("e4b_has_chunked_lm_loss") is True),
                           ("chunked_calls > 0", int(c.get("chunked_calls") or 0) > 0),
                           ("runtime_refusals 0", int(c.get("runtime_refusals") or 0) == 0)) if not ok]
    return "" if not bad else f"chunked LM loss not engaged ({', '.join(bad)}; record {c})"


def packed_why(r):
    """Amendment 39's engagement predicate, every framework: the arm read PACKED rows of exactly PACKED4K_SEQ tokens at PACKED4K_RECIPE --
    its tokens file packed, its --seq PACKED4K_SEQ, no padded token in any step, every step's real tokens seq x micro-batch x accum, and
    TC1_FREE_OUTPUTS=1 (arm_facts.free_outputs: each micro-batch's output released before the next forward).
    The family is read against its own fixture: the field recipe's seq / micro-batch / tokens never satisfy it."""
    r = r or {}
    bad = []
    if not (r.get("tokens") or {}).get("pack"):
        bad.append("the tokens file is not packed")
    if r.get("seq") != PACKED4K_SEQ:
        bad.append(f"seq {r.get('seq')} != {PACKED4K_SEQ}")
    if (r.get("micro_batch"), r.get("accum")) != PACKED4K_RECIPE:
        bad.append(f"micro-batch {r.get('micro_batch')} x accum {r.get('accum')} != {PACKED4K_RECIPE[0]} x {PACKED4K_RECIPE[1]}")
    tps, pad = r.get("tokens_per_step") or [], r.get("tokens_padded_per_step") or []
    want = PACKED4K_SEQ * int(r.get("micro_batch") or 1) * int(r.get("accum") or 1)
    if not tps or any(t != want for t in tps):
        bad.append(f"tokens per step {sorted(set(tps))[:3] or 'missing'} != {want}")
    if any(pad):
        bad.append(f"padded tokens on {sum(1 for p in pad if p)} step(s)")
    if (r.get("arm_facts") or {}).get("free_outputs") is not True:          # TC1_FREE_OUTPUTS=1: each micro-batch's output released
        bad.append(f"arm_facts.free_outputs {(r.get('arm_facts') or {}).get('free_outputs')!r}, not True")
    return "" if not bad else f"packed regime not engaged ({'; '.join(bad)})"


# ----------------------------------------------------------------------------- TC1 amendment 26: prebound Triton launches off vs on
PREBIND_FAM = "qwen3prebindab"    # E4B_TRITON_PREBIND + GNF4_TRITON_PREBIND both 0 (side pb0) vs both 1 (pb1), shipped and matched arms, venv-e4b
PREBIND_PAIRS = (("P53", "shipped", "fused_attn4_shipped"), ("P54", "matched", "fused_attn4_m"))   # each: <tag>_pb0 vs <tag>_pb1, two draws a side
PREBIND_BANDS = {"P53": (0.90, 0.98), "P54": (0.92, 0.99)}   # pb1 / pb0 s/step on stable pairs
PREBIND_HELDOUT_MAX = 0.005       # P55: |mean held-out at N, pb1 - pb0| on each arm (the same compiled kernels)
FAMS.append(PREBIND_FAM)
NAMES[PREBIND_FAM] = "Qwen3-30B-A3B (amendment 26: Triton launches prebound off vs on, e4b + grouped-nf4-gemm, venv-e4b)"
N_LAYERS[PREBIND_FAM] = 48
ATTN_CENSUS[PREBIND_FAM] = 192
DENSE_PINS[PREBIND_FAM] = DENSE_PINS[QDENSE_FAM]   # the qwen3 pin, read through amendment 22's check
FAM_ANCHOR[PREBIND_FAM] = ("e4b", "fused_attn4_m_pb0")
EXPECTED[PREBIND_FAM] = [("e4b", "fused_attn4_shipped_pb0"), ("e4b", "fused_attn4_shipped_pb1"), ("e4b", "fused_attn4_m_pb0"), ("e4b", "fused_attn4_m_pb1"),
                         ("e4b", "fused_attn4_m_pb1_d2"), ("e4b", "fused_attn4_m_pb0_d2"), ("e4b", "fused_attn4_shipped_pb1_d2"), ("e4b", "fused_attn4_shipped_pb0_d2")]
MATCHED |= {"fused_attn4_m_pb0", "fused_attn4_m_pb1", "fused_attn4_m_pb0_d2", "fused_attn4_m_pb1_d2"}
for _p, _n, _t in PREBIND_PAIRS:
    for _side in ("pb0", "pb1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")

# ----------------------------------------------------------------------------- TC1 amendment 35: amendment 26's A/B under triton 3.7.1
PREBIND37_FAM = "qwen3prebind37"  # the same eight arms as qwen3prebindab, every one in venv-unsloth (torch 2.12.1+cu130, triton 3.7.1)
PREBIND37_PAIRS = (("P66", "shipped", "fused_attn4_shipped"), ("P67", "matched", "fused_attn4_m"))
PREBIND37_BANDS = {"P66": (0.97, 1.00), "P67": (0.97, 1.00)}   # pb1 / pb0 s/step on stable pairs
PREBIND37_ENV = ("2.12", "3.7")   # the env.torch and prebind_ab triton prefixes every arm must carry
FAMS.append(PREBIND37_FAM)
NAMES[PREBIND37_FAM] = "Qwen3-30B-A3B (amendment 35: Triton launches prebound off vs on under triton 3.7.1, e4b + grouped-nf4-gemm, venv-unsloth)"
N_LAYERS[PREBIND37_FAM] = 48
ATTN_CENSUS[PREBIND37_FAM] = 192
DENSE_PINS[PREBIND37_FAM] = DENSE_PINS[QDENSE_FAM]   # the qwen3 pin, read through amendment 22's check
FAM_ANCHOR[PREBIND37_FAM] = ("e4b", "fused_attn4_m_pb0")
EXPECTED[PREBIND37_FAM] = list(EXPECTED[PREBIND_FAM])
# per family: the speed pairs, their bands, and the held-out prediction's id (MATCHED and DRAW2 already hold these tags)
PREBIND_SPECS = {PREBIND_FAM: (PREBIND_PAIRS, PREBIND_BANDS, "P55"), PREBIND37_FAM: (PREBIND37_PAIRS, PREBIND37_BANDS, "P68")}


def prebind_ab_why(tag, r, fam=None):
    """Amendment 26's engagement predicate, read off the receipt's `prebind_ab` record: a pb1 arm requested both flags and both e4b and
    grouped-nf4-gemm counted prebound launches; a pb0 arm requested neither and counted none. Amendment 35's family also requires every
    arm to have run torch 2.12 and triton 3.7. Empty string = engaged."""
    pa = (r or {}).get("prebind_ab")
    if not isinstance(pa, dict):
        return "no prebind_ab record on the receipt: whether the launches were prebound cannot be verified"
    on = "_pb1" in tag
    bad = []
    for side in ("e4b", "gnf4"):
        d = pa.get(side) or {}
        n = int((d.get("stats") or {}).get("prebound", 0))
        if not d.get("has"):
            bad.append(f"{side} has no prebound path")
        elif d.get("requested") is not on:
            bad.append(f"{side} requested {d.get('requested')}")
        elif on and n <= 0:
            bad.append(f"{side} counted no prebound launch")
        elif not on and n != 0:
            bad.append(f"{side} counted {n} prebound launches with the flag off")
    if fam == PREBIND37_FAM:
        tv, trv = str(((r or {}).get("env") or {}).get("torch") or ""), str(pa.get("triton") or "")
        if not tv.startswith(PREBIND37_ENV[0]):
            bad.append(f"env.torch {tv or 'missing'} is not {PREBIND37_ENV[0]}*")
        if not trv.startswith(PREBIND37_ENV[1]):
            bad.append(f"triton {trv or 'missing'} is not {PREBIND37_ENV[1]}*")
    return "" if not bad else f"prebind A/B not engaged ({'; '.join(bad)}; triton {pa.get('triton')})"


# ----------------------------------------------------------------------------- TC1 amendment 36: the compact padded LoRA delta (gnf4#445)
COMPACT_FAM = "qwen3compactab"    # NF4_QLORA_COMPACT_DELTA=0 (side cd0, the default) vs =1 (cd1), shipped and matched arms, venv-unsloth
COMPACT_PAIRS = (("P70", "matched", "fused_attn4_m"), ("P71", "shipped", "fused_attn4_shipped"))   # each: <tag>_cd0 vs <tag>_cd1, two draws a side
COMPACT_SPEED_BAND = (0.97, 1.02)  # cd1 / cd0 s/step on stable pairs
COMPACT_PEAK_DROP = (0.3, 2.0)     # P69: GB, median peak cd0 - cd1 on the matched arm
COMPACT_HELDOUT_MAX = 0.005        # P72: |mean held-out at N, cd1 - cd0| on each arm (the same gradient bytes)
FAMS.append(COMPACT_FAM)
NAMES[COMPACT_FAM] = "Qwen3-30B-A3B (amendment 36: grouped-nf4-gemm's padded LoRA delta saving its block vs its input, venv-unsloth)"
N_LAYERS[COMPACT_FAM] = 48
ATTN_CENSUS[COMPACT_FAM] = 192
DENSE_PINS[COMPACT_FAM] = DENSE_PINS[QDENSE_FAM]   # the qwen3 pin, read through amendment 22's check
FAM_ANCHOR[COMPACT_FAM] = ("e4b", "fused_attn4_m_cd0")
EXPECTED[COMPACT_FAM] = [("e4b", "fused_attn4_shipped_cd0"), ("e4b", "fused_attn4_shipped_cd1"), ("e4b", "fused_attn4_m_cd0"), ("e4b", "fused_attn4_m_cd1"),
                         ("e4b", "fused_attn4_m_cd1_d2"), ("e4b", "fused_attn4_m_cd0_d2"), ("e4b", "fused_attn4_shipped_cd1_d2"), ("e4b", "fused_attn4_shipped_cd0_d2")]
MATCHED |= {"fused_attn4_m_cd0", "fused_attn4_m_cd1", "fused_attn4_m_cd0_d2", "fused_attn4_m_cd1_d2"}
for _p, _n, _t in COMPACT_PAIRS:
    for _side in ("cd0", "cd1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")
# TC1 amendment 37: the same A/B with grouped-nf4-gemm#473's backward (each intermediate released at its last use), on another host
COMPACT2_FAM = "qwen3compactab2"
FAMS.append(COMPACT2_FAM)
NAMES[COMPACT2_FAM] = "Qwen3-30B-A3B (amendment 37: the compact padded LoRA delta off vs on with its backward freeing early, venv-unsloth)"
N_LAYERS[COMPACT2_FAM] = 48
ATTN_CENSUS[COMPACT2_FAM] = 192
DENSE_PINS[COMPACT2_FAM] = DENSE_PINS[QDENSE_FAM]
FAM_ANCHOR[COMPACT2_FAM] = ("e4b", "fused_attn4_m_cd0")
EXPECTED[COMPACT2_FAM] = list(EXPECTED[COMPACT_FAM])
# TC1 amendment 38: the default decision -- Qwen3-30B-A3B on a third host (both arms) and Mixtral-8x7B (matched arm, resident, defaults),
# one-sided bands: faster or level, the peak not above the default's
COMPACT3_FAM = "qwen3compactab3"
MCOMPACT_FAM = "mixtralcompactab"
FAMS += [COMPACT3_FAM, MCOMPACT_FAM]
NAMES[COMPACT3_FAM] = "Qwen3-30B-A3B (amendment 38: the compact padded LoRA delta's default decision, a third host, venv-unsloth)"
NAMES[MCOMPACT_FAM] = "Mixtral-8x7B-Instruct-v0.1 (amendment 38: the compact padded LoRA delta's default decision, matched arm, resident, venv-unsloth)"
N_LAYERS.update({COMPACT3_FAM: 48, MCOMPACT_FAM: 32})
ATTN_CENSUS.update({COMPACT3_FAM: 192, MCOMPACT_FAM: 128})
DENSE_PINS.update({COMPACT3_FAM: DENSE_PINS[QDENSE_FAM], MCOMPACT_FAM: DENSE_PINS[MDENSE_FAM]})
FAM_ANCHOR.update({COMPACT3_FAM: ("e4b", "fused_attn4_m_cd0"), MCOMPACT_FAM: ("e4b", "fused_attn4_m_cd0")})
EXPECTED[COMPACT3_FAM] = list(EXPECTED[COMPACT_FAM])
EXPECTED[MCOMPACT_FAM] = list(EXPECTED[COMPACT_FAM])      # the shipped arms are TC1_SKIP stubs on Mixtral
# per family: (peak id, peak-drop band GB (cd0 - cd1), the speed pairs, the speed band, the held-out id)
COMPACT_SPECS = {COMPACT_FAM: ("P69", COMPACT_PEAK_DROP, COMPACT_PAIRS, COMPACT_SPEED_BAND, "P72"),
                 COMPACT2_FAM: ("P73", (-0.05, 0.50), (("P74", "matched", "fused_attn4_m"), ("P75", "shipped", "fused_attn4_shipped")),
                                (0.95, 0.99), "P76"),
                 COMPACT3_FAM: ("P79", (-0.05, 99.0), (("P77", "matched", "fused_attn4_m"), ("P78", "shipped", "fused_attn4_shipped")),
                                (0.0, 0.99), "P82"),
                 MCOMPACT_FAM: ("P81", (-0.05, 99.0), (("P80", "matched", "fused_attn4_m"),), (0.0, 1.01), "P83")}


# TC1 amendment 41: e4b's chunked LM loss (E4B_CHUNKED_LM_LOSS, #1142) off vs on at the field recipe -- its default decision
CHUNKAB_FAM = "qwen3chunkab"      # E4B_CHUNKED_LM_LOSS=0 (side ce0, the default) vs =1 (ce1), shipped and matched arms, venv-unsloth
FAMS.append(CHUNKAB_FAM)
NAMES[CHUNKAB_FAM] = "Qwen3-30B-A3B (amendment 41: e4b's chunked LM loss off vs on at the field recipe, venv-unsloth)"
N_LAYERS[CHUNKAB_FAM] = 48
ATTN_CENSUS[CHUNKAB_FAM] = 192
DENSE_PINS[CHUNKAB_FAM] = DENSE_PINS[QDENSE_FAM]
FAM_ANCHOR[CHUNKAB_FAM] = ("e4b", "fused_attn4_m_ce0")
EXPECTED[CHUNKAB_FAM] = [("e4b", "fused_attn4_shipped_ce0"), ("e4b", "fused_attn4_shipped_ce1"), ("e4b", "fused_attn4_m_ce0"), ("e4b", "fused_attn4_m_ce1"),
                         ("e4b", "fused_attn4_m_ce1_d2"), ("e4b", "fused_attn4_m_ce0_d2"), ("e4b", "fused_attn4_shipped_ce1_d2"), ("e4b", "fused_attn4_shipped_ce0_d2")]
MATCHED |= {"fused_attn4_m_ce0", "fused_attn4_m_ce1", "fused_attn4_m_ce0_d2", "fused_attn4_m_ce1_d2"}
for _t in ("fused_attn4_m", "fused_attn4_shipped"):
    for _side in ("ce0", "ce1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")
COMPACT_SPECS[CHUNKAB_FAM] = ("P92", (-0.05, 99.0), (("P90", "matched", "fused_attn4_m"), ("P91", "shipped", "fused_attn4_shipped")), (0.0, 1.01), "P93",
                              ("ce0", "ce1"))   # amendment 41, one-sided: no slower than 1.01, the matched peak not above the default's + 0.05 GB
# TC1 amendment 44: E4B_CHUNKED_LM_LOSS=auto (experts4bit-qlora#1178: chunk a training forward only when its stock fp32 logits would reach
# 1 GiB) against the default at the field recipe -- amendment 41's box with `auto` in place of `1`; its default decision with amendment 43's P98
CHUNKAUTO_FAM = "qwen3chunkauto"  # E4B_CHUNKED_LM_LOSS=0 (side ca0, the default) vs =auto (ca1), shipped and matched arms, venv-unsloth
FAMS.append(CHUNKAUTO_FAM)
NAMES[CHUNKAUTO_FAM] = "Qwen3-30B-A3B (amendment 44: e4b's chunked LM loss off vs auto at the field recipe, venv-unsloth)"
N_LAYERS[CHUNKAUTO_FAM] = 48
ATTN_CENSUS[CHUNKAUTO_FAM] = 192
DENSE_PINS[CHUNKAUTO_FAM] = DENSE_PINS[QDENSE_FAM]
FAM_ANCHOR[CHUNKAUTO_FAM] = ("e4b", "fused_attn4_m_ca0")
EXPECTED[CHUNKAUTO_FAM] = [("e4b", "fused_attn4_shipped_ca0"), ("e4b", "fused_attn4_shipped_ca1"), ("e4b", "fused_attn4_m_ca0"), ("e4b", "fused_attn4_m_ca1"),
                           ("e4b", "fused_attn4_m_ca1_d2"), ("e4b", "fused_attn4_m_ca0_d2"), ("e4b", "fused_attn4_shipped_ca1_d2"), ("e4b", "fused_attn4_shipped_ca0_d2")]
MATCHED |= {"fused_attn4_m_ca0", "fused_attn4_m_ca1", "fused_attn4_m_ca0_d2", "fused_attn4_m_ca1_d2"}
for _t in ("fused_attn4_m", "fused_attn4_shipped"):
    for _side in ("ca0", "ca1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")
COMPACT_SPECS[CHUNKAUTO_FAM] = ("P102", (-0.05, 99.0), (("P101", "matched", "fused_attn4_m"), ("P100", "shipped", "fused_attn4_shipped")), (0.0, 1.02),
                                "P103", ("ca0", "ca1"))   # amendment 44, one-sided: no slower than 1.02, the matched peak not above the default's + 0.05 GB
CHUNKAUTO_FORWARDS = 240          # amendment 44's P99: every training forward of a 60-step, accum-4 arm ran stock under the gate (small_calls)
# TC1 amendment 45: OMP_NUM_THREADS at the host's physical cores (om0, every box so far) vs at the container's CPU allotment (om1), e4b's matched
# arm and Unsloth's, both in venv-unsloth
OMPAB_FAM = "qwen3ompab"
FAMS.append(OMPAB_FAM)
NAMES[OMPAB_FAM] = "Qwen3-30B-A3B (amendment 45: OMP_NUM_THREADS at the physical cores vs the container's CPU allotment, e4b and Unsloth in venv-unsloth)"
N_LAYERS[OMPAB_FAM] = 48
ATTN_CENSUS[OMPAB_FAM] = 192
DENSE_PINS[OMPAB_FAM] = DENSE_PINS[QDENSE_FAM]
FAM_ANCHOR[OMPAB_FAM] = ("e4b", "fused_attn4_m_om0")
EXPECTED[OMPAB_FAM] = [("e4b", "fused_attn4_m_om0"), ("e4b", "fused_attn4_m_om1"), ("unsloth", "ckpt_unsloth_m_om0"), ("unsloth", "ckpt_unsloth_m_om1"),
                       ("unsloth", "ckpt_unsloth_m_om1_d2"), ("unsloth", "ckpt_unsloth_m_om0_d2"), ("e4b", "fused_attn4_m_om1_d2"), ("e4b", "fused_attn4_m_om0_d2")]
MATCHED |= {"fused_attn4_m_om0", "fused_attn4_m_om1", "fused_attn4_m_om0_d2", "fused_attn4_m_om1_d2",
            "ckpt_unsloth_m_om0", "ckpt_unsloth_m_om1", "ckpt_unsloth_m_om0_d2", "ckpt_unsloth_m_om1_d2"}
for _fw, _t in (("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m")):
    for _side in ("om0", "om1"):
        DRAW2[(_fw, f"{_t}_{_side}")] = (_fw, f"{_t}_{_side}_d2")
#: (prediction, framework, arm stem, one-sided band on om1 / om0): e4b faster by at least 3 %, Unsloth no slower than 1 %
OMPAB_PAIRS = (("P104", "e4b", "fused_attn4_m", (0.0, 0.97)), ("P105", "unsloth", "ckpt_unsloth_m", (0.0, 1.01)))
OMPAB_HELDOUT_MAX = 0.005         # P106, on each framework

# ----------------------------------------------------------------------------- TC1 amendment 46: grouped-nf4-gemm's decoded route vs its fused kernels
QDEC_FAM = "qwen3decab"           # GNF4_TRAIN_GEMM=fused (side dec0) vs =decoded (dec1, gnf4#487) on the matched arm, venv-e4b, every other default
ODEC_FAM = "olmoedecab"           # the same on OLMoE-1B-7B-0924-Instruct at TC2's pin, TC1's field recipe
DEC_FAMS = (ODEC_FAM, QDEC_FAM)   # scored in this order: P108 (OLMoE), P109 (Qwen3-30B-A3B), then P110 / P111 over both
DEC_ARM = "fused_attn4_m"         # each family's one arm: <arm>_dec0 vs <arm>_dec1, two draws a side in ABBA order
DEC_PREDS = {ODEC_FAM: ("P108", "OLMoE-1B-7B"), QDEC_FAM: ("P109", "Qwen3-30B-A3B")}
DEC_BANDS = {"P108": (0.0, 0.95), "P109": (0.97, 1.25)}   # dec1 / dec0 s/step on stable pairs (P108 one-sided: at most 0.95)
DEC_HELDOUT_MAX = 0.01            # P110: |mean held-out at N over the dec1 draws - over the dec0 draws| on each family
DEC_PEAK_MAX = 0.30               # P111: the matched peak dec1 - dec0 (GB) on each family; the route's decode transient is at most 256 MiB a call
DEC_TORCH = "2.8."                # every arm in venv-e4b: RD1's software (torch 2.8.0+cu128, triton 3.4.0)
DEC_GATE_FILE = "decgate.json"    # P107: the box's first step, written by tc1_run.sh's tc1_decoded_gate
DEC_AUTO_MIN_ROWS = 48            # the decision rule's line: RD1's per-call reading, rows per present group (quoted in the evidence)
DENSE_PINS.update({QDEC_FAM: DENSE_PINS[QDENSE_FAM], ODEC_FAM: TC2_MODELS["olmoe"][:2]})   # read through amendment 22's pin check
FAMS += [ODEC_FAM, QDEC_FAM]
NAMES[QDEC_FAM] = "Qwen3-30B-A3B (amendment 46: grouped-nf4-gemm's fused 4-bit kernels vs its decoded route, matched arm, venv-e4b)"
NAMES[ODEC_FAM] = "OLMoE-1B-7B-0924-Instruct (amendment 46: grouped-nf4-gemm's fused 4-bit kernels vs its decoded route, matched arm, venv-e4b, TC2's pin)"
N_LAYERS.update({QDEC_FAM: 48, ODEC_FAM: TC2_MODELS["olmoe"][2]})
ATTN_CENSUS.update({QDEC_FAM: 192, ODEC_FAM: ATTN_CENSUS["olmoe"]})
for _fam in DEC_FAMS:
    FAM_ANCHOR[_fam] = ("e4b", f"{DEC_ARM}_dec0")
    EXPECTED[_fam] = [("e4b", f"{DEC_ARM}_dec0"), ("e4b", f"{DEC_ARM}_dec1"), ("e4b", f"{DEC_ARM}_dec1_d2"), ("e4b", f"{DEC_ARM}_dec0_d2")]
MATCHED |= {f"{DEC_ARM}_dec0", f"{DEC_ARM}_dec1", f"{DEC_ARM}_dec0_d2", f"{DEC_ARM}_dec1_d2"}
for _side in ("dec0", "dec1"):
    DRAW2[("e4b", f"{DEC_ARM}_{_side}")] = ("e4b", f"{DEC_ARM}_{_side}_d2")


def chunk_ab_why(tag, r):
    """Amendment 41's engagement predicate: the arm ran torch 2.12 (venv-unsloth) and its `chunked_lm_loss` record shows the side its tag
    names -- ce1: E4B_CHUNKED_LM_LOSS set, e4b has the loss, chunked training forwards, no run-time fallback; ce0: no chunked forward.
    Empty string = engaged."""
    r = r or {}
    tv = str((r.get("env") or {}).get("torch") or "")
    if not tv.startswith("2.12"):
        return f"chunked-loss A/B not engaged (env.torch {tv or 'missing'} is not 2.12*)"
    if "_ce1" in tag:
        return chunked_lm_loss_why(r)
    c = r.get("chunked_lm_loss")
    if not isinstance(c, dict):
        return "no chunked_lm_loss record on the receipt: whether the loss was chunked cannot be verified"
    return "" if int(c.get("chunked_calls") or 0) == 0 else f"chunked-loss A/B not engaged (the ce0 side made {c.get('chunked_calls')} chunked calls)"


def chunk_auto_why(tag, r):
    """Amendment 44's engagement predicate: the arm ran torch 2.12 (venv-unsloth) and its `chunked_lm_loss` record shows the side its tag
    names -- ca1: E4B_CHUNKED_LM_LOSS=auto, e4b has the loss and patched the model, no run-time fallback, the record carries small_calls
    (#1178); ca0: nothing patched, no chunked and no gated forward. Whether ca1's gate fired is P99's to score, not validity's. Empty
    string = engaged."""
    r = r or {}
    tv = str((r.get("env") or {}).get("torch") or "")
    if not tv.startswith("2.12"):
        return f"chunked-loss auto A/B not engaged (env.torch {tv or 'missing'} is not 2.12*)"
    c = r.get("chunked_lm_loss")
    if not isinstance(c, dict):
        return "no chunked_lm_loss record on the receipt: whether the loss was gated cannot be verified"
    if "_ca1" in tag:
        bad = [k for k, ok in (("E4B_CHUNKED_LM_LOSS=auto", str(c.get("env") or "").strip().lower() == "auto"),
                               ("e4b has the chunked loss", c.get("e4b_has_chunked_lm_loss") is True),
                               ("patched 1", int(c.get("patched") or 0) == 1),
                               ("small_calls recorded", "small_calls" in c),
                               ("runtime_refusals 0", int(c.get("runtime_refusals") or 0) == 0)) if not ok]
        return "" if not bad else f"chunked-loss auto not engaged ({', '.join(bad)}; record {c})"
    bad = [k for k, ok in (("patched 0", int(c.get("patched") or 0) == 0), ("chunked_calls 0", int(c.get("chunked_calls") or 0) == 0),
                           ("small_calls 0", int(c.get("small_calls") or 0) == 0)) if not ok]
    return "" if not bad else f"chunked-loss auto A/B not engaged (the ca0 side: {', '.join(bad)}; record {c})"


def ompab_why(r):
    """Amendment 45's engagement predicate, either framework: the arm ran torch 2.12 (venv-unsloth) and recorded the OpenMP thread count
    it ran with, with torch's intra-op pool at that count. Which side ran fewer threads is the scorer's to check. Empty string = engaged."""
    r = r or {}
    tv = str((r.get("env") or {}).get("torch") or "")
    if not tv.startswith("2.12"):
        return f"threads A/B not engaged (env.torch {tv or 'missing'} is not 2.12*)"
    af = r.get("arm_facts") or {}
    omp, tn = af.get("omp_num_threads"), af.get("torch_num_threads")
    if not str(omp or "").isdigit() or tn is None:
        return f"threads A/B not engaged (arm_facts omp_num_threads {omp!r}, torch_num_threads {tn!r}: the thread count cannot be verified)"
    if int(omp) != int(tn):
        return f"threads A/B not engaged (OMP_NUM_THREADS {omp} but torch ran {tn} intra-op threads)"
    return ""


def score_ompab(F, fam=OMPAB_FAM):
    """TC1-PREREG amendment 45: P104 (e4b) and P105 (Unsloth) -- om1 / om0 s/step within OMPAB_PAIRS' one-sided band, each over two VALID
    draws a side with each side's draws within 5 %, and every om1 receipt at FEWER OpenMP threads than every om0 receipt of its framework
    (else no contrast: UNTESTED); P106 -- on each framework |mean held-out at N, om1 - om0| <= OMPAB_HELDOUT_MAX."""
    R = F.get(fam)
    if not R:
        return []
    rows = {(x["fw"], x["tag"]): x for x in R["rows"]}
    out, p106 = [], []
    for pid, fw, t, (lo, hi) in OMPAB_PAIRS:
        O, N = R["draws"].get((fw, f"{t}_om0"), {}), R["draws"].get((fw, f"{t}_om1"), {})
        thr = {side: sorted({int(((rows.get((fw, tag)) or {}).get("r") or {}).get("arm_facts", {}).get("omp_num_threads") or 0)
                             for tag in (f"{t}_{side}", f"{t}_{side}_d2")}) for side in ("om0", "om1")}
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("om0", O), ("om1", N)))
            out.append((pid, fam, "UNTESTED", f"{fw}: two stable VALID draws a side are registered -- {why}"))
            p106.append((fw, None, why))
            continue
        if not (thr["om1"] and thr["om0"] and max(thr["om1"]) < min(thr["om0"]) and min(thr["om1"]) > 0):
            out.append((pid, fam, "UNTESTED", f"{fw}: no contrast -- OpenMP threads om0 {thr['om0']} vs om1 {thr['om1']} (om1 must run fewer)"))
            p106.append((fw, None, "no contrast"))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        h0, h1 = O.get("heldout_list") or [], N.get("heldout_list") or []
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p106.append((fw, dq, f"held-out at N om0 {[round(v, 4) for v in h0 if v is not None]} om1 {[round(v, 4) for v in h1 if v is not None]}"))
        out.append((pid, fam, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{fw}: om1 / om0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}; OpenMP threads om0 "
                    f"{thr['om0']} vs om1 {thr['om1']}; s/step om0 {O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), om1 "
                    f"{N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} (within {100 * N['stability']:.1f}%)"))
    ev = "; ".join(f"{n}: " + (f"mean held-out om1 - om0 {d:+.4f} (|.| <= {OMPAB_HELDOUT_MAX}); {e}" if d is not None else e) for n, d, e in p106)
    if any(d is not None and abs(d) > OMPAB_HELDOUT_MAX for _, d, _ in p106):
        out.append(("P106", fam, "FALSIFIED", ev))
    elif any(d is None for _, d, _ in p106):
        out.append(("P106", fam, "UNTESTED", ev))
    else:
        out.append(("P106", fam, "HELD", ev))
    return out
def decoded_ab_why(tag, r):
    """Amendment 46's engagement predicate, read off the arm's `route_ab` record (grouped-nf4-gemm's nf4_route.ROUTE_STATS for the process) and
    its torch: every arm ran venv-e4b (env.torch 2.8.*); a dec1 arm ran with the route resolved to `decoded` and counted decoded forward AND
    decoded dgrad calls, no dense forward; a dec0 arm ran `fused` and counted no decoded and no dense forward (a record without the
    `decoded_fwd` counter -- grouped-nf4-gemm before #487 -- cannot show that). Empty string = engaged."""
    r = r or {}
    tv = str((r.get("env") or {}).get("torch") or "")
    if not tv.startswith(DEC_TORCH):
        return f"decoded-route A/B not engaged (env.torch {tv or 'missing'} is not {DEC_TORCH}*: venv-e4b is registered)"
    ra = r.get("route_ab")
    if not isinstance(ra, dict):
        return "no route_ab record on the receipt: the training GEMM route this arm took cannot be verified"
    st = ra.get("stats") or {}
    if "_dec1" in tag:
        checks = [("gnf4_train_gemm decoded", ra.get("gnf4_train_gemm") == "decoded"), ("decoded_fwd > 0", (st.get("decoded_fwd") or 0) > 0),
                  ("decoded_dgrad > 0", (st.get("decoded_dgrad") or 0) > 0), ("dense_fwd 0", (st.get("dense_fwd") or 0) == 0)]
    else:
        checks = [("gnf4_train_gemm fused", ra.get("gnf4_train_gemm") == "fused"), ("decoded_fwd 0", st.get("decoded_fwd") == 0),
                  ("dense_fwd 0", (st.get("dense_fwd") or 0) == 0)]
    bad = [k for k, ok in checks if not ok]
    return "" if not bad else f"decoded-route A/B not engaged ({', '.join(bad)}; record {ra})"


def score_chunkauto_gate(F, fam=CHUNKAUTO_FAM):
    """TC1-PREREG amendment 44's P99: on every ca1 arm that ran (VALID), the size gate never fired. FALSIFIED iff a VALID ca1 arm chunked a
    forward (chunked_calls > 0: the gate fired). HELD iff all four ca1 arms are VALID and each reads exactly chunked 0 / small
    CHUNKAUTO_FORWARDS. UNTESTED otherwise -- a ca1 arm not VALID, or one that chunked nothing but gated a count other than
    CHUNKAUTO_FORWARDS (an extra or missing training forward: a question about the instrument or the trainer, reported, not the gate)."""
    R = F.get(fam)
    if not R:
        return []
    rows = {x["tag"]: x for x in R["rows"] if x["fw"] == "e4b"}
    ca1 = [t for _, t in EXPECTED[fam] if "_ca1" in t]
    ev, bad, missing, count = [], [], [], []
    for t in ca1:
        x = rows.get(t)
        if not x or x.get("verdict") != "VALID":
            missing.append(f"{t} {x.get('verdict') if x else 'absent'}")
            continue
        c = (x.get("r") or {}).get("chunked_lm_loss") or {}
        ch, sm = int(c.get("chunked_calls") or 0), int(c.get("small_calls") or 0)
        ev.append(f"`{t}` chunked {ch} / small {sm}")
        if ch > 0:
            bad.append(t)
        elif sm != CHUNKAUTO_FORWARDS:
            count.append(f"{t} gated {sm}, not {CHUNKAUTO_FORWARDS}")
    if bad:
        v = "FALSIFIED"
    elif missing or count:
        v = "UNTESTED"
    else:
        v = "HELD"
    tail = ("; not VALID: " + ", ".join(missing)) if missing else ""
    tail += ("; forward count (instrument, not the gate): " + ", ".join(count)) if count else ""
    return [("P99", fam, v, f"want chunked 0 / small {CHUNKAUTO_FORWARDS} on every ca1 arm: " + "; ".join(ev) + tail)]


def compact_ab_why(tag, r):
    """Amendment 36's engagement predicate: the arm ran torch 2.12 (venv-unsloth), grouped-nf4-gemm resolved the compact delta to its side
    (keep_ab.gnf4_compact_delta "1" on cd1, "0" on cd0), no layer kept its MoE activations, and the padded LoRA path ran (lean_ab's
    lora_path_calls.padded > 0), so the compact node is the route the switch changes. Empty string = engaged."""
    r = r or {}
    ka, la = r.get("keep_ab"), r.get("lean_ab") or {}
    if not isinstance(ka, dict):
        return "no keep_ab record on the receipt: whether the compact delta was in force cannot be verified"
    want = "1" if "_cd1" in tag else "0"
    tv = str((r.get("env") or {}).get("torch") or "")
    padded = int(((la.get("lora_path_calls") or {}).get("padded")) or 0)
    bad = [k for k, ok in ((f"gnf4_compact_delta {want}", ka.get("gnf4_compact_delta") == want), ("layers_kept 0", int(ka.get("layers_kept") or 0) == 0),
                           ("the padded LoRA path ran", padded > 0), ("env.torch 2.12*", tv.startswith("2.12"))) if not ok]
    return "" if not bad else (f"compact-delta A/B not engaged ({', '.join(bad)}; compact {ka.get('gnf4_compact_delta')!r}, kept {ka.get('layers_kept')!r}, "
                               f"padded calls {padded}, torch {tv or 'missing'})")


# ----------------------------------------------------------------------------- TC1 amendment 28: the double-quantized expert absmax as a default
QDQ_FAM = "qwen3dqab"             # E4B_ABSMAX_DQ=0 (side dq0, the default) vs =1 (dq1, #1040) on the matched arm, resident, every other default
MDQ_FAM = "mixtraldqab"           # the same on Mixtral-8x7B-Instruct at TC2's pin and field recipe
DQ_FAMS = (QDQ_FAM, MDQ_FAM)      # scored in this order: P56 (Qwen3-30B-A3B), P57 (Mixtral), then P58 over both
DQ_ARM = "fused_attn4_m"
DQ_PREDS = {QDQ_FAM: ("P56", "Qwen3-30B-A3B"), MDQ_FAM: ("P57", "Mixtral-8x7B")}
DQ_SPEED_BAND = (0.97, 1.03)      # dq1 / dq0 s/step on stable pairs
DQ_PEAK_DROP = {QDQ_FAM: (1.25, 1.45), MDQ_FAM: (1.9, 2.3)}   # GB, median peak dq0 - dq1
DQ_HELDOUT_MAX = 0.005            # P58: |mean held-out at N, dq1 - dq0| on each family
FAMS += [QDQ_FAM, MDQ_FAM]
NAMES[QDQ_FAM] = "Qwen3-30B-A3B (amendment 28: the expert absmax fp32 vs double-quantized, matched arm, resident)"
NAMES[MDQ_FAM] = "Mixtral-8x7B-Instruct-v0.1 (amendment 28: the expert absmax fp32 vs double-quantized, matched arm, resident)"
N_LAYERS.update({QDQ_FAM: 48, MDQ_FAM: 32})
ATTN_CENSUS.update({QDQ_FAM: 192, MDQ_FAM: 128})
DENSE_PINS.update({QDQ_FAM: DENSE_PINS[QDENSE_FAM], MDQ_FAM: DENSE_PINS[MDENSE_FAM]})   # read through amendment 22's pin check
for _fam in DQ_FAMS:
    FAM_ANCHOR[_fam] = ("e4b", f"{DQ_ARM}_dq0")
    EXPECTED[_fam] = [("e4b", f"{DQ_ARM}_dq0"), ("e4b", f"{DQ_ARM}_dq1"), ("e4b", f"{DQ_ARM}_dq1_d2"), ("e4b", f"{DQ_ARM}_dq0_d2")]
MATCHED |= {f"{DQ_ARM}_dq0", f"{DQ_ARM}_dq1", f"{DQ_ARM}_dq0_d2", f"{DQ_ARM}_dq1_d2"}
for _side in ("dq0", "dq1"):
    DRAW2[("e4b", f"{DQ_ARM}_{_side}")] = ("e4b", f"{DQ_ARM}_{_side}_d2")


def dq_ab_why(fam, tag, r):
    """Amendment 28's engagement predicate: a dq1 arm double-quantized the absmax of every MoE layer (absmax_dq true, absmax_dq_modules ==
    the family's registered n_layers); a dq0 arm kept it fp32 (absmax_dq false). Empty string = engaged."""
    r = r or {}
    if "_dq1" in tag:
        bad = [k for k, ok in (("absmax_dq true", r.get("absmax_dq") is True),
                               (f"absmax_dq_modules {N_LAYERS[fam]}", r.get("absmax_dq_modules") == N_LAYERS[fam])) if not ok]
    else:
        bad = [] if r.get("absmax_dq") is not True else ["absmax_dq false"]
    return "" if not bad else f"absmax A/B not engaged ({', '.join(bad)}; absmax_dq {r.get('absmax_dq')!r}, modules {r.get('absmax_dq_modules')!r})"


# ----------------------------------------------------------------------------- TC1 amendment 32: one variable, Triton (3.4 vs 3.7.1 in venv-e4b)
TRITON_FAM = "qwen3tritonab"      # venv-e4b with its triton 3.4 (side tr0) vs triton 3.7.1 first on PYTHONPATH (tr1); prebound launches off both sides
TRITON_PAIRS = (("P59", "matched", "fused_attn4_m"), ("P60", "shipped", "fused_attn4_shipped"))   # each: <tag>_tr0 vs <tag>_tr1, two draws a side
TRITON_BANDS = {"P59": (0.82, 0.95), "P60": (0.85, 0.98)}   # tr1 / tr0 s/step on stable pairs
TRITON_HELDOUT_MAX = 0.005        # P61: |mean held-out at N, tr1 - tr0| on each arm
TRITON_WANT = {"tr0": "3.4.", "tr1": "3.7."}
FAMS.append(TRITON_FAM)
NAMES[TRITON_FAM] = "Qwen3-30B-A3B (amendment 32: venv-e4b with triton 3.4 vs 3.7.1, prebound launches off, matched and shipped arms)"
N_LAYERS[TRITON_FAM] = 48
ATTN_CENSUS[TRITON_FAM] = 192
DENSE_PINS[TRITON_FAM] = DENSE_PINS[QDENSE_FAM]   # the qwen3 pin, read through amendment 22's check
FAM_ANCHOR[TRITON_FAM] = ("e4b", "fused_attn4_m_tr0")
EXPECTED[TRITON_FAM] = [("e4b", f"{_t}_{_s}") for _, _, _t in TRITON_PAIRS for _s in ("tr0", "tr1", "tr1_d2", "tr0_d2")]
MATCHED |= {f"fused_attn4_m_{_s}" for _s in ("tr0", "tr1", "tr0_d2", "tr1_d2")}
for _p, _n, _t in TRITON_PAIRS:
    for _side in ("tr0", "tr1"):
        DRAW2[("e4b", f"{_t}_{_side}")] = ("e4b", f"{_t}_{_side}_d2")


def triton_ab_why(tag, r):
    """Amendment 32's engagement predicate: the arm ran the Triton its tag names (env.triton 3.4.* on tr0, 3.7.* on tr1) with the prebound
    launches off on both e4b and grouped-nf4-gemm (prebind_ab requested false). Empty string = engaged."""
    r = r or {}
    side = "tr1" if "_tr1" in tag else "tr0"
    tv = str((r.get("env") or {}).get("triton") or "")
    bad = [] if tv.startswith(TRITON_WANT[side]) else [f"env.triton {tv or 'missing'} is not {TRITON_WANT[side]}*"]
    pa = r.get("prebind_ab")
    if not isinstance(pa, dict):
        bad.append("no prebind_ab record (the prebound launches must be off on both sides)")
    else:
        bad += [f"{sd} prebind requested {(pa.get(sd) or {}).get('requested')!r}" for sd in ("e4b", "gnf4") if (pa.get(sd) or {}).get("requested") is not False]
    return "" if not bad else f"Triton A/B not engaged ({'; '.join(bad)})"


# ----------------------------------------------------------------------------- TC1 amendment 34: amendment 24's environment gain, split
ENVSPLIT_FAM = "qwen3envsplit"    # matched arm: e0 venv-e4b (torch 2.8, tf 5.18), e1 venv-e4b-tf55 (torch 2.8, tf 5.5), e2 venv-unsloth (torch 2.12, tf 5.5)
ENVSPLIT_SIDES = {"e0": ("2.8.", "5.18."), "e1": ("2.8.", "5.5."), "e2": ("2.12.", "5.5.")}   # (torch prefix, transformers prefix)
ENVSPLIT_PREDS = (("P62", "transformers 5.5 over 5.18 (torch 2.8)", "e1", "e0", (0.90, 1.00)),
                  ("P63", "torch 2.12 + triton 3.7 over torch 2.8 + triton 3.4 (transformers 5.5)", "e2", "e1", (0.85, 0.99)),
                  ("P64", "the whole environment (amendment 24's 0.882, re-read)", "e2", "e0", (0.80, 0.95)))
ENVSPLIT_HELDOUT_MAX = 0.005      # P65: |mean held-out at N| e1 - e0 and e2 - e0
FAMS.append(ENVSPLIT_FAM)
NAMES[ENVSPLIT_FAM] = "Qwen3-30B-A3B (amendment 34: the matched arm in three environments -- transformers 5.18 / 5.5 on torch 2.8, and torch 2.12)"
N_LAYERS[ENVSPLIT_FAM] = 48
ATTN_CENSUS[ENVSPLIT_FAM] = 192
DENSE_PINS[ENVSPLIT_FAM] = DENSE_PINS[QDENSE_FAM]   # the qwen3 pin, read through amendment 22's check
FAM_ANCHOR[ENVSPLIT_FAM] = ("e4b", "fused_attn4_m_e0")
EXPECTED[ENVSPLIT_FAM] = [("e4b", f"fused_attn4_m_{t}") for t in ("e0", "e1", "e2", "e2_d2", "e1_d2", "e0_d2")]
MATCHED |= {f"fused_attn4_m_{t}" for t in ("e0", "e1", "e2", "e0_d2", "e1_d2", "e2_d2")}
for _side in ENVSPLIT_SIDES:
    DRAW2[("e4b", f"fused_attn4_m_{_side}")] = ("e4b", f"fused_attn4_m_{_side}_d2")


def envsplit_why(tag, r):
    """Amendment 34's engagement predicate: the arm ran the torch and transformers its side names, with the prebound launches off."""
    r = r or {}
    side = next((sd for sd in ENVSPLIT_SIDES if f"_{sd}" in tag), None)
    if side is None:
        return f"tag {tag!r} names no environment"
    want_t, want_tf = ENVSPLIT_SIDES[side]
    env = r.get("env") or {}
    tv, tfv = str(env.get("torch") or ""), str(env.get("transformers") or "")
    bad = [] if tv.startswith(want_t) else [f"env.torch {tv or 'missing'} is not {want_t}*"]
    if not tfv.startswith(want_tf):
        bad.append(f"env.transformers {tfv or 'missing'} is not {want_tf}*")
    pa = r.get("prebind_ab")
    if not isinstance(pa, dict):
        bad.append("no prebind_ab record (the prebound launches must be off on every side)")
    else:
        bad += [f"{sd} prebind requested {(pa.get(sd) or {}).get('requested')!r}" for sd in ("e4b", "gnf4") if (pa.get(sd) or {}).get("requested") is not False]
    return "" if not bad else f"environment split not engaged ({'; '.join(bad)})"


def dense_ab_why(fam, tag, r):
    """Amendment 22's engagement predicate, read off the arm's `route_ab` record (grouped-nf4-gemm's nf4_route.ROUTE_STATS for the process):
    a dense1 arm ran with the route resolved to `dense` and counted dense forward AND dense dgrad calls; a dense0 arm ran `fused` and counted
    no dense forward (a record without the `dense_fwd` counter -- grouped-nf4-gemm before #459 -- cannot show that); on mixtral both sides
    also ran with the double-quantized expert absmax (the receipt's `absmax_dq`). Empty string = engaged."""
    r = r or {}
    ra = r.get("route_ab")
    if not isinstance(ra, dict):
        return "no route_ab record on the receipt: the training GEMM route this arm took cannot be verified"
    st = ra.get("stats") or {}
    if "_dense1" in tag:
        checks = [("gnf4_train_gemm dense", ra.get("gnf4_train_gemm") == "dense"), ("dense_fwd > 0", (st.get("dense_fwd") or 0) > 0),
                  ("dense_dgrad > 0", (st.get("dense_dgrad") or 0) > 0)]
    else:
        checks = [("gnf4_train_gemm fused", ra.get("gnf4_train_gemm") == "fused"), ("dense_fwd 0", st.get("dense_fwd") == 0)]
    if fam == MDENSE_FAM:
        checks.append(("absmax_dq true", r.get("absmax_dq") is True))
    bad = [k for k, ok in checks if not ok]
    if not bad:
        return ""
    return f"dense-route A/B not engaged ({', '.join(bad)}; record {ra}" + (f"; absmax_dq {r.get('absmax_dq')!r}" if fam == MDENSE_FAM else "") + ")"


# ----------------------------------------------------------------------------- TC1 amendment 23: the memory census, e4b against Unsloth (no speed read)
MEMCENSUS_FAM = "qwen3memcensus"   # e4b fp32 absmax, e4b double-quantized absmax (E4B_ABSMAX_DQ=1), Unsloth; micro-batch 1 x accum 8; one draw each; --mem-census 1
MEMCENSUS_ARMS = (("e4b", "fused_attn4_m_mb1"), ("e4b", "fused_attn4_m_mb1_dq"), ("unsloth", "ckpt_unsloth_m_mb1"))
MEMCENSUS_LABELS = {MEMCENSUS_ARMS[0]: "e4b fp32 absmax", MEMCENSUS_ARMS[1]: "e4b dq absmax", MEMCENSUS_ARMS[2]: "Unsloth"}
MEMCENSUS_PIN = ("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
MEMCENSUS_RECIPE = (1, 8)          # (micro-batch, accum): the mb1 recipe, TC1's tokens per step
P41_ANALYTIC = 29.0e9 / 64 * 4    # 1.8125e9 B: 29.0 B expert parameters / 64 x 4 bytes, the registration's rounding -- the receipt's own expert_params wins when present
P41_TOL = 0.02                     # P41: each absmax within 2 % of its analytic value
P41_DQ_RATIO = 3.94                # ... the dq arm's analytic value = the fp32 one / 3.94 (#1040's measured ratio)
P42_MIN = 0.90                     # P42: attributed_fraction >= 0.90 on every arm
P43_BAND = (0.5e9, 2.5e9)          # P43: e4b dq peak - Unsloth peak (bytes, peak allocated) inside this band
NO_SPEED_FAMS = {MEMCENSUS_FAM: "the memory census slows the step (amendment 23), so no speed is read on this token; positions stay with the boxes that read them"}
FAMS.append(MEMCENSUS_FAM)
NAMES[MEMCENSUS_FAM] = "Qwen3-30B-A3B (amendment 23: the memory census -- e4b fp32 absmax, e4b double-quantized absmax, Unsloth; micro-batch 1 × accum 8; no speed read)"
N_LAYERS[MEMCENSUS_FAM] = 48
ATTN_CENSUS[MEMCENSUS_FAM] = 192
FAM_ANCHOR[MEMCENSUS_FAM] = MEMCENSUS_ARMS[0]
EXPECTED[MEMCENSUS_FAM] = list(MEMCENSUS_ARMS)
MATCHED |= {"fused_attn4_m_mb1_dq"}                     # fused_attn4_m_mb1 and ckpt_unsloth_m_mb1 are registered matched already


def memcensus_why(r):
    """Amendment 23's predicates on an OK row: the pin, the mb1 recipe, a census on the receipt (an ERRORED census is recorded, never VOID --
    the predictions read it UNTESTED), and on e4b the absmax the tag names (absmax_dq true on `_dq`, false otherwise). Empty string = as registered."""
    r = r or {}
    bad = []
    mid, rev = MEMCENSUS_PIN
    if r.get("model") not in (None, mid) or r.get("revision") not in (None, rev):
        bad.append(f"model/revision {r.get('model')} @ {str(r.get('revision'))[:12]} != the registered pin {mid} @ {rev[:12]}")
    if (r.get("micro_batch"), r.get("accum")) != MEMCENSUS_RECIPE:
        bad.append(f"recipe micro-batch {r.get('micro_batch')} x accum {r.get('accum')} != the registered {MEMCENSUS_RECIPE[0]} x {MEMCENSUS_RECIPE[1]}")
    if not isinstance(r.get("mem_census"), dict):
        bad.append("no mem_census on the receipt: the registered instrument (--mem-census 1) did not run")
    if r.get("framework") == "e4b":
        want = (r.get("tag") or "").endswith("_dq")
        if (r.get("absmax_dq") is True) != want:
            bad.append(f"absmax_dq {r.get('absmax_dq')!r} on {r.get('tag')}: the tag names the {'double-quantized' if want else 'fp32'} expert absmax")
    return "; ".join(bad)


def _gb(b, nd=3):
    return "—" if b is None else f"{b / 1e9:.{nd}f}"


def _mc_reading(R, key):
    """(the census of one VALID arm, "") or (None, why it cannot be read): missing, not VALID, no census, or an errored census."""
    x = next((x for x in R["rows"] if (x["fw"], x["tag"]) == key), None)
    if x is None or x["r"] is None:
        return None, f"{key[0]}/{key[1]} missing"
    if x["verdict"] != "VALID":
        w = x.get("why") or x.get("reason") or ""
        return None, f"{key[0]}/{key[1]} {x['verdict']}" + (f" ({w[:140]})" if w else "")
    mc = x["r"].get("mem_census")
    if not isinstance(mc, dict):
        return None, f"{key[0]}/{key[1]}: no mem_census"
    if mc.get("error"):
        return None, f"{key[0]}/{key[1]}: the census errored ({str(mc['error'])[:140]})"
    return mc, ""


def _mc_at_peak(mc):
    """The live bytes at the peak by class: the static classes the census labelled there (other_frozen.* folded into other_frozen) and
    `transient` = every other live block (activations, workspaces, backward temporaries) -- the window's requested-size basis."""
    pw = (mc or {}).get("peak_window") or {}
    sp = pw.get("static_at_peak") or {}
    out = {}
    for k, v in sp.items():
        cls = k.split(".", 1)[0]
        out[cls] = out.get(cls, 0) + v
    if pw.get("live_bytes") is not None:
        out["transient"] = pw["live_bytes"] - sum(sp.values())
    return out


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
FIELD_MICRO_BATCH = 2                  # TC1-PREREG "Fixture": the field recipe's micro-batch (x accum 4); TC2 amendment 8's box Q runs its PRIMARY pair at 1 x 8


def anchor_of(fam):
    """R11: the family's e4b anchor arm -- fused_attn4_m everywhere but gpt-oss, where e4b trains attention only (attn_only_m)."""
    return TC2_ANCHOR.get(fam) or FAM_ANCHOR.get(fam, QUALITY_ANCHOR)


def registered_draw2(fam, key):
    """R11: the second-draw key of `key` when the family REGISTERS it (in EXPECTED), else None -- DRAW2 is global, the registration per family."""
    k2 = DRAW2.get(key)
    return k2 if (k2 is not None and k2 in EXPECTED.get(fam, EXPECTED["qwen3"])) else None


def pair_recipe(e_rec, o_rec, label):
    """TC2 amendment 8: the micro-batch x accum a quoted PRIMARY pair ran, named on its position line when it is not the field recipe's
    micro-batch -- box Q's pair runs at 1 x 8 under the primary tags. "" on the field recipe (every pair before amendment 8) or when a side
    does not record it; a pair whose two sides record different recipes names both."""
    e = (e_rec.get("micro_batch"), e_rec.get("accum")) if is_ok(e_rec) else (None, None)
    o = (o_rec.get("micro_batch"), o_rec.get("accum")) if is_ok(o_rec) else (None, None)
    if e[0] is None or o[0] is None:
        return ""
    if e == o:
        return "" if e[0] == FIELD_MICRO_BATCH else f"micro-batch {e[0]} × accum {e[1]}"
    return f"recipes differ: e4b micro-batch {e[0]} × accum {e[1]}, {label} micro-batch {o[0]} × accum {o[1]}"


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
    if fam in DENSE_PINS:                                                # amendment 22: each dense-route family's pin (mixtral: TC2's)
        mid, rev = DENSE_PINS[fam]
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
            elif r.get("lora_path_loop_steps") and not (fam in LOOP_ROUTE_SHARE_MAX and max(
                    [v for v in (r.get("lora_loop_share") or []) if v is not None] or [1.0]) <= LOOP_ROUTE_SHARE_MAX[fam]):
                st = r["lora_path_loop_steps"]                       # amendment 43: a share under the family's bound is a recorded route
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
    if fam in (SYNC_FAM, PROF945_FAM, LEAN_FAM, TILE_FAM, RMS_FAM, REUSE_FAM, KEEP_FAM) and fw == "e4b":   # amendments 10 / 12-15 / 20-21: the arm ran the sync path its tag names (13-15, 20-21: the new one)
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
    if fam == RMS_FAM and fw == "e4b":                  # amendment 15: ... and the RMSNorm path its tag names, on the trimmed delta
        w = rms_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == REUSE_FAM and fw == "e4b":                # amendment 20: ... and the host-reuse setting its tag names, on the trimmed delta
        w = reuse_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == KEEP_FAM and fw == "e4b":                 # amendment 21: ... and the layers-kept setting its tag names, on the trimmed delta
        w = keep_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam in DENSE_FAMS and fw == "e4b":              # amendment 22: the training GEMM route its tag names (mixtral: on the double-quantized absmax)
        w = dense_ab_why(fam, r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam in DEC_FAMS and fw == "e4b":                # amendment 46: venv-e4b, and the training GEMM route its tag names
        w = decoded_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == MEMCENSUS_FAM:                            # amendment 23: the pin, the mb1 recipe, the census on the receipt, the absmax the tag names
        w = memcensus_why(r)
        if w:
            why.append(w)
    if fam == BMMAB_FAM and fw == "e4b":               # amendment 24: the environment and adapter dtype its tag names
        w = bmm_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam in SAMESTACK_SPECS and fw == "e4b":         # amendment 25 / TC1c amendment 9: the venv its tag names
        w = samestack_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam in PACKED_FAMS:                             # amendments 39 / 40: packed rows of exactly 4,096 tokens at micro-batch 1 x accum 4, every framework
        w = packed_why(r)
        if w:
            why.append(w)
    if fam == CHUNKAB_FAM and fw == "e4b":             # amendment 41: the chunked loss its tag names, torch 2.12
        w = chunk_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == CHUNKAUTO_FAM and fw == "e4b":           # amendment 44: the auto side's record, the default side unpatched, torch 2.12
        w = chunk_auto_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == OMPAB_FAM:                               # amendment 45: torch 2.12 and the recorded thread count, either framework
        w = ompab_why(r)
        if w:
            why.append(w)
    if fam in CHUNKED_FAMS and fw == "e4b":            # amendments 40 / 43: the chunked LM loss on every e4b arm
        w = chunked_lm_loss_why(r)
        if w:
            why.append(w)
    if fam in PREBIND_SPECS and fw == "e4b":           # amendments 26 / 35: the prebound launches its tag names, engaged on both sides
        w = prebind_ab_why(r.get("tag") or "", r, fam)
        if w:
            why.append(w)
    if fam in COMPACT_SPECS and fam not in (CHUNKAB_FAM, CHUNKAUTO_FAM) and fw == "e4b":   # amendments 36-38: the padded LoRA delta its tag names (41 / 44 share the scorer only)
        w = compact_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam in DQ_FAMS and fw == "e4b":                 # amendment 28: the expert absmax its tag names
        w = dq_ab_why(fam, r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == TRITON_FAM and fw == "e4b":              # amendment 32: the Triton its tag names, prebound launches off
        w = triton_ab_why(r.get("tag") or "", r)
        if w:
            why.append(w)
    if fam == ENVSPLIT_FAM and fw == "e4b":            # amendment 34: the torch and transformers its tag names, prebound launches off
        w = envsplit_why(r.get("tag") or "", r)
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
                positions[other]["recipe"] = pair_recipe(e_anchor, recs[k], label)      # TC2 amendment 8: named off the field recipe's micro-batch
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
    if fam in NO_SPEED_FAMS:                            # amendment 23: no speed is read on a census token -- every position on it is marked not quoted
        for group in (positions, secondary, labelled, native):
            for k in list(group):
                group[k] = {"label": group[k].get("label", str(k)), "quoted": False, "why": f"not quoted: {NO_SPEED_FAMS[fam]}"}
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
    # R11: the footprint line reads e4b under expert offload against a resident framework, so an e4b anchor that ran RESIDENT (TC2 amendment 6) has none
    anchor_offload = e_anchor.get("offload") if is_ok(e_anchor) else None
    fp = (footprint(fam, recs, draws, verdicts)
          if ((fam in FOOTPRINT_FAMS and anchor_offload is not False) or anchor_offload) else None)
    return {"fam": fam, "rows": rows, "V": V, "verdicts": verdicts, "parity": par, "draws": draws, "positions": positions, "secondary": secondary,
            "native": native, "labelled": labelled, "equivalence": equiv, "frozen": frozen, "anchor_probe": anchor_probe, "profile": profile,
            "noise_floor": noise_floor, "equiv_band": band, "anchor_sha": anchor_sha, "footprint": fp, "common_set": common, "anchor_key": qa,
            "anchor_offload": anchor_offload,
            "anchor_micro_batch": e_anchor.get("micro_batch") if is_ok(e_anchor) else None,     # TC2 amendment 8: P4 reads the field recipe's step
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


def score_rmsab(F):
    """TC1-PREREG amendment 15 (#945), on the qwen3rmsab box: P24 (shipped) and P25 (matched) -- e4b's fused training RMSNorm steps at
    rms1 / rms0 within RMS_BANDS[pid], the median over two VALID draws a side with each side's draws within 5 %; P26 -- on each arm the
    two sides' mean held-out at N agree within RMS_QUALITY_MAX. Outside FALSIFIED; a missing / non-VALID / unstable side UNTESTED."""
    R = F.get(RMS_FAM)
    if not R:
        return []
    out, qual = [], []
    for pid, name, t in RMS_PAIRS:
        O, N = R["draws"].get(("e4b", f"{t}_rms0"), {}), R["draws"].get(("e4b", f"{t}_rms1"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("rms0", O), ("rms1", N)))
            out.append((pid, RMS_FAM, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            qual.append((name, None, why))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        lo, hi = RMS_BANDS[pid]
        h0 = [x["r"].get("eval_loss_final") for x in R["rows"] if x["fw"] == "e4b" and x["tag"] in (f"{t}_rms0", f"{t}_rms0_d2")]
        h1 = [x["r"].get("eval_loss_final") for x in R["rows"] if x["fw"] == "e4b" and x["tag"] in (f"{t}_rms1", f"{t}_rms1_d2")]
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        qual.append((name, dq, f"held-out at N rms0 {[round(v, 4) for v in h0]} rms1 {[round(v, 4) for v in h1]}"))
        out.append((pid, RMS_FAM, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name}: rms1 / rms0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}; "
                    f"s/step rms0 {O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), rms1 {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} "
                    f"(within {100 * N['stability']:.1f}%); {'flip-eligible on speed' if ratio_ <= RMS_FLIP_AT_OR_BELOW else 'no flip on speed'}"))
    if any(d is None for _, d, _ in qual):
        out.append(("P26", RMS_FAM, "UNTESTED", "; ".join(f"{n}: {e}" for n, d, e in qual if d is None)))
    else:
        held = all(abs(d) <= RMS_QUALITY_MAX for _, d, _ in qual)
        out.append(("P26", RMS_FAM, "HELD" if held else "FALSIFIED",
                    "; ".join(f"{n}: mean held-out rms1 - rms0 {d:+.4f} (|.| <= {RMS_QUALITY_MAX}); {e}" for n, d, e in qual)))
    return out


def score_reuseab(F):
    """TC1-PREREG amendment 20 (#945), on the qwen3reuseab box: P33 (shipped arm) and P34 (matched arm) -- gnf4's per-pass host reuse steps at
    reuse1 / reuse0 within REUSE_BANDS[pid], the median over two VALID draws a side with each side's draws within 5 %. Outside the band
    FALSIFIED; an unstable, missing or non-VALID side UNTESTED. The evidence names the decision rule's reading (flip / keep / no effect),
    each side's held-out at N and the reuse1 side's hit counts."""
    R = F.get(REUSE_FAM)
    if not R:
        return []
    out = []
    for pid, name, t in REUSE_PAIRS:
        O, N = R["draws"].get(("e4b", f"{t}_reuse0"), {}), R["draws"].get(("e4b", f"{t}_reuse1"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("reuse0", O), ("reuse1", N)))
            out.append((pid, REUSE_FAM, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        lo, hi = REUSE_BANDS[pid]
        rule = ("flip-eligible" if ratio_ <= REUSE_FLIP_AT_OR_BELOW else "KEEP off" if ratio_ > REUSE_KEEP_ABOVE else "no measurable effect")
        r1 = next((x["r"] for x in R["rows"] if (x["fw"], x["tag"]) == ("e4b", f"{t}_reuse1")), None) or {}
        hits = (r1.get("reuse_ab") or {}).get("stats") or {}
        out.append((pid, REUSE_FAM, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name}: reuse1 / reuse0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}; decision reading: {rule}; "
                    f"s/step reuse0 {O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), reuse1 {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} "
                    f"(within {100 * N['stability']:.1f}%); held-out at N reuse0 {f(O.get('heldout'), 4)} / reuse1 {f(N.get('heldout'), 4)}; "
                    f"reuse1 hits (process) {json.dumps(hits, sort_keys=True)}"))
    return out


def score_keepab(F):
    """TC1-PREREG amendment 21 (#945), on the qwen3keepab box: P35 (shipped arm, 32 layers kept) and P36 (matched arm, 16 kept) -- keep1 / keep0
    s/step within KEEP_BANDS[pid], the median over two VALID draws a side with each side's draws within 5 %; P37 -- on each arm the keep1
    side's median peak is at or under KEEP_PEAK_MAX_GB and the two sides' mean held-out at N agree within KEEP_HELDOUT_MAX. Outside
    FALSIFIED; a missing / non-VALID / unstable side UNTESTED (an OOM keep1 side is a non-VALID side: P37 FALSIFIED reads it)."""
    R = F.get(KEEP_FAM)
    if not R:
        return []
    out, p37 = [], []
    for pid, name, t in KEEP_PAIRS:
        O, N = R["draws"].get(("e4b", f"{t}_keep0"), {}), R["draws"].get(("e4b", f"{t}_keep1"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("keep0", O), ("keep1", N)))
            out.append((pid, KEEP_FAM, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            oom = R["verdicts"].get(("e4b", f"{t}_keep1")) == "OOM" or R["verdicts"].get(("e4b", f"{t}_keep1_d2")) == "OOM"
            p37.append((name, "OOM" if oom else None, why))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        lo, hi = KEEP_BANDS[pid]
        h0 = [x["r"].get("eval_loss_final") for x in R["rows"] if x["fw"] == "e4b" and x["tag"] in (f"{t}_keep0", f"{t}_keep0_d2")]
        h1 = [x["r"].get("eval_loss_final") for x in R["rows"] if x["fw"] == "e4b" and x["tag"] in (f"{t}_keep1", f"{t}_keep1_d2")]
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p37.append((name, (N.get("peak"), dq), f"peak keep0 {f(O.get('peak'), 2)} / keep1 {f(N.get('peak'), 2)} GB; held-out at N keep0 "
                                               f"{[round(v, 4) for v in h0]} keep1 {[round(v, 4) for v in h1]}"))
        out.append((pid, KEEP_FAM, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name} ({KEEP_N[t]} of 48 layers kept): keep1 / keep0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs "
                    f"{[lo, hi]}; s/step keep0 {O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), keep1 "
                    f"{N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} (within {100 * N['stability']:.1f}%); peak keep0 {f(O.get('peak'), 2)} / keep1 {f(N.get('peak'), 2)} GB"))
    if any(v == "OOM" for _, v, _ in p37):
        out.append(("P37", KEEP_FAM, "FALSIFIED", "; ".join(f"{n}: {'keep1 OOM' if v == 'OOM' else e}" for n, v, e in p37)))
    elif any(v is None or v[0] is None or v[1] is None for _, v, _ in p37):
        out.append(("P37", KEEP_FAM, "UNTESTED", "; ".join(f"{n}: {e}" for n, _, e in p37)))
    else:
        held = all(v[0] <= KEEP_PEAK_MAX_GB and abs(v[1]) <= KEEP_HELDOUT_MAX for _, v, _ in p37)
        out.append(("P37", KEEP_FAM, "HELD" if held else "FALSIFIED",
                    "; ".join(f"{n}: keep1 peak {v[0]:.2f} GB (<= {KEEP_PEAK_MAX_GB}), mean held-out keep1 - keep0 {v[1]:+.4f} (|.| <= {KEEP_HELDOUT_MAX}); {e}"
                              for n, v, e in p37)))
    return out


def score_denseab(F):
    """TC1-PREREG amendment 22, on the mixtraldenseab / qwen3denseab tokens: P38 (Mixtral) and P39 (Qwen3-30B-A3B) -- grouped-nf4-gemm's
    dense route steps at dense1 / dense0 within DENSE_BANDS[pid], the median over two VALID draws a side with each side's draws within 5 %;
    P40 -- on each family the two sides' mean held-out at N agree within DENSE_HELDOUT_MAX. Outside FALSIFIED; a missing, non-VALID (a side
    the engagement predicate voided) or unstable side UNTESTED. P40 is one reading over both families: FALSIFIED when either family is read
    outside the bound, else UNTESTED while either family is unread, else HELD."""
    if not any(fam in F for fam in DENSE_FAMS):
        return []
    out, p40 = [], []
    for fam in DENSE_FAMS:
        pid, name = DENSE_PREDS[fam]
        R = F.get(fam)
        if not R:
            out.append((pid, fam, "UNTESTED", f"{name}: no {fam} receipts in this directory"))
            p40.append((fam, None, "no receipts"))
            continue
        D0, D1 = R["draws"].get(("e4b", f"{DENSE_ARM}_dense0"), {}), R["draws"].get(("e4b", f"{DENSE_ARM}_dense1"), {})
        if not (D0.get("usable") and D1.get("usable") and D0.get("draws") == 2 and D1.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("dense0", D0), ("dense1", D1)))
            out.append((pid, fam, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            p40.append((fam, None, why))
            continue
        ratio_ = D1["s"] / D0["s"]
        cross = [n / o for n in D1["s_list"] for o in D0["s_list"]]
        lo, hi = DENSE_BANDS[pid]
        h0, h1 = D0.get("heldout_list") or [], D1.get("heldout_list") or []
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p40.append((fam, dq, f"held-out at N dense0 {[round(v, 4) for v in h0 if v is not None]} dense1 {[round(v, 4) for v in h1 if v is not None]}"))
        r1 = next((x["r"] for x in R["rows"] if (x["fw"], x["tag"]) == ("e4b", f"{DENSE_ARM}_dense1")), None) or {}
        stats = (r1.get("route_ab") or {}).get("stats") or {}
        rule = ""
        if fam == QDENSE_FAM:
            rule = (f"; at or below {DENSE_ALL_GROUPS_AT_OR_BELOW} (the decision rule's every-group-count reading)" if ratio_ <= DENSE_ALL_GROUPS_AT_OR_BELOW
                    else f"; above {DENSE_ALL_GROUPS_AT_OR_BELOW} (fused stays at every group count on this family)")
        out.append((pid, fam, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name}: dense1 / dense0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}{rule}; "
                    f"s/step dense0 {D0['s_list'][0]:.3f} / {D0['s_list'][1]:.3f} (within {100 * D0['stability']:.1f}%), dense1 {D1['s_list'][0]:.3f} / {D1['s_list'][1]:.3f} "
                    f"(within {100 * D1['stability']:.1f}%); peak dense0 {f(D0.get('peak'), 2)} / dense1 {f(D1.get('peak'), 2)} GB; "
                    f"dense1 route counts (process) {json.dumps(stats, sort_keys=True)}"))
    ev = "; ".join(f"{fam}: " + (f"mean held-out dense1 - dense0 {d:+.4f} (|.| <= {DENSE_HELDOUT_MAX}); {e}" if d is not None else e) for fam, d, e in p40)
    if any(d is not None and abs(d) > DENSE_HELDOUT_MAX for _, d, _ in p40):
        out.append(("P40", "denseab", "FALSIFIED", ev))
    elif any(d is None for _, d, _ in p40):
        out.append(("P40", "denseab", "UNTESTED", ev))
    else:
        out.append(("P40", "denseab", "HELD", ev))
    return out


def decgate_reading(d):
    """Amendment 46's P107 record: `decgate.json` in the receipts directory (tc1_run.sh's tc1_decoded_gate) -> (record or None, why)."""
    p = os.path.join(d or "", DEC_GATE_FILE)
    if not (d and os.path.exists(p)):
        return None, f"no {DEC_GATE_FILE} in this directory: the gate's record is missing"
    try:
        g = json.load(open(p))
    except Exception as e:
        return None, f"{DEC_GATE_FILE} unreadable: {type(e).__name__}: {e}"
    return (g, "") if isinstance(g, dict) else (None, f"{DEC_GATE_FILE} is not an object")


def score_decgate(d):
    """TC1-PREREG amendment 46's P107: grouped-nf4-gemm's compiled correctness tests for the decoded route passed on the box's card, the box's
    first step. FALSIFIED iff the gate ran and a test failed (`failures` > 0 in a run). HELD iff it ran, every run collected tests and exited 0
    with no failure, error or skip, every required test passed, and the record says passed. UNTESTED otherwise -- no record, a gate that
    could not run, an error that is not a failed assertion (collection, setup), a skipped test (the card did not run it), or a required test
    that did not pass (`missing`: a grouped-nf4-gemm without the route has none of them, and `-k` alone would then pass on the dequant tests):
    the box is refused on all of these, but only a failed test speaks to the route."""
    g, why = decgate_reading(d)
    if g is None:
        return [("P107", "decgate", "UNTESTED", why)]
    runs = [x for x in (g.get("runs") or []) if isinstance(x, dict)]
    ev = (f"grouped-nf4-gemm @{str(g.get('gnf4_sha') or '?')[:12]} on {g.get('gpu') or '?'} (torch {g.get('torch') or '?'}, triton {g.get('triton') or '?'}): "
          + ("; ".join(f"{x.get('name')} rc {x.get('rc')}: {x.get('tests')} tests, {x.get('failures')} failed, {x.get('errors')} errors, "
                       f"{x.get('skipped')} skipped" + (f", required tests not passed: {x.get('missing')}" if x.get("missing") else "")
                       for x in runs) or "no run recorded")
          + (f"; {g.get('reason')}" if g.get("reason") else ""))
    if not g.get("ran"):
        return [("P107", "decgate", "UNTESTED", ev)]
    if any(int(x.get("failures") or 0) > 0 for x in runs):
        return [("P107", "decgate", "FALSIFIED", ev)]
    clean = bool(runs) and all(x.get("rc") == 0 and int(x.get("tests") or 0) > 0 and "missing" in x and not x.get("missing")
                               and not (int(x.get("errors") or 0) or int(x.get("skipped") or 0)) for x in runs)
    return [("P107", "decgate", "HELD" if clean and g.get("passed") is True else "UNTESTED", ev)]


def score_decodedab(F):
    """TC1-PREREG amendment 46, on the olmoedecab / qwen3decab tokens: P108 (OLMoE, at most 0.95) and P109 (Qwen3-30B-A3B, in [0.97, 1.25]) --
    grouped-nf4-gemm's decoded route steps at dec1 / dec0 within DEC_BANDS[pid], the median over two VALID draws a side with each side's
    draws within 5 %; P110 -- on each family the two sides' mean held-out at N agree within DEC_HELDOUT_MAX; P111 -- on each family the matched
    peak dec1 - dec0 is at most DEC_PEAK_MAX GB. Outside FALSIFIED; a missing, non-VALID or unstable side UNTESTED. P110 and P111 are each one
    reading over both families: FALSIFIED when either family is read outside the bound, else UNTESTED while either is unread, else HELD."""
    if not any(fam in F for fam in DEC_FAMS):
        return []
    out, p110, p111 = [], [], []
    for fam in DEC_FAMS:
        pid, name = DEC_PREDS[fam]
        R = F.get(fam)
        if not R:
            out.append((pid, fam, "UNTESTED", f"{name}: no {fam} receipts in this directory"))
            p110.append((fam, None, "no receipts"))
            p111.append((fam, None, "no receipts"))
            continue
        D0, D1 = R["draws"].get(("e4b", f"{DEC_ARM}_dec0"), {}), R["draws"].get(("e4b", f"{DEC_ARM}_dec1"), {})
        if not (D0.get("usable") and D1.get("usable") and D0.get("draws") == 2 and D1.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("dec0", D0), ("dec1", D1)))
            out.append((pid, fam, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            p110.append((fam, None, why))
            p111.append((fam, None, why))
            continue
        ratio_ = D1["s"] / D0["s"]
        cross = [n / o for n in D1["s_list"] for o in D0["s_list"]]
        lo, hi = DEC_BANDS[pid]
        h0, h1 = D0.get("heldout_list") or [], D1.get("heldout_list") or []
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p110.append((fam, dq, f"held-out at N dec0 {[round(v, 4) for v in h0 if v is not None]} dec1 {[round(v, 4) for v in h1 if v is not None]}"))
        dp = (D1["peak"] - D0["peak"]) if (D0.get("peak") is not None and D1.get("peak") is not None) else None
        p111.append((fam, dp, f"peak dec0 {f(D0.get('peak'), 2)} / dec1 {f(D1.get('peak'), 2)} GB"))
        r1 = next((x["r"] for x in R["rows"] if (x["fw"], x["tag"]) == ("e4b", f"{DEC_ARM}_dec1")), None) or {}
        stats = (r1.get("route_ab") or {}).get("stats") or {}
        band = f"<= {hi}" if lo <= 0 else f"in {[lo, hi]}"
        note = ""
        if pid == "P109" and ratio_ < lo:
            note = f"; below {lo}: Qwen3-30B-A3B gains below RD1's {DEC_AUTO_MIN_ROWS}-row line too (a lower line is re-asked)"
        out.append((pid, fam, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name}: dec1 / dec0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {band}{note}; "
                    f"s/step dec0 {D0['s_list'][0]:.3f} / {D0['s_list'][1]:.3f} (within {100 * D0['stability']:.1f}%), dec1 {D1['s_list'][0]:.3f} / "
                    f"{D1['s_list'][1]:.3f} (within {100 * D1['stability']:.1f}%); dec1 route counts (process) {json.dumps(stats, sort_keys=True)}"))
    for pid, rows, bound, label in (("P110", p110, DEC_HELDOUT_MAX, "mean held-out dec1 - dec0"), ("P111", p111, DEC_PEAK_MAX, "peak dec1 - dec0 GB")):
        ev = "; ".join(f"{fam}: " + (f"{label} {d:+.4f} (bound {bound}); {e}" if d is not None else e) for fam, d, e in rows)
        over = (lambda d: abs(d) > bound + 1e-9) if pid == "P110" else (lambda d: d > bound + 1e-9)   # a reading AT the bound holds (float noise aside)
        if any(d is not None and over(d) for _, d, _ in rows):
            out.append((pid, "decodedab", "FALSIFIED", ev))
        elif any(d is None for _, d, _ in rows):
            out.append((pid, "decodedab", "UNTESTED", ev))
        else:
            out.append((pid, "decodedab", "HELD", ev))
    return out


def _mc_top(mc, n=3, static=False):
    """The `n` largest live-at-peak groups of one census, static: groups excluded unless `static`."""
    gs = [g for g in (mc or {}).get("live_at_peak_top") or [] if static or not str(g.get("group", "")).startswith("static:")]
    return ", ".join(f"{g['group']} {_gb(g.get('bytes'))} GB ×{g.get('count')}" for g in gs[:n]) or "none"


def score_memcensus(F):
    """TC1-PREREG amendment 23, on the qwen3memcensus token, read from the receipts' mem_census (a missing or non-VALID arm, or an errored
    census, leaves every prediction that needs it UNTESTED):
      P41 -- e4b's fp32 expert absmax (the static census at the end of setup) within 2 % of the analytic expert_params / 64 x 4 bytes (the
             receipt's own expert_params; 29.0e9 / 64 x 4 when absent), and the dq arm's within 2 % of that / 3.94: HELD iff both legs are
             inside, FALSIFIED iff a leg read is outside;
      P42 -- attributed_fraction >= 0.90 on every arm, each read only when its reduced window holds the run's peak (peak_in_window);
      P43 -- excess_after_absmax = e4b dq peak allocated - Unsloth peak allocated, HELD iff inside [0.5e9, 2.5e9] bytes; the evidence names the
             largest class of the excess at the peak (the static classes the census labelled there and `transient`, dq minus Unsloth) and
             each side's largest live-at-peak groups."""
    R = F.get(MEMCENSUS_FAM)
    if not R:
        return []
    ke, kd, ku = MEMCENSUS_ARMS
    reads = {k: _mc_reading(R, k) for k in MEMCENSUS_ARMS}
    out = []
    me, md, mu = reads[ke][0], reads[kd][0], reads[ku][0]
    ep = next((v for v in ((m.get("static_after_setup") or {}).get("expert_params") for m in (me, md) if m) if v), None)
    analytic = ep / 64 * 4 if ep else P41_ANALYTIC
    basis = f"expert_params {ep:,} / 64 × 4 bytes" if ep else "29.0e9 / 64 × 4 bytes (no expert_params on the receipts)"
    legs = []
    for key, target, how in ((ke, analytic, "the analytic value"), (kd, analytic / P41_DQ_RATIO, f"the analytic value / {P41_DQ_RATIO}")):
        m, why = reads[key]
        name = MEMCENSUS_LABELS[key]
        b = (m.get("static_after_setup") or {}).get("expert_absmax") if m else None
        if b is None:
            legs.append((None, f"{name}: UNREAD -- {why or 'no expert_absmax in static_after_setup'}"))
            continue
        d = (b - target) / target
        legs.append((abs(d) <= P41_TOL, f"{name} {_gb(b, 4)} GB vs {how} {_gb(target, 4)} GB (Δ {100 * d:+.2f} %, within {100 * P41_TOL:.0f} %: {'yes' if abs(d) <= P41_TOL else 'NO'})"))
    be, bd = ((m.get("static_after_setup") or {}).get("expert_absmax") if m else None for m in (me, md))
    ratio_txt = f"; fp32 / dq bytes {be / bd:.3f}" if (be and bd) else ""
    v = "FALSIFIED" if any(ok is False for ok, _ in legs) else ("HELD" if all(ok for ok, _ in legs) else "UNTESTED")
    out.append(("P41", MEMCENSUS_FAM, v, f"analytic {_gb(analytic, 4)} GB = {basis}; " + "; ".join(e for _, e in legs) + ratio_txt))
    legs = []
    for key in MEMCENSUS_ARMS:
        m, why = reads[key]
        name = MEMCENSUS_LABELS[key]
        if m is None:
            legs.append((None, f"{name}: UNREAD -- {why}"))
            continue
        af, pw = m.get("attributed_fraction"), m.get("peak_window") or {}
        if af is None:
            legs.append((None, f"{name}: no live-at-peak reduction (trace: {m.get('trace')})"))
            continue
        if pw.get("peak_in_window") is not True:
            legs.append((None, f"{name}: the reduced window does not hold the run's peak (window peak {_gb(pw.get('peak_bytes'))} GB at {pw.get('checkpoint')}, "
                               f"max allocated {_gb(pw.get('max_allocated_at_checkpoint'))} GB there)"))
            continue
        un = [g for g in m.get("live_at_peak_top") or [] if str(g.get("group", "")).startswith("unattributed")]
        legs.append((af >= P42_MIN, f"{name} {af:.4f} ({'>=' if af >= P42_MIN else '<'} {P42_MIN}; largest unattributed: "
                                    + (f"{un[0]['group']} {_gb(un[0].get('bytes'))} GB" if un else "none") + ")"))
    v = "FALSIFIED" if any(ok is False for ok, _ in legs) else ("HELD" if all(ok for ok, _ in legs) else "UNTESTED")
    out.append(("P42", MEMCENSUS_FAM, v, "attributed fraction at the peak: " + "; ".join(e for _, e in legs)))
    (md, wd), (mu, wu) = reads[kd], reads[ku]
    pd_, pu = (md or {}).get("peak_allocated_bytes"), (mu or {}).get("peak_allocated_bytes")
    if pd_ is None or pu is None:
        why = "; ".join(w for w in (wd, wu) if w) or "a peak_allocated_bytes is missing"
        out.append(("P43", MEMCENSUS_FAM, "UNTESTED", f"e4b dq and Unsloth peaks are both registered -- {why}"))
        return out
    ex, (lo, hi) = pd_ - pu, P43_BAND
    ad, au = _mc_at_peak(md), _mc_at_peak(mu)
    diffs = sorted(((c, ad.get(c, 0) - au.get(c, 0)) for c in set(ad) | set(au)), key=lambda kv: (-kv[1], kv[0]))
    largest = f"{diffs[0][0]} {diffs[0][1] / 1e9:+.3f} GB" if (diffs and diffs[0][1] > 0) else "none (no class is larger on e4b at the peak)"
    fp32 = f" (e4b fp32-absmax peak {_gb(me.get('peak_allocated_bytes'))} GB)" if me else ""
    out.append(("P43", MEMCENSUS_FAM, "HELD" if lo <= ex <= hi else "FALSIFIED",
                f"excess_after_absmax = e4b dq peak {_gb(pd_)} - Unsloth peak {_gb(pu)} = {ex / 1e9:+.3f} GB vs [{lo / 1e9}, {hi / 1e9}] GB{fp32}; "
                f"the excess's largest group at the peak: {largest}; by class at the peak, dq - Unsloth (GB): "
                + ", ".join(f"{c} {v / 1e9:+.3f}" for c, v in diffs)
                + f"; e4b dq's largest non-static groups: {_mc_top(md)}; Unsloth's: {_mc_top(mu)}"))
    return out


MEMCENSUS_TOP_ROWS = 20                 # the printed side-by-side rows (the receipts carry 40)
_MC_CLASSES = ("frozen_expert_weights", "expert_absmax", "other_frozen", "trainable_adapters", "adapter_grads", "optimizer_state", "other_buffers")


def _mc_static(st, c):
    """One class of a static census: `x.y` reads a sub-key (expert_absmax.fp32 -> expert_absmax_parts, other_frozen.bf16 -> other_frozen),
    other_frozen alone is the sum over its dtypes."""
    st = st or {}
    if c.startswith("expert_absmax."):
        return (st.get("expert_absmax_parts") or {}).get(c.split(".", 1)[1])
    if c.startswith("other_frozen."):
        return (st.get("other_frozen") or {}).get(c.split(".", 1)[1])
    if c == "other_frozen":
        return sum((st.get("other_frozen") or {}).values()) if isinstance(st.get("other_frozen"), dict) else None
    return st.get(c)


def memcensus_block(F):
    """Amendment 23's tables, the three arms side by side: the headline per arm, the static census by class (GB at the end of setup / of
    training), the live bytes at the peak by class, and the top live-at-peak groups."""
    R = F.get(MEMCENSUS_FAM)
    if not R:
        return []
    reads = [(k,) + _mc_reading(R, k) for k in MEMCENSUS_ARMS]
    heads = [MEMCENSUS_LABELS[k] for k in MEMCENSUS_ARMS]
    hdr = ["| | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]

    def row(label, fn):
        return f"| {label} | " + " | ".join((fn(m) if m else f"— ({w[:90]})") for _, m, w in reads) + " |"

    def pw(m):
        return m.get("peak_window") or {}
    lines = ["\n## Amendment 23: the memory census (Qwen3-30B-A3B, micro-batch 1 × accum 8, one draw per arm; GB = 1e9 bytes; no speed is read)", ""] + hdr
    lines.append(row("peak allocated / reserved (GB)", lambda m: f"{_gb(m.get('peak_allocated_bytes'))} / {_gb(m.get('peak_reserved_bytes'))}"))
    lines.append(row("attributed fraction at the peak", lambda m: f"{m['attributed_fraction']:.4f}" if m.get("attributed_fraction") is not None else "—"))
    lines.append(row("peak at (checkpoint / phase)", lambda m: f"{pw(m).get('checkpoint')} / {pw(m).get('peak_phase')}"))
    lines.append(row("window holds the run's peak / events / ring full", lambda m: f"{pw(m).get('peak_in_window')} / {pw(m).get('window_events')} / {pw(m).get('ring_full')}"))
    lines.append(row("snapshots / census s / torch", lambda m: f"{m.get('snapshots')} / {m.get('census_seconds')} / {m.get('torch')}"))
    parts = sorted({p for _, m, _ in reads if m for st in (m.get("static_after_setup"), m.get("static_after_train")) for p in ((st or {}).get("expert_absmax_parts") or {})})
    dts = sorted({p for _, m, _ in reads if m for st in (m.get("static_after_setup"), m.get("static_after_train")) for p in ((st or {}).get("other_frozen") or {})})
    keys = (["frozen_expert_weights", "expert_absmax"] + [f"expert_absmax.{p}" for p in parts] + ["other_frozen"] + [f"other_frozen.{p}" for p in dts]
            + ["trainable_adapters", "adapter_grads", "optimizer_state", "other_buffers", "other", "allocated_bytes"])
    lines += ["", "**Static census by class (GB: end of setup / end of training)**", ""] + hdr
    for c in keys:
        lines.append(row(c, lambda m, c=c: f"{_gb(_mc_static(m.get('static_after_setup'), c))} / {_gb(_mc_static(m.get('static_after_train'), c))}"))
    at = [_mc_at_peak(m) if m else {} for _, m, _ in reads]
    lines += ["", "**Live bytes at the peak by class (GB; transient = every live block that holds no static tensor)**", ""] + hdr
    for c in [c for c in _MC_CLASSES + ("transient",) if any(c in a for a in at)]:
        lines.append(row(c, lambda m, c=c: _gb(_mc_at_peak(m).get(c))))
    n = max([len(m.get("live_at_peak_top") or []) for _, m, _ in reads if m] or [0])
    if n:
        lines += ["", f"**Top live-at-peak groups (GB ×count; {min(n, MEMCENSUS_TOP_ROWS)} of the receipts' {n} shown)**", "",
                  "| rank | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
        for i in range(min(n, MEMCENSUS_TOP_ROWS)):
            cells = []
            for _, m, _ in reads:
                gs = (m or {}).get("live_at_peak_top") or []
                cells.append(f"{gs[i]['group']} {_gb(gs[i].get('bytes'))} ×{gs[i].get('count')}" if i < len(gs) else "")
            lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    return lines
def bmm_replay(d):
    """Amendment 24's replay files: {label: {"file": ..., "rows": {(dtype, blas): row}, "error": ...}} for each of BMM_FILES present in d."""
    out = {}
    for label, name in BMM_FILES.items():
        p = os.path.join(d or "", name)
        if not (d and os.path.exists(p)):
            continue
        try:
            js = json.load(open(p))
            out[label] = {"file": name, "rows": {(x.get("dtype"), x.get("blas")): x for x in js.get("results", [])}}
        except Exception as e:
            out[label] = {"file": name, "rows": {}, "error": f"{type(e).__name__}: {e}"}
    return out


def _bmm_row(rep, label, dtype, blas="default"):
    """(row, why): the replay row for one variant, or None with the reason it cannot be read."""
    if label not in rep:
        return None, f"{BMM_FILES[label]} not in this directory (the replay did not run in that environment)"
    x = rep[label]["rows"].get((dtype, blas))
    if not x:
        return None, f"{BMM_FILES[label]}: no {dtype}/{blas} row" + (f" ({rep[label]['error']})" if rep[label].get("error") else "")
    if x.get("error"):
        return None, f"{BMM_FILES[label]}: {dtype}/{blas} errored: {str(x['error'])[-160:]}"
    if list(x.get("cap") or []) != BMM_CAP:
        return None, f"{BMM_FILES[label]}: {dtype}/{blas} ran on {x.get('gpu')} (cap {x.get('cap')}), not the registered sm_120 card"
    want = "2.8." if label == "t28" else "2.12."
    if not str(x.get("torch") or "").startswith(want):
        return None, f"{BMM_FILES[label]}: {dtype}/{blas} ran torch {x.get('torch')}, not {want}*"
    return x, ""


def score_bmm_replay(d):
    """TC1-PREREG amendment 24, the replay (BMMBENCH-t28.json / BMMBENCH-t212.json): P44 -- torch 2.8's fp32 median host us per forward bmm at
    the recorded shapes >= BMM_P44_MIN_US and >= BMM_P44_MIN_X x bf16's; P45 -- its fp32 cold median >= BMM_P45_MIN_X x its repeated-shape
    median; P46 -- torch 2.12's fp32 cold median <= BMM_P46_MAX_X x torch 2.8's. Outside FALSIFIED; a missing file / row / card UNTESTED."""
    rep = bmm_replay(d)
    if not rep:
        return []
    out = []
    fp, why_fp = _bmm_row(rep, "t28", "fp32")
    bf, why_bf = _bmm_row(rep, "t28", "bf16")
    if fp and bf:
        a_, b_ = fp["recorded"]["fwd_host_us_median"], bf["recorded"]["fwd_host_us_median"]
        ok = a_ >= BMM_P44_MIN_US and a_ >= BMM_P44_MIN_X * b_
        out.append(("P44", "bmmbench", "HELD" if ok else "FALSIFIED",
                    f"torch {fp['torch']} on {fp['gpu']}: fp32 {a_:.1f} us vs bf16 {b_:.1f} us per forward bmm at the recorded shapes "
                    f"(x{a_ / b_:.2f}; registered >= {BMM_P44_MIN_US:.0f} us and >= x{BMM_P44_MIN_X:.0f})"))
    else:
        out.append(("P44", "bmmbench", "UNTESTED", "; ".join(w for w in (why_fp, why_bf) if w)))
    if fp:
        c_, g_ = fp["cold"]["fwd_host_us_median"], fp["recorded_again"]["fwd_host_us_median"]
        out.append(("P45", "bmmbench", "HELD" if c_ >= BMM_P45_MIN_X * g_ else "FALSIFIED",
                    f"torch {fp['torch']} fp32: cold {c_:.1f} us ({fp['cold']['calls']} calls, {fp['cold']['distinct_shapes']} new shapes) vs repeated "
                    f"{g_:.1f} us (x{c_ / g_:.2f}; registered >= x{BMM_P45_MIN_X:.0f}); fixed shape {fp['fixed']['fwd_host_us_median']:.1f} us, "
                    f"bucketed (multiple of {fp.get('bucket_multiple')}) {fp['bucket']['fwd_host_us_median']:.1f} / again {fp['bucket_again']['fwd_host_us_median']:.1f} us"))
    else:
        out.append(("P45", "bmmbench", "UNTESTED", why_fp))
    fn, why_fn = _bmm_row(rep, "t212", "fp32")
    if fp and fn:
        c8, c12 = fp["cold"]["fwd_host_us_median"], fn["cold"]["fwd_host_us_median"]
        out.append(("P46", "bmmbench", "HELD" if c12 <= BMM_P46_MAX_X * c8 else "FALSIFIED",
                    f"fp32 cold median torch {fn['torch']} {c12:.1f} us vs torch {fp['torch']} {c8:.1f} us (x{c12 / c8:.3f}; registered <= x{BMM_P46_MAX_X})"))
    else:
        out.append(("P46", "bmmbench", "UNTESTED", "; ".join(w for w in (why_fp, why_fn) if w)))
    return out


def bmm_replay_table(d):
    """Amendment 24, descriptive: every replay row's medians per regime (host us per forward bmm; backward host and device ms)."""
    rep = bmm_replay(d)
    if not rep:
        return []
    out = ["\n**Replay rows** (median host us per forward bmm; `bwd` = backward host / device ms at the recorded shapes)",
           "| env | torch | dtype | blas | recorded | again | cold | fixed | bucket | bucket again | bwd host / device |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label in BMM_FILES:
        for (dt, blas), x in (rep.get(label) or {}).get("rows", {}).items():
            if x.get("error"):
                out.append(f"| {label} | {x.get('torch', '?')} | {dt} | {blas} | ERROR: {str(x['error'])[-80:]} | | | | | | |")
                continue
            m = lambda k: f"{x[k]['fwd_host_us_median']:.1f}"
            out.append(f"| {label} | {x['torch']} | {dt} | {blas} | {m('recorded')} | {m('recorded_again')} | {m('cold')} | {m('fixed')} | {m('bucket')} | "
                       f"{m('bucket_again')} | {x['recorded']['bwd_host_ms_median']:.3f} / {x['recorded']['bwd_device_ms_median']:.3f} |")
    return out


def score_bmmab(F):
    """TC1-PREREG amendment 24, the training A/B on the qwen3bmmab box: P47 -- the matched arm's tv1 / tv0 s/step within BMM_P47_BAND, the
    median over two VALID draws a side with each side's draws within 5 %; P48 -- the shipped arm's tv1 / tv0 >= the matched arm's + BMM_P48_MIN_GAP;
    P49 -- on each arm |mean held-out at N, tv1 - tv0| <= BMM_HELDOUT_MAX. Outside FALSIFIED; a missing / non-VALID / unstable side UNTESTED."""
    R = F.get(BMMAB_FAM)
    if not R:
        return []
    out, ratios, p49 = [], {}, []
    for name, t in BMMAB_PAIRS:
        O, N = R["draws"].get(("e4b", f"{t}_tv0"), {}), R["draws"].get(("e4b", f"{t}_tv1"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {dd.get('verdict') or 'missing'}: {dd.get('why') or ''}".strip() for side, dd in (("tv0", O), ("tv1", N)))
            ratios[name] = (None, f"{name}: two stable VALID draws a side are registered -- {why}")
            p49.append((name, None, why))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        h0, h1 = O.get("heldout_list") or [], N.get("heldout_list") or []
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p49.append((name, dq, f"held-out at N tv0 {[round(v, 4) for v in h0 if v is not None]} tv1 {[round(v, 4) for v in h1 if v is not None]}"))
        ratios[name] = (ratio_, f"{name}: tv1 / tv0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios]; s/step tv0 "
                                f"{O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), tv1 {N['s_list'][0]:.3f} / "
                                f"{N['s_list'][1]:.3f} (within {100 * N['stability']:.1f}%); peak tv0 {f(O.get('peak'), 2)} / tv1 {f(N.get('peak'), 2)} GB")
    m, ev_m = ratios["matched"]
    lo, hi = BMM_P47_BAND
    out.append(("P47", BMMAB_FAM, "UNTESTED" if m is None else ("HELD" if lo <= m <= hi else "FALSIFIED"),
                ev_m + ("" if m is None else f" vs {[lo, hi]}")))
    sh, ev_s = ratios["shipped"]
    if m is None or sh is None:
        out.append(("P48", BMMAB_FAM, "UNTESTED", "; ".join(e for r_, e in (ratios["matched"], ratios["shipped"]) if r_ is None)))
    else:
        out.append(("P48", BMMAB_FAM, "HELD" if sh >= m + BMM_P48_MIN_GAP else "FALSIFIED",
                    f"shipped {sh:.3f} vs matched {m:.3f} (gap {sh - m:+.3f}; registered >= +{BMM_P48_MIN_GAP}); {ev_s}"))
    ev = "; ".join(f"{n}: " + (f"mean held-out tv1 - tv0 {dq:+.4f} (|.| <= {BMM_HELDOUT_MAX}); {e}" if dq is not None else e) for n, dq, e in p49)
    if any(dq is not None and abs(dq) > BMM_HELDOUT_MAX for _, dq, _ in p49):
        out.append(("P49", BMMAB_FAM, "FALSIFIED", ev))
    elif any(dq is None for _, dq, _ in p49):
        out.append(("P49", BMMAB_FAM, "UNTESTED", ev))
    else:
        out.append(("P49", BMMAB_FAM, "HELD", ev))
    return out


def score_samestack(F, fam=SAMESTACK_FAM):
    """TC1-PREREG amendment 25, on the qwen3samestack box: P50 -- the matched position Unsloth/e4b (both on venv-unsloth) within
    SAMESTACK_P50_BAND, both pairs two STABLE draws; P51 -- e4b's matched arm venv-unsloth / venv-e4b within SAMESTACK_P51_BAND, both sides
    two STABLE draws; P52 -- e4b's reference and Unsloth each read EQUIVALENT or INSIDE-DRAW-NOISE against e4b's fused arm, and e4b's
    parity PASSES. Outside FALSIFIED; a missing / non-VALID / unstable side UNTESTED. TC1c amendment 9 (fam=SAMESTACK_H100_FAM): the same
    two speed readings as P27 / P28 with its own bands, no matched-set prediction (no reference arm), and P29 -- every e4b receipt's
    route_ab names grouped_mm with GNF4_TRAIN_GEMM unset (the fused e4b arms that ran)."""
    pid_pos, band_pos, pid_env, band_env, pid_set, pid_route, cite = SAMESTACK_SPECS[fam]
    R = F.get(fam)
    if not R:
        return []
    out = []
    pz = R["positions"].get("unsloth") or {}
    lo, hi = band_pos
    if pz.get("quoted") and pz.get("e4b_draws") == 2 and pz.get("other_draws") == 2:
        v = pz["ratio"]
        out.append((pid_pos, fam, "HELD" if lo <= v <= hi else "FALSIFIED",
                    f"Unsloth/e4b on one stack {v:.3f} [{pz['ratio_min']:.3f}, {pz['ratio_max']:.3f} over {pz['n_cross']} cross-draw ratios] vs {[lo, hi]}; "
                    f"s/step e4b {pz['e4b_s']:.3f} (within {100 * (pz.get('e4b_stability') or 0):.1f}%), Unsloth {pz['other_s']:.3f} (within "
                    f"{100 * (pz.get('other_stability') or 0):.1f}%); peak e4b {f(pz.get('peak_e4b'), 2)} / Unsloth {f(pz.get('peak_other'), 2)} GB; "
                    f"held-out at N e4b {f(pz.get('heldout_e4b'), 4)} / Unsloth {f(pz.get('heldout_other'), 4)}"))
    else:
        out.append((pid_pos, fam, "UNTESTED", f"two stable VALID draws a side are registered -- {pz.get('why') or 'one side single-draw'}"))
    E, T = R["draws"].get(("e4b", "fused_attn4_m"), {}), R["draws"].get(("e4b", SAMESTACK_T28), {})
    lo, hi = band_env
    if E.get("usable") and T.get("usable") and E.get("draws") == 2 and T.get("draws") == 2:
        v = E["s"] / T["s"]
        cross = [a / b for a in E["s_list"] for b in T["s_list"]]
        out.append((pid_env, fam, "HELD" if lo <= v <= hi else "FALSIFIED",
                    f"e4b matched arm venv-unsloth / venv-e4b {v:.3f} [{min(cross):.3f}, {max(cross):.3f}] vs {[lo, hi]}; s/step venv-unsloth "
                    f"{E['s_list'][0]:.3f} / {E['s_list'][1]:.3f}, venv-e4b {T['s_list'][0]:.3f} / {T['s_list'][1]:.3f}; {cite}"))
    else:
        why = "; ".join(f"{n} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for n, d in (("venv-unsloth", E), ("venv-e4b", T)))
        out.append((pid_env, fam, "UNTESTED", f"two stable VALID draws a side are registered -- {why}"))
    if pid_route:
        want_route, route_ok = SAMESTACK_ROUTE[fam]
        ras = {x["tag"]: x["r"].get("route_ab") for x in R["rows"]       # the fused e4b arms that ran (a skipped reference is a stub)
               if x["fw"] == "e4b" and x["tag"].startswith("fused") and (x.get("r") or {}).get("status") == "ok"}
        bad = [f"{t}: {(ra or {}).get('gnf4_train_gemm') if ra else 'no route_ab record'} {json.dumps((ra or {}).get('stats'), sort_keys=True)}" + (f" (GNF4_TRAIN_GEMM={ra.get('gnf4_train_gemm_env')})" if ra and ra.get("gnf4_train_gemm_env") else "")
               for t, ra in sorted(ras.items()) if not (ra and route_ok(ra) and not ra.get("gnf4_train_gemm_env"))]
        ev = (f"{len(ras)} e4b receipts; route {want_route} with GNF4_TRAIN_GEMM unset on " + ("all" if not bad else f"{len(ras) - len(bad)}; not on {'; '.join(bad)}"))
        out.append((pid_route, fam, "UNTESTED" if not ras else ("HELD" if not bad else "FALSIFIED"), ev if ras else "no e4b receipts"))
    if not pid_set:
        return out
    eq, par = R.get("equivalence") or {}, (R.get("parity") or {}).get("verdict")
    ok_read = ("EQUIVALENT", "INSIDE-DRAW-NOISE")
    reads = {k: (eq.get(k) or {}).get("reading") for k in (("e4b", "reference_attn4_m"), ("unsloth", "ckpt_unsloth_m"))}
    ev = "; ".join(f"`{k[0]}/{k[1]}` {v or 'not read'}" for k, v in reads.items()) + f"; e4b parity {par or 'not read'}"
    if any(v in (None, "—", "N-A") for v in reads.values()) or par not in ("PASS", "FAIL"):
        out.append((pid_set, fam, "UNTESTED", ev))
    else:
        out.append((pid_set, fam, "HELD" if all(v in ok_read for v in reads.values()) and par == "PASS" else "FALSIFIED", ev))
    return out


def score_packed4k(F, fam=PACKED4K_FAM):
    """TC1-PREREG amendment 39, on the qwen3samestack4k box: P84 / P85 -- amendment 25's two speed readings (score_samestack with
    SAMESTACK_SPECS[fam]: no matched-set prediction, no route check) on the packed rows; P86 -- every e4b arm that ran completed resident:
    FALSIFIED if any e4b arm's status is OOM, else HELD if every e4b row that ran is VALID (and ran resident, offload off), else UNTESTED.
    An arm that did not run (NOT_RUN: the registered box skips e4b's reference) is not read. Amendment 40 (fam=PACKED4KCE_FAM): the same
    reading as P87 / P88 / P89, e4b with its chunked LM loss."""
    pid_fit = PACKED_FIT_ID[fam]
    R = F.get(fam)
    if not R:
        return []
    out = score_samestack(F, fam)
    ran = [x for x in R["rows"] if x["fw"] == "e4b" and x["status"] != "NOT_RUN"]
    ooms = [x for x in ran if x["status"] == "OOM"]
    offl = [x["tag"] for x in ran if x["verdict"] == "VALID" and (x.get("r") or {}).get("offload")]
    peaks = "; ".join(f"{x['tag']} {x['verdict']}" + (f" peak {f((x.get('r') or {}).get('peak_vram_gb'), 2)} GB" if (x.get("r") or {}).get("peak_vram_gb") is not None else "")
                      for x in ran)
    if ooms:
        out.append((pid_fit, fam, "FALSIFIED", f"e4b OOM on {', '.join(x['tag'] + ' (' + (x.get('reason') or '')[:80] + ')' for x in ooms)}; {len(ran)} e4b arm(s) ran: {peaks}"))
    elif ran and all(x["verdict"] == "VALID" for x in ran) and not offl:
        out.append((pid_fit, fam, "HELD", f"all {len(ran)} e4b arms that ran completed resident and VALID: {peaks}"))
    else:
        why = (f"VALID under offload: {offl}" if offl else "") or ("no e4b arm ran" if not ran else
                                                                     f"not every e4b arm that ran is VALID and none OOMed: {peaks}")
        out.append((pid_fit, fam, "UNTESTED", why))
    return out


def score_prebindab(F, fam=PREBIND_FAM):
    """TC1-PREREG amendment 26, on the qwen3prebindab box: P53 (shipped) and P54 (matched) -- pb1 / pb0 s/step within PREBIND_BANDS[pid],
    the median over two VALID draws a side with each side's draws within 5 %; P55 -- on each arm |mean held-out at N, pb1 - pb0| <=
    PREBIND_HELDOUT_MAX. Outside FALSIFIED; a missing / non-VALID / unstable side UNTESTED. Amendment 35 (fam=PREBIND37_FAM): the same
    reading of the qwen3prebind37 box as P66 / P67 / P68 with PREBIND37_BANDS."""
    pairs, bands, pheld = PREBIND_SPECS[fam]
    R = F.get(fam)
    if not R:
        return []
    out, p55 = [], []
    for pid, name, t in pairs:
        O, N = R["draws"].get(("e4b", f"{t}_pb0"), {}), R["draws"].get(("e4b", f"{t}_pb1"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("pb0", O), ("pb1", N)))
            out.append((pid, fam, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            p55.append((name, None, why))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        lo, hi = bands[pid]
        h0, h1 = O.get("heldout_list") or [], N.get("heldout_list") or []
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p55.append((name, dq, f"held-out at N pb0 {[round(v, 4) for v in h0 if v is not None]} pb1 {[round(v, 4) for v in h1 if v is not None]}"))
        r1 = next((x["r"] for x in R["rows"] if (x["fw"], x["tag"]) == ("e4b", f"{t}_pb1")), None) or {}
        pa = r1.get("prebind_ab") or {}
        counts = {sd: (pa.get(sd) or {}).get("stats") for sd in ("e4b", "gnf4")}
        out.append((pid, fam, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name}: pb1 / pb0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}; s/step pb0 "
                    f"{O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), pb1 {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} "
                    f"(within {100 * N['stability']:.1f}%); pb1 launch counts (process) {json.dumps(counts, sort_keys=True)}; triton {pa.get('triton')}"))
    ev = "; ".join(f"{n}: " + (f"mean held-out pb1 - pb0 {dq:+.4f} (|.| <= {PREBIND_HELDOUT_MAX}); {e}" if dq is not None else e) for n, dq, e in p55)
    if any(dq is not None and abs(dq) > PREBIND_HELDOUT_MAX for _, dq, _ in p55):
        out.append((pheld, fam, "FALSIFIED", ev))
    elif any(dq is None for _, dq, _ in p55):
        out.append((pheld, fam, "UNTESTED", ev))
    else:
        out.append((pheld, fam, "HELD", ev))
    return out


def score_dqab(F):
    """TC1-PREREG amendment 28, on the qwen3dqab / mixtraldqab tokens: P56 (Qwen3-30B-A3B) and P57 (Mixtral) -- dq1 / dq0 s/step within
    DQ_SPEED_BAND and the median peak falling by DQ_PEAK_DROP[fam], over two VALID draws a side with each side's draws within 5 %; P58 --
    on each family |mean held-out at N, dq1 - dq0| <= DQ_HELDOUT_MAX (one reading over both: FALSIFIED when either family is read outside,
    else UNTESTED while either is unread). Outside FALSIFIED; a missing / non-VALID / unstable side UNTESTED."""
    if not any(fam in F for fam in DQ_FAMS):
        return []
    out, p58 = [], []
    for fam in DQ_FAMS:
        pid, name = DQ_PREDS[fam]
        R = F.get(fam)
        if not R:
            out.append((pid, fam, "UNTESTED", f"{name}: no {fam} receipts in this directory"))
            p58.append((fam, None, "no receipts"))
            continue
        O, N = R["draws"].get(("e4b", f"{DQ_ARM}_dq0"), {}), R["draws"].get(("e4b", f"{DQ_ARM}_dq1"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("dq0", O), ("dq1", N)))
            out.append((pid, fam, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            p58.append((fam, None, why))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        drop = (O.get("peak") or 0) - (N.get("peak") or 0) if (O.get("peak") is not None and N.get("peak") is not None) else None
        lo, hi = DQ_SPEED_BAND
        plo, phi = DQ_PEAK_DROP[fam]
        h0, h1 = O.get("heldout_list") or [], N.get("heldout_list") or []
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p58.append((fam, dq, f"held-out at N dq0 {[round(v, 4) for v in h0 if v is not None]} dq1 {[round(v, 4) for v in h1 if v is not None]}"))
        ok = lo <= ratio_ <= hi and drop is not None and plo <= drop <= phi
        out.append((pid, fam, "HELD" if ok else "FALSIFIED",
                    f"{name}: dq1 / dq0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}; peak dq0 "
                    f"{f(O.get('peak'), 2)} / dq1 {f(N.get('peak'), 2)} GB, drop {f(drop, 3)} vs {[plo, phi]}; s/step dq0 {O['s_list'][0]:.3f} / "
                    f"{O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), dq1 {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} (within "
                    f"{100 * N['stability']:.1f}%)"))
    ev = "; ".join(f"{fam}: " + (f"mean held-out dq1 - dq0 {d:+.4f} (|.| <= {DQ_HELDOUT_MAX}); {e}" if d is not None else e) for fam, d, e in p58)
    if any(d is not None and abs(d) > DQ_HELDOUT_MAX for _, d, _ in p58):
        out.append(("P58", "dqab", "FALSIFIED", ev))
    elif any(d is None for _, d, _ in p58):
        out.append(("P58", "dqab", "UNTESTED", ev))
    else:
        out.append(("P58", "dqab", "HELD", ev))
    return out


def score_compactab(F, fam=COMPACT_FAM):
    """TC1-PREREG amendment 36, on the qwen3compactab box: P69 -- on the matched arm the median peak falls by COMPACT_PEAK_DROP (cd0 - cd1);
    P70 (matched) and P71 (shipped) -- cd1 / cd0 s/step within COMPACT_SPEED_BAND; each over two VALID draws a side with each side's draws
    within 5 %; P72 -- on each arm |mean held-out at N, cd1 - cd0| <= COMPACT_HELDOUT_MAX. Outside FALSIFIED; a missing / non-VALID /
    unstable side UNTESTED. Amendment 37 (fam=COMPACT2_FAM): the same reading as P73 (peak) / P74 / P75 (speed) / P76 with its
    own bands, from COMPACT_SPECS."""
    spec = COMPACT_SPECS[fam]
    p_peak, peak_band, pairs, speed_band, p_held = spec[:5]
    s0, s1 = spec[5] if len(spec) > 5 else ("cd0", "cd1")        # amendment 41: the chunked loss's sides are ce0 / ce1
    R = F.get(fam)
    if not R:
        return []
    out, p72, drop_ev = [], [], None
    for pid, name, t in pairs:
        O, N = R["draws"].get(("e4b", f"{t}_{s0}"), {}), R["draws"].get(("e4b", f"{t}_{s1}"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in ((s0, O), (s1, N)))
            out.append((pid, fam, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            p72.append((name, None, why))
            if name == "matched":
                drop_ev = ("UNTESTED", f"matched: two stable VALID draws a side are registered -- {why}")
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        lo, hi = speed_band
        h0, h1 = O.get("heldout_list") or [], N.get("heldout_list") or []
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p72.append((name, dq, f"held-out at N {s0} {[round(v, 4) for v in h0 if v is not None]} {s1} {[round(v, 4) for v in h1 if v is not None]}"))
        drop = (O["peak"] - N["peak"]) if (O.get("peak") is not None and N.get("peak") is not None) else None
        out.append((pid, fam, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name}: {s1} / {s0} {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}; s/step {s0} "
                    f"{O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), {s1} {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} "
                    f"(within {100 * N['stability']:.1f}%); peak {s0} {f(O.get('peak'), 2)} / {s1} {f(N.get('peak'), 2)} GB"))
        if name == "matched":
            plo, phi = peak_band
            ev = f"matched: peak {s0} {f(O.get('peak'), 3)} / {s1} {f(N.get('peak'), 3)} GB, drop {f(drop, 3)} vs {[plo, phi]}"
            drop_ev = ("UNTESTED", ev + " (no peak recorded)") if drop is None else ("HELD" if plo <= drop <= phi else "FALSIFIED", ev)
    out.insert(0, (p_peak, fam) + (drop_ev or ("UNTESTED", "matched: no pair")))
    ev = "; ".join(f"{n}: " + (f"mean held-out {s1} - {s0} {d:+.4f} (|.| <= {COMPACT_HELDOUT_MAX}); {e}" if d is not None else e) for n, d, e in p72)
    if any(d is not None and abs(d) > COMPACT_HELDOUT_MAX for _, d, _ in p72):
        out.append((p_held, fam, "FALSIFIED", ev))
    elif any(d is None for _, d, _ in p72):
        out.append((p_held, fam, "UNTESTED", ev))
    else:
        out.append((p_held, fam, "HELD", ev))
    return out


def score_tritonab(F):
    """TC1-PREREG amendment 32, on the qwen3tritonab box: P59 (matched) and P60 (shipped) -- tr1 / tr0 s/step within TRITON_BANDS[pid], the
    median over two VALID draws a side with each side's draws within 5 %; P61 -- on each arm |mean held-out at N, tr1 - tr0| <=
    TRITON_HELDOUT_MAX. Outside FALSIFIED; a missing / non-VALID / unstable side UNTESTED."""
    R = F.get(TRITON_FAM)
    if not R:
        return []
    out, p61 = [], []
    for pid, name, t in TRITON_PAIRS:
        O, N = R["draws"].get(("e4b", f"{t}_tr0"), {}), R["draws"].get(("e4b", f"{t}_tr1"), {})
        if not (O.get("usable") and N.get("usable") and O.get("draws") == 2 and N.get("draws") == 2):
            why = "; ".join(f"{side} {d.get('verdict') or 'missing'}: {d.get('why') or ''}".strip() for side, d in (("tr0", O), ("tr1", N)))
            out.append((pid, TRITON_FAM, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            p61.append((name, None, why))
            continue
        ratio_ = N["s"] / O["s"]
        cross = [n / o for n in N["s_list"] for o in O["s_list"]]
        lo, hi = TRITON_BANDS[pid]
        h0, h1 = O.get("heldout_list") or [], N.get("heldout_list") or []
        dq = (sum(h1) / len(h1) - sum(h0) / len(h0)) if (h0 and h1 and None not in h0 + h1) else None
        p61.append((name, dq, f"held-out at N tr0 {[round(v, 4) for v in h0 if v is not None]} tr1 {[round(v, 4) for v in h1 if v is not None]}"))
        out.append((pid, TRITON_FAM, "HELD" if lo <= ratio_ <= hi else "FALSIFIED",
                    f"{name}: tr1 / tr0 {ratio_:.3f} [{min(cross):.3f}, {max(cross):.3f} over 4 cross-draw ratios] vs {[lo, hi]}; s/step tr0 "
                    f"{O['s_list'][0]:.3f} / {O['s_list'][1]:.3f} (within {100 * O['stability']:.1f}%), tr1 {N['s_list'][0]:.3f} / {N['s_list'][1]:.3f} "
                    f"(within {100 * N['stability']:.1f}%); amendment 24's environment A/B (torch, transformers and triton together) read 0.882 on the matched arm"))
    ev = "; ".join(f"{n}: " + (f"mean held-out tr1 - tr0 {dq:+.4f} (|.| <= {TRITON_HELDOUT_MAX}); {e}" if dq is not None else e) for n, dq, e in p61)
    if any(dq is not None and abs(dq) > TRITON_HELDOUT_MAX for _, dq, _ in p61):
        out.append(("P61", TRITON_FAM, "FALSIFIED", ev))
    elif any(dq is None for _, dq, _ in p61):
        out.append(("P61", TRITON_FAM, "UNTESTED", ev))
    else:
        out.append(("P61", TRITON_FAM, "HELD", ev))
    return out

def loadgate_lines(d):
    """TC1 amendment 33: the box's LOADGATE lines (summary.txt), one per attempt -- the median host load1 over the arm's own run, the gate,
    and the attempts the gate voided (set aside to loadvoid/ and run again). Empty when the box ran without TC1_LOAD_GATE."""
    p = os.path.join(d or "", "summary.txt")
    if not (d and os.path.exists(p)):
        return []
    ls = [ln.rstrip("\n") for ln in open(p, errors="replace") if ln.startswith("LOADGATE ")]
    if not ls:
        return []
    void = sum(1 for ln in ls if " VOID " in ln)
    return [f"\n## Load gate (TC1-PREREG amendment 33): {void} draw(s) voided for host load and run again",
            "Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; "
            "the last attempt of each arm stands whatever its load.", ""] + [f"- `{ln[len('LOADGATE '):]}`" for ln in ls]


def score_envsplit(F):
    """TC1-PREREG amendment 34, on the qwen3envsplit box: P62 (e1 / e0), P63 (e2 / e1) and P64 (e2 / e0) -- each ratio of medians over two
    VALID draws a side (each side's draws within 5 %) within its band; P65 -- |mean held-out at N| e1 - e0 and e2 - e0 <= ENVSPLIT_HELDOUT_MAX.
    Outside FALSIFIED; a missing / non-VALID / unstable side UNTESTED."""
    R = F.get(ENVSPLIT_FAM)
    if not R:
        return []
    D = {sd: R["draws"].get(("e4b", f"fused_attn4_m_{sd}"), {}) for sd in ENVSPLIT_SIDES}
    ok = {sd: bool(d.get("usable") and d.get("draws") == 2) for sd, d in D.items()}
    out = []
    for pid, name, num, den, (lo, hi) in ENVSPLIT_PREDS:
        if not (ok[num] and ok[den]):
            why = "; ".join(f"{sd} {D[sd].get('verdict') or 'missing'}: {D[sd].get('why') or ''}".strip() for sd in (num, den) if not ok[sd])
            out.append((pid, ENVSPLIT_FAM, "UNTESTED", f"{name}: two stable VALID draws a side are registered -- {why}"))
            continue
        v = D[num]["s"] / D[den]["s"]
        cross = [a / b for a in D[num]["s_list"] for b in D[den]["s_list"]]
        out.append((pid, ENVSPLIT_FAM, "HELD" if lo <= v <= hi else "FALSIFIED",
                    f"{name}: {num} / {den} {v:.3f} [{min(cross):.3f}, {max(cross):.3f}] vs {[lo, hi]}; s/step {num} {D[num]['s_list'][0]:.3f} / "
                    f"{D[num]['s_list'][1]:.3f}, {den} {D[den]['s_list'][0]:.3f} / {D[den]['s_list'][1]:.3f}"))
    reads, missing = [], []
    for sd in ("e1", "e2"):
        h0, h1 = D["e0"].get("heldout_list") or [], D[sd].get("heldout_list") or []
        if ok["e0"] and ok[sd] and h0 and h1 and None not in h0 + h1:
            reads.append((sd, sum(h1) / len(h1) - sum(h0) / len(h0)))
        else:
            missing.append(sd)
    ev = "; ".join(f"{sd} - e0 {d:+.4f}" for sd, d in reads) + (f"; unread: {', '.join(missing)}" if missing else "") + f" (|.| <= {ENVSPLIT_HELDOUT_MAX})"
    if any(abs(d) > ENVSPLIT_HELDOUT_MAX for _, d in reads):
        out.append(("P65", ENVSPLIT_FAM, "FALSIFIED", ev))
    elif missing:
        out.append(("P65", ENVSPLIT_FAM, "UNTESTED", ev))
    else:
        out.append(("P65", ENVSPLIT_FAM, "HELD", ev))
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
    elif Q.get("anchor_micro_batch") not in (None, FIELD_MICRO_BATCH):
        out.append(("P4", "qwen3_5", "UNTESTED", f"e4b's anchor ran micro-batch {Q['anchor_micro_batch']}: P4's e4b leg reads the field recipe's step "
                    f"(tp4's {TP4_QWEN3_5_E4B_S_PER_STEP} s/step at micro-batch {FIELD_MICRO_BATCH}); TC2 amendment 8's box Q quotes its pair on the position line"))
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
    elif MX.get("anchor_offload") is False:
        out.append(("P5", "mixtral", "UNTESTED", "e4b's anchor ran RESIDENT: P5 is the offload pair's prediction (TC2 amendment 6 scores a resident box)"))
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
    if pos.get("recipe"):                                                   # TC2 amendment 8: a primary pair off the field recipe's micro-batch says so
        prefix = f"{prefix} ({pos['recipe']})"
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
                     f"fixture template {src.get('template')} seq {src.get('seq')}{' (packed rows, amendment 39)' if (src.get('tokens') or {}).get('pack') else ''} "
                     f"micro-batch {src.get('micro_batch')} × accum {src.get('accum')} lr {src.get('lr')} r {src.get('r')} α {src.get('alpha')} "
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
    if RMS_FAM in F:
        out += ["\n## Predictions P24 / P25 / P26 (TC1-PREREG amendment 15, #945: e4b's fused training RMSNorm vs the composite, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_rmsab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if REUSE_FAM in F:
        out += ["\n## Predictions P33 / P34 (TC1-PREREG amendment 20, #945: gnf4's per-pass host reuse off vs on, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_reuseab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if KEEP_FAM in F:
        out += ["\n## Predictions P35 / P36 / P37 (TC1-PREREG amendment 21, #945: whole-layer checkpointing vs keeping MoE activations, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_keepab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if any(fam in F for fam in DENSE_FAMS):
        out += ["\n## Predictions P38 / P39 / P40 (TC1-PREREG amendment 22: grouped-nf4-gemm's dense route vs its fused kernels, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_denseab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if MEMCENSUS_FAM in F:
        out += memcensus_block(F)
        out += ["\n## Predictions P41 / P42 / P43 (TC1-PREREG amendment 23: the memory census, e4b against Unsloth at micro-batch 1; scored mechanically from the receipts)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_memcensus(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if BMMAB_FAM in F or bmm_replay(d):
        out += ["\n## Predictions P44–P49 (TC1-PREREG amendment 24: the RTX 5090's fp32 bmm host cost, replay and venv-e4b vs venv-unsloth A/B; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_bmm_replay(d) + score_bmmab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
        out += bmm_replay_table(d)
    if SAMESTACK_FAM in F:
        out += ["\n## Predictions P50 / P51 / P52 (TC1-PREREG amendment 25: the matched position with both frameworks on one stack; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_samestack(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if SAMESTACK_HOST2_FAM in F:
        out += ["\n## Predictions P94 / P95 (TC1-PREREG amendment 42: the same-stack position on a second host, the current code; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_samestack(F, SAMESTACK_HOST2_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if SAMESTACK_MIXTRAL_FAM in F:
        out += ["\n## Predictions P29 / P30 / P31 (TC2-PREREG amendment 9: Mixtral's position with both frameworks on one stack; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_samestack(F, SAMESTACK_MIXTRAL_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if SAMESTACK_H100_FAM in F:
        out += ["\n## Predictions P27 / P28 / P29 (TC1C-PREREG amendment 9: the H100 position with both frameworks on one stack; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_samestack(F, SAMESTACK_H100_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if PACKED4KCE2_FAM in F:
        out += ["\n## Predictions P96 / P97 / P98 (TC1-PREREG amendment 43: the packed 4,096-token regime on one stack, e4b with its chunked LM loss, the LoRA loop a recorded route; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_packed4k(F, PACKED4KCE2_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
        loops = [(x["tag"], max([v for v in ((x.get("r") or {}).get("lora_loop_share") or []) if v is not None] or [0.0]))
                 for x in F[PACKED4KCE2_FAM]["rows"] if x["fw"] == "e4b" and (x.get("r") or {}).get("status") == "ok"]
        out.append("- the per-expert LoRA loop's share of a step's delta calls (max per arm, the recorded route): "
                   + "; ".join(f"`{t}` {m:.3f}" for t, m in loops))
    if PACKED4KCE_FAM in F:
        out += ["\n## Predictions P87 / P88 / P89 (TC1-PREREG amendment 40: the packed 4,096-token regime on one stack, e4b with its chunked LM loss; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_packed4k(F, PACKED4KCE_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if PACKED4K_FAM in F:
        out += ["\n## Predictions P84 / P85 / P86 (TC1-PREREG amendment 39: the packed 4,096-token regime with both frameworks on one stack; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_packed4k(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if PREBIND_FAM in F:
        out += ["\n## Predictions P53 / P54 / P55 (TC1-PREREG amendment 26: prebound Triton launches off vs on, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_prebindab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if PREBIND37_FAM in F:
        out += ["\n## Predictions P66 / P67 / P68 (TC1-PREREG amendment 35: prebound Triton launches off vs on under triton 3.7.1, venv-unsloth, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_prebindab(F, PREBIND37_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if COMPACT_FAM in F:
        out += ["\n## Predictions P69 / P70 / P71 / P72 (TC1-PREREG amendment 36: the compact padded LoRA delta off vs on, venv-unsloth, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_compactab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if CHUNKAB_FAM in F:
        out += ["\n## Predictions P90 / P91 / P92 / P93 (TC1-PREREG amendment 41: e4b's chunked LM loss off vs on at the field recipe; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_compactab(F, CHUNKAB_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if OMPAB_FAM in F:
        out += ["\n## Predictions P104 / P105 / P106 (TC1-PREREG amendment 45: OMP_NUM_THREADS at the physical cores vs the container's allotment; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_ompab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if CHUNKAUTO_FAM in F:
        out += ["\n## Predictions P99 / P100 / P101 / P102 / P103 (TC1-PREREG amendment 44: e4b's chunked LM loss off vs auto at the field recipe; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_chunkauto_gate(F) + score_compactab(F, CHUNKAUTO_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if any(fam in F for fam in DEC_FAMS) or (d and os.path.exists(os.path.join(d, DEC_GATE_FILE))):
        out += ["\n## Predictions P107 / P108 / P109 / P110 / P111 (TC1-PREREG amendment 46: grouped-nf4-gemm's decoded route vs its fused kernels, the sm_120 gate first, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_decgate(d) + score_decodedab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    for _cf, _ids in ((COMPACT3_FAM, "P77 / P78 / P79 / P82"), (MCOMPACT_FAM, "P80 / P81 / P83")):
        if _cf in F:
            out += [f"\n## Predictions {_ids} (TC1-PREREG amendment 38: the compact padded LoRA delta's default decision, {NAMES[_cf].split(' (')[0]}; scored mechanically)",
                    "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
            for pid, fam, v, ev in score_compactab(F, _cf):
                out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if COMPACT2_FAM in F:
        out += ["\n## Predictions P73 / P74 / P75 / P76 (TC1-PREREG amendment 37: the compact padded LoRA delta off vs on with its backward freeing early, venv-unsloth; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_compactab(F, COMPACT2_FAM):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if any(fam in F for fam in DQ_FAMS):
        out += ["\n## Predictions P56 / P57 / P58 (TC1-PREREG amendment 28: the expert absmax fp32 vs double-quantized, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_dqab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if TRITON_FAM in F:
        out += ["\n## Predictions P59 / P60 / P61 (TC1-PREREG amendment 32: venv-e4b with triton 3.4 vs 3.7.1, two stable draws a side; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_tritonab(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    if ENVSPLIT_FAM in F:
        out += ["\n## Predictions P62 / P63 / P64 / P65 (TC1-PREREG amendment 34: the matched arm in three environments, two stable draws each; scored mechanically)",
                "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
        for pid, fam, v, ev in score_envsplit(F):
            out.append(f"| {pid} | {fam} | **{v}** | {ev} |")

    out += loadgate_lines(d)
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


def _reuse_set(ship=((5.00, 5.05), (4.70, 4.72)), match=((5.30, 5.33), (5.05, 5.07)), flags=("0", "1"), hits=(1536, 768), lean="1"):
    """Amendment 20: e4b against itself on every current default -- each pair as (reuse0 draws, reuse1 draws); `flags` = the flag each
    side's record says was in force; `hits` = the reuse1 side's (upload_hits, plan_hits)."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for side, ss, flag in (("reuse0", old, flags[0]), ("reuse1", new, flags[1])):
            on = side == "reuse1"
            st = {"upload_hits": hits[0] if on else 0, "upload_misses": 2048, "plan_hits": hits[1] if on else 0, "plan_misses": 768 if on else 0}
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                R[("e4b", tag)] = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000), matched=matched,
                                           sync_ab={"e4b_grouping": "default", "gnf4_pinned_ring": "1", "e4b_has_group_by_expert": True,
                                                    "gnf4_has_ring": True, "ring_staged": 9216, "ring_waits": 0},
                                           lean_ab={"gnf4_lean_delta": lean, "gnf4_lean_delta_env": None, "gnf4_has_lean_delta": True,
                                                    "lora_path_calls": {"loop": 0, "padded": 9216, "grouped_mm": 0}},
                                           reuse_ab={"gnf4_host_reuse": flag, "gnf4_host_reuse_env": flag, "gnf4_has_host_reuse": True, "stats": st})
    for r in R.values():
        r["fam"] = REUSE_FAM
    return R


def _keep_set(ship=((5.00, 5.05), (4.40, 4.42)), match=((5.30, 5.33), (4.90, 4.92)), kept=None, compact="1", held_shift=0.0, peak1=29.5, lean="1"):
    """Amendment 21: e4b against itself -- each pair as (keep0 draws, keep1 draws); `kept` overrides the keep1 sides' layers_kept (default
    each arm's KEEP_N); `held_shift` moves the keep1 sides' held-out; `peak1` = the keep1 sides' peak GB."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for side, ss in (("keep0", old), ("keep1", new)):
            on = side == "keep1"
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                R[("e4b", tag)] = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000) + (held_shift if on else 0.0), matched=matched,
                                           sync_ab={"e4b_grouping": "default", "gnf4_pinned_ring": "1", "e4b_has_group_by_expert": True,
                                                    "gnf4_has_ring": True, "ring_staged": 9216, "ring_waits": 0},
                                           lean_ab={"gnf4_lean_delta": lean, "gnf4_lean_delta_env": None, "gnf4_has_lean_delta": True,
                                                    "lora_path_calls": {"loop": 0, "padded": 9216, "grouped_mm": 0}},
                                           keep_ab={"requested_env": str(KEEP_N[t]) if on else "0", "e4b_has_moe_keep": True,
                                                    "layers_kept": ((kept if kept is not None else KEEP_N[t]) if on else 0),
                                                    "gnf4_compact_delta": compact if on else "0", "gnf4_compact_delta_env": compact if on else None})
                R[("e4b", tag)]["peak_vram_gb"] = peak1 if on else 27.2
    for r in R.values():
        r["fam"] = KEEP_FAM
    return R


def _bmm_set(m=((5.00, 5.03), (4.20, 4.23)), sh=((3.90, 3.92), (3.86, 3.88)), torch_v=("2.8.0+cu128", "2.12.1+cu130"), held_shift=0.0, wrong_dtype=False):
    """Amendment 24: e4b against itself -- each arm as (tv0 draws, tv1 draws) s/step; `torch_v` = the torch each side's receipts record;
    `held_shift` moves the tv1 sides' held-out; `wrong_dtype` gives the matched tv1 side native adapters."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_m", m, True), ("fused_attn4_shipped", sh, False)):
        for side, ss, tv in (("tv0", old, torch_v[0]), ("tv1", new, torch_v[1])):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                r = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000) + (held_shift if side == "tv1" else 0.0), matched=matched,
                             env={"box_class": "RTX 5090", "gpu": "NVIDIA GeForce RTX 5090", "torch": tv})
                if wrong_dtype and matched and side == "tv1":
                    r["adapter_dtype"] = "native"
                R[("e4b", tag)] = r
    for r in R.values():
        r["fam"] = BMMAB_FAM
    return R


def _bmm_replay_files(d, t28=None, t212=None, cap=(12, 0)):
    """Amendment 24: write BMMBENCH-t28.json / -t212.json into d. Each arg = {(dtype, blas): (recorded, again, cold)} medians in us (None =
    no file). Rows not given read 50 us in every regime."""
    def row(label, torch_v, dt, blas, vals):
        rec, again, cold = vals
        reg = lambda v: {"fwd_host_us_median": v, "fwd_host_us_p90": v * 1.2, "bwd_host_ms_median": 0.4, "bwd_device_ms_median": 1.0, "calls": 96, "distinct_shapes": 60}
        return {"label": label, "dtype": dt, "blas": blas, "torch": torch_v, "cuda": "x", "gpu": "NVIDIA GeForce RTX 5090", "cap": list(cap), "n_calls": 96,
                "median_shape": [107, 138], "bucket_multiple": 32, "r": 16, "recorded": reg(rec), "recorded_again": reg(again), "cold": reg(cold),
                "fixed": reg(again), "bucket": reg(again), "bucket_again": reg(again)}
    for label, torch_v, spec in (("t28", "2.8.0+cu128", t28), ("t212", "2.12.1+cu130", t212)):
        if spec is None:
            continue
        rows = []
        for dt, blas in (("fp32", "default"), ("bf16", "default"), ("fp32", "cublaslt"), ("fp32tf32", "default")):
            rows.append(row(label, torch_v, dt, blas, spec.get((dt, blas), (50.0, 50.0, 50.0))))
        json.dump({"label": label, "results": rows}, open(os.path.join(d, BMM_FILES[label]), "w"))


def _samestack_set(e=(3.44, 3.46), t28=(3.90, 3.92), u=(7.90, 7.95), torch_v=("2.12.1+cu130", "2.8.0+cu128"), u_held=1.8100, drop=(), fam=None, routes=None):
    """Amendment 25: the matched set on one stack -- e4b's anchor pair and reference with `torch_v[0]`, its _t28 pair with `torch_v[1]`,
    Unsloth's pair; `u_held` = Unsloth's held-out at N (e4b 1.8000, reference 1.8050: band 0.015); `drop` removes arms. TC1c amendment 9:
    `fam` (default amendment 25's) and `routes`, a {tag: route} map written into every e4b receipt's route_ab (grouped_mm unless named)."""
    env = lambda tv: {"box_class": "RTX 5090", "gpu": "NVIDIA GeForce RTX 5090", "torch": tv}
    R = {}
    for i, sfx in enumerate(("", "_d2")):
        R[("e4b", "fused_attn4_m" + sfx)] = _receipt("e4b", "fused_attn4_m" + sfx, "fused", s=e[i], env=env(torch_v[0]))
        R[("e4b", SAMESTACK_T28 + sfx)] = _receipt("e4b", SAMESTACK_T28 + sfx, "fused", s=t28[i], env=env(torch_v[1]))
        R[("unsloth", "ckpt_unsloth_m" + sfx)] = _receipt("unsloth", "ckpt_unsloth_m" + sfx, "unsloth", s=u[i], heldout_n=u_held)
    R[("e4b", "reference_attn4_m")] = _receipt("e4b", "reference_attn4_m", "reference", s=9.00, heldout_n=1.8050, env=env(torch_v[0]))
    for k in drop:
        R.pop(k, None)
    for (fw, tag), r in R.items():
        r["fam"] = fam or SAMESTACK_FAM
        if routes is not None and fw == "e4b":
            r["route_ab"] = {"gnf4_train_gemm": routes.get(tag, "grouped_mm"), "gnf4_train_gemm_env": None, "gnf4_has_route": True,
                             "stats": {"fwd": 16896, "dgrad": 7680}}
    return R

def _packed4k_set(e=(14.0, 14.2), t28=(15.6, 15.8), u=(19.6, 19.8), torch_v=("2.12.1+cu130", "2.8.0+cu128"), oom=(), over=None, fam=None, chunked=None,
                  loop_share=None):
    """Amendment 39: amendment 25's set on the packed rows -- every receipt at seq 4096, micro-batch 1 x accum 4, N 30, a packed tokens file,
    16,384 real tokens and none padded on every step, resident; e4b's reference a NOT_RUN stub (the registered box skips it). `oom` names
    tags written as OOM stubs (e4b or Unsloth); `over` = {tag: {field: value}} overrides on a receipt."""
    R = {}
    for i, sfx in enumerate(("", "_d2")):
        for fw, tag, arm, ss, tv in (("e4b", "fused_attn4_m" + sfx, "fused", e[i], torch_v[0]), ("e4b", SAMESTACK_T28 + sfx, "fused", t28[i], torch_v[1]),
                                     ("unsloth", "ckpt_unsloth_m" + sfx, "unsloth", u[i], None)):
            R[(fw, tag)] = _receipt(fw, tag, arm, steps=30, s=ss, heldout_n=1.8100 if fw == "unsloth" else 1.8000, seq=4096, micro_batch=1, accum=4, offload=False,
                                    tokens={"sha256": "p" * 64, "pack": True}, tokens_per_step=[16384] * 30, tokens_padded_per_step=[0] * 30,
                                    arm_facts={"free_outputs": True},
                                    **({"env": {"box_class": "RTX 5090", "gpu": "NVIDIA GeForce RTX 5090", "torch": tv}} if tv else {}))
    R[("e4b", "reference_attn4_m")] = _stub("e4b", "reference_attn4_m", "reference", "not_run", "skipped by TC1_SKIP")
    for (fw, tag) in list(R):
        if tag in oom:
            R[(fw, tag)] = {**_stub(fw, tag, R[(fw, tag)]["arm"], "oom", "OOM at step 1: CUDA out of memory. Tried to allocate 2.32 GiB"), "steps": 30}
        R[(fw, tag)]["fam"] = fam or PACKED4K_FAM
        if loop_share is not None and fw == "e4b" and R[(fw, tag)].get("status") == "ok":   # amendment 43: the loop on every step at this share
            R[(fw, tag)]["lora_loop_share"] = [loop_share] * 30
            R[(fw, tag)]["lora_path_loop_steps"] = list(range(1, 31)) if loop_share else []
        if fam in CHUNKED_FAMS and fw == "e4b" and R[(fw, tag)].get("status") == "ok":   # amendments 40 / 43: the chunked loss on every e4b arm
            R[(fw, tag)]["chunked_lm_loss"] = dict(chunked) if chunked is not None else {
                "env": "1", "e4b_has_chunked_lm_loss": True, "chunked_calls": 120, "stock_calls": 16, "runtime_refusals": 0, "refused": {}}
        R[(fw, tag)].update((over or {}).get(tag, {}))
    return R


def _prebind_set(ship=((3.10, 3.12), (2.92, 2.94)), match=((3.90, 3.92), (3.72, 3.74)), held_shift=0.0, pb1_counts=(4000, 9000), pb0_counts=(0, 0),
                 requested=None, record=True, fam=PREBIND_FAM, triton=None, torch=None):
    """Amendment 26: e4b against itself -- each pair as (pb0 draws, pb1 draws) s/step; `pb1_counts` / `pb0_counts` = (e4b, gnf4) prebound
    launches each side counted; `requested` overrides the pb1 side's (e4b, gnf4) requested flags; `record=False` drops prebind_ab.
    Amendment 35 (fam=PREBIND37_FAM): `triton` / `torch` default to 3.7.1 / 2.12.1+cu130 there (3.4.0 / unset for amendment 26)."""
    triton = triton or ("3.7.1" if fam == PREBIND37_FAM else "3.4.0")
    torch = torch or ("2.12.1+cu130" if fam == PREBIND37_FAM else None)
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for side, ss in (("pb0", old), ("pb1", new)):
            on = side == "pb1"
            cnt = pb1_counts if on else pb0_counts
            req = (requested if (on and requested is not None) else (on, on))
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                r = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000) + (held_shift if on else 0.0), matched=matched)
                if record:
                    r["prebind_ab"] = {"e4b_env": "1" if on else "0", "gnf4_env": "1" if on else "0", "triton": triton,
                                       "e4b": {"has": True, "requested": req[0], "stats": {"prebound": cnt[0], "triton": 12}},
                                       "gnf4": {"has": True, "requested": req[1], "stats": {"prebound": cnt[1], "triton": 30}}}
                if torch:
                    r["env"]["torch"] = torch
                R[("e4b", tag)] = r
    for r in R.values():
        r["fam"] = fam
    return R

def _compact_set(match=((3.50, 3.52), (3.48, 3.50)), ship=((3.00, 3.02), (2.99, 3.01)), peaks=(27.50, 26.80), held_shift=0.0, flags=("0", "1"),
                 kept=0, padded=49152, torch="2.12.1+cu130", record=True, fam=None):
    """Amendment 36: e4b against itself -- each pair as (cd0 draws, cd1 draws) s/step; `peaks` = the matched arm's (cd0, cd1) GB (the shipped
    arm's sit 2 GB lower); `flags` = each side's resolved gnf4_compact_delta; `kept` / `padded` / `torch` as every receipt records them;
    `held_shift` moves the cd1 sides' held-out; `record=False` drops keep_ab."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for i_side, (side, ss) in enumerate((("cd0", old), ("cd1", new))):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                r = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000) + (held_shift if side == "cd1" else 0.0), matched=matched)
                r["peak_vram_gb"] = peaks[i_side] - (0.0 if matched else 2.0)
                r["env"]["torch"] = torch
                r["lean_ab"] = {"gnf4_lean_delta": "1", "gnf4_lean_delta_env": None, "gnf4_has_lean_delta": True,
                                "lora_path_calls": {"loop": 0, "padded": padded, "grouped_mm": 0}}
                if record:
                    r["keep_ab"] = {"requested_env": None, "e4b_has_moe_keep": True, "layers_kept": kept, "gnf4_compact_delta": flags[i_side],
                                    "gnf4_compact_delta_env": flags[i_side]}
                r["fam"] = fam or COMPACT_FAM
                R[("e4b", tag)] = r
    return R

def _chunkab_set(match=((3.50, 3.52), (3.51, 3.53)), ship=((3.00, 3.02), (3.01, 3.03)), peaks=(27.50, 27.30), held_shift=0.0, ce1_calls=240,
                 ce0_calls=0, refusals=0, torch="2.12.1+cu130", record=True):
    """Amendment 41: e4b against itself -- each pair as (ce0 draws, ce1 draws) s/step; `peaks` = the matched arm's (ce0, ce1) GB (the shipped
    arm's sit 2 GB lower); `ce1_calls` / `ce0_calls` = chunked training forwards each side recorded; `refusals` = ce1's run-time fallbacks."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for i_side, (side, ss) in enumerate((("ce0", old), ("ce1", new))):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                r = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000) + (held_shift if side == "ce1" else 0.0), matched=matched)
                r["peak_vram_gb"] = peaks[i_side] - (0.0 if matched else 2.0)
                r["env"]["torch"] = torch
                if record:
                    on = side == "ce1"
                    r["chunked_lm_loss"] = {"env": "1" if on else "0", "e4b_has_chunked_lm_loss": True, "chunked_calls": ce1_calls if on else ce0_calls,
                                            "stock_calls": 16, "runtime_refusals": refusals if on else 0, "refused": {}}
                r["fam"] = CHUNKAB_FAM
                R[("e4b", tag)] = r
    return R

def _chunkauto_set(match=((3.50, 3.52), (3.51, 3.53)), ship=((3.00, 3.02), (3.01, 3.03)), peaks=(27.50, 27.50), held_shift=0.0, ca1_chunked=0,
                   ca1_small=240, ca0_patched=0, env="auto", torch="2.12.1+cu130"):
    """Amendment 44: e4b against itself -- each pair as (ca0 draws, ca1 draws) s/step; `peaks` = the matched arm's (ca0, ca1) GB (the
    shipped arm's sit 2 GB lower); `ca1_chunked` / `ca1_small` = the auto side's chunked / gated training forwards; `ca0_patched` = whether
    the default side's model was patched; `env` = the auto side's E4B_CHUNKED_LM_LOSS."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for i_side, (side, ss) in enumerate((("ca0", old), ("ca1", new))):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                r = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000) + (held_shift if side == "ca1" else 0.0), matched=matched)
                r["peak_vram_gb"] = peaks[i_side] - (0.0 if matched else 2.0)
                r["env"]["torch"] = torch
                on = side == "ca1"
                r["chunked_lm_loss"] = {"env": env if on else "0", "e4b_has_chunked_lm_loss": True, "chunked_calls": ca1_chunked if on else 0,
                                        "stock_calls": 16 if on else 0, "small_calls": ca1_small if on else 0, "patched": 1 if on else ca0_patched,
                                        "runtime_refusals": 0, "refused": {}}
                r["fam"] = CHUNKAUTO_FAM
                R[("e4b", tag)] = r
    return R

def _ompab_set(e=((3.50, 3.52), (3.20, 3.22)), u=((7.90, 7.95), (7.88, 7.93)), omp=(128, 32), held_shift=0.0, torch="2.12.1+cu130", torch_threads=None):
    """Amendment 45: e4b's and Unsloth's matched arms, each as (om0 draws, om1 draws) s/step; `omp` = the (om0, om1) OpenMP thread counts every
    receipt records; `torch_threads` overrides torch's recorded intra-op count (default: the OMP count)."""
    R = {}
    for fw, tag0, arm, pairs, held in (("e4b", "fused_attn4_m", "fused", e, 1.8000), ("unsloth", "ckpt_unsloth_m", "unsloth", u, 1.8100)):
        for i_side, side in enumerate(("om0", "om1")):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{tag0}_{side}{sfx}"
                r = _receipt(fw, tag, arm, s=pairs[i_side][i], heldout_n=held + (held_shift if side == "om1" else 0.0))
                r.setdefault("env", {})["torch"] = torch
                r["arm_facts"] = dict(r.get("arm_facts") or {}, omp_num_threads=str(omp[i_side]),
                                      torch_num_threads=omp[i_side] if torch_threads is None else torch_threads)
                r["fam"] = OMPAB_FAM
                R[(fw, tag)] = r
    return R

def _dense_set(fam, d0=None, d1=None, routes=("fused", "dense"), dense_fwd=(0, 1536), dense_dgrad=(0, 1536), absmax=(True, True), held_shift=0.0, drop_route=False):
    """Amendment 22: e4b against itself on one family's matched arm -- (dense0 draws, dense1 draws) s/step; `routes` = the route each side's
    record resolved; `dense_fwd` / `dense_dgrad` = each side's process counts (None drops the counter, as grouped-nf4-gemm before #459);
    `absmax` = each side's absmax_dq; `held_shift` moves the dense1 side's held-out at N; `drop_route` removes the route_ab record."""
    d0 = d0 or ((5.50, 5.52) if fam == MDENSE_FAM else (5.30, 5.33))
    d1 = d1 or ((4.10, 4.12) if fam == MDENSE_FAM else (4.95, 4.97))
    R = {}
    for i_side, (side, ss) in enumerate((("dense0", d0), ("dense1", d1))):
        for i, sfx in enumerate(("", "_d2")):
            tag = f"{DENSE_ARM}_{side}{sfx}"
            held = 1.8000 + (held_shift if side == "dense1" else 0.0)
            r = (_tc2_receipt("mixtral", "e4b", tag, "fused", s=ss[i], heldout_n=held) if fam == MDENSE_FAM
                 else _receipt("e4b", tag, "fused", s=ss[i], heldout_n=held))
            stats = {"fwd": 0, "dgrad": 0}
            for k, v in (("dense_fwd", dense_fwd[i_side]), ("dense_dgrad", dense_dgrad[i_side])):
                if v is not None:
                    stats[k] = v
            r["route_ab"] = None if drop_route else {"gnf4_train_gemm": routes[i_side], "gnf4_train_gemm_env": routes[i_side], "gnf4_has_route": True, "stats": stats}
            r["absmax_dq"] = absmax[i_side]
            r["fam"] = fam
            R[("e4b", tag)] = r
    return R


def _decoded_set(fam, d0=None, d1=None, routes=("fused", "decoded"), dec_fwd=(0, 1536), dec_dgrad=(0, 1536), peaks=None, held_shift=0.0,
                 drop_route=False, torch="2.8.0+cu128", drop_counter=False):
    """Amendment 46: e4b against itself on one family's matched arm -- (dec0 draws, dec1 draws) s/step; `routes` = the route each side's
    record resolved; `dec_fwd` / `dec_dgrad` = each side's decoded counts; `peaks` = (dec0, dec1) GB; `held_shift` moves the dec1 side's
    held-out at N; `drop_route` removes the route_ab record; `drop_counter` removes the dec0 side's decoded counters (grouped-nf4-gemm before
    #487); `torch` = every arm's env.torch."""
    d0 = d0 or ((2.00, 2.02) if fam == ODEC_FAM else (5.30, 5.33))
    d1 = d1 or ((1.70, 1.72) if fam == ODEC_FAM else (5.60, 5.62))
    peaks = peaks or ((12.00, 12.20) if fam == ODEC_FAM else (27.50, 27.70))
    R = {}
    for i_side, (side, ss) in enumerate((("dec0", d0), ("dec1", d1))):
        for i, sfx in enumerate(("", "_d2")):
            tag = f"{DEC_ARM}_{side}{sfx}"
            held = 1.8000 + (held_shift if side == "dec1" else 0.0)
            r = (_tc2_receipt("olmoe", "e4b", tag, "fused", s=ss[i], heldout_n=held) if fam == ODEC_FAM
                 else _receipt("e4b", tag, "fused", s=ss[i], heldout_n=held))
            r["peak_vram_gb"] = peaks[i_side]
            r["env"]["torch"] = torch
            stats = {"fwd": 0, "dgrad": 0, "dense_fwd": 0, "dense_dgrad": 0}
            if not (drop_counter and side == "dec0"):
                stats.update(decoded_fwd=dec_fwd[i_side], decoded_dgrad=dec_dgrad[i_side])
            r["route_ab"] = None if drop_route else {"gnf4_train_gemm": routes[i_side], "gnf4_train_gemm_env": routes[i_side], "gnf4_has_route": True,
                                                     "stats": stats}
            r["fam"] = fam
            R[("e4b", tag)] = r
    return R


def _dq_set(fam, d0=None, d1=None, peaks=None, flags=(False, True), modules=None, held_shift=0.0):
    """Amendment 28: e4b against itself on one family's matched arm -- (dq0 draws, dq1 draws) s/step; `peaks` = (dq0, dq1) GB; `flags` =
    each side's absmax_dq; `modules` overrides the dq1 side's absmax_dq_modules (default the family's n_layers); `held_shift` moves dq1."""
    mix = fam == MDQ_FAM
    d0 = d0 or ((3.57, 3.58) if mix else (3.90, 3.92))
    d1 = d1 or ((3.62, 3.63) if mix else (3.95, 3.97))
    peaks = peaks or ((31.07, 28.97) if mix else (27.16, 25.81))
    R = {}
    for i_side, (side, ss) in enumerate((("dq0", d0), ("dq1", d1))):
        for i, sfx in enumerate(("", "_d2")):
            tag = f"{DQ_ARM}_{side}{sfx}"
            held = 1.8000 + (held_shift if side == "dq1" else 0.0)
            r = (_tc2_receipt("mixtral", "e4b", tag, "fused", s=ss[i], heldout_n=held) if mix
                 else _receipt("e4b", tag, "fused", s=ss[i], heldout_n=held))
            r["absmax_dq"] = flags[i_side]
            if flags[i_side]:
                r["absmax_dq_modules"] = N_LAYERS[fam] if modules is None else modules
            r["peak_vram_gb"] = peaks[i_side]
            r["fam"] = fam
            R[("e4b", tag)] = r
    return R

def _triton_set(match=((3.90, 3.92), (3.45, 3.47)), ship=((3.05, 3.07), (2.85, 2.87)), triton_v=("3.4.0", "3.7.1"), held_shift=0.0, prebind=(False, False)):
    """Amendment 32: e4b against itself -- each pair as (tr0 draws, tr1 draws) s/step; `triton_v` = the triton each side's receipts record;
    `prebind` = the (e4b, gnf4) requested flags on every arm; `held_shift` moves the tr1 sides' held-out."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_m", match, True), ("fused_attn4_shipped", ship, False)):
        for side, ss, tv in (("tr0", old, triton_v[0]), ("tr1", new, triton_v[1])):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                r = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000) + (held_shift if side == "tr1" else 0.0), matched=matched,
                             env={"box_class": "RTX 5090", "gpu": "NVIDIA GeForce RTX 5090", "torch": "2.8.0+cu128", "triton": tv})
                r["prebind_ab"] = {"e4b_env": "0", "gnf4_env": "0", "triton": tv, "e4b": {"has": True, "requested": prebind[0], "stats": {"prebound": 0}},
                                   "gnf4": {"has": True, "requested": prebind[1], "stats": {"prebound": 0}}}
                R[("e4b", tag)] = r
    for r in R.values():
        r["fam"] = TRITON_FAM
    return R

def _envsplit_set(e0=(3.90, 3.92), e1=(3.80, 3.82), e2=(3.44, 3.46), envs=None, held=(0.0, 0.0), prebind=False):
    """Amendment 34: the matched arm's draws per side; `envs` overrides the (torch, transformers) each side's receipts record; `held` =
    the held-out shift of e1 and e2 against e0; `prebind` = the requested flags on every arm."""
    envs = envs or {"e0": ("2.8.0+cu128", "5.18.0"), "e1": ("2.8.0+cu128", "5.5.0"), "e2": ("2.12.1+cu130", "5.5.0")}
    R = {}
    for side, ss, h in (("e0", e0, 0.0), ("e1", e1, held[0]), ("e2", e2, held[1])):
        for i, sfx in enumerate(("", "_d2")):
            tag = f"fused_attn4_m_{side}{sfx}"
            r = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=1.8000 + h,
                         env={"box_class": "RTX 5090", "gpu": "NVIDIA GeForce RTX 5090", "torch": envs[side][0], "transformers": envs[side][1]})
            r["prebind_ab"] = {"e4b": {"has": True, "requested": prebind}, "gnf4": {"has": True, "requested": prebind}}
            r["fam"] = ENVSPLIT_FAM
            R[("e4b", tag)] = r
    return R

QWEN3_EXPERT_PARAMS = 48 * 128 * (1536 * 2048 + 2048 * 768)      # 28,991,029,248: Qwen3-30B-A3B's logical expert weights
QWEN3_ABSMAX_FP32 = QWEN3_EXPERT_PARAMS // 64 * 4                 # 1,811,939,328 B
QWEN3_ABSMAX_DQ = QWEN3_EXPERT_PARAMS // 64 + QWEN3_EXPERT_PARAMS // 64 // 256 * 4 + 96 * (4 + 1024)   # q + scales + offsets + codes: 460,161,408 B


def _mc_census(peak, absmax, af=0.95, expert_params=QWEN3_EXPERT_PARAMS, in_window=True, transient=6.0e9, other_bf16=1_900_000_000, error=None):
    """Amendment 23: a hand-built mem_census as tc1_arm.py writes it (bytes) -- `peak` = peak_allocated_bytes, `absmax` = the static expert
    absmax, `transient` = the live bytes at the peak that hold no static tensor; `error` = an errored census."""
    base = {"torch": "2.8.0+cu128", "max_entries": 1000000, "stacks": "python", "context": "alloc"}
    if error:
        return {"error": error, **base}
    parts = {"fp32": absmax} if absmax > 1e9 else {"nested_q": absmax}
    st = {"frozen_expert_weights": QWEN3_EXPERT_PARAMS // 2, "expert_absmax": absmax, "expert_absmax_parts": parts,
          "other_frozen": {"bf16": other_bf16, "nf4_linear4bit": 300_000_000}, "trainable_adapters": 2_570_059_776, "trainable_by_dtype": {"fp32": 2_570_059_776},
          "adapter_grads": 0, "optimizer_state": 0, "other_buffers": 1_000_000, "expert_params": expert_params, "expert_stacks": 96}
    at = {"frozen_expert_weights": QWEN3_EXPERT_PARAMS // 2, "expert_absmax": absmax, "other_frozen.bf16": other_bf16, "other_frozen.nf4_linear4bit": 300_000_000,
          "trainable_adapters": 2_570_059_776, "adapter_grads": 2_570_059_776, "optimizer_state": 1_285_029_888}
    live = sum(at.values()) + int(transient)
    top = sorted([{"group": "static:" + k, "bytes": v, "count": 96} for k, v in at.items()]
                 + [{"group": "site:transformers/loss/loss_utils.py:60 fixed_cross_entropy", "bytes": int(transient * 0.6), "count": 2},
                    {"group": "unattributed:backward, no Python frame (an autograd C++ op or allocator-internal)", "bytes": int(transient * 0.4), "count": 7}],
                 key=lambda g: -g["bytes"])
    return {"peak_allocated_bytes": int(peak), "peak_reserved_bytes": int(peak) + 500_000_000, "static_after_setup": st,
            "static_after_train": dict(st, optimizer_state=1_285_029_888), "live_at_peak_top": top, "attributed_fraction": af,
            "peak_window": {"checkpoint": "s2.mb5", "peak_bytes": live, "peak_phase": "s2.mb5.backward", "peak_in_window": in_window,
                            "max_allocated_at_checkpoint": int(peak), "allocated_at_checkpoint": int(peak) - 3_000_000_000, "window_events": 412000, "ring_full": False,
                            "live_bytes": live, "live_blocks": 5000, "attributed_bytes": int(live * af), "inconsistent_events": 0, "n_groups": len(top), "static_at_peak": at},
            "snapshots": 6, "snapshot_cap_hits": 0, "census_seconds": 14.2, "trace": "recorded", **base}


def _memcensus_set(e4b_peak=26.06e9, dq_peak=24.71e9, un_peak=23.50e9, absmax=(QWEN3_ABSMAX_FP32, QWEN3_ABSMAX_DQ), af=(0.95, 0.94, 0.92), in_window=(True, True, True)):
    """Amendment 23's three arms at micro-batch 1 x accum 8, each VALID with its census: e4b fp32 absmax, e4b dq absmax, Unsloth (whose
    transient at the peak is 1 GB smaller than e4b's, and whose absmax is the nested form)."""
    R = {}
    ub = {"unsloth_grouped_mm": 48 * 8, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": 48 * 8}
    for i, (fw, tag) in enumerate(MEMCENSUS_ARMS):
        if fw == "e4b":
            r = _receipt("e4b", tag, "fused", s=6.0, accum=8, micro_batch=1, kernel_calls_per_step_min=2 * 48 * 8, absmax_dq=tag.endswith("_dq"),
                         mem_census=_mc_census(e4b_peak if i == 0 else dq_peak, absmax[i], af=af[i], in_window=in_window[i]))
        else:
            r = _receipt("unsloth", tag, "unsloth", s=4.0, heldout_n=1.8100, accum=8, micro_batch=1, experts_forward_calls_per_step_min=48 * 8,
                         unsloth_backend_calls_per_step_min=ub, unsloth_backend_calls_per_step_max=ub, unsloth_grouped_mm_calls_per_step_min=GMM_FACTOR * 48 * 8,
                         unsloth_grouped_mm_calls_per_step_max=GMM_FACTOR * 48 * 8, mem_census=_mc_census(un_peak, QWEN3_ABSMAX_DQ, af=af[2], in_window=in_window[2], transient=5.0e9))
        r["fam"] = MEMCENSUS_FAM
        r["peak_vram_gb"] = round(r["mem_census"]["peak_allocated_bytes"] / 1e9, 3)
        R[(fw, tag)] = r
    return R


def _rms_set(ship=((5.00, 5.05), (4.60, 4.62)), match=((5.30, 5.33), (5.00, 5.02)), held_shift=0.0, patched=192, calls=1536):
    """Amendment 15: e4b against itself on the trimmed delta and the post-#945 sync path -- each pair as (rms0 draws, rms1 draws);
    `held_shift` moves the rms1 side's held-out; `patched` / `calls` = the rms1 side's fused-RMSNorm record."""
    R = {}
    for t, (old, new), matched in (("fused_attn4_shipped", ship, False), ("fused_attn4_m", match, True)):
        for side, ss in (("rms0", old), ("rms1", new)):
            for i, sfx in enumerate(("", "_d2")):
                tag = f"{t}_{side}{sfx}"
                on = side == "rms1"
                R[("e4b", tag)] = _receipt("e4b", tag, "fused", s=ss[i], heldout_n=(1.7800 if not matched else 1.8000) + (held_shift if on else 0.0), matched=matched,
                                           sync_ab={"e4b_grouping": "default", "gnf4_pinned_ring": "1", "e4b_has_group_by_expert": True,
                                                    "gnf4_has_ring": True, "ring_staged": 9216, "ring_waits": 0},
                                           lean_ab={"gnf4_lean_delta": "1", "gnf4_lean_delta_env": None, "gnf4_has_lean_delta": True,
                                                    "lora_path_calls": {"loop": 0, "padded": 9216, "grouped_mm": 0}},
                                           rms_ab={"requested_env": "1" if on else "0", "e4b_has_fused_rmsnorm": True,
                                                   "patched": patched if on else 0, "calls": calls if on else 0})
    for r in R.values():
        r["fam"] = RMS_FAM
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
    R = _tc2_set("mixtral")                                  # TC2 amendment 6: every e4b arm resident
    for key in [k for k in R if k[0] == "e4b" and is_ok(R[k])]:
        R[key]["offload"] = False
    M3 = trun("mixtral", R)
    assert M3["anchor_offload"] is False and M3["footprint"] is None, M3["footprint"]
    assert not family_block(M3)[1].startswith("- **FOOTPRINT") and M3["positions"]["unsloth"]["quoted"]
    p5 = TP({**T_ALL, "mixtral": M3})["P5"]
    assert p5 == "UNTESTED", p5
    print("FAILING-CASE TC2-resident (reducer): e4b resident on mixtral -> no footprint line, P5", p5)
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
    # 60. amendment 15 (#945): the RMSNorm composite vs e4b's fused training kernel -- every arm VALID when it ran the path its tag names,
    #     P24 / P25 HELD inside their bands and P26 HELD on matching held-out, P26 FALSIFIED on a 0.02 held-out shift, an rms1 arm that
    #     patched nothing VOID (P24-P26 UNTESTED)
    def pr(R):
        return {p: v for p, _, v, _ in score_rmsab({RMS_FAM: reduce_family(RMS_FAM, R, {}, 20)})}
    RF = {RMS_FAM: reduce_family(RMS_FAM, _rms_set(), {}, 20)}
    assert [(x["fw"], x["tag"]) for x in RF[RMS_FAM]["rows"]] == EXPECTED[RMS_FAM]
    assert all(x["verdict"] == "VALID" for x in RF[RMS_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RF[RMS_FAM]["rows"]]
    assert pr(_rms_set()) == {"P24": "HELD", "P25": "HELD", "P26": "HELD"}
    assert pr(_rms_set(held_shift=0.02)) == {"P24": "HELD", "P25": "HELD", "P26": "FALSIFIED"}
    assert pr(_rms_set(ship=((5.00, 5.05), (4.99, 5.02)))) == {"P24": "FALSIFIED", "P25": "HELD", "P26": "HELD"}
    RR = reduce_family(RMS_FAM, _rms_set(patched=0), {}, 20)
    assert RR["verdicts"][("e4b", "fused_attn4_shipped_rms1")] == "VOID" and "patched > 0" in next(x["why"] for x in RR["rows"] if x["tag"] == "fused_attn4_shipped_rms1")
    assert pr(_rms_set(patched=0)) == {"P24": "UNTESTED", "P25": "UNTESTED", "P26": "UNTESTED"}
    text = render(RF, "x")
    assert "## Predictions P24 / P25 / P26" in text and "| P26 | qwen3rmsab | **HELD** |" in text
    cases += 1
    # 61. amendment 20 (#945): gnf4's per-pass host reuse off vs on -- every arm VALID when it ran the setting its tag names on the trimmed
    #     delta, P33 / P34 HELD inside their bands with the flip reading, FALSIFIED and KEEP above 1.01, a reuse1 arm whose memos never hit
    #     or that ran with the flag off VOID (UNTESTED), an arm on the previous delta body VOID
    def pu(R):
        return {p: v for p, _, v, _ in score_reuseab({REUSE_FAM: reduce_family(REUSE_FAM, R, {}, 20)})}
    UF = {REUSE_FAM: reduce_family(REUSE_FAM, _reuse_set(), {}, 20)}
    assert [(x["fw"], x["tag"]) for x in UF[REUSE_FAM]["rows"]] == EXPECTED[REUSE_FAM]
    assert all(x["verdict"] == "VALID" for x in UF[REUSE_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in UF[REUSE_FAM]["rows"]]
    PU = {p: (v, ev) for p, _, v, ev in score_reuseab(UF)}
    assert PU["P33"][0] == "HELD" and PU["P34"][0] == "HELD" and "reuse1 / reuse0 0.937 [" in PU["P33"][1] and "flip-eligible" in PU["P33"][1], PU
    assert '"plan_hits": 768' in PU["P33"][1], PU["P33"][1]
    slow = score_reuseab({REUSE_FAM: reduce_family(REUSE_FAM, _reuse_set(match=((5.00, 5.02), (5.20, 5.22))), {}, 20)})
    assert [(p, v) for p, _, v, _ in slow] == [("P33", "HELD"), ("P34", "FALSIFIED")] and "KEEP off" in slow[1][3]
    assert pu(_reuse_set(ship=((5.00, 5.05), (4.99, 5.02)))) == {"P33": "FALSIFIED", "P34": "HELD"}        # no gain: 0.995, no measurable effect
    RR = reduce_family(REUSE_FAM, _reuse_set(hits=(1536, 0)), {}, 20)
    assert RR["verdicts"][("e4b", "fused_attn4_shipped_reuse1")] == "VOID" and "plan_hits > 0" in next(x["why"] for x in RR["rows"] if x["tag"] == "fused_attn4_shipped_reuse1")
    assert pu(_reuse_set(hits=(0, 0))) == {"P33": "UNTESTED", "P34": "UNTESTED"} and pu(_reuse_set(flags=("0", "0"))) == {"P33": "UNTESTED", "P34": "UNTESTED"}
    assert reduce_family(REUSE_FAM, _reuse_set(flags=("1", "1")), {}, 20)["verdicts"][("e4b", "fused_attn4_m_reuse0")] == "VOID"
    assert reduce_family(REUSE_FAM, _reuse_set(lean="0"), {}, 20)["verdicts"][("e4b", "fused_attn4_m_reuse0")] == "VOID"
    text = render(UF, "x")
    assert "## Predictions P33 / P34" in text and "| P33 | qwen3reuseab | **HELD** |" in text and "| P34 | qwen3reuseab | **HELD** |" in text
    cases += 1
    # 62. amendment 21 (#945): whole-layer checkpointing vs keeping MoE activations -- every arm VALID when it kept the layers its tag names
    #     with the compact delta, P35 / P36 HELD inside their bands, P37 HELD on peak and held-out, P37 FALSIFIED on a 31.5 GB peak or a 0.01
    #     held-out shift, a keep1 arm that kept the wrong count or ran without the compact delta VOID (UNTESTED)
    def pk(R):
        return {p: v for p, _, v, _ in score_keepab({KEEP_FAM: reduce_family(KEEP_FAM, R, {}, 20)})}
    KF = {KEEP_FAM: reduce_family(KEEP_FAM, _keep_set(), {}, 20)}
    assert [(x["fw"], x["tag"]) for x in KF[KEEP_FAM]["rows"]] == EXPECTED[KEEP_FAM]
    assert all(x["verdict"] == "VALID" for x in KF[KEEP_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in KF[KEEP_FAM]["rows"]]
    PK = {p: (v, ev) for p, _, v, ev in score_keepab(KF)}
    assert PK["P35"][0] == "HELD" and PK["P36"][0] == "HELD" and PK["P37"][0] == "HELD", PK
    assert "keep1 / keep0 0.878 [" in PK["P35"][1] and "(32 of 48 layers kept)" in PK["P35"][1] and "(16 of 48 layers kept)" in PK["P36"][1], PK
    assert pk(_keep_set(peak1=31.5)) == {"P35": "HELD", "P36": "HELD", "P37": "FALSIFIED"}
    assert pk(_keep_set(held_shift=0.01)) == {"P35": "HELD", "P36": "HELD", "P37": "FALSIFIED"}
    assert pk(_keep_set(ship=((5.00, 5.05), (4.99, 5.02)))) == {"P35": "FALSIFIED", "P36": "HELD", "P37": "HELD"}
    RR = reduce_family(KEEP_FAM, _keep_set(kept=8), {}, 20)
    assert RR["verdicts"][("e4b", "fused_attn4_shipped_keep1")] == "VOID" and "layers_kept 32" in next(x["why"] for x in RR["rows"] if x["tag"] == "fused_attn4_shipped_keep1")
    assert pk(_keep_set(kept=8)) == {"P35": "UNTESTED", "P36": "UNTESTED", "P37": "UNTESTED"}
    assert reduce_family(KEEP_FAM, _keep_set(compact="0"), {}, 20)["verdicts"][("e4b", "fused_attn4_m_keep1")] == "VOID"
    text = render(KF, "x")
    assert "## Predictions P35 / P36 / P37" in text and "| P35 | qwen3keepab | **HELD** |" in text and "| P37 | qwen3keepab | **HELD** |" in text
    cases += 1
    # 63. amendment 22: grouped-nf4-gemm's dense route vs its fused kernels on both families -- every arm VALID when it took the route its tag
    #     names (mixtral: with the double-quantized absmax), P38 / P39 HELD inside their bands, P40 HELD on matching held-out; the printer's table
    def DF(m=None, q=None):
        return {MDENSE_FAM: reduce_family(MDENSE_FAM, m if m is not None else _dense_set(MDENSE_FAM), {}, 20),
                QDENSE_FAM: reduce_family(QDENSE_FAM, q if q is not None else _dense_set(QDENSE_FAM), {}, 20)}

    def pd(m=None, q=None):
        return {p: v for p, _, v, _ in score_denseab(DF(m, q))}
    DD = DF()
    for fam in DENSE_FAMS:
        assert [(x["fw"], x["tag"]) for x in DD[fam]["rows"]] == EXPECTED[fam]
        assert all(x["verdict"] == "VALID" for x in DD[fam]["rows"]), [(fam, x["tag"], x["verdict"], x["why"]) for x in DD[fam]["rows"]]
    PD = {p: (fam, v, ev) for p, fam, v, ev in score_denseab(DD)}
    assert [p for p, _, _, _ in score_denseab(DD)] == ["P38", "P39", "P40"]
    assert PD["P38"][:2] == (MDENSE_FAM, "HELD") and "dense1 / dense0 0.746 [" in PD["P38"][2] and '"dense_dgrad": 1536' in PD["P38"][2], PD["P38"]
    assert PD["P39"][:2] == (QDENSE_FAM, "HELD") and "dense1 / dense0 0.933 [" in PD["P39"][2] and "every-group-count reading" in PD["P39"][2], PD["P39"]
    assert PD["P40"][1] == "HELD" and "mixtraldenseab: mean held-out dense1 - dense0 +0.0000" in PD["P40"][2], PD["P40"]
    assert pd(q=_dense_set(QDENSE_FAM, d1=(5.10, 5.12))) == {"P38": "HELD", "P39": "HELD", "P40": "HELD"}           # 0.961: HELD, fused stays at every group count
    assert "fused stays at every group count" in score_denseab(DF(q=_dense_set(QDENSE_FAM, d1=(5.10, 5.12))))[1][3]
    assert pd(m=_dense_set(MDENSE_FAM, held_shift=0.009)) == {"P38": "HELD", "P39": "HELD", "P40": "HELD"}          # 0.009 <= 0.01
    text = render(DD, "x")
    assert ("## Predictions P38 / P39 / P40" in text and "| P38 | mixtraldenseab | **HELD** |" in text and "| P39 | qwen3denseab | **HELD** |" in text
            and "| P40 | denseab | **HELD** |" in text), text[-1500:]
    cases += 1
    # 64. amendment 22, each band edge: P38 FALSIFIED below 0.55 and above 0.90, P39 FALSIFIED below 0.85 and above 1.15; P40 FALSIFIED on a
    #     0.02 held-out shift on either family (the speed readings unchanged)
    assert pd(m=_dense_set(MDENSE_FAM, d1=(2.90, 2.92))) == {"P38": "FALSIFIED", "P39": "HELD", "P40": "HELD"}       # 0.528
    assert pd(m=_dense_set(MDENSE_FAM, d1=(5.10, 5.12))) == {"P38": "FALSIFIED", "P39": "HELD", "P40": "HELD"}       # 0.927
    assert pd(q=_dense_set(QDENSE_FAM, d1=(4.40, 4.42))) == {"P38": "HELD", "P39": "FALSIFIED", "P40": "HELD"}       # 0.829
    assert pd(q=_dense_set(QDENSE_FAM, d1=(6.20, 6.22))) == {"P38": "HELD", "P39": "FALSIFIED", "P40": "HELD"}       # 1.168
    assert pd(m=_dense_set(MDENSE_FAM, held_shift=0.02)) == {"P38": "HELD", "P39": "HELD", "P40": "FALSIFIED"}
    assert pd(q=_dense_set(QDENSE_FAM, held_shift=-0.02)) == {"P38": "HELD", "P39": "HELD", "P40": "FALSIFIED"}
    lo_ev = score_denseab(DF(m=_dense_set(MDENSE_FAM, d1=(2.90, 2.92))))[0][3]
    print("FAILING-CASE A22-P38 (reducer):", lo_ev[:120])
    print("FAILING-CASE A22-P40 (reducer):", score_denseab(DF(m=_dense_set(MDENSE_FAM, held_shift=0.02)))[2][3][:120])
    cases += 1
    # 65. amendment 22, instability and missing sides: a dense1 side whose draws differ by > 5 % is UNSTABLE (P38 UNTESTED and P40 UNTESTED
    #     with the other family still read), a missing second draw UNMEASURED, a family absent from the directory UNTESTED
    DU = DF(m=_dense_set(MDENSE_FAM, d1=(4.10, 4.40)))
    assert DU[MDENSE_FAM]["draws"][("e4b", "fused_attn4_m_dense1")]["verdict"] == "UNSTABLE"
    assert {p: v for p, _, v, _ in score_denseab(DU)} == {"P38": "UNTESTED", "P39": "HELD", "P40": "UNTESTED"}
    print("FAILING-CASE A22-unstable (reducer):", score_denseab(DU)[0][3][:160])
    Q = _dense_set(QDENSE_FAM)
    del Q[("e4b", "fused_attn4_m_dense0_d2")]
    assert pd(q=Q) == {"P38": "HELD", "P39": "UNTESTED", "P40": "UNTESTED"}
    assert DF(q=Q)[QDENSE_FAM]["draws"][("e4b", "fused_attn4_m_dense0")]["verdict"] == "UNMEASURED"
    only_q = {QDENSE_FAM: reduce_family(QDENSE_FAM, _dense_set(QDENSE_FAM), {}, 20)}
    assert {p: v for p, _, v, _ in score_denseab(only_q)} == {"P38": "UNTESTED", "P39": "HELD", "P40": "UNTESTED"}
    assert score_denseab({}) == [] and "## Predictions P38 / P39 / P40" not in render({}, "x")
    cases += 1
    # 66. amendment 22, engagement: a side that did not take the route its tag names reads VOID with the reason and its prediction UNTESTED --
    #     a dense1 side with no dense dgrad, a dense1 side resolved to fused, a dense0 side resolved to dense or counting dense forwards, a dense0
    #     record without the dense_fwd counter (grouped-nf4-gemm before #459), a receipt without a route_ab record
    def void_why(fam, R, tag):
        F1 = reduce_family(fam, R, {}, 20)
        return F1["verdicts"][("e4b", tag)], next(x["why"] for x in F1["rows"] if x["tag"] == tag)
    v, why = void_why(QDENSE_FAM, _dense_set(QDENSE_FAM, dense_dgrad=(0, 0)), "fused_attn4_m_dense1")
    assert v == "VOID" and "dense-route A/B not engaged (dense_dgrad > 0;" in why, why
    print("FAILING-CASE A22-engagement (reducer):", why[:160])
    assert pd(q=_dense_set(QDENSE_FAM, dense_dgrad=(0, 0))) == {"P38": "HELD", "P39": "UNTESTED", "P40": "UNTESTED"}
    v, why = void_why(QDENSE_FAM, _dense_set(QDENSE_FAM, routes=("fused", "fused"), dense_fwd=(0, 0), dense_dgrad=(0, 0)), "fused_attn4_m_dense1_d2")
    assert v == "VOID" and "gnf4_train_gemm dense, dense_fwd > 0, dense_dgrad > 0" in why, why
    v, why = void_why(MDENSE_FAM, _dense_set(MDENSE_FAM, routes=("dense", "dense")), "fused_attn4_m_dense0")
    assert v == "VOID" and "gnf4_train_gemm fused" in why and "dense_fwd 0" not in why, why
    v, why = void_why(MDENSE_FAM, _dense_set(MDENSE_FAM, dense_fwd=(64, 1536)), "fused_attn4_m_dense0_d2")
    assert v == "VOID" and "(dense_fwd 0;" in why, why
    assert pd(m=_dense_set(MDENSE_FAM, dense_fwd=(64, 1536))) == {"P38": "UNTESTED", "P39": "HELD", "P40": "UNTESTED"}
    v, why = void_why(QDENSE_FAM, _dense_set(QDENSE_FAM, dense_fwd=(None, 1536)), "fused_attn4_m_dense0")
    assert v == "VOID" and "dense_fwd 0" in why, why
    v, why = void_why(QDENSE_FAM, _dense_set(QDENSE_FAM, drop_route=True), "fused_attn4_m_dense0")
    assert v == "VOID" and "no route_ab record" in why, why
    assert dense_ab_why(QDENSE_FAM, "fused_attn4_m_dense1", {}).startswith("no route_ab record")
    cases += 1
    # 67. amendment 22, mixtral's absmax_dq requirement and its pin: a mixtral side without the double-quantized absmax reads VOID (either side),
    #     while qwen3denseab requires nothing of it; a mixtral receipt off TC2's pin or with Qwen3's layer count reads VOID
    v, why = void_why(MDENSE_FAM, _dense_set(MDENSE_FAM, absmax=(True, False)), "fused_attn4_m_dense1")
    assert v == "VOID" and "(absmax_dq true;" in why and "absmax_dq False" in why, why
    print("FAILING-CASE A22-absmax (reducer):", why[:160])
    v, why = void_why(MDENSE_FAM, _dense_set(MDENSE_FAM, absmax=(False, True)), "fused_attn4_m_dense0_d2")
    assert v == "VOID" and "absmax_dq true" in why, why
    assert pd(m=_dense_set(MDENSE_FAM, absmax=(False, True))) == {"P38": "UNTESTED", "P39": "HELD", "P40": "UNTESTED"}
    assert all(x["verdict"] == "VALID" for x in reduce_family(QDENSE_FAM, _dense_set(QDENSE_FAM, absmax=(False, False)), {}, 20)["rows"])
    M = _dense_set(MDENSE_FAM)
    M[("e4b", "fused_attn4_m_dense1")]["revision"] = "0" * 40
    v, why = void_why(MDENSE_FAM, M, "fused_attn4_m_dense1")
    assert v == "VOID" and "the registered pin mistralai/Mixtral-8x7B-Instruct-v0.1 @ eba92302a286" in why, why
    M = _dense_set(MDENSE_FAM)
    M[("e4b", "fused_attn4_m_dense0_d2")]["n_layers"] = 48
    v, why = void_why(MDENSE_FAM, M, "fused_attn4_m_dense0_d2")
    assert v == "VOID" and "config n_layers 48 != registered 32" in why, why
    Q = _dense_set(QDENSE_FAM)
    Q[("e4b", "fused_attn4_m_dense0")]["model"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    v, why = void_why(QDENSE_FAM, Q, "fused_attn4_m_dense0")
    assert v == "VOID" and "the registered pin Qwen/Qwen3-30B-A3B @ ad44e777bcd1" in why, why
    cases += 1
    # 68. TC2 amendment 8, box Q (tc2qwen35mb1): Qwen3.6 alone, resident, the PRIMARY pair at micro-batch 1 x accum 8 on both sides under the
    #     primary tags (e4b with the absmax double-quantized and the frozen projections in NF4; Unsloth with the expert target parameters), two
    #     draws a side; the reference, HF, both axolotl arms and e4b as shipped not_run; no _mb1 receipt. It reads as the family's ordinary
    #     matched pair: every arm VALID on the accum-8 counters, both pairs STABLE, the position quoted and named with its micro-batch, the
    #     step-0 class NEAR, COMPARABLE the ceiling without the reference, no secondary line; P4 UNTESTED (its e4b leg is the field recipe's step)
    def box_q(fam="qwen3_5"):
        L, A = N_LAYERS[fam], 8
        mb1 = {"micro_batch": 1, "accum": A}
        uns = {**mb1, "experts_forward_calls_per_step_min": L * A, "unsloth_grouped_mm_calls_per_step_min": GMM_FACTOR * L * A,
               "unsloth_grouped_mm_calls_per_step_max": GMM_FACTOR * L * A,
               "unsloth_backend_calls_per_step_min": {"unsloth_grouped_mm": L * A, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": L * A},
               "unsloth_backend_calls_per_step_max": {"unsloth_grouped_mm": L * A, "unsloth_triton": 0, "unsloth_loop": 0, "moe_bnb4bit_backend": L * A},
               "unsloth_knobs": {"target_parameters_requested": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]},
               "eval_loss_step0": 2.012, "eval_curve": [{"step": 0, "heldout_loss": 2.012}, {"step": 20, "heldout_loss": 1.803}],
               "losses": [round(2.003 - 0.01 * i, 5) for i in range(20)], "loss_last": round(2.003 - 0.19, 5), "loss_step2": 1.983}
        R = {}
        for tag, s_, pk in (("fused_attn4_m", 9.24, 31.36), ("fused_attn4_m_d2", 9.30, 31.40)):
            R[("e4b", tag)] = _tc2_receipt(fam, "e4b", tag, "fused", s=s_, peak_vram_gb=pk, offload=False, absmax_dq=True, frozen_4bit=True,
                                           kernel_calls_per_step_min=2 * L * A, **mb1)
        for tag, s_, pk in (("ckpt_unsloth_m", 18.49, 30.41), ("ckpt_unsloth_m_d2", 18.60, 30.45)):
            R[("unsloth", tag)] = _tc2_receipt(fam, "unsloth", tag, "unsloth", s=s_, heldout_n=1.803, peak_vram_gb=pk, **uns)
        for fw, tag, arm_ in (("hf", "hf_peft_m", "hf"), ("axolotl", "ckpt_axolotl_m", "axolotl"), ("axolotl", "ckpt_axolotl_best", "axolotl"),
                              ("e4b", "fused_attn4_shipped", "fused"), ("e4b", "reference_attn4_m", "reference")):
            R[(fw, tag)] = {**_stub(fw, tag, arm_, "not_run", "skipped by TC1_SKIP"), "fam": fam}
        return R
    QB = trun("qwen3_5", box_q())
    for key in (("e4b", "fused_attn4_m"), ("e4b", "fused_attn4_m_d2"), ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_d2")):
        assert QB["verdicts"][key] == "VALID", (key, row(QB, *key)["why"])
    assert all(QB["verdicts"][k] == "NOT_RUN" for k in (("e4b", "reference_attn4_m"), ("hf", "hf_peft_m"), ("e4b", "fused_attn4_shipped"), ("unsloth", "ckpt_unsloth_m_experts")))
    assert QB["draws"][("e4b", "fused_attn4_m")]["verdict"] == "STABLE" and QB["draws"][("unsloth", "ckpt_unsloth_m")]["verdict"] == "STABLE"
    pq = QB["positions"]["unsloth"]
    assert pq["quoted"] and abs(pq["ratio"] - 18.545 / 9.27) < 1e-9 and pq["e4b_draws"] == 2 and pq["other_draws"] == 2 and pq["quality"] == "COMPARABLE", pq
    assert pq["recipe"] == "micro-batch 1 × accum 8" and QB["anchor_micro_batch"] == 1 and QB["secondary"] == {} and QB["anchor_offload"] is False and QB["footprint"] is None
    eq = QB["equivalence"][("unsloth", "ckpt_unsloth_m")]
    assert eq["reading"] == "COMPARABLE" and eq["step0_class"] == "NEAR" and abs(eq["d_step0"] - 0.012) < 1e-9 and QB["parity"]["verdict"] == "NO-REF", (eq, QB["parity"])
    ln = pos_lines(pq, QB["N"], prefix="MATCHED POSITION")
    assert ln[0].startswith("- **MATCHED POSITION (micro-batch 1 × accum 8): s/step ratio unsloth/e4b = 2.001** [1.988, 2.013 over 4 cross-draw ratios]"), ln[0]
    assert "step-0 Δ +0.0120" in ln[1], ln[1]
    blk = "\n".join(family_block(QB))
    assert "micro-batch 1 × accum 8 lr" in blk and "SECONDARY POSITION" not in blk and "FOOTPRINT" not in blk and "`unsloth/ckpt_unsloth_m` **COMPARABLE**" in blk
    p4 = next(x for x in score_tc2_predictions({**T_ALL, "qwen3_5": QB}) if x[0] == "P4")
    assert p4[2] == "UNTESTED" and "e4b's anchor ran micro-batch 1" in p4[3], p4
    assert TP(T_ALL)["P4"] == "HELD" and T_ALL["qwen3_5"]["anchor_micro_batch"] == 2 and all(not pz.get("recipe") for pz in T_ALL["qwen3_5"]["positions"].values())
    print("FAILING-CASE TC2-mb1-primary (reducer): P4", p4[2], "--", p4[3][:110])
    Rm = box_q()                                             # a side that ran the field recipe beside a micro-batch-1 side is named, never folded in
    Rm[("unsloth", "ckpt_unsloth_m")].update({"micro_batch": 2, "accum": 4})
    assert trun("qwen3_5", Rm)["positions"]["unsloth"].get("recipe") == "recipes differ: e4b micro-batch 1 × accum 8, unsloth micro-batch 2 × accum 4"
    td = tempfile.mkdtemp(prefix="tc2am8_reduce_selftest_")    # through the files, with the field N the box passes (--steps 20)
    for (fw, tag), r in box_q().items():
        json.dump(r, open(os.path.join(td, f"qwen3_5_{fw}_{tag}.json"), "w"))
    F = reduce_dir(td, 20)
    text = render(F, td)
    for needle in ("MATCHED POSITION (micro-batch 1 × accum 8): s/step ratio unsloth/e4b = 2.001**", "| P4 | qwen3_5 | **UNTESTED** | e4b's anchor ran micro-batch 1",
                   "draws (R1): `e4b/fused_attn4_m` STABLE (9.240/9.300 s"):
        assert needle in text, needle
    cases += 1
    # 69. TC2 amendment 8, box M (tc2mixtralres): Mixtral alone at the field recipe, every e4b arm resident at e4b's defaults, Unsloth x2, e4b x2
    #     and the reference; HF, both axolotl arms and e4b as shipped not_run. The pair reads a quoted position with no recipe note and no
    #     footprint line (amendment 6's gate: the anchor ran resident), P5 UNTESTED, the parity control and the EQUIVALENT band read
    Rx = _tc2_set("mixtral")
    for tag, s_ in (("fused_attn4_m", 3.62), ("fused_attn4_m_d2", 3.66), ("reference_attn4_m", 6.8)):
        Rx[("e4b", tag)].update({"offload": False, "peak_vram_gb": 31.2, "s_per_step_median_11plus": s_})
    for tag, s_ in (("ckpt_unsloth_m", 3.40), ("ckpt_unsloth_m_d2", 3.44)):
        Rx[("unsloth", tag)].update({"peak_vram_gb": 29.13, "s_per_step_median_11plus": s_})
    for fw, tag, arm_ in (("hf", "hf_peft_m", "hf"), ("axolotl", "ckpt_axolotl_m", "axolotl"), ("axolotl", "ckpt_axolotl_best", "axolotl"), ("e4b", "fused_attn4_shipped", "fused")):
        Rx[(fw, tag)] = {**_stub(fw, tag, arm_, "not_run", "skipped by TC1_SKIP"), "fam": "mixtral"}
    MB = trun("mixtral", Rx)
    pm = MB["positions"]["unsloth"]
    assert MB["anchor_offload"] is False and MB["footprint"] is None and not family_block(MB)[1].startswith("- **FOOTPRINT"), MB["footprint"]
    assert pm["quoted"] and abs(pm["ratio"] - 3.42 / 3.64) < 1e-9 and pm["recipe"] == "" and pm["peak_e4b"] == 31.2, pm
    assert MB["verdicts"][("e4b", "reference_attn4_m")] == "VALID" and MB["parity"]["verdict"] == "PASS" and MB["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "EQUIVALENT"
    assert pos_lines(pm, MB["N"], prefix="MATCHED POSITION")[0].startswith("- **MATCHED POSITION: s/step ratio unsloth/e4b = 0.940**")
    tm = TP({**T_ALL, "mixtral": MB})
    assert tm["P5"] == "UNTESTED" and MB["anchor_micro_batch"] == 2, tm
    print("FAILING-CASE TC2-mixtral-default (reducer): resident at e4b's defaults -> quoted", f"{pm['ratio']:.3f}", "no footprint line, P5", tm["P5"])
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
    # 70. amendment 23: the memory census -- every arm VALID, NO position quoted on the token (the census slows the step), P41 / P42 / P43 HELD
    #     with the evidence naming the excess's largest class at the peak; the printer's side-by-side tables and the prediction table
    def MC(R):
        return {MEMCENSUS_FAM: reduce_family(MEMCENSUS_FAM, R, {}, 20)}

    def pm(R):
        return {p: v for p, _, v, _ in score_memcensus(MC(R))}
    M = MC(_memcensus_set())
    MR = M[MEMCENSUS_FAM]
    assert [(x["fw"], x["tag"]) for x in MR["rows"]] == EXPECTED[MEMCENSUS_FAM]
    assert all(x["verdict"] == "VALID" for x in MR["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in MR["rows"]]
    assert MR["secondary"] and not any(p.get("quoted") for grp in ("positions", "secondary", "labelled", "native") for p in MR[grp].values())
    assert "the memory census slows the step" in MR["secondary"]["unsloth"]["why"] and "the memory census slows the step" in MR["positions"]["unsloth"]["why"]
    PM = {p: (v, ev) for p, _, v, ev in score_memcensus(M)}
    assert [p for p, _, _, _ in score_memcensus(M)] == ["P41", "P42", "P43"]
    assert PM["P41"][0] == "HELD" and "expert_params 28,991,029,248 / 64 × 4 bytes" in PM["P41"][1] and "fp32 / dq bytes 3.938" in PM["P41"][1], PM["P41"]
    assert "e4b fp32 absmax 1.8119 GB vs the analytic value 1.8119 GB (Δ +0.00 %" in PM["P41"][1], PM["P41"]
    assert PM["P42"][0] == "HELD" and "Unsloth 0.9200 (>= 0.9" in PM["P42"][1], PM["P42"]
    assert PM["P43"][0] == "HELD" and "= +1.210 GB vs [0.5, 2.5] GB" in PM["P43"][1] and "the excess's largest group at the peak: transient +1.000 GB" in PM["P43"][1], PM["P43"]
    text = render(M, "x")
    for needle in ("## Amendment 23: the memory census", "| | e4b fp32 absmax | e4b dq absmax | Unsloth |", "| peak allocated / reserved (GB) | 26.060 / 26.560 | 24.710 / 25.210 | 23.500 / 24.000 |",
                   "| expert_absmax | 1.812 / 1.812 | 0.460 / 0.460 | 0.460 / 0.460 |", "| optimizer_state | 0.000 / 1.285 |", "**Top live-at-peak groups",
                   "| 1 | static:frozen_expert_weights 14.496 ×96 |", "## Predictions P41 / P42 / P43", "| P41 | qwen3memcensus | **HELD** |",
                   "| P42 | qwen3memcensus | **HELD** |", "| P43 | qwen3memcensus | **HELD** |", "NO MATCHED POSITION QUOTED (unsloth)"):
        assert needle in text, needle
    assert "MATCHED POSITION: s/step" not in text and "SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed): s/step" not in text
    cases += 1
    # 71. amendment 23, each prediction's failing leg: P41 on the fp32 arm (+4.9 %) and on the dq arm (+30 %); P42 on Unsloth's 0.85; P43 above
    #     2.5 GB and below 0.5 GB (the band's edge reads HELD)
    assert pm(_memcensus_set(absmax=(1_900_000_000, QWEN3_ABSMAX_DQ))) == {"P41": "FALSIFIED", "P42": "HELD", "P43": "HELD"}
    assert pm(_memcensus_set(absmax=(QWEN3_ABSMAX_FP32, 600_000_000))) == {"P41": "FALSIFIED", "P42": "HELD", "P43": "HELD"}
    assert pm(_memcensus_set(af=(0.95, 0.94, 0.85))) == {"P41": "HELD", "P42": "FALSIFIED", "P43": "HELD"}
    assert pm(_memcensus_set(dq_peak=26.10e9)) == {"P41": "HELD", "P42": "HELD", "P43": "FALSIFIED"}                     # +2.60 GB
    assert pm(_memcensus_set(dq_peak=23.90e9)) == {"P41": "HELD", "P42": "HELD", "P43": "FALSIFIED"}                     # +0.40 GB
    assert pm(_memcensus_set(dq_peak=24.00e9))["P43"] == "HELD" and pm(_memcensus_set(dq_peak=26.00e9))["P43"] == "HELD"   # the band's edges
    print("FAILING-CASE A23-P41 (reducer):", score_memcensus(MC(_memcensus_set(absmax=(1_900_000_000, QWEN3_ABSMAX_DQ))))[0][3][:150])
    print("FAILING-CASE A23-P43 (reducer):", score_memcensus(MC(_memcensus_set(dq_peak=26.10e9)))[2][3][:120])
    cases += 1
    # 72. amendment 23, UNTESTED and VOID: a missing Unsloth arm, an errored census (recorded, not VOID), a reduced window that missed the run's peak,
    #     a dq arm that ran the fp32 absmax, an arm without its census, an arm off the mb1 recipe; P41 falls back to 29.0e9 / 64 x 4 without expert_params
    R = _memcensus_set()
    del R[MEMCENSUS_ARMS[2]]
    assert pm(R) == {"P41": "HELD", "P42": "UNTESTED", "P43": "UNTESTED"}
    R = _memcensus_set()
    R[MEMCENSUS_ARMS[1]]["mem_census"] = _mc_census(0, 0, error="checkpoint s1.mb1: RuntimeError: injected")
    MR = MC(R)
    assert MR[MEMCENSUS_FAM]["verdicts"][MEMCENSUS_ARMS[1]] == "VALID" and pm(R) == {"P41": "UNTESTED", "P42": "UNTESTED", "P43": "UNTESTED"}
    assert "the census errored (checkpoint s1.mb1: RuntimeError: injected)" in score_memcensus(MR)[2][3]
    assert pm(_memcensus_set(in_window=(True, False, True))) == {"P41": "HELD", "P42": "UNTESTED", "P43": "HELD"}
    R = _memcensus_set()
    R[MEMCENSUS_ARMS[1]]["absmax_dq"] = False
    MR = MC(R)
    assert MR[MEMCENSUS_FAM]["verdicts"][MEMCENSUS_ARMS[1]] == "VOID" and "the tag names the double-quantized expert absmax" in next(x["why"] for x in MR[MEMCENSUS_FAM]["rows"] if x["tag"] == "fused_attn4_m_mb1_dq")
    assert pm(R) == {"P41": "UNTESTED", "P42": "UNTESTED", "P43": "UNTESTED"}
    R = _memcensus_set()
    del R[MEMCENSUS_ARMS[0]]["mem_census"]
    assert MC(R)[MEMCENSUS_FAM]["verdicts"][MEMCENSUS_ARMS[0]] == "VOID" and pm(R) == {"P41": "UNTESTED", "P42": "UNTESTED", "P43": "HELD"}
    R = _memcensus_set()
    R[MEMCENSUS_ARMS[2]]["accum"] = 4
    MR = MC(R)
    assert MR[MEMCENSUS_FAM]["verdicts"][MEMCENSUS_ARMS[2]] == "VOID" and "recipe micro-batch 1 x accum 4" in next(x["why"] for x in MR[MEMCENSUS_FAM]["rows"] if x["fw"] == "unsloth")
    R = _memcensus_set()
    for k in MEMCENSUS_ARMS[:2]:
        R[k]["mem_census"]["static_after_setup"]["expert_params"] = None
    p41 = score_memcensus(MC(R))[0]
    assert p41[2] == "HELD" and "29.0e9 / 64 × 4 bytes (no expert_params on the receipts)" in p41[3] and "(Δ -0.03 %" in p41[3], p41
    cases += 1
    # 73. amendment 23 end to end through the files: the token's receipts in a directory, reduce_dir + render
    md_ = tempfile.mkdtemp(prefix="tc1_reduce_selftest_memcensus_")
    for (fw, tag), r in _memcensus_set().items():
        json.dump(r, open(os.path.join(md_, f"{MEMCENSUS_FAM}_{fw}_{tag}.json"), "w"))
    F = reduce_dir(md_, 20)
    assert set(F) == {MEMCENSUS_FAM} and [(x["fw"], x["tag"]) for x in F[MEMCENSUS_FAM]["rows"]] == EXPECTED[MEMCENSUS_FAM]
    text = render(F, md_)
    assert "| P43 | qwen3memcensus | **HELD** |" in text and "### Qwen3-30B-A3B (amendment 23: the memory census" in text and "## Predictions P1–P10" not in text
    cases += 1
    # 74. TC1 amendment 24 (qwen3bmmab): the environment A/B (P47-P49) -- matched 0.841 and shipped 0.990 HELD; no matched gain, a shipped
    #     gain as large as the matched one, and a held-out gap each FALSIFY their prediction
    BF = lambda R: {BMMAB_FAM: reduce_family(BMMAB_FAM, R, {}, 20)}
    RB = BF(_bmm_set())
    assert [(x["fw"], x["tag"]) for x in RB[BMMAB_FAM]["rows"]] == EXPECTED[BMMAB_FAM]
    assert all(x["verdict"] == "VALID" for x in RB[BMMAB_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RB[BMMAB_FAM]["rows"]]
    pb = lambda **kw: {p: v for p, _, v, _ in score_bmmab(BF(_bmm_set(**kw)))}
    assert pb() == {"P47": "HELD", "P48": "HELD", "P49": "HELD"}, pb()
    assert "tv1 / tv0 0.840 [" in score_bmmab(RB)[0][3], score_bmmab(RB)[0]
    assert pb(m=((5.00, 5.03), (4.95, 4.97))) == {"P47": "FALSIFIED", "P48": "FALSIFIED", "P49": "HELD"}     # 0.989: no matched gain
    assert pb(sh=((3.90, 3.92), (3.40, 3.42))) == {"P47": "HELD", "P48": "FALSIFIED", "P49": "HELD"}        # shipped 0.872: not fp32-specific
    assert pb(m=((5.00, 5.03), (3.40, 3.42))) == {"P47": "FALSIFIED", "P48": "HELD", "P49": "HELD"}         # 0.680: below the band
    assert pb(held_shift=0.02) == {"P47": "HELD", "P48": "HELD", "P49": "FALSIFIED"}
    cases += 1
    # 75. FAILING CASES: a tv1 side that ran torch 2.8 (the venv fell back) or the wrong adapter dtype is VOID, and the A/B reads UNTESTED
    RV = BF(_bmm_set(torch_v=("2.8.0+cu128", "2.8.0+cu128")))
    v = RV[BMMAB_FAM]["verdicts"][("e4b", "fused_attn4_m_tv1")]
    why = next(x["why"] for x in RV[BMMAB_FAM]["rows"] if x["tag"] == "fused_attn4_m_tv1")
    print("FAILING-CASE TC1-am24-env (reducer):", v, "--", str(why)[-140:])
    assert v == "VOID" and "env.torch 2.8.0+cu128 is not 2.12.*" in str(why), (v, why)
    assert pb(torch_v=("2.8.0+cu128", "2.8.0+cu128")) == {"P47": "UNTESTED", "P48": "UNTESTED", "P49": "UNTESTED"}
    RW = BF(_bmm_set(wrong_dtype=True))
    assert RW[BMMAB_FAM]["verdicts"][("e4b", "fused_attn4_m_tv1")] == "VOID"
    assert "adapter_dtype 'native' != 'fp32'" in str(next(x["why"] for x in RW[BMMAB_FAM]["rows"] if x["tag"] == "fused_attn4_m_tv1"))
    assert bmm_ab_why("fused_attn4_shipped_tv0", {"env": {"torch": "2.8.0+cu128"}, "adapter_dtype": "native"}) == ""
    cases += 1
    # 76. the replay (P44-P46): HELD, each FALSIFIED in turn, UNTESTED without the 2.12 file or off the registered card, nothing without files
    pr = lambda dd: {p: v for p, _, v, _ in score_bmm_replay(dd)}
    T28 = {("fp32", "default"): (320.0, 40.0, 310.0), ("bf16", "default"): (15.0, 14.0, 15.0)}
    bd = tempfile.mkdtemp(prefix="tc1_bmm_selftest_")
    _bmm_replay_files(bd, t28=T28, t212={("fp32", "default"): (40.0, 30.0, 45.0)})
    assert pr(bd) == {"P44": "HELD", "P45": "HELD", "P46": "HELD"}, pr(bd)
    assert "cold 310.0 us" in score_bmm_replay(bd)[1][3] and "multiple of 32" in score_bmm_replay(bd)[1][3]
    bd = tempfile.mkdtemp(prefix="tc1_bmm_selftest_")
    _bmm_replay_files(bd, t28={("fp32", "default"): (90.0, 40.0, 95.0), ("bf16", "default"): (15.0, 14.0, 15.0)}, t212={("fp32", "default"): (40.0, 30.0, 45.0)})
    assert pr(bd) == {"P44": "FALSIFIED", "P45": "HELD", "P46": "HELD"}, pr(bd)        # 90 us: the anomaly is not the call's own
    bd = tempfile.mkdtemp(prefix="tc1_bmm_selftest_")
    _bmm_replay_files(bd, t28={("fp32", "default"): (320.0, 300.0, 310.0), ("bf16", "default"): (15.0, 14.0, 15.0)}, t212={("fp32", "default"): (250.0, 240.0, 250.0)})
    assert pr(bd) == {"P44": "HELD", "P45": "FALSIFIED", "P46": "FALSIFIED"}, pr(bd)   # every call slow, and torch 2.12 no better
    bd = tempfile.mkdtemp(prefix="tc1_bmm_selftest_")
    _bmm_replay_files(bd, t28=T28)
    print("FAILING-CASE TC1-am24-replay (reducer):", score_bmm_replay(bd)[2])
    assert pr(bd) == {"P44": "HELD", "P45": "HELD", "P46": "UNTESTED"} and "BMMBENCH-t212.json not in this directory" in score_bmm_replay(bd)[2][3]
    bd = tempfile.mkdtemp(prefix="tc1_bmm_selftest_")
    _bmm_replay_files(bd, t28=T28, t212={("fp32", "default"): (40.0, 30.0, 45.0)}, cap=(8, 6))
    assert pr(bd) == {"P44": "UNTESTED", "P45": "UNTESTED", "P46": "UNTESTED"} and "not the registered sm_120 card" in score_bmm_replay(bd)[0][3]
    assert score_bmm_replay(tempfile.mkdtemp(prefix="tc1_bmm_selftest_")) == [] and bmm_replay_table(None) == []
    cases += 1
    # 77. end to end through the files: the A/B receipts and both replay files -> the P44-P49 table and the replay rows
    bd = tempfile.mkdtemp(prefix="tc1_bmm_selftest_")
    for (fw, tag), r in _bmm_set().items():
        json.dump(r, open(os.path.join(bd, f"{BMMAB_FAM}_{fw}_{tag}.json"), "w"))
    _bmm_replay_files(bd, t28=T28, t212={("fp32", "default"): (40.0, 30.0, 45.0)})
    F = reduce_dir(bd, 20)
    assert set(F) == {BMMAB_FAM} and [(x["fw"], x["tag"]) for x in F[BMMAB_FAM]["rows"]] == EXPECTED[BMMAB_FAM]
    text = render(F, bd)
    for needle in ("## Predictions P44–P49", "| P44 | bmmbench | **HELD** |", "| P46 | bmmbench | **HELD** |", "| P47 | qwen3bmmab | **HELD** |",
                   "| P49 | qwen3bmmab | **HELD** |", "**Replay rows**", "| t212 | 2.12.1+cu130 | fp32 | default | 40.0 | 30.0 | 45.0 |"):
        assert needle in text, needle
    cases += 1
    # 78. TC1 amendment 25 (qwen3samestack): the same-stack position P50 (2.297), the environment replication P51 (0.882), the matched set P52;
    #     each FALSIFIED in turn
    SF = lambda R: {SAMESTACK_FAM: reduce_family(SAMESTACK_FAM, R, {}, 20)}
    RS = SF(_samestack_set())
    assert [(x["fw"], x["tag"]) for x in RS[SAMESTACK_FAM]["rows"]] == EXPECTED[SAMESTACK_FAM]
    assert all(x["verdict"] == "VALID" for x in RS[SAMESTACK_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RS[SAMESTACK_FAM]["rows"]]
    ps = lambda **kw: {p: v for p, _, v, _ in score_samestack(SF(_samestack_set(**kw)))}
    assert ps() == {"P50": "HELD", "P51": "HELD", "P52": "HELD"}, (ps(), score_samestack(RS))
    assert "Unsloth/e4b on one stack 2.297 [" in score_samestack(RS)[0][3] and "venv-unsloth / venv-e4b 0.882 [" in score_samestack(RS)[1][3]
    assert ps(u=(10.40, 10.45)) == {"P50": "FALSIFIED", "P51": "HELD", "P52": "HELD"}          # 3.02
    assert ps(e=(3.88, 3.90)) == {"P50": "HELD", "P51": "FALSIFIED", "P52": "HELD"}           # 0.995: no environment gain
    assert ps(u_held=1.8300) == {"P50": "HELD", "P51": "HELD", "P52": "FALSIFIED"}            # Unsloth 0.03 off: COMPARABLE, not EQUIVALENT
    cases += 1
    # 79. FAILING CASES: the anchor ran torch 2.8 (venv-unsloth's install fell back) -> VOID, P50-P52 UNTESTED; no reference -> P52 UNTESTED;
    #     an Unsloth single draw -> P50 UNTESTED
    RV = SF(_samestack_set(torch_v=("2.8.0+cu128", "2.8.0+cu128")))
    v = RV[SAMESTACK_FAM]["verdicts"][("e4b", "fused_attn4_m")]
    why = next(x["why"] for x in RV[SAMESTACK_FAM]["rows"] if x["tag"] == "fused_attn4_m")
    print("FAILING-CASE TC1-am25-env (reducer):", v, "--", str(why)[-120:])
    assert v == "VOID" and "env.torch 2.8.0+cu128 is not 2.12.*" in str(why), (v, why)
    assert ps(torch_v=("2.8.0+cu128", "2.8.0+cu128")) == {"P50": "UNTESTED", "P51": "UNTESTED", "P52": "UNTESTED"}
    assert ps(drop=(("e4b", "reference_attn4_m"),))["P52"] == "UNTESTED"
    assert ps(drop=(("unsloth", "ckpt_unsloth_m_d2"),))["P50"] == "UNTESTED"
    assert samestack_why(SAMESTACK_T28 + "_d2", {"env": {"torch": "2.8.0+cu128"}}) == ""
    cases += 1
    # 80. TC1 amendment 26 (qwen3prebindab): shipped 0.942 and matched 0.954 HELD; no gain, too large a gain and a held-out gap each FALSIFY
    PF = lambda R: {PREBIND_FAM: reduce_family(PREBIND_FAM, R, {}, 20)}
    RP = PF(_prebind_set())
    assert [(x["fw"], x["tag"]) for x in RP[PREBIND_FAM]["rows"]] == EXPECTED[PREBIND_FAM]
    assert all(x["verdict"] == "VALID" for x in RP[PREBIND_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RP[PREBIND_FAM]["rows"]]
    pp = lambda **kw: {p: v for p, _, v, _ in score_prebindab(PF(_prebind_set(**kw)))}
    assert pp() == {"P53": "HELD", "P54": "HELD", "P55": "HELD"}, score_prebindab(RP)
    assert "pb1 / pb0 0.942 [" in score_prebindab(RP)[0][3] and "triton 3.4.0" in score_prebindab(RP)[0][3]
    assert pp(ship=((3.10, 3.12), (3.08, 3.10))) == {"P53": "FALSIFIED", "P54": "HELD", "P55": "HELD"}         # 0.994: no gain
    assert pp(match=((3.90, 3.92), (3.40, 3.42))) == {"P53": "HELD", "P54": "FALSIFIED", "P55": "HELD"}        # 0.872: more than registered
    assert pp(held_shift=0.008) == {"P53": "HELD", "P54": "HELD", "P55": "FALSIFIED"}
    cases += 1
    # 81. FAILING CASES: a pb1 side whose gnf4 never prebound, a pb0 side that prebound, a pb1 side that did not request, no record -> VOID
    RV = PF(_prebind_set(pb1_counts=(4000, 0)))
    v = RV[PREBIND_FAM]["verdicts"][("e4b", "fused_attn4_m_pb1")]
    why = next(x["why"] for x in RV[PREBIND_FAM]["rows"] if x["tag"] == "fused_attn4_m_pb1")
    print("FAILING-CASE TC1-am26-engagement (reducer):", v, "--", str(why)[-120:])
    assert v == "VOID" and "gnf4 counted no prebound launch" in str(why), (v, why)
    assert pp(pb1_counts=(4000, 0)) == {"P53": "UNTESTED", "P54": "UNTESTED", "P55": "UNTESTED"}
    assert PF(_prebind_set(pb0_counts=(5, 0)))[PREBIND_FAM]["verdicts"][("e4b", "fused_attn4_m_pb0")] == "VOID"
    assert PF(_prebind_set(requested=(True, False)))[PREBIND_FAM]["verdicts"][("e4b", "fused_attn4_shipped_pb1")] == "VOID"
    assert "no prebind_ab record" in prebind_ab_why("fused_attn4_m_pb1", {})
    assert PF(_prebind_set(record=False))[PREBIND_FAM]["verdicts"][("e4b", "fused_attn4_m_pb0")] == "VOID"
    cases += 1
    # 82. TC1 amendment 28 (qwen3dqab / mixtraldqab): Qwen3 1.013 / -1.35 GB and Mixtral 1.014 / -2.10 GB HELD, P58 HELD; each leg FALSIFIED in turn
    QF = lambda q=None, m=None: {QDQ_FAM: reduce_family(QDQ_FAM, q if q is not None else _dq_set(QDQ_FAM), {}, 20),
                                 MDQ_FAM: reduce_family(MDQ_FAM, m if m is not None else _dq_set(MDQ_FAM), {}, 20)}
    QB = QF()
    for fam in DQ_FAMS:
        assert [(x["fw"], x["tag"]) for x in QB[fam]["rows"]] == EXPECTED[fam]
        assert all(x["verdict"] == "VALID" for x in QB[fam]["rows"]), (fam, [(x["tag"], x["verdict"], x["why"]) for x in QB[fam]["rows"]])
    pq = lambda **kw: {p: v for p, _, v, _ in score_dqab(QF(**kw))}
    assert pq() == {"P56": "HELD", "P57": "HELD", "P58": "HELD"}, score_dqab(QB)
    assert "dq1 / dq0 1.013 [" in score_dqab(QB)[0][3] and "drop 1.350 vs [1.25, 1.45]" in score_dqab(QB)[0][3], score_dqab(QB)[0][3]
    assert pq(q=_dq_set(QDQ_FAM, d1=(4.10, 4.12))) == {"P56": "FALSIFIED", "P57": "HELD", "P58": "HELD"}          # 1.051: costs too much
    assert pq(q=_dq_set(QDQ_FAM, peaks=(27.16, 26.66))) == {"P56": "FALSIFIED", "P57": "HELD", "P58": "HELD"}     # drop 0.50 GB
    assert pq(m=_dq_set(MDQ_FAM, peaks=(31.07, 29.57))) == {"P56": "HELD", "P57": "FALSIFIED", "P58": "HELD"}     # drop 1.50 GB
    assert pq(m=_dq_set(MDQ_FAM, held_shift=0.008)) == {"P56": "HELD", "P57": "HELD", "P58": "FALSIFIED"}
    cases += 1
    # 83. FAILING CASES: a dq1 side that kept the fp32 absmax, or double-quantized only some layers, is VOID; a missing family leaves its leg UNTESTED
    RV = QF(q=_dq_set(QDQ_FAM, flags=(False, False)))
    v = RV[QDQ_FAM]["verdicts"][("e4b", "fused_attn4_m_dq1")]
    why = next(x["why"] for x in RV[QDQ_FAM]["rows"] if x["tag"] == "fused_attn4_m_dq1")
    print("FAILING-CASE TC1-am28-engagement (reducer):", v, "--", str(why)[-120:])
    assert v == "VOID" and "absmax_dq true" in str(why), (v, why)
    assert pq(q=_dq_set(QDQ_FAM, flags=(False, False))) == {"P56": "UNTESTED", "P57": "HELD", "P58": "UNTESTED"}
    assert QF(q=_dq_set(QDQ_FAM, modules=40))[QDQ_FAM]["verdicts"][("e4b", "fused_attn4_m_dq1")] == "VOID"
    assert QF(m=_dq_set(MDQ_FAM, flags=(True, True)))[MDQ_FAM]["verdicts"][("e4b", "fused_attn4_m_dq0")] == "VOID"
    only = {p: v for p, _, v, _ in score_dqab({QDQ_FAM: QB[QDQ_FAM]})}
    assert only == {"P56": "HELD", "P57": "UNTESTED", "P58": "UNTESTED"}, only
    cases += 1
    # 84. TC1 amendment 32 (qwen3tritonab): matched 0.885 and shipped 0.935 HELD; no gain, a held-out gap FALSIFY; the wrong Triton or the
    #     prebound launches on -> VOID and UNTESTED
    TF = lambda R: {TRITON_FAM: reduce_family(TRITON_FAM, R, {}, 20)}
    RT = TF(_triton_set())
    assert [(x["fw"], x["tag"]) for x in RT[TRITON_FAM]["rows"]] == EXPECTED[TRITON_FAM]
    assert all(x["verdict"] == "VALID" for x in RT[TRITON_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RT[TRITON_FAM]["rows"]]
    pt = lambda **kw: {p: v for p, _, v, _ in score_tritonab(TF(_triton_set(**kw)))}
    assert pt() == {"P59": "HELD", "P60": "HELD", "P61": "HELD"}, score_tritonab(RT)
    assert "tr1 / tr0 0.885 [" in score_tritonab(RT)[0][3], score_tritonab(RT)[0][3]
    assert pt(match=((3.90, 3.92), (3.86, 3.88))) == {"P59": "FALSIFIED", "P60": "HELD", "P61": "HELD"}        # 0.990: Triton is not the gain
    assert pt(held_shift=0.008) == {"P59": "HELD", "P60": "HELD", "P61": "FALSIFIED"}
    RV = TF(_triton_set(triton_v=("3.4.0", "3.4.0")))
    v = RV[TRITON_FAM]["verdicts"][("e4b", "fused_attn4_m_tr1")]
    why = next(x["why"] for x in RV[TRITON_FAM]["rows"] if x["tag"] == "fused_attn4_m_tr1")
    print("FAILING-CASE TC1-am32-engagement (reducer):", v, "--", str(why)[-120:])
    assert v == "VOID" and "env.triton 3.4.0 is not 3.7.*" in str(why), (v, why)
    assert pt(triton_v=("3.4.0", "3.4.0")) == {"P59": "UNTESTED", "P60": "UNTESTED", "P61": "UNTESTED"}
    assert TF(_triton_set(prebind=(True, False)))[TRITON_FAM]["verdicts"][("e4b", "fused_attn4_m_tr0")] == "VOID"
    cases += 1
    # 85. TC1 amendment 33: the load gate's LOADGATE lines render; none, nothing
    gd = tempfile.mkdtemp(prefix="tc1_loadgate_selftest_")
    open(os.path.join(gd, "summary.txt"), "w").write("BOX x\nLOADGATE qwen3samestack/e4b/fused_attn4_m attempt 0 load1_median 12.0 gate 6.0 status ok over 1\n"
                                                      "LOADGATE qwen3samestack/e4b/fused_attn4_m attempt 0 VOID (host load1 median 12.0 > 6.0): re-run 1 of 2\n"
                                                      "LOADGATE qwen3samestack/e4b/fused_attn4_m attempt 1 load1_median 3.1 gate 6.0 status ok over 0\n")
    lg = loadgate_lines(gd)
    assert lg[0].endswith("1 draw(s) voided for host load and run again") and len(lg) == 6 and "attempt 1 load1_median 3.1" in lg[-1], lg
    open(os.path.join(gd, "summary.txt"), "w").write("BOX x\n")
    assert loadgate_lines(gd) == [] and loadgate_lines(None) == []
    cases += 1
    # 86. TC1 amendment 34 (qwen3envsplit): transformers 0.974, torch 0.905, whole 0.882 HELD; a held-out gap FALSIFIES P65; the wrong
    #     transformers or the prebound launches on -> VOID and UNTESTED
    EF = lambda R: {ENVSPLIT_FAM: reduce_family(ENVSPLIT_FAM, R, {}, 20)}
    RE = EF(_envsplit_set())
    assert [(x["fw"], x["tag"]) for x in RE[ENVSPLIT_FAM]["rows"]] == EXPECTED[ENVSPLIT_FAM]
    assert all(x["verdict"] == "VALID" for x in RE[ENVSPLIT_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RE[ENVSPLIT_FAM]["rows"]]
    pe = lambda **kw: {p: v for p, _, v, _ in score_envsplit(EF(_envsplit_set(**kw)))}
    assert pe() == {"P62": "HELD", "P63": "HELD", "P64": "HELD", "P65": "HELD"}, score_envsplit(RE)
    assert "e1 / e0 0.974 [" in score_envsplit(RE)[0][3] and "e2 / e0 0.882 [" in score_envsplit(RE)[2][3], score_envsplit(RE)
    assert pe(e1=(3.44, 3.46))["P62"] == "FALSIFIED" and pe(e1=(3.44, 3.46))["P63"] == "FALSIFIED"   # transformers is all of it
    assert pe(held=(0.0, 0.008))["P65"] == "FALSIFIED"
    RV = EF(_envsplit_set(envs={"e0": ("2.8.0+cu128", "5.18.0"), "e1": ("2.8.0+cu128", "5.18.0"), "e2": ("2.12.1+cu130", "5.5.0")}))
    v = RV[ENVSPLIT_FAM]["verdicts"][("e4b", "fused_attn4_m_e1")]
    why = next(x["why"] for x in RV[ENVSPLIT_FAM]["rows"] if x["tag"] == "fused_attn4_m_e1")
    print("FAILING-CASE TC1-am34-engagement (reducer):", v, "--", str(why)[-120:])
    assert v == "VOID" and "env.transformers 5.18.0 is not 5.5.*" in str(why), (v, why)
    assert pe(envs={"e0": ("2.8.0+cu128", "5.18.0"), "e1": ("2.8.0+cu128", "5.18.0"), "e2": ("2.12.1+cu130", "5.5.0")}) == \
        {"P62": "UNTESTED", "P63": "UNTESTED", "P64": "HELD", "P65": "UNTESTED"}
    assert EF(_envsplit_set(prebind=True))[ENVSPLIT_FAM]["verdicts"][("e4b", "fused_attn4_m_e0")] == "VOID"
    cases += 1
    # 87. TC1 amendment 35 (qwen3prebind37): amendment 26's A/B under triton 3.7.1 -- shipped 0.981 and matched 0.985 HELD; a slower pb1, a
    #     larger gain than registered and a held-out gap each FALSIFY; an arm that ran triton 3.4 or torch 2.8 is not engaged (VOID -> UNTESTED)
    P7 = lambda R: {PREBIND37_FAM: reduce_family(PREBIND37_FAM, R, {}, 20)}
    s37 = lambda ship=((3.10, 3.12), (3.04, 3.06)), match=((3.90, 3.92), (3.84, 3.86)), **kw: _prebind_set(ship=ship, match=match, fam=PREBIND37_FAM, **kw)
    R7 = P7(s37())
    assert [(x["fw"], x["tag"]) for x in R7[PREBIND37_FAM]["rows"]] == EXPECTED[PREBIND37_FAM]
    assert all(x["verdict"] == "VALID" for x in R7[PREBIND37_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in R7[PREBIND37_FAM]["rows"]]
    p7 = lambda **kw: {p: v for p, _, v, _ in score_prebindab(P7(s37(**kw)), PREBIND37_FAM)}
    assert p7() == {"P66": "HELD", "P67": "HELD", "P68": "HELD"}, score_prebindab(R7, PREBIND37_FAM)
    assert "pb1 / pb0 0.981 [" in score_prebindab(R7, PREBIND37_FAM)[0][3] and "triton 3.7.1" in score_prebindab(R7, PREBIND37_FAM)[0][3]
    assert p7(ship=((3.10, 3.12), (3.14, 3.16))) == {"P66": "FALSIFIED", "P67": "HELD", "P68": "HELD"}      # 1.013: slower with the flags on
    assert p7(match=((3.90, 3.92), (3.70, 3.72))) == {"P66": "HELD", "P67": "FALSIFIED", "P68": "HELD"}      # 0.949: more than registered
    assert p7(held_shift=0.008) == {"P66": "HELD", "P67": "HELD", "P68": "FALSIFIED"}
    for kw, frag in (({"triton": "3.4.0"}, "triton 3.4.0 is not 3.7*"), ({"torch": "2.8.0+cu128"}, "env.torch 2.8.0+cu128 is not 2.12*")):
        RV7 = P7(s37(**kw))
        why = next(x["why"] for x in RV7[PREBIND37_FAM]["rows"] if x["tag"] == "fused_attn4_m_pb0")
        assert RV7[PREBIND37_FAM]["verdicts"][("e4b", "fused_attn4_m_pb0")] == "VOID" and frag in str(why), why
        assert p7(**kw) == {"P66": "UNTESTED", "P67": "UNTESTED", "P68": "UNTESTED"}
    print("FAILING-CASE TC1-am35-engagement (reducer):", "VOID", "--", str(why)[-120:])
    assert pp() == {"P53": "HELD", "P54": "HELD", "P55": "HELD"}           # amendment 26's box reads as before (triton 3.4, no torch check)
    cases += 1
    # 88. TC1 amendment 36 (qwen3compactab): matched peak -0.70 GB, matched 0.994 and shipped 0.997 HELD; too small a drop, a slower cd1 and a
    #     held-out gap each FALSIFY; a side that resolved the wrong flag, kept layers, ran no padded call or ran torch 2.8 is VOID
    CF = lambda R: {COMPACT_FAM: reduce_family(COMPACT_FAM, R, {}, 20)}
    RC = CF(_compact_set())
    assert [(x["fw"], x["tag"]) for x in RC[COMPACT_FAM]["rows"]] == EXPECTED[COMPACT_FAM]
    assert all(x["verdict"] == "VALID" for x in RC[COMPACT_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RC[COMPACT_FAM]["rows"]]
    pc = lambda **kw: {p: v for p, _, v, _ in score_compactab(CF(_compact_set(**kw)))}
    assert pc() == {"P69": "HELD", "P70": "HELD", "P71": "HELD", "P72": "HELD"}, score_compactab(RC)
    assert "drop 0.700 vs [0.3, 2.0]" in score_compactab(RC)[0][3] and "cd1 / cd0 0.994 [" in score_compactab(RC)[1][3], score_compactab(RC)
    assert pc(peaks=(27.50, 27.40)) == {"P69": "FALSIFIED", "P70": "HELD", "P71": "HELD", "P72": "HELD"}            # 0.10 GB: too small a drop
    assert pc(match=((3.50, 3.52), (3.62, 3.64))) == {"P69": "HELD", "P70": "FALSIFIED", "P71": "HELD", "P72": "HELD"}  # 1.034: slower
    assert pc(held_shift=0.008) == {"P69": "HELD", "P70": "HELD", "P71": "HELD", "P72": "FALSIFIED"}
    for kw, frag in (({"flags": ("0", "0")}, "gnf4_compact_delta 1"), ({"kept": 4}, "layers_kept 0"), ({"padded": 0}, "the padded LoRA path ran"),
                     ({"torch": "2.8.0+cu128"}, "env.torch 2.12*"), ({"record": False}, "no keep_ab record")):
        RV = CF(_compact_set(**kw))
        tag = "fused_attn4_m_cd1"
        why = next(x["why"] for x in RV[COMPACT_FAM]["rows"] if x["tag"] == tag)
        assert RV[COMPACT_FAM]["verdicts"][("e4b", tag)] == "VOID" and frag in str(why), (kw, why)
    assert pc(flags=("0", "0")) == {"P69": "UNTESTED", "P70": "UNTESTED", "P71": "UNTESTED", "P72": "UNTESTED"}
    print("FAILING-CASE TC1-am36-engagement (reducer):", "VOID", "--", str(why)[-120:])
    cases += 1
    # 89. TC1c amendment 9 (qwen3samestackh100): the H100 same-stack position 1.181 and environment 0.917 HELD, the route on every e4b arm;
    #     Unsloth faster on one stack FALSIFIES P27 (still a stable reading), an e4b arm off the route FALSIFIES P29, no gain FALSIFIES P28
    HF_ = lambda R: {SAMESTACK_H100_FAM: reduce_family(SAMESTACK_H100_FAM, R, {}, 20)}
    hset = lambda routes=None, **kw: _samestack_set(e=kw.pop("e", (2.20, 2.22)), t28=kw.pop("t28", (2.40, 2.42)), u=kw.pop("u", (2.60, 2.62)),
                                                    fam=SAMESTACK_H100_FAM, routes=routes if routes is not None else {}, **kw)
    RH = HF_(hset())
    assert [(x["fw"], x["tag"]) for x in RH[SAMESTACK_H100_FAM]["rows"]] == EXPECTED[SAMESTACK_H100_FAM]
    ph = lambda **kw: {p: v for p, _, v, _ in score_samestack(HF_(hset(**kw)), SAMESTACK_H100_FAM)}
    assert ph() == {"P27": "HELD", "P28": "HELD", "P29": "HELD"}, score_samestack(RH, SAMESTACK_H100_FAM)
    assert "Unsloth/e4b on one stack 1.181 [" in score_samestack(RH, SAMESTACK_H100_FAM)[0][3]
    assert "0.900 on an RTX 5090" in score_samestack(RH, SAMESTACK_H100_FAM)[1][3]
    assert ph(u=(2.00, 2.02)) == {"P27": "FALSIFIED", "P28": "HELD", "P29": "HELD"}             # 0.910: Unsloth faster on one stack
    assert ph(e=(2.50, 2.52)) == {"P27": "HELD", "P28": "FALSIFIED", "P29": "HELD"}             # 1.041: slower in Unsloth's environment
    assert ph(routes={"fused_attn4_m_t28": "fused"}) == {"P27": "HELD", "P28": "HELD", "P29": "FALSIFIED"}
    assert "fused_attn4_m_t28: fused" in score_samestack(HF_(hset(routes={"fused_attn4_m_t28": "fused"})), SAMESTACK_H100_FAM)[2][3]
    assert ph(torch_v=("2.8.0+cu128", "2.8.0+cu128"))["P27"] == "UNTESTED"
    RS9 = hset()                                                     # the registered box skips e4b's reference: a NOT_RUN stub, no route_ab
    RS9[("e4b", "reference_attn4_m")] = {**_stub("e4b", "reference_attn4_m", "reference", "not_run", "skipped by TC1_SKIP"), "fam": SAMESTACK_H100_FAM}
    assert {p: v for p, _, v, _ in score_samestack(HF_(RS9), SAMESTACK_H100_FAM)} == {"P27": "HELD", "P28": "HELD", "P29": "HELD"}
    assert ps() == {"P50": "HELD", "P51": "HELD", "P52": "HELD"}                                # amendment 25's reading unchanged
    print("FAILING-CASE TC1c-am9-route (reducer):", "FALSIFIED", "-- P29 with the _t28 arm on the fused route")
    cases += 1
    # 90. TC1 amendment 37 (qwen3compactab2): the fixed backward -- matched 0.969 / shipped 0.970 with the peak 0.10 GB lower HELD; a peak
    #     above the default's by more than 0.05 GB, a step slower than 0.99, a held-out gap each FALSIFY; engagement as amendment 36
    C2 = lambda R: {COMPACT2_FAM: reduce_family(COMPACT2_FAM, R, {}, 20)}
    c2set = lambda **kw: _compact_set(**{"match": ((3.50, 3.52), (3.39, 3.41)), "ship": ((3.00, 3.02), (2.91, 2.93)), "peaks": (27.50, 27.40),
                                         "fam": COMPACT2_FAM, **kw})
    RC2 = C2(c2set())
    assert [(x["fw"], x["tag"]) for x in RC2[COMPACT2_FAM]["rows"]] == EXPECTED[COMPACT2_FAM]
    p2 = lambda **kw: {p: v for p, _, v, _ in score_compactab(C2(c2set(**kw)), COMPACT2_FAM)}
    assert p2() == {"P73": "HELD", "P74": "HELD", "P75": "HELD", "P76": "HELD"}, score_compactab(RC2, COMPACT2_FAM)
    assert "drop 0.100 vs [-0.05, 0.5]" in score_compactab(RC2, COMPACT2_FAM)[0][3], score_compactab(RC2, COMPACT2_FAM)[0][3]
    assert p2(peaks=(27.50, 27.72))["P73"] == "FALSIFIED"                       # amendment 36's +0.22 GB
    assert p2(match=((3.50, 3.52), (3.49, 3.51)))["P74"] == "FALSIFIED"         # 0.997: no gain
    assert p2(held_shift=0.008)["P76"] == "FALSIFIED"
    assert p2(flags=("0", "0")) == {"P73": "UNTESTED", "P74": "UNTESTED", "P75": "UNTESTED", "P76": "UNTESTED"}
    assert pc() == {"P69": "HELD", "P70": "HELD", "P71": "HELD", "P72": "HELD"}  # amendment 36's reading unchanged
    cases += 1
    # 91. TC2 amendment 9 (mixtralsamestack): Unsloth/e4b on one stack 1.050 and the environment 0.950 HELD, the dense route on every fused e4b
    #     arm; Unsloth faster on one stack FALSIFIES P29 (still a stable reading); an arm on the fused route FALSIFIES P31
    MF_ = lambda R: {SAMESTACK_MIXTRAL_FAM: reduce_family(SAMESTACK_MIXTRAL_FAM, R, {}, 20)}
    def mset(routes=None, e=(3.40, 3.42), t28=(3.58, 3.60), u=(3.57, 3.59), torch_v=("2.12.1+cu130", "2.8.0+cu128")):
        """Mixtral's receipts at TC2's pin (TC2's helper), e4b's same-stack pair on torch_v[0], its _t28 pair on torch_v[1], Unsloth's pair;
        every fused e4b arm records the dense route (`routes` overrides a tag's), as grouped-nf4-gemm's auto takes it off sm_90."""
        R, rt = {}, {"fused_attn4_m": "dense", "fused_attn4_m_d2": "dense", "fused_attn4_m_t28": "dense", "fused_attn4_m_t28_d2": "dense", **(routes or {})}
        for i, sfx in enumerate(("", "_d2")):
            for fw, tag, arm, ss, tv in (("e4b", "fused_attn4_m" + sfx, "fused", e[i], torch_v[0]), ("e4b", SAMESTACK_T28 + sfx, "fused", t28[i], torch_v[1]),
                                         ("unsloth", "ckpt_unsloth_m" + sfx, "unsloth", u[i], None)):
                r = _tc2_receipt("mixtral", fw, tag, arm, s=ss)
                if tv:
                    r["env"]["torch"] = tv
                if fw == "e4b":
                    dense = rt[tag] == "dense"                   # as box M records it: the mode stays `fused`, the counts say dense
                    r["route_ab"] = {"gnf4_train_gemm": "fused", "gnf4_train_gemm_env": None, "gnf4_has_route": True,
                                     "stats": {"fwd": 0 if dense else 11264, "dgrad": 0 if dense else 5120,
                                               "dense_fwd": 11264 if dense else 0, "dense_dgrad": 5120 if dense else 0}}
                r["fam"] = SAMESTACK_MIXTRAL_FAM
                R[(fw, tag)] = r
        R[("e4b", "reference_attn4_m")] = {**_stub("e4b", "reference_attn4_m", "reference", "not_run", "skipped by TC1_SKIP"), "fam": SAMESTACK_MIXTRAL_FAM}
        return R
    RM = MF_(mset())
    assert [(x["fw"], x["tag"]) for x in RM[SAMESTACK_MIXTRAL_FAM]["rows"]] == EXPECTED[SAMESTACK_MIXTRAL_FAM]
    pm = lambda **kw: {p: v for p, _, v, _ in score_samestack(MF_(mset(**kw)), SAMESTACK_MIXTRAL_FAM)}
    assert pm() == {"P29": "HELD", "P30": "HELD", "P31": "HELD"}, (score_samestack(RM, SAMESTACK_MIXTRAL_FAM),
                                                                  [(x["tag"], x["verdict"], x["why"]) for x in RM[SAMESTACK_MIXTRAL_FAM]["rows"]])
    assert "route dense with GNF4_TRAIN_GEMM unset on all" in score_samestack(RM, SAMESTACK_MIXTRAL_FAM)[2][3]
    assert pm(u=(2.70, 2.72))["P29"] == "FALSIFIED"                                      # 0.795: Unsloth well ahead on one stack
    assert pm(routes={"fused_attn4_m_d2": "fused"})["P31"] == "FALSIFIED"
    assert ph() == {"P27": "HELD", "P28": "HELD", "P29": "HELD"} and ps() == {"P50": "HELD", "P51": "HELD", "P52": "HELD"}
    cases += 1
    # 92. TC1 amendment 38 (qwen3compactab3 + mixtralcompactab): one-sided bands -- Qwen3 0.967 / 0.948 with the peak 0.29 GB lower HELD (the
    #     shipped 0.948 that FALSIFIED amendment 37's P75 holds here); a step slower than 0.99, a peak 0.10 GB higher each FALSIFY; Mixtral
    #     matched-only with its shipped arms skipped: 1.000 HELD (level is allowed), 1.03 FALSIFIES, a peak 0.10 GB higher FALSIFIES
    C3 = lambda R: {COMPACT3_FAM: reduce_family(COMPACT3_FAM, R, {}, 20)}
    c3set = lambda **kw: _compact_set(**{"match": ((3.90, 3.92), (3.77, 3.79)), "ship": ((3.00, 3.02), (2.84, 2.86)), "peaks": (27.48, 27.19),
                                         "fam": COMPACT3_FAM, **kw})
    p3 = lambda **kw: {p: v for p, _, v, _ in score_compactab(C3(c3set(**kw)), COMPACT3_FAM)}
    assert p3() == {"P79": "HELD", "P77": "HELD", "P78": "HELD", "P82": "HELD"}, score_compactab(C3(c3set()), COMPACT3_FAM)
    assert p3(match=((3.90, 3.92), (3.89, 3.91)))["P77"] == "FALSIFIED"        # 0.997: no gain
    assert p3(peaks=(27.48, 27.58))["P79"] == "FALSIFIED"                      # +0.10 GB
    def mcset(match=((3.56, 3.58), (3.56, 3.58)), peaks=(31.07, 31.00), **kw):
        R = _compact_set(match=match, peaks=peaks, fam=MCOMPACT_FAM, **kw)
        out_ = {}
        for (fw, tag), r in R.items():
            if "shipped" in tag:
                out_[(fw, tag)] = {**_stub("e4b", tag, "fused", "not_run", "skipped by TC1_SKIP"), "fam": MCOMPACT_FAM}
                continue
            m = _tc2_receipt("mixtral", "e4b", tag, "fused", s=r["s_per_step_median_11plus"] if "s_per_step_median_11plus" in r else r["s_per_step"],
                             heldout_n=r["heldout_final"] if "heldout_final" in r else 1.8)
            for k in ("peak_vram_gb", "keep_ab", "lean_ab"):
                if k in r:
                    m[k] = r[k]
            m["env"]["torch"] = r["env"]["torch"]
            m["fam"] = MCOMPACT_FAM
            out_[(fw, tag)] = m
        return out_
    MC = lambda R: {MCOMPACT_FAM: reduce_family(MCOMPACT_FAM, R, {}, 20)}
    RMC = MC(mcset())
    assert all(x["verdict"] in ("VALID", "NOT_RUN") for x in RMC[MCOMPACT_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RMC[MCOMPACT_FAM]["rows"]]
    pmc = lambda **kw: {p: v for p, _, v, _ in score_compactab(MC(mcset(**kw)), MCOMPACT_FAM)}
    assert pmc() == {"P81": "HELD", "P80": "HELD", "P83": "HELD"}, score_compactab(RMC, MCOMPACT_FAM)
    assert pmc(match=((3.56, 3.58), (3.67, 3.69)))["P80"] == "FALSIFIED"      # 1.031: slower on Mixtral
    assert pmc(peaks=(31.07, 31.17))["P81"] == "FALSIFIED"                    # +0.10 GB on a 32 GB card
    assert p2() == {"P73": "HELD", "P74": "HELD", "P75": "HELD", "P76": "HELD"}  # amendment 37's reading unchanged
    cases += 1
    # 93. TC1 amendment 39 (qwen3samestack4k): amendment 25's family on packed 4,096-token rows -- Unsloth/e4b 1.397 and the environment
    #     0.898 HELD, every e4b arm that ran resident and VALID (P86 HELD, the skipped reference not read); each FALSIFIED in turn (an e4b OOM
    #     for P86); the field recipe's receipts under this token VOID (read against its own fixture); amendment 25's reading unchanged
    kd = tempfile.mkdtemp(prefix="tc1_packed4k_selftest_")
    for (fw, tag), r in _packed4k_set().items():
        json.dump(r, open(os.path.join(kd, f"{PACKED4K_FAM}_{fw}_{tag}.json"), "w"))
    FK = reduce_dir(kd, 30)
    assert set(FK) == {PACKED4K_FAM}, set(FK)                        # the file names parse to the family (the token ends in digits + k)
    RK = FK[PACKED4K_FAM]
    assert [(x["fw"], x["tag"]) for x in RK["rows"]] == EXPECTED[PACKED4K_FAM] == EXPECTED[SAMESTACK_FAM]
    assert all(x["verdict"] == "VALID" for x in RK["rows"] if x["tag"] != "reference_attn4_m"), [(x["tag"], x["verdict"], x["why"]) for x in RK["rows"]]
    assert RK["verdicts"][("e4b", "reference_attn4_m")] == "NOT_RUN"
    assert {p: v for p, _, v, _ in score_packed4k(FK)} == {"P84": "HELD", "P85": "HELD", "P86": "HELD"}, score_packed4k(FK)
    sk = score_packed4k(FK)
    assert "Unsloth/e4b on one stack 1.397 [" in sk[0][3] and "venv-unsloth / venv-e4b 0.898 [" in sk[1][3] and "0.900 at the field recipe" in sk[1][3], sk[:2]
    assert "all 4 e4b arms that ran completed resident and VALID" in sk[2][3], sk[2][3]
    text = render(FK, kd)
    for needle in ("## Predictions P84 / P85 / P86 (TC1-PREREG amendment 39: the packed 4,096-token regime with both frameworks on one stack",
                   "| P84 | qwen3samestack4k | **HELD** |", "| P85 | qwen3samestack4k | **HELD** |", "| P86 | qwen3samestack4k | **HELD** |",
                   "seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4"):
        assert needle in text, needle
    def KF_(R):
        return {PACKED4K_FAM: reduce_family(PACKED4K_FAM, R, {}, 30)}

    def pk(**kw):
        return {p: v for p, _, v, _ in score_packed4k(KF_(_packed4k_set(**kw)))}
    assert pk(u=(25.0, 25.2)) == {"P84": "FALSIFIED", "P85": "HELD", "P86": "HELD"}             # 1.780: Unsloth further behind than the band
    assert pk(u=(10.0, 10.2)) == {"P84": "FALSIFIED", "P85": "HELD", "P86": "HELD"}             # 0.716: Unsloth ahead on the packed rows
    assert pk(t28=(13.0, 13.2)) == {"P84": "HELD", "P85": "FALSIFIED", "P86": "HELD"}           # 1.076: no gain from Unsloth's environment
    assert pk(oom=("fused_attn4_m_t28_d2",)) == {"P84": "HELD", "P85": "UNTESTED", "P86": "FALSIFIED"}
    assert "e4b OOM on fused_attn4_m_t28_d2 (OOM at step 1" in score_packed4k(KF_(_packed4k_set(oom=("fused_attn4_m_t28_d2",))))[2][3]
    assert pk(oom=("ckpt_unsloth_m_d2",)) == {"P84": "UNTESTED", "P85": "HELD", "P86": "HELD"}  # an Unsloth OOM is not P86's
    # the fixture predicate, every framework: unpacked tokens, the field seq, the field micro-batch, padded steps -- each VOIDs its row; an e4b
    # VOID leaves P86 UNTESTED (not every e4b arm that ran is VALID, none OOMed), an e4b arm VALID under offload too
    for tag, ov, frag in (("fused_attn4_m_t28", {"tokens": {"sha256": "p" * 64, "pack": False}}, "the tokens file is not packed"),
                          ("fused_attn4_m_t28", {"seq": 2048}, "seq 2048 != 4096"),
                          ("ckpt_unsloth_m", {"micro_batch": 2, "tokens_per_step": [32768] * 30}, "micro-batch 2 x accum 4 != 1 x 4"),
                          ("fused_attn4_m_d2", {"tokens_per_step": [16200] * 30, "tokens_padded_per_step": [184] * 30}, "padded tokens on 30 step(s)"),
                          ("fused_attn4_m", {"arm_facts": {"free_outputs": False}}, "arm_facts.free_outputs False, not True")):
        RV = KF_(_packed4k_set(over={tag: ov}))
        fwk = "unsloth" if tag.startswith("ckpt") else "e4b"
        why = next(x["why"] for x in RV[PACKED4K_FAM]["rows"] if x["tag"] == tag)
        assert RV[PACKED4K_FAM]["verdicts"][(fwk, tag)] == "VOID" and "packed regime not engaged" in why and frag in why, (tag, why)
        want86 = "HELD" if fwk == "unsloth" else "UNTESTED"
        assert {p: v for p, _, v, _ in score_packed4k(RV)}["P86"] == want86, (tag, score_packed4k(RV))
    assert pk(over={"fused_attn4_m_d2": {"offload": True}})["P86"] == "UNTESTED"
    FR = _samestack_set(fam=PACKED4K_FAM)                            # the field recipe's receipts (seq 2048, micro-batch 2, no pack) under this token
    RF = KF_(FR)
    assert all(x["verdict"] == "VOID" and "packed regime not engaged" in x["why"] for x in RF[PACKED4K_FAM]["rows"]), [(x["tag"], x["verdict"]) for x in RF[PACKED4K_FAM]["rows"]]
    assert {p: v for p, _, v, _ in score_packed4k(RF)} == {"P84": "UNTESTED", "P85": "UNTESTED", "P86": "UNTESTED"}
    assert ps() == {"P50": "HELD", "P51": "HELD", "P52": "HELD"} and score_packed4k({}) == []    # amendment 25's reading unchanged
    print("FAILING-CASE TC1-am39-oom (reducer):", "FALSIFIED", "-- P86 with an e4b arm OOM on the packed rows")
    cases += 1
    # 94. TC1 amendment 40 (qwen3samestack4kce): amendment 39's box with e4b's chunked LM loss -- P87 / P88 / P89 HELD on the packed rows; an
    #     e4b OOM FALSIFIES P89; an e4b arm without the chunked record, with the env unset, with no chunked call or a run-time fallback is VOID
    CE_ = lambda R: {PACKED4KCE_FAM: reduce_family(PACKED4KCE_FAM, R, {}, 30)}
    pce = lambda **kw: {p: v for p, _, v, _ in score_packed4k(CE_(_packed4k_set(fam=PACKED4KCE_FAM, **kw)), PACKED4KCE_FAM)}
    RCE = CE_(_packed4k_set(fam=PACKED4KCE_FAM))
    assert all(x["verdict"] in ("VALID", "NOT_RUN") for x in RCE[PACKED4KCE_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RCE[PACKED4KCE_FAM]["rows"]]
    assert pce() == {"P87": "HELD", "P88": "HELD", "P89": "HELD"}, score_packed4k(RCE, PACKED4KCE_FAM)
    assert pce(oom=("fused_attn4_m_d2",))["P89"] == "FALSIFIED"
    for ch, frag in (({"env": "0", "e4b_has_chunked_lm_loss": True, "chunked_calls": 120, "runtime_refusals": 0}, "E4B_CHUNKED_LM_LOSS set"),
                     ({"env": "1", "e4b_has_chunked_lm_loss": True, "chunked_calls": 0, "runtime_refusals": 0}, "chunked_calls > 0"),
                     ({"env": "1", "e4b_has_chunked_lm_loss": True, "chunked_calls": 120, "runtime_refusals": 2}, "runtime_refusals 0"),
                     ({"env": "1", "e4b_has_chunked_lm_loss": False}, "e4b has the chunked loss")):
        RV = CE_(_packed4k_set(fam=PACKED4KCE_FAM, chunked=ch))
        why = next(x["why"] for x in RV[PACKED4KCE_FAM]["rows"] if x["tag"] == "fused_attn4_m")
        assert RV[PACKED4KCE_FAM]["verdicts"][("e4b", "fused_attn4_m")] == "VOID" and frag in str(why), (ch, why)
    assert "no chunked_lm_loss record" in chunked_lm_loss_why({})
    assert {p: v for p, _, v, _ in score_packed4k(KF_(_packed4k_set()))} == {"P84": "HELD", "P85": "HELD", "P86": "HELD"}   # amendment 39 unchanged
    cases += 1
    # 95. TC1 amendment 41 (qwen3chunkab): one-sided -- matched 1.003 / shipped 1.003 with the peak 0.20 GB lower HELD; a step slower than 1.01,
    #     a peak 0.10 GB higher, a held-out gap each FALSIFY; a ce1 side with no chunked call, a ce0 side with chunked calls, torch 2.8 -> VOID
    CK = lambda R: {CHUNKAB_FAM: reduce_family(CHUNKAB_FAM, R, {}, 20)}
    RCK = CK(_chunkab_set())
    assert [(x["fw"], x["tag"]) for x in RCK[CHUNKAB_FAM]["rows"]] == EXPECTED[CHUNKAB_FAM]
    assert all(x["verdict"] == "VALID" for x in RCK[CHUNKAB_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RCK[CHUNKAB_FAM]["rows"]]
    pck = lambda **kw: {p: v for p, _, v, _ in score_compactab(CK(_chunkab_set(**kw)), CHUNKAB_FAM)}
    assert pck() == {"P92": "HELD", "P90": "HELD", "P91": "HELD", "P93": "HELD"}, score_compactab(RCK, CHUNKAB_FAM)
    assert "ce1 / ce0 1.003 [" in score_compactab(RCK, CHUNKAB_FAM)[1][3], score_compactab(RCK, CHUNKAB_FAM)
    assert pck(match=((3.50, 3.52), (3.57, 3.59)))["P90"] == "FALSIFIED"            # 1.020: slower than allowed
    assert pck(peaks=(27.50, 27.60))["P92"] == "FALSIFIED"                         # +0.10 GB
    assert pck(held_shift=0.008)["P93"] == "FALSIFIED"
    for kw, tag, frag in (({"ce1_calls": 0}, "fused_attn4_m_ce1", "chunked_calls > 0"), ({"ce0_calls": 7}, "fused_attn4_m_ce0", "the ce0 side made 7"),
                          ({"torch": "2.8.0+cu128"}, "fused_attn4_m_ce0", "is not 2.12*"), ({"refusals": 1}, "fused_attn4_m_ce1", "runtime_refusals 0"),
                          ({"record": False}, "fused_attn4_m_ce0", "no chunked_lm_loss record")):
        RV = CK(_chunkab_set(**kw))
        why = next(x["why"] for x in RV[CHUNKAB_FAM]["rows"] if x["tag"] == tag)
        assert RV[CHUNKAB_FAM]["verdicts"][("e4b", tag)] == "VOID" and frag in str(why), (kw, why)
    assert pc() == {"P69": "HELD", "P70": "HELD", "P71": "HELD", "P72": "HELD"}     # amendment 36's reading unchanged
    cases += 1
    # 96. TC1 amendment 42 (qwen3samestackh2): amendment 33's box on a second host -- 2.297 / 0.882 HELD; 3.02 and 0.995 each FALSIFY; no
    #     matched-set or route prediction is read
    H2 = lambda R: {SAMESTACK_HOST2_FAM: reduce_family(SAMESTACK_HOST2_FAM, R, {}, 20)}
    p42 = lambda **kw: {p: v for p, _, v, _ in score_samestack(H2(_samestack_set(fam=SAMESTACK_HOST2_FAM, **kw)), SAMESTACK_HOST2_FAM)}
    assert p42() == {"P94": "HELD", "P95": "HELD"}, score_samestack(H2(_samestack_set(fam=SAMESTACK_HOST2_FAM)), SAMESTACK_HOST2_FAM)
    assert p42(u=(10.40, 10.45))["P94"] == "FALSIFIED" and p42(e=(3.88, 3.90))["P95"] == "FALSIFIED"
    assert ps() == {"P50": "HELD", "P51": "HELD", "P52": "HELD"}
    cases += 1
    # 97. TC1 amendment 43 (qwen3samestack4kce2): the loop at 1.5 % of delta calls on every step reads VALID here and VOID under amendment 40's
    #     family (the same receipts); 8 % is VOID here too; P96 / P97 / P98 HELD at 1.40 / 0.89, a 1.75 FALSIFIES P96, an e4b OOM FALSIFIES P98
    C2_ = lambda R, fam=PACKED4KCE2_FAM: {fam: reduce_family(fam, R, {}, 30)}
    lp = dict(e=(11.2, 11.2), t28=(12.6, 12.6), u=(15.7, 15.7), loop_share=0.015)
    R43 = C2_(_packed4k_set(fam=PACKED4KCE2_FAM, **lp))
    assert all(x["verdict"] in ("VALID", "NOT_RUN") for x in R43[PACKED4KCE2_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in R43[PACKED4KCE2_FAM]["rows"]]
    R40 = C2_(_packed4k_set(fam=PACKED4KCE_FAM, **lp), PACKED4KCE_FAM)
    assert R40[PACKED4KCE_FAM]["verdicts"][("e4b", "fused_attn4_m")] == "VOID"                      # amendment 40's rule unchanged
    R8 = C2_(_packed4k_set(fam=PACKED4KCE2_FAM, **{**lp, "loop_share": 0.08}))
    assert R8[PACKED4KCE2_FAM]["verdicts"][("e4b", "fused_attn4_m")] == "VOID"
    p43 = lambda **kw: {p: v for p, _, v, _ in score_packed4k(C2_(_packed4k_set(fam=PACKED4KCE2_FAM, **{**lp, **kw})), PACKED4KCE2_FAM)}
    assert p43() == {"P96": "HELD", "P97": "HELD", "P98": "HELD"}, score_packed4k(R43, PACKED4KCE2_FAM)
    assert p43(u=(19.6, 19.6))["P96"] == "FALSIFIED" and p43(oom=("fused_attn4_m_d2",))["P98"] == "FALSIFIED"
    assert "`fused_attn4_m` 0.015" in render(R43, "x")
    cases += 1
    # 98. TC1 amendment 44 (qwen3chunkauto): auto vs off at the field recipe -- every arm VALID; P99..P103 HELD on a gate that never fired;
    #     a ca1 arm that chunked forwards FALSIFIES P99 (VALID still); a ca0 side patched, or a ca1 side not on `auto`, VOIDs the arm;
    #     1.03 on the shipped pair FALSIFIES P100 and leaves P101 HELD; a matched peak +0.10 GB FALSIFIES P102
    CA = lambda R: {CHUNKAUTO_FAM: reduce_family(CHUNKAUTO_FAM, R, {}, 20)}
    RCA = CA(_chunkauto_set())
    assert [(x["fw"], x["tag"]) for x in RCA[CHUNKAUTO_FAM]["rows"]] == EXPECTED[CHUNKAUTO_FAM]
    assert all(x["verdict"] == "VALID" for x in RCA[CHUNKAUTO_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in RCA[CHUNKAUTO_FAM]["rows"]]
    pca = lambda **kw: {p: v for p, _, v, _ in score_chunkauto_gate(CA(_chunkauto_set(**kw))) + score_compactab(CA(_chunkauto_set(**kw)), CHUNKAUTO_FAM)}
    assert pca() == {"P99": "HELD", "P102": "HELD", "P101": "HELD", "P100": "HELD", "P103": "HELD"}, pca()
    R99 = CA(_chunkauto_set(ca1_chunked=12, ca1_small=228))
    assert all(x["verdict"] == "VALID" for x in R99[CHUNKAUTO_FAM]["rows"]) and pca(ca1_chunked=12, ca1_small=228)["P99"] == "FALSIFIED"
    assert pca(ca1_chunked=0, ca1_small=236)["P99"] == "UNTESTED"                 # nothing chunked, a forward count off: the instrument, not the gate
    assert "gated 236, not 240" in score_chunkauto_gate(CA(_chunkauto_set(ca1_small=236)))[0][3]
    for kw, tag, frag in (({"ca0_patched": 1}, "fused_attn4_m_ca0", "patched 0"), ({"env": "1"}, "fused_attn4_m_ca1", "E4B_CHUNKED_LM_LOSS=auto"),
                          ({"torch": "2.8.0+cu128"}, "fused_attn4_m_ca0", "is not 2.12*")):
        RV = CA(_chunkauto_set(**kw))
        why = next(x["why"] for x in RV[CHUNKAUTO_FAM]["rows"] if x["tag"] == tag)
        assert RV[CHUNKAUTO_FAM]["verdicts"][("e4b", tag)] == "VOID" and frag in str(why), (kw, why)
    p_ship = pca(ship=((3.00, 3.02), (3.09, 3.11)))
    assert p_ship["P100"] == "FALSIFIED" and p_ship["P101"] == "HELD", p_ship
    assert pca(peaks=(27.50, 27.60))["P102"] == "FALSIFIED"
    assert "P99" in render(RCA, "x") and "amendment 44" in render(RCA, "x")
    cases += 1
    # 99. TC1 amendment 45 (qwen3ompab): every arm VALID; P104 HELD at e4b 0.914, P105 HELD at Unsloth 0.997, P106 HELD; e4b at 0.99 FALSIFIES
    #     P104; equal thread counts on both sides read UNTESTED (no contrast); a receipt whose torch pool differs from OMP_NUM_THREADS is VOID
    OM = lambda R: {OMPAB_FAM: reduce_family(OMPAB_FAM, R, {}, 20)}
    ROM = OM(_ompab_set())
    assert [(x["fw"], x["tag"]) for x in ROM[OMPAB_FAM]["rows"]] == EXPECTED[OMPAB_FAM]
    assert all(x["verdict"] == "VALID" for x in ROM[OMPAB_FAM]["rows"]), [(x["tag"], x["verdict"], x["why"]) for x in ROM[OMPAB_FAM]["rows"]]
    pom = lambda **kw: {p: v for p, _, v, _ in score_ompab(OM(_ompab_set(**kw)))}
    assert pom() == {"P104": "HELD", "P105": "HELD", "P106": "HELD"}, score_ompab(ROM)
    assert pom(e=((3.50, 3.52), (3.46, 3.48)))["P104"] == "FALSIFIED"
    assert pom(omp=(32, 32)) == {"P104": "UNTESTED", "P105": "UNTESTED", "P106": "UNTESTED"}
    RV = OM(_ompab_set(torch_threads=64))
    assert RV[OMPAB_FAM]["verdicts"][("e4b", "fused_attn4_m_om0")] == "VOID" and "torch ran 64" in str(next(x["why"] for x in RV[OMPAB_FAM]["rows"] if x["tag"] == "fused_attn4_m_om0"))
    assert pom(held_shift=0.01)["P106"] == "FALSIFIED"
    assert "P104" in render(ROM, "x") and "amendment 45" in render(ROM, "x")
    cases += 1
    # 100. TC1 amendment 46 (olmoedecab / qwen3decab): grouped-nf4-gemm's decoded route vs its fused kernels -- every arm VALID when it ran
    #     venv-e4b and took the route its tag names; P108 HELD at 0.851 (OLMoE), P109 HELD at 1.056 (Qwen3-30B-A3B), P110 / P111 HELD; the table
    def DEC(o=None, q=None):
        return {ODEC_FAM: reduce_family(ODEC_FAM, o if o is not None else _decoded_set(ODEC_FAM), {}, 60),
                QDEC_FAM: reduce_family(QDEC_FAM, q if q is not None else _decoded_set(QDEC_FAM), {}, 20)}

    def pdec(o=None, q=None):
        return {p: v for p, _, v, _ in score_decodedab(DEC(o, q))}
    DDEC = DEC()
    for fam in DEC_FAMS:
        assert [(x["fw"], x["tag"]) for x in DDEC[fam]["rows"]] == EXPECTED[fam]
        assert all(x["verdict"] == "VALID" for x in DDEC[fam]["rows"]), [(fam, x["tag"], x["verdict"], x["why"]) for x in DDEC[fam]["rows"]]
    PDEC = {p: (fam, v, ev) for p, fam, v, ev in score_decodedab(DDEC)}
    assert [p for p, _, _, _ in score_decodedab(DDEC)] == ["P108", "P109", "P110", "P111"]
    assert PDEC["P108"][:2] == (ODEC_FAM, "HELD") and "dec1 / dec0 0.851 [" in PDEC["P108"][2] and "vs <= 0.95" in PDEC["P108"][2], PDEC["P108"]
    assert PDEC["P109"][:2] == (QDEC_FAM, "HELD") and "dec1 / dec0 1.056 [" in PDEC["P109"][2] and '"decoded_dgrad": 1536' in PDEC["P109"][2], PDEC["P109"]
    assert PDEC["P110"][1] == "HELD" and PDEC["P111"][1] == "HELD" and "peak dec1 - dec0 GB +0.2000" in PDEC["P111"][2], PDEC["P111"]
    text = render(DDEC, "x")
    assert ("## Predictions P107 / P108 / P109 / P110 / P111" in text and "| P108 | olmoedecab | **HELD** |" in text
            and "| P109 | qwen3decab | **HELD** |" in text and "| P107 | decgate | **UNTESTED** |" in text), text[-1500:]
    cases += 1
    # 101. amendment 46, each bound: P108 FALSIFIED at 0.965; P109 FALSIFIED below 0.97 (with the lower-line note) and above 1.25; P110 FALSIFIED
    #      on a 0.02 held-out shift on either family; P111 FALSIFIED on a +0.40 GB peak, and HELD at +0.30
    assert pdec(o=_decoded_set(ODEC_FAM, d1=(1.93, 1.95))) == {"P108": "FALSIFIED", "P109": "HELD", "P110": "HELD", "P111": "HELD"}
    low = score_decodedab(DEC(q=_decoded_set(QDEC_FAM, d1=(5.10, 5.12))))
    assert {p: v for p, _, v, _ in low} == {"P108": "HELD", "P109": "FALSIFIED", "P110": "HELD", "P111": "HELD"} and "a lower line is re-asked" in low[1][3]
    assert pdec(q=_decoded_set(QDEC_FAM, d1=(6.70, 6.72))) == {"P108": "HELD", "P109": "FALSIFIED", "P110": "HELD", "P111": "HELD"}
    assert pdec(o=_decoded_set(ODEC_FAM, held_shift=0.02))["P110"] == "FALSIFIED" and pdec(q=_decoded_set(QDEC_FAM, held_shift=-0.02))["P110"] == "FALSIFIED"
    assert pdec(o=_decoded_set(ODEC_FAM, peaks=(12.00, 12.40)))["P111"] == "FALSIFIED"
    assert pdec(q=_decoded_set(QDEC_FAM, peaks=(27.50, 27.80)))["P111"] == "HELD"
    print("FAILING-CASE A46-P108 (reducer):", score_decodedab(DEC(o=_decoded_set(ODEC_FAM, d1=(1.93, 1.95))))[0][3][:120])
    cases += 1
    # 102. amendment 46, engagement: an arm that did not take the route its tag names, or ran outside venv-e4b, reads VOID with the reason and
    #      its prediction UNTESTED -- dec1 resolved to fused, dec1 with no decoded dgrad, a dec0 record without the decoded counter (gnf4 before
    #      #487), torch 2.12, no route_ab record; P110 / P111 UNTESTED with the other family read
    for kw, tag, frag in (({"routes": ("fused", "fused")}, "fused_attn4_m_dec1", "gnf4_train_gemm decoded"),
                          ({"dec_dgrad": (0, 0)}, "fused_attn4_m_dec1", "decoded_dgrad > 0"),
                          ({"drop_counter": True}, "fused_attn4_m_dec0", "decoded_fwd 0"),
                          ({"torch": "2.12.1+cu130"}, "fused_attn4_m_dec0", "is not 2.8.*"),
                          ({"drop_route": True}, "fused_attn4_m_dec1", "no route_ab record")):
        RV = DEC(q=_decoded_set(QDEC_FAM, **kw))
        why = next(x["why"] for x in RV[QDEC_FAM]["rows"] if x["tag"] == tag)
        assert RV[QDEC_FAM]["verdicts"][("e4b", tag)] == "VOID" and frag in str(why), (kw, why)
        assert {p: v for p, _, v, _ in score_decodedab(RV)} == {"P108": "HELD", "P109": "UNTESTED", "P110": "UNTESTED", "P111": "UNTESTED"}, kw
    only_o = {ODEC_FAM: reduce_family(ODEC_FAM, _decoded_set(ODEC_FAM), {}, 60)}
    assert {p: v for p, _, v, _ in score_decodedab(only_o)} == {"P108": "HELD", "P109": "UNTESTED", "P110": "UNTESTED", "P111": "UNTESTED"}
    assert score_decodedab({}) == []
    cases += 1
    # 103. amendment 46's P107, the sm_120 gate's record (decgate.json): no record UNTESTED; a clean pass HELD; a failed test FALSIFIED; a gate
    #      that could not run, an error that is not a failed assertion, a skipped test, a required test not passed (a grouped-nf4-gemm without
    #      the route), or a record without the required-test field UNTESTED; the table renders from the record alone
    gd = tempfile.mkdtemp(prefix="tc1_decgate_selftest_")
    assert score_decgate(gd)[0][2] == "UNTESTED" and score_decgate(None)[0][2] == "UNTESTED"
    run_ok = [{"name": "test_nf4_route.py -k decoded", "rc": 0, "tests": 24, "failures": 0, "errors": 0, "skipped": 0, "missing": []},
              {"name": "test_nf4_route_decision.py", "rc": 0, "tests": 38, "failures": 0, "errors": 0, "skipped": 0, "missing": []}]
    for rec, want in (({"ran": True, "passed": True, "runs": run_ok}, "HELD"),
                      ({"ran": True, "passed": False, "runs": [dict(run_ok[0], rc=1, failures=2), run_ok[1]]}, "FALSIFIED"),
                      ({"ran": False, "passed": False, "runs": [], "reason": "the grouped-nf4-gemm checkout failed"}, "UNTESTED"),
                      ({"ran": True, "passed": False, "runs": [dict(run_ok[0], rc=2, errors=1), run_ok[1]]}, "UNTESTED"),
                      ({"ran": True, "passed": False, "runs": [dict(run_ok[0], skipped=23), run_ok[1]]}, "UNTESTED"),
                      # a grouped-nf4-gemm without the route: -k selects only the dequant tests, all pass, the required ones are missing
                      ({"ran": True, "passed": False, "runs": [dict(run_ok[0], tests=6, missing=["test_decoded_route_passes_the_rd1_gate_against_the_dense_route"]),
                                                               dict(run_ok[1], tests=17, missing=["test_decoded_is_taken_only_when_asked_for"])]}, "UNTESTED"),
                      ({"ran": True, "passed": True, "runs": [{k: v for k, v in run_ok[0].items() if k != "missing"}, run_ok[1]]}, "UNTESTED")):
        json.dump(dict(rec, gnf4_sha="g" * 40, gpu="NVIDIA GeForce RTX 5090", torch="2.8.0+cu128", triton="3.4.0"), open(os.path.join(gd, DEC_GATE_FILE), "w"))
        got = score_decgate(gd)[0]
        assert got[2] == want, (rec, got)
    json.dump({"ran": True, "passed": False, "runs": [dict(run_ok[0], rc=1, failures=2)]}, open(os.path.join(gd, DEC_GATE_FILE), "w"))
    gate_only = render({}, gd)
    assert "## Predictions P107 / P108 / P109 / P110 / P111" in gate_only and "| P107 | decgate | **FALSIFIED** |" in gate_only, gate_only[-800:]
    print("FAILING-CASE A46-P107 (reducer):", score_decgate(gd)[0][3][:140])
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
