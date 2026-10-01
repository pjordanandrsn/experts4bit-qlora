# P94 — K8 against the arithmetic a T > 1 route replaces: K25 against the served NF4 M-tile kernel at T == 1, with the scalar GEMV beside them, on Granite-3.1-3B-A800M and OLMoE-1B-7B on one RTX 5090 (registered 2026-10-01, before any run)

Issue: experts4bit-qlora#564.

**Why.**
- **P93** (`bench/p93/RESULTS-p93.md`, #837) read the served-precision K25 route at B=16 ×0.594 (Granite) and ×0.598
  (OLMoE). Granite was LICENSED; OLMoE was QUALITY_FAIL on c4val1, its K8 moved +0.155 ppl.
- **P92** read the same route in bf16 and OLMoE's c4val1 moved **−0.107**. Making the weight rounding 8× finer flipped
  the sign and grew the move, so weight rounding is not the cause.
- **A design error in P92 and P93, found after P93's read.**
  - Their K8 compared K25 at T == 1 (ON) against the scalar fp32 decode GEMV (OFF, the route T == 1 runs today).
  - The default under consideration (`auto`) changes only rows above T == 1. There the route it replaces is the served
    NF4 M-tile kernel, TF32 on the fp32 dequant, not the GEMV.
  - So their quality comparison had the wrong baseline for the change they proposed. P92's and P93's verdicts stand
    under their own registered rules. This lane reads the comparison that the T > 1 default actually needs.
- **The instrument** `E4B_NF4_T1_DEVICE_GROUPING=1` (#838) lets K8 read the served M-tile kernel at T == 1. On the A2000,
  a token's rows on that kernel are bit-equal alone and inside B=16, so its T == 1 reading stands for the batched rows,
  as K25's does.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- receipts and ledger rows;
- proven teardown.

## The lane

- **Box, software, families:** P93's. One RTX 5090; e4b at the launch commit (carrying #834's route and #838's
  instrument); grouped-nf4-gemm at `908a2ca`. Granite `r12epi` and OLMoE `nf4` at P44's revisions, each on its licensed
  env.
- **Three arithmetics at T == 1:**
  - **g:** the scalar fp32 decode GEMV (`E4B_NF4_GROUPED_SMALLM=0`, `E4B_NF4_T1_DEVICE_GROUPING=0`). P93's OFF.
  - **m:** the served NF4 M-tile kernel (`E4B_NF4_GROUPED_SMALLM=0`, `E4B_NF4_T1_DEVICE_GROUPING=1`). B=16's arithmetic
    today.
  - **t:** K25, the select tree through TF32 (`E4B_NF4_GROUPED_SMALLM=1`). P93's ON.
- **The premise, on the card, before anything is fetched:**
  - `tests/test_k25_row_exact_gpu.py` and `tests/test_nf4_t1_device_grouping_gpu.py`. Each arithmetic's rows must be
    bit-equal alone and inside B=16 (rc 25).
  - grouped-nf4-gemm's K25 contract compiled (rc 23).
- **Per family:**
  1. m at B=1, censused, for engagement;
  2. K8 g, m, t on wikitext, then g, m, t on c4val1 (P44-a's arm: 2048 steps, eager B=1, `--no-fuse-qkv`).

**The reducer** (`p94_reduce.py`, 10-case self-test):
- **VOID** if any of these holds:
  - an arm is missing;
  - m's B=1 census does not run `_gemm_nf4_grouped` exactly 2 × layers per step with no `_gemv_nf4*`;
  - m or t is bit-equal to g on both texts of a family;
  - the K8 arms scored different text.
- **LICENSED** if `experts4bit_qlora.k8_gate.verdict` (uncalibrated: |Δppl| ≤ 0.05 on every text) passes with base m
  and candidate t, in both families.
- **QUALITY_FAIL** otherwise.
- **Reported, not gated:** m − g and t − g per text. `BASELINE_SHIFT` marks a family whose m − g alone exceeds 0.05.

**The registered consequence.**
- **LICENSED**, with P93's speed (both families B=16 ≤ 0.95, already read): `E4B_NF4_GROUPED_SMALLM` defaults to `auto`.
  That is rows above T == 1 only. T == 1 stays on the GEMV, which this lane does not license moving.
- **QUALITY_FAIL:** the default stays `0`.

## Predictions (written before the data)

- **|t − m| < 0.01 ppl** on both texts in both families: LICENSED. Both multiply TF32 on the same fp32 weights over
  64-wide K chunks; they differ only in tiling and in where each chunk's sum lands.
- **m − g on OLMoE c4val1 about +0.10 to +0.20** (BASELINE_SHIFT), near P93's t − g (+0.155). P93's swing would then be
  the GEMV-versus-tile difference that B=16 production already carries. Granite's |m − g| ≈ 0.02–0.03.
- **t − g reproduces P93's ON − OFF** exactly (the same card class, the same build, a deterministic instrument).

## Box and cost

- **`p94-5090-<n>`:** one RTX 5090. **Guard 1.0 h at ≤ $0.75/h (≤ $0.75)**, so there is no proving rental.
- About 40 minutes:
  - install and premise ~5;
  - per family, fetch and bake ~2.5, one B=1 arm ~1, six K8 arms ~2.5 each.
- **Lane ceiling $1.50; hard stop $2.00.**

## Rehearsal

The whole runner ran on the NAS RTX A2000 (sm_86) from e4b `52200d0`, this branch before this section, with
grouped-nf4-gemm at the pin. It ran with `P94_GPU_CLASS=A2000 P94_MIN_DISK_GB=20`, so it was marked REHEARSAL.
- **Up to the arms, it held:**
  - install and the tripwire (the instrument, the tree, the TF32 plan);
  - the reducer's self-test (10 cases);
  - the premise: both row-exact files 6/6, K25's contract compiled 30/30;
  - both families fetched and baked.
- **Every arm stopped** at the fp8 paged-KV append, which sm_86 cannot compile, as in P91–P93's rehearsals.
- **The reducer** then read VOID with every arm missing, as it must.
- **Two faults, each fixed before this section:**
  - the first attempt's reducer self-test failed, on an exact-0.05 boundary case that binary floats cannot represent
    and on a zero-shift case its own engagement rule voids (rc 21);
  - in the second, the box's `git clone` of e4b stalled on the QNAP's GitHub link and was retried; the runner's one pip
    retry handled it.
- No time is quoted (the A2000 is correctness-only).

Amendments, dated, go below this line before any data is read.
