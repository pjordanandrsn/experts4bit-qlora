# P92 — does K25 make the NF4 families' B=16 decode faster on an RTX 5090 without moving their K8? Granite-3.1-3B-A800M and OLMoE-1B-7B, OFF vs ON (registered 2026-10-01, before any run)

Issue: experts4bit-qlora#564. This lane tests the change lane P91's registered decision named.

**Why now.**
- **P91** (`bench/p91/RESULTS-p91.md`, #826) censused the NF4 families' decode steps on one RTX 5090. The served NF4
  grouped GEMM (`_gemm_nf4_grouped`) is **61.7 %** of B=16 kernel time on Granite (`r12epi`) and **71.9 %** on OLMoE
  (`nf4`). Its decision: an NF4 grouped small-M kernel.
- **K25** (grouped-nf4-gemm #429) is that kernel: K19's grouped small-M tensor-core GEMM with the NF4 codebook dequant.
  - It reuses K19's 16-row tiles, in-kernel gather, sorted output, K23's `scatter` / `gather_div` and K21's masked
    tail.
  - Its weight operand is the bf16 dequant. The served GEMM multiplies TF32 on the fp32 dequant, so the arithmetic
    changes and the lane gates on quality.
  - On the A2000 its contract held compiled (28/28), and its outputs were bit-identical across plans and codebook
    decodes. No speed number exists for it.
- **The route** (`E4B_NF4_GROUPED_SMALLM`, #827) serves the NF4 store's device-grouped decode rows with K25 under
  `auto` (T > 1) or `1` (T == 1 too); `0`, the default, is today's route.
  - At Granite's expert shapes on the A2000, a token's K25 rows are bit-equal alone and inside B=16, and the T == 1
    route captures.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- receipts and ledger rows;
- proven teardown.

## The lane

- **Box.** One RTX 5090, any CPU vendor.
- **Software.**
  - e4b at the launch commit, carrying #827's route.
  - grouped-nf4-gemm at **#429's merge** (K25 with its default plan).
  - The tripwire refuses an install without `nf4_smallm.gemm_nf4_grouped_smallm`, or whose route plan
    (`_K25_PLAN`) is not BLOCK_N 32 / KC 256 / 4 warps / 2 stages / the paired decode.
- **Families,** at P44's pinned revisions, each with its licensed env (`bench/p44/serve_stack.py`'s `arm_env`,
  verbatim; the pin test checks it): Granite `ibm-granite/granite-3.1-3b-a800m-instruct @a0278068` (`r12epi`), then
  OLMoE `allenai/OLMoE-1B-7B-0924-Instruct @7f1c97f4` (`nf4`).
- **Arms.** OFF = `E4B_NF4_GROUPED_SMALLM=0` (today's route). ON = `E4B_NF4_GROUPED_SMALLM=1`: K25 for every NF4
  decode row, T == 1 included, so K8's B=1 eager loop reads the kernel. Everything else is the family's env, and the
  lean glue is at its default (`auto`, so ON's gate_up reads token rows and its down scatters, as on K19's rows).

**The premise, on the card, before anything is fetched:**
1. e4b's `tests/test_k25_row_exact_gpu.py`, staged and pinned. A token's K25 rows must be bit-equal alone (T = 1) and
   inside a B=16 step, the T = 1 output must match the fp32 oracle, lean token rows must be bit-equal to the expanded
   rows, and the T = 1 route must capture and replay bit-equal. Without it a B=1 K8 does not stand for the B=16 rows:
   **rc 25**, the lane stops.
2. grouped-nf4-gemm's `kernel/test_nf4_grouped_smallm_interp.py` at the pin, compiled. K25 has never compiled for
   sm_120 (the A2000 is sm_86): **rc 23**, the lane stops.

**Per family, in this order:**
1. **Speed** (P91's harness: `bench/p39/step_decomp.py` + the bench/p42 hook, graph-window timing, prompt 512, 128
   generated tokens, `--no-fuse-qkv`, P42's replay census):
   - B=16 OFF, ON, OFF, ON; the first draw of each arm censused;
   - B=1 OFF, ON, one draw each, both censused.
2. **Quality** (P44-a's K8 arm: 2048 steps, the eager B=1 loop, `--no-fuse-qkv`): OFF and ON on wikitext, then OFF
   and ON on c4val1.

**The reducer** (`p92_reduce.py`, with a 14-case self-test), per family, in this order:
- **VOID** if any of these holds:
  - an arm is missing;
  - a B=16 arm's two draws are more than **3 %** apart;
  - engagement fails. At B=16 and at B=1, the ON census must run `_gemm_nf4_grouped_smallm` exactly 2 × layers times
    per step (64 on Granite, 32 on OLMoE), with no `_gemm_nf4_grouped` and no `_gemv_nf4*`. The OFF census must run
    no `_gemm_nf4_grouped_smallm`;
  - K8 ON is bit-equal to OFF on both texts (the eager loop did not read K25);
  - the K8 arms scored different text or step counts.
- **QUALITY_FAIL** if `experts4bit_qlora.k8_gate.verdict` fails in its uncalibrated regime: |Δppl| ≤ 0.05 on every
  text. K25 is an arithmetic change, not a calibrated pack, so a large move in either direction is a failure.
- **LICENSED** if B=16 ON/OFF (the mean of each arm's two draws) is **≤ 0.95**.
- **NOT_FASTER** otherwise.
- B=1 ON/OFF is reported beside the verdict.

**The registered consequence.**
- **Both families LICENSED:** `E4B_NF4_GROUPED_SMALLM` defaults to `auto` (rows above T == 1). If B=1 ON/OFF is at
  most 1.00 in both families, T == 1 is included too.
  - The route is global, so the default also changes the NF4 rows of families this lane does not read. The register's
    other NF4 positions (Mixtral, Gemma-4) are then labelled measured before K25 and unmeasured under it.
- **Otherwise:** the default stays `0`, each family's verdict is recorded, and the next lane follows from it. A
  NOT_FASTER family points at K25's plan; a QUALITY_FAIL family points at the weight rounding.

## Predictions (written before the data)

- **B=16 ON/OFF:**
  - Granite **0.66–0.85**; OLMoE **0.65–0.82**. Both LICENSED.
  - Why: grouped-nf4-gemm's K22/K24 benches put the served NF4 GEMM at about 25 % of the byte floor on gpt-oss's
    shapes, where K21, K25's sibling, read 50 %. K25 at 1.5–2× the served GEMM (Granite 5.48 ms, OLMoE 8.61) saves
    1.83–2.74 / 2.87–4.31 ms. With Granite's `[R, H]` `index_select` gone too (0.39 ms in P91), the steps fall from
    9.19 / 12.47 ms to about 6.06–6.97 / 8.16–9.60: 0.66–0.76 / 0.65–0.77. The bands' upper ends leave room for a K25
    that does less well on these smaller shapes than K21 did on gpt-oss's.
- **B=1 ON/OFF:** **0.95–1.20** in both. At one row per expert, K25 spends 15 of 16 MMA rows on padding, and the served
  route is a decode GEMV. K19 read 1.103 against its GEMV (P88).
- **K8:** **|Δppl| < 0.01** on both texts in both families.
  - Rounding the weight to bf16 is the rounding the dequant-then-GEMM path makes, about 2^-9 relative, against NF4's
    own quantisation error of about 10 %.
  - On synthetic weights, K25's rms error against an fp64 product was 1.37× the served kernel's. That is output-level
    noise, well inside a K8 budget.
- **Engagement** holds in every census, and the K8 ON arms differ from OFF in the last digits.
- **OFF reproduces P91's steps** (Granite 9.19 / 3.34 ms, OLMoE 12.47 / 3.96) within host spread. Reported, not gated.

## Box and cost

- **`p92-5090-<n>`:** one RTX 5090, **guard 1.0 h at ≤ $0.75/h (≤ $0.75)**. The guard is not over 1 h, so there is
  no proving rental under the rule; the premise runs on the card before anything is fetched.
- Expected wall-clock is about 40 minutes:
  - install about 3, premise about 3;
  - per family, fetch and bake about 2.5;
  - six speed arms at about 1.2 each (P91: four arms and two fetch/bakes in 8.5 minutes);
  - four K8 arms at about 1.6 each (P44-a's OLMoE `nf4` arms).
- **Lane ceiling $1.50; hard stop $2.00.**

## Rehearsal

The whole runner ran on the NAS RTX A2000 (sm_86) from e4b `967b1d6`, this branch before this section, with
grouped-nf4-gemm at the pin. It ran with `P92_GPU_CLASS=A2000 P92_MIN_DISK_GB=20`, so it was marked REHEARSAL.
- **Two earlier attempts each found a fault, fixed before this page:**
  - the reducer's self-test ran before the install, but the reducer applies the installed `k8_gate` (rc 21); it now
    runs after the tripwire;
  - a commit that had not reached the remote could not be installed (rc 9).
- **Up to the arms, it held:**
  - install and tripwire (K25 present, `_K25_PLAN` as registered);
  - the reducer's self-test (14 cases);
  - the premise: the row-exact test 4/4 and K25's contract compiled 28/28;
  - both families fetched at their pins and baked.
- **Every arm stopped** at the fp8 paged-KV append (`fp8e4nv`), which sm_86 cannot compile, as in P91's rehearsal.
  That is the rehearsal card's limit, not the runner's.
- **The reducer** then read VOID for both families, with every arm missing, as it must.
- No time is quoted (the A2000 is correctness-only).

Amendments, dated, go below this line before any data is read.
