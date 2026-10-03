# P105 — the Gated DeltaNet kernels under the hybrid paged path, with a premise that is a distribution: with flash-linear-attention and causal-conv1d installed, does e4b's hybrid paged path stay within its control on every seed and replay its decode graphs exactly, and what do the kernels buy, on one RTX 5090 and one host (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#928. Follows P103 (#932) and P104 (#938), both stopped at their proving rentals.

**Why.**
- P103 and P104 asked this question, and each failed its premise under fla on one test: the hybrid parity check, which
  scores a tiny 4-expert top-2 MoE hybrid's worst relative logit error at **one seed**.
- **That statistic is a lottery** (P104, `bench/p104/RESULTS-p104.md`). On the A2000, with e4b's `PagedModelRunner`
  and SDPA standing in for the fp8 kernel, it is bimodal across 10 seeds in every kernel arm, transformers' torch path
  included: 5/10 torch-path seeds fail its own bound. Seed 0 lands low on the torch path and high under fla, and that
  is the whole of both failures.
- **A dense hybrid, by contrast,** sits at its control's error on every seed, identically across the torch path, fla
  and fla + causal-conv1d.
- **The new test** `tests/test_linear_state_dense_parity_gpu.py` gates on that distribution. A dense hybrid
  `[LIN, ATT, LIN, LIN]` runs through `PagedModelRunner` with the compact fp8 KV pool, against transformers' cache
  prefilled in the same 32-token chunks, with batched decode in changing row order. On **every one of 8 seeds**, the
  hybrid's worst relative logit error must be within 2× that seed's dense all-attention control. Greedy-token agreement
  is reported, not gated: the control itself flips argmax near-ties against its bf16 reference.
  - Variants: SDPA standing in for the attention on any CUDA card, and gnf4's real fp8 kernel on sm_89+.
  - **On the A2000:** the hybrid-to-control ratio is 0.66–1.60 on all 8 seeds in all three kernel arms. A mutant that
    writes the state rows back in reversed order (sequences' states swapped) fails at 9.2–14.6× on every seed, under
    the torch path and under fla + causal-conv1d. Its greedy-token agreement stays at 15–21/21, which is why tokens
    are not the gate.
- **What the kernels might buy** (indicative A2000 per-call times, #928, `bench/p103/a2000/probe6.py`): fla's
  decode-step recurrent rule is about 6× the torch path per layer, including under CUDA graphs (at 16 rows, 0.27
  against 1.64 ms). Its prefill chunk rule is 4–10× faster.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

**P104's lane, with the premise changed and nothing else:**
- P98's measurement at its registered bytes: arms **g** (bucketed graphs 1–16), **e** (the padded-eager oracle) and
  **p** (plain eager), over W16 and W1.
- Three phases on ONE box: **t** (the torch path, asserted kernel-free), **f** (`flash-linear-attention==0.5.2`) and
  **fc** (plus `causal-conv1d==1.7.0`), with torch held and an engagement record per phase.
- P103's reducer and rule, unchanged.

**The premise per phase:**
- **t:** the three pinned hybrid files, the chunk-matched file and the dense parity file, **11 passed**, none skipped.
  This keeps P97–P101's premise intact on the torch path.
- **f, fc:** `test_hybrid_decode_graphs_gpu.py`, `test_linear_state_graph_gpu.py`, the chunk-matched **all-linear**
  test (bit for bit, 4 seeds) and the dense parity file (8 seeds each, stand-in and real kernel), **9 passed**, none
  skipped.
- **Reported in every kernel phase, never gating:** the two single-seed tiny-MoE checks (`test_linear_state_gpu.py`,
  and the chunk-matched file's hybrid test).

**The rule** (`p105_reduce.py`, P103's, self-test 20 cases):
- **Per phase:** P98's registered rule, unchanged. A wrong engagement record is **VOID**. A kernel phase whose install
  fails is **UNAVAILABLE**, one that is skipped is **NOT_RUN**, and one whose premise fails with the kernels engaged is
  **NOT_SUPPORTED**.
- **The lane:** **VOID** unless phase t is SUPPORTED; otherwise phase fc's result, else phase f's, else
  **UNAVAILABLE**.
- **`recommend`** is true iff phase fc is SUPPORTED and its graph arm decodes at least 1.05× phase t's on W16 or W1.

**The registered consequence:**
- **SUPPORTED, recommend:** `docs/SERVING.md` lists the two kernels as supported for hybrid serving and recommends
  them, with the measured gain, the install line, and the caveat that fla's chunked prefill depends on chunk
  boundaries (as transformers' own does). A register row.
- **SUPPORTED, not recommend:** the same, without the recommendation.
- **NOT_SUPPORTED:** the failing phase, bucket or test is recorded on #928, and SERVING.md warns against that kernel
  for hybrid serving.
- **UNAVAILABLE / VOID:** nothing moves.

**Not measured:** quality in nats against the torch path; prefill speed (P98's harness times decode only); any host but
this one.

## Predictions (written before the data)

- **The lane is SUPPORTED, and `recommend` is true.** The dense parity premise holds in every phase, every seed within
  2× its control.
- **fc's graph arm over t's: W16 between 1.15× and 1.60×, W1 between 1.02× and 1.15×** (P104's bands, unread).
- **f's graph gain is at least 80 % of fc's.**
- **fc's plain-eager arm over t's on W16: at least 1.15×.**
- **The reported single-seed MoE checks:** at least one of the two lands above its bound in some kernel phase. It is a
  lottery, so this is a weak prediction.
- **Peak GPU memory** stays at or below 28 GB.

## Box and cost

- **`p105-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P105_PROVE=1`, no model.
- **`p105-5090-<n>`:** one RTX 5090 with ≥ 200 GB of disk. **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).**
- **Lane ceiling $3.00; hard stop $4.00.**

## Rehearsal

- **The new test** on the A2000 at e4b `a1bd83d`:
  - 1 passed, 1 skipped (the real-kernel variant, sm_86) on the torch path, fla and fla + causal-conv1d;
  - the reversed-write-back mutant fails on the bar in the two arms where it ran.
- **The runner's proving path** is rehearsed on the A2000 before the merge.

**Rehearsed 2026-10-03, 06:24–06:30Z, at `506ce06`, on the NAS RTX A2000**, staged exactly as `p105_drive.sh` stages.
`P105_PROVE=1` with the knobs class A2000, disk 10 GB and premise skips allowed. Exit rc 0, with `PROVED` and the
`REHEARSAL` marker.
- **Install and tripwire held:** e4b `506ce06`, gnf4 `34da93d`, torch 2.8.0+cu128, transformers 5.17.0. Phase t is
  kernel-free. The reducer's self-test passed 20 cases.
- **Premise t:** 7 passed, 4 skipped (the fp8 tests on sm_86). The dense parity stand-in ran: hybrid-to-control ratios
  0.68–1.60 over the 8 seeds.
- **Phase f:** fla 0.5.2 engaged as registered. Premise 7 passed, 2 skipped; dense ratios 0.66–1.36. The two
  single-seed MoE checks were reported (they skip on sm_86). `phase f ok`.
- **Phase fc:** causal-conv1d 1.7.0 engaged as registered. Premise 7 passed, 2 skipped; dense ratios 0.74–1.36.
  `phase fc ok`.
- **Reporting:** the per-seed dense lines and the reported-only MoE lines reach `summary.txt` as designed.
- **HF CDN probe:** 55.1 MB/s.

Amendments, dated, go below this line before any data is read.
