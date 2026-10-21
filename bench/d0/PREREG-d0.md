# D0: how much of a trained expert LoRA delta survives repacking into a serving format (OLMoE-1B-7B)

**Status:** registered before any number. USD 0, CPU only, on two stored fixtures. It is a premise for serving trained
expert adapters (`docs/ADAPTER_PORTABILITY.md`), and it decides nothing by itself. The owner decides between (a) the
merged, re-quantized arena and (b) the fused-kernel epilogue, on this number plus the registered cross-format served
read. One lane executes and evaluates. One independent reviewer, then the maintainer's pass.

## Why

Training adds the expert delta ΔW = s·B·A on top of the frozen NF4 base Q_train(W). The fused serving route has no
seam for a low-rank term, so path (a) bakes the delta into the arena: repack T(S + ΔW) for a serving format T. A
round-to-nearest grid drops any part of ΔW smaller than half a code gap. That loss can be total when S already
sits on T's grid (S = Q_train(W), T = NF4), and becomes rounding noise when S is off the grid. D0 measures it per
expert, and measures its effect on held-out loss, before anyone writes arena code.

## Inputs (fixed before any number)

- **Fixtures:** `fixture-dq` (trained at the default double-quantized expert absmax) and `fixture-nodq`
  (`E4B_ABSMAX_DQ=0`). Both are Loggetta defaults on OLMoE-1B-7B-0924-Instruct, generated on P224's shakedown box
  (`bench/p224/PREREG-p224.md`, "Fixture generation"), and sha-pinned in the private receipt store. A fixture that box refused is absent here.
  The D0 numbers are produced only for fixtures that exist.
- **Base:** `allenai/OLMoE-1B-7B-0924-Instruct` at `7f1c97f440f06ce36705e4f2b843edb5925f4498`. The three shards are
  verified against the Hub's LFS sha256 (done: all three OK).
- **Code:** e4b at the commit the fixture's receipt records (the quantizer, `compress_expert_absmax_` and the
  ExpertsLoRA layout); grouped-nf4-gemm `c6215964` for the reference packers (`int4_pack_ref.pack_int4_b32`,
  `mxfp4_pack_ref.quantize_pack_mxfp4`); Loggetta `90a7697d` for the Alpaca formatting and `_expert_digest`;
  torch 2.8.0 (CPU), bitsandbytes 0.50.2, transformers 5.17.0. All of this is in one CPU venv, and its pip freeze is
  recorded.
- **Scale:** s is read from the manifest (alpha/r). ΔW_e = s·B_e·A_e is computed in fp32 for gate_up `[2I, H]` and
  down `[H, I]`.

## Step 0: the training base is `fixture-base`, verified on the box

- **Step 0 is `fixture-base` (P224 Amendment 1).** On the training card, e4b's own loader runs through Loggetta's
  `prepare_qlora_training` with the fixture's plan. Its `_expert_digest` must equal `fixture-nodq`'s receipt (stacks 0
  and 15, exact). After `compress_expert_absmax_`, it must equal `fixture-dq`'s receipt. Each fixture's Q_train(W) is
  read from those stored tensors: codes plus fp32 absmax, and the DQ payloads for `fixture-dq`.
- **Without a verified `fixture-base`, that fixture's D0 is NOT RUN.** It waits for one. It is never a bit-exact CPU
  attempt and never a strike.
- **The CPU rebuild is REPORTED only.** That is the e4b stack quantizer on CPU (bitsandbytes `quantize_4bit`, nf4,
  b64), and the report is the share of codes that differ from `fixture-base`. A CPU round trip of the same weight is
  known to differ from CUDA's, since CUDA multiplies by the reciprocal.

## Step 1: per-expert retention (all 16 layers × 64 experts × {gate_up, down})

**Targets T:**
- **NF4:** bitsandbytes nf4, b64, fp32 absmax: the served grouped-NF4 arena's storage.
- **INT4:** int4-b32 RTN, `pack_int4_b32`. It is uncalibrated, since OLMoE has no GPTQ calibration; this is stated,
  not corrected.
- **MXFP4:** e2m1 with an e8m0 scale per 32, `quantize_pack_mxfp4`.

**Sources S:**
- **Q** = Q_train(W) dequantized in fp32 (the trained base).
- **W** = the raw bf16 checkpoint in fp32. This is the raw-W merge. Per grouped-nf4-gemm's packing rule it is the
  natural source for INT4 and MXFP4, but it is not the base the adapter was trained against.

That gives six cells, T×S. For each cell and each expert projection, D = deq(T(S + ΔW)) − deq(T(S)), and:
- **ρ** = ⟨D, ΔW⟩ / ‖ΔW‖²: the share of the delta carried in its own direction.
- **c** = cos(D, ΔW).
- **e** = ‖D − ΔW‖ / ‖ΔW‖.
- **n** = ‖D‖ / ‖ΔW‖: the spec's ratio. It is reported only. Off-grid noise can push it above 1 with nothing
  retained.
- **f**: the fraction of elements whose dequantized value changed, because its code or its block's scale changed.

**Per cell, reported:**
- ρ̄ = Σ⟨D, ΔW⟩ / Σ‖ΔW‖² (energy-weighted over experts);
- ρ̄_route: per-expert ρ weighted by each expert's held-out routing count (its top-8 selections over the scored tokens
  in REF's forward). This is what the served loss feels. It is reported when step 2 runs;
- c̄ the same way;
- the median, p10 and p90 of per-expert ρ;
- ρ̄ per layer and per projection;
- the base conversion error ‖deq(T(S)) − Q‖ / ‖Q‖.

The NF4-from-Q base is expected to reproduce Q exactly, deq(NF4(Q)) == Q in fp32, for BOTH fixtures. Each block's
largest element sits at code ±1, so re-quantizing returns the same absmax: the fp32 one for `fixture-nodq` and the
dequantized DQ absmax for `fixture-dq`. A failure is a finding and is reported as one. L(NF4-Q base) reuses L(Q) only if
the equality holds.

## Step 2: held-out loss (all arms on one CPU host, one fixed code path)

**Data:**
- The last 64 rows of `tatsu-lab/alpaca` train (rows 51,938-52,001), at a dataset revision pinned before the run.
- **Unseen is SHOWN, per fixture, before any scoring (Step 2 is VOID for that fixture otherwise; Step 1 stands):**
  - The receipt's data record shows that training read only rows below 51,938 (`examples_used`).
  - Re-encoding exactly rows `[0, examples_used)` with Loggetta's own demonstration path (at `90a7697d`, on the
    dataset I load) reproduces the receipt's `token_stream_sha256`. This proves which rows were fed and that my
    dataset copy has the same bytes there. *(superseded by Amendment 1)*
  - The dataset fingerprint is compared too when the receipt records one.
- **Scored as training scored it:** the demonstration trains with loss on ALL tokens and with examples concatenated
  by EOS (Loggetta's legacy demo setting). Each held-out row is the demo's own alpaca text plus EOS, scored on all its
  tokens after the first. *(superseded by Amendment 1)*
- **One deviation, stated:** each example is scored alone, never packed beside its neighbours. This makes the
  per-example bootstrap possible, and it removes cross-example context that training had.
- L = total NLL / total scored tokens, truncated at 512.

**The forward:**
- My layer-major OLMoE forward, in fp32 compute. Expert weights are held as bf16 values of each arm's dequantized
  weights, as the serving kernels dequantize to bf16.
- The attention LoRA is applied unmerged in every arm except BASE0, so only the expert path differs.
- Plumbing gate (VOID otherwise): on a 2-layer truncation of the base, my forward matches transformers'
  `OlmoeForCausalLM` (fp32, same weights) to max |Δlogit| ≤ 1e-3 on two held-out rows.

**Arms:**
- **BASE0:** no adapter, context only.
- **Q:** the training base plus the attention LoRA, with no expert delta.
- **REF:** Q plus the expert LoRA applied unmerged in fp32 (the trainer's function).
- For each cell T×S: **T(S)** and **T(S+ΔW)**.

**Retention by loss:** R = [L(T(S)) − L(T(S+ΔW))] / [L(Q) − L(REF)].
- Each R comes with a paired bootstrap 90% CI (2,000 resamples over examples, seed 0).
- Also reported: L(T(S+ΔW)) − L(REF), which includes the format's own base cost, and the teacher-forced max and mean
  |Δlogit| against REF.

**Resolution:**
- The denominator has its own CI. The expert delta's improvement L(Q) − L(REF) is RESOLVED when it exceeds 3 × its
  bootstrap SE.
- R and its CI are reported ONLY when it is resolved. Otherwise the loss part is UNRESOLVED: the arms' losses are
  reported, no R is, and the labels come from ρ̄ alone, marked so.
- The paired bootstrap resamples the SAME 64 rows for the numerator and the denominator together. Each resample
  recomputes both from the same row draw, so R's CI carries their correlation.

**Order, host and budget (stated before anything runs):**
- **Order:** `fixture-dq` first, then `fixture-nodq`.
- **Forwards:** 15 per fixture, fewer when a cell's base reproduces Q exactly and reuses L(Q).
- **Host:** a capped CPU container on a lab host, never the control host: 2 CPUs, shares 256, a 16 GB hard ceiling,
  layer-major streaming. One layer's weights are resident at a time, so RSS stays under 8 GB and the full model never
  loads. The host's free RAM is recorded at start.
- **Estimate:**
  - About 1.18 B active parameters per token (attention, 8 of 64 experts, router, lm_head), so ~2.4 GFLOP per token.
  - The upper bound is 64 rows × 512 tokens, ~7.7e13 FLOP per forward: ~26 min at 50 GFLOP/s, so ~6.5 h per fixture
    plus the repacking.
  - The real rows are shorter. The scored-token count is measured from the tokenized held-out set before the first
    forward and stated in the record.
- **Caps:** Step 1 is cheap and always runs. Each fixture's step 2 has an 8 h cap. Hitting it is a PLANNED outcome:
  the loss part is reported INCOMPLETE with the arms done, never extrapolated.

## The labels (per cell, per fixture; fixed now)

- **RETAINED:** R ≥ 0.95.
- **PARTIAL:** 0.5 ≤ R < 0.95.
- **LOST:** R < 0.5.
- **When the loss is UNRESOLVED,** all three thresholds apply to ρ̄ instead (RETAINED ρ̄ ≥ 0.95, PARTIAL 0.5-0.95, LOST
  < 0.5). Each such label is marked "(ρ̄; loss unresolved)". A ρ̄ label never claims a loss effect.
- **A label whose 90% CI crosses its boundary says so,** e.g. "PARTIAL (CI crosses 0.95)", so a borderline call is
  never read as firm.

c̄ is reported beside every label. A delta carried in expectation (ρ̄ ≈ 1) but buried in noise (c̄ ≪ 1) is called
out in words, because its loss effect is what decides it.

## Predictions (graded, never ruling)

| prediction | basis |
|---|---|
| **NF4 from Q: LOST**, ρ̄ < 0.5 and f < 0.2 | Q is on NF4's grid. A delta under half a code gap rounds back, and the absmax element (code ±1) only rescales its block |
| **NF4 from W: ρ̄ in 0.8-1.2, c̄ < 0.6**, and its base equals Q for `fixture-nodq` up to the CPU-vs-CUDA tie flips (that share reported) | raw W is off the grid, so rounding dithers the delta: unbiased but noisy |
| **INT4 and MXFP4, either source: ρ̄ in 0.8-1.2, c̄ < 0.6** | the same dithering, and Q is off these grids too |
| **By loss, the off-grid cells are PARTIAL or RETAINED; NF4 from Q is LOST** | the noise is a second quantization error of the base's own size; the signal is the delta's mean |
| **`fixture-dq` and `fixture-nodq` give the same labels** | the double-quantized absmax moves each block's scale only slightly (8-bit nested statistics) |
| **The expert delta's improvement is RESOLVED** | 200 steps on Alpaca teach the response format |

Confidence is moderate at best: ‖ΔW‖ against the code gap after 200 steps is the open quantity. The heavily
routed experts may move far enough for NF4 from Q to keep a real share.

## What follows (never automatic)

- **Some off-grid cell RETAINED** while NF4-from-Q is LOST: path (a) is viable only as a repack from a non-trained
  base (raw W, or another format). The registered cross-format served read then prices that mismatch on the served model. The arena code is its own PR.
- **Everything LOST or PARTIAL:** path (a) fails its premise, and (b), the fused low-rank epilogue, is the expert
  path. It is a grouped-nf4-gemm project, registered separately.
- **UNRESOLVED:** the fixture is too weak to rule on loss. A stronger fixture (more steps) is a new data-generation
  item, never a re-read of this one.

## Records

These go to the private store under the D0 fixture entry:
- the step 0 digests;
- the per-expert table (CSV);
- the per-cell summaries;
- the per-arm losses and per-example NLLs;
- the bootstrap outputs;
- the pip freeze, the host and the wall times.

There are no tensors in the store. Every number in the result cites a file there.

## Amendment 1: the fixtures' data (registered before any number)

**Why:** at Loggetta `90a7697d`, omitting `--dataset` cannot supply 200 × 512 tokens. The demonstration's streamed
path validates every row, and `tatsu-lab/alpaca` row 284 has an empty output. So data preparation raises before any
weight loads; rows 0-283 give 23,377 of the 102,400 tokens. This was reproduced on CPU with Loggetta's own
`prepare_data`. P224 Amendment 1 therefore trains both fixtures on Alpaca's own full-text column, pinned:
`--dataset tatsu-lab/alpaca --dataset-revision dce01c9b08f87459cf36a430d809084718273017 --format text --text-field
text`. The text format keeps loss on all tokens and concatenated packing (958 rows, 102,400 tokens).

**What changes in D0** (Step 2's data only; nothing else moves):
- **Unseen is SHOWN** by re-encoding through Loggetta's own data path at `90a7697d`: `prepare_data` with the
  receipt's spec and the plan's data profile. It must reproduce the receipt's `token_stream_sha256` and
  `examples_used`. Training must have read only rows below 51,938, and 958 is expected. The dataset revision must be
  the pinned `dce01c9b`. A failure VOIDs that fixture's Step 2, as before.
- **Scored as the fixtures trained:** each held-out row (rows 51,938-52,001 of the same pinned revision) is that
  row's `text` plus EOS, Loggetta's `encode_example` for the text format. It is scored on all its tokens after the
  first and truncated at 512. The one stated deviation stands: each example is scored alone.

