# Status — what this package does, what changed, what is open

**As of 2026-09-04, version 0.34.0.** One page. The README argues the
case; this page states the position. Every line has an entry in
[`docs/claims.json`](claims.json) with its evidence path, and nothing is
here that does not.

Evidence words, used strictly: **measured** = a run with a receipt in
this repository. **measured-private** = the run happened and the number
is real, but the receipt lives in a private audit tree, so *you cannot
check it from here* — those are named as such. **retired** = published,
now known wrong, kept so the retraction is findable. **superseded** =
still true as measured, but a later entry is the number to quote.

---

## What you get today

**It fits, and it trains.** Full bf16 OLMoE-1B-7B OOMs a 12 GB card; in
4-bit it loads at 4.70 GB and trains under 8 GB, and QLoRA on the frozen
NF4 experts moves a held-out Alpaca eval from 1.4813 to 1.0290. The
streaming loader never materialises the bf16 model in CPU or GPU RAM.

**It scales past VRAM.** With expert offload, Qwen3-30B-A3B peaks at
7.16 GB and Gemma-4-26B-A4B at 8.47 GB during a training step — both OOM
without it — for about +11% s/step at OLMoE scale. Against a descending
host-RAM cap the on-disk arena needs 2.56× to 6.40× less host RAM than
the pinned-RAM path, and at 8.59 GB Qwen3-30B is OOM-killed on host RAM
and completes on the arena.

**The fused training path is faster at equal loss.** Across two 30B-class
MoEs, five datasets each, 200 steps per cell: 1.52–1.81× per step at
0.75–0.81× peak VRAM and 0.86–0.92× energy, with loss parity on both
registered criteria and the frozen 4-bit stack bit-identical over
16.31 GB hashed. Default to `enable_fast_train(model, dgrad=True)`.

**Serving is at parity with the model's own attention on three of four
families, and not on the fourth.** This is the part that changed most
this week. Measured against a *chunk-free* reference — one full forward,
no chunk boundaries — the paged decode path is indistinguishable from
the model's own attention on Granite, gpt-oss and Qwen3, and is **not**
on Gemma-4:

| family | paged vs reference | that model's noise floor | reading |
|---|---|---|---|
| Granite-3.1-3B-A800M | 0.00229 nats | 0.00330 | indistinguishable |
| gpt-oss-20b | 0.00288 nats | 0.01758 | indistinguishable |
| Qwen3-30B-A3B | 0.00173 nats | 0.00641 | indistinguishable |
| Gemma-4-26B-A4B | +0.093 … +0.247 (three windows) | no stable floor: HF's own cache −0.107 … +0.271 | **no reference at this resolution** ([#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)) |

**Read that table with its floor column or not at all.** Two
arithmetically equivalent forwards of a mixture-of-experts model do not
agree, because rounding flips which experts the router picks — 4.52% of
layer-token choices on gpt-oss, 6.77% on Qwen3 — and the disagreement is
carried almost entirely by the flipped tokens (KL 0.0504 against 0.0013).
A parity delta below the floor means *indistinguishable*, never "a small
cost". Method in [`METHODOLOGY.md`](METHODOLOGY.md) §13.1; per-family
table in [`SERVING-PARITY.md`](SERVING-PARITY.md).

**Gemma-4 has no reference at this resolution.** On three 512-token
windows the paged path sits +0.093, +0.114 and +0.247 nats from a
one-shot forward — and transformers' *own* cached forward sits −0.107,
+0.271 and +0.081 from that same one-shot forward on the same windows.
The cause is the model, not a path: plain transformers with no e4b code
gives the same 255 tokens an NLL that moves by 0.4 nats depending only
on which tokens follow them (bf16 batch-shape variance in the expert
gathers, 0.2% at layer 1, amplified by the router to 35% of the hidden
state by layer 19; Qwen3 shows the mechanism at a tenth of the
amplitude and loses 0.001). Running the router in fp32 (plain transformers) does not remove the amplification (layer-19 divergence 0.25 against 0.34, top layer unchanged) and itself moves the same tokens' NLL by 1.1 nats, so router precision is not a lever; the sensitivity is the model's. The paged path's one localised,
measured cost is the fp8 cache and dot: 0.046 nats, concentrated on the
five 512-dim layers, 0.017 with 32-wide K groups. Method: METHODOLOGY
§13.2; numbers: SERVING-PARITY.

**Serving speed**, single-stream Qwen3-30B-A3B on a rented RTX 5090:
about 100 tok/s on the NF4 baseline, 204.6 tok/s with calibrated int4
attention and int4 experts, and about 1,238 tok/s aggregate at B=16.
On the 2026-09-04 validation box the same stack went from 156.1 to
177.9 tok/s at B=1 (×1.14) when 0.34.0's round-2 fold started engaging
on the calibrated int4 attention it had silently skipped (#375).
Those figures are **measured-private**. On the same box, vLLM is ahead by
1.47× at B=1 and 1.55× at B=16 with identical prompts — that comparison
is the honest one and it is also measured-private.

**Per-family throughput is now measured in-repo** (2026-09-04): six
families under one protocol on one rented 5090 class, with every refused
arm named — that list became the build-out, and 0.33.0 ships it. Table
and receipt: [`SERVING-THROUGHPUT.md`](SERVING-THROUGHPUT.md).

**One more family reaches the reference's ratio with the build-out,
and one claimed to has been retracted** (0.33.0, **measured-private**
until the validation lane's receipt lands in-repo; the numbers are on
the merged PRs): Gemma-4-26B-A4B at 121 tok/s B=1 with int4 experts +
round-1 norms + router epilogue (×1.69 over NF4; int4 experts alone
×1.20 B=1 / ×1.39 B=16; ×1.68 at B=16). Mixtral reaches ×2.14 (×2.29
with calibrated attention). gpt-oss stays NF4-only: a uniform int4 grid
cannot hold its MXFP4 experts (+0.63 nats measured), and the native path
is not yet a lever. **Granite-3.1-3B-A800M's "302 tok/s, ×1.59" row is
retracted as a parity claim**: its int4 experts cost +0.063 ppl on the
same window, over the registered 0.05-ppl gate (`k8_gate`, uncalibrated
rule), and the 0.33.0 text quoted the row without applying that gate.
The speed is real; the configuration is not licensed. Granite's licensed
stack keeps NF4 experts (round-1 + round-2 folds + router epilogue) and
its combined number is being measured.

---

## What changed — retired, superseded, corrected

- **"The fp8 paged KV cache costs +0.047 ppl on Qwen3" — RETIRED.** That
  is +0.0058 nats, below the model's own 0.0095-nat floor.
  Indistinguishable from reordering the arithmetic; not attributable to
  the cache. The rule derived from it ("buy headroom back from the cache
  first") goes with it — there was nothing to buy back.
- **"gpt-oss's +0.078 nats is a real signal about sinks and sliding
  windows" — RETIRED.** Against a chunk-free reference the path sits at
  0.00288 nats. The chunked oracle it had been compared against is 6×
  further from the truth than the path it was judging; the gap tracked
  the oracle's chunk-boundary count.
- **"Chunked scoring breaks on sliding-window families" — mechanism
  RETIRED.** The measurement stands; widening the window past the context
  leaves the gap, and every cache class reproduces it. The cause is
  router flips, which applies to every MoE model.
- **The pre-registered KL gate is FALSIFIED**, by its own first
  measurement: it rejects NF4 experts, which this project ships
  (0.029 nats against a 0.01 threshold). The threshold was calibrated
  from a signed NLL difference and applied to a full-vocabulary KL. It is
  left textually unchanged in METHODOLOGY §13 and marked falsified rather
  than retuned.
- **"Granite reaches the Qwen3 ratio: 302 tok/s, ×1.59 with int4
  experts" (0.33.0 changelog and this page) — RETRACTED.** The int4
  experts on that row cost +0.0118 nats = +0.063 ppl against NF4 on the
  same 2048-step window, over the registered 0.05-ppl uncalibrated gate.
  The lane table read nats against the family's 0.0033-nat noise floor
  (which the row clears by 3.6×) and never against the budget; a floor
  says an effect is real, a budget says whether it ships. The pattern was
  already on record — int4-b32 experts are quality-neutral at ≥13B
  active and cost ~1.2–1.8% ppl at ≤1B active — and this row is that
  pattern. The 0.32.0 throughput table's Granite int4 rows carry the
  same delta (1.6741 → 1.6859) and are re-labelled in
  [`SERVING-THROUGHPUT.md`](SERVING-THROUGHPUT.md) and `claims.json`.
- **"Gemma-4 behaves (−0.0078 nats)" — SUPERSEDED**, and then **"Gemma-4
  is not at parity: 0.247 nats, 3× its floor" — SUPERSEDED the same
  day.** Both compared one 512-token window to one reference. Three
  windows and a three-forward test in plain transformers show the model
  has no reference at that resolution (above). What survives is the
  fp8 share, 0.046 nats. [#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359) stays open, re-scoped.
- **The 13.47× training speedup is ~7.2× against a current baseline.**
  transformers v5 fused the per-expert loop upstream, moving the baseline
  from 50.86 to 26.6 s/step. The grouped arm did not regress. Roughly
  half the published multiple is now upstream's work.
- **`docs/INFERENCE.md`'s decode grid is superseded** for decode by the
  pipelined and paged engines (that document says so itself).
- **"int8-offload posts the best training eval" is confounded** — the
  audit found an evaluator offset the same order as the effect, and the
  bundle's CSV mislabels host for half the repeat jobs.
- **Informed hot sets did not replicate on an A6000** with a 128-expert
  model and were withdrawn as evidence there.

---

## What is open

- **#344 — Gemma-4 fails to load on 2 of 6 rented hosts** with
  `CUDA error: invalid argument`, after the experts quantise. A 2 GiB
  host-hop fix was merged and reverted the same day: the model's largest
  tensor is 1.375 GiB, so it never triggered. The live lead is that CUDA
  reports asynchronously, so the traceback site need not be the faulting
  kernel.
- **#341 — a flaky end-to-end KV test** (unseeded inputs, an f32-mode
  tolerance applied to the fp8 default on sm_120).
- **No shipped tool bakes the arena.** Reproducing the training receipt
  from published artifacts still needs a quantise-and-emit step you write
  yourself.
- **[#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359) — Gemma-4, re-scoped**: (1) DONE in 0.32.0 with
  grouped-nf4-gemm 0.26.0 — 32-wide key scales on the 512-dim heads
  take the paged path from 3.59239 to 3.57228 nats on the P26b window;
  (2) still open: a parity instrument that survives batch-shape
  variance — a long window, or matched routing — before any verdict is
  quoted for this family.
- **Several older documents carry open debts of their own**, and say so:
  `POST_AUDIT_WORK_QUEUE.md` (quarantines Q1–Q4 in force),
  `TRAIN_PLACEMENT_CERTIFICATE.md` (a scoped S10 — one same-host bf16
  pair unexplained by any measured mechanism), `LAYOUT_FACTS.md`
  (full-run training determinism UNKNOWN).

---

## Two things about the documentation itself

**Anchored documents are never edited in place.** Several docs here carry
an OpenTimestamps footer, and their bytes must keep matching their proof.
On 2026-09-03 three PRs appended a serving-parity section to the anchored
`support_matrix.md`; that content now lives in
[`SERVING-PARITY.md`](SERVING-PARITY.md) and the anchored file is
restored to its anchored bytes. The precedent for doing it this way is
`ARCHITECTURE_SUPPORT.md`, which exists as a separate file for exactly
this reason.

Separately, and predating that: `support_matrix.md`'s footer discloses a
pre-footer content hash that no longer matches the file's pre-footer
bytes. That discrepancy is older than this cleanup and is **not** fixed
here, because fixing it means editing an anchored document. It is
recorded so a reader is not surprised by a failing check.

**`measured-private` is not a synonym for measured.** The serving speed
numbers, the calibrated-int4 quality numbers and the head-to-head against
vLLM come from a private audit tree. They are real runs with real
receipts that this repository does not carry, and they are labelled that
way in `claims.json`. Treat them as you would any number you cannot
check.
