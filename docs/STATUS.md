# Status — what this package does, what changed, what is open

**As of 2026-09-05, version 0.35.2** (the version of record is
`pyproject.toml`'s). One page. The README argues the case; this page
states the position. Every line has an entry in
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

**Training on real weights is receipted per family** (lane tp1, 2026-09-05,
one rented RTX 5090 under the shipped 0.35.0 / 0.30.0 code; **measured** —
receipt [`bench/train-parity-20260905/tp1/`](../bench/train-parity-20260905/tp1/README.md),
table in its [`RESULTS-tp1.md`](../bench/train-parity-20260905/tp1/RESULTS-tp1.md);
register `e4b.train.parity.tp1.<family>.<arm>.2026-09-05`; all six families
through the registered arms, 18 result lines, every attempt a row, nothing
pending). Each family goes through the direct
`load_moe_4bit_streaming` + `verify_moe_4bit(strict=True)` path on real
weights, then `reference` (the per-expert loop), `fused`
(`enable_fast_train(dgrad=True)`) and `batched` (`enable_batched_train`)
for 60 steps on the registered `clinical` text, judged in the registered
units — |Δ final train loss| ≤ 0.05 and median step-wise |Δ| ≤ 0.05 against
the family's own reference, same box — with cost reported and never gated.
**The fused path PASSES on every family that has one:** Granite-3.1-3B-A800M
(0.01329 / 0.01270; ×5.86 per step at ×0.20 J/step — on the corrected-counter
re-run after a HARNESS_ERROR first attempt, a closure bug in the harness's
kernel counter, amendment 3, not the shipped code; the family's first direct
real-weight load and first training receipt), OLMoE-1B-7B-Instruct
(0.01327 / 0.01249; ×3.22 per step at ×0.34 J/step — the first reading on a
registered text with real weights for that family), Qwen3-30B-A3B resident
on the 32 GB card (0.01315 / 0.01050; ×2.56, ×0.41 J/step), Gemma-4-26B-A4B-it
— the `-it` checkpoint, loaded on that host without the #344 fault —
(0.02385 / 0.04742: inside the band by 0.0026, the tightest cell, read with
that margin; ×2.37, ×0.48 J/step) and Mixtral-8x7B-Instruct under
`offload=True` (0.00953 / 0.00945; ×1.26 at ×0.49 the reference loop's peak
VRAM — its first training receipt, and the family enters the capability's
list on it). **The batched path PASSES where its kernel engages and is VOID
where it does not:** PASS on Granite-3.1-3B-A800M (0.01553 / 0.01681;
×3.93 — the family's first direct real-weight load) and Mixtral
(0.00766 / 0.01057; ×1.27); VOID on OLMoE, Qwen3 and Gemma-4 —
`enable_batched_train` patched every layer, but `engines/batched.py` falls
back to the reference forward per call above `_PAD_WASTE_LIMIT` with no
counter, and on ≈100-token rows over 64 or 128 experts the kernel was
reached on only some layers (Gemma-4: on 9 of 60 steps on none) — a VOID
row carries no parity number, however its loss curve reads. **gpt-oss
`fused` / `batched` are REFUSED** (0 patched: the loader builds its experts
bare); attention-only QLoRA over its frozen experts trains with the stacks
bit-exact (10.75 GB hashed), and the kernel package's experimental MXFP4
route trains its experts on its own text with the step-0 canary passing
(top-1 0.906, KL 0.025) and provenance holding — experimental, never
licensed. The capability list (`qlora-fused-moe-experts.model_families` in
[`capabilities.json`](capabilities.json)) is exactly the families whose
`fast_train` is `supported`: `olmoe`, `qwen3_moe`, `gemma4_text`, `mixtral`,
`granitemoe` — the last two entered on this lane; `gpt_oss` stays out. No
convergence claim, no cross-family ratio, no training throughput position;
a PASS is a PASS on one text.

**Training support is stated per path, never as a flat flag** (phase directive
2026-09-05): the tp1 receipt classifies every row as one of OK / REFUSED /
HARNESS_ERROR / ALARM / OOM / NOT_RUN / EXPERIMENTAL, mechanically from
its artefacts, with the parity verdict as a separate column; Granite's
first fused attempt stays a HARNESS_ERROR row beside its re-run (and the
follow-up script's own abort between them, amendment 5, is listed in the
re-run's reason), and every amendment stays visible in the bundle's
README. The same reading, per family and per path:

| model_type | quantize | reference_train | fast_train (the headline path) | batched_train | nvme_train | native_mxfp4_train |
|---|---|---|---|---|---|---|
| `olmoe` | supported (tp1) | supported (tp1; `e4b.train.olmoe-converges`) | **supported** — tp1 OK · PASS on the registered text with real weights (`e4b.train.parity.tp1.olmoe.fused.2026-09-05`) | **void** — tp1 OK · VOID: the `_PAD_WASTE_LIMIT` fallback engaged without a counter (`…olmoe.batched…`) | not_tested (the arena ladder is measured-private, no shipped bake) | n/a |
| `qwen3_moe` | supported (tp1, resident on a 32 GB card; flagship) | supported (tp1; flagship) | **supported** — tp1 OK · PASS resident on one 5090 (`…qwen3.fused…`), beside the flagship's five datasets (`e4b.train.flagship-matrix`) | **void** — tp1 OK · VOID: the kernel reached on a fraction of the layers every step (`…qwen3.batched…`); the dgrad-gate trajectory stands on its own fixture | not_tested (measured-private) | n/a |
| `gemma4_text` | supported (tp1: the `-it` checkpoint loaded on this host, no #344; flagship: base) | supported (tp1; flagship) | **supported** — tp1 OK · PASS with the step-wise median inside the band by a small margin (`…gemma4.fused…`), beside the model-2 flagship (`e4b.train.flagship-matrix`) | **void** — tp1 OK · VOID: on some steps no layer reached the kernel (`…gemma4.batched…`) | not_tested (measured-private) | n/a |
| `granitemoe` | supported (tp1: the first direct real-weight load) | supported (tp1 OK) | **supported** — tp1 OK · PASS on the corrected-counter re-run (`…granite.fused…`; attempt 1 a kept HARNESS_ERROR of the harness's counter, `…granite.fused.attempt1…`, amendment 3) — **entered `model_families` on it** | supported — tp1 OK · PASS (`…granite.batched…`) | not_tested | n/a |
| `gpt_oss` | supported (bare `GptOssExperts4bit`; tp1) | refused — no `ExpertsLoRA`; attention-only QLoRA trains (`…gptoss.attn_only…`, OK · no pair) | refused — `enable_fast_train` returns 0 (`…gptoss.fused…`, REFUSED) | refused — `enable_batched_train` returns 0 (`…gptoss.batched…`, REFUSED) | refused — `enable_mxfp4_nvme_residency` refuses bias-carrying modules (#402; it had defaulted to the V4 epilogue, #397), `enable_nvme_train_residency` refuses bare modules, and the `arena_train=True` wrap is refused on structure | **experimental** — grouped-nf4-gemm's `ExpertsMxfp4LoRA`; tp1 canary and provenance passed on its own text (`…gptoss.mxfp4…`, EXPERIMENTAL); never licensed |
| `mixtral` | supported (tp1: the first real-weight pass through the `w1/w3/w2` fusion, `offload=True`) | supported (tp1, offload) | **supported** — tp1 OK · PASS under offload at half the reference loop's peak VRAM (`…mixtral.fused…`); **entered `model_families` on this row** | supported — tp1 OK · PASS, the kernel reached everywhere (the 8-expert shape; `…mixtral.batched…`) | not_tested | n/a |

Each cell is one of `supported` (completed under the registered protocol with a PASS/OK receipt), `refused` (with the reason), `void` (ran, unreadable), `harness_error`, `not_tested`, `experimental`, `n/a` — per path, never a flat flag; the machine-readable form, with the claim id behind every `supported` / `void` / `refused` cell, is `training_support` in [`capabilities.json`](capabilities.json), validated by `scripts/check_capabilities.py`, and `model_families` is exactly the families whose `fast_train` is `supported`. Row statuses in the tp1 receipt are one of OK / REFUSED / HARNESS_ERROR / ALARM / OOM / NOT_RUN / EXPERIMENTAL with the parity verdict (PASS / FAIL / VOID) as a separate column.

**Against Unsloth, end-to-end, on one identical training problem** (lane
p38, 2026-09-05, one rented RTX 5090, box 49975389; **measured** — receipt
[`bench/h2h-20260905/p38/`](../bench/h2h-20260905/p38/README.md), table in
its [`RESULTS-p38.md`](../bench/h2h-20260905/p38/RESULTS-p38.md), the
pre-registration verbatim as its `PREREG.md`; register
`e4b.train.h2h.unsloth.qwen3.5090.2026-09-05` with its `.quality-n60`,
`.curve-n200` and `.e4b-internal-parity` rows and one row per arm).
Qwen3-30B-A3B at one pinned revision, the registered `clinical` fixture
tokenised once and asserted by sha in every arm, seq 512, r 8 / α 16 on
attention q/k/v/o and every expert, the router frozen, 321,257,472
trainable parameters asserted, the same optimizer, LR, batch, steps,
precision and held-out eval on both sides — experts4bit-qlora 0.35.0 +
grouped-nf4-gemm 0.30.0 (the fused `dgrad` path with NF4 attention, the
shipped `TRAIN_ATTN_4BIT` mechanism) against Unsloth 2026.9.2 +
unsloth_zoo 2026.9.1 (its 4-bit MoE path, `native_torch` backend; the two
stacks' transformers/peft versions differ and are recorded). **At 60 steps:
s/step ratio Unsloth/e4b 1.413** (2.151 vs 1.522 s — e4b faster per step at
this workload), peak VRAM 21.371 vs 23.141 GB, 157.1 vs 224.7 J/step, time
to a held-out loss of 0.32 92.5 vs 130.3 s, held-out loss comparable
(0.2923 vs 0.2975, |Δ| 0.0052 ≤ the 0.05 reading threshold). **At 200 steps
the curves separate in Unsloth's favour: 0.2713 vs 0.2881** — e4b's
flattens near 0.29 from step 60 while Unsloth's keeps falling. That row is
quoted beside the position wherever the position is quoted; its causes —
the eval schedule, the checkpointing mode, the two stacks' transformers /
peft versions, the expert adapter's precision (bf16 on this side, because
the loader passes the model dtype to `ExpertsLoRA`; fp32 on Unsloth's) —
are candidates, not established. e4b's own fused-vs-reference pair passes
its band on that box (0.00131 / 0.01138, ×2.92 per step; informational,
tp1 owns the licence). The pre-registration predicted the opposite sign at
this workload and said the finding ships either way; it does. One workload
(≈86 tokens per step, batch 1, resident), one box, one family: no general
speed claim, nothing licensed, and the 2026-08-26 "1.17× ahead" memory
(never a claim) is disqualified as a comparison.

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

**Serving speed**, Qwen3-30B-A3B on a rented RTX 5090: the licensed
position is the census's, below — **×2.067 at B=1 (238.1 tok/s on box
49916675) and ×2.602 at B=16 (1327.5 tok/s)**, the streamed-calibrated
stack bo6c licensed on both texts
(`e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05` /
`e4b.serve.census.bo7.qwen3.b16.5090.2026-09-05`, **measured**). The
2026-09-03 numbers this paragraph used to lead with — about 100 tok/s on
the NF4 baseline, 204.6 tok/s with calibrated int4 attention and
round-to-nearest int4 experts, about 1,238 tok/s aggregate at B=16 — are
real and stand as measured (`e4b.serve.b1.qwen3-30b.int4attn-calib.5090`,
**measured-private**; `e4b.serve.b16.qwen3-30b.int4.5090`, **measured**),
but **that RTN-int4 configuration class later FAILED the registered gate
on its second text** (lane bo5, +0.063 ppl on C4 validation, below) and
**is not the licensed stack**: quote them as the speed of an unlicensed
configuration on its box, never as the position. On the 2026-09-04
validation box the same class went from 156.1 to 177.9 tok/s at B=1
(×1.14) when 0.34.0's round-2 fold started engaging on the calibrated int4
attention it had silently skipped (#375) — measured, receipt in the bo3
bundle below, the same caveat.

**Against vLLM, same box, same session, identical prompt token ids** (lane
p37, 2026-09-05, Vast 49975016, an RTX 5090 on an EPYC 7Q83 host;
**measured** — receipt [`bench/h2h-20260905/p37/`](../bench/h2h-20260905/p37/README.md),
table in its [`RESULTS-p37.md`](../bench/h2h-20260905/p37/RESULTS-p37.md),
the pre-registration verbatim as its `PREREG.md`; register
`e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05` and one row per arm):
vLLM 0.28.0 serving Qwen's GPTQ-Int4 checkpoint (Marlin, default CUDA
graphs) decodes at **286.0 tok/s at B=1 and 2030.0 aggregate at B=16**
(fp8-KV arm 300.9 / 2206.5; eager 20.8 / 322.5); this package's NF4 control
on the same box reads 113.4 / 500.1 (repeats 113.5 / 499.9) — **vLLM /
e4b-NF4 2.52 at B=1 and 4.06 at B=16**, the only licence-free ratio the lane
can quote, and it is against the slowest configuration this package ships.
**The ratio against the licensed stack is not quoted:** every licensed e4b
arm on that box is VOID under the pre-registered pack-fingerprint rule —
the streamed calibration there packed 11522 gptq / 766 rtn expert matrices
where the licensed pack (bo6b, bo6c, bo7) reads 11512 / 776, the same
recipe but not the licensed bytes, and speed cannot inherit a licence. The
recipe's speed on that box — 236.4 tok/s at B=1 (repeat 235.8), 1305.3 at
B=16, ×2.08 / ×2.61 over its own NF4 control, bo7's ×2.067 / ×2.602
reproduced within 1% — is measured and **unlicensed**, an observation never
divided into a position. **Amendment 3 (pre-registered after `TP_DONE`)
ran the registered K8 gate on that box's own pack and it FAILED**
(`e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05.gate`, measured): wikitext
Δ −0.0230 ppl, pass; C4 validation **Δ +0.1093 ppl against the +0.05
budget** — where the licensed pack read −0.0662 on the same window. VOID
stands; no ratio against the licensed stack exists on that lane; the
recipe re-derived on another host is a different pack with a different
verdict (an open item, below). The 2026-09-03 comparison (`e4b.serve.h2h.vllm.same-box` (superseded): ×1.47 /
×1.55 on a different box against the 0.27.0/0.21.0 RTN stack, vLLM version
unrecorded) is **superseded** for current-position use and stays true as
measured. Quality is quoted, never equated: neither checkpoint is scored on
that lane.

**Per-family throughput is now measured in-repo** (2026-09-04): six
families under one protocol on one rented 5090 class, with every refused
arm named — that list became the build-out, and 0.33.0 ships it. Table
and receipt: [`SERVING-THROUGHPUT.md`](SERVING-THROUGHPUT.md).

**One more family reaches the reference's ratio with the build-out,
and one claimed to has been retracted** (0.33.0; **measured** — the validation lane's full receipt, every run and
its verdict in the gate's own units, is
[`bench/hybrid-g9/throughput-20260904/bo3/`](../bench/hybrid-g9/throughput-20260904/bo3/README.md),
and the licensed-best table is in [`SERVING-THROUGHPUT.md`](SERVING-THROUGHPUT.md)): Gemma-4-26B-A4B at 121 tok/s B=1 with int4 experts +
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
its combined number is 259.1 tok/s at B=1 (×1.37) and 1689.6 at B=16
(×1.18), +0.019 ppl — measured, in the same receipt.

**The second text (2026-09-04, lane bo5; measured — receipt
[`bench/hybrid-g9/throughput-20260904/bo5/`](../bench/hybrid-g9/throughput-20260904/bo5/README.md),
table in its [`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo5/RESULTS.md)).**
The registered K8 gate is in perplexity: an uncalibrated arm |Δppl| ≤ 0.05
on every text, a calibrated pack ≤ +0.05 on every text with an improvement
claimable only when it holds with the same sign on two; nats are quoted
beside the verdict and never change it. bo3 had scored one text; bo5 scored
C4 validation on one box (49841214, the same 5090 class) for every
calibrated pack bo3 left at "one text", and **every one FAILS as
registered**: Qwen3's `all` stack +0.063 ppl (+0.0038 nats — every
attribution arm on that text is inside the family's 0.0095-nat floor and
the exact folds alone read −0.073, so the reading is noise, not a component;
the verdict is not retuned), Mixtral's `all` +0.116 ppl, and Granite's
C4-calibrated int4 experts (e4b#384, draft) +0.387 ppl at 10× the floor —
that route is refused. Mixtral's *uncalibrated* int4-expert stack — called
licensed in P30 and bo3 on wikitext (−0.046 ppl there) — fails the second
text by 0.008 (+0.058 ppl, +0.0070 nats; this family's floor is
unmeasured), so that label is **withdrawn** under the rule as written,
pending a decision on re-registering the gate in nats against a measured
floor. Licensed and re-measured on the bo5 box: Granite's NF4 stack
294.1 tok/s B=1 / 1736.1 B=16; gpt-oss's MXFP4 store under the route rule
(GEMV for single rows, NF4 kept for batched rows) 173.3 (×1.270) / 719.7
(×0.971 — the ×0.81 B=16 penalty is recovered), quality gate open on that
family as before. Measured speed of configurations NOT licensed under the
registered rule: Qwen3 `all` 204.1 / 1251.6 (with #385's glue, ×1.057 at
B=1 on that box), Mixtral `all` 123.3 / 377.3, Mixtral's int4-expert stack
112.0 / 376.5. #387's fused q/k/v is quality-clean (−0.0011 nats on
Granite, +0.0002 on Mixtral) and buys nothing (Granite ×0.968 B=1, Mixtral
×1.032) — it stays a draft.

**Closing the gap, not the gate (2026-09-04, lane bo6; measured — receipt
[`bench/hybrid-g9/throughput-20260904/bo6/`](../bench/hybrid-g9/throughput-20260904/bo6/README.md),
table in its [`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo6/RESULTS.md)).**
The registered gate stays in perplexity; the int4-expert arms that failed
their second text with round-to-nearest experts were re-run with
per-expert GPTQ calibration (e4b#384) on one box, against NF4 re-scored on
that box. **Sequential calibration is the mechanism that ships** (0.35.0):
on Qwen3-30B-A3B the calibrated experts alone read c4val1 +0.150 ppl
(FAIL) when every layer's Hessian is accumulated against the unquantised
prefix and packed afterwards, and −0.050 (pass on that text) when each
layer chunk is packed before the next chunk's Hessians accumulate — the
same box, batches and damping; the order alone moves the reading 0.200 ppl
and flips the verdict
(`e4b.serve.buildout.bo6.qwen3.calibration-order.c4val1.2026-09-04`; the
instrument is run-to-run deterministic there,
`e4b.serve.buildout.bo6.qwen3.k8-deterministic.5090.2026-09-04`). More
calibration text helps (64k tokens −0.211, 256k −0.141; damping 0.1 fails
at +0.054), and none of those is claimed as an improvement until wikitext
agrees. **Qwen3's calibrated int4 experts pass the registered gate under
it:** the full calibrated stack — calibrated experts + calibrated int4
attention + folds + epilogue + glue — passes on both texts (wikitext
−0.060 / c4val1 +0.035,
`e4b.serve.buildout.bo6.qwen3.all-calibexp-allatonce.k8.2026-09-04`; that
pack was calibrated all-at-once) and reads 158.0 tok/s at B=1 / 993.6 at
B=16 on that Threadripper-hosted box (`e4b.serve.buildout.bo6.qwen3.b1.5090.2026-09-04` /
`e4b.serve.buildout.bo6.qwen3.b16.5090.2026-09-04` — quoted with its box, no ratio: the lane has no NF4 speed arm).
**Qwen3-30B-A3B's licensed serving stack is the streamed one** (bo6c,
2026-09-05, same box, receipt in the same bundle): sequentially calibrated
int4 experts at 64k C4-validation tokens + C4-calibrated int4 attention +
round-1/2 folds + router epilogue + decode glue — wikitext −0.053 /
c4val1 −0.066 ppl (−0.0083 / −0.0040 nats, both inside the 0.0095-nat
floor): pass on both texts as registered, **licensed under the unchanged
gate**, at parity or better on both texts, no improvement claimed by a
number
(`e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`).
The 64k experts-only pack is at parity on wikitext (−0.0002) and −0.211
in-domain (`e4b.serve.buildout.bo6c.qwen3.calibexp-streamed-64k.k8.2026-09-05`),
and the streamed calibration is itself run-to-run deterministic
(`e4b.serve.buildout.bo6c.qwen3.calib-deterministic.5090.2026-09-05`).
That stack's speed is not on this lane — bo7 measures the calibrated stack
at the hook's 16k-token default, not 64k. Granite
stays NF4 (bo5: calibrated experts +0.387 ppl, not closable with this
lever). **Sequential calibration closed Qwen3's gap and did not close
Mixtral's:** its calibrated int4-expert stack passes the in-domain text
(c4val1 +0.039) and fails the out-of-domain one (wikitext +0.077 ppl,
+0.0234 nats; this family's floor is unmeasured) — FAIL as registered,
measured, not licensed
(`e4b.serve.buildout.bo6.mixtral.lic-calibexp-streamed.k8.2026-09-04`),
the mirror image of bo5's RTN stack; not a reference shift (same window
sha, NF4 agrees with bo5c to 0.001 nats). Next levers are not gate
changes: a per-expert NF4 fallback for the largest-residual experts, or
the 64k calibration set scored on wikitext. The lane's Mixtral speed
arms and its 64k-token arm were killed by their own per-arm alarms
during the ~85-min streamed calibration and were not measured — a
harness limit, not a model result; bo7 dropped those arms too. Qwen3's
NF4 reference sits 0.006 nats from bo5's
on the identical window while Mixtral's agree to 0.001 — that shift stays
open, and no sub-0.01-nat number is compared across lanes.

**The throughput census (2026-09-05, lane bo7; measured — receipt
[`bench/hybrid-g9/throughput-20260904/bo7/`](../bench/hybrid-g9/throughput-20260904/bo7/README.md),
table in its [`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo7/RESULTS.md)).**
Speed only, under the **shipped** code (0.35.0 + 0.30.0 at their `main`,
hook v6 at its 16k default), all six families on one rented RTX 5090 (EPYC
7Q83 host, instance 49916675), 48 arms at B=1 and B=16, every ratio to
that family's own NF4 arm on that box and every licence label copied from
this register — bo7 licenses nothing, and no bo3/bo5/bo6 number is divided
into a bo7 number. What the box says, three axes per licensed best (ratio
over NF4 on this box; rental-measured tok/s on this box; anchor-class
projection — which exists only for Qwen3-30B at B=1 and is not computed
here): **Granite's licensed stack** (NF4 experts + folds + epilogue) is
×1.341 at B=1 (304.9 tok/s) and ×1.160 at B=16 (1836.8;
`e4b.serve.census.bo7.granite.b1.5090.2026-09-05` / `e4b.serve.census.bo7.granite.b16.5090.2026-09-05`); **OLMoE's position is NF4**
(282.5 / 1347.5, ×1.000; `e4b.serve.census.bo7.olmoe.b1.5090.2026-09-05` / `e4b.serve.census.bo7.olmoe.b16.5090.2026-09-05`) because nothing above it is licensed on this
register — the tp row's "best licensed" label predates the two-text
clause and its calibrated attention is refused on this family, so its full
stack is ×2.070 / ×2.289 measured, not licensed; **gpt-oss's quoted best is
its own reference arm** (NF4 + exact folds, 144.5 / 761.6; `e4b.serve.census.bo7.gptoss.b1.5090.2026-09-05` / `e4b.serve.census.bo7.gptoss.b16.5090.2026-09-05`) and the MXFP4
store under the route rule reads ×1.293 / ×0.970 with the quality gate open;
**Qwen3's licensed stack** — the streamed 64k calibrated pack bo6c
licensed on both texts — measured on the same box under the lane's
amendment 2 (pre-registered 06:05Z, run after `TP_DONE`): **×2.067 at B=1
(238.1 tok/s; anchor-class projection 159.2 × 2.067 ≈ 329 tok/s, a
projection from an uncertified class) and ×2.602 at B=16 (1327.5 tok/s;
no anchor projection)** — `e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05` / `e4b.serve.census.bo7.qwen3.b16.5090.2026-09-05`. Its
speed is identical to the lane's 16k arm (4.20 vs 4.20 ms; 1327.5 vs
1338.8, within 1%) and to the RTN stack: a calibrated pack's kernels do
not depend on the calibration size, as the amendment predicted — the pack
changes the values, not the kernel or the bytes. **Gemma-4 has no K8
instrument, so no arm carries a K8 licence**; the register's position with
that caveat is the exact round-1 fold + epilogue on NF4 (`r1epi`), ×1.281
at B=1 (103.6 tok/s) and ×1.106 at B=16 (675.8;
`e4b.serve.census.bo7.gemma4.b1.5090.2026-09-05` / `e4b.serve.census.bo7.gemma4.b16.5090.2026-09-05`), and the quoted int4 best
(bo3's `stack`) reads ×1.705 / ×1.697 measured, no quality verdict —
Gemma-4-it loaded on this host without the #344 fault. **Mixtral's position
is NF4** (50.3 / 191.4, ×1.000; `e4b.serve.census.bo7.mixtral.b1.5090.2026-09-05` /
`e4b.serve.census.bo7.mixtral.b16.5090.2026-09-05`): the exact folds are ×1.062 / ×1.018 but unscored as a combined
arm, the RTN int4 stack ×2.329 / ×1.959 and the calibrated-attention stack
×2.597 / ×1.962 are measured, not licensed (bo5's second-text FAILs stand),
and the calibrated-expert arms were dropped under the lane's amendment
(FAIL as registered on bo6b, ~85-min calibration) and print as a row that
says so. Two int4 facts the box adds: a calibrated int4-expert pack costs
nothing over an RTN one in speed (Granite 2.04 vs 2.05 ms, Qwen3 4.197 vs
4.204), and the streamed calibration's pack counts reproduce across hosts
(Granite 2524/36, Qwen3 10820/1468 — bo5's and bo6's counts). All 50 arms
ran — 48 in the lane (`TP_DONE` 07:00Z, 5.0 h) and amendment 2's two
(`TP2_DONE` 08:52Z) — with no alarm, refusal or traceback.

---

## What changed — retired, superseded, corrected

- **The 2026-09-04 Qwen3 "best licensed" throughput rows are SUPERSEDED**
  (2026-09-05): `e4b.serve.tp.qwen3.b1.5090.2026-09-04` (superseded) and
  `e4b.serve.tp.qwen3.b16.5090.2026-09-04` (superseded) now point at the
  census rows of the licensed stack
  (`e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05` /
  `e4b.serve.census.bo7.qwen3.b16.5090.2026-09-05`); the RTN-int4 class they
  quoted failed its second text on bo5, and their numbers stand as measured.
  The other four families' 2026-09-04 rows keep their numbers with the
  "best licensed" label withdrawn in the sentence (measured, not licensed),
  and every active claim whose sentence asserts a licence now names its K8
  verdict row (`licensed_by`) — `scripts/check_claims_register.py` holds the
  register to that and to its own structure (evidence paths that exist,
  dated measured rows, successors that resolve, no "pending" on an active
  row). The 2026-09-03 single-stream stack this page led with is the same
  RTN class and is quoted above as unlicensed speed.
- **The 2026-09-03 same-box vLLM comparison is SUPERSEDED** (2026-09-05):
  `e4b.serve.h2h.vllm.same-box` (superseded; ×1.47 / ×1.55 on box 49702459,
  the 0.27.0/0.21.0 RTN stack, vLLM version unrecorded) points at
  `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05` — lane p37, vLLM 0.28.0
  pinned, identical prompt ids, every knob recorded — which quotes vLLM
  against this package's NF4 control (2.52× / 4.06×) and, as pre-registered,
  quotes **no** ratio against the licensed stack because its arms were VOID on
  that box (the pack fingerprint 11522/766 against the licensed 11512/776).
  Amendment 3's gate on that pack failed on C4 validation (+0.1093 ppl;
  `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05.gate`), so no
  head-to-head against the licensed stack exists on that lane.
- **The 2026-08 "vLLM 6.31× ahead" figure is RETIRED by id** — the
  retired row is `e4b.retired.vllm-6.31x-ahead`: template prompts and a different box;
  the same-box, same-prompt comparison (`e4b.serve.h2h.vllm.same-box` (superseded))
  supersedes it and now names it, with its own two limits stated (vLLM
  version unrecorded; the e4b arm is the unlicensed RTN class).
- **`e4b.retired.13.47x-training-speedup` is `retired`, not `superseded`**:
  the "about 7.2× against a current baseline" restatement has no receipt of
  its own in this repository, so there was no successor row to name; the
  correction below stands as written.
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
- **"4-bit on a card that already fits is a 1.2–2.3× energy penalty:
  NF4 is storage-only and the GEMM runs in bf16 either way" —
  SUPERSEDED, number unchanged** (2026-09-04). The measurement stands as
  its receipt made it — one OLMoE-dims expert projection on an RTX A2000,
  dequantize-then-`linear` and a bitsandbytes 0.50-dev fork build's
  `matmul_4bit` routing against native bf16 — and is re-registered with
  that comparator and version named as `e4b.train.energy-honest.scoped-a2000`
  (`e4b.train.energy-honest` is `superseded`, pointing at it). What is
  withdrawn is the mechanism sentence as a universal: bitsandbytes ≥ 0.50.0
  CUDA inference can consume packed 4-bit weights directly for supported
  ordinary 2-D cells, routed grouped MoE execution is a separate contract,
  and training's input gradient is separate again
  ([`BITSANDBYTES.md`](BITSANDBYTES.md)). The receipt names its build only
  as "0.50.0.dev0 / the fork", so which path its 4-bit arm exercised is not
  recoverable from it — remeasure with a recorded version:
  [#392](https://github.com/pjordanandrsn/experts4bit-qlora/issues/392).
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

- **The licensed serving pack does not reproduce bit-for-bit across boxes.**
  Lane p37 (`bench/h2h-20260905/p37/`) re-derived Qwen3's streamed 64k
  calibrated pack with the same recipe on another host and read 11522 gptq /
  766 rtn expert matrices against the licensed 11512 / 776 (bo6b, bo6c,
  bo7): ten of 12,288 matrices crossed the `min_rows` threshold the other
  way — and the registered K8 gate run on that pack (amendment 3,
  `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05.gate`) **fails on C4
  validation: +0.1093 ppl against the +0.05 budget** (wikitext −0.0230,
  pass), where the licensed pack read −0.0662 on the same window and the
  two boxes' NF4 references agree to 0.0002 ppl. The reference travels; the
  calibrated pack does not: the streamed calibration recipe does not
  reproduce its licence across hosts. bo6c's licence stands on its box as
  measured; what is open is the recipe's host-dependence — same checkpoint,
  same text, same knobs, a different pack and a different verdict on
  another host ([#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405)).
- **`enable_batched_train`'s engagement envelope.** It falls back to the
  reference forward per call above `_PAD_WASTE_LIMIT` (`engines/batched.py`);
  a positive return value is a patch count, not kernel engagement. In the
  code tp1 measured (0.35.0) the fallback was silent: OLMoE's, Qwen3's and
  Gemma-4's `batched` arms are VOID on exactly this (on 9 of 60 Gemma-4
  steps no layer reached the kernel); it engaged everywhere only on
  Mixtral's 8-expert and Granite's 40-expert shapes. 0.35.1 (#402) makes the
  fallback countable — `batched_fallback_stats(model)` — so a batched arm can
  be read from the path itself; the envelope (which shapes engage) is still
  open and is measured, not tuned.
- **gpt-oss has no expert-LoRA training path under the shipped e4b code.**
  `enable_fast_train` and `enable_batched_train` refuse it (0 patched), by
  design, and 0.35.1 (#402) makes the refusal a contract: a wrapper whose
  base breaks the stock epilogue raises `EpilogueContractError`, and
  `enable_mxfp4_nvme_residency` refuses bias-carrying modules (it had
  defaulted to the V4 epilogue, #397). What stays open is a gpt-oss-aware
  adapter; the kernel package's `ExpertsMxfp4LoRA` route is the experimental
  alternative (tp1: canary and provenance pass, never licensed).
- **#344 — Gemma-4 fails to load on 2 of 6 rented hosts** with
  `CUDA error: invalid argument`, after the experts quantise. A 2 GiB
  host-hop fix was merged and reverted the same day: the model's largest
  tensor is 1.375 GiB, so it never triggered. The live lead is that CUDA
  reports asynchronously, so the traceback site need not be the faulting
  kernel.
- **#341 — a flaky end-to-end KV test** (unseeded inputs, an f32-mode
  tolerance applied to the fp8 default on sm_120).
- **[#392](https://github.com/pjordanandrsn/experts4bit-qlora/issues/392) —
  the energy receipt does not record its bitsandbytes build.**
  `docs/METHODOLOGY.md` names the build only as `0.50.0.dev0` (§1) and
  "the fork (bnb 0.50-dev)" (the packaging note covering §9–§10), with no
  commit; the harness prints the GPU name, not `bitsandbytes.__version__`. Until it is rerun on a recorded release,
  `e4b.train.energy-honest.scoped-a2000` is a one-card, one-build number.
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

**`measured-private` is not a synonym for measured.** The 2026-09-03
single-stream serving speed numbers and the calibrated-int4 quality numbers
come from a private audit tree (the same-box head-to-head against vLLM is
now in this repository, `bench/h2h-20260905/p37/`; its 2026-09-03
predecessor was private and is superseded). They are real runs with real
receipts that this repository does not carry, and they are labelled that
way in `claims.json`. Treat them as you would any number you cannot
check.
