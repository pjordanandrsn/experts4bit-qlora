# Status — what this package does, what changed, what is open

**As of 2026-09-21, version 0.36.4** (the version of record is
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

**The field-recipe position, on the 2026-09-19 kernel cut** (lane tp4
re-run, `bench/tp4/RESULTS-tp4-p46cut.md`; ordered by P46's decision rule
and extended to the second box by its amendment 2; register
`e4b.train.h2h.unsloth.qwen3.5090.2026-09-19` with its `.quality-n20`,
`.e4b-internal-parity`, `.secondary-mb1`, per-arm and `.coverage` rows).
One rule changed in the kernel package: the grouped-LoRA delta's `auto`
path now pads unless the padded block would not fit, where it used to send
any call past a 4× padding-waste ratio to a per-expert Python loop
(grouped-nf4-gemm 0.32.1). On **Qwen3-30B-A3B at the Unsloth notebooks'
own recipe** — alpaca, seq 2048, micro-batch 2 × accum 4, r 16, AdamW-8bit,
one RTX 5090, both frameworks training the same 642,514,944 parameters —
**e4b takes 6.4707 s/step against Unsloth's 29.0547: a ratio of 4.490**,
at 237 vs 48 tokens/s and 1,186 vs 3,341 J/step, the same peak VRAM, and a
held-out gap of 0.0283 nats (COMPARABLE). On the previous cut this arm did
not finish at all — tp4 alarmed it at 3600 s, and P43's T1 diagnosis
measured 29.30 s/step. e4b's fused path passes its own parity control on
the same box at **0.00140 nats** from its dense reference (×9.31 slower),
and that control passed on every family measured; it was the registered
condition that would otherwise have stopped the release. The gain is
family-dependent and smaller elsewhere, as pre-registered: Granite 2.853 →
2.366 (×1.21) and OLMoE 2.739 → 1.395 (×1.96), both with parity passing.
**Nothing is quoted against Unsloth on Granite or Qwen3.6** — its arm
trains 5.2 M parameters against e4b's 99.6 M there, attention only, VOID
by tp4's regime rule — nor on OLMoE, whose Unsloth arm died before its
first step. The two Unsloth positions on this page (1.413 at p38's
clinical fixture, 4.490 here) are different workloads on different cuts
and neither supersedes the other.


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

**tp2 / P40 (2026-09-06): the Unsloth head-to-head now covers all six
families, one box, one fixture** (lane tp2 / P40, one rented RTX 5090, Vast
box 50005568 on a Ryzen 7 5700X3D host, train-anchor class
`pcie-full/launch-fast`; **measured** — receipt
[`bench/h2h-20260906/tp2/`](../bench/h2h-20260906/tp2/README.md), reducer
table in its [`RESULTS-tp2.md`](../bench/h2h-20260906/tp2/RESULTS-tp2.md),
the pre-registration verbatim as its `P40-PREREG.md`; register
`e4b.train.h2h.unsloth.<family>.5090.2026-09-06` — one row per attempt under
`….arm.<framework>.<arm>`, position and `.quality-n60` rows on Qwen3 and
Mixtral, a `.footprint` row on Mixtral, `.coverage` rows on Granite and
OLMoE, `.e4b-internal-parity` rows on Granite, OLMoE, Qwen3 and Mixtral).
P38's fixture exactly (seq 512, r 8 / α 16, lr 1e-4, accum 1, N=60, held-out
every 20), tokenised once per family with that family's tokenizer; e4b at
the shipped cut a user installs today — 0.35.1 + grouped-nf4-gemm 0.30.2
from PyPI, NF4 attention via the shipped `TRAIN_ATTN_4BIT` mechanism —
against Unsloth 2026.9.2 in its own venv. **Positions exist on two
families; on the other four the statuses are the result.** Qwen3-30B-A3B:
s/step ratio Unsloth/e4b **1.457** (5.986 vs 4.108 s — e4b faster per
step), peak 21.371 vs 23.141 GB, 383.9 vs 561.4 J/step, held-out
COMPARABLE (0.2935 vs 0.3087, Δ +0.0152 ≤ the 0.05 reading threshold;
`e4b.train.h2h.unsloth.qwen3.5090.2026-09-06`, `….quality-n60`) — the
cross-lane anchor: **+3.1% from P38's 1.413, inside the pre-registered
±10%** (prediction P2 held: the sign and size of the per-step position
reproduce on a second box; P38's 200-step curve row, in Unsloth's favour,
still stands beside any position of this family). Mixtral-8x7B: **the
footprint trade leads the row** — the e4b arm trained its experts under
CPU offload at a **3.223 GB** peak (the registered design for this family,
tp1's `offload=True`; a trainable-on-smaller-cards capability, its own row
`e4b.train.h2h.unsloth.mixtral.5090.2026-09-06.footprint`) while Unsloth
ran resident (its only mode) at **29.163 GB**, and what that VRAM buys it
is speed per step: ratio **0.361** — a footprint-vs-speed trade under the
registered design, not a kernel deficit; e4b's energy is lower (298.3 vs
350.2 J/step); held-out COMPARABLE (0.2590 vs 0.2502, Δ −0.0087;
`e4b.train.h2h.unsloth.mixtral.5090.2026-09-06`, `….quality-n60`);
prediction P4 (Unsloth resident OOMs at seq 512 on 32 GB) **falsified**.
No position on the other four, and the coverage rows are results, not
empty cells: **Granite** — the comparator could not train the experts:
Unsloth's arm completed but is **VOID** — it attached LoRA to the
attention only, 2,621,440 trainable parameters against e4b's 49,807,360
with `ExpertsLoRA` on all 32 MoE layers, 0 Params4bit expert stacks, no
'Enabling LoRA on MoE parameters' banner — its MoE-LoRA path never engaged
on `granitemoe`
(`e4b.train.h2h.unsloth.granite.5090.2026-09-06.coverage`,
`….arm.unsloth.ckpt_unsloth`); e4b's arms are VALID (fused 0.641 s/step,
×5.59 its own reference). **OLMoE** — the comparator could not train the
family at all: Unsloth's process died at MoE-LoRA engage, rc=1, before its
first receipt write (HARNESS_ERROR;
`e4b.train.h2h.unsloth.olmoe.5090.2026-09-06.coverage`,
`….arm.unsloth.ckpt_unsloth`); e4b's arms are VALID (fused 0.708 s/step,
×3.73). **gpt-oss** — three
refusals, as pre-registered (P5 held): e4b's attention-4-bit refuses on
structure (96 of 96 attention projections carry a bias), its fused path
patches nothing (tp1's REFUSED row re-proven on this box), and Unsloth's
load fails on the MXFP4 weight conversion
(`e4b.train.h2h.unsloth.gptoss.5090.2026-09-06.arm.*`); the attention-only
secondary row trains (1.464 s/step, 14.687 GB, `….arm.e4b.attn_only`).
**Gemma-4** — both e4b attention-4-bit arms died on the converter's own
count check (`quantize_attention_projections_4bit converted 100
projections, expected 120` —
[#412](https://github.com/pjordanandrsn/experts4bit-qlora/issues/412); the
bf16-attention `fast_train` path stays exactly as tp1 left it), while
Unsloth's arm is OK · VALID at 3.510 s/step
(`e4b.train.h2h.unsloth.gemma4.5090.2026-09-06.arm.unsloth.ckpt_unsloth`)
— recorded with the receipt. e4b's internal fused-vs-reference parity
PASSES on all four families that ran both arms (×5.59 / ×3.73 / ×2.69 /
×1.23 per step on Granite / OLMoE / Qwen3 / Mixtral,
`e4b.train.h2h.unsloth.<family>.5090.2026-09-06.e4b-internal-parity`;
informational, tp1 owns the licence). `training_support` in
[`capabilities.json`](capabilities.json) now records the attention-4-bit
configuration per family from these receipts — inside the per-path
structure, never a flat flag. No gate, threshold or licence moved; nothing
here supersedes P38 (two boxes, two measurements, never averaged).

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

**Quality measured from the checkpoint, not from e4b's own reference
(P44, 2026-09-19).** A second instrument scores each served stack
against the family's bf16 checkpoint (full-vocabulary KL, 200 committed
prompts), after a control that first asks the reference whether it
agrees with *itself* under the two forward shapes the scorer compares.
Three readings moved the register
([`bench/p44/RESULTS-p44.md`](../bench/p44/RESULTS-p44.md)):
**gpt-oss-20b's native MXFP4 store route licenses** (0.0019 nats from a
dequant of the same bytes, against 0.0222 for the NF4 requant control;
every stratum ≤ 0.13× the control) — the first quality verdict on that
family that is not out-of-domain flattery, carried as a KL-from-reference
licence rather than a K8 one; **OLMoE's int4 recipes FAIL the second
text** (RTN int4 experts +0.255 ppl on C4; the Qwen3 calibrated recipe
+0.443 — worse than RTN, so OLMoE stays NF4); and **Gemma-4's served NF4
stack is 1.08 nats/token from the bf16 checkpoint with 36 % top-1
disagreement**, lever-independent, reproduced across three runs, where
the same instrument reads the other four families at 0.02–0.1. That is
a served-model defect, not a quantisation cost, filed as
[#597](https://github.com/pjordanandrsn/experts4bit-qlora/issues/597)
with a three-arm diagnostic to pre-register; no Gemma-4 serving position
is quoted. The per-expert error census on Granite (126 of 1,270 routed
experts carry half the GPTQ error — 9.9 %, at the 10 % threshold) and
Mixtral (flat: 22.7 % on the 16 layers measured) is data for a
per-expert fallback that is not built.

**Calibration does not rescue Gemma-4's experts, and the more principled
calibration is worse.** With all 30 expert layers quantised, KL against the
bf16 checkpoint reads **1.0772** for NF4 round-to-nearest, **1.1050** for
calibrated int4 packed all at once, and **1.1564** for calibrated int4 packed
sequentially — batches, source and Hessian budget identical, calibration
**order** the only difference. The ordering effect that licensed sequential
calibration on Qwen3-30B does not transfer to this family; it **inverts**.
Both registered predictions were refuted, and the third — which deliberately
declined to predict the direction — is why the reading stands
([`bench/p53/RESULTS-p53.md`](../bench/p53/RESULTS-p53.md)). Together with P49,
which refuted shrinking the perturbation, **both addressable axes are now
closed**: the only lever on Gemma-4's positional sensitivity remains keeping
early expert layers in high precision, and that lever is memory.

**Gemma-4's fused training path does not agree with e4b's own dense
reference, and a kernel change did not fix it.** On the same box, the same
tokens and identical trainable counts, the fused expert path and e4b's
per-expert dense reference end 0.08257 nats apart on held-out loss, with a
median step-wise gap of 0.12421, against a 0.05/0.05 band — a FAIL that
reproduces the 0.09037 / 0.11801 first measured on the previous kernel cut.
This is e4b against itself, not against another framework, which is exactly
what the control exists to catch. The direction was the one the P47–P51 lanes
predicted: Gemma-4's sensitivity is **positional and lives in the quantised
model**, not in the adapter path, so changing the adapter path was not
expected to move it and did not. [`e4b.parity.gemma4.train-internal`](claims.json) is **SUPERSEDED** 2026-09-22 by lane P56
([`e4b.parity.gemma4.train-floor`](claims.json),
[`bench/p56/RESULTS-p56.md`](../bench/p56/RESULTS-p56.md)): the disagreement is
**not attributable to the fused path**. e4b's kernel-free batched path — 13×
less composed gradient error, never touching `grouped-nf4-gemm` — fails the
same band on the same box, at 0.05425 final / 0.08487 median. Cutting the
per-op error 13× closes only 1.9× of the gap, so no achievable arithmetic
change reaches 0.05. The FAIL itself reproduced a third time (0.10223 /
0.10639), so it is real and stable; what changed is what it is evidence
*about*. P56 measures this family's training-parity floor for the first time —
**≥ 0.054 final / 0.085 median step-wise** — and `tp4_reduce.parity()` compares
against 0.05 **and against zero**, with no floor term. So #558 stands open as a
**gate** defect, not a fused-path defect, and the serving side already fixed
the same class of error by moving to a measured floor. On the same pair the
fused path is 11.65× faster per step; that is an internal comparison and no
competitive position is quoted from it.

**Gemma-4's experts take a graded store map, not one store** (lanes P47–P51,
2026-09-19; `bench/p47`–`bench/p51`, all against the bf16 checkpoint on the
same 200 prompts). This family's per-layer sensitivity to expert
quantisation spans **159×** — NF4 in layer 0 alone puts the model 0.892 nats
from the checkpoint, NF4 in layer 27 alone 0.0056 — and the cause is
positional rather than a matter of how much precision is lost: the same ~8 %
expert-branch damage costs 159× more at layer 0 than at layer 27, while an
8.5× *smaller* damage at layer 0 buys only 22 %. So no store rescues the
early layers (on layer 0 alone: int8 0.693, fp8 0.774, NF4 0.892, FP4 1.051),
e4b's own Gemma-4 modelling is faithful (0.0056 nats with 29 of 30 expert
stacks in bf16), and the tail cannot be crushed further — `moe_intermediate_size`
is 704 = 64 × 11, so no quantisation block above 64 divides it and NF4 at
block 64 is already the smallest store this package ships. What works is a
**graded map**, which the loader accepts since `quantize_layers` learned to
take a per-layer mapping: `{0..9: None, 10..19: "int8", 20..29: ("nf4", 64)}`.
**At matched expert bytes it is 1.44× better than a uniform high-precision
head** (0.1695 nats at 25.70 GB against 0.2448 at 25.21 GB), and it saves
6.65 GB against the uniform configuration that reaches the 0.05 fidelity
floor. The crossover is real and cuts both ways: with a 5-layer bf16 head
grading *loses* to uniform, because an int8 tier starting at layer 5 still
covers layers that cannot take it. **No Gemma-4 position is quoted, and the
map is an option rather than a default.** The gate originally registered for
it — K8 on the served stack — cannot be built on this family: Gemma-4's own
NLL moves 0.4 nats with batch shape against K8's 0.05 budget
(`e4b.parity.gemma4.no-reference`), and the K8 runner requires the arena
path, which refuses per-layer store maps by design. The replacement bar,
taken from configurations this package already ships (≤ 0.10 nats and top-1
≥ 0.93 — the band gpt-oss's licensed NF4 requant and the other families
occupy), is **not cleared**: the graded map reads 0.1695 nats and top-1
0.855, about one token in seven disagreeing with bf16. Two boundaries travel
with the recommendation: it applies to the **loader path only**, and its
quality is materially below every other family's shipped quantisation.

That bar has since been applied on **held-out prompts** — a fresh 100-prompt
set written after the design was fixed, disjoint from the committed set by
assertion — and the map does not clear it there either: **0.1319 nats and
top-1 0.874**, missing on both axes, on every stratum, and on the *more*
favourable of the two sets. The same run carries its own control: gpt-oss's
licensed NF4 requant, the configuration the bar was taken from, reads 0.0217
against its committed 0.0222, so the bar transfers between prompt sets to
within 2 %. What survives the gate is the shape, not a position — the graded
map still beat a matched-bytes uniform head by 1.39× on the fresh prompts.
The gate is therefore **run and not passed**, which is a stronger and less
comfortable statement than not run. Curve and rows in
[`bench/p51/RESULTS-p51.md`](../bench/p51/RESULTS-p51.md); the gate in
[`bench/p52/RESULTS-p52.md`](../bench/p52/RESULTS-p52.md).


**Serving speed**, Qwen3-30B-A3B on a rented RTX 5090: the licensed
position is the census's, below — **×2.067 at B=1 (238.1 tok/s on box
49916675) and ×2.602 at B=16 (1327.5 tok/s) vs e4b's own NF4 control on
the same box**, the streamed-calibrated stack bo6c licensed on both texts
(`e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05` /
`e4b.serve.census.bo7.qwen3.b16.5090.2026-09-05`, **measured**). Those
ratios are not a field-engine speedup. The
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

**Against vLLM, same box, same session, identical prompt token ids — CURRENT vs
CURRENT (lane P58, 2026-09-22, Vast 52069847, an RTX 5090 on an EPYC 9655 host;
**measured** — [`bench/p58/RESULTS-p58.md`](../bench/p58/RESULTS-p58.md), register
`e4b.serve.h2h.vllm-0.30.0.p58.qwen3.b1.5090.2026-09-22` /
`e4b.serve.h2h.vllm-0.30.0.p58.qwen3.b16.5090.2026-09-22` /
`e4b.serve.h2h.vllm-0.29.0.p58.qwen3.b16.5090.2026-09-22` and one row per arm):**
vLLM **0.30.0** (released that morning) serving Qwen's GPTQ-Int4 checkpoint via
Marlin decodes at **260.3 tok/s at B=1 and 1925.6 aggregate at B=16**; this
package's **current int4 stack** (RTN int4 experts + uncalibrated int4 attention +
K16 route, fused q/k/v at B=1 — P54's) reads **239.4 / 1379.2** on the same box —
**vLLM / e4b-int4 1.087 at B=1 and 1.396 at B=16**, vLLM ahead, both inside the
pre-registered bands; vLLM 0.29.0 reads 1912.7 at B=16 (1.387; build-to-build
1.007) and its two B=1 engine starts disagreed by 7 % (DRIFT, no B=1 ratio for
that build). Same-box e4b int4 / NF4: ×2.356 / ×2.851. **Bounded:** one box,
one prompt set, B=1/B=16 only; vLLM's number includes its serving loop and
e4b's does not, so the engine advantage is understated; quality quoted, never
equated; footprint and TTFT not compared. The 2026-09-05 comparison below
stays as measured history.

**Against vLLM 0.28.0 (history, lane p37, 2026-09-05,** same box, same session, identical prompt token ids; Vast 49975016, an RTX 5090 on an EPYC 7Q83 host;
**measured** — receipt [`bench/h2h-20260905/p37/`](../bench/h2h-20260905/p37/README.md),
table in its [`RESULTS-p37.md`](../bench/h2h-20260905/p37/RESULTS-p37.md),
the pre-registration verbatim as its `PREREG.md`; register
`e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05` and one row per arm):
vLLM 0.28.0 serving Qwen's GPTQ-Int4 checkpoint (MarlinExperts, default CUDA
graphs) decodes at **286.0 tok/s at B=1 and 2030.0 aggregate at B=16**
(fp8-KV arm 300.9 / 2206.5; eager 20.8 / 322.5); this package's NF4 control
on the same box reads 113.4 / 500.1 (repeats 113.5 / 499.9) — **vLLM /
e4b-NF4 2.52 at B=1 and 4.06 at B=16**, the only licence-free ratio the lane
can quote, and it is against the slowest configuration this package ships.
**Bounded:** behind vLLM's graph decode at B=1 and B=16 on one RTX 5090 box
with one prompt set (P37) — never a general position; other batch shapes,
prefill/TTFT and vLLM's resident footprint were **not recorded**.
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
here; every ratio below is vs e4b's own NF4 control on the same box, never
a field engine): **Granite's licensed stack** (NF4 experts + folds + epilogue) is
×1.341 at B=1 (304.9 tok/s) and ×1.160 at B=16 (1836.8;
`e4b.serve.census.bo7.granite.b1.5090.2026-09-05` / `e4b.serve.census.bo7.granite.b16.5090.2026-09-05`; **no field comparator measured**); **OLMoE's position is NF4**
(282.5 / 1347.5, ×1.000; `e4b.serve.census.bo7.olmoe.b1.5090.2026-09-05` / `e4b.serve.census.bo7.olmoe.b16.5090.2026-09-05`; **no field comparator measured**) because nothing above it is licensed on this
register — the tp row's "best licensed" label predates the two-text
clause and its calibrated attention is refused on this family, so its full
stack is ×2.070 / ×2.289 measured, not licensed; **gpt-oss's quoted best is
its own reference arm** (NF4 + exact folds, 144.5 / 761.6; `e4b.serve.census.bo7.gptoss.b1.5090.2026-09-05` / `e4b.serve.census.bo7.gptoss.b16.5090.2026-09-05`; **no field comparator measured**) and the MXFP4
store under the route rule reads ×1.293 / ×0.970 with the quality gate open;
**Qwen3's licensed stack** — the streamed 64k calibrated pack artifact bo6c
licensed on both texts (11512 gptq / 776 rtn) — measured on the same box under the lane's
amendment 2 (pre-registered 06:05Z, run after `TP_DONE`): **×2.067 at B=1
(238.1 tok/s; anchor-class projection 159.2 × 2.067 ≈ 329 tok/s, a
projection from an uncertified class) and ×2.602 at B=16 (1327.5 tok/s;
no anchor projection) vs e4b's own NF4 control** — `e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05` / `e4b.serve.census.bo7.qwen3.b16.5090.2026-09-05`. The
same-box field comparator is now P58's vLLM 0.30.0 GPTQ-Int4 / MarlinExperts
(260.3 / 1925.6 graph vs the current int4 stack's 239.4 / 1379.2 — 1.087 / 1.396,
2026-09-22; P37's 0.28.0 rows 286.0 / 2030.0 are history); #405 is the P37
reproduction item (c4val1 FAIL), not a licence withdrawal. Its
speed is identical to the lane's 16k arm (4.20 vs 4.20 ms; 1327.5 vs
1338.8, within 1%) and to the RTN stack: a calibrated pack's kernels do
not depend on the calibration size, as the amendment predicted — the pack
changes the values, not the kernel or the bytes. **Gemma-4 has no K8
instrument, so no arm carries a K8 licence**; the register's position with
that caveat is the exact round-1 fold + epilogue on NF4 (`r1epi`), ×1.281
at B=1 (103.6 tok/s) and ×1.106 at B=16 (675.8;
`e4b.serve.census.bo7.gemma4.b1.5090.2026-09-05` / `e4b.serve.census.bo7.gemma4.b16.5090.2026-09-05`; **no field comparator measured**), and the quoted int4 best
(bo3's `stack`) reads ×1.705 / ×1.697 measured, no quality verdict —
Gemma-4-it loaded on this host without the #344 fault. **Mixtral's position
is NF4** (50.3 / 191.4, ×1.000; `e4b.serve.census.bo7.mixtral.b1.5090.2026-09-05` /
`e4b.serve.census.bo7.mixtral.b16.5090.2026-09-05`; **no field comparator measured**): the exact folds are ×1.062 / ×1.018 but unscored as a combined
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

- **The licensed serving pack is now artifact-backed, and the recipe's
  cross-host reproduction is narrower than it looked.** Lane P55x
  (`bench/p55x/`, 2026-09-22) built Qwen3's streamed 64k calibrated pack,
  dumped it as a hash-pinned artifact, and ran the registered K8 gate **on
  those bytes loaded back by fingerprint** — the path a user gets. It
  **passes both texts**: wikitext −0.05275 ppl, c4val1 −0.06622, against an
  NF4 reference on that box that is bit-identical to bo6c's. `pack_fingerprint
  sha256:0c9955a9f06d8326…` is in the register, the bytes are retained, and
  they were verified after transfer by two independent implementations on two
  machines. A loader given that fingerprint refuses anything else and never
  rebuilds from the recipe, so the licence now travels as bytes
  ([`e4b.serve.p55x.qwen3.all-calibexp-streamed-64k.k8.2026-09-22`](claims.json)).
  That pack is also **byte-identical to the one lane P39 built on a different
  5090 on 2026-09-10 under e4b 0.35.3**, where P55x ran under 0.36.4 — so four
  builds across at least three boxes and a release boundary agree on every
  byte, and **P37's divergence (11522/766, c4val1 +0.109) is an outlier rather
  than the rule**. The open question is no longer whether the recipe
  reproduces but what was different about that host
  ([#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405)).
  Two things stay open and are not small. **The fingerprint covers the experts
  only**: `engines/int4_attn_calib.py` has no serialisation, so the 192
  calibrated attention projections are re-derived on every load — and the same
  pinned expert bytes with RTN attention **fail** c4val1 at +0.13237, so that
  unpinnable half is the half carrying the quality. And bo6c's own row keeps no
  fingerprint, because its bytes were not retained; it stays active rather than
  superseded, since the bo7 census rows take their licence from it and nothing
  establishes that their pack is this pack. `min_rows`, damping and the K8
  budget are unchanged.
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
- **[#412](https://github.com/pjordanandrsn/experts4bit-qlora/issues/412) —
  Gemma-4 attention 4-bit: `quantize_attention_projections_4bit` converted
  100 projections where 120 were expected**, so both e4b attention-4-bit
  arms of lane tp2/P40 died on the converter's own count check before a
  step ran (`e4b.train.h2h.unsloth.gemma4.5090.2026-09-06.arm.e4b.*`,
  receipt status `void_attn4`). Attention-4-bit training on `gemma4_text`
  has no receipt and is not supported pending it; the bf16-attention
  `fast_train` path stays as tp1 left it.
- **[#344](https://github.com/pjordanandrsn/experts4bit-qlora/issues/344) —
  Gemma-4 fails to load on 2 of 6 rented hosts** with `CUDA error: invalid
  argument`, after the experts quantise. A 2 GiB host-hop fix was merged and
  reverted the same day: the model's largest tensor is 1.375 GiB, so it never
  triggered. **The six hosts' forensics have now been put side by side**
  (`bench/p55/P55-PREREG.md`), and two leads are dead by construction: driver
  580.159.03 appears on both sides, and every host is the same RTX 5090. What
  orders every outcome is **host RAM against this checkpoint's 49.9 GiB single
  shard** — 30 GiB refused the map outright (`Cannot allocate memory`), 64 GiB
  mapped it and died opaquely, 96 / 125 / 188 GiB passed — and one failing host
  had already run two hours of clean CUDA on smaller models before Gemma-4
  reached it. All three outcomes sit inside the `safe_open(device="cuda")`
  mapping, which maps the whole shard and copies each tensor out of it. That is
  a reading of six points, not a mechanism; **lane P55 is registered to test it
  and has not been launched** (its authorisation is the open item, not its
  code). Meanwhile the loader no longer fails silently about it: a shard-read
  failure now prints the shard, its size and the host's `MemTotal` /
  `MemAvailable` / cgroup limit, and `E4B_LOAD_SYNC_DEBUG=1` inserts staged
  `torch.cuda.synchronize()` checkpoints (arming `CUDA_LAUNCH_BLOCKING=1` while
  CUDA is still uninitialised) so a fault is bound to a load stage rather than
  to whichever CUDA call observed it.
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
- **Fusing q/k/v on the int4 attention store changes the arithmetic at
  B=16** (lane P54, `bench/p54/RESULTS-p54.md`). The fusion is byte-identical
  by construction and **token-identical at B=1**, where it is worth 0.516
  ms/step (12.4 %) on one 5090. At B=16 the fused arm's tokens diverge from
  the control's on 14 of 16 sequences, reproducibly, while the control's own
  two draws are bit-identical — the K16 small-M GEMM's accumulation order is a
  function of N. Leading hypothesis only; a second candidate (the round-2 glue
  path taken when `qkv_proj` exists) **was excluded by P57 on 2026-09-22**
  (`bench/p57/RESULTS-p57.md`): with the round-2 glue forced OFF on both legs the
  fused arm still diverges on 15 of 16 sequences. **And the K16 GEMM is excluded
  too (P59, same day, `bench/p59/RESULTS-p59.md`)**: the kernel is bitwise
  invariant to fusing at 2–16 rows (A2000 probe) and the fused and unfused stacks
  read KL exactly 0 at 16-row decode on the 5090. The divergence lives in the
  >16-row path — the harness's 128-token prefill steps run cuBLAS on the cached
  bf16 weight, whose kernel choice depends on N. **Bounded and licensed (P59
  amendment 1, `e4b.serve.p59b.qwen3.b16.fqkv-kl.5090.2026-09-22`):** through a
  128-row prefill the fused stack sits **0.0044 nats/token** (top-1 0.9775) from
  the unfused one at B=16, with a fresh-process rebuild of the unfused stack
  bit-identical (so the number is the fusion's, not noise) — reorder-class, well
  inside the shipped bar (≤ 0.10 nats, top-1 ≥ 0.93). **The fused q/k/v path is
  now the default on the int4 lanes at B=16 as well as B=1**; its same-box B=16
  saving is P54's 0.223 ms/step (1401 → 1429 tok/s on that box).
- **K17's fused split-K reduce is exact and slower in the consumer** (P57,
  same page): −0.038 ms/step at B=1, −0.217 at B=16, the epilogue landing inside
  the critical-path GEMV while the launches it removed were overlapped. Stays
  opt-in in grouped-nf4-gemm; the census row it targeted was GPU time, not a launch.
- **The distinct-expert count at B=16 is MEASURED (P57 amendment 3, 2026-09-22,
  `e4b.serve.p57.qwen3.b16.distinct-experts.5090.2026-09-22`): 58.7 distinct
  experts per layer per decode step** (layer means 50.9–71.7; uniform expectation
  82.4; 7–41 of 128 experts per layer never touched in 73 steps × 16 rows), from an
  on-device router hook that counts every graph-replayed step. Against
  [#564](https://github.com/pjordanandrsn/experts4bit-qlora/issues/564)'s byte
  roofline the floor at 58.7 is 4.89 ms/step vs the measured 6.34 ms expert-GEMV
  row: **~77 % of roofline at real routing, ~1.4 ms/step of headroom** in the
  largest row of the B=16 step — the lever the P58 gap analysis said was absent
  is present after all, sized by the routing it actually sees.
- **Where that headroom is (P60–P61, `bench/p60/`, `bench/p61/`):** on the
  recorded B=16 routing, one row per distinct expert instead of one per routed
  row runs **0.92 ms/step** faster (`e4b.serve.p60.qwen3.b16.expert-gemv-repeat-cost.5090.2026-09-22`),
  and row order is worth nothing. Sharing loads does not reach it: K18's
  grouped GEMV is exact and 1.49× slower (`gnf4.kernel.k18-grouped-expert-gemv.5090.2026-09-22`).
  P61 finds row work and expert bytes overlap rather than add, so no lever is
  licensed (`e4b.serve.p61.qwen3.b16.expert-gemv-cost-split.5090.2026-09-23`).
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
