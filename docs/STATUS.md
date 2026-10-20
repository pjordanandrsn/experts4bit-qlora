# Status — what this package does, what is current, what is open

**As of 2026-10-20, version 0.56.0** (the version of record is `pyproject.toml`'s). This page states the current
position in each area, with the claim id behind each number; [`docs/claims.json`](claims.json) holds every claim's full
text and evidence. The dated narrative behind these positions, with the readings they replaced and why, is
[`STATUS-RECORD.md`](STATUS-RECORD.md).

Evidence words: **measured** = a run with a receipt in this repository. **measured-private** = a real run whose receipt
is in a private audit tree, so you cannot check it from here; such numbers say so. **retired** = published, now known
wrong, kept so the retraction is findable. **superseded** = true as measured, but a later row is the one to quote.
Speeds are within-box readings on the named card: absolute times do not travel between hosts, and a wall-clock ratio
can move with the host's CPU.

---

## What you get today

### Fit and memory

**It fits, and it trains, on a 12 GB card.** Full bf16 OLMoE-1B-7B runs out of memory on a 12 GB card; in 4-bit it
loads at 4.70 GB and trains under 8 GB, and the streaming loader never materialises the bf16 model in CPU or GPU RAM.
QLoRA on the frozen NF4 experts moves a held-out Alpaca eval from 1.4813 to 1.0290. `e4b.train.olmoe-fits`,
`e4b.train.olmoe-converges` ([METHODOLOGY](METHODOLOGY.md)).

**Expert offload takes 30B-class MoEs past VRAM.** Qwen3-30B-A3B trains TC1's matched set (fp32 adapters) on a 12 GB RTX
A2000 at 10.463 GB peak allocated (micro-batch 1 × accum 8, the registered secondary after micro-batch 2 ran out of
memory on that shared card; one draw; `e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`). An older 5-step
reading put it at 7.16 GB and Gemma-4-26B-A4B at 8.47 GB, both out of memory without offload; that reading was not
reproduced (a later attempt, at a different configuration on a 5090, read 9.13 GB). Offload costs about +11 % s/step at OLMoE scale. The
on-disk arena needs 2.56× to 6.40× less host RAM than the pinned-RAM path: at an 8.59 GB cap Qwen3-30B is OOM-killed on
host RAM and completes on the arena. `e4b.offload.fits-30b-class`, `e4b.offload.arena-vs-host-ram`
([results](../bench/host-ram-ceiling/RESULTS-scaling.md)).

**On a 24 GB card, Qwen3-30B-A3B trains the field recipe under expert offload:** TC1's matched set (642,514,944 fp32
adapters) at 11.88 GB peak and 10.52 s/step, its trajectory equivalent to resident training on a 5090 (median per-step
|Δ| 0.0025). Resident, e4b runs out of memory (24.45 GB). Unsloth 2026.9.14 trains the same set resident there at
24.22 GB and 10.03 s/step; HF neither fits nor offloads, and axolotl's rows are a harness failure, not readings. A fit
table, one draw per row, not a speed position. `e4b.train.frontier.qwen3.4090-24gb.2026-10-02` with its
`.unsloth-resident`, `.e4b-resident-oom`, `.hf-axolotl` and `.offload-keeps-trajectory` rows
([TC3](../bench/h2h-2026-10-02/tc3/README.md)).

**Under expert offload, layer-major accumulation is 2.365× faster per step than the stock loop.** `layer_major_step`
(`experts4bit_qlora.engines.layer_major`) runs a step's micro-batches layer by layer. Each offloaded layer is staged
once per step instead of once per micro-batch: 95 stagings against 380. With `enable_fast_train(dgrad=True)` on
Qwen3-30B-A3B on one RTX 4090, the stock loop steps 7.641 / 7.514 s and layer-major 3.231 / 3.173 s, at 7.98 GB peak
against 8.41. The predicted band [1.25, 2.0] was missed high, and the held-out term (0.015 against 0.01) sits inside
the 0.031 same-loop spread at 34 steps. From the CLI trainer, `OFFLOAD_EXPERTS=1 LAYER_MAJOR=1` (which implies
`FAST_TRAIN=1`) makes the same calls, checked bit for bit against the library on a fixture. At the CLI's own default
recipe (SEQ 192, token-budget batches of 2048, r 8 with attention LoRA, torch AdamW) on one RTX 4090 it steps 1.455×
faster than `OFFLOAD_EXPERTS=1 FAST_TRAIN=1` (11.596 / 11.595 s against 7.956 / 7.968 s): its larger micro-batches
carry more compute, so the copy is a smaller share of the step, and the a-priori band [1.5, 2.6] was missed low.
It replicates on one RTX 5090: 2.311× (6.017 / 6.014 s against 2.597 / 2.602 s), inside the band [1.8, 3.0]
predicted a priori from the host link.
It holds on Mixtral-8x7B on one RTX 4090: 2.118× (11.386 / 11.386 s against 5.372 / 5.375 s), inside the band
[1.5, 2.6] predicted a priori, at 5.64 GiB peak against 5.72.
`e4b.train.p137.layer-major-offload.qwen3.4090.2026-10-18` ([P137](../bench/p137/RESULTS-p137.md)),
`e4b.train.p140.layer-major-offload.qwen3.5090.2026-10-18` ([P140](../bench/p140/RESULTS-p140.md)),
`e4b.train.p144.layer-major-offload.mixtral.4090.2026-10-18` ([P144](../bench/p144/RESULTS-p144.md)),
`e4b.train.p146.layer-major-cli.qwen3.4090.2026-10-18` ([P146](../bench/p146/RESULTS-p146.md)).

**On the 12 GB RTX A2000 the same set trains under offload at micro-batch 1**, at 10.46 GB peak, held-out 0.8483 beside
the 5090's 0.8516 / 0.8487. Unsloth refuses on that card, axolotl's wheels need a newer driver, and HF runs out of
memory at load. The A2000 is a correctness-only testbed, so no step time is quoted.
`e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`.

**Kimi-K3 runs at full depth on the 12 GB A2000:** all 93 layers at 4.07 GB peak VRAM once Triton's cache is warm, the
MXFP4 experts streamed from a 1.446 TB SSD arena and the 108.76 GB dense side read from the checkpoint's byte offsets,
with 0 bytes pinned in host RAM. Five processes complete "The capital city of France is" as " Paris. It is"; the
per-token NLL correlates with Fireworks' kimi-k3 at r = 0.9965; on grouped-nf4-gemm 0.33.6 the forward is
bit-identical run to run. Feasibility and fidelity only, with no decode speed quoted, through a custom driver rather
than `load_moe_4bit_streaming`. `e4b.offload.kimi-k3.full-depth.a2000.five-runs.2026-09-28`,
`e4b.quality.kimi-k3.vs-fireworks.per-token.a2000.2026-09-28`,
`e4b.parity.kimi-k3.reproducible-on-gnf4-0.33.6.a2000.2026-09-29`
([results](../bench/kimi-k3-a2000/RESULTS-kimi-k3-a2000.md)).

**Dense execution remains in development.** DQ7's out-of-sample reading is VOID; no reserve calibration or 24 GB
boundary has passed. DQ9 completed sixteen known-subject diagnostics with the corrected full-logit workspace:
allocator estimates were above measured peaks, but all ten streamed full-device estimates remained below sampled
driver use, by up to 1.14 GB. One setup clear released 32 MiB on one arm and changed no training peak, so it provides
no training-memory remedy. DQ9 licenses neither capacity nor calibration; the opt-in and enforced streamed margin
remain. `e4b.train.dense-executor.dq9.5090.2026-10-08` ([diagnostic record](../bench/dq9/receipts/dq9-5090-2/README.md));
`e4b.train.dense-executor.dq7.5090.2026-10-08` ([original VOID reading](../bench/dq7/RESULTS-dq7.md)).

**DQ11 is a fusion build candidate, measured-private.** Unsloth (U) was faster than e4b's dense path (L) in this matched SDPA, eager, full-logits
configuration, not Unsloth's default. The registered quality gates pass and the independently rederived decision is
`BUILD_CANDIDATE`; this licenses a separate fusion investigation and carries no capacity or shipping claim.
`e4b.train.dense-adapter.dq11.matched-sdpa.5090.2026-10-11` ([results](../bench/dq11/RESULTS-dq11.md)).

### Training against other frameworks

**Qwen3-30B-A3B against Unsloth 2026.9.14: Unsloth spends 1.92× e4b's GPU time per step; wall-clock 2.80× on an AMD EPYC
7713.** TC1's field recipe (alpaca, seq 2048, micro-batch 2 × accum 4, r 16, AdamW-8bit) on one RTX 5090, the work
matched, both frameworks on one stack, e4b at every current default. The wall ratio depends on the host: Unsloth runs
14× e4b's CPU ops per step and keeps its GPU busy for 0.33 of its step against e4b's 0.49, so a slower host stretches
Unsloth's step more. Held-out loss agrees within 0.0012; e4b peaks 1.89 GB higher (26.16 against 24.27 GB). HF + PEFT
runs out of memory on the 32 GB card. `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio`,
`e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.hf-oom` ([results](../bench/h2h-2026-10-02/tc1/RESULTS-tc1-pos69.md)).

**On packed 4,096-token rows, Unsloth 2026.9.14 spends 1.160× e4b's GPU time per step.** The wall-clock ratio follows the host:
1.472 on the profiled host, 1.287 and 1.773 on two others. At e4b's defaults its training phase peaks 1.04 GB above
Unsloth's 24.86 GB, and 0.30 GB above with the opt-in `E4B_CKPT_OFFLOAD=1`.
`e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio`,
`e4b.train.memory.packed-4k-position-defaults.5090.2026-10-07`
([results](../bench/h2h-2026-10-02/tc1/RESULTS-tc1-pos68.md)).

**Mixtral-8x7B against Unsloth 2026.9.14: e4b is faster, Unsloth/e4b 1.144**, both resident on one stack (torch 2.12.1 /
transformers 5.5.0), on an AMD EPYC 7B13: 3.24 against 3.71 s/step, at 2.07 GB more peak on e4b (its fp32 expert
absmax). The register's other Mixtral ratios answer other questions: 0.836 put e4b on the field image's torch 2.8 stack
on a Core Ultra 9 285K, the host most favourable to Unsloth seen (the same-stack box read that stack at 1.013), and
0.361 is tp2's footprint trade (2026-09-06, Unsloth 2026.9.2), e4b under CPU expert offload at 3.22 GB against Unsloth
resident at 29.16 GB, not a
resident position. `e4b.train.h2h.unsloth.mixtral.5090.2026-10-05.same-stack`,
`e4b.train.h2h.unsloth.mixtral.5090.2026-10-04.dense-default`, `e4b.train.h2h.unsloth.mixtral.5090.2026-09-06`
([results](../bench/h2h-2026-10-02/tc2/RESULTS-tc2-mixtral-samestack.md)).

**Granite and OLMoE: e4b is faster than each comparator** (TC2, 2026-10-04, one RTX 5090; transformers 5.18.0 + PEFT
0.21.2, axolotl 0.20.0, Unsloth 2026.9.14): HF/e4b 1.299 and
axolotl/e4b 1.150 on Granite-3.1-3B-A800M, Unsloth/e4b 1.821 on OLMoE-1B-7B, each pair comparable or equivalent on
held-out loss. e4b's peak is about half HF's on Granite and 1.70 GB above Unsloth's on OLMoE. Unsloth adapts no expert
parameter of Granite, so there is no Unsloth position there, and gpt-oss-20b has no adapter set common to the
frameworks. `e4b.train.h2h.hf.granite.5090.2026-10-04`, `e4b.train.h2h.axolotl.granite.5090.2026-10-04`,
`e4b.train.h2h.unsloth.olmoe.5090.2026-10-04`, `e4b.train.h2h.unsloth.granite.5090.2026-10-02.coverage`,
`e4b.train.h2h.unsloth.gptoss.5090.2026-10-02.no-common-set` ([TC2](../bench/h2h-2026-10-02/tc2/README.md)).

**Qwen3.6-35B-A3B: at its defaults e4b does not fit the matched set resident on 32 GB; Unsloth 2026.9.14 does**, at 30.47 GB.
Under expert offload e4b trains the set at 19.35 GB. With the comparator's bytes at micro-batch 1 (a measurement hook,
not a training option) e4b trains it resident and Unsloth/e4b reads 2.049, a labelled row.
`e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-04`, `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02.e4b-offload`,
`e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-04.mb1-dq-frozen4`.

**axolotl 0.20.0 on Qwen3-30B-A3B: e4b is faster, axolotl/e4b 2.775** at matched work (1.979 on a second host). At steady
state (steps 101–200) axolotl's scattermoe native-best / e4b as shipped reads 1.238 and 1.146 on two hosts, while at
step 200 axolotl's held-out is 0.7705 against e4b-as-shipped's 0.7948. Both were read on 2026-10-03, before the later
training defaults. `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03`,
`e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state`
([results](../bench/h2h-2026-10-02/tc1/RESULTS-tc1-matched19.md)).

**On an H100 NVL at e4b 0.45.0's default settings, e4b is faster per step than Unsloth 2026.9.14: Unsloth/e4b
1.061**, at 2.93 GB more peak
on e4b and ×1.23 Unsloth's energy per step. `e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0`
([TC1c](../bench/h2h-2026-10-02/tc1c/README.md)).

### Training support per family

**Reach for the fused path: `enable_fast_train(model, dgrad=True)`.** Across two 30B-class MoEs, five datasets each,
200 steps per cell, it runs 1.52–1.81× faster per step at 0.75–0.81× peak VRAM and 0.80–0.92× energy, with loss parity
and the frozen 4-bit stack bit-identical. `e4b.train.flagship-matrix`
([results](../bench/flagship-matrix/RESULTS-flagship-matrix.md)).

**Support is stated per family and per path.** Each family's real weights go through `load_moe_4bit_streaming` +
`verify_moe_4bit(strict=True)`, then each path trains 60 steps against the family's own reference loop (|Δ final loss|
and median step-wise |Δ| ≤ 0.05). A PASS is a PASS on one text, not a convergence claim. The first six rows are lane
tp1's ([results](../bench/train-parity-20260905/tp1/RESULTS-tp1.md)), the last five lane MG1's
([results](../bench/moegen/mg1/mg1-5090-2/RESULTS-mg1.md)).

| model_type | quantize | reference_train | fast_train | batched_train | nvme_train | native_mxfp4_train |
|---|---|---|---|---|---|---|
| `olmoe` | supported | supported | **supported** · `e4b.train.parity.tp1.olmoe.fused.2026-09-05`, `e4b.train.parity.mg1.olmoe.fused.2026-10-04` | void · `e4b.train.parity.tp1.olmoe.batched.2026-09-05` | not_tested | n/a |
| `qwen3_moe` | supported | supported | **supported** · `e4b.train.parity.tp1.qwen3.fused.2026-09-05` | void · `e4b.train.parity.tp1.qwen3.batched.2026-09-05` | not_tested | n/a |
| `gemma4_text` | supported | supported | **supported** · `e4b.train.parity.tp1.gemma4.fused.2026-09-05`, `e4b.train.parity.mg1.gemma4.fused.2026-10-04` | void · `e4b.train.parity.tp1.gemma4.batched.2026-09-05` | not_tested | n/a |
| `granitemoe` | supported | supported | **supported** · `e4b.train.parity.tp1.granite.fused.2026-09-05` | supported · `e4b.train.parity.tp1.granite.batched.2026-09-05` | not_tested | n/a |
| `mixtral` | supported (offload) | supported (offload) | **supported** under offload · `e4b.train.parity.tp1.mixtral.fused.2026-09-05` | supported · `e4b.train.parity.tp1.mixtral.batched.2026-09-05` | not_tested | n/a |
| `gpt_oss` | supported | refused; attention-only QLoRA trains · `e4b.train.parity.tp1.gptoss.attn_only.2026-09-05` | refused · `e4b.train.parity.tp1.gptoss.fused.2026-09-05` | refused · `e4b.train.parity.tp1.gptoss.batched.2026-09-05` | refused (#402) | experimental · `e4b.train.parity.tp1.gptoss.mxfp4.2026-09-05` |
| `lfm2_moe` | supported | supported | **supported** · `e4b.train.parity.mg1.lfm2.fused.2026-10-04` | not_tested | not_tested | n/a |
| `granitemoehybrid` | supported | supported | **supported** · `e4b.train.parity.mg1.graniteh.fused.2026-10-04` | not_tested | not_tested | n/a |
| `ernie4_5_moe` | supported | supported | **supported** · `e4b.train.parity.mg1.ernie.fused.2026-10-04` | not_tested | not_tested | n/a |
| `nemotron_h` | supported | supported | **supported** · `e4b.train.parity.mg1.nemotron.fused.2026-10-04` | not_tested | not_tested | n/a |
| `qwen3_5_moe` | supported | supported | **supported** · `e4b.train.parity.mg1.qwen3_5.fused.2026-10-04`, `e4b.train.parity.mg1.qwen3_5.p2.2026-10-05` | not_tested | not_tested | n/a |

`void` means the arm ran but the kernel was not reached on every layer, so it carries no parity number. On
`gemma4_text` the attention-4-bit configuration is supported too, judged against the family's measured training-parity
floor (0.121 final / 0.1275 median). `e4b.train.p67.gemma4.attn4-reorder-floor.5090.2026-09-24`. The machine-readable
form is `training_support` in [`capabilities.json`](capabilities.json), whose `model_families` is exactly the families
whose `fast_train` is supported.

### Training defaults and opt-ins

**`enable_fast_train`'s defaults each come from a registered A/B** on Qwen3-30B-A3B on one RTX 5090 (table below).
Five earlier step-time changes (one host sync per MoE layer pass, the lean LoRA delta, the tile rule, fused RMSNorm,
host reuse) each stepped the field recipe at 0.847–0.968 of the code before, held-out within 0.0021.
`e4b.train.host-syncs.qwen3.5090.2026-10-03`, `e4b.train.lora-delta-lean.qwen3.5090.2026-10-03`,
`e4b.train.prefill-tile-rule.qwen3.5090.2026-10-03`, `e4b.train.fused-rmsnorm.qwen3.5090.2026-10-03`,
`e4b.train.host-reuse.qwen3.5090.2026-10-04`.

**Opt-ins, each read and left off:**
- `E4B_CKPT_OFFLOAD=1`: 0.74 GB off the packed-row training peak for 1.023 of the step.
  `e4b.train.ckpt-flavour.default.5090.2026-10-07`
- `E4B_MOE_KEEP_LAYERS` (with grouped-nf4-gemm's compact delta): 0.835 of the step with 32 of 48 layers' MoE
  activations kept, at +4.5 GB; use it where that headroom exists ([CHOOSING](CHOOSING.md)).
  `e4b.train.moe-keep.qwen3.5090.2026-10-04`
- `GNF4_TRAIN_GEMM=decoded`: 1.066 of the fused kernels' step on Qwen3, so `auto` does not take it.
  `e4b.train.decoded-route.qwen3.5090.2026-10-06`
- `NF4_QLORA_SINGLE_LADDER` (grouped-nf4-gemm): its `auto`, the ladder only with fp32 adapters, becomes the default.
  With fp32 adapters it stepped 0.797 of the time on a host-bound box and 1.031 on a GPU-bound one (registered bound
  1.05). With bf16 adapters, the shipped arm, it does not engage (1.002). `e4b.train.single-ladder.field.5090.2026-10-08`,
  `e4b.train.single-ladder-auto.gpu-bound.5090.2026-10-08`

**Energy: on a card that already fits the model, 4-bit costs energy.** `bnb.matmul_4bit` costs 1.748× native bf16's
J/op at decode, 1.601× at prefill and 1.965× in training. The fused 4-bit MoE forward's J/token at batch 4096 is 0.063
of batch 64's. `e4b.train.energy-honest.5090.2026-10-05` ([results](../bench/p114/RESULTS-p114.md)).

### Serving

**`serve_paged` decodes under bucketed CUDA graphs by default:** 5.597× the eager server with 16 requests and 9.020×
with one (Qwen3-30B-A3B NF4, one RTX 5090), at no measurable quality cost (+0.00036 nats against eager, P110
AT_PARITY). `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03`,
`e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.2026-10-03` ([SERVING](SERVING.md)).

**Capacity: 64 slots serve 12 req/s on one RTX 5090, against 4 at the old 16 slots** (Qwen3-30B-A3B int4, 512-token
prompts, 2,048 tokens a slot, TTFT ≤ 1.0 s and TPOT ≤ 100 ms). Since 0.50.0 `serve_paged` sizes its slots to the serve
estimate and decodes the widest step as one graph replay. One 64-row piece costs no measurable quality against four
16-row pieces (−0.0032 nats, P117 AT_PARITY). Under load, output text depends on which requests a step batches
together; serial output is unchanged. `e4b.serve.sc2e.64-slots-buckets-auto.qwen3.5090.2026-10-07` (12 req/s),
`e4b.serve.sc2e.64-slots-default-buckets.qwen3.5090.2026-10-07` (8 req/s on the 1–16 buckets),
`e4b.serve.p117.wide-bucket-quality.qwen3.5090.2026-10-08` ([SC2e](../bench/h2h-2026-10-02/sc2e/README.md)).

**Prefill: K19's lean dispatch by default.** On the int4 K19 prefill route, gate_up reads the token rows itself and down
stores in the caller's order, so the 4,096-row gather and the unsort are gone. It is bit-identical on every window and
×0.968 [0.966, 0.970] per 512-token first-chunk prefill replay (Qwen3-30B-A3B int4, one RTX 5090, P130).
`e4b.serve.p130.prefill-lean-dispatch.qwen3.5090.2026-10-10` ([results](../bench/p130/RESULTS-p130.md)).

**Prefill: the decode folds take prompt rows on Qwen3-MoE.** On `qwen3_moe`, a prefill's norms, residual-and-norm and
q/k norm with the rotary run decode's fused kernels: ×1.118 [1.116, 1.119] per 512-token first-chunk prefill replay, at
parity on a teacher-forced read (bias −0.0035 nats; Qwen3-30B-A3B int4, one RTX 5090, P130). Other families keep the
eager chain until a read of their own. `e4b.serve.p130.prefill-glue-speed.qwen3.5090.2026-10-10`,
`e4b.serve.p130.prefill-glue-quality.qwen3.5090.2026-10-10` ([results](../bench/p130/RESULTS-p130.md)).

**Single-request decode.** grouped-nf4-gemm's bandwidth-targeted NF4 decode GEMV decodes one request 1.2417× as fast,
16 unchanged, within P110's quality bar; it is grouped-nf4-gemm's default from 0.43.0 at Qwen3's NF4 expert shapes on
GPUs with 160 or more SMs. `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`. On Mixtral-8x7B, at plans chosen on an RTX
5090, the same GEMV decodes one request **2.41×** as fast (19.5 → 8.1 ms a step), 16 unchanged, within the same bar.
`e4b.serve.p141.gemv-bw.mixtral.5090.2026-10-18` ([results](../bench/p141/RESULTS-p141.md)). Two more reads set
grouped-nf4-gemm 0.47.0's `auto` scope. On an RTX 4090 (128 SMs), Qwen3-30B-A3B decodes one request **1.5145×** as fast
(`e4b.serve.p133.gemv-bw.qwen3.4090.2026-10-18`, [results](../bench/p133/RESULTS-p133.md)). On an RTX 5090 at a 400 W
limit, Granite-3.1-3b-a800m decodes it **1.3979×** as fast and OLMoE-1B-7B **1.5011×**
(`e4b.serve.p134.gemv-bw.granite.5090.2026-10-18`, `e4b.serve.p134.gemv-bw.olmoe.5090.2026-10-18`,
[results](../bench/p134/RESULTS-p134.md)). Sixteen requests are unchanged and quality is within the same bar in each.
- **Scope since grouped-nf4-gemm 0.47.0:** its default also takes Mixtral-8x7B's, Granite-3.1-3B-A800M's and OLMoE-1B-7B's
  expert shapes on GPUs with 160 or more SMs, and the RTX 4090 (128 SMs, sm_89) at Qwen3-30B-A3B's two shapes only. The
  shapes are licensed, not the families: any model with those expert shapes takes them. Every other shape and smaller
  GPU keeps dot-pad; `GNF4_GEMV_BW=0` restores it.

**The B=1 fused stack is the default on Qwen3-MoE, Qwen3.5/3.6-MoE and Mixtral.** On top of grouped-nf4-gemm 0.43.0's
bandwidth GEMV, fused q/k/v
plus three glue folds decode one request 1.5902× and 16 requests 1.1644× as fast on Qwen3-30B-A3B NF4, and P115 Phase
D's SANE read at T == 1 passes (bias +0.00541 nats, argmax agreement 0.9674). The four knobs resolve to `auto` on the
families with such reads and stay off elsewhere. Lane FAM read the others at T == 1, each against its own neutral floor
(#1362). Granite-MoE fails on one gated entry of 12, by 2 positions in 1536. gpt-oss-20b fails on every knob, because its
own floor sits below the 0.90 agreement backstop; that floor also contains Phase C's 0.924, so Phase C's miss was the
gate, not the knob. Qwen3.6-MoE, where only the router epilogue engages, passes, and the epilogue decodes one request
there in 0.973 of the step (FAM Amendment 4), so the family joined the allowlist. Mixtral-8x7B passes on every knob,
resolved at a ×0.90 softmax-scale change, and its three folds (no fused q/k/v engages there) decode one request in
0.9393 and 0.9277 of the step, 20.2–20.5 → 19.0 ms (P135, FAM Amendment 5's speed read), so the family joined the
allowlist too. Gemma-4-26B-A4B's knobs stay off: the family's own neutral floor sits below the
agreement backstop, so the instrument cannot license any knob on its NF4 stack (closed at its proof; the knobs pass
the relative gate). `0` on each knob is the way back.
`e4b.serve.p115.fused-stack-combined.qwen3.5090.2026-10-08`, `e4b.serve.p115.fused-stack-speed.qwen3.5090.2026-10-07`,
`e4b.serve.p115.fused-stack-quality.qwen3.5090.2026-10-07`, `e4b.serve.fam.fused-stack-t1.granite.5090.2026-10-09`,
`e4b.serve.fam.fused-stack-t1.gptoss.5090.2026-10-09`, `e4b.serve.fam.fused-stack-t1.qw36.5090.2026-10-09`,
`e4b.serve.fam.router-epilogue-speed.qw36.5090.2026-10-09`, `e4b.serve.fam.fused-stack-t1.mixtral.5090.2026-10-09`, `e4b.serve.fam.fused-stack-t1.gemma4.5090.2026-10-10`,
`e4b.serve.p135.fold-speed.mixtral.5090.2026-10-18`
([P115](../bench/p115/RESULTS-p115.md), [FAM](../bench/fam/RESULTS-fam.md), [P135](../bench/p135/RESULTS-p135.md)).

**Where the default single-request step goes.** A kernel-class census of the shipped default on Qwen3-30B-A3B NF4,
run on a 5090 with the fused stack and the bandwidth GEMV both on, puts the one-request step at 4.75 ms. The dense bf16
GEMVs (attention projections, router logits, lm_head) lead at 38 %, then the NF4 experts at 26 %, attention at 15 %
and launch-bound routing glue at 11 %; the GPU is busy 92 % of the step. At 16 requests the small-M grouped NF4 GEMM is
69 % of an 18 ms step. The census moves no default: it prices the next single-request lever, the dense GEMVs, which
needs a quality bar of its own. Its router census of 48 is the 5090 check of the fp32 router probe on the shipped
default. `e4b.serve.p123.default-decode-census.qwen3.5090.2026-10-09` ([P123](../bench/p123/RESULTS-p123.md)).

**The paged FP8 attention decode is capped at 16 splits on Qwen3-MoE.** The library plans its split count from the
block table's capacity: 32 at one request on 4096-token slots, most of them empty at served contexts. A cap of 16 splits
moves buckets 1, 2 and 4 only.
- **Speed:** one request decodes **1.044×** as fast on a 5090 (255.6 → 267.0 tok/s; the smallest of four ABBA pairs,
  binding). Sixteen requests are unchanged (0.992).
- **Quality:** SANE exact (bias 0.0, argmax 1.0). The cap is bitwise-identical to the plan at contexts ≤ 1024, and
  within A2's fp64 contract (error 0.9976–1.0000× the plan's) at 2048 and above.
- **Default:** on for `qwen3_moe` (16); `E4B_ATTN_SPLIT_CAP=0` restores the plan.
- **Measured basis:** Qwen3-30B-A3B on an RTX 5090 (170 SMs). Other `qwen3_moe` checkpoints (e.g. Qwen3-235B-A22B)
  and other cards take the default by family without a read of their own. The plan is SM-scaled, so the cap moves
  B 1–6 on a PRO 6000 and B 1–4 on a 4090.

`e4b.serve.p143.attn-split-cap.qwen3.5090.2026-10-19` ([P143](../bench/p143/RESULTS-p143.md)).

**Qwen3-30B-A3B's lm_head serves one-row decode from an int8 copy (on by default).** `E4B_INT8_HEAD_BW=1` rounds the
151936 x 2048 head in place to its per-row int8 grid, P150's I8HEAD. One-row calls go to grouped-nf4-gemm K36's int8
GEMV, which reads half the head's bytes.
- **Speed:** one request decodes **1.052×** as fast on a 5090 (263.7 → 277.5 tok/s; the smallest of four ABBA pairs).
  The step saving is 0.189 ms, 0.998 of P148's microbench projection.
- **Sixteen requests:** 0.992, inside the 0.99 line. S1 is below S0 in every W16 pair, though the W16 rate counts only
  16-row steps, where the head is passed to `F.linear`. The cost is observed, not yet attributed. Token divergence,
  named at review, is bounded out by a $0 router replay (distinct experts per step 0.9990, S1/S0; bf16, teacher-forced
  on the recorded streams). The resident int8 copy is the remaining named candidate.
- **Quality:** SANE inside the bars (bias +0.0022, argmax 0.989), in line with P150's GO at both of P125's gates. The
  knob rounds the head at every row count, so every row count's logits change.
- **Default:** on for `qwen3_moe` at this head shape (151936 x 2048). It needs K36's GEMV, first released in
  grouped-nf4-gemm 0.48.0. With an older release the default resolves off, recorded in `/health`'s
  `levers.int8_head_bw.source`. `E4B_INT8_HEAD_BW=0` opts out.
- **Quality basis:** read on Qwen3-30B-A3B @ad44e777. Same-shape checkpoints inherit the default unread.
- **Measured basis:** this checkpoint's head on an RTX 5090 (170 SMs), at K36's tuned plan. Other heads take K36's
  untuned default plan, and other cards are not read.

`e4b.serve.p153.int8-head-bw.qwen3.5090.2026-10-19` ([P153](../bench/p153/RESULTS-p153.md)).

**Qwen3-30B-A3B's batched decode runs K25 with K39's byte-permute decode (on by default).** `E4B_K25_LUT=prmt` swaps
K25's select-tree codebook decode for grouped-nf4-gemm K39's (`lut="prmt"`), which yields the same values bit for bit.
- **Speed:** sixteen requests decode **1.0925×** as fast on a 5090 (about 976 → 1067 tok/s; the smallest of four ABBA
  pairs, 1.0925-1.0934). One request is unchanged (0.9999): K25 is not on the T == 1 path.
- **Identity:** every token the prmt arms decoded is the tree's, at 16 and 1 requests; the tree's own null held across
  five processes.
- **Clocks:** the median SM clocks differ by 0.25 %. The tree met the software power-cap event (0x4) in more samples
  (0.40-0.44 of its samples, against 0.07-0.18), under an unchanged 575 W limit: part of the served difference rather
  than a confound.
- **Default:** on for `qwen3_moe`. It needs K39's decode, first released in grouped-nf4-gemm 0.50.0, and an NVIDIA
  device. With an older release or on ROCm the default resolves to the tree (the same values), recorded in
  `info["k25_lut"]["source"]`. `E4B_K25_LUT=tree` opts out; an explicit `E4B_K25_LUT=prmt` still refuses a package or
  device that cannot serve it.
- **Scope:** the default covers every `qwen3_moe` checkpoint, Qwen3-235B-A22B included. The decode is bit-identical
  to the tree, so the values are the same everywhere; the speed was read only at Qwen3-30B-A3B's expert shapes on an
  RTX 5090 (170 SMs) at today's K25 plan, and no speed is claimed for other shapes (a 235B scale-up lane is where it
  gets read). Other families stay on the tree.

`e4b.serve.p158.k25-prmt.qwen3.5090.2026-10-20` ([P158](../bench/p158/RESULTS-p158.md)).

**qwen3_moe's default on main after 0.56.0, in no release yet: K25 at plan `c1` under K39's decode (`E4B_K25_PLAN`,
e4b#1877; the default from lane P161).** `c1` runs K25 at P156's first carried plan pair, gate_up `{block_n 64, kc 256,
warps 4, stages 2}` and down `{block_n 128, kc 128, warps 8, stages 3}`, in place of today's single plan.
- **The default:** an unset `E4B_K25_PLAN` resolves at engine build to `c1` on `qwen3_moe` where K25's decode resolved
  to `prmt`, and to `today` everywhere else, with the reason in `info["k25_plan"]["source"]`. It never refuses: where the
  decode is not `prmt`, an unset knob serves `today`. `E4B_K25_PLAN=today` opts out. Other families stay on today's plan
  until a read of their own.
- **Only under `prmt`.** The select tree at `c1`'s shapes is slower than at today's (P155's selection: gate_up
  64/256/4/2 at 14.23 ms against 8.32, at 255 registers; [RESULTS-p156](../bench/p156/RESULTS-p156.md), "The cause is
  the decode"), so an explicit `c1` under any other decode is refused at engine build and in the forward, never run.
  It is also refused with K25 off or absent.
- **Measured: a kernel microbench, then a served read.** At Qwen3-30B-A3B's W16 shapes on an RTX 5090, under `prmt`,
  both K25 GEMMs ran 1.6598× faster at `c1` than at today's plan (kernel time today / `c1`, above 1 is faster; the median
  of 126 paired reps, lower bound 1.6568, clock-adjusted 1.6525; P160 `p160-5090-2`,
  [RESULTS](../bench/p160/RESULTS-p160.md); UNVERIFIED: no register row). In the served step, P161 (`p161-5090-1`, one
  RTX 5090 at 575 W, e4b `da3eb5c5`) read sixteen concurrent requests decoding **1.465433×** as fast at `c1` as at
  today's plan, both under `prmt` (rate, the smallest of four ABBA pairs), one request unchanged (0.9966), and every
  token identical: `e4b.serve.p161.k25-plan-c1.qwen3.5090.2026-10-20`, [RESULTS](../bench/p161/RESULTS-p161.md). One read
  on one card and one family; qwen3_moe takes `c1` by default on it.
- **Same bits.** The tree, `prmt`, and `prmt` with `c1` give bit-identical outputs through the served call
  (`tests/test_k25_lut_gpu.py`, including a K that neither of `c1`'s KCs divides).
- **Records:** `info["k25_plan"]` (the build's resolution) and `prefill_routes()["seen"]["moe_k25_plan"]` (each K25
  call's plan).

**Int8 attention projections on Mixtral are opt-in, with no speed claimed.**
- **What it does:** `E4B_SERVE_ATTN_INT8=1` (#1848) holds Mixtral-8x7B's q, k, v and o projections in int8 with a
  per-row fp32 scale. Decode calls of 1-16 rows run grouped-nf4-gemm 0.49.0's K37 kernels (`dense_i8.linear_i8`), and
  larger calls dequantize transiently.
- **Where it refuses:** on other families, without K37, and alongside int4 attention or `E4B_DENSE_GEMV_BW=1`.
- **Its premises have read:**
  - the int8 round trip keeps P147's quality gate: I8ATTN GO, `p147-a2000-1`;
  - the kernels beat cuBLAS bf16 at Mixtral's shapes: P149 box K, K_GO, `p149k-5090-8`, a microbench premise.
- **Why it stays off by default:** the served read, P149's box Q+S, has not run. There is no claim id.

[P147](../bench/p147/RESULTS-p147.md), [P149](../bench/p149/RESULTS-p149.md).

**Calibrated int4 attention is not licensed on the default.** On Qwen3-30B-A3B, `E4B_SERVE_ATTN_INT4_CALIB=1` decodes
one request 1.154× as fast (1.238× with the int4 lm_head too). The mean NLL bias stays inside K8's budget, but only
92.9 % of next-token argmaxes agree with the default at one window per pass (91.8 % with the head), against the 0.95
gate. It stays opt-in. A K8-style ppl gate alone would have passed it. Calibration is deterministic and takes 95–225 s a
build, depending on the host. `e4b.serve.p125.int4-attn-calib-default.qwen3.5090.2026-10-09`
([P125](../bench/p125/RESULTS-p125.md)).

**The launch-bound glue is cut, bitwise.**
- **What changed.** P127 fused or removed the small launches around the MoE routing and rotary on the default decode
  path: three host casts (#1448), and five grouped-nf4-gemm options (#526–#530) wired in by #1472, #1477 and #1482.
- **What it buys.** On Qwen3-30B-A3B, with grouped-nf4-gemm 0.45.0 or later, the default now decodes one request 1.132×
  as fast as the stack before #1448 (226 → 256 tok/s, 0.52 ms saved per step), and 16 requests 1.016× as fast. With
  0.44.0 or older, only Phase 1 runs.
- **What it costs.** Tokens and per-step logits are bitwise identical, and memory does not change.
- **In context.** The saving is about the size of the launch-bound routing glue that P123 priced at 11 % of a 4.75 ms
  step.
- **Scope.** The claim is the P127 delta at `7f044dd9`.
`e4b.serve.p127.launch-bound-glue.qwen3.5090.2026-10-09` ([P127](../bench/p127/RESULTS-p127.md)).

**Other opt-ins.** `E4B_PAGED_DECODE_LOOKAHEAD=1` recovers the whole host gap between decode steps, but the gap is
small: 1.0198× at one request, SLOWER against its 1.02 bar. `E4B_PAGED_LAST_LOGITS=1` projects only the final prompt
position through the LM head (#1337). `e4b.serve.p118.decode-lookahead.qwen3.5090.2026-10-08`.

**The tile table over 4 programs.** `E4B_INT4_TILE_PROGRAMS=4` splits the one-launch cumsum tile table over 4 programs
(#1433, grouped-nf4-gemm #524). On Qwen3-30B-A3B int4 it makes the captured 64-row step about 15.5 % faster with identical
tokens: P126 read DEFAULT_ON_4 on attempt 2, under its Amendment 1. It is the default (`auto`) since #1476; see the
defaults table.
`e4b.serve.p126.tile-programs.qwen3-int4.5090.2026-10-09` ([P126](../bench/p126/RESULTS-p126.md)).

**Qwen3-30B-A3B's licensed int4 stack** (calibrated int4 experts and attention, folds, router epilogue) passes the K8
gate on both texts loaded by fingerprint (expert pack `sha256:0c9955a9…`; wikitext −0.05275, c4val1 −0.06622 ppl). On
the bo7 census (the bo6c pack) it decodes ×2.067 at B=1 (238.1 tok/s) and ×2.602 at B=16 (1327.5 tok/s) against
e4b's own NF4 control on the same box.
`e4b.serve.p55x.qwen3.all-calibexp-streamed-64k.k8.2026-09-22`, `e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05`,
`e4b.serve.census.bo7.qwen3.b16.5090.2026-09-05` ([bo7](../bench/hybrid-g9/throughput-20260904/bo7/RESULTS.md)).

**Against vLLM, vLLM is ahead: 1.087 at B=1 and 1.396 at B=16** (P58: one box, identical prompt ids; vLLM 0.30.0 on
Qwen's GPTQ-Int4 checkpoint against e4b's int4 stack). The B=16 gap is mostly the experts (P86); K19 and K23's lean
glue have since taken the int4 B=16 step to 0.905× and a further 0.950×. No later same-box vLLM reading is in the
register. `e4b.serve.h2h.vllm-0.30.0.p58.qwen3.b1.5090.2026-09-22`,
`e4b.serve.h2h.vllm-0.30.0.p58.qwen3.b16.5090.2026-09-22`,
`e4b.serve.p86.qwen3.b16.kernel-census-vs-vllm-0.30.0.5090.2026-10-01` ([P58](../bench/p58/RESULTS-p58.md)).

**The other families**, each against its own NF4 control on the same box:
- **Granite:** the licensed stack decodes ×1.341 at B=1 and ×1.160 at B=16. `e4b.serve.census.bo7.granite.b1.5090.2026-09-05`
- **OLMoE and Mixtral:** the position is NF4; their int4 recipes fail the second text.
  `e4b.serve.census.bo7.olmoe.b1.5090.2026-09-05`, `e4b.serve.census.bo7.mixtral.b1.5090.2026-09-05`,
  `e4b.serve.p44.olmoe.two-text-k8.2026-09-19`
- **gpt-oss-20b:** the native MXFP4 store is licensed by KL from the dequantised reference (0.0019 nats, against 0.0222
  for an NF4 requant). `e4b.serve.p44.gptoss.store-r12.kl-vs-bf16.2026-09-19`
- **Qwen3.6-35B-A3B (hybrid):** the paged state matches transformers' own; graphs decode 2.02× (16 requests) and 3.28×
  (one) eager, and with flash-linear-attention and causal-conv1d installed the Gated DeltaNet kernels add 1.135× /
  1.118× at no measurable quality cost.
  `e4b.serve.p97.qwen36-hybrid-paged-state.5090.2026-10-02`, `e4b.serve.p101.qwen36-hybrid-decode-graphs.5090.2026-10-03`,
  `e4b.serve.p105.qwen36-gdn-kernels.5090.2026-10-03`, `e4b.serve.p106.qwen36-gdn-kernel-quality.5090.2026-10-03`
- **Gemma-4:** no serving position. The paged path is at parity with transformers' own perturbations, but the served NF4
  stack is 1.077 nats/token from the bf16 checkpoint, a cost in the early expert layers that calibration does not
  rescue (measured-private) and a graded store map does not bring under the bar
  ([P52](../bench/p52/RESULTS-p52.md)). `e4b.parity.gemma4.p108.paged-vs-floor.5090.2026-10-03`,
  `e4b.serve.p44.gemma4.nf4-vs-bf16.2026-09-19`, `e4b.quality.gemma4.calibration-refuted`

**Parity behind the serving defaults.** Paged decode is indistinguishable from the model's own attention on Granite,
gpt-oss-20b and Qwen3-30B-A3B (measured-private; [SERVING-PARITY](SERVING-PARITY.md)), read against each model's
routing-flip floor, never zero. Served T > 1 paths are inside the shipped KL bar against T = 1 decode (KL ≤ 0.0169,
top-1 ≥ 0.946), and the int8 expert activation step and the bf16 router-weight cast are indistinguishable from their
controls. `e4b.parity.granite.paged-vs-own-attention`, `e4b.parity.gptoss.paged-vs-own-attention`,
`e4b.parity.qwen3.paged-vs-own-attention`, `e4b.parity.moe-routing-flip-floor`,
`e4b.serve.p68.qwen3.t-gt-1-vs-t1.within-bar.5090.2026-09-24`, `e4b.serve.p64.qwen3.b1.expert-int8-step.5090.2026-09-24`,
`e4b.serve.p70.qwen3.b1.router-weight-cast.5090.2026-09-25`.

---

## Defaults and the reads behind them

| default | where | way back | read |
|---|---|---|---|
| chunked LM loss where fp32 logits would reach 1 GiB | `enable_fast_train` | `E4B_CHUNKED_LM_LOSS=0` | `e4b.train.chunked-lm-loss.auto.default-decision.5090.2026-10-05` |
| double-quantized expert absmax | `enable_fast_train` | `E4B_ABSMAX_DQ=0` or `absmax_dq=False` (offload, batched, residency and NVMe engines need fp32) | `e4b.train.absmax-dq.qwen3.5090.2026-10-05`, `e4b.train.absmax-dq.packed-4k.5090.2026-10-07` |
| prebound Triton launches | e4b, grouped-nf4-gemm | `E4B_TRITON_PREBIND=0`, `GNF4_TRITON_PREBIND=0` | `e4b.train.prebind.qwen3.5090.2026-10-05` |
| PyTorch's reentrant checkpoint | `enable_fast_train` (a `Trainer` with `gradient_checkpointing=True` re-enables Hugging Face's) | `E4B_CKPT_OFFLOAD=0` | `e4b.train.ckpt-flavour.default.5090.2026-10-07` |
| chunked held-out loss | `enable_fast_train` | `E4B_CHUNKED_EVAL_LOSS=0` | `e4b.train.chunked-eval-loss.packed-4k.5090.2026-10-07` |
| expert combine over row chunks | `enable_fast_train` | `E4B_COMBINE_CHUNK=0` | `e4b.train.combine-row-chunks.packed-4k.5090.2026-10-07` |
| fused q/k/v training projection (Qwen3-MoE, NF4 attention with adapters; 0.868–0.906 of the step on two RTX 5090 hosts, 13.9–14.6 % fewer launches, about 24 MB; `disable_fast_train` restores the three projections) | `enable_fast_train` | `E4B_TRAIN_FUSE_QKV=0` | `e4b.train.p129.fused-qkv.qwen3.5090.2026-10-10`, `e4b.train.p129.fused-qkv.second-host.5090.2026-10-10` |
| bucketed LoRA-delta padding, compact | grouped-nf4-gemm | `NF4_QLORA_PAD_BUCKETS=0`, `NF4_QLORA_COMPACT_BUCKETS=0` | `e4b.train.pad-buckets.auto.default-decision.5090.2026-10-06`, `e4b.train.compact-buckets.packed-4k.5090.2026-10-07` |
| bucketed CUDA-graph decode (sm_89+) | `serve_paged` | `E4B_PAGED_GRAPHS=0` | `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03`, `e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.2026-10-03` |
| estimate-sized slots | `serve_paged` | `E4B_PAGED_MAX_SEQS=16` | `e4b.serve.sc2e.64-slots-default-buckets.qwen3.5090.2026-10-07` |
| one graph per decode step | `serve_paged` | `E4B_PAGED_BUCKETS=1,2,4,8,16` | `e4b.serve.sc2e.64-slots-buckets-auto.qwen3.5090.2026-10-07`, `e4b.serve.p117.wide-bucket-quality.qwen3.5090.2026-10-08` |
| K19's lean dispatch on prefill rows (`auto`: wherever the installed K19 takes `scatter=` / `gather_div=`; bitwise) | serving | `E4B_PREFILL_LEAN_DISPATCH=0` | `e4b.serve.p130.prefill-lean-dispatch.qwen3.5090.2026-10-10` |
| the decode folds on prefill rows (`qwen3_moe` only; changes prefill arithmetic, read at parity) | `serve_paged` | `E4B_FUSE_PREFILL_GLUE=0` | `e4b.serve.p130.prefill-glue-speed.qwen3.5090.2026-10-10`, `e4b.serve.p130.prefill-glue-quality.qwen3.5090.2026-10-10` |
| one KV-table selection per step | decode graphs | `E4B_KV_STEP_SELECT=0` | `e4b.serve.p111.kv-step-select.qwen3.5090.2026-10-04` |
| programmatic dependent launch, capped at 8 rows | grouped-nf4-gemm | `GNF4_PDL=0` | `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04` |
| bandwidth-targeted NF4 decode GEMV | grouped-nf4-gemm | `GNF4_GEMV_BW=0` | `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07` |
| grouped small-M routes above T == 1 (K19, K23, K21, K25; K25 read on Granite, OLMoE and Qwen3's W16 step) | serving | `E4B_INT4_GROUPED_SMALLM=0`, `E4B_INT4_LEAN_GLUE=0`, `E4B_MXFP4_GROUPED_SMALLM=0`, `E4B_NF4_GROUPED_SMALLM=0` | `e4b.serve.p88.qwen3.int4.k19-b16.5090.2026-10-01`, `e4b.serve.p89.qwen3.int4.k23-lean-glue-b16.5090.2026-10-01`, `e4b.serve.p90.gptoss.mxfp4.k21-b16.5090.2026-10-01`, `e4b.serve.p96.nf4-families.k25-windowed-k8.5090.2026-10-02`, `e4b.serve.p121.k25-w16.qwen3.5090.2026-10-08` |
| the one-launch tile table above 256 routed rows, up to the size read (`next_pow2(E) × next_pow2(R) ≤ 128 × 512`; takes effect with a grouped-nf4-gemm release carrying #519's chunked table; the one-piece table read 1.44× slower) | serving | `E4B_INT4_WIDE_TILES=0` | `e4b.serve.p122.wide-tiles-chunked.qwen3-int4.5090.2026-10-08`, `e4b.serve.p120.wide-tiles.qwen3-int4.5090.2026-10-08` |
| the one-launch tile table over 4 programs, inside the size read (`next_pow2(E) × next_pow2(R) ≤ 128 × 512`; takes effect with a grouped-nf4-gemm release carrying #524's `programs=`; Qwen3-30B-A3B's 64-row step about 15.5 % faster) | serving | `E4B_INT4_TILE_PROGRAMS=1` | `e4b.serve.p126.tile-programs.qwen3-int4.5090.2026-10-09` |
| attention projections of 17–64 decode rows on the int4 small-M GEMM, from the int4 grid instead of a cached bf16 copy (with the K16 route on; takes effect with a grouped-nf4-gemm release carrying #522's `block_m=`; Qwen3-30B-A3B's 64- and 32-row steps 3.2–3.8 % faster) | `Int4Linear` (serving) | `E4B_ATTN_INT4_WIDE=0` | `e4b.serve.p124.attn-int4-wide.qwen3-int4.5090.2026-10-09` |
| router weights cast to bf16 at ≤ 64 rows (`softmax_topk`) | fused router epilogue | `E4B_ROUTER_EPI_CAST=0` | `e4b.serve.p70.qwen3.b1.router-weight-cast.5090.2026-09-25` |
| the paged FP8 attention decode capped at 16 splits (`qwen3_moe` only; buckets 1, 2 and 4 move, 32 / 32 / 23 -> 16; bitwise-identical to the plan at contexts ≤ 1024; one request 1.044× as fast, 16 unchanged, measured on Qwen3-30B-A3B on an RTX 5090 (170 SMs); other `qwen3_moe` checkpoints (e.g. Qwen3-235B-A22B) and other cards (the cap moves B 1–6 on a PRO 6000, B 1–4 on a 4090) take it by family without a read of their own) | `serve_paged` | `E4B_ATTN_SPLIT_CAP=0` | `e4b.serve.p143.attn-split-cap.qwen3.5090.2026-10-19` |
| the lm_head on its per-row int8 grid, its one-row decode GEMV through grouped-nf4-gemm K36 (`qwen3_moe` at the 151936 x 2048 head only; needs grouped-nf4-gemm 0.48.0 or later, else it resolves off, recorded; one request 1.052× as fast, 16 requests 0.992, an observed cost not attributed, measured on Qwen3-30B-A3B on an RTX 5090 (170 SMs); quality read on Qwen3-30B-A3B @ad44e777, same-shape checkpoints inherit the default unread; other head shapes, e.g. Qwen3-235B-A22B's 151936 x 4096, stay off) | `serve_paged` | `E4B_INT8_HEAD_BW=0` | `e4b.serve.p153.int8-head-bw.qwen3.5090.2026-10-19` |
| K25's codebook decode by grouped-nf4-gemm K39's byte permute (`lut="prmt"`; `qwen3_moe` only; bit-identical to the select tree, every token identical; needs grouped-nf4-gemm 0.50.0 or later and an NVIDIA device, else it resolves to the tree, recorded in `info["k25_lut"]`; sixteen requests 1.0925× as fast (the smallest of four ABBA pairs), one request unchanged, measured on Qwen3-30B-A3B on an RTX 5090 (170 SMs) at the `today` plan; other `qwen3_moe` checkpoints (e.g. Qwen3-235B-A22B) take it by family with no speed claimed; other families stay on the tree) | `serve_paged` | `E4B_K25_LUT=tree` | `e4b.serve.p158.k25-prmt.qwen3.5090.2026-10-20` |
| K25's tile plan `c1` under K39's decode (on main after 0.56.0; `qwen3_moe` only, where K25's decode resolved to `prmt`, else `today`, recorded in `info["k25_plan"]["source"]`; bit-identical to `today`'s plan, every token identical; sixteen requests 1.4654× as fast (the smallest of four ABBA pairs), one request unchanged (0.9966), measured on Qwen3-30B-A3B on an RTX 5090 (170 SMs); other `qwen3_moe` checkpoints take it by family with no speed claimed; other families stay on `today`) | `serve_paged` | `E4B_K25_PLAN=today` | `e4b.serve.p161.k25-plan-c1.qwen3.5090.2026-10-20` |
| B=1 fused stack (fused q/k/v and three glue folds) on Qwen3-MoE; its router epilogue, the only fold that engages, on Qwen3.5/3.6-MoE; its three glue folds on Mixtral | `serve_paged` | `E4B_PAGED_FUSE_QKV=0`, `E4B_FUSE_T1_GLUE=0`, `E4B_FUSE_T1_GLUE_R2=0`, `E4B_FUSE_ROUTER_EPI=0` | `e4b.serve.p115.fused-stack-combined.qwen3.5090.2026-10-08`, `e4b.serve.p115.fused-stack-speed.qwen3.5090.2026-10-07`, `e4b.serve.fam.router-epilogue-speed.qw36.5090.2026-10-09`, `e4b.serve.p135.fold-speed.mixtral.5090.2026-10-18` |

---

## What changed

Superseded and retired readings are not repeated here. The register keeps each with its successor or its reason, and
[`STATUS-RECORD.md`](STATUS-RECORD.md) keeps the dated account. The retractions most often met elsewhere:
- The 13.47× training speedup is retired (`e4b.retired.13.47x-training-speedup`): transformers v5 fused the per-expert loop upstream, so roughly half the multiple is now upstream's work.
- "vLLM 6.31× ahead" is retired (`e4b.retired.vllm-6.31x-ahead`): template prompts on a different box.
- "The fp8 paged KV cache costs +0.047 ppl on Qwen3" is retired (`e4b.retired.fp8-kv-cost`): +0.0058 nats, below the model's own floor.
- The pre-registered KL gate is retired (`e4b.retired.kl-gate-threshold`): its first measurement rejected NF4 experts, which this package ships.
- Granite's "302 tok/s, ×1.59 with int4 experts" is retracted: those experts cost +0.063 ppl, over the 0.05-ppl gate.

---

## What is open

- **#344:** Gemma-4 failed to load on 2 of 6 rented hosts; lane P55 did not reproduce it on the current loader, and
  `E4B_LOAD_SYNC_DEBUG=1` now binds a fault to a load stage. `e4b.open.gemma4-load-fault`
- **gpt-oss has no expert-LoRA training path** here; grouped-nf4-gemm's `ExpertsMxfp4LoRA` route is experimental.
- **`enable_batched_train`'s envelope** (which shapes reach its kernel) is unmeasured; `batched_fallback_stats(model)`
  counts the fallback.
- **#728:** the int4 attention projections' int8 activation step is unpriced, and so is the int8 step at B = 16.
- **The B=16 expert GEMV** runs at about 77 % of #564's byte roofline at real routing, about 1.4 ms/step of headroom,
  but no lever is licensed. `e4b.serve.p57.qwen3.b16.distinct-experts.5090.2026-09-22`,
  `e4b.serve.p61.qwen3.b16.expert-gemv-cost-split.5090.2026-09-23`
- **The residency cold-cost model** under-predicts the pipelined engine's gather by 1.57–2.02× on a PCIe gen 4 link;
  the follow-up is filed against grouped-nf4-gemm's `kernel/cold_deadline.py`.
  `e4b.serve.p66.qwen3.pipelined-gather-over-cold-deadline.5090.2026-09-24`
- **P37's divergent pack:** four builds on at least three boxes produce the licensed pack byte for byte; why P37's host
  built a different one is unexplained, and no open issue carries it.
- **Open register rows:** reproducing the TR2 training receipt from published artifacts (`e4b.open.tr2-repro-gap`);
  int8-offload's best training eval, confounded by an evaluator offset (`e4b.open.int8-offload-confounded`).
- **Open campaigns:** training #835, serving #846, single-stream decode #1313. P127 read: its launch-bound glue changes (Phase 1,
  #1448; Phase 2, #1472/#1477/#1482 with grouped-nf4-gemm #526–#530, first released in 0.45.0) decode one request
  1.1321× as fast and 16 requests 1.0156× on Qwen3-30B-A3B on an RTX 5090, bitwise identical
  (`e4b.serve.p127.launch-bound-glue.qwen3.5090.2026-10-09`, [results](../bench/p127/RESULTS-p127.md)). #1472's kernel
  options take effect with grouped-nf4-gemm 0.45.0 or later.
- **Held-out loss parity against Unsloth and axolotl is unread.** HP1 closed at its calibration: at 48 layers the
  shared evaluator's per-row zero-adapter gap set a tolerance of 0.044, against a 0.01 ceiling
  ([results](../bench/hp1/RESULTS-HP1.md)). HP2 stopped at G1, because its instrument could not read the expected
  margin ([results](../bench/hp2/RESULTS-HP2.md)). Neither makes a parity, ahead or behind claim. The held-out losses in
  "Training against other frameworks" above are each framework's own in-loop reading, not a shared evaluator's. HPN
  ([design](../bench/hpn/HPN-DESIGN.md), $0 on an RTX A2000) reads where the evaluator-vs-in-loop gap comes from.
- **Older documents' debts:** `POST_AUDIT_WORK_QUEUE.md` (Q1–Q4), `TRAIN_PLACEMENT_CERTIFICATE.md` (a scoped S10),
  `LAYOUT_FACTS.md` (training determinism UNKNOWN), and `support_matrix.md`'s footer hash, which no longer matches its
  bytes and is recorded, not fixed, because the file is anchored.

---

## Where the detail is

- [`STATUS-RECORD.md`](STATUS-RECORD.md): the dated narrative to 0.50.0, every superseded reading with its reason.
  Frozen; it is not appended to.
- [`claims.json`](claims.json) ([schema](claims-schema.md)): every claim's full text, conditions, status and evidence.
- [`CHANGELOG.md`](../CHANGELOG.md): what each release changed and the reads behind it.
- `bench/<lane>/RESULTS-*.md`: each lane's receipt and verdict, linked from the positions above.
- [`capabilities.json`](capabilities.json), [`SOLUTIONS.md`](SOLUTIONS.md), [`SERVING.md`](SERVING.md),
  [`SERVING-PARITY.md`](SERVING-PARITY.md) and [`METHODOLOGY.md`](METHODOLOGY.md): entry points, serving and method.
