# Status — what this package does, what is current, what is open

**As of 2026-10-08, version 0.50.0** (the version of record is `pyproject.toml`'s). This page states the current
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

**Expert offload takes 30B-class MoEs past VRAM.** In a training step Qwen3-30B-A3B peaks at 7.16 GB and
Gemma-4-26B-A4B at 8.47 GB, where both run out of memory without offload, for about +11 % s/step at OLMoE scale. The
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

### Training against other frameworks

**Qwen3-30B-A3B against Unsloth: Unsloth spends 1.92× e4b's GPU time per step; wall-clock 2.80× on an AMD EPYC
7713.** TC1's field recipe (alpaca, seq 2048, micro-batch 2 × accum 4, r 16, AdamW-8bit) on one RTX 5090, the work
matched, both frameworks on one stack, e4b at every current default. The wall ratio depends on the host: Unsloth runs
14× e4b's CPU ops per step and keeps its GPU busy for 0.33 of its step against e4b's 0.49, so a slower host stretches
Unsloth's step more. Held-out loss agrees within 0.0012; e4b peaks 1.89 GB higher (26.16 against 24.27 GB). HF + PEFT
runs out of memory on the 32 GB card. `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio`,
`e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.hf-oom` ([results](../bench/h2h-2026-10-02/tc1/RESULTS-tc1-pos69.md)).

**On packed 4,096-token rows, Unsloth spends 1.160× e4b's GPU time per step.** The wall-clock ratio follows the host:
1.472 on the profiled host, 1.287 and 1.773 on two others. At e4b's defaults its training phase peaks 1.04 GB above
Unsloth's 24.86 GB, and 0.30 GB above with the opt-in `E4B_CKPT_OFFLOAD=1`.
`e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio`,
`e4b.train.memory.packed-4k-position-defaults.5090.2026-10-07`
([results](../bench/h2h-2026-10-02/tc1/RESULTS-tc1-pos68.md)).

**Mixtral-8x7B against Unsloth: e4b is faster, Unsloth/e4b 1.144**, both resident on one stack (torch 2.12.1 /
transformers 5.5.0), on an AMD EPYC 7B13: 3.24 against 3.71 s/step, at 2.07 GB more peak on e4b (its fp32 expert
absmax). The register's other Mixtral ratios answer other questions: 0.836 put e4b on the field image's torch 2.8 stack
on a Core Ultra 9 285K, the host most favourable to Unsloth seen (the same-stack box read that stack at 1.013), and
0.361 is tp2's footprint trade, e4b under CPU expert offload at 3.22 GB against Unsloth resident at 29.16 GB, not a
resident position. `e4b.train.h2h.unsloth.mixtral.5090.2026-10-05.same-stack`,
`e4b.train.h2h.unsloth.mixtral.5090.2026-10-04.dense-default`, `e4b.train.h2h.unsloth.mixtral.5090.2026-09-06`
([results](../bench/h2h-2026-10-02/tc2/RESULTS-tc2-mixtral-samestack.md)).

**Granite and OLMoE: e4b is faster than each comparator** (TC2, 2026-10-04, one RTX 5090): HF/e4b 1.299 and
axolotl/e4b 1.150 on Granite-3.1-3B-A800M, Unsloth/e4b 1.821 on OLMoE-1B-7B, each pair comparable or equivalent on
held-out loss. e4b's peak is about half HF's on Granite and 1.70 GB above Unsloth's on OLMoE. Unsloth adapts no expert
parameter of Granite, so there is no Unsloth position there, and gpt-oss-20b has no adapter set common to the
frameworks. `e4b.train.h2h.hf.granite.5090.2026-10-04`, `e4b.train.h2h.axolotl.granite.5090.2026-10-04`,
`e4b.train.h2h.unsloth.olmoe.5090.2026-10-04`, `e4b.train.h2h.unsloth.granite.5090.2026-10-02.coverage`,
`e4b.train.h2h.unsloth.gptoss.5090.2026-10-02.no-common-set` ([TC2](../bench/h2h-2026-10-02/tc2/README.md)).

**Qwen3.6-35B-A3B: at its defaults e4b does not fit the matched set resident on 32 GB; Unsloth does**, at 30.47 GB.
Under expert offload e4b trains the set at 19.35 GB. With the comparator's bytes at micro-batch 1 (a measurement hook,
not a training option) e4b trains it resident and Unsloth/e4b reads 2.049, a labelled row.
`e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-04`, `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02.e4b-offload`,
`e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-04.mb1-dq-frozen4`.

**axolotl on Qwen3-30B-A3B: e4b is faster, axolotl/e4b 2.775** at matched work (1.979 on a second host). At steady
state (steps 101–200) axolotl's scattermoe native-best / e4b as shipped reads 1.238 and 1.146 on two hosts, while at
step 200 axolotl's held-out is 0.7705 against e4b-as-shipped's 0.7948. Both were read on 2026-10-03, before the later
training defaults. `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03`,
`e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state`
([results](../bench/h2h-2026-10-02/tc1/RESULTS-tc1-matched19.md)).

**On an H100 NVL at e4b 0.45.0's default settings, e4b is faster per step: Unsloth/e4b 1.061**, at 2.93 GB more peak
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
- `NF4_QLORA_SINGLE_LADDER=1` (grouped-nf4-gemm): 0.797 of the step with fp32 adapters on a host-bound box,
  1.015 with bf16 adapters (the shipped arm). `e4b.train.single-ladder.field.5090.2026-10-08`
  Its `auto` (the ladder only with fp32 adapters) held its mechanism on a second host, but the step time went unread
  there: the host was loaded. `e4b.train.single-ladder-auto.field.5090.2026-10-08`

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

**Single-request decode.** grouped-nf4-gemm's bandwidth-targeted NF4 decode GEMV decodes one request 1.2417× as fast,
16 unchanged, within P110's quality bar; it is grouped-nf4-gemm's default from 0.43.0 at Qwen3's NF4 expert shapes on
GPUs with 160 or more SMs. `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`.

**The B=1 fused stack is the default on Qwen3-MoE.** On top of grouped-nf4-gemm 0.43.0's bandwidth GEMV, fused q/k/v
plus three glue folds decode one request 1.5902× and 16 requests 1.1644× as fast on Qwen3-30B-A3B NF4, and P115 Phase
D's SANE read at T == 1 passes (bias +0.00541 nats, argmax agreement 0.9674). The four knobs resolve to `auto` on the one
family with that read and stay off elsewhere: gpt-oss-20b failed Phase C's argmax gate (0.924 against 0.95), and
Qwen3.5/3.6-MoE and Granite-MoE wait for lane FAM's reads at T == 1 (#1362). `0` on each knob is the way back.
`e4b.serve.p115.fused-stack-combined.qwen3.5090.2026-10-08`, `e4b.serve.p115.fused-stack-speed.qwen3.5090.2026-10-07`,
`e4b.serve.p115.fused-stack-quality.qwen3.5090.2026-10-07`,
`e4b.serve.p115.fused-stack-engagement.gptoss-qwen36.5090.2026-10-08`,
`e4b.serve.p115.fused-stack-quality.granite.5090.2026-10-08` ([P115](../bench/p115/RESULTS-p115.md)).

**Other opt-ins.** `E4B_PAGED_DECODE_LOOKAHEAD=1` recovers the whole host gap between decode steps, but the gap is
small: 1.0198× at one request, SLOWER against its 1.02 bar. `E4B_PAGED_LAST_LOGITS=1` projects only the final prompt
position through the LM head (#1337). `E4B_INT4_WIDE_TILES=1` builds the tile table above 256 routed rows in one
launch; on Qwen3-30B-A3B int4 the 64-row step is 1.44× slower (the table costs 10.4 ms a step), so it is SLOWER
and stays opt-in. `e4b.serve.p118.decode-lookahead.qwen3.5090.2026-10-08`,
`e4b.serve.p120.wide-tiles.qwen3-int4.5090.2026-10-08`.

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
| bucketed LoRA-delta padding, compact | grouped-nf4-gemm | `NF4_QLORA_PAD_BUCKETS=0`, `NF4_QLORA_COMPACT_BUCKETS=0` | `e4b.train.pad-buckets.auto.default-decision.5090.2026-10-06`, `e4b.train.compact-buckets.packed-4k.5090.2026-10-07` |
| bucketed CUDA-graph decode (sm_89+) | `serve_paged` | `E4B_PAGED_GRAPHS=0` | `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03`, `e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.2026-10-03` |
| estimate-sized slots | `serve_paged` | `E4B_PAGED_MAX_SEQS=16` | `e4b.serve.sc2e.64-slots-default-buckets.qwen3.5090.2026-10-07` |
| one graph per decode step | `serve_paged` | `E4B_PAGED_BUCKETS=1,2,4,8,16` | `e4b.serve.sc2e.64-slots-buckets-auto.qwen3.5090.2026-10-07`, `e4b.serve.p117.wide-bucket-quality.qwen3.5090.2026-10-08` |
| one KV-table selection per step | decode graphs | `E4B_KV_STEP_SELECT=0` | `e4b.serve.p111.kv-step-select.qwen3.5090.2026-10-04` |
| programmatic dependent launch, capped at 8 rows | grouped-nf4-gemm | `GNF4_PDL=0` | `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04` |
| bandwidth-targeted NF4 decode GEMV | grouped-nf4-gemm | `GNF4_GEMV_BW=0` | `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07` |
| grouped small-M routes above T == 1 (K19, K23, K21, K25; K25 read on Granite, OLMoE and Qwen3's W16 step) | serving | `E4B_INT4_GROUPED_SMALLM=0`, `E4B_INT4_LEAN_GLUE=0`, `E4B_MXFP4_GROUPED_SMALLM=0`, `E4B_NF4_GROUPED_SMALLM=0` | `e4b.serve.p88.qwen3.int4.k19-b16.5090.2026-10-01`, `e4b.serve.p89.qwen3.int4.k23-lean-glue-b16.5090.2026-10-01`, `e4b.serve.p90.gptoss.mxfp4.k21-b16.5090.2026-10-01`, `e4b.serve.p96.nf4-families.k25-windowed-k8.5090.2026-10-02`, `e4b.serve.p121.k25-w16.qwen3.5090.2026-10-08` |
| router weights cast to bf16 at ≤ 64 rows (`softmax_topk`) | fused router epilogue | `E4B_ROUTER_EPI_CAST=0` | `e4b.serve.p70.qwen3.b1.router-weight-cast.5090.2026-09-25` |
| B=1 fused stack (fused q/k/v and three glue folds) on Qwen3-MoE | `serve_paged` | `E4B_PAGED_FUSE_QKV=0`, `E4B_FUSE_T1_GLUE=0`, `E4B_FUSE_T1_GLUE_R2=0`, `E4B_FUSE_ROUTER_EPI=0` | `e4b.serve.p115.fused-stack-combined.qwen3.5090.2026-10-08`, `e4b.serve.p115.fused-stack-speed.qwen3.5090.2026-10-07` |

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
- **Registered, not yet read:** TC1 amendment 72 (amendment 71's `auto` box again, on a third host). The head-to-head
  campaigns stay open: training #835, serving #846, single-stream decode #1313.
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
