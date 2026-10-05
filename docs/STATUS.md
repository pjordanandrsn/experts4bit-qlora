# Status — what this package does, what changed, what is open

**As of 2026-10-05, version 0.48.0** (the version of record is
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

**At the field recipe, on the cards people own** (lane TC3, 2026-10-02, one
rented RTX 4090 and the owned RTX A2000,
[`bench/h2h-2026-10-02/tc3/`](../bench/h2h-2026-10-02/tc3/README.md); register
`e4b.train.frontier.qwen3.4090-24gb.2026-10-02` with its `.unsloth-resident`,
`.e4b-resident-oom`, `.hf-axolotl` and `.offload-keeps-trajectory` rows, and
`e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`). Qwen3-30B-A3B at TC1's
matched set — the same 642,514,944 fp32 adapters, init and tokens — **trains
on a 24 GB card under e4b's expert offload at 11.88 GB peak and 10.52
s/step**, its trajectory EQUIVALENT-TO-RESIDENT against the 5090 reading
(median per-step |Δ| 0.0025, held-out |Δ| 0.0032), while resident e4b needs
more than the card has (OOM at 24.45 GB at both recipes and at step 18 as
shipped). **Unsloth 2026.9.14 trains the same set resident on the 24 GB
card** (24.22 GB peak, 10.03 s/step, the same loss) — the lane predicted it
would not, and that row is the finding; HF neither fits nor offloads there.
axolotl's three rows there are not readings of axolotl: its plain arm failed
on this harness's no-autocast loop (TC1 amendment 4, corrected 2026-10-02)
and is re-asked. On the 12 GB card e4b trains
the set under offload at micro-batch 1 (10.46 GB peak on an A2000 behind a
6-core Xeon, held-out 0.8483 beside the 5090's 0.8516 / 0.8487; the A2000 is
a correctness-only testbed, so its step time is not quoted); Unsloth's loader
dispatches modules to the CPU there and refuses,
axolotl's cu130 wheels need a newer driver than the host has, and HF, given the
budget to reach the card, OOMs at load (TC3 P2 HELD).
A fit table, one draw per row — no position; the 32 GB position is TC1's.

**Kimi-K3 runs at full depth on a 12 GB card** (2026-09-28, the released
0.37.5 / 0.33.4; **measured** — [`bench/kimi-k3-a2000/`](../bench/kimi-k3-a2000/RESULTS-kimi-k3-a2000.md),
`e4b.offload.kimi-k3.full-depth.a2000.five-runs.2026-09-28`). All 93 layers
run on one RTX A2000 at 4.07 GB peak VRAM once Triton's cache is warm. The
MXFP4 experts stream from a 1.446 TB SSD arena, and the 108.76 GB dense
side is served from the checkpoint's byte offsets, with 0 bytes pinned in
host RAM. In five processes the model completed "The capital city of France
is" as " Paris. It is" (a feasibility and fidelity result: the A2000 is a
correctness-only testbed, so no decode speed is quoted). Against Fireworks'
kimi-k3 on the same raw paragraph, the per-token NLL correlates at r = 0.9965
(`e4b.quality.kimi-k3.vs-fireworks.per-token.a2000.2026-09-28`), and a
routing-replayed cache gate passes the real cache at cos 0.999966 and fails
a cache with its KDA state zeroed at 0.877523
(`e4b.parity.kimi-k3.cache-gate.replayed-routing.a2000.2026-09-28`). **The
forward is not reproducible run to run:** p(" Paris") ranged 68.90–71.59 %
over the five processes. Autotuning is ruled out. The source is the MXFP4
prefill combine's atomic `index_add_`: with torch's deterministic algorithms on,
three processes are bit-identical at all 92 MoE calls. grouped-nf4-gemm#410's
ordered combine alone, with deterministic mode off, reproduces them bit for bit
(`e4b.parity.kimi-k3.prefill-drift-is-the-combine.a2000.2026-09-28`).
grouped-nf4-gemm 0.33.6 ships the fix. On it, three released-package processes
are bit-identical with deterministic mode off
(`e4b.parity.kimi-k3.reproducible-on-gnf4-0.33.6.a2000.2026-09-29`). A custom
driver wires the engines, so this is not the `load_moe_4bit_streaming` path.

**The fused training path is faster at equal loss.** Across two 30B-class
MoEs, five datasets each, 200 steps per cell: 1.52–1.81× per step at
0.75–0.81× peak VRAM and 0.80–0.92× energy, with loss parity on both
registered criteria and the frozen 4-bit stack bit-identical (12.85 GB per
Gemma-4 cell, 16.31 GB for Qwen3). *(Corrected 2026-09-30: energy read
0.86–0.92×, Gemma-4's alone.)* Default to
`enable_fast_train(model, dgrad=True)`.

**Training on real weights is receipted per family** (lane tp1, 2026-09-05,
one rented RTX 5090 under the shipped 0.35.0 / 0.30.0 code; **measured** —
receipt [`bench/train-parity-20260905/tp1/`](../bench/train-parity-20260905/tp1/README.md),
table in its [`RESULTS-tp1.md`](../bench/train-parity-20260905/tp1/RESULTS-tp1.md);
register `e4b.train.parity.tp1.<family>.<arm>.2026-09-05`, whose rows carry
every number here; all six families through the registered arms, 18 result
lines, every attempt a row). Each family goes through the direct
`load_moe_4bit_streaming` + `verify_moe_4bit(strict=True)` path on real
weights, then `reference` (the per-expert loop), `fused`
(`enable_fast_train(dgrad=True)`) and `batched` (`enable_batched_train`)
for 60 steps on the registered `clinical` text, judged against the family's
own reference on the same box — |Δ final train loss| ≤ 0.05 and median
step-wise |Δ| ≤ 0.05 — with cost reported and never gated.
**The fused path PASSES on every family that has one:** Granite-3.1-3B-A800M
(×5.86 per step, on the corrected-counter re-run after a HARNESS_ERROR first
attempt — a closure bug in the harness's kernel counter, amendment 3, not the
shipped code), OLMoE-1B-7B-Instruct (×3.22), Qwen3-30B-A3B resident on the
32 GB card (×2.56), Gemma-4-26B-A4B-it (×2.37; inside the band by 0.0026, the
tightest cell; the `-it` checkpoint loaded on that host without the #344
fault) and Mixtral-8x7B-Instruct under `offload=True` (×1.26 at ×0.49 the
reference loop's peak VRAM). **The batched path PASSES where its kernel
engages and is VOID where it does not:** PASS on Granite (×3.93) and Mixtral
(×1.27); VOID on OLMoE, Qwen3 and Gemma-4 — `engines/batched.py` fell back
to the reference forward per call above `_PAD_WASTE_LIMIT` with no counter
(Gemma-4: on 9 of 60 steps no layer reached the kernel) — and a VOID row
carries no parity number, however its loss curve reads. **gpt-oss `fused` /
`batched` are REFUSED** (0 patched: the loader builds its experts bare);
attention-only QLoRA over its frozen experts trains with the stacks
bit-exact, and the kernel package's experimental MXFP4 route trains its
experts on its own text with the step-0 canary passing — experimental,
never licensed. The capability list (`qlora-fused-moe-experts.model_families` in
[`capabilities.json`](capabilities.json)) is exactly the families whose
`fast_train` is `supported`: `olmoe`, `qwen3_moe`, `gemma4_text`, `mixtral`,
`granitemoe` — the last two entered on this lane — and, since lane MG1
(2026-10-04), `lfm2_moe`, `granitemoehybrid`, `ernie4_5_moe` and `nemotron_h`, and since MG1 amendment 3 (2026-10-05)
`qwen3_5_moe`;
`gpt_oss` stays out. No
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
| `olmoe` | supported (tp1) | supported (tp1; `e4b.train.olmoe-converges`) | **supported** — tp1 OK · PASS on the registered text with real weights (`e4b.train.parity.tp1.olmoe.fused.2026-09-05`); re-read on the moe-generalize code by MG1, PASS (`e4b.train.parity.mg1.olmoe.fused.2026-10-04`) | **void** — tp1 OK · VOID: the `_PAD_WASTE_LIMIT` fallback engaged without a counter (`…olmoe.batched…`) | not_tested (outside tp1; the arena ladder, `e4b.offload.arena-vs-host-ram`, is a separate receipt) | n/a |
| `qwen3_moe` | supported (tp1, resident on a 32 GB card; flagship) | supported (tp1; flagship) | **supported** — tp1 OK · PASS resident on one 5090 (`…qwen3.fused…`), beside the flagship's five datasets (`e4b.train.flagship-matrix`) | **void** — tp1 OK · VOID: the kernel reached on a fraction of the layers every step (`…qwen3.batched…`); the dgrad-gate trajectory stands on its own fixture | not_tested (outside tp1; the arena ladder, `e4b.offload.arena-vs-host-ram`, is a separate receipt) | n/a |
| `gemma4_text` | supported (tp1: the `-it` checkpoint loaded on this host, no #344; flagship: base) | supported (tp1; flagship) | **supported** — tp1 OK · PASS with the step-wise median inside the band by a small margin (`…gemma4.fused…`); MG1 re-read it after #1048 moved its fused RMSNorm to its own fp32 multiply, PASS with the median further inside the band (`e4b.train.parity.mg1.gemma4.fused.2026-10-04`), beside the model-2 flagship (`e4b.train.flagship-matrix`) | **void** — tp1 OK · VOID: on some steps no layer reached the kernel (`…gemma4.batched…`) | not_tested (outside tp1; the arena ladder, `e4b.offload.arena-vs-host-ram`, is a separate receipt) | n/a |
| `granitemoe` | supported (tp1: the first direct real-weight load) | supported (tp1 OK) | **supported** — tp1 OK · PASS on the corrected-counter re-run (`…granite.fused…`; attempt 1 a kept HARNESS_ERROR of the harness's counter, `…granite.fused.attempt1…`, amendment 3) — **entered `model_families` on it** | supported — tp1 OK · PASS (`…granite.batched…`) | not_tested | n/a |
| `gpt_oss` | supported (bare `GptOssExperts4bit`; tp1) | refused — no `ExpertsLoRA`; attention-only QLoRA trains (`…gptoss.attn_only…`, OK · no pair) | refused — `enable_fast_train` returns 0 (`…gptoss.fused…`, REFUSED) | refused — `enable_batched_train` returns 0 (`…gptoss.batched…`, REFUSED) | refused — `enable_mxfp4_nvme_residency` refuses bias-carrying modules (#402; it had defaulted to the V4 epilogue, #397), `enable_nvme_train_residency` refuses bare modules, and the `arena_train=True` wrap is refused on structure | **experimental** — grouped-nf4-gemm's `ExpertsMxfp4LoRA`; tp1 canary and provenance passed on its own text (`…gptoss.mxfp4…`, EXPERIMENTAL); never licensed |
| `mixtral` | supported (tp1: the first real-weight pass through the `w1/w3/w2` fusion, `offload=True`) | supported (tp1, offload) | **supported** — tp1 OK · PASS under offload at half the reference loop's peak VRAM (`…mixtral.fused…`); **entered `model_families` on this row** | supported — tp1 OK · PASS, the kernel reached everywhere (the 8-expert shape; `…mixtral.batched…`) | not_tested | n/a |
| `lfm2_moe` | supported (MG1) | supported (MG1) | **supported** — MG1 OK · PASS on the released LFM2-8B-A1B (`e4b.train.parity.mg1.lfm2.fused.2026-10-04`); its `out_proj` attention gets LoRA since #1048; **entered `model_families` on this row** | not_tested | not_tested | n/a |
| `granitemoehybrid` | supported (MG1) | supported (MG1) | **supported** — MG1 OK · PASS on granite-4.0-h-tiny, its Mamba blocks on mamba-ssm / causal-conv1d (`e4b.train.parity.mg1.graniteh.fused.2026-10-04`); **entered `model_families` on this row** | not_tested | not_tested | n/a |
| `ernie4_5_moe` | supported (MG1) | supported (MG1) | **supported** — MG1 OK · PASS on ERNIE-4.5-21B-A3B-PT, its interleaved rotary refused by the fused-RoPE semantics probe and kept (`e4b.train.parity.mg1.ernie.fused.2026-10-04`); **entered `model_families` on this row** | not_tested | not_tested | n/a |
| `nemotron_h` | supported (MG1) | supported (MG1) | **supported** — MG1 OK · PASS on Nemotron-3.5-Lightning-30B-A3B, the non-gated relu² expert path's first full-model reading (`e4b.train.parity.mg1.nemotron.fused.2026-10-04`); **entered `model_families` on this row** | not_tested | not_tested | n/a |
| `qwen3_5_moe` | supported (MG1, resident) | supported (MG1) | **supported** — MG1 OK · PASS on Qwen3.6-35B-A3B, resident (`e4b.train.parity.mg1.qwen3_5.fused.2026-10-04`); P2 read by MG1 amendment 3 on tp1's own fused arm, the dgrad kernel on every frozen-GEMM backward (`e4b.train.parity.mg1.qwen3_5.p2.2026-10-05`, [read](../bench/moegen/mg1/mg1-a3-5090-1/RESULTS-mg1-a3.md)); amendment 2's ladder OOM stays a row ([read](../bench/moegen/mg1/mg1-a2-5090-2/RESULTS-mg1-a2.md)); **entered `model_families` on this row** | not_tested | not_tested | n/a |

**Lane MG1 (2026-10-04, [`bench/moegen/mg1/mg1-5090-2/RESULTS-mg1.md`](../bench/moegen/mg1/mg1-5090-2/RESULTS-mg1.md)) put seven families through tp1's arm driver and verdict on one RTX 5090 behind the train anchor**, on the moe-generalize code (#1048: the training stack's glue matched by structure, not by Qwen names). Every family PASSES. LFM2, Granite-4.0-H, ERNIE-4.5 and Nemotron-H enter on their rows above. Qwen3.6-35B-A3B passes too, and entered once MG1 amendment 3 read P2 on tp1's own fused arm (`e4b.train.parity.mg1.qwen3_5.p2.2026-10-05`). The two regression anchors re-read clean: OLMoE (`e4b.train.parity.mg1.olmoe.fused.2026-10-04`) and Gemma-4 (`e4b.train.parity.mg1.gemma4.fused.2026-10-04`), whose fused RMSNorm numerics #1048 changed.

Each cell is one of `supported` (completed under the registered protocol with a PASS/OK receipt), `refused` (with the reason), `void` (ran, unreadable), `harness_error`, `not_tested`, `experimental`, `n/a` — per path, never a flat flag; the machine-readable form, with the claim id behind every `supported` / `void` / `refused` cell, is `training_support` in [`capabilities.json`](capabilities.json), validated by `scripts/check_capabilities.py`, and `model_families` is exactly the families whose `fast_train` is `supported`.

**The field-recipe position, at matched work with the comparator's grouped
path engaged** (lane TC1, 2026-10-02, three rented RTX 5090 hosts,
[`bench/h2h-2026-10-02/tc1/`](../bench/h2h-2026-10-02/tc1/README.md); register
`e4b.train.h2h.unsloth.qwen3.5090.2026-10-02` with its `.quality-n20`,
`.e4b-internal-parity`, `.secondary-mb1`, `.curve-n200` and labelled rows). On
**Qwen3-30B-A3B at the Unsloth notebooks' own recipe** — alpaca, seq 2048,
micro-batch 2 × accum 4, r 16, AdamW-8bit — with the work matched (the same
642,514,944 parameters, one per-slot LoRA init and fp32 adapters in both
frameworks, the same tokens) and Unsloth 2026.9.14 on torch 2.12.1+cu130
with its `grouped_mm` backend engaged on every step, **e4b takes 5.688 s/step
against Unsloth's 8.171: a ratio of 1.437 [1.434, 1.440]** over two draws
each, at 265 vs 176 tokens/s. Quoted beside it, in Unsloth's favour: peak VRAM
3.57 GB lower and energy per step ×0.72 at that configuration (the fp32
adapters are the matched set's choice; e4b as shipped matches Unsloth's
footprint). Held-out loss is EQUIVALENT at N = 20 and the 200-step curves are
EQUIVALENT at every eval (`.curve-n200`, largest paired |Δ| 0.0020); e4b's
parity control PASSES on the same box at **0.00041 nats**. **This supersedes
the 2026-09-19 position (4.490)**: that Unsloth arm ran on torch 2.8, where
`torch._grouped_mm` is sm_90-only and Unsloth's loader silently takes its
per-expert loop — the same loop still reads 6.565 × e4b on a torch-2.8 install
(`.loop-fallback-t28`). Both steps are host-launch-bound on this card
(device-busy 0.48 e4b vs 0.23 Unsloth; the fused path records 4.3 × fewer
device events and 9.9 × fewer CPU-side ops per step), and absolute s/step does not travel between hosts (the same e4b
arm: 3.2 / 5.7 / 9.3 s on three hosts, `.host-variance`) while the within-box
ratio does. **axolotl 0.20.0 trains it too** (`e4b.train.h2h.axolotl.qwen3.5090.2026-10-02`,
re-run once the harness stopped breaking its fp32 routers, TC1 amendment 4):
**axolotl/e4b 1.416 [1.398, 1.435]** at the same matched work, e4b faster per
step, axolotl 0.93 GB lower at peak and ×2.31 the energy; its scattermoe
native-best arm steps 7 % faster than e4b's matched path (0.929, one draw,
its own init — `.scattermoe-native`, a labelled row). **After amendments 10-15 both positions moved** (TC1 amendment 19, e4b with every default,
no environment): **Unsloth/e4b 1.997 [1.980, 2.014]** (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-03`; e4b 3.973 s/step vs 7.933, the matched set
inside the draw noise, Unsloth still 2.95 GB lower at peak) and **axolotl/e4b 2.775 [2.735, 2.814]** (`e4b.train.h2h.axolotl.qwen3.5090.2026-10-03`; 1.979 on
the matched-set box's host -- e4b's host-launch-bound step varies more by host than axolotl's, so the ratio does too).
**With both frameworks on one stack** (torch 2.12.1+cu130 / transformers 5.5.0, TC1 amendments 25, 29 and 33, load-gated
60-step draws) **Unsloth/e4b reads 2.352 [2.348, 2.356]** (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.same-stack`; e4b 3.494 s/step vs
8.218, held-out at N=60 COMPARABLE, Unsloth 3.22 GB lower at peak). It replicated on a second host, an EPYC 7K62, at **2.468**
(`e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.same-stack-host2`, TC1 amendment 42, the current code). That is the Qwen3-30B-A3B position to quote; 1.997 is the reading
with e4b in the field image's environment (torch 2.8.0+cu128 / transformers 5.18.0) and Unsloth in its own.
**Those positions read short rows.** The field recipe's Alpaca rows carry about 1,000–1,400 real tokens per step. On packed rows of
4,096 real tokens (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.packed-4k`, TC1 amendment 39) e4b at its defaults runs out of memory at
step 1, allocating the fp32 copy of the full-vocabulary logits (2.32 GiB), while Unsloth trains the same rows at 24.86 GB: an e4b loss
in that regime.
With the opt-in chunked LM loss (`E4B_CHUNKED_LM_LOSS=1`, TC1 amendment 40) e4b trains those rows at 32.5 GB, but that box reads
UNTESTED: grouped-nf4-gemm's `auto` route took its per-expert LoRA loop for ~1.5 % of delta calls, which TC1's field-recipe rule VOIDs.
Read again with that loop as a recorded route (TC1 amendment 43, another host), e4b with the flag steps those rows at 11.20 s against
Unsloth's 14.32 on one stack: Unsloth/e4b **1.278** [1.271, 1.284] (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.packed-4k-chunked`, a
LABELLED position: the flag is opt-in), with e4b at 32.5 GB against Unsloth's 24.86. Read with caveats: three of the six standing attempts ran above the 6.0 load gate on a shared host, both of e4b's quoted draws among them (machine 145701; standing load1 e4b 8.0 / 52.2, Unsloth 4.7 / 18.2, venv-e4b 5.7 / 3.6; the draws stayed within 0.3 % and 0.7 %), and amendment 43's bands were set with amendment 40's 1.43 in view, so this is a replication, not a blind test.
At the field recipe the same flag costs e4b's shipped arm 4.9 % of its step (1.049 [1.017, 1.082]) for 1.17 GB of peak
(`e4b.train.chunked-lm-loss.default-decision.5090.2026-10-05`, TC1 amendment 41), so it stays opt-in. `E4B_CHUNKED_LM_LOSS=auto` chunks only where a
forward's fp32 logits would reach 1 GiB. At the field recipe its gate never fired and the step ran 0.992 (shipped) / 0.999 (matched) of
the stock loss's (`e4b.train.chunked-lm-loss.auto.default-decision.5090.2026-10-05`, TC1 amendment 44), so `auto` becomes e4b's default. That box ran every attempt above its 6.0 load gate on a shared host, so the decision stands but the 4.9 % is not a clean magnitude.
The 2026-10-02 figures above stand for the code before #945. **Native-best against
native-best on one box** (`.native-vs-native`, TC1 amendments 5-7): Unsloth's
native-best / e4b as shipped **1.794 [1.790, 1.797]**, e4b faster per step.
axolotl's scattermoe draws on that box were 10.8 % apart, with large warm-up
steps inside the 20-step window, so its ratio is not quoted, P13 is UNTESTED,
and no "fastest" statement is made. **At steady state axolotl's scattermoe is
faster than e4b as shipped** (`e4b.train.h2h.axolotl.qwen3.5090.2026-10-02.native-steady-state`,
amendments 8 and 9, two hosts): over steps 101..200 of a 200-step run, axolotl /
e4b shipped is **0.911 [0.902, 0.921]** on a Xeon E5-2698 v4 host and **0.901
[0.892, 0.910]** on a Ryzen 9 3900X host, axolotl 9-10 % faster per step on both.
Over the whole run e4b finishes first on both (summed step time 1,353-1,372 s
against axolotl's 1,934-2,023 s, and 743-752 s against 1,219-1,231 s, axolotl's
warm-up included), and axolotl spends ×1.32-1.39 the energy per step. axolotl's native configuration also reaches the matched
held-out curve at step 200, while e4b as shipped sits 0.024-0.027 above it.
**e4b's own step got faster afterwards** (`e4b.train.host-syncs.qwen3.5090.2026-10-03`,
TC1 amendment 10, #945): a fused MoE training layer pass made 13 host syncs, and
#946 plus grouped-nf4-gemm's pinned index ring take it to 1. On one 5090 that steps
the field recipe at **0.866** (shipped) and **0.847** (matched) of the legacy path,
with unchanged held-out loss and peak VRAM. The positions above predate it and stand
as measured; none is restated from an A/B of e4b against itself.
**And again** (`e4b.train.lora-delta-lean.qwen3.5090.2026-10-03`, TC1 amendment 13): grouped-nf4-gemm's padded LoRA delta
trimmed exactly (#440: no zero fill, no scalar pass at scaling 1, a sort-free gather) steps the field
recipe at **0.939** (shipped) and **0.911** (matched) of the previous body on one 5090, held-out loss
unchanged.
**And the grouped GEMM's tile** (`e4b.train.prefill-tile-rule.qwen3.5090.2026-10-03`, TC1 amendment 14): sizing its M-tile from the
actual group sizes instead of the largest group (grouped-nf4-gemm#441) steps it at **0.924** (shipped) and
**0.968** (matched) of before on one 5090, outputs identical.
**And the norms** (`e4b.train.fused-rmsnorm.qwen3.5090.2026-10-03`, TC1 amendment 15): fusing the frozen RMSNorms into one launch each way
(#961, near-exact) steps it at **0.924** (shipped) and **0.959** (matched) of the composite on one 5090, held-out
within 0.0021.
**And the host reuse** (`e4b.train.host-reuse.qwen3.5090.2026-10-04`, TC1 amendment 20): grouped-nf4-gemm reusing repeated index
uploads and the LoRA plan inside each MoE layer pass (#444, values identical) steps it at **0.933** (shipped) and **0.951**
(matched) of before on one 5090, held-out unchanged; on by default since grouped-nf4-gemm#446.
**And keeping MoE activations** (`e4b.train.moe-keep.qwen3.5090.2026-10-04`, TC1 amendment 21): checkpointing attention only in the
last n layers instead of whole layers (`E4B_MOE_KEEP_LAYERS` with grouped-nf4-gemm's compact delta, gradients identical) steps it at
**0.835** with 32 of 48 layers kept (shipped, +4.5 GB) and **0.926** with 16 kept (matched, +2.3 GB) on one 5090 — opt-in, the
recommended setting where that headroom exists (`docs/CHOOSING.md`); the cross-framework positions are quoted with the defaults.
**With all of that, the steady-state ordering flipped** (`e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state`, TC1 amendments 16 and 18, two hosts):
over steps 101..200 of the same 200-step run, axolotl's scattermoe / e4b as shipped is **1.238 [1.231, 1.246]**
on an EPYC 7663 host and **1.146 [1.129, 1.164]** on an EPYC 7C13 host -- **e4b as shipped now steps 15-24 % faster at steady state**, finishes 200 steps in about
half the summed step time (650-654 s against 1,373-1,389 s), and axolotl spends x1.79 the energy per step. axolotl
still reaches the matched held-out curve while e4b as shipped sits 0.024-0.027 above it. The 2026-10-02 rows above
stand as measured for the code before #945.
**On an H100 NVL at default settings e4b is now faster per step** (lane TC1c, the same matched set, one rented box per reading,
[`bench/h2h-2026-10-02/tc1c/`](../bench/h2h-2026-10-02/tc1c/README.md)). On e4b 0.45.0 with grouped-nf4-gemm 0.37.0 and nothing set,
`auto` takes the grouped_mm route, and Unsloth takes 2.544 s/step against e4b's 2.397: **Unsloth/e4b 1.061 [1.047, 1.075]**
(`e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0`, TC1c amendment 8). That is at 2.93 GB more peak VRAM on e4b (27.20 vs 24.27 GB)
and ×1.23 Unsloth's energy per step, with the matched set EQUIVALENT. Before the route, with e4b after TC1 amendments 10–15
(`e4b.train.h2h.unsloth.qwen3.h100.2026-10-04`, TC1c amendment 1), Unsloth took 2.571 s/step against e4b's 3.146 —
**Unsloth/e4b 0.817 [0.799, 0.836]**, Unsloth faster by 1.22 ×, at 2.98 GB less peak VRAM and ×0.79 the energy, with the matched set
EQUIVALENT. e4b's own step fell ×0.768 since the first H100 box; on this card it is now mostly device-bound (device-busy 0.665), and
it spends ~2.4 × Unsloth's device time per step. **With e4b keeping all 48 layers' MoE activations** (an opt-in memory-for-time setting,
`E4B_MOE_KEEP_LAYERS=all` with the compact delta; TC1c amendment 2, a LABELLED row `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.moe-keep`,
same machine) **e4b is faster per step on the H100 too: Unsloth/e4b 1.100 [1.088, 1.111]** (2.577 against 2.343 s), at 9.81 GB more
peak VRAM (34.08 vs 24.27 GB) and about equal energy — quoted beside the default-settings 0.817, never in place of it. grouped-nf4-gemm's opt-in grouped_mm route made e4b SLOWER on this card as shipped (TC1c amendment 4: 0.664 alone, 0.934 with the activations kept; labelled rows `.route` / `.moe-keep-route`): its dequant kernel ran ~4x slower than the bitsandbytes dequant the kernel replay had timed. With the dequant at bandwidth (grouped-nf4-gemm #452; TC1c amendment 6) the route makes e4b faster per step than Unsloth on the H100: **1.030 [1.016, 1.045] alone and 1.325 [1.296, 1.356] with the activations kept** (labelled rows `.route-v2` / `.moe-keep-route-v2`, matched set EQUIVALENT), and it is the sm_90 default from grouped-nf4-gemm 0.37.0 (grouped-nf4-gemm#454); the default-settings re-read on 0.45.0 (TC1c amendment 8, above) is the H100 position of record. The first H100 reading, e4b before #945
(`e4b.train.h2h.unsloth.qwen3.h100.2026-10-02`), was 0.621 [0.615, 0.628]: Unsloth 2.546
s/step against e4b's 4.097, faster by 1.61 × at 3.59 GB less peak VRAM and ×0.63 the energy, with the
same loss (EQUIVALENT). The profiles then said why (`.dispatch-profile`): e4b
issues ~139 k device events per step on both cards, Unsloth 612 k on the
5090 and 82 k on the H100 — the grouped GEMM its path routes through is one
launch per call on Hopper and not on Blackwell. The 1.437 is therefore a
5090 position; by TC1c's registered decision rule no "e4b faster" position
is quoted for the H100 class. The clinical-fixture positions (1.413 p38, 1.457 tp2/P40) are
dated measurements on earlier cuts and do not reproduce on the current ones
(`.anchor-p38-fixture`: 2.428 with Unsloth's grouped path, 5.147 on its loop).
tp4's other-family rows stand as measured on the 2026-09-19 cut — Granite
2.853 → 2.366 (×1.21) and OLMoE 2.739 → 1.395 (×1.96), both with parity
passing; **nothing is quoted against Unsloth on Granite or Qwen3.6** (its arm
trains the attention only there, VOID by tp4's regime rule) **nor on OLMoE**
(`e4b.train.h2h.unsloth.coverage.5090.2026-09-19`) — lane TC2 re-asked
them on the current cuts; the next paragraph is what it found.
**And grouped-nf4-gemm's dense route** (`e4b.train.dense-route.*.5090.2026-10-04`, TC1 amendment 22): one expert dequantized at a
time into `torch.mm`. Against the fused kernels it reads 0.651 on Mixtral-8x7B (few large experts) and 2.947 on Qwen3-30B-A3B
(many experts, a launch-bound step), with held-out unchanged. Off sm_90, grouped-nf4-gemm's `auto` takes it for calls with at most
16 present groups.
**Where the memory goes** (`e4b.train.memory-census.qwen3.5090.2026-10-04`, TC1 amendment 23): a census of the CUDA allocator
on Qwen3-30B-A3B at micro-batch 1 finds every static class byte-for-byte the same in e4b and Unsloth except the expert absmax.
e4b keeps it in fp32 by default (1.81 GB); `E4B_ABSMAX_DQ=1` stores it in 0.46 GB, as Unsloth does. With that switch e4b peaks
0.43 GB above Unsloth (24.68 vs 24.24 GB), all of it transient, mostly grouped-nf4-gemm's padded LoRA delta; at e4b's defaults
the gap is 1.78 GB. grouped-nf4-gemm's opt-in compact delta (`NF4_QLORA_COMPACT_DELTA=1`) was meant to shrink that transient; on one
stack it did not (`e4b.train.compact-delta.qwen3.5090.2026-10-05`, TC1 amendment 36): the matched peak rose 0.23 GB, while the step ran
0.969 (matched) and 0.970 (shipped) of the default's. It stays opt-in. With grouped-nf4-gemm#473's backward (each intermediate released
at its last use) a second host read the matched peak 0.288 GB lower and the step 0.967 (matched) / 0.948 (shipped)
(`e4b.train.compact-delta.free.qwen3.5090.2026-10-05`, TC1 amendment 37); the shipped ratio fell below its registered band, so it stays
opt-in until a registration whose bands allow that reading. A third, faster host read it slower (1.016 / 1.013 on Qwen3, 1.013 on Mixtral;
`e4b.train.compact-delta.default-decision.5090.2026-10-05`, TC1 amendment 38), so it stays opt-in: it trades host work for device work,
and wins only where the step is host-bound. Its peak saving held (−0.31 GB on Qwen3's fp32 arm).
**The double-quantized absmax, as an A/B** (`e4b.train.absmax-dq.mixtral.5090.2026-10-05`, TC1 amendment 28): on Mixtral-8x7B
`E4B_ABSMAX_DQ=1` costs 2.3 % of the step (1.023) for 2.04 GB of peak (31.07 → 29.03 GB), held-out within 0.003. Qwen3-30B-A3B's pair
was unstable on a busy host; amendment 31 read it over 60 steps on a quiet one
(`e4b.train.absmax-dq.qwen3.5090.2026-10-05`): 1.014 for 1.34 GB, held-out within 0.003. All three predictions held, so the
double-quantized absmax becomes the default for resident training (`E4B_ABSMAX_DQ=0` turns it off).
**Prebound Triton launches, as an A/B** (`e4b.train.prebind.qwen3.5090.2026-10-05`, TC1 amendments 26 and 30): with both prebind
flags on, Qwen3-30B-A3B's training step is 0.973 of the flags-off step on the matched arm and 0.980 [0.957, 1.003] on the shipped arm
(60 steps), with held-out within 0.002 and bit-identical kernels. All three predictions held, so `E4B_TRITON_PREBIND` and
grouped-nf4-gemm's `GNF4_TRITON_PREBIND` are on by default (`=0` turns each off). The gain is small, and the shipped interval reaches
1.0. They cover triton 3.7 too (`e4b.train.prebind.triton37.qwen3.5090.2026-10-05`, TC1 amendment 35, torch 2.12.1 / triton 3.7.1):
0.986 [0.975, 0.997] on the matched arm and 0.996 [0.978, 1.013] on the shipped arm, held-out within 0.003.
**The environment** (`e4b.train.env-ab.qwen3.5090.2026-10-04`, TC1 amendment 24): every 5090 position before amendment 33 ran e4b on the
field image's torch 2.8.0+cu128 / transformers 5.18.0 and Unsloth on torch 2.12.1+cu130 / transformers 5.5.0. In Unsloth's environment
e4b's matched arm steps 0.882× as long. The cause is not the padded LoRA delta's fp32 `bmm`, whose host cost per new shape is the same
on both torch versions (`e4b.train.bmm-host-replay.5090.2026-10-04`). The same-stack box (TC1 amendment 33) replicated the gain on
another host: 0.900, with e4b's prebound launches engaged on the field-image side only.
Nor is it triton (`e4b.train.triton37.qwen3.5090.2026-10-05`, TC1 amendment 32): triton 3.7.1 alone in the field image reads 0.992 on
the matched arm and 0.971 on the shipped arm. **It is torch 2.12's** (`e4b.train.env-split.qwen3.5090.2026-10-05`, TC1 amendment 34):
torch 2.12.1 + triton 3.7.1 over torch 2.8 + triton 3.4 reads 0.905 at transformers 5.5, and transformers 5.5 over 5.18 on torch 2.8 reads
1.005.

**The other families at matched work** (lane TC2, 2026-10-02, two rented
RTX 5090 hosts, [`bench/h2h-2026-10-02/tc2/`](../bench/h2h-2026-10-02/tc2/README.md);
register `e4b.train.h2h.hf.granite.5090.2026-10-02`,
`e4b.train.h2h.axolotl.granite.5090.2026-10-02`,
`e4b.train.h2h.unsloth.olmoe.5090.2026-10-02`,
`e4b.train.h2h.hf.olmoe.5090.2026-10-02` and their companions). **Re-read on
2026-10-04 on the current code, e4b is faster than each of them** (TC2 amendment 7,
`…granite.5090.2026-10-04`, `…olmoe.5090.2026-10-04`): HF/e4b **1.299** and axolotl/e4b
**1.150** on Granite, Unsloth/e4b **1.821** on OLMoE, with parity PASS and the pairs reading as
before. The 2026-10-02 readings that follow were e4b 0.38.1's: **then the 5090
per-step edge did not generalise to small experts.** On Granite-3.1-3B-A800M
HF + PEFT with bf16 experts reads **HF/e4b 0.971 [0.965, 0.978]**, faster per
step than e4b's fused 4-bit experts, which train at 0.53 × HF's VRAM (4.54
vs 8.50 GB) — a footprint position against HF. **axolotl's 4-bit path beats
e4b's on Granite** at 0.903 [0.901, 0.905] and a lower peak (4.08 GB): its
`quantize_moe_experts` packs all 64 expert stacks as NF4 (corrected
2026-10-02 — first read as bf16 because the census does not see parametrized
storage). TC2 P1 FALSIFIED. On OLMoE-1B-7B, Unsloth
2026.9.14 with its grouped path engaged reads **Unsloth/e4b 1.201 [1.199,
1.204]** (e4b faster per step) at 2.21 GB less peak VRAM and ×0.71 the
energy in Unsloth's favour, with the same loss — the first Unsloth position
on this family; HF reads 1.255 [1.253, 1.258] at twice e4b's VRAM (P2's
HF band FALSIFIED). Unsloth still adapts no expert parameter of Granite on
either target list (`e4b.train.h2h.unsloth.granite.5090.2026-10-02.coverage`),
and gpt-oss-20b has no common adapter set across frameworks — e4b trains its
attention only there, an e4b limitation beside the wins
(`e4b.train.h2h.unsloth.gptoss.5090.2026-10-02.no-common-set`). e4b's parity
control PASSES on every family with a reference (P6 HELD). Box B (Qwen3.6-35B-A3B,
Mixtral-8x7B) was lost with its instance and re-run under amendments 2 and 3.
**On Qwen3.6-35B-A3B at matched work, Unsloth trains the set resident on
32 GB and e4b does not** (`e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02`):
the matched 926,187,520 fp32 adapters over 20,520 slots OOM e4b's fused
path at both micro-batches and in its reference loop (32.5 GB peak on a
31.4 GiB card), while Unsloth, given the family's own expert names, trains
them at 30.47 GB and 10.59 s/step; e4b fits only as shipped (bf16 adapters,
32.48 GB, 5.09 s/step — a labelled row, not a comparison). That loss is
**resident only: under expert offload e4b trains the matched set at 19.35 GB**
(`.e4b-offload`, 12.4 / 13.8 s/step, parity PASS). No ratio is quoted: e4b's
offload draws are 11 % apart, and the frameworks do not start from the same
base — e4b keeps the 30 linear-attention layers' projections and the 40
shared experts in bf16 (about 1.14 B parameters) where Unsloth stores them in
4-bit, worth about 1.6 GB of the resident footprint and a 0.05-nat step-0
gap that VOIDs the same-box pair by rule. Re-asked resident on 2026-10-04 on the current code
(TC2 amendment 6, `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-04`): step 1 now completes, but e4b
still OOMs at step 2 at both micro-batches. **At micro-batch 1, on the comparator's bytes,
e4b trains it resident and steps 2.05× faster than Unsloth**
(`e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-04.mb1-dq-frozen4`, TC2 amendment 8, labelled):
with the absmax double-quantized and the non-routed projections in NF4, Unsloth/e4b
**2.049** [2.028, 2.070], 18.95 against 9.25 s/step, at 0.94 GB more peak (31.35 vs
30.41 GB), the pair COMPARABLE. e4b's defaults keep those projections bf16 and do not fit.
**On Mixtral-8x7B with both frameworks on one stack, e4b is faster: Unsloth/e4b 1.144 [1.140, 1.148]**
(`e4b.train.h2h.unsloth.mixtral.5090.2026-10-05.same-stack`, TC2 amendment 9; torch 2.12.1 / transformers 5.5.0, resident, e4b at
defaults on the dense route, an EPYC 7B13 host): e4b 3.24 s/step against Unsloth's 3.71, COMPARABLE, at 2.07 GB more peak (the fp32
absmax) and ×1.07 the energy per step. That is the position to quote. In the same box e4b on the field image's stack reads 1.013, so the
earlier loss below was mostly the host and the stack (e4b's code also moved between the two boxes). **With e4b on the field image's stack, on a Core Ultra 9 285K, Unsloth was faster,
0.836** (`e4b.train.h2h.unsloth.mixtral.5090.2026-10-04.dense-default`, TC2 amendment 8): with
grouped-nf4-gemm's dense route under `auto` (gnf4#463, run at `bb56b42`, a main commit whose version still read
0.38.0; it ships in 0.39.0), which `auto` takes for Mixtral's calls, e4b steps in
3.57 s against Unsloth's 2.98 s, at 1.95 GB more peak (31.1 vs 29.1 GB), the pair
COMPARABLE. The host, a Core Ultra 9 285K, is the Unsloth-favouring end of the range seen.
Amendment 6's 0.697 on the fused kernels (`e4b.train.h2h.unsloth.mixtral.5090.2026-10-04`) is superseded by it.
The offload footprint row below stays as the offload lever's reading.
**On Mixtral-8x7B, re-measured with the counters fixed**
(`e4b.train.footprint.unsloth.mixtral.5090.2026-10-02`): Unsloth resident steps
in 3.74 s at 29.1 GB after one 34-second compile, e4b under expert offload in
19.8 / 21.2 s at 7.16 GB — a footprint row, e4b's peak ×4.07 lower and
Unsloth about 5.3× faster per step. No position: e4b's offload draws are 7 %
apart. The 2026-10-02 Mixtral rows that read Unsloth at 6.2 / 7.0 s/step after
a 76–89-minute compile are retired: that was this harness's counters
recompiling Unsloth's compiled MoE block on every call (TC2 amendment 5).


**Against Unsloth, end-to-end, on one identical training problem** (lane
p38, 2026-09-05, one rented RTX 5090, box 49975389; **measured** — receipt
[`bench/h2h-20260905/p38/`](../bench/h2h-20260905/p38/README.md), table in
its [`RESULTS-p38.md`](../bench/h2h-20260905/p38/RESULTS-p38.md), the
pre-registration verbatim as its `PREREG.md`; register
`e4b.train.h2h.unsloth.qwen3.5090.2026-09-05` with its `.quality-n60`,
`.curve-n200` and `.e4b-internal-parity` rows and one row per arm, which
carry the numbers and the pins). Qwen3-30B-A3B on the registered `clinical`
fixture, r 8 / α 16 on attention and every expert, 321,257,472 trainable
parameters asserted and everything but the framework held equal —
experts4bit-qlora 0.35.0 + grouped-nf4-gemm 0.30.0 (the fused `dgrad` path
with NF4 attention) against Unsloth 2026.9.2 (its 4-bit MoE path; the two
stacks' transformers/peft versions differ and are recorded). **At 60 steps: s/step ratio Unsloth/e4b
1.413** (e4b faster per step at this workload, at a lower peak and lower
energy), held-out loss comparable (0.2923 vs 0.2975). **At 200 steps
the curves separate in Unsloth's favour: 0.2713 vs 0.2881** — e4b's
flattens near 0.29 from step 60 while Unsloth's keeps falling. That row is
quoted beside the position wherever the position is quoted; its causes —
the eval schedule, the checkpointing mode, the two stacks' transformers /
peft versions, the expert adapter's precision (bf16 on this side, because
the loader passes the model dtype to `ExpertsLoRA`; fp32 on Unsloth's) —
are candidates, not established. e4b's own fused-vs-reference pair passes
its band on that box (informational; tp1 owns the licence). The pre-registration predicted the opposite sign at
this workload and said the finding ships either way; it does. One workload
(≈86 tokens per step, batch 1, resident), one box, one family: no general
speed claim, nothing licensed, and the 2026-08-26 "1.17× ahead" memory
(never a claim) is disqualified as a comparison.

**tp2 / P40 (2026-09-06): the Unsloth head-to-head now covers all six
families, one box, one fixture** (one rented RTX 5090, Vast box 50005568 on
a Ryzen 7 5700X3D host, train-anchor class `pcie-full/launch-fast`;
**measured** — receipt [`bench/h2h-20260906/tp2/`](../bench/h2h-20260906/tp2/README.md),
reducer table in its [`RESULTS-tp2.md`](../bench/h2h-20260906/tp2/RESULTS-tp2.md),
the pre-registration verbatim as its `P40-PREREG.md`; register
`e4b.train.h2h.unsloth.<family>.5090.2026-09-06` — one row per attempt under
`….arm.<framework>.<arm>`, plus position, `.quality-n60`, `.footprint`,
`.coverage` and `.e4b-internal-parity` rows, which carry the numbers).
P38's fixture exactly, tokenised once per family; e4b at the cut a user
installs today — 0.35.1 + grouped-nf4-gemm 0.30.2 from PyPI, NF4 attention
via `TRAIN_ATTN_4BIT` — against Unsloth 2026.9.2 in its own venv.
**Positions exist on two families; on the other four the statuses are the
result.** Qwen3-30B-A3B: s/step ratio Unsloth/e4b **1.457** (e4b faster per
step), held-out COMPARABLE (`e4b.train.h2h.unsloth.qwen3.5090.2026-09-06`,
`….quality-n60`) — the cross-lane anchor: **+3.1% from P38's 1.413, inside
the pre-registered ±10%** (P2 held; P38's 200-step curve row, in Unsloth's
favour, still stands beside any position of this family). Mixtral-8x7B:
**the footprint trade leads the row** — e4b trained its experts under CPU
offload at a **3.223 GB** peak (the registered design, tp1's `offload=True`;
its own row `e4b.train.h2h.unsloth.mixtral.5090.2026-09-06.footprint`) while
Unsloth ran resident (its only mode) at **29.163 GB**, and what that VRAM buys
it is speed per step: ratio **0.361** — a footprint-vs-speed trade under the
registered design, not a kernel deficit; e4b's energy is lower; held-out
COMPARABLE (`e4b.train.h2h.unsloth.mixtral.5090.2026-09-06`, `….quality-n60`);
P4 (Unsloth resident OOMs on 32 GB) **falsified**. No position on the other
four, and the coverage rows are results, not empty cells: **Granite** —
Unsloth's arm completed but is **VOID**: it attached LoRA to the attention
only (2,621,440 trainable parameters against e4b's 49,807,360), its MoE-LoRA
path never engaged on `granitemoe`
(`e4b.train.h2h.unsloth.granite.5090.2026-09-06.coverage`). **OLMoE** —
Unsloth's process died at MoE-LoRA engage, before its first receipt
(HARNESS_ERROR; `e4b.train.h2h.unsloth.olmoe.5090.2026-09-06.coverage`). On
both, e4b's arms are VALID. **gpt-oss** — three refusals, as pre-registered
(P5 held): e4b's attention-4-bit refuses on structure (96 of 96 attention
projections carry a bias), its fused path patches nothing, and Unsloth's
load fails on the MXFP4 weight conversion
(`e4b.train.h2h.unsloth.gptoss.5090.2026-09-06.arm.*`); the attention-only
secondary row trains. **Gemma-4** — both e4b attention-4-bit arms died on the
harness's projection-count check (the converter returned 100; the harness expected
4 · n_layers = 120 —
[#412](https://github.com/pjordanandrsn/experts4bit-qlora/issues/412), fixed
since by #435; the attention-4-bit arms have run cleanly since, in P56,
tp4 and P67, and are **licensed since lane P67** (2026-09-24) under the family's
measured floor, [#713](https://github.com/pjordanandrsn/experts4bit-qlora/issues/713); the
bf16-attention `fast_train` path stays as tp1 left it), while Unsloth's arm
is OK · VALID (`e4b.train.h2h.unsloth.gemma4.5090.2026-09-06.arm.unsloth.ckpt_unsloth`).
e4b's internal fused-vs-reference parity PASSES on all four families that
ran both arms (`e4b.train.h2h.unsloth.<family>.5090.2026-09-06.e4b-internal-parity`;
informational, tp1 owns the licence). `training_support` in
[`capabilities.json`](capabilities.json) now records the attention-4-bit
configuration per family from these receipts — inside the per-path
structure, never a flat flag. No gate, threshold or licence moved; nothing
here supersedes P38 (two boxes, two measurements, never averaged).

**Serving is at parity with the model's own attention on three of four
families, and not on the fourth.** Measured against a *chunk-free* reference — one full forward,
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

**Gemma-4's paged path is at parity with transformers' own perturbations** (lane P108, 2026-10-03, one rented
RTX 5090; **measured** — [`bench/p108/RESULTS-p108.md`](../bench/p108/RESULTS-p108.md),
`e4b.parity.gemma4.p108.paged-vs-floor.5090.2026-10-03`).
- **The setup:** 32 wikitext windows with the 1,024-token sliding window binding. The paged path is set against a
  floor of three arithmetically neutral transformers draws.
- **The result:** the paged path's mean NLL (5.360) lies between the floor draws' (5.298–5.415). Its bias against
  transformers' cached forward is −0.196 nats with spread 0.326, against the floor's 0.258 / 0.349. A halved
  decode scale reads +2.26.
- **What it supersedes:** the single-window statement below. It still holds at that resolution.

**Gemma-4 has no reference at this resolution.** On three 512-token
windows transformers' *own* cached forward sits as far from a one-shot
forward as the paged path does (`e4b.parity.gemma4.no-reference`).
The cause is the model, not a path: plain transformers with no e4b code
gives the same 255 tokens an NLL that moves by 0.4 nats depending only
on which tokens follow them (bf16 batch-shape variance in the expert
gathers, 0.2% at layer 1, amplified by the router to 35% of the hidden
state by layer 19; Qwen3 shows the mechanism at a tenth of the
amplitude and loses 0.001). Running the router in fp32 (plain transformers) does not remove the amplification (layer-19 divergence 0.25 against 0.34, top layer unchanged) and itself moves the same tokens' NLL by 1.1 nats, so router precision is not a lever; the sensitivity is the model's. The paged path's one localised,
measured cost is the fp8 cache and dot: 0.046 nats, concentrated on the
five 512-dim layers, 0.017 with 32-wide K groups. Method: METHODOLOGY
§13.2; numbers: SERVING-PARITY.

**Hybrid linear-attention models serve paged on the card** (lane P97, 2026-10-02, one rented RTX 5090; **measured** —
[`bench/p97/RESULTS-p97.md`](../bench/p97/RESULTS-p97.md), `e4b.serve.p97.qwen36-hybrid-paged-state.5090.2026-10-02`).
On Qwen3.6-35B-A3B (30 Gated DeltaNet + 10 attention layers), the paged runner keeps each sequence's linear state at
transformers' own (7.7e-3 relative error where both paths see the same tokens) and tracks transformers' forward at
4.43e-3 nats, argmax agreement 0.972, through gnf4's fp8 kernel on a 10-layer pool. A slot-mapping mutant reads 4.05
nats. Decode was eager (graphs were then refused for hybrids; P101 below reads them), so this is a correctness
reading, not a speed one.

**Hybrid models decode under CUDA graphs** (lane P101, 2026-10-03, one rented RTX 5090; **measured** —
[`bench/p101/RESULTS-p101.md`](../bench/p101/RESULTS-p101.md), `e4b.serve.p101.qwen36-hybrid-decode-graphs.5090.2026-10-03`).
- **The reading:** on the fixed code (#918, #913), Qwen3.6-35B-A3B through `build_engine` captured every bucketed decode
  graph (1–16) and replayed them with no eager step, with tokens equal to the padded eager step's on all 17 requests.
- **The speed:** 449.0 tok/s on 16 staggered requests and 79.7 on one, 2.02x and 3.28x plain eager, on a Ryzen 9 7950X
  host.
- **The host matters:** eager hybrid decode is launch-bound, and P98's EPYC 7663 host ran the same eager arms at about
  half the speed. The ratio is not claimed for another host.
- **`E4B_PAGED_GRAPHS=1` now serves hybrid models.** The Gated DeltaNet layers still ran transformers' torch path.

**`serve_paged` keeps eager decode by default** (lane P109, 2026-10-03, one rented RTX 5090; **measured** —
[`bench/p109/RESULTS-p109.md`](../bench/p109/RESULTS-p109.md), `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03`).
- **Speed:** on the default server (Qwen3-30B-A3B NF4), graphs ran ×5.60 the eager default at 16 concurrent requests and
  ×9.02 at one, on an EPYC 7C13 host.
- **Exactness:** the replay is bit-identical to its padded eager step.
- **Verdict DIVERGENT:** 9 of 16 rows agree with the eager default for 16 tokens, against a bar of 12. Grouping alone,
  with no graphs, reads 10 of 16. The default waited on a teacher-forced quality reading: P110, below.

**The graph arithmetic costs no measurable quality** (lane P110, 2026-10-03, one rented RTX 5090; **measured** —
[`bench/p110/RESULTS-p110.md`](../bench/p110/RESULTS-p110.md), `e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.2026-10-03`).
- **Quality:** on the default server (Qwen3-30B-A3B NF4), teacher-forced over 48 wikitext windows, device grouping plus
  bucket padding (bitwise the graph path) reads +0.0004 nats against the eager default. The eager default's own
  half-batch and prefill-split draws read +0.0018 and +0.0005.
- **Verdict AT_PARITY:** a halved decode scale reads +1.05. Graphs are now `serve_paged`'s default
  (`E4B_PAGED_GRAPHS=auto`; `0` keeps eager decode).

**One KV-table selection per decode step** (lane P111, 2026-10-04, one rented RTX 5090; **measured** —
[`bench/p111/RESULTS-p111.md`](../bench/p111/RESULTS-p111.md), `e4b.serve.p111.kv-step-select.qwen3.5090.2026-10-04`).
- **Speed:** on the default graph server (Qwen3-30B-A3B NF4), it runs 1.0352× / 1.0096× the per-layer selection
  (16 requests / one; min over two pairs) with identical tokens. The 16-row step falls 20.72 → 19.99 ms, SC1b's census
  of per-layer KV-table glue.
- **Default:** on (`E4B_KV_STEP_SELECT`; `0` keeps the per-layer form).

**Programmatic dependent launch, capped to small launches** (lane P113, 2026-10-04, one rented RTX 5090; **measured** —
[`bench/p113/RESULTS-p113.md`](../bench/p113/RESULTS-p113.md), `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04`).
- **Speed:** on SC1's int4 serving configuration (Qwen3-30B-A3B), grouped-nf4-gemm's `GNF4_PDL=1` with
  `GNF4_PDL_MAX_ROWS=8` runs 1.0404× / 1.0000× the switch off (one request / 16; min over two pairs,
  decode-only timing) with identical tokens. Uncapped it runs 1.0401× / 0.9787×: PDL helps the B=1 step
  (4.38 → 4.21 ms) and costs the 16-row one, which the cap leaves alone.
- **Default:** on, capped at 8, from grouped-nf4-gemm 0.37.0 (`GNF4_PDL=0` turns it off). P112, the
  first served read, closed VOID twice (`bench/p112/RESULTS-p112.md`).

**The Gated DeltaNet kernels speed hybrid decode** (lane P105, 2026-10-03, one rented RTX 5090; **measured** —
[`bench/p105/RESULTS-p105.md`](../bench/p105/RESULTS-p105.md), `e4b.serve.p105.qwen36-gdn-kernels.5090.2026-10-03`).
- **Speed:** with `flash-linear-attention` 0.5.2 and `causal-conv1d` 1.7.0 installed, Qwen3.6-35B-A3B's bucketed decode
  graphs ran 1.135× (W16) and 1.118× (W1) the torch path's on the same host, 536.5 and 100.7 tok/s.
- **Exactness:** every bucket still replays exactly as the padded eager step.
- **Correctness:** a dense hybrid stays within 2× its all-attention control on every seed, on the real fp8 kernel.
- **Caveats:** token streams differ from the torch path's (the kernels' arithmetic). Quality in nats: lane P106, below.

**The Gated DeltaNet kernels cost no measurable quality** (lane P106, 2026-10-03, one rented RTX 5090; **measured** —
[`bench/p106/RESULTS-p106.md`](../bench/p106/RESULTS-p106.md), `e4b.serve.p106.qwen36-gdn-kernel-quality.5090.2026-10-03`).
- **How:** transformers' Gated DeltaNet functions were switched in place inside one process, proven bit-exact against
  a kernel-free process on the card.
- **Quality:** against transformers' torch path on Qwen3.6-35B-A3B, teacher-forced on wikitext:
  - KL 5.7e-3 nats on prompt positions and 4.9e-3 on decode steps;
  - argmax agreement 0.969 and 0.973;
  - d_nll within ±0.001 nats;
  - NEUTRAL by the registered rule.
- **Prefill:** TTFT 1.10–1.12× the torch path's at 512–4,096 tokens.
- **Earlier lanes:** P103 and P104 stopped on a single-seed tiny-MoE premise, which proved to be a seed lottery on
  every kernel set.

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
the same instrument reads the other four families at 0.02–0.1. It was
filed as a served-model defect
([#597](https://github.com/pjordanandrsn/experts4bit-qlora/issues/597)), and
#597 closed 2026-09-19 the other way. Lanes P47–P51 (below) place the cost
in the quantised model's early expert layers, not in the serving stack:
NF4 in layer 0 alone costs 0.892 of the 1.08 nats. No Gemma-4 serving
position is quoted. The per-expert error census on Granite and Mixtral
(`e4b.serve.p44.census.granite-mixtral.2026-09-19`) is data for a per-expert
fallback that is not built.

Lane P65 (`bench/p65/`, 2026-09-24) asked which per-expert ranking would
choose that fallback's experts, and whether it survives a change of
calibration text. On Granite every ranking survives:
- `rel_act` 0.932 cross-text Spearman;
- Colla-Q's entropy-weighted error 0.910;
- routing frequency 0.891.

So S-C's selector there has two arms, written in
[`SPECULATIVE_LANES_ADDENDUM_4.md`](SPECULATIVE_LANES_ADDENDUM_4.md)
(`e4b.quality.p65.granite.selector-two-arms.5090.2026-09-24`). OLMoE and
Mixtral have no per-expert premise, so neither gets a selector. On both,
the entropy ranking transfers across texts better than routing frequency,
as Colla-Q reported: +0.421 and +0.479
(`e4b.quality.p65.collaq-stability.olmoe-mixtral.5090.2026-09-24`).
Routing-frequency hot sets agree across the two texts at only 0.14
(OLMoE) and 0.125 (Mixtral, chance) Jaccard. No mixed-precision cell has
run.

**Calibration does not rescue Gemma-4's experts, and the more principled
calibration is worse** (P53, [`bench/p53/RESULTS-p53.md`](../bench/p53/RESULTS-p53.md),
`e4b.quality.gemma4.calibration-refuted`): with all 30 expert layers
quantised, KL against the bf16 checkpoint reads **1.0772** NF4
round-to-nearest, **1.1050** calibrated all-at-once and **1.1564** calibrated
sequentially — calibration **order** the only difference, so the ordering
effect that licensed sequential calibration on Qwen3-30B **inverts** here
(both registered predictions refuted; the third, which declined to predict
the direction, is why the reading stands). Together with P49,
which refuted shrinking the perturbation, **both addressable axes are now
closed**: the only lever on Gemma-4's positional sensitivity remains keeping
early expert layers in high precision, and that lever is memory.

**Gemma-4's fused training path does not agree with e4b's own dense
reference, and a kernel change did not fix it.** On one box, the same tokens
and identical trainable counts, fused and dense reference end 0.08257 nats
apart on final train loss (median step-wise 0.12421) against a 0.05/0.05 band,
reproducing the 0.09037 / 0.11801 of the previous kernel cut; the fused path
is 11.65× faster on that pair (internal, no competitive position). As the
P47–P51 lanes predicted, the sensitivity is **positional and lives in the
quantised model**, not the adapter path. [`e4b.parity.gemma4.train-internal`](claims.json) is **SUPERSEDED** 2026-09-22 by lane P56
([`e4b.parity.gemma4.train-floor`](claims.json), itself superseded 2026-09-24 by lane P67, below;
[`bench/p56/RESULTS-p56.md`](../bench/p56/RESULTS-p56.md)): the disagreement is
**not attributable to the fused path** — e4b's kernel-free batched path, with
13× less composed gradient error, fails the same band, and no achievable
arithmetic change reaches 0.05. The FAIL is real and stable; P56 measures this
family's training-parity floor for the first time (**≥ 0.054 final / 0.085
median step-wise**), and `tp4_reduce.parity()` compares against 0.05 **and
against zero**, with no floor term. So the defect is the **gate**, not the fused
path — [#558](https://github.com/pjordanandrsn/experts4bit-qlora/issues/558)
closed 2026-09-20 as explained, not fixed, and the missing per-family floor was
filed as [#713](https://github.com/pjordanandrsn/experts4bit-qlora/issues/713);
the serving side already fixed the same class of error by moving to a measured
floor. That proxy floor is in turn **SUPERSEDED**
2026-09-24 by lane P67 ([`e4b.train.p67.gemma4.attn4-reorder-floor.5090.2026-09-24`](claims.json),
[`bench/p67/RESULTS-p67.md`](../bench/p67/RESULTS-p67.md)): measured as the
reference against its own reorderings (one repeat, four fixed permutations of the
per-expert loop, same session), the floor is **0.121 final / 0.128 median** — larger
than the proxy — and read against `max(0.05, 3 · F_hi)` both accelerated arms
**PASS** (fused 0.013 / 0.036, batched 0.037 / 0.048), as do the two earlier
sessions' constant-band FAILs. #558 does not reopen; the attention-4-bit paths
are supported on `gemma4_text`, as a statement of resolution: two correct
reference runs land 0.011–0.121 apart at step 20 on this model.

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
ratios are not a field-engine speedup. The 2026-09-03 numbers this
paragraph used to lead with (about 100 tok/s NF4, 204.6 with calibrated int4
attention and round-to-nearest int4 experts, about 1,238 aggregate at B=16;
`e4b.serve.b1.qwen3-30b.int4attn-calib.5090`, **measured-private**;
`e4b.serve.b16.qwen3-30b.int4.5090`, **measured**) are real, but **that
RTN-int4 class later FAILED the registered gate on its second text** (bo5,
+0.063 ppl on C4 validation, below): quote them as an unlicensed
configuration's speed on its box, never as the position.

**Against vLLM, same box, same session, identical prompt token ids — CURRENT vs
CURRENT (lane P58, 2026-09-22, Vast 52069847, an RTX 5090 on an EPYC 9655 host;
**measured** — [`bench/p58/RESULTS-p58.md`](../bench/p58/RESULTS-p58.md), register
`e4b.serve.h2h.vllm-0.30.0.p58.qwen3.b1.5090.2026-09-22` /
`e4b.serve.h2h.vllm-0.30.0.p58.qwen3.b16.5090.2026-09-22` /
`e4b.serve.h2h.vllm-0.29.0.p58.qwen3.b16.5090.2026-09-22` and one row per arm):**
vLLM **0.30.0** serving Qwen's GPTQ-Int4 checkpoint via
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
stays as measured history. **B=16 gap (P86, same box, both censused):** the experts (int4 GEMV 6.98 ms vs Marlin MoE
4.78; 2.86 of 3.32 ms), glue +0.62, our attention faster (0.96 vs 1.35)
(`e4b.serve.p86.qwen3.b16.kernel-census-vs-vllm-0.30.0.5090.2026-10-01`); K19 now takes the step to
0.905× (`e4b.serve.p88.qwen3.int4.k19-b16.5090.2026-10-01`), K23's lean glue a further 0.950×
(`e4b.serve.p89.qwen3.int4.k23-lean-glue-b16.5090.2026-10-01`).

**Against vLLM 0.28.0 (history, lane p37, 2026-09-05;** one RTX 5090 box, Vast
49975016; **measured** — [`RESULTS-p37.md`](../bench/h2h-20260905/p37/RESULTS-p37.md),
register `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05`, its `.gate` row
and one row per arm): vLLM 286.0 / 2030.0 tok/s against the NF4 control's
113.4 / 500.1 — **vLLM / e4b-NF4 2.52 at B=1 and 4.06 at B=16**, the lane's only
licence-free ratio, against the slowest configuration this package ships. No
ratio against the licensed stack exists there: the recipe's arms were VOID on
the pack fingerprint (11522/766 vs 11512/776), and amendment 3's gate on that
pack FAILED c4val1 (+0.1093 ppl). Its predecessor
(`e4b.serve.h2h.vllm.same-box`, superseded: ×1.47 / ×1.55, another box, the RTN
stack) stays true as measured.

**Per-family throughput is now measured in-repo** (2026-09-04): six
families under one protocol on one rented 5090 class, with every refused
arm named — that list became the build-out, and 0.33.0 ships it. Table
and receipt: [`SERVING-THROUGHPUT.md`](SERVING-THROUGHPUT.md).

**The 0.33.0 build-out (lane bo3, 2026-09-04; measured — [receipt](../bench/hybrid-g9/throughput-20260904/bo3/README.md),
licensed-best table in [`SERVING-THROUGHPUT.md`](SERVING-THROUGHPUT.md)) is history under the bo7 census
below:** its Gemma-4 int4 stack (121 tok/s B=1, ×1.69 over NF4) has no
quality verdict, Mixtral's int4 arms (×2.14; ×2.29 with calibrated
attention) failed their second text on bo5, gpt-oss stays NF4-only (a
uniform int4 grid cannot hold its MXFP4 experts: +0.63 nats), and Granite's
"302 tok/s, ×1.59" row is **retracted** (below); Granite's licensed NF4 stack
read 259.1 / 1689.6 tok/s there.

**The second text (2026-09-04, lane bo5; measured — receipt
[`bench/hybrid-g9/throughput-20260904/bo5/`](../bench/hybrid-g9/throughput-20260904/bo5/README.md),
table in its [`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo5/RESULTS.md)).**
The registered K8 gate is in perplexity: an uncalibrated arm |Δppl| ≤ 0.05
on every text, a calibrated pack ≤ +0.05 on every text with an improvement
claimable only when it holds with the same sign on two; nats are quoted
beside the verdict and never change it. bo5 scored C4 validation for every
calibrated pack bo3 left at one text, and **every one FAILS as registered**:
Qwen3's `all` stack +0.063 ppl (+0.0038 nats, inside the family's 0.0095-nat
floor, so noise rather than a component; the verdict is not retuned), Mixtral's `all`
+0.116, and Granite's C4-calibrated int4 experts +0.387 at 10× the floor
(that route is refused). Mixtral's *uncalibrated* int4-expert stack, licensed
in P30 and bo3 on wikitext, fails the second text (+0.058 ppl), so that
label is **withdrawn**. Licensed and re-measured there: Granite's NF4 stack
294.1 / 1736.1 tok/s; gpt-oss's MXFP4 store under the route rule 173.3
(×1.270) / 719.7 (×0.971, recovering the ×0.81 B=16 penalty), its quality
gate open. The unlicensed configurations' speeds are the
`e4b.serve.buildout.bo5.*` rows; #387's fused q/k/v was quality-clean and
bought nothing.

**Closing the gap, not the gate (2026-09-04, lane bo6; measured — receipt
[`bench/hybrid-g9/throughput-20260904/bo6/`](../bench/hybrid-g9/throughput-20260904/bo6/README.md),
table in its [`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo6/RESULTS.md)).**
The int4-expert arms that failed their second text were re-run with
per-expert GPTQ calibration (e4b#384) against NF4 re-scored on the same box.
**Sequential calibration is the mechanism that ships** (0.35.0): on
Qwen3-30B-A3B the order of calibration and packing alone moves c4val1 by
0.200 ppl and flips the verdict
(`e4b.serve.buildout.bo6.qwen3.calibration-order.c4val1.2026-09-04`, on a
deterministic instrument, `e4b.serve.buildout.bo6.qwen3.k8-deterministic.5090.2026-09-04`);
more calibration text helps (16k → 64k → 256k tokens:
`e4b.serve.buildout.bo6.qwen3.calibexp-streamed-16k.c4val1.2026-09-04`,
`e4b.serve.buildout.bo6.qwen3.calibexp-streamed-64k.c4val1.2026-09-04`,
`e4b.serve.buildout.bo6.qwen3.calibexp-streamed-256k.c4val1.2026-09-04`),
none of it claimed as an improvement until wikitext agrees. **Qwen3-30B-A3B's
licensed serving stack is the streamed one** (bo6c, 2026-09-05): sequentially
calibrated int4 experts at 64k C4-validation tokens + C4-calibrated int4
attention + round-1/2 folds + router epilogue + decode glue passes both texts
inside the 0.0095-nat floor (wikitext −0.053 / c4val1 −0.066 ppl), **licensed
under the unchanged gate**, no improvement claimed by a number
(`e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`; the
calibration is deterministic, `e4b.serve.buildout.bo6c.qwen3.calib-deterministic.5090.2026-09-05`);
its speed is bo7's, below. The all-at-once calibrated stack also passed both
texts (`e4b.serve.buildout.bo6.qwen3.all-calibexp-allatonce.k8.2026-09-04`).
**Sequential calibration did not close Mixtral's gap:** c4val1 +0.039, wikitext
+0.077 ppl — FAIL as registered, not licensed
(`e4b.serve.buildout.bo6.mixtral.lic-calibexp-streamed.k8.2026-09-04`), the
mirror image of bo5's RTN stack; Granite stays NF4. Next levers are not gate
changes: a per-expert NF4 fallback for the largest-residual experts, or the
64k set scored on wikitext. Mixtral's speed arms died on their own alarms
during the ~85-min calibration (a harness limit, not a model result). Qwen3's
NF4 reference sits 0.006 nats from bo5's on the identical window while
Mixtral's agree to 0.001 — that shift stays open, and no sub-0.01-nat number
is compared across lanes.

**The throughput census (2026-09-05, lane bo7; measured — receipt
[`bench/hybrid-g9/throughput-20260904/bo7/`](../bench/hybrid-g9/throughput-20260904/bo7/README.md),
table in its [`RESULTS.md`](../bench/hybrid-g9/throughput-20260904/bo7/RESULTS.md)).**
Speed only, under the **shipped** code (0.35.0 + 0.30.0 at their `main`,
hook v6 at its 16k default), all six families on one rented RTX 5090 (EPYC
7Q83 host, instance 49916675), 48 arms at B=1 and B=16, every ratio to
that family's own NF4 arm on that box and every licence label copied from
this register — bo7 licenses nothing, and no bo3/bo5/bo6 number is divided
into a bo7 number. Every ratio below is vs e4b's own NF4 control on the same
box, never a field engine, and only Qwen3 has a field comparator measured:
**Granite's licensed stack** (NF4 experts + folds + epilogue) is ×1.341 at B=1
(304.9 tok/s) and ×1.160 at B=16 (1836.8;
`e4b.serve.census.bo7.granite.b1.5090.2026-09-05` / `e4b.serve.census.bo7.granite.b16.5090.2026-09-05`); **OLMoE's position is NF4**
(282.5 / 1347.5, ×1.000; `e4b.serve.census.bo7.olmoe.b1.5090.2026-09-05` / `e4b.serve.census.bo7.olmoe.b16.5090.2026-09-05`) because nothing above it is licensed on this
register — its full stack is ×2.070 / ×2.289 measured, not licensed (the tp
row's label predates the two-text clause; its calibrated attention is refused
on this family); on both families' position configs the NF4 grouped expert GEMM is 61.7 % (Granite) and 71.9 %
(OLMoE) of B=16 decode kernel time (P91, descriptive, licenses nothing;
`e4b.serve.p91.nf4-families.kernel-census.5090.2026-10-01`), so the next family lane is an NF4 grouped small-M kernel;
that kernel (K25, `E4B_NF4_GROUPED_SMALLM`, default `auto` since P96) reads ×0.937 at B=16 on Granite with K8 inside the gate, and
QUALITY_FAIL on OLMoE (c4val1 K8 −0.107 ppl, step ×1.024); its own GEMM runs within 4 % of the served one (P92,
`e4b.serve.p92.nf4-families.k25-b16.5090.2026-10-01`);
at the served precision (the select tree through TF32 MMA, P93) the route takes Granite's B=16 step to ×0.594 (B=1
×0.854, K8 inside the gate) and OLMoE's to ×0.598, but OLMoE's c4val1 K8 moved +0.155 ppl, the other sign from P92's
−0.107, so the default stays `0` (`e4b.serve.p93.nf4-families.k25-tree-tf32-b16.5090.2026-10-01`);
against the arithmetic it would replace above T == 1 (the served M-tile), K25 reads QUALITY_FAIL in both families
(c4val1 K8 +0.102 Granite, +0.168 OLMoE; wikitext inside), and the served M-tile itself sits −0.078 from the GEMV on
Granite c4val1, so the default stays `0` and c4val1's spread is the next question
(P94, `e4b.serve.p94.nf4-families.k25-vs-mtile-k8.5090.2026-10-01`);
on disjoint windows, K8's own spread across those equal-error arithmetics is 0.054 / 0.055 ppl on c4val1 and
0.026 / 0.030 on wikitext, so a single-window 0.05 gate cannot resolve these families; a windowed gate needs
W >= 5 / 2 windows, and P94's window was an outlier for K25 (P95, UNDER_RESOLVED,
`e4b.serve.p95.nf4-families.k8-window-spread.5090.2026-10-02`);
under that windowed gate, on fresh windows, K25 against the M-tile reads mean t − m −0.001 / −0.015 (Granite) and
−0.016 / −0.004 (OLMoE), LICENSED, so `E4B_NF4_GROUPED_SMALLM` defaults to `auto` for rows above T == 1 (P96,
`e4b.serve.p96.nf4-families.k25-windowed-k8.5090.2026-10-02`);
**gpt-oss's quoted best is its own reference arm** (NF4 + exact folds, 144.5 / 761.6; `e4b.serve.census.bo7.gptoss.b1.5090.2026-09-05` / `e4b.serve.census.bo7.gptoss.b16.5090.2026-09-05`) and the MXFP4
store under the route rule reads ×1.293 / ×0.970; with K21 serving the store's batched rows, its B=16 step reads
×0.581 against the NF4 fallback, at a lower KL from the reference (0.00147 vs 0.00192;
`e4b.serve.p90.gptoss.mxfp4.k21-b16.5090.2026-10-01`);
**Qwen3's licensed stack** — the streamed 64k calibrated pack artifact bo6c
licensed on both texts (11512 gptq / 776 rtn), run under the lane's amendment
2: **×2.067 at B=1 (238.1 tok/s; anchor-class projection ≈ 329 tok/s, from an
uncertified class) and ×2.602 at B=16 (1327.5 tok/s) vs e4b's own NF4
control** — `e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05` / `e4b.serve.census.bo7.qwen3.b16.5090.2026-09-05`. The
same-box field comparator is now P58's vLLM 0.30.0 GPTQ-Int4 / MarlinExperts
(260.3 / 1925.6 graph vs the current int4 stack's 239.4 / 1379.2 — 1.087 / 1.396,
2026-09-22; P37's 0.28.0 rows 286.0 / 2030.0 are history); #405 (closed
2026-09-06) is the P37 reproduction item (c4val1 FAIL), not a licence
withdrawal. Its
speed matches the lane's 16k arm and the RTN stack within 1%: the pack
changes the values, not the kernel or the bytes. **Gemma-4 has no K8
instrument, so no arm carries a K8 licence**; the register's position with
that caveat is the exact round-1 fold + epilogue on NF4 (`r1epi`), ×1.281
at B=1 (103.6 tok/s) and ×1.106 at B=16 (675.8;
`e4b.serve.census.bo7.gemma4.b1.5090.2026-09-05` / `e4b.serve.census.bo7.gemma4.b16.5090.2026-09-05`), and the quoted int4 best
(bo3's `stack`) reads ×1.705 / ×1.697 measured, no quality verdict (Gemma-4-it
loaded on this host without the #344 fault). **Mixtral's position
is NF4** (50.3 / 191.4, ×1.000; `e4b.serve.census.bo7.mixtral.b1.5090.2026-09-05` /
`e4b.serve.census.bo7.mixtral.b16.5090.2026-09-05`): the exact folds are ×1.062 / ×1.018 but unscored as a combined
arm, and the RTN int4 stack ×2.329 / ×1.959 and the calibrated-attention stack
×2.597 / ×1.962 are measured, not licensed (bo5's second-text FAILs stand).
A calibrated int4-expert pack costs nothing over an RTN one in speed
(Granite 2.04 vs 2.05 ms, Qwen3 4.197 vs 4.204), and the streamed calibration's
pack counts reproduce across hosts (Granite 2524/36, Qwen3 10820/1468 — bo5's
and bo6's). All 50 arms ran with no alarm, refusal or traceback.

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
  verdict row (`licensed_by`), which `scripts/check_claims_register.py`
  enforces with the register's structure. The 2026-09-03 single-stream stack
  this page led with is the same RTN class, quoted above as unlicensed speed.
- **The 2026-09-03 same-box vLLM comparison is SUPERSEDED** (2026-09-05):
  `e4b.serve.h2h.vllm.same-box` (superseded; ×1.47 / ×1.55 on box 49702459,
  the 0.27.0/0.21.0 RTN stack, vLLM version unrecorded) points at lane p37,
  `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05` (vLLM pinned, identical
  prompt ids, every knob recorded; above), itself now history beside P58.
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
  ([`BITSANDBYTES.md`](BITSANDBYTES.md)). The unrecorded build was
  **remeasured on a release** on 2026-10-04: the same card and harness, on
  bitsandbytes 0.50.2 (#392, `e4b.train.energy-honest.a2000-bnb0502.2026-10-04`).
  `matmul_4bit` reads decode 0.91–1.06× (break-even), prefill 1.29–1.49× and
  train 1.64–2.15× over three passes.
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
  (`bench/p55x/`, 2026-09-22) built Qwen3's streamed 64k calibrated pack as a
  hash-pinned artifact and ran the registered K8 gate **on those bytes loaded
  back by fingerprint** — the path a user gets — and it **passes both texts**
  against an NF4 reference bit-identical to bo6c's
  ([`e4b.serve.p55x.qwen3.all-calibexp-streamed-64k.k8.2026-09-22`](claims.json)).
  The bytes are retained and were verified after transfer by two independent
  implementations on two machines; a loader given the fingerprint refuses
  anything else and never rebuilds from the recipe, so the licence now travels
  as bytes. The pack is **byte-identical to lane P39's of 2026-09-10** (another
  5090, e4b 0.35.3 against 0.36.4): four builds across at least three boxes and
  a release boundary agree on every byte, so **P37's divergence (11522/766,
  c4val1 +0.109) is an outlier rather than the rule**, and the open question is
  what was different about that host — the question
  [#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405) was
  narrowed to on 2026-09-22, though #405 has been closed since 2026-09-06 and no
  open issue carries it.
  Two things stay open and are not small. **The fingerprint covers the experts
  only**: `engines/int4_attn_calib.py` has no serialisation, so the 192
  calibrated attention projections are re-derived on every load — and the same
  pinned expert bytes with RTN attention **fail** c4val1 at +0.13237, so that
  unpinnable half is the half carrying the quality. And bo6c's own row keeps no
  fingerprint, because its bytes were not retained; it stays active rather than
  superseded, since the bo7 census rows take their licence from it and nothing
  establishes that their pack is this pack. `min_rows`, damping and the K8
  budget are unchanged.
- **Residency's launch cost is a fixed per-layer count, and the cold cost
  model's transfer term is wrong on a gen 4 link (lane P66, `bench/p66/`,
  [#711](https://github.com/pjordanandrsn/experts4bit-qlora/issues/711),
  2026-09-24).** A pre-registered census on one RTX 5090 read the pipelined
  engine at **+10 launches, +2 copies, 0 syncs per MoE layer per token**
  against the all-resident step, spread exactly 0 across cold fraction 0 /
  0.5 / 1.0, eager and captured
  ([`e4b.serve.p66.qwen3.pipelined-residency-fixed-count.5090.2026-09-24`](claims.json));
  the all-hot fixed tax is 1.184 ms/token of device time captured, 8.05 ms
  of eager wall
  ([`…pipelined-fixed-tax.captured…`](claims.json)), and at the RFC's 17 %
  cold it is 8.9 % of residency's captured cost — launches cannot explain an
  RFC-size gap in this engine. What is open is the transfer: grouped-nf4-gemm's
  `cold_deadline` (bytes over link + bytes over VRAM) **under-predicts the
  measured UVA gather by 1.57–2.02×** there
  ([`…pipelined-gather-over-cold-deadline…`](claims.json)). The gather runs at
  the box's single-copy link rate (14.4 GB/s implied, 14.72 GB/s probed), not
  the back-to-back rate the calibration blob records (23.07 GB/s). Which
  constant the model should carry, and why the two differ on a gen 4 × 16
  link, is the follow-up filed against `kernel/cold_deadline.py`
  (a per-step fixed term, a measured efficiency factor before any RFC
  comparison, the hybrid tier's mixed-layer dispatch term the deadline
  destination omits). The MXFP4 NVMe engine's 4 syncs per layer and the
  hybrid tier's composition-dependent 4 / 10 / 8 are structural, as
  registered ([`…gptoss.mxfp4-and-hybrid-sync-census…`](claims.json)).
- **`enable_batched_train`'s engagement envelope.** It falls back to the
  reference forward per call above `_PAD_WASTE_LIMIT` (`engines/batched.py`);
  a positive return value is a patch count, not kernel engagement. In the
  code tp1 measured (0.35.0) the fallback was silent, which is why three
  `batched` arms are VOID (above); it engaged everywhere only on Mixtral's
  8-expert and Granite's 40-expert shapes. 0.35.1 (#402) makes the
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
- **[#713](https://github.com/pjordanandrsn/experts4bit-qlora/issues/713) — answered by lane P67 (2026-09-24): Gemma-4's
  training-parity floor is measured, and the attention-4-bit paths are licensed under it.** Since #435 the arms
  convert all 115 structural projections (25 sliding layers × 4, 5 full-attention layers × 3 without `v_proj`) and
  run VALID. tp1's constant 0.05 band against zero failed every accelerated path here, the kernel-free one included
  (P56). P67 drew the floor the registration asked for — the reference against itself, one plain repeat and four
  fixed permutations of the per-expert loop, one session on one RTX 5090 — and read F_hi **0.121 final / 0.128
  median** ([`e4b.train.p67.gemma4.attn4-reorder-floor.5090.2026-09-24`](claims.json)): the reorderings move the
  step-0 loss by 0.052–0.217 where the fused kernels move it 0.034, and the plain repeat diverges at step 2. Against
  `max(0.05, 3 · F_hi)` = 0.363 / 0.383 the fused arm (0.013 / 0.036) and the batched arm (0.037 / 0.048) **PASS**,
  carried by the tolerance and not detectable, and the two earlier sessions' FAILs PASS the same band, so the
  reading is not MIXED. Under the registered table `fast_train` and `reference_train` in the attention-4-bit
  configuration are **supported** on `gemma4_text` ([`capabilities.json`](capabilities.json)); `batched_train`
  stays void (its arm ran at `pad_waste_limit` 64, not the shipped default). What stays open is resolution, not
  licence: a 20-step trajectory cannot detect a 0.05-nat defect on this model, so a per-op or matched-routing
  instrument is the next step if one is ever wanted ([`bench/p67/RESULTS-p67.md`](../bench/p67/RESULTS-p67.md)).
  tp2/P40's harness still hard-codes 4 · n_layers ([#412](https://github.com/pjordanandrsn/experts4bit-qlora/issues/412)).
- **[#344](https://github.com/pjordanandrsn/experts4bit-qlora/issues/344) —
  Gemma-4 failed to load on 2 of 6 rented hosts** (2026-09-03) with `CUDA error:
  invalid argument`, after the experts quantise. A 2 GiB host-hop fix was merged
  and reverted the same day: the model's largest tensor is 1.375 GiB, so it never
  triggered. The six hosts' forensics, put side by side
  (`bench/p55/P55-PREREG.md`), killed the driver and GPU leads by construction
  and left one reading: host RAM against the checkpoint's 49.9 GB (46.48 GiB)
  single shard. **Lane P55 tested that reading and refuted it** (2026-10-04,
  [`bench/p55/RESULTS-p55.md`](../bench/p55/RESULTS-p55.md)). On a host of the
  failing class (an RTX 5090 with 58.1 GiB of effective memory, its cgroup
  limit), the checkpoint loaded cleanly three times:
  - unarmed;
  - under staged synchronisation with `CUDA_LAUNCH_BLOCKING=1`, every stage clean;
  - with the cgroup's headroom down to 5.7 GiB during the load.

  The fault is **unreproduced on the current loader**; the failing runs used
  e4b as of 2026-09-03. The issue stays open as host-specific, with its leads
  dead. The loader no longer fails silently about it: a shard-read
  failure now prints the shard, its size and the host's `MemTotal` /
  `MemAvailable` / cgroup limit, and `E4B_LOAD_SYNC_DEBUG=1` inserts staged
  `torch.cuda.synchronize()` checkpoints (arming `CUDA_LAUNCH_BLOCKING=1` while
  CUDA is still uninitialised) so a fault is bound to a load stage rather than
  to whichever CUDA call observed it.
- **`e4b.open.tr2-repro-gap` stays open in the register**: reproducing the
  TR2 training receipt from published artifacts. The bake step it names as
  missing ships in grouped-nf4-gemm: `nvme_bake_nf4` writes the NF4 arena
  that `TRAIN_ARENA` (`enable_hybrid_train`) and `enable_nvme_train_residency`
  read (`bench/hybrid-g5/g5_run.py`, `bench/host-ram-ceiling/prep.sh`); no
  TR2 rerun from such an arena is recorded.
- **[#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359) — Gemma-4, re-scoped**: (1) DONE in 0.32.0 with
  grouped-nf4-gemm 0.26.0 — 32-wide key scales on the 512-dim heads
  take the paged path from 3.59239 to 3.57228 nats on the P26b window;
  (2) still open: a parity instrument that survives batch-shape
  variance — a long window, or matched routing — before any verdict is
  quoted for this family.
- **Fusing q/k/v on the int4 attention store changes the arithmetic at
  B=16** (lanes P54 / P57 / P59, `bench/p54/`, `bench/p57/`, `bench/p59/`).
  Byte-identical by construction and **token-identical at B=1** (0.516 ms/step,
  12.4 %, on one 5090), it diverged from the control on 14 of 16 B=16
  sequences. P57 excluded the round-2 glue and P59 the K16 GEMM (bitwise
  invariant to fusing at 2–16 rows; KL exactly 0 at 16-row decode), so the
  divergence lives in the >16-row path: the harness's 128-token prefill runs
  cuBLAS on the cached bf16 weight, whose kernel choice depends on N.
  **Bounded and licensed** (P59 amendment 1,
  `e4b.serve.p59b.qwen3.b16.fqkv-kl.5090.2026-09-22`): 0.0044 nats/token from
  the unfused stack, a fresh-process rebuild bit-identical — reorder-class,
  inside the shipped bar (≤ 0.10 nats, top-1 ≥ 0.93). **The fused q/k/v path is
  now the default on the int4 lanes at B=16 as well as B=1**, saving P54's
  0.223 ms/step at B=16.
- **K17's fused split-K reduce is exact and slower in the consumer** (P57,
  same page): −0.038 ms/step at B=1, −0.217 at B=16, the epilogue landing inside
  the critical-path GEMV while the launches it removed were overlapped. Stays
  opt-in in grouped-nf4-gemm; the census row it targeted was GPU time, not a launch.
- **The distinct-expert count at B=16 is MEASURED (P57 amendment 3, 2026-09-22,
  `e4b.serve.p57.qwen3.b16.distinct-experts.5090.2026-09-22`): 58.7 distinct
  experts per layer per decode step** (uniform expectation 82.4), from an
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
- **A token decoded alone and the same token inside a verify or prefill do not
  get the same bits (lane P63, `bench/p63/`,
  [#708](https://github.com/pjordanandrsn/experts4bit-qlora/issues/708),
  2026-09-24, one RTX 5090, Qwen3-30B-A3B).** The expert routes split three ways:
  - **Row-exact.** Every experts module (48/48) gives each token its T = 1 bits
    at 16, 17 and 160 tokens on three routes:
    - the int4 store under `FORCE_SINGLETON_GROUPS`
      (`e4b.serve.p63.qwen3.int4-singleton.row-exact.5090.2026-09-24`);
    - the int4 store under `DEVICE_GROUPING` up to 256 routed rows
      (`e4b.serve.p63.qwen3.int4-device-gemv.row-exact.5090.2026-09-24`);
    - NF4 under `FORCE_SINGLETON_GROUPS` on the dot-pad GEMV
      (`e4b.serve.p63.qwen3.nf4-singleton.row-exact.5090.2026-09-24`).
  - **A different function at T > 1 than at T = 1.** Each path keeps the
    quality licence it was measured on, and no licence transfers between row
    counts. P63 moves no default.
    - The default T > 1 routes: the int4 store's dequant + bf16 matmul (0 of
      1,280 rows equal, rel L2 ≤ 1.5e-2), and NF4's M-tile on TF32 weights
      (rel L2 2.7e-3).
    - The fused router epilogue's fp32 routing weights at ≤ 64 rows, against
      the upstream router's bf16 above
      ([#726](https://github.com/pjordanandrsn/experts4bit-qlora/issues/726)).
  - **Summation order only (reorder-class).** The grouped int4 GEMM above 256
    rows, and the scalar NF4 GEMV under `GNF4_GEMV_DOTPAD=0`, as grouped-nf4-gemm's
    register records.
  - **End to end.** No position is exact on any route, and every first
    difference is at layer 0 in attention, never in the experts. KL mean is
    0.008–0.033 nats/token and top-1 agreement 0.88–0.98. 16 of 45 cells are
    under the shipped top-1 bar (0.93) over 64–160 positions, while no cell is
    near its KL bar. **Lane P68 (`bench/p68/`,
    [#725](https://github.com/pjordanandrsn/experts4bit-qlora/issues/725),
    closed) answered both halves:**
    - **What makes the difference.** Forcing the attention projections and the
      core to their T = 1 calls, row by row, moves the first difference out of
      layer-0 attention. Forcing everything that picks its arithmetic by row count
      makes a verify **bit-identical** to decode on both stacks
      (`e4b.serve.p68.qwen3.verify-from-t1-calls.row-exact.5090.2026-09-24`).
    - **The size, over ~1,000–2,000 positions per cell.** Every served T > 1 cell
      is inside the bar: KL ≤ 0.017, top-1 ≥ 0.946
      (`e4b.serve.p68.qwen3.t-gt-1-vs-t1.within-bar.5090.2026-09-24`).
  - **`E4B_FUSE_COMBINE=0` at T = 1** (lane B393's size) is KL 1.18e-04 on
    NF4, and 1.70e-02 with 7 of 160 flips on int4.
- **The int4 experts' int8 activation step costs no measurable quality at
  B = 1 decode; the attention projections' is still unpriced** (lane P64,
  `bench/p64/`, 2026-09-24, one RTX 5090, the licensed pack `0c9955a9…`).
  - **What was compared.** The experts' T = 1 calls were routed through bf16
    activations instead of the per-32 int8 quantise, over the same int4 bytes.
  - **The result.** KL was 0.0034 / 0.0025 nats/token on wikitext / c4val1:
    0.61× and 0.77× of the instrument's own arithmetic-order floor. dNLL
    spans 0, so the step is **INDISTINGUISHABLE**, and W4A8 expert decode
    stays (`e4b.serve.p64.qwen3.b1.expert-int8-step.5090.2026-09-24`).
  - **Unread.** The attention half was skipped for time and stays open
    ([#728](https://github.com/pjordanandrsn/experts4bit-qlora/issues/728)).
    Nothing prices the int8 step at B = 16 either: P59's B = 16 KL, whose
    register row now says so, never ran it, because its scorer leaves
    `DEVICE_GROUPING` off.
- **The fused router's weights can be cast to the model's dtype at no
  measurable quality cost at B = 1 decode** (lane P70, `bench/p70/`,
  2026-09-25, one RTX 5090, the licensed pack `0c9955a9…`).
  - **What was compared.** `E4B_ROUTER_EPI_CAST=1` rounds the fused router's
    fp32 top-k weights to bf16 at ≤ 64 rows, as the upstream router does above
    64 ([#726](https://github.com/pjordanandrsn/experts4bit-qlora/issues/726)).
    On the RTX 5090 the cast weights are bit-equal to upstream's at the same
    row count. The experts match too. Only the slot order differs.
  - **The result.** KL was 0.0033 / 0.0028 nats/token on wikitext / c4val1:
    0.60× and 0.92× of the instrument's own floor. dNLL spans 0, so the cast
    is **INDISTINGUISHABLE**
    (`e4b.serve.p70.qwen3.b1.router-weight-cast.5090.2026-09-25`). By the
    registered rule it is now the default for the `softmax_topk` kind
    (since 0.37.5). `E4B_ROUTER_EPI_CAST=0` restores fp32. It stays until #674's
    K8 question is measured, then goes with notice (#782).
    `topk_softmax` keeps fp32 until it is read.
- **Bucketed CUDA-graph decode decodes exactly as the eager runner and is
  4.8× faster on a host-bound host** (lane B771b, `bench/b771b/`,
  2026-09-29, one RTX 5090, NF4 Qwen3-30B-A3B). It supersedes P80's 2.3×.
  - **What was fixed first.** P80 measured a path with a bug. With one active
    row, bucket 1 appended its K/V to a scratch slot, so the row decoded that
    phase without its own new tokens (#777). grouped-nf4-gemm 0.33.7 also
    made the fused fp8 append write the eager quantize's bytes.
  - **Now.** Over P80's trace (16 → 8 → 4 → 2 → 1 active rows), the graphs,
    the bucket step and the eager runner with the same grouping decode
    identical tokens in every row. Aggregate decode is 267.5 against 55.6 /
    55.8 tok/s: B1/A1 = 4.815 and B2/A2 = 4.791
    (`e4b.serve.b771b.qwen3.dynb.graph-buckets.fixed-path.5090.2026-09-29`).
  - **Where it comes from.** The graph step is within 4–10% of P80's at every
    row count. The eager step is host-bound and 2.15–2.34× slower on this
    Zen 2 host than on P80's Zen 5 one, so the ratio does not travel.
  - **Still open.** It stays opt-in. The HTTP shim and `infer` use
    transformers' `generate`, not `PagedModelRunner`. It is not measured on
    other families or under arrivals. P82 (next item) re-measured the int4
    stack.
- **On the int4 serving recipe, the graph path decodes exactly as the eager
  runner and is 12× faster on a host-bound host** (lane P82, `bench/p82/`,
  2026-09-29, one RTX 5090 on an AMD EPYC 7R13 host, Qwen3-30B-A3B). It
  supersedes P81's 14×, which was measured on the path with the two bugs below.
  - **What was compared.** P80's trace and rule, on P55x's recipe: calibrated
    int4 experts and attention, the T=1 folds, and the fused router epilogue
    at fp32 weights. Both packs came from one build and were installed by
    fingerprint in every arm, all with the device grouping. Aggregate decode
    was 811.9 / 814.2 tok/s against 67.6 / 65.9: B1/A1 = 12.01, B2/A2 = 12.36
    (`e4b.serve.p82.qwen3.licensed-int4-fp32router.dynb.graph-buckets.fixed-path.5090.2026-09-29`).
  - **Correctness.** The eager runner, its bucket step and both graph arms
    decode identical tokens in every row. In P81 they did not. There were two
    causes, both now fixed: grouped-nf4-gemm#413's fp8 append, and #777's
    bucket-1 append to a scratch slot.
  - **What the ratio is.** Eager decode takes 88–102 ms per step at every row
    count, so it is host cost. The graphs take 13.6 ms at 16 rows and 4.4 ms
    at one. The ratio does not travel to another host. A `PagedModelRunner`
    caller on this stack that does not enable graphs leaves most of the
    throughput unused. The package's serving entry points use transformers'
    `generate`, not the paged runner.
  - **#674: answered.** The licensed int4 recipe's fp32 K8 moved from
    6.36709 to 6.36396 because of one change, gnf4#413 (the fused fp8 KV
    append's rounding). P70's build with that append off reads 6.36396 bit
    for bit (P85, `e4b.serve.p85.qwen3.int4-recipe.k8.fused-append-413.5090.2026-09-30`;
    P83/P84 isolated it). Box-invariant on AMD hosts; an Intel host's
    attention calibrated differently (P84).
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
come from a private audit tree. They are real runs with real
receipts that this repository does not carry, and they are labelled that
way in `claims.json`. Treat them as you would any number you cannot
check.
