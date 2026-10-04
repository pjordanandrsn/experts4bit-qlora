# moe-generalize: per-family training ladders on an RTX A2000 (informational)

*Branch `session/moe-generalize`, 2026-10-04. These are within-box readings on one RTX A2000 12 GB (sm_86): the owned seat
card, in a container capped at 2 CPU cores and 30 GB, shared with other workloads. They are not positions, licences or
cross-framework ratios. Lane MG1 ([`MG1-PREREG.md`](MG1-PREREG.md)) reads the licence pairs and the same ladder on an RTX 5090.*

Receipts: [`receipts/`](receipts/) (one JSON per family plus its log; `report.py receipts` regenerates every table below) and
[`tp1-a2000/`](tp1-a2000/) (`mg1_reduce.py tp1-a2000`).

## How to read the ladder

[`ladder.py`](ladder.py) loads a family once and runs each rung in one process, interleaved A..Z Z..A (reference once,
last). Six timed steps follow three warmups, at seq 512, micro-batch 1, r 16, with attention LoRA and 4-bit attention on.

| rung | what it is |
|---|---|
| `reference` | no `enable_fast_train`: the `ExpertsLoRA` reference forward and its recompute backward |
| `fused_pre` | `enable_fast_train(dgrad=True)` with every Qwen-campaign default turned back off (legacy grouping, no pinned ring, no host reuse, untrimmed padded delta, max-group tile, composite RMSNorm and RoPE) |
| `fused` | the shipped path, at the package defaults |
| `keep` | `fused` plus MoE-activation retention (`E4B_MOE_KEEP_LAYERS=all`, `NF4_QLORA_COMPACT_DELTA=1`) |
| `…@GNF4_TRAIN_GEMM=dequant_mm` | this branch's prototype dequantize-then-GEMM route (one dequant of the present experts, then one cuBLAS GEMM per group on this card). It never merged: grouped-nf4-gemm#459's `dense` route answers the same question with one expert's weight at a time |

**Compare rungs by device seconds, not wall clock.** On this seat, wall clock for host-bound steps swings by up to 2.3×
between runs. The device-busy time of one profiled step repeats within about 8 % across the ABBA pairs. Wall clock is shown
because the host-bound rungs (`reference` above all) are a host-cost finding in their own right.

## The ladders

Device seconds per step, with the ratio to `fused` in parentheses. Then wall seconds per step, host→device syncs per step,
and the routed experts' share of the fused step's device time.

| family (receipt) | fused | fused_pre | keep | reference | dequant_mm probe | fused wall | ref wall | syncs fused / pre / ref | experts' share |
|---|---|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B, fp32 adapters (`olmoe_fp32adapters`) | 1.346 | 1.522 (1.13) | 1.062 (0.79) | 1.841 (1.37) | — | 1.61 | 10.40 | 35 / 369 / 24,610 | 52 % |
| Granite-3.1-3B-A800M, fp32 adapters (`granite31_fp32adapters`) | 1.233 | 1.511 (1.23) | 0.946 (0.77) | 1.453 (1.18) | — | 2.16 | 7.29 | 67 / 737 / 30,090 | 39 % |
| LFM2-8B-A1B (`lfm2`) | 1.227 | 1.333 (1.09) | 0.925 (0.75) | 1.162 (0.95) | — | 1.28 | 3.65 | 47 / 507 / 16,798 | 72 % |
| Granite-4.0-H-tiny, transformers' torch Mamba scan (`graniteh`) | 1.751 | 1.940 (1.11) | 1.505 (0.86) | 2.623 (1.50) | — | 1.84 | 11.41 | 82 / 920 / 58,233 | 33 % |
| Granite-4.0-H-tiny, mamba-ssm 2.3.2 + causal-conv1d 1.7.0 (`graniteh_mambak`) | 1.124 | 1.294 (1.15) | 0.857 (0.76) | — | 0.907 (0.81) | 1.28 | — | 82 / 920 / — | 50 % |
| Qwen3-30B-A3B, first 4 layers (`qwen3_4L`) | 0.299 | 0.362 (1.21) | 0.232 (0.78) | 0.620 (2.07) | 0.261 (0.87); with keep 0.200 (0.67) | 0.38 | 3.76 | 11 / 93 / 11,098 | 58 % |
| OLMoE-1B-7B, bf16 adapters, the shipped configuration (`olmoe`) | 1.017 | 1.391 (1.37) | 0.861 (0.85) | — | 0.803 (0.79) | 3.95 | — | 35 / 369 / — | 66 % |
| LFM2-8B-A1B, route rungs (`lfm2_route`) | 1.225 | — | 0.912 (0.74) | — | 0.874 (0.71); with keep 0.617 (0.50) | 1.27 | — | 47 / — / — | 73 % |
| ERNIE-4.5-21B-A3B, first 4 layers (`ernie_4L`) | 0.326 | 0.390 (1.20) | 0.254 (0.78) | 0.363 (1.11) | 0.248 (0.76); with keep 0.190 (0.58) | 0.34 | 1.09 | 9 / 70 / 4,616 | 65 % |
| Nemotron-3.5-Lightning-30B-A3B, first 8 layers: 3 MoE, Mamba and attention (`nemotron_8L`) | 0.334 | 0.424 (1.27) | 0.257 (0.77) | 0.851 (2.55) | 0.358 (1.07); with keep 0.257 (0.77) | 1.23 | 14.05 | 9 / 70 / 9,224 | 62 % |
| Qwen3.6-35B-A3B, first 4 layers: 3 Gated DeltaNet and 1 attention (`qwen36_4L`) | 0.400 | 0.495 (1.24) | 0.314 (0.79) | 1.433 (3.58) | 0.389 (0.97) | 0.45 | 6.33 | 11 / 93 / 23,290 | 49 % |

Peak VRAM: `keep` costs 0.18 to 1.32 GB over `fused` (OLMoE 6.81 → 7.44 GB with fp32 adapters and 5.39 → 6.22 with bf16,
Granite-3.1 4.09 → 5.06, LFM2 6.32 → 7.36, Granite-H with its kernels 6.94 → 8.26, Qwen3 slice 3.76 → 4.00, ERNIE slice
2.87 → 3.11, Nemotron-H slice 4.94 → 5.12, Qwen3.6 slice 5.95 → 6.18). Apart from the dequant probe, which allocates the decoded
stack (up to +0.66 GB), no other rung raises peak VRAM by more than 0.07 GB.

**Engagement, every `fused` rung.** Every MoE layer was patched (OLMoE 16, Granite-3.1 32, LFM2 22, Granite-H 40, Qwen3
slice 4, ERNIE slice 3, Nemotron-H slice 3, Qwen3.6 slice 4), and none was skipped. Every frozen-GEMM dgrad was served by the
kernel (`DGRAD_STATS` loop 0), including Nemotron-H's non-gated relu² experts on real weights. The LoRA delta took the padded
route on every call, and the pinned ring never overflowed. The train-GEMM route was `fused`.
- **RMSNorm:** every norm was fused, with no fallback calls, and the probe read each family's own formula.
  - `w·round(n)`: OLMoE 65, Granite-3.1 65, LFM2 61, Granite-H 81, Qwen3 slice 17, ERNIE slice 9.
  - `w·n` fp32: Nemotron-H slice 9.
  - Centered `(1+w)·n` fp32: Qwen3.6 slice 11, its norms' first fusion.
- **RoPE:** one rotary function per family was fused. ERNIE's was **refused on semantics** (it rotates interleaved pairs), so
  ERNIE keeps its own.
- **Recurrent blocks:** Granite-H's Mamba, Nemotron-H's Mamba and Qwen3.6's Gated DeltaNet ran on their kernels (`recurrent
  fallback []`) once mamba-ssm, causal-conv1d and flash-linear-attention were installed.

## tp1 parity pairs on the A2000

These use tp1's arm driver unchanged: N 60, seq 512, r 8, the registered clinical fixture, reference vs
`enable_fast_train(dgrad=True)`. The A2000 is outside tp1's train-anchor band, so these pairs validate the harness path per
family. They license nothing.

| family | verdict | d_final | median d | reference s/step | fused s/step | peak GB (both arms) | patched |
|---|---|---|---|---|---|---|---|
| LFM2-8B-A1B | PASS | 0.0088 | 0.0165 | 2.748 | 0.576 | 6.11 | 22 |
| Granite-4.0-H-tiny | PASS | 0.0062 | 0.0120 | 11.040 | 1.267 | 6.18 | 40 |

Both Granite-H arms ran the mamba-ssm / causal-conv1d kernels: neither log has a fallback warning.

## What the ladders say

1. **The Qwen-general optimisations carry over to every family measured.** `fused_pre` takes 1.09 to 1.37× `fused`'s device
   time. Turning them on also removes about nine in ten host syncs: 369 → 35 on OLMoE, 737 → 67 on Granite-3.1, 507 → 47 on LFM2, 920 → 82
   on Granite-H, 93 → 11 on the Qwen3 and Qwen3.6 slices and 70 → 9 on the ERNIE and Nemotron-H slices. No family needed a
   family-specific patch to get them.
2. **Activation retention saves 15 to 26 % of device time on every family, the hybrids included** (14 % on Granite-H while it
   runs the torch scan). Before this branch it matched
   no layer on Granite, LFM2 or Granite-H, because it looked for `self_attn` + `mlp` by name. The structural matcher reaches every
   MoE-bearing layer, and on Granite-H it keeps the Mamba mixers checkpointed on their own (80 children for 40 layers).
3. **On the hybrids, the recurrent kernels matter more than anything on the expert side.** Without mamba-ssm and causal-conv1d,
   transformers runs its PyTorch scan, and installing the kernels cut Granite-H's fused step by 36 % of device time
   (1.751 → 1.124 s), more than every Qwen-general optimisation together. `enable_fast_train` now names the fallback (`FAST_TRAIN_STATS["recurrent_fallbacks"]`, plus a
   `RuntimeWarning`). MG1's box installs the kernels.
4. **LFM2's geometry favours dequantize-then-GEMM.** LFM2 routes top-4 of 32 experts, so at seq 512 each expert sees about
   64 rows. Its reference arm (bitsandbytes dequant, then per-expert cuBLAS) uses 0.95× the fused arm's device time. The fused
   arm's 2.85× wall-clock lead comes from the host side (16,798 → 47 syncs per step), not from kernel speed. The prototype route was also device-cheaper on
   Granite-H (0.81×) and on the Qwen3 slice (0.87×). On Granite-H the extra launches cost more wall time on this 2-core seat
   than they saved (1.28 → 1.61 s). That is the question grouped-nf4-gemm#459's `dense` route is for. The geometry is
   measured; the dispatch decision needs a full-step reading on a card with a real host.
5. **Granite-3.1 is not expert-bound.** The routed experts are 39 % of its fused step, and the dense GEMMs and elementwise glue
   58 %. The expert runtime cannot reproduce Qwen's relative gain there; that is the model's cost structure, not a missed
   optimisation.
6. **The dequantize-then-GEMM route's sign follows rows per expert.** This is the prototype's device time against `fused` on
   this card, at seq 512:

   | rows per expert | family | device ratio |
   |---|---|---|
   | 64 | LFM2 (top-4 of 32) | 0.71 |
   | 64 | OLMoE (top-8 of 64) | 0.79 |
   | 48 | ERNIE (top-6 of 64) | 0.76 |
   | 48 | Granite-H (top-6 of 64) | 0.81 |
   | 32 | Qwen3 (top-8 of 128) | 0.87 |
   | 24 | Nemotron-H (top-6 of 128) | 1.07 |
   | 16 | Qwen3.6 (top-8 of 256) | 0.97 |

   At 32 or more rows per expert it was cheaper on every family; at 16 to 24 it was not. That is the geometry a measured
   dispatch rule would key on. TC1 amendment 22 read grouped-nf4-gemm#459's `dense` route at 0.651× the fused step on Mixtral
   (top-2 of 8, four times OLMoE's rows per expert at equal tokens) on an RTX 5090. A per-card, per-geometry full-step reading is the next step before
   any default moves. Wall clock is a separate question on a host-bound card: OLMoE's prototype rung was device-cheaper and
   wall-slower on this 2-core seat.
7. **On the hybrid slices, the recurrent kernels are cheap once installed.** `recurrent_ssm` is 1 % of the fused step on both
   the Qwen3.6 and Nemotron-H slices. The routed experts are 49 % and 62 % of those steps; attention, the dense and
   shared-expert GEMMs and the elementwise glue are the rest. These shares describe the slices, not the full models, because a
   slice's mix of layer types differs from its model's (Nemotron-H's first 8 layers hold 3 of its 23 MoE layers). MG1 reads
   the full models.
