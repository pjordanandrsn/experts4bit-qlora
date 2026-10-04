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

Peak VRAM: `keep` costs 0.24 to 1.32 GB over `fused` (OLMoE 6.81 → 7.44 GB, Granite-3.1 4.09 → 5.06, LFM2 6.32 → 7.36,
Granite-H with its kernels 6.94 → 8.26, Qwen3 slice 3.76 → 4.00). No other rung raises peak VRAM by more than 0.07 GB.

**Engagement, every `fused` rung.** Every MoE layer was patched (OLMoE 16, Granite-3.1 32, LFM2 22, Granite-H 40, Qwen3
slice 4), and none was skipped. Every frozen-GEMM dgrad was served by the kernel (`DGRAD_STATS` loop 0). The LoRA delta took the
padded route on every call, and the pinned ring never overflowed. Every RMSNorm was fused, with no fallback calls (OLMoE 65,
Granite-3.1 65, LFM2 61, Granite-H 81, Qwen3 slice 17). One rotary function per family was fused, and none was refused. The
train-GEMM route was `fused`.

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

1. **The Qwen-general optimisations carry over to every family measured.** `fused_pre` takes 1.09 to 1.23× `fused`'s device
   time. Turning them on also removes about nine in ten host syncs: 369 → 35 on OLMoE, 737 → 67 on Granite-3.1, 507 → 47 on LFM2, 920 → 82
   on Granite-H and 93 → 11 on the Qwen3 slice. No family needed a family-specific patch to get them.
2. **Activation retention saves 21 to 25 % of device time on every family, the hybrid included** (14 % on Granite-H while it
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
