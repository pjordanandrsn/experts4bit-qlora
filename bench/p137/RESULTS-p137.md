# P137 results — layer-major accumulation under expert offload, RTX 4090: LICENSED

**Verdict: LICENSED, S = 2.365** (`p137-4090-7`, 2026-10-18). On Qwen3-30B-A3B with expert offload and
`enable_fast_train(dgrad=True)`, on an RTX 4090, `layer_major_step` runs the registered training step at 2.365× the
speed of the stock accumulation loop. The bar was S ≥ 1.10. The verdict was re-derived independently twice, and both
re-derivations produced byte-identical `verdict.json` (sha256 `feef3cfd…`).

Registration: [PREREG-p137.md](PREREG-p137.md) (#1718), with Amendments 1 (#1722), 2 (#1730) and 3 (#1734).

## What was run

- **Subject:** e4b `b4366b9a` (0.55.0), the registered library pin; grouped-nf4-gemm 0.46.0 `6b7c2e74`; torch
  2.8.0+cu128; transformers 5.18.0; bitsandbytes 0.50.2.
- **Lane source:** e4b `b3567954` (#1734's merge). The staged runner, box, reducer and fixture were byte-identical to it.
- **Model and setup:** Qwen3-30B-A3B @ `ad44e777`. `load_moe_4bit_streaming(..., offload=True, pin=True)` with all 48
  MoE layers offloaded. Gradient checkpointing is enabled first (`use_reentrant=False`), then
  `enable_fast_train(model, dgrad=True)` patches 48 layers and, by its default since TC1 amendment 64, routes all 48
  decoder layers through e4b's reentrant checkpoint; each arm records `ckpt {layers: 48, mode: reentrant}`.
- **Recipe:** alpaca `train[:400]` (sha256 `68e3d31c…`, staged from the controller), seq 2048, micro-batch 2 × accum 4,
  r 16 / α 16, AdamW 8-bit, lr 2e-4, seed 3407.
- **Steps per arm:** 2 warm and 32 timed.
- **Arms:** A0 is the stock accumulation loop, which runs each micro-batch's forward and backward in turn. L1 is
  `experts4bit_qlora.engines.layer_major.layer_major_step`, which runs one step's micro-batches layer by layer so that
  each offloaded layer is staged once per step instead of once per micro-batch.
- **Order:** one process, one load, then ABBA: A0a L1a L1b A0b. Every arm starts from the same hash-checked adapter
  snapshot. An untimed warm-up of both loops runs before the arms.
- **Box:** one rented RTX 4090 (sm_89, 24 GB, driver 595.71.05), on a Vast.ai verified/secure host: machine 51612,
  instance 56546276, AMD EPYC 7742, 256 threads. Cost $0.745.

## Numbers

| arm | median s/step (32 timed) | stagings, step 0 / steady | peak allocated | held-out at N |
|---|---|---|---|---|
| A0a (stock) | 7.641 | 381 / 380 | 8.41 GB | 1.0787 |
| L1a (layer-major) | 3.231 | 96 / 95 | 7.98 GB | 1.0439 |
| L1b (layer-major) | 3.173 | 96 / 95 | 7.98 GB | 1.0749 |
| A0b (stock) | 7.514 | 381 / 380 | 8.41 GB | 1.0701 |

- **The gain.** S = min(A0a/L1a, A0b/L1b) = min(2.3650, 2.3682) = **2.365**.
- **Self-pairs** (VOID outside [0.97, 1.03]): A0b/A0a 0.9833, L1b/L1a 0.9820.
- **STRICT rungs, all clean:**
  - The warm-up record `{stock 96, layer_major 336}` is identical in all four arms.
  - Step 0 is cold: 381 in both A0 arms, 96 in both L1 arms.
  - Steady stagings are exactly M × (2n − 1) = 380 for A0 and 2n − 1 = 95 for L1.
  - Step 0's per-micro-batch losses are bit-identical across the four arms.
  - Every arm starts from the snapshot, with allocated memory within 5 KiB of the baseline.
  - L1's peak is below A0's.

**Where the gain comes from.** It is consistent with the stagings. Each pass over the 48 offloaded layers, a forward
and a backward, stages 2n − 1 = 95 layers, because the last layer stays resident between the two. The stock loop makes
that pass once per micro-batch, 4 × 95 = 380 per step; layer-major makes it once per step, 95. On this host-bound path
the step time follows the copies: 4× fewer stagings and 2.4× the speed.

## Predictions, scored

- **Q3 FALSIFIED, high.** The prediction was S in [1.25, 2.0]. S = 2.365, and the two blocks agree to 0.13 %.
- **Q5 FALSIFIED (REPORTING).** The held-out losses were L1 1.0594 and A0 1.0744: |Δ| 0.015 against a 0.01 term, with
  L1 lower. Arms of the same loop, from an identical snapshot, seed and data, end 0.031 apart (L1a / L1b) and 0.009
  apart (A0a / A0b) after 34 steps. At this length the term sits inside the run-to-run spread, so it is reported, not
  read. Gradients are a different summation order, not bitwise.
- **Q6 HELD.**
- One A0a step took 18.29 s; the medians are unaffected.

## The path to the reading

- `p137-4090-1`: HARNESS_ERROR, $0.028. The hub/httpx2 decode failed at the data step; Amendment 1 fixed it.
- `p137-4090-2`: ABORTED, $0.974. Each arm re-fused the experts on the host CPU; Amendment 2 put all four arms in one
  load.
- `p137-4090-3`: refused at $0 before the create.
- `p137-4090-4`: VOID, $0.630. The 8-step L1 self-pair was 0.900. Amendment 3 fixed 32 timed steps and a warm-up of
  both loops, before any data; the median and the band were unchanged.
- `p137-4090-5` and `-6`: NOT_RUN at the launcher's pre-flight ($0.009, $0.046). One host was stuck loading; the other
  had HF CDN bandwidth of 6.4 MB/s.
- `p137-4090-7`: this reading.

Lane total: $2.432.

## Scope of the licence (registered consequence)

The licence covers exactly this configuration:
- `layer_major_step` with `enable_fast_train(dgrad=True)` and expert offload;
- Qwen3-30B-A3B;
- an RTX 4090 (sm_89);
- this recipe.

It is not a licence for `LAYER_MAJOR=1`, which is the CLI trainer's knob. That trainer does not take the fused path,
so the knob runs the bitsandbytes reference loop. No default changes.

## Receipt

[`receipts/p137-4090-7/`](receipts/p137-4090-7/SHA256SUMS) holds the four arm records, `verdict.json`, `summary.txt`,
`versions.txt`, `forensics.txt`, the logs and the launcher's teardown proof. `SHA256SUMS` lists them. The data slice
is identified by its sha256 and not committed.
