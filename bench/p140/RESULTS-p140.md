# P140 results — P137's layer-major gain replicated on an RTX 5090: LICENSED

**Verdict: LICENSED, S = 2.311** (`p140-5090-1`, 2026-10-18). On Qwen3-30B-A3B with expert offload and
`enable_fast_train(dgrad=True)`, on an RTX 5090, `layer_major_step` runs the registered training step at 2.311× the
speed of the stock accumulation loop. The bar was S ≥ 1.10. The maintainer re-derived the verdict independently and got
a byte-identical `verdict.json` (sha256 `20bd7656…`).

P137 read 2.365 on an RTX 4090 (`e4b.train.p137.layer-major-offload.qwen3.4090.2026-10-18`). The gain replicates across
card and host.

Registration: [PREREG-p140.md](PREREG-p140.md) (#1746). It is P137's kit at its Amendment 3, by named substitutions;
every STRICT term is P137's.

## What was run

- **Subject:** e4b `b4366b9a` (0.55.0), the registered library pin; grouped-nf4-gemm 0.46.0 `6b7c2e74`; torch
  2.8.0+cu128; transformers 5.18.0; bitsandbytes 0.50.2.
- **Lane source:** e4b `5bcae060` (#1746's merge). The staged runner, box, reducer and fixture were byte-identical to it.
- **Recipe and arms:** P137's. Qwen3-30B-A3B @ `ad44e777`, all 48 MoE layers offloaded and patched. Gradient
  checkpointing is enabled first, then routed by `enable_fast_train` through e4b's reentrant checkpoint on all 48
  layers. Seq 2048, micro-batch 2 × accum 4, r 16 / α 16, AdamW 8-bit, lr 2e-4, seed 3407. ABBA in one process after
  one load and an untimed warm-up of both loops; 2 warm and 32 timed steps per arm.
- **Box:** one rented RTX 5090 (sm_120, 32 GB, driver 615.71.09) on a Vast.ai verified/secure host: machine 34018,
  AMD Ryzen 7 5700X3D, 16 threads, pinned host-to-device ceiling 22.80 GB/s. The load took 65.1 s. Cost $0.489.

## Numbers

| arm | median s/step (32 timed) | stagings, step 0 / steady | peak allocated | held-out at N |
|---|---|---|---|---|
| A0a (stock) | 6.017 | 381 / 380 | 8.41 GB | 1.0708 |
| L1a (layer-major) | 2.597 | 96 / 95 | 7.98 GB | 1.0728 |
| L1b (layer-major) | 2.602 | 96 / 95 | 7.98 GB | 1.0826 |
| A0b (stock) | 6.014 | 381 / 380 | 8.41 GB | 1.0690 |

- **The gain.** S = min(A0a/L1a, A0b/L1b) = min(2.3167, 2.3113) = **2.311**.
- **Self-pairs:** 0.9994 (stock) and 1.0018 (layer-major), against the [0.97, 1.03] band.
- **STRICT rungs, all clean:**
  - the warm-up record `{stock 96, layer_major 336}` is identical in all four arms;
  - step 0 is cold (381 and 96);
  - steady stagings are exactly 380 and 95;
  - step 0 is bit-identical across the four arms;
  - every arm starts from the snapshot;
  - the checkpoint is reentrant on 48/48 layers in every arm;
  - layer-major's peak is below the stock loop's.
- One A0a step took 10.80 s; the medians are unaffected.

## Predictions, scored

- **Q3 HELD.** S is inside the a-priori band [1.8, 3.0].
- **The host-link model:**
  - A0 = C + 4T and L1 = C + T fit at T = 1.139 s and C = 1.461 s, an effective rate of 32.3 GB / T = 28.3 GB/s.
  - That is above the box's 22.8 GB/s pinned ceiling, so some copy overlaps compute.
  - On the 4090 the same fit gave T = 1.459 s and C = 1.743 s.
  - Both the copy-bound and the compute-bound part came down on the 5090.
- **Q5 HELD this time (REPORTING).** The held-out losses are L1 1.0777 and A0 1.0699, |Δ| 0.0079 against the 0.01
  term.
- **Q1, Q2 and Q4 HELD.**

## A note the load time made

`P140_LOAD` was 65.1 s on this host, against 2,078 s on `p137-4090-7`'s AMD EPYC 7742 host, for the same model and
source. The 30-minute-plus load seen in P137 is a property of that host, not of the loader. That is recorded here as an
observation; it is not part of P140's question.

## Scope of the licence

The licence covers exactly this configuration:
- `layer_major_step` with `enable_fast_train(dgrad=True)` and expert offload;
- Qwen3-30B-A3B;
- the RTX 5090 class (sm_120);
- P137's recipe.

From the CLI trainer, that is `OFFLOAD_EXPERTS=1 LAYER_MAJOR=1`, which implies `FAST_TRAIN=1` (#1744). No default
changes.

## Receipt

[`receipts/p140-5090-1/`](receipts/p140-5090-1/SHA256SUMS) holds the four arm records, `verdict.json`, `summary.txt`,
`versions.txt`, `forensics.txt`, the logs and the launcher's teardown proof, listed in `SHA256SUMS`. The data slice is
identified by its sha256 and not committed.
