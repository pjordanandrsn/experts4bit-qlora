# P87 read — VOID by the rule. The speed arms still answer the question: on an RTX 5090, K19 is 1.03× at B=16 and 1.24× slower at B=1

Registered in [`PREREG-p87.md`](PREREG-p87.md) (#805, `9b5654e`). Issue: experts4bit-qlora#564.

**Verdict: VOID** (`p87_reduce.py`: `K8 off: missing or failed`).

The calibrated K8 build ran on an Intel Xeon E5-2698 v4 host. It needed 1,120 s for its first calibration chunk;
P85's AMD host needed 360 s. It hit its 3,600 s per-arm alarm after 3 of its 5 chunks and dumped no pack, so the K8
OFF and ON arms never ran. The lane exited rc 142, the alarm's signal, and the launcher recorded `HARNESS_ERROR`.

**Every check the speed arms are subject to passed.** The reducer applies them before the K8 check:
- the premise held on the card;
- both draws of every arm agree within 3 %;
- engagement held at both batch sizes: ON ran K19 96 times per step and the experts' 96 GEMV calls left the GEMV; OFF
  ran no K19.

So the numbers below are measured and valid as descriptions. They are not a registered verdict. No register row is
written from a VOID read.

**What they say: K19 is not the 5090 win.** At B=16 the step is 0.971× the GEMV's; the bar to license a default was
≤ 0.95, and I predicted 0.75–0.85. At B=1 it is 1.236× slower. The quality read would not have changed the outcome:
LICENSED needs both quality and speed, and the speed fails.

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---|
| `p87-prove-1` | **PROVED**, lane rc 0. RTX 5090 (sm_120, driver 595.58), Intel Xeon Platinum 8347C. Install and tripwire OK; **the premise held** (3 passed); K19's and K16's contract tests compiled on the card (20 passed); egress 35.4 MB/s | $0.0329 |
| `p87-5090-1` | **refused before any rental**: the receipt store's HEAD was a local commit not yet pushed (the proof's receipt). Pushed, recorded, relaunched | $0.0000 |
| `p87-5090-2` | **the reading: VOID.** RTX 5090 (sm_120, driver 595.71, 575 W), Intel Xeon E5-2698 v4. Premise held (3 passed); all eight speed arms ran; the build was cut by its alarm. Destroyed 06:34:19Z, absent | $0.9522 |
| **total** | | **$0.9851** of the $3.00 ceiling |

## The premise, on the card

`tests/test_k19_row_exact_gpu.py` passed on both 5090s (3 passed in 20.71 s and 26.00 s):
- a token's K19 rows are bit-equal alone and inside a B=16 step;
- K19 matches the fp32 oracle at T == 1;
- the captured T == 1 route replays bit-equal to eager.

So the design question, whether a B=1 K8 stands for B=16's arithmetic, is answered yes on sm_120. That is the
lane's one confirmed piece.

## Speed (graph-replay window, `step_ms_clean`)

| arm | draw 1 | draw 2 | median | ON / OFF |
|---|---:|---:|---:|---:|
| B=16 OFF (int8 GEMV) | 12.0307 | 11.9956 | **12.013** | |
| B=16 ON (K19) | 11.6669 | 11.6674 | **11.667** | **0.971** |
| B=1 OFF | 4.2764 | 4.2778 | **4.277** | |
| B=1 ON | 5.2808 | 5.2919 | **5.286** | **1.236** |

The OFF arms reproduce P86's steps on the same host class (12.06 and 4.27 ms; P86's reading also ran on an
E5-2698 v4).

## Where the time went (P42 replay census, 8 replays, ms per step)

**B=16:**

| kernel | OFF | ON | Δ |
|---|---:|---:|---:|
| `_gemv_int4_b32` (experts) | 7.000 (96 calls) | — | −7.000 |
| `_gemm_int4_b32_grouped_smallm` (K19) | — | 6.495 (96) | +6.495 |
| `_reduce_partials` | 0.384 | — | −0.384 |
| `_quant_x_rows` | 0.345 | — | −0.345 |
| `_tile_table_r1` (the tile build) | — | 0.498 (48) | +0.498 |
| gather / scatter / elementwise glue | 0.117 | 0.415 | +0.298 |
| everything else | 4.531 | 4.466 | −0.065 |
| **kernel total** | **12.377** | **11.874** | **−0.503** |

**B=1:**
- K19 takes 1.773 ms, against the experts' share of the GEMV, 0.928 (the GEMV's count halves from 192 to 96 per
  step: the attention projections stay on it).
- The tile build (0.069) and the gather/index/elementwise kernels that grew (0.337) add 0.41 ms, against 0.28 ms of
  quantize and reduce removed.
- Kernel total: 4.284 → 5.256.

Three readings:
1. **Kernel for kernel, K19 is only 1.08× the GEMV on the 5090** (6.50 vs 7.00 ms). Against the int4-b32 byte floor at
   the measured routing (4.89 ms; P57/P86), K19 runs at about **75 %** and the GEMV at about 70 %. Marlin MoE runs at
   about **94 %** of its own floor (4.78 ms; P86). The 5090's expert matmul is not won by grouping rows into tensor-core
   tiles alone.
2. **Grouping costs about 0.8 ms per step.** The tile build is about 10 µs per layer (0.50 ms per step), and the
   gather, unsort and scatter glue another 0.30 ms. That nearly cancels the 0.73 ms of reduce and quantize that K19
   removes. The OFF route builds no grouping at all, so this cost is new.
3. **At B=1, every expert gets a one-row tile**, so the MMA wastes 15 of every 16 rows. K19 is then 1.91× slower than
   the GEMV, which is efficient at one row.

## The registered predictions

| prediction | read |
|---|---|
| B=16 ratio 0.75–0.85 | **0.971 — refuted** |
| B=1 NEUTRAL (within ±3 %) | **1.236 SLOWER — refuted** |
| \|ΔK8\| < 0.003 nats | not read |
| the build equals OFF to the bit | not read |

**Why the speed prediction was wrong.** It came from the A2000 probe, where K19 beat the GEMV 1.89× on the same
recorded routing. That probe's own README said the ratio does not transfer, and the house rule is that the A2000 is a
correctness testbed, never a timing one. I predicted from it anyway.

The 5090's GEMV is 11.4× faster than the A2000's (6.98 vs 79.3 ms per step), about twice the cards' bandwidth ratio.
So on the A2000 the GEMV ran far below its own floor, and K19 beat a weak baseline. The right first step was a kernel
microbench on the target card, against the target card's floor, before an end-to-end lane.

**Why K8 did not run.** The registration allowed any CPU vendor because every comparison is on one box. It sized the
2.5 h guard and the build's 3,600 s alarm from P85's reading, whose AMD host finished the first calibration chunk in
360 s. The calibration is CPU-heavy: this Broadwell host needed 1,120 s for the same chunk, about one third of the
speed, and three chunks filled the hour. A quality lane that builds
a calibrated pack needs either a host-speed floor or a pack built once and reused.

## What follows (the registration's NOT_FASTER branch, applied descriptively)

- **K19 stays opt-in.** No default changes. `E4B_INT4_GROUPED_SMALLM` remains an instrument.
- **No re-run of P87 as registered.** Its speed bar cannot be met by this kernel on this card, so its quality arms
  would buy nothing.
- **The next lane is a kernel microbench on the 5090, before any end-to-end lane.** On P60's recorded B=16 routing, at
  Qwen3's expert shapes:
  - K19's plan space (BLOCK_N, KC, warps, stages) against the served GEMV and the byte floor;
  - Marlin MoE (`fused_marlin_moe`, vLLM 0.30.0) on the same routing, in its own venv, as K15 did for the dense
    projections;
  - the tile build alone.

  It costs about $0.10–0.20 of GPU time. An end-to-end lane follows only if a kernel beats the GEMV by ≥ 1.3× there.
- **The grouping glue is a lever whichever kernel wins.** 0.8 ms per step of tile build and gather/scatter. It is
  folded into nothing today.

## Receipts

[`receipts/p87-prove-1/`](receipts/p87-prove-1/) and [`receipts/p87-5090-2/`](receipts/p87-5090-2/), byte-identical to
the receipt store's (`adertha-agents` receipts store `21616a2`). For each run:
- summary, forensics, versions, the premise result;
- for the reading, also: the eight step receipts, the four censuses, `verdict.json`, the build's log (the calibration
  chunks, then the alarm) and `verify_artifact`'s refusal (no manifest);
- `census_diff.txt`, generated here by the census parser from the four censuses;
- `SHA256SUMS`.
