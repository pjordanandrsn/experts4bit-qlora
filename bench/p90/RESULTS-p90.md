# P90 — results: **LICENSED** (under amendment 1). K21 takes gpt-oss-20b's B=16 decode step on an RTX 5090 from 22.51 to 13.09 ms (×0.581), and the MXFP4 store's KL from the reference falls (0.00192 → 0.00147 nats)

Registration: `bench/p90/PREREG-p90.md` (#817, `ba43ce8`), amendment 1 (#820, `147a9c6`). Issue: #564. The route
under test is `E4B_MXFP4_GROUPED_SMALLM=1` (#816) over grouped-nf4-gemm's K21 (#422) with its masked K tail (#425), at
`4cc831c`.

**Verdict by `p90_reduce.py --speed-dir p90-5090-2 --quality-dir p90-h100-1`: LICENSED.**

```
P90_VERDICT LICENSED B=16 13.086 vs 22.511 ms (x0.581); KL 0.00192 -> 0.00147; B=1 SLOWER (x1.075)
```

| | OFF | ON | ratio |
|---|---:|---:|---:|
| B=16 step, ms (two draws) | 22.52 / 22.51 | 13.08 / 13.09 | **0.581** (bar 0.90) |
| B=16 aggregate, tok/s | 710.6 | 1,223.1 | 1.72× |
| B=1 step, ms (two draws) | 5.81 / 5.81 | 6.25 / 6.25 | 1.075, **SLOWER** |
| KL from the reference, nats/token (5,757 tokens) | **0.001922** | **0.001466** | −0.000456 |
| top-1 agreement | 0.9814 | 0.9840 | +0.0026 |

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---:|
| `p90-prove-1` | PROVED on an RTX 5090: premise 3 passed, K0, K21 + K16 contracts 23 passed | $0.0449 |
| `p90-5090-1` | **VOID**: both KL arms crashed in the reference (below). Speed recorded descriptively | $0.1177 |
| `p90-5090-2` | the speed half (amendment 1): PART_OK | $0.1560 |
| `p90-h100-prove-1` | PROVED on an H100 NVL: premise 3 passed, K0, contracts 23 passed | $0.1391 |
| `p90-h100-1` | the quality half (amendment 1): PART_OK | $0.4926 |
| **total** | | **$0.9503** of the $8.00 ceiling |

Every teardown is proven (`vast-destroy`, HTTP 200, instance absent).

## The VOID read and amendment 1

`p90-5090-1` read **VOID** because both KL arms exited rc 1. The failure was in the **reference**, not the route:
- gpt-oss-20b's bf16 dequant reference is about 40 GB and does not fit a 32 GB card;
- transformers offloaded part of it to the CPU, and its grouped-MM fallback then raised a device mismatch.

P44-b had scored this reference on an H100 NVL. I registered the KL arms on a 5090 without checking the reference's
size.

Amendment 1 (#820) split a reading into two runs:
- a speed run on the RTX 5090, with the registered speed arms;
- a quality run on an H100 NVL (P44-b's card class), with the registered KL arms.

Both runs hold the premise and K0, and the verdict combines them under the registered rule. Run 1's speed was recorded
descriptively only (×0.584 at B=16, ×1.074 at B=1). The two readings below are fresh.

## Speed (`p90-5090-2`)

- **Host.** One RTX 5090 (driver 580.173.02, power.limit 575 W) on an Intel Xeon E5-2680 v4 (machine 149299). Image
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- **Software.** e4b 0.37.8 at `147a9c6` and grouped-nf4-gemm at `4cc831c`, with torch 2.8.0, triton 3.4.0 and
  transformers 5.16.1.
- **Premise.** `tests/test_k21_row_exact_gpu.py` passed 3 of 3 on the card:
  - a token's K21 rows are bit-equal alone (T = 1) and inside a B=16 step, through gpt-oss's epilogue;
  - T = 1 matches the fp32 dequant oracle;
  - the T = 1 route captures and replays bit-equal.
- **Routes, from the censuses** (per decode step), all as registered:
  - B=16 ON: 48 K21 calls and no NF4 grouped GEMM;
  - B=16 OFF: 48 NF4 grouped GEMMs;
  - B=1 ON: 48 K21 calls and no GEMV;
  - B=1 OFF: 48 MXFP4 GEMVs.
- **Where the B=16 step goes now** (ON census, 12.72 ms of kernel time per step):
  - K21: **7.98 ms** (63 %), against the NF4 grouped GEMM's 17.86 in OFF;
  - bf16 cutlass GEMMs (the attention projections, router and head): 2.41;
  - elementwise: 0.74;
  - index kernels: about 0.5;
  - attention: 0.46.

## Quality (`p90-h100-1`)

- **Host.** One H100 NVL (driver 595.71.05, 400 W) on an AMD EPYC 9V84 (machine 145679), same image and software.
- **Premise and K0.** The premise passed 3 of 3 on the card. K0's controls all passed. The reference agreed with
  itself decode-vs-prefill at 0.00051 nats, so control (i) chose decode on both sides.
- **The rows** (store_r12, 200 committed prompts, 5,757 scored tokens each):

| arm | KL mean | KL p95 | top-1 | general / technical / code / longctx |
|---|---:|---:|---:|---|
| OFF (the licensed store: split-K GEMV at T = 1) | 0.001922 | 0.006784 | 0.98141 | 0.00163 / 0.00140 / 0.00141 / 0.00236 |
| ON (K21 at T = 1) | **0.001466** | 0.005034 | **0.98402** | 0.00119 / 0.00128 / 0.00094 / 0.00184 |

- **Instrument check.** OFF reproduces P44's licensed row (0.0019) within its band.
- **ON is lower in every stratum.** The weights are the same exact MXFP4 bytes either way. The GEMV quantizes
  activations to int8 per 32-block, and K21 keeps them bf16; that is the likely reason ON reads lower (inferred).

## Against the predictions

| prediction | outcome |
|---|---|
| B=16 ON/OFF 0.55–0.70, LICENSED | **0.581**, LICENSED |
| B=1 SLOWER 1.03–1.30 | **1.075**, SLOWER, so the default covers T > 1 only |
| KL(ON) ≤ KL(OFF), about 0.0015–0.0019; top-1 within ±0.001 | KL **0.00147** ≤ 0.00192 held; top-1 **+0.0026**, outside the ±0.001 I expected, in the good direction |
| OFF reproduces the census step (22.5 ms) | 22.51, held |

## An observation on K22 and K24 (descriptive)

In-model, K21 reads 7.98 ms per step. K24's bench read it at 7.62 on recorded routing, 4.5 % apart. The same benches
read the served NF4 kernel 17 % under its census in both K22 and K24, which is what voided them. So the instrument gap
appears specific to the NF4 grouped kernel, not to the routing. That is inferred from these two numbers, not tested.

## What follows

1. **A default.** `E4B_MXFP4_GROUPED_SMALLM` becomes `auto` for T > 1, in its own PR citing this row:
   - device-grouped MXFP4 decode rows above T == 1 take K21 when the kernel package carries it with its masked tail;
   - T == 1 stays on the GEMV, because B=1 reads SLOWER;
   - `1` keeps routing T == 1 too.
2. **The rest of gpt-oss's B=16 step.**
   - K21 itself reads at about 48 % of K24's MXFP4 byte floor (3.81 ms on the recorded routing, 1,521 GB/s copy).
     That floor is the bench's, not this census's, but the gap makes K21 the next kernel lever.
   - The bf16 attention projections (2.41 ms) have no licensed quantized route on this family.

## Receipts

- [`receipts/p90-5090-2/`](receipts/p90-5090-2/) (speed): the timed JSONs, the four censuses, the premise and K0 logs,
  the part report, summary, forensics, versions and the teardown proof.
- [`receipts/p90-h100-1/`](receipts/p90-h100-1/) (quality): both KL receipts and logs, the premise, K0 and the part
  report.
- [`receipts/p90-5090-1/`](receipts/p90-5090-1/): the VOID run.
- [`receipts/verdict-p90-5090-2+p90-h100-1.json`](receipts/verdict-p90-5090-2+p90-h100-1.json): the combined
  verdict.
- Each run directory has a `SHA256SUMS`.
