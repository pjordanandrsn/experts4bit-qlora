# P117 — results: **AT_PARITY**. Decoding 32 or 64 rows in one graph (`E4B_PAGED_BUCKETS=auto`) costs no measurable quality against decode in pieces of at most 16 rows: teacher-forced on SC2e's int4 stack, W32 reads +0.0006 nats, W64 −0.0032 and W64pad −0.0033, inside the 16-row arithmetic's own neutral perturbations (one RTX 5090, 2026-10-08)

Registration: `bench/p117/PREREG-p117.md` (#1335, merged `9ea9528`, the launch commit). Issue: #846. Follows SC2e
(#1320, read #1333) and the ruling on #1320.

Code under test:
- e4b 0.48.0 at `9ea9528`;
- grouped-nf4-gemm 0.42.0 at `b4f93f1`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`: SC2e's served stack (int4 experts on 48 layers, int4 RTN attention on 96
  projections, the three T=1 folds, fused q/k/v), NF4 arena baked on the box.

**Verdict by `p117_reduce.py`: `AT_PARITY`.** All three subjects pass P110's bar (bias ≤ B_floor + 0.01 = 0.0125;
spread ≤ 2 × S_floor = 0.0296). No VOID condition fired:
- the commits and the revision are the registered ones; every arm has its windows (64; W64pad the first 48);
- engagement is exact: decode attention calls = 127 steps × 48 layers × pieces per step (R 24,384; half 48,768; W32
  12,192; W64 6,096; G64 none); device grouping on in every pass; every bucket `eager: capture=False`, G64's `graph`;
- every arm ran its registered split: R, rep, chunk and rev 508 bucket-16 steps; half 1,016 bucket-8; W32 254
  bucket-32; W64 and the mutant 127 bucket-64; W64pad 127 bucket-64 steps with 2,032 padding rows; G64 127 replays;
- **FUNCTION:** G64's captured 64-row replay emitted W64's tokens at all 8,128 positions;
- `mutant_scale` fails the bar.

## The reading (`p117-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 595.91.07) on an AMD Ryzen 9 9950X (32 threads, 123 GiB RAM), Vast instance
54778599. **Cost:** $0.914. Teardown proven at 05:32:21Z.

**Timeline:** install and tripwire from 04:55:46Z; premise 7 passed, none skipped (04:57:55Z); fetch until 05:25:53Z;
bake until 05:26:57Z; the box (load 91 s with the int4 repack, ten passes 227 s); lane complete 05:32:20Z.

| arm | what it is | bias vs R (nats) | spread | se | max \|d\| | mean KL vs R | argmax agree |
|---|---|---:|---:|---:|---:|---:|---:|
| rep | R again | +0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 1.0000 |
| half (floor) | buckets 1–8: eight 8-row pieces | −0.00251 | 0.01449 | 0.00224 | 0.04477 | 1.1e-2 | 0.9583 |
| chunk (floor) | prompts prefilled in 256-token chunks | −0.00066 | 0.01478 | 0.00246 | 0.07201 | 1.3e-2 | 0.9537 |
| rev (floor) | windows bound and decoded in reverse | +0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 1.0000 |
| **W32** | **two 32-row pieces** | **+0.00059** | **0.01520** | 0.00233 | 0.06048 | 1.1e-2 | 0.9543 |
| **W64** | **one 64-row piece** | **−0.00317** | **0.01449** | 0.00223 | 0.04803 | 1.1e-2 | 0.9583 |
| **W64pad** | **48 windows padded to 64** | **−0.00331** | **0.01397** | 0.00250 | 0.04803 | 1.1e-2 | 0.9606 |
| mutant_scale | W64 with the decode softmax scale halved | +1.14650 | 1.14650 | 0.03620 | 2.07153 | 1.27 | 0.5764 |

R's mean NLL is 2.1787 over the 64 windows (1.192 to 3.168).

- **The wide paths sit inside the floor.** Each subject's bias is within 1.5 standard errors of zero, and on the
  better side for W64 and W64pad. Their spreads (0.014–0.015) and KL against R (1.1e-2) match the floor's draws.
- **What moves the arithmetic is the piece's row count, not row order.** 64 windows reverse into the same four 16-row
  sets, so `rev` changes only the order inside each piece and the slots; it read bit-identical to R, as did the
  repeat. Halving the piece (half) or widening it (W32, W64) moves single windows by up to 0.06 nats.
- **The instrument can fail:** a halved decode scale reads +1.15 nats.
- **Step times, for the record** (eager padded steps on this host, not serving speed, which is SC2e's): R 157 ms per
  step as four 16-row pieces, W32 73, W64 54; G64's replay 16.9 ms.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: R repeats bit for bit | **yes** |
| Q2: floor \|bias\| ≤ 0.003, S_floor in [0.008, 0.020]; rev no longer bit-identical | bias **yes** (0.0025), S_floor **yes** (0.0148). rev **no**: bit-identical. The prediction's reason was wrong: 64 windows reverse into the same four 16-row sets, so rev varies only row order and slots, which leave the arithmetic unchanged |
| Q3: W32, W64, W64pad bias in [−0.003, +0.003], spread ≤ 0.016 | spread **yes** (≤ 0.0152). Bias **yes** for W32 (+0.0006), **no** for W64 (−0.0032) and W64pad (−0.0033): outside by 0.0002–0.0003 nats, on the better side |
| Q4: FUNCTION holds | **yes** (8,128 positions, 0 differ) |
| Q5: mutant_scale's bias above +0.3 | **yes** (+1.147) |
| Q6: engagement exact | **yes** |
| Q7: the box ≤ 30 minutes, the int4 repack included | **yes** (5.3 min) |
| Q8: AT_PARITY | **yes** |

## The registered consequence (AT_PARITY)

AT_PARITY licenses `E4B_PAGED_BUCKETS=auto` as a default beside `E4B_PAGED_MAX_SEQS=auto` (#1334). The licence rests on
two readings, both on this model and card: SC2e's speed (64 slots with buckets up to 64 serve 12 req/s, against 8 on
the default list and 4 at 16 slots) and this lane's quality.

What follows, in a separate PR:
- **The default.** `E4B_PAGED_BUCKETS` unset reads `auto`; the explicit list `1,2,4,8,16` restores today's buckets.
  `E4B_PAGED_MAX_SEQS=auto` then chooses among 64, 32 and 16.
- **The docs.** Greedy outputs under load change at the bf16 level, at no measured quality cost.
- **Scope, in one line:** read on Qwen3-30B-A3B int4 on an RTX 5090; other models get the wide buckets on the strength
  of this read, not their own.
- **The register.** The row `e4b.serve.p117.wide-bucket-quality.qwen3.5090.2026-10-08` carries W64's bias, from the
  verdict file. It lands with this read.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p117-prove-1` | OK, PROVED | $0.154 | the whole box on Granite (40 windows, 32 positions); its verdict (not a reading) AT_PARITY, FUNCTION 1,240 positions 0 differ, mutant +3.04 |
| `p117-5090-1` | **OK, AT_PARITY** | $0.914 | the reading |

**The lane cost $1.068**, inside its $3.00 ceiling.

**Receipts** are in `receipts/p117-5090-1/`, with `SHA256SUMS`:
- `box.json` (every arm's per-window NLL, KL and agreement, and every pass's engagement);
- `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the install, premise, fetch, bake and box logs;
- the teardown proof.

The launcher's receipts and ledger rows are in the receipt store, adertha-receipts `19d91e4` (`p117-prove-1`) and
`2bb857e` (`p117-5090-1`).
