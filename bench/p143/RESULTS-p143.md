# P143 results: the paged FP8 attention decode's split cap (e4b#1313)

**Status:** closed. **FLIP** on the binding read (Amendment 3). `E4B_ATTN_SPLIT_CAP` defaults to 16 on `qwen3_moe` with
the flip PR that cites this record. Claim: `e4b.serve.p143.attn-split-cap.qwen3.5090.2026-10-19`.

P142's per-name census put the paged FP8 attention decode at the top of the addressable B = 1 step: 12.7 %, 48 launches
of about 11.6 µs. The library plans its split count from the block table's **capacity**, not the live length. At B = 1
on 4096-token slots that is 32 splits, most of them empty at served contexts. P143 asked whether fewer splits help, and
whether the gain survives the served step.

| run | what | cost | verdict |
|---|---|---|---|
| `p143-5090-1` | step 1, the microbench | $0.008 | HARNESS_ERROR (a forward `rewind`; strike 1; [Amendment 1](PREREG-p143.md)) |
| `p143-5090-2` | step 1, rerun | $0.029 | **GO_FIXED**: 16 splits beat 32 at every context |
| `p143a2-5090-1` | Amendment 2: kernel contract, kernel timing, served ABBA + SANE | $0.406 | **NO_FLIP**: g1 1.0297 under a noisy self-pair |
| `p143a3-5090-1` | Amendment 3: the binding served read, 8 arms x 5 reps + SANE | $0.479 | **FLIP**: g1 1.0442, g16 0.9920 |

Total $0.922 on four RTX 5090 launches. Each receipt's public copy is under [`receipts/`](receipts/), with a README and
`SHA256SUMS`. The launcher's receipts and ledger rows are in the receipt store. The maintainer re-derived every verdict
from the store byte for byte: step 1 on #1759, A2 on #1778, A3 on #1788.

## Step 1 (`p143-5090-2`): the microbench

The served call was e4b's own `Fp8PagedKV` at Qwen3-30B-A3B's geometry, timed as one CUDA graph of 48 launches per arm.
Two passes in opposite orders, contexts {256, 640, 1024, 2048, 4096} x {plan, 4, 8, 16, 32} splits.
- **16 splits:** save **0.171 ms** a step at 640 tokens (bar 0.131), and are 0.040 ms faster at 4096. They beat 32 at
  every context.
- **8 splits:** save 0.179 ms at 640 but lose 0.149 ms at 4096.
- **The per-context best** is an upper bound, and it adds only 0.008 ms over a fixed 16.

The full grid is in the [receipt's README](receipts/p143-5090-2/README.md). This was data seen before step 2's design.
The 0.171 ms is a microbench, never the quoted gain.

## The lever

- **grouped-nf4-gemm 0.47.1** (#558; tag `38b30888`) makes `decode_split_plan` a pure function and adds an optional
  `split_cap` to `fp8_paged_decode_attention`. The default is byte-identical.
- **e4b#1768** adds `E4B_ATTN_SPLIT_CAP` (0..32), resolved per family.
  - The KV hands the kernel `split_cap` only on the fp8 decode path, never on speculative verify and never over an
    explicit `n_split`.
  - `info["attn_split_cap"]` records what each captured bucket planned.
- **At Qwen3's geometry on the 5090,** the plan is 32 / 32 / 23 / 11 / 5 at buckets 1 / 2 / 4 / 8 / 16. With the cap it
  is 16 / 16 / 16 / 11 / 5: only buckets 1, 2 and 4 move.

## Amendment 2 (`p143a2-5090-1`): NO_FLIP

**Kernel contract.** Fifteen cells, B {1, 2, 4} x five contexts. The cap's error against an fp64 reference was 0.9976 to
1.0000 times the plan's. At every context ≤ 1024, and at B 4 / 2048, the two were **bitwise identical**.

**Kernel timing.** The cap was faster in every B 2 / 4 cell, by 0.09 to 0.20 ms per 48 launches.

**Served read.**
- **Speed:** W1 pairs 1.0405 and 1.0297, so g1 = **1.0297**, missing 1.03 by 0.0003. g16 was 1.0113.
- **Noise:** the W1 self-pair S1b/S1a read **0.9845**, 1.55 % of drift and about 50x the miss. P141's form would have
  called the read NOISY. A2 registered no such rung, so NO_FLIP stands on the record.
- **Quality:** SANE read bias 0.0, argmax 1.0 and KL 0.0. That is exact rather than a miss, because SANE's contexts sit
  where the cap is bitwise-neutral.

Two pre-data rulings shaped A2:
- **The SANE pass's KV is built at the server's 4096 tokens.** P115 Phase D's pass builds 656 tokens, where the plan
  is 11 and a cap of 16 cannot bind.
- **W16 is a speed no-regression check, not an identity check.** The prefill ramp decodes W16's first steps at 1-4
  rows, on the capped buckets.

## Amendment 3 (`p143a3-5090-1`): FLIP, binding

The rule was stricter than A2's and binding: no Amendment 4 on the flip, and a NO_FLIP would have closed the lever.
- **Arms:** eight, ABBA twice, at five reps.
- **FLIP rule:** the minimum of all four W1 pairs, unrounded, ≥ 1.03, and g16 ≥ 0.99.
- **NOISY:** any W1 self-pair outside [0.985, 1.015] would have VOIDed the read and counted as strike two.

| | S1a/S0a | S1b/S0b | S1c/S0c | S1d/S0d | min |
|---|---|---|---|---|---|
| W1 | 1.0442 | 1.0445 | 1.0448 | 1.0447 | **1.044184** |
| W16 | 0.9920 | 1.0052 | 1.0003 | 0.9992 | **0.991994** |

- **Noise:** W1 self-pairs 0.9991 to 0.9995. A2's drift did not recur at five reps.
- **Speed:** one request ran 255.6 -> 267.0 tok/s, **+4.4 %**. That matches the kernel prior: 0.161 ms of a ~3.9 ms
  step.
- **Sixteen requests:** unchanged (geomean 0.9992), as the 5-split plan predicts.
- **Quality and tokens:** SANE exact again, and tokens identical between arms.
- **Host:** the card ran at a 450 W power limit. That bounds the absolute rates, not the ratios.

## Predictions, graded

| amendment | prediction | result |
|---|---|---|
| A2 | every contract ratio in [0.98, 1.02] | **hit** (0.9976-1.0000) |
| A2 | B 2/4 no slower, faster at 640 and 1024 | **hit** |
| A2 | SANE \|bias\| ≤ 0.002, argmax ≥ 0.99 | **hit** (0.0, 1.0) |
| A2 | W1 tokens differ from S0's within 160 | **missed**: identical (bitwise-neutral at ≤ 1024) |
| A2 | g1 1.025-1.045 | **hit** (1.0297) |
| A2 | g16 within 0.99-1.01 | **missed** (1.0113) |
| A2 | verdict FLIP | **missed** (NO_FLIP) |
| A3 | every W1 pair in [1.03, 1.05] | **hit** (1.0442-1.0448) |
| A3 | every self-pair within [0.99, 1.01] | **missed** at W16 (S0b/S0a 0.9859); hit at W1 |
| A3 | SANE 0.0 / 1.0 | **hit** |
| A3 | verdict FLIP | **hit** |

## What it changes, and what it doesn't

- **The default:** `qwen3_moe` takes `E4B_ATTN_SPLIT_CAP=16`, with the flip PR. `E4B_ATTN_SPLIT_CAP=0` restores the
  library's plan.
- **What moves:** buckets 1, 2 and 4. At served contexts ≤ 1024 no output changes, only speed.
- **Mixtral** (8 KV heads; B 1-2 move under the cap) and every other family keep the plan until a read of their own.
- **Not read here:** long contexts beyond 4096, other cards, and speculative verify (which never takes the cap).
