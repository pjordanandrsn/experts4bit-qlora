# P114 — results: **READ**. On one rented RTX 5090, `bnb.matmul_4bit` costs 1.6–2.0× native bf16's GPU energy per op on the one-projection harness, and the fused 4-bit MoE forward's J/token falls to 0.063 of batch 64's at batch 4096

Registration: `bench/p114/PREREG-p114.md` (#1197, merged `726aed43`, the launch commit). Work item: #1133, decision 1 (iii).
Authorization: #1133 issuecomment-6001804551.

**Verdict by `bench/p114/p114_reduce.py`: READ.** Every spread is ≤ 0.047 against the registered 0.10.

**Registered consequence.**
- `e4b.train.energy-honest.5090.2026-10-05` supersedes `e4b.train.energy-honest.scoped-a2000` and
  `e4b.train.energy-honest.a2000-bnb0502.2026-10-04`.
- Every quote is restated from the medians below, with their scope.

## The run

- **`p114-5090-1`:** one NVIDIA GeForce RTX 5090 on `vast:verified-secure`.
  - power limit **500 W** (max 600 W), driver **570.133.07**, PCIe gen 4 x16;
  - an AMD EPYC 7B12 host, 503 GB RAM.
- **Stack:** experts4bit-qlora 0.48.0 @ `726aed43`, bitsandbytes 0.50.2, torch 2.8.0+cu128 (`versions.txt`).
- **Cost:** $0.148, with teardown proven (`teardown-proof.json`, instance absent at 20:02:05Z). The receipt is in the
  private record.
- **Before any pass:**
  - The card check passed, and `power.draw` read 6.81 W at rest.
  - The gate passed (`logs/gate.log`). Dequant and `matmul_4bit` agree to 6.2e-3 of max |out| at M = 1 and exactly at
    M = 32 and 512, so bitsandbytes takes `matmul_4bit` through dequantize-then-matmul above one row.
  - The rule's self-test read 10/10.
- **Six passes**, 19:51:08–20:01:41Z, each after 60 s of rest.
  - Each pass's `nvidia-smi pmon` shows only the pass's own process.
  - The card held no compute process before any pass.
  - Idle power was 48–50 W before the projection passes and 65–66 W before the Part B passes.

## The reading (medians of three passes; one RTX 5090, 500 W power limit, driver 570.133.07)

Total J/op relative to native bf16 on one OLMoE-dims gate_up projection (out 2048, in 2048), eager op loop:

| workload | `bnb.matmul_4bit` | dequantize → `linear` | spread (`matmul_4bit` / dequant) |
|---|---:|---:|---:|
| decode M=1 | **1.748** | **3.293** | 0.006 / 0.047 |
| prefill M=512 | **1.601** | **1.539** | 0.013 / 0.017 |
| train fwd+bwd M=32 | **1.965** | **1.405** | 0.005 / 0.007 |

Per-arm medians behind those ratios (`p114_verdict.json`, `cells`):

| workload | arm | ops/s | W | µJ/op |
|---|---|---:|---:|---:|
| decode | native | 60,474 | 201.2 | 3,335 |
| decode | dequant | 12,696 | 137.7 | 11,010 |
| decode | `matmul_4bit` | 16,827 | 98.3 | 5,844 |
| prefill | native | 30,483 | 416.5 | 13,689 |
| prefill | dequant | 11,304 | 236.7 | 21,074 |
| prefill | `matmul_4bit` | 10,087 | 221.0 | 21,914 |
| train | native | 2,684 | 97.0 | 36,154 |
| train | dequant | 1,946 | 98.7 | 50,732 |
| train | `matmul_4bit` | 1,372 | 97.3 | 70,904 |

**Part B**, the fused 4-bit MoE forward, J/token relative to batch 64 (medians; spreads ≤ 0.013):

| batch (tokens) | tok/s | W | µJ/token | vs batch 64 |
|---:|---:|---:|---:|---:|
| 64 | 15,128 | 106.0 | 7,009 | 1.000 |
| 256 | 60,907 | 138.0 | 2,262 | 0.322 |
| 1,024 | 239,332 | 167.2 | 699 | 0.100 |
| 4,096 | 672,400 | 298.0 | 442 | **0.063** (≈16× lower) |

**Part A** (allocation only): on this 32 GB card both the bf16 (12.9 GB) and the 4-bit (3.2 GB) expert footprints
allocate. That is recorded, not compared. The memory-wall statement stays the A2000's allocation fact, per the
registration.

## Against the predictions

| prediction (direction only) | outcome |
|---|---|
| `matmul_4bit` near 1 at decode | **refuted: 1.748.** It draws less than half native's power (98.3 vs 201.2 W) but runs at 0.28 of native's op rate, so its energy per op is higher |
| `matmul_4bit` above 1 at prefill, close to the dequant arm | held: 1.601 against 1.539 (the gate shows the same arithmetic above one row) |
| both 4-bit arms above 1 at train | held: 1.965 and 1.405 |
| dequant above 1 throughout | held: 3.293, 1.539 and 1.405 |
| Part B J/token falls as batch grows | held: 1.000 → 0.322 → 0.100 → 0.063 |
| verdict READ (NOISY at decode the likeliest alternative) | READ; the largest spread was the dequant decode cell's, 0.047 |

## What it supersedes, and what it does not say

- **The two A2000 rows become `superseded`**, pointing at `e4b.train.energy-honest.5090.2026-10-05`.
  - They stand as measured on their card.
  - Their numbers are not compared with these: this reading replaces them as the number to quote, on a rented card.
- **Scope:**
  - one RTX 5090 at a 500 W power limit, one projection, an eager loop, total J only;
  - not grouped MoE execution, not CUDA-graphed serving, not other cards.
- **The host is part of every launch-bound cell.** The decode cells are host-bound: about 60k native ops/s is the Python
  loop's rate.

## Reproduce

The registered reducer, run on the committed receipts, prints exactly these lines (and writes a `p114_verdict.json`
identical to the box's):

```
python bench/p114/p114_reduce.py bench/p114/receipts/p114-5090-1 /tmp/p114_verdict.json
```

```
P114 matmul_4bit / native J/op (median of 3): decode 1.748, prefill 1.601, train 1.965
P114 dequant / native J/op (median of 3): decode 3.293, prefill 1.539, train 1.405
P114 Part B J/token vs batch 64 (median of 3): 64 1.000, 256 0.322, 1024 0.100, 4096 0.063
P114 spreads: matmul_4bit decode 0.006, matmul_4bit prefill 0.013, matmul_4bit train 0.005, dequant decode 0.047, dequant prefill 0.017, dequant train 0.007, partb 64 0.000, partb 256 0.013, partb 1024 0.012, partb 4096 0.010
P114_VERDICT READ on NVIDIA GeForce RTX 5090
```

## Receipts

`bench/p114/receipts/p114-5090-1/` holds:
- `runs/`: every pass's harness output, `pmon`, pre-pass state and time stamps;
- `p114_verdict.json`, `summary.txt`, `forensics.txt` and `versions.txt`;
- `logs/`: the gate, the reducer and pip;
- `outer.log`, the teardown proof and `SHA256SUMS`.
