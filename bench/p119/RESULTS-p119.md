# P119 — results: **READ**. Where the 64-slot server's steps spend their device time: at 64 decode rows the int4 experts (K19) take 44 % of a 15.6 ms step and the tile table above 256 routed rows another 2.9 ms; the 512-token prefill is 41.2 ms of device time, 48 % int4 experts and 26 % elementwise work (one RTX 5090, 2026-10-08)

Registration: `bench/p119/PREREG-p119.md` (#1350, merged `3a1ff60`; amendment 1, #1352, merged `caac7f0`, the launch
commit). Issue: #846. Descriptive only: no gate, licence, default or claim.

Code under test:
- e4b 0.50.0 at `caac7f0`;
- grouped-nf4-gemm 0.42.0 at `b4f93f1`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`: SC2e's served stack (int4 experts on 48 layers, int4 RTN attention, the folds,
  fused q/k/v), bulk KV bookkeeping, device grouping; NF4 arena baked on the box.

**Verdict by `p119_reduce.py`: `READ`.** No VOID condition fired: the commits, revision and stack are the registered
ones; every decode bracket profiled 8 steps on its registered split, every bucket `eager: capture=False`, bulk KV on;
both prefill brackets ran in their mode (ON was not refused); every bracket saw device kernels.

## The reading (`p119-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 595.71.05) on an AMD EPYC 7C13 (256 threads, 1,007 GiB RAM), Vast instance 54864544.
**Cost:** $0.827. Teardown proven at 15:04:53Z.

**Timeline:** install and tripwire from 14:51:12Z; premise 7 passed, none skipped (14:52:15Z); fetch until 14:59:36Z;
bake until 15:00:39Z; the box (load 159 s with the int4 repack, brackets 54 s); lane complete 15:04:52Z.

### Decode: device ms per step by class (`torch.profiler`, eager twins)

| class | d16 | d32 | d64 | d64x4 |
|---|---:|---:|---:|---:|
| int4 experts (K19) | 4.61 | 5.87 | **6.81** | 18.36 |
| dense GEMM (bf16 cuBLAS; LM head) | 0.56 | 1.98 | 2.54 | 2.25 |
| dense int4 (`Int4Linear` ≤ 16 rows) | 0.78 | — | — | 3.13 |
| elementwise | 0.51 | 0.53 | **2.67** | 2.06 |
| attention | 0.79 | 1.11 | 1.67 | 3.16 |
| MoE routing | 0.64 | 1.13 | 1.41 | 2.56 |
| norms and glue | 0.27 | 0.29 | 0.32 | 1.07 |
| copies, sampling, other | 0.06 | 0.07 | 0.21 | 0.23 |
| **total** | **8.22** | **10.99** | **15.64** | **32.83** |

- **The marginal cost per decode row** is 0.155 ms from 16 to 64 rows: 0.173 from 16 to 32 and 0.145 from 32 to 64.
- **Bucket 32 swaps the attention projections' kernel.** `Int4Linear` above 16 rows runs its cached bf16 weight
  on cuBLAS: dense int4 0.78 ms gives way to dense GEMM 1.98 ms.
- **Bucket 64 crosses 256 routed rows** (64 rows × top-8). The tile table is then built by the chained builder: scatter,
  gather, sort, scan and small integer kernels. Grouped by name after the fact, that is 2.87 ms per step, against
  0.93 ms for the one-launch `_tile_table_r1` at 32 rows. It is most of the elementwise jump.
- **Four 16-row pieces cost 2.10× one 64-row piece.** Each piece repeats the expert weight reads (K19 18.36 ms against
  6.81).

### Prefill: device ms per 512-token prefill (forward and flush)

| class | OFF | ON (`E4B_PAGED_LAST_LOGITS=1`) |
|---|---:|---:|
| int4 experts | 19.90 | 19.95 |
| elementwise | 10.59 | 10.59 |
| dense GEMM | 6.65 | 5.66 |
| MoE routing | 2.10 | 2.10 |
| attention (flash) | 1.50 | 1.50 |
| copies, sampling, other | 0.47 | 0.47 |
| **total** | **41.21** | **40.28** |

- **The elementwise 10.6 ms** is many small kernels: grouped by name after the fact, the RMSNorm arithmetic about 1.5 ms
  (pow, mean, rsqrt and add, 193 calls each) plus its multiplies about 2.0, dtype and layout copies 2.1, `index_copy`
  1.1, concatenation 0.6, the routed-row scatter and gather, and RoPE.
- **Dense GEMM** is the attention projections on bf16 cuBLAS (4.83 ms) and the LM head (1.36 ms, one call). With
  last-logits on, the head runs on one row and the prefill saves 0.94 ms.
- **Device-to-device copies are 4.0 per layer.** P107's ~131 per layer predate bulk KV bookkeeping (#1200).

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: at 64 rows the int4 expert class is the largest, ≥ 40 % | **yes** (6.81 ms, 44 %) |
| Q2: `d64` within ±15 % of SC2e's 16.0 ms device time | **yes** (15.64 ms, −2.3 %) |
| Q3: `d64x4` / `d64` ≥ 1.8 | **yes** (2.10) |
| Q4: marginal device cost per row 16 → 64 ≤ 0.20 ms | **yes** (0.155) |
| Q5: `p512_off` in [30, 42] ms | **yes** (41.21) |
| Q6: D2D copies per layer in `p512_off` ≤ 10 | **yes** (4.0) |
| Q7: LM head ≥ 1.5 ms at 512 rows and ≤ 0.3 ms at 1 row; ON saves ≥ 1 ms | **no**: the head is 1.36 ms in the forward and ON saves 0.94 ms |
| Q8: the box ≤ 15 minutes | **yes** (3.5 min) |

**The standalone head bracket undercounted.** The profiler recorded 2 of the 5 calls at every row count (0.4 per
requested call), so its `head_ms` (0.53 ms at 512 rows, 0.15 at 1 row) divide two calls' time by five. Per recorded
call it reads 1.33 ms at 512 rows and 0.38 ms at 1 row: an estimate from the recorded calls, not a recovery of the
unrecorded ones. It agrees with the head's kernel inside the prefill forward (1.36 ms) and with the bandwidth floor for
reading its 622 MB weight once. Q7 misses on either reading. The decode and prefill brackets record whole calls per
step at the multiples their layers imply (48, 144, 193), and `d64` agrees with SC2e's served device time; that is
consistent with complete capture, not proof of it.

**The frozen classes behaved as registered on the reading.** On the NF4 proof, `_gemm_nf4_grouped` fell into dense
GEMM; the raw kernel tables in `box.json` carry every name either way.

## What it names (the registered consequences)

- **Decode (Q1 held):** the next decode lane is K19 above 256 routed rows, 6.81 ms of the 15.64 ms step. Beside it, the
  tile table above 256 routed rows (2.87 ms against 0.93 below) and `Int4Linear` above 16 rows (2.05 ms on cuBLAS at 64
  rows) are the next largest levers.
- **Prefill (Q5 held, device-bound):** the int4 experts (19.9 ms) and the elementwise work (10.6 ms) are the levers.
  Prefill speed is the TTFT lane's; this census is its prior, not its reading.
- **Last-logits (Q7 missed):** its prefill saving is small, 0.94 ms of 41.2 (2.3 %).

Each is its own registration.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p119-prove-1` | HARNESS_ERROR | $0.069 | the box called `len()` on `kv_layers()`'s int; amendment 1 |
| `p119-prove-2` | OK, PROVED | $0.133 | every bracket on Granite; its tables are not a reading |
| `p119-5090-1` | **OK, READ** | $0.827 | the reading |

**The lane cost $1.029**, inside its $3.00 ceiling.

**Receipts** are in `receipts/p119-5090-1/`, with `SHA256SUMS`: `box.json` (every bracket's per-kernel table),
`verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`, the logs and the teardown proof. These
committed copies are the public record of the reading.

**Re-derive:**
```bash
python3 bench/p119/p119_reduce.py --dir bench/p119/receipts/p119-5090-1 --out /tmp/v.json --e4b-sha caac7f0881ff410682aedd1326279b00e282bac8
```
The output equals the committed `verdict.json` byte for byte on Python 3.9 and 3.14.
