# SC2e: 64 slots with decode-graph buckets up to 64 lift e4b `serve_paged`'s capacity ceiling from 4 to 12 req/s on one RTX 5090, with identical serial output; `SLOTS_LICENSED(64, auto)` (lane SC2e of #846; 2026-10-07)

Pre-registration: [`../../sc2/SC2e-PREREG.md`](../../sc2/SC2e-PREREG.md) (#1320, reviewed by the maintainer before it was
registered). The code under test: #1319 (`a99b857e`, `experts4bit_qlora/` tree `4d96736c`), which made
`E4B_PAGED_BUCKETS=auto` capture decode-graph buckets up to `max_seqs`, opt-in. The box ran the registration's merge
commit `3bbcb01b`, whose package tree is the same `4d96736c`.

| run (adertha-receipts commit) | e4b | host (driver; board power limit) | outcome | $ |
|---|---|---|---|---|
| `sc2e-prove-1` (`a573f575`) | — | — | REFUSED by the launcher before any box: the launch script did not set `E4B_RENT_LIVE=1` | 0.000 |
| `sc2e-prove-2` (`7069bae1`) | `3bbcb01b` | one RTX 5090 | **PROVED box=L** on Granite: all four arms on the registered routes and slots, every bucket captured, the burst replayed each arm's widest bucket with no eager step, serial smokes IDENTICAL against s16, the wide steps in every step trace | 0.245 |
| `sc2e-5090-1` (`e9db4ebf`) | `3bbcb01b` | AMD EPYC 7352, 48 threads (610.43.02; **575 W**, 3105 MHz), machine 58411 | **OK**: eight servers (draw 1 s16 → s32a → s64c → s64a, draw 2 reversed), every gate passed | 1.017 |

SC2e total: **$1.262 across 3 receipts**, against the registered guards (proof ≤ $0.85, reading ≤ $3.18) and the lane
ceiling of $8. Every run sat under #846's standing no-ask tier.

**Stack.** e4b `3bbcb01b`; grouped-nf4-gemm v0.42.0 (`b4f93f1c`); torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1.
Qwen3-30B-A3B int4 (SC2's levers, `E4B_PAGED_FUSE_QKV=1`), 2,048 tokens a slot, chunk 512, bulk KV bookkeeping and the
prefill graph at their defaults (engaged on every server). Routes from every server's own `/health`: k19 / k19 / flash,
device grouping on, `seen` K19 above 256 rows. The arms differed only in `E4B_PAGED_MAX_SEQS` and `E4B_PAGED_BUCKETS`:

| arm | `max_seqs` | buckets captured (`/health`) | KV pool (`levers.kv.pool_mib`) |
|---|---|---|---|
| s16 (control, today's server) | 16 | 1, 2, 4, 8, 16 | 1,669.7 MiB |
| s32a | 32 | 1, 2, 4, 8, 16, 32 | 3,339.4 MiB |
| s64c | 64 | 1, 2, 4, 8, 16 (a wider step chains 16-row replays) | 6,638.8 MiB |
| s64a | 64 | 1, 2, 4, 8, 16, 32, 64 | 6,678.8 MiB |

**The board is not SC2's or SC2c's.** Absolute times are comparable only within this box. No comparator ran here, so
nothing below is a position against vLLM or SGLang.

## The outcome by the registered rule

`bench/sc2/sc2e_reduce.py` on the box; re-derived here from the committed receipts with an identical
`verdict_sc2e.json`.

**Gates: all pass.** ROUTES, SLOTS, ENGAGED (including the burst's replay of each arm's widest bucket, no eager decode
step on any server) and PROMPTS on all eight servers; DETERMINISM IDENTICAL; IDENTITY IDENTICAL for s32a, s64c and s64a
against s16 in both draws.

**Attainment** (TTFT ≤ 1.0 s and TPOT ≤ 100 ms), draw 1 / draw 2; the ceiling is the largest rate at ≥ 0.95 in both:

| arm | 1 | 2 | 4 | 8 | 12 | 16 | ceiling |
|---|---|---|---|---|---|---|---|
| s16 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.700 / 0.258 (UNSTABLE) | 0.208 / 0.167 | 0.158 / 0.142 | **4** |
| s32a | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.425 / 0.450 | 0.317 / 0.275 | **8** |
| s64c | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.633 / 0.608 | 0.525 / 0.533 | **8** |
| s64a | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.967 | 0.558 / 0.542 | **12** |

**Predictions:**

| # | prediction | measured | verdict |
|---|---|---|---|
| P1 | decode-only step p50 at the largest bucket: s32a in [12, 20] ms; s64a in [20, 34] ms | s32a 12.84 / 12.89 ms; s64a **18.41 / 18.27 ms** | **REFUTED**, on the fast side (s64a) |
| P1b | s64a's bucket-64 step / s64c's four-replay step ≤ 0.85 | 18.41 / 36.54 = **0.504**; 18.27 / 36.35 = **0.503** | HOLDS |
| P2 | serial p50 TTFT and TPOT of each wide arm within ±5 % of s16 | every ratio in [0.997, 1.009] | HOLDS |
| P3 | s64a's ceiling ≥ 8 | **12** | HOLDS |
| P4 | s32a at 8 req/s ≥ 0.90 in both draws | 1.00 / 1.00 (ceiling 8) | HOLDS |
| P5 | s64a at 12 ≥ 0.50 in both draws; at 16 < 0.95 in one | 1.000 / 0.967; 0.558 / 0.542 | HOLDS |
| P6 | no regression at 1–4 req/s; each wide ceiling ≥ s16's | every pair at 1.00 | HOLDS |
| P7 | VRAM at ready minus s16's: s32a [1,600, 2,400]; s64c [4,900, 5,600]; s64a [4,900, 6,200] MiB; the prefill graph engaged everywhere | s32a **+1,350**; s64c +4,980; s64a **+4,746**; engaged everywhere | **REFUTED**, low (s32a, s64a) |
| P8 | s64a's TPOT p50 at 8 req/s ≤ 40 ms | 18.5 / 20.3 ms | HOLDS |

**Licence: `SLOTS_LICENSED(64, auto)`, `buckets_auto` LICENSED.** Every wide arm was LICENSABLE (no failure listed for
s64a, s64c or s32a); s64a comes first in the registered order. The two misses are both on the safe side:
- **P1** missed because the decode step grows more slowly above 16 rows than the linear fit below 16 assumed
  (census below). The registered consequence of a miss was for a *high* miss; this one is low.
- **P7** missed low: the `auto` arms rose 263–320 MiB *less* than their KV pools did. The serve estimate prices the
  pool, so it over-prices these servers, the safe direction for an estimate-sized default. The registered consequence
  (price the wide buckets' graph pools before any default) was for a high miss and does not apply.

## The census (reported, no bar)

`bench/sc2/sc2e_census.py`, from each server's own step trace, draw 1 (draw 2 within 1 % unless shown).

**Decode-only steps by key.** A one-piece step is keyed by its bucket; a chained step by `<largest>x<pieces>`:

| key | arm | step p50 | device p50 | steps |
|---|---|---|---|---|
| 1 | every arm | 4.33 ms | 3.86 ms | ~16–21k |
| 16 | s16 | 9.56 ms | 8.40 ms | 4,601 |
| 32 | s32a | 12.84 ms | 11.25 ms | 1,669 |
| 32 | s64a | 12.42 ms | 10.87 ms | 550 |
| 64 | s64a | **18.41 ms** | 15.99 ms | 601 |
| 16x2 | s64c | 17.19 ms | — | 423 |
| 16x3 | s64c | 27.20 ms | — | 223 |
| 16x4 | s64c | **36.54 ms** | — | 407 |

- Device time for a chained step is not shown: the trace's events name only the last piece.
- **Above 16 rows each row costs ~0.18–0.21 ms** (16 → 32: +3.3 ms over 16 rows; 16 → 64: +8.9 ms over 48 rows),
  against the ~0.35 ms per bucket row SC2c's fit found below 16. That is P1's miss.
- **One 64-row graph runs at half the time of four 16-row replays.** Each chained replay pays a full pass over the
  step's weights and a host sync, which is consistent with 36.5 ≈ 4 × 9.1.
- `dec_pieces` on s64c: 516 two-piece, 306 three-piece and 568 four-piece steps in draw 1.

**Prefill.** The prefill step is 48.7–49.3 ms with a 40.0 ms forward on every server, and the direct stall per
prefill is 40.5 ms on every server: slots and buckets do not change it.

**Queueing** (draw 1: mean requests in the server / queue wait p50):

| arm | 8 req/s | 12 req/s | 16 req/s |
|---|---|---|---|
| s16 | 13.4 / 299 ms | 14.8 / 3,228 ms | 14.9 / 4,588 ms |
| s32a | 18.7 / 10 ms | 26.7 / 1,025 ms | 27.6 / 2,302 ms |
| s64c | 27.2 / 13 ms | 48.2 / 45 ms | 50.4 / 56 ms |
| s64a | 19.9 / 8 ms | 42.5 / 19 ms | 47.1 / 50 ms |

At 16 slots the server is full from 8 req/s on, and TTFT is queue wait. At 64 slots the queue wait stays under 60 ms
at every rate. s64c then fails on time per token instead: its TPOT p50 at 12 req/s is 48–50 ms against s64a's 30–32
(TPOT p99 54 ms against 37 ms).

**Throughput** (output tok/s, draw 1 / draw 2):

| arm | 8 req/s | 12 req/s | 16 req/s |
|---|---|---|---|
| s16 | 1,002 / 1,096 | 1,093 / 1,073 | 1,073 / 1,089 |
| s32a | 1,085 / 1,177 | 1,350 / 1,327 | 1,338 / 1,381 |
| s64c | 974 / 1,064 | 1,100 / 1,088 | 1,091 / 1,120 |
| s64a | 1,075 / 1,175 | 1,482 / 1,457 | **1,514 / 1,569** |

**Memory at ready** (`nvidia-smi` used / `prefill_graph.free_after_mib`): s16 23,686 / 8,465; s32a 25,036 / 7,115;
s64c 28,666 / 3,485; s64a 28,432 / 3,719 MiB. s64a used 234 MiB less than s64c with 40 MiB more pool. The census does
not separate allocations, so that difference is reported, not explained.

**Busy-loop gaps** between steps: 1.1–1.3 % of busy time on every server.

**Text agreement with s16, no bar** (share of requests with byte-equal streamed text, same seeds; draw 1 / draw 2):

| arm | serial | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| s32a | 1.00 / 1.00 | 0.84 / 0.72 | 0.57 / 0.38 | 0.08 / 0.05 | 0.04 / 0.03 | 0.01 / 0.01 |
| s64c | 1.00 / 1.00 | 0.84 / 0.79 | 0.58 / 0.31 | 0.46 / 0.52 | 0.68 / 0.66 | 0.72 / 0.74 |
| s64a | 1.00 / 1.00 | 0.88 / 0.73 | 0.50 / 0.32 | 0.12 / 0.04 | 0.05 / 0.04 | 0.04 / 0.03 |

Under load, batch composition differs between servers, so agreement below 1 is expected everywhere. The split is
sharp, though: s64c, whose decode pieces stay at ≤ 16 rows, keeps 0.46–0.74 of its text at 8–16 req/s; s32a and s64a,
whose wide steps take `Int4Linear`'s cached bf16 weight above 16 rows and (at 64) the chained tile table above 256
routed rows, keep 0.01–0.12. This says nothing about which arithmetic is better; it is why the `auto` bucket default
waits on a teacher-forced read (the #1320 ruling).

## What it means

- **Slots were what bound e4b at 8 req/s.** With 32 or 64 slots the queue empties and 8 req/s holds in both draws; at
  64 slots with buckets up to 64 the ceiling is 12, three times today's 4, with byte-identical serial output.
- **The wide graph is what reaches 12.** 64 slots on the default list chain 16-row replays and lose at 12 req/s on time
  per token; one 64-row graph runs that step in half the time.
- **What a single request sees does not change.** Serial TTFT and TPOT are within 1 % on every arm.
- **Memory.** 64 slots at 2,048 tokens cost +4.6–4.9 GiB on this card and left 3.4–3.6 GiB free with the prefill graph
  engaged. At the default 4,096 tokens a slot, on a 24 GB card or on a hybrid, 64 slots do not fit: the default must be
  sized by the estimate, as registered.

## Post hoc: the capacity model (descriptive; no rule reads it)

`serve_capacity` (#1134) on each server's own fitted costs and plans
([`sc2e_capacity_check.py`](sc2e_capacity_check.py) →
[`capacity_check.json`](receipts/sc2e-5090-1/capacity_check.json)); chained steps are left out of s64c's fit and the
model chains pieces itself. **42 of 48 cells fall within 0.03 of the measured attainment, and all within 0.075**
(largest: s64c d2 at 8 req/s, model 0.925 against 1.00; s64a d2 at 12, 0.892 against 0.967). The model's first test
above 16 rows reproduces every ceiling. Its fitted per-row cost on s64a is 0.234–0.238 ms (buckets 1–64), against SC2c's
0.35 below 16.

## Next

- **The default flip, as ruled on #1320:** `E4B_PAGED_MAX_SEQS=auto`, the largest of {64, 32, 16} whose serve estimate
  fits the free memory with 1 GiB to spare, on the **default bucket list**. That configuration is s64c here
  (LICENSABLE, ceiling 8) and needs no teacher-forced read: every decode piece stays a ≤ 16-row bucket. The PR carries
  the three "never 64" unit tests, names `E4B_PAGED_MAX_SEQS=16` as the way back, and states this scope.
- **`E4B_PAGED_BUCKETS=auto` as a default** (s64a, ceiling 12) waits on P110's teacher-forced read at buckets 32 and 64,
  registered as its own lane now that P1b HOLDS and s64a is licensed.
- **The decode step**: at 64 rows it is 18.4 ms, 16.0 ms of it on the device. The expert route above 256 rows and the
  bf16 attention path set it; that is the next lane's census.

## Reproduce

- **Rerun:** `SC1_BOX=L bash bench/sc1/sc1_drive.sh` at `3bbcb01b`.
- **Re-reduce:** gunzip the step traces into a scratch copy, then run the reducer and the model check:

  ```bash
  cp -r receipts/sc2e-5090-1/sc2 /tmp/sc2e && gunzip /tmp/sc2e/steps_*.gz
  python bench/sc2/sc2e_reduce.py --dir /tmp/sc2e --out /tmp/sc2e/verdict.json
  python bench/h2h-2026-10-02/sc2e/sc2e_capacity_check.py receipts/sc2e-5090-1/sc2
  ```

  The verdict equals the committed `verdict_sc2e.json`; the model check reads the gzipped traces directly and equals
  `capacity_check.json`. Each run file drops the driver's per-chunk `chunk_gaps_s` (the full record is in the receipt
  store); no rule reads it.
