# The 64-slot decode-step census (#846): where SC2e's bucket-64 step goes, and what each lever can still save

Reported, no bar. $0, from committed receipts only. It registers nothing; the next lane is the maintainer's pick.

Every number below is in `decode64_census.json`, which `decode64_census.py` writes from:
- SC2e `sc2e-5090-1`: the s64a step traces, both draws, and `capacity_check.json`. Code: e4b `3bbcb01b`,
  grouped-nf4-gemm `b4f93f1c` (0.42.0);
- P119 `p119-5090-1`: the d64 eager-twin profile;
- P122 `p122-5090-1`, P124 `p124-5090-4` and P126 `p126-5090-2`: the levers already read at 64 rows;
- #1469's `loc-a4000-1` decode routing trace;
- `bench/sc2/sc2f_basis.py`'s model.

## 1. The bucket-64 step closes

Each served bucket-64 decode step (one piece, decode only) splits exactly by its own CUDA events. The device replay
splits by P119's d64 classes, and an explicit residual covers the rest. Means over the steps, so the parts add:

| | draw 1 | draw 2 |
|---|---:|---:|
| steps (rows, mean) | 601 (51.7) | 670 (50.2) |
| **the step** (p50) | **18.392** (18.411) | **18.273** (18.270) |
| host before the replay (`gpu.dec_prep`) | 0.879 | 0.932 |
| device replay (`gpu.dec_issue − gpu.dec_prep`) | 15.906 | 15.800 |
| host tail (`step_ms − gpu.dec_issue`) | 1.607 | 1.541 |
| left over | 0.000 | 0.000 |

| device replay by class (P119 d64, one profile) | ms |
|---|---:|
| int4 experts (K19) | 6.811 |
| elementwise (holds most of the chained tile table above 256 routed rows) | 2.666 |
| dense GEMM (cuBLAS: the attention projections above 16 rows, the LM head) | 2.543 |
| attention | 1.675 |
| MoE routing | 1.413 |
| norms and glue | 0.324 |
| device-to-device copies | 0.119 |
| other, sampling, other copies | 0.091 |
| **P119's total** | **15.641** |
| **residual** (served replay − P119's eager twins), draw 1 / draw 2 | **+0.265 / +0.160** |

- **P119 can stand for SC2e's replay.** Between SC2e's commit and P119's (`caac7f08`), e4b changed only config defaults
  and opt-ins, on the same grouped-nf4-gemm.
- **The capacity model's `decode_step_at_max_seqs`** (19.87 / 19.71 ms) is its line at 64 rows. The model fits the
  step on the bucket over every decode-only step (4.664 + 0.2376 ms a row; 4.710 + 0.2344). That line runs above the
  measured bucket-64 step, whose steps carry ~51 live rows padded to 64.
- **The host segments** at bucket 64, draw 1 / 2 (ms):
  - `dispatch` 1.203 / 1.141
  - `dec_prep` 0.795 / 0.845
  - `dec_issue` 0.211 / 0.223
  - `dec_mirror` 0.182 / 0.177
  - `retire` 0.140 / 0.142
  - `ops`, `plan`, `dec_ready` and `dec_emit` < 0.06 each
  - `dec_sync` is the wait on the replay.

## 2. The levers, ranked by what they can still save per bucket-64 step

| | lever | at SC2e | saved since, read | **ceiling left** | basis |
|---|---|---:|---:|---:|---|
| (c) | the exposed host gap | 2.486 / 2.473 | 0 | **≤ 2.473** | SC2e traces: host before the replay + host tail |
| (a) | `Int4Linear` above 16 rows | 1.880 | 0.497 (P124) | **≤ 1.050** | P124's K16 time (1.383) − the int4 bytes at the copy bandwidth (0.333) |
| (d) | K19 at 512 routed rows | 6.811 | — | **≤ 0.749** | K19 × (1 − K20's best-plan efficiency at 16 rows, 0.89) |
| (b) | the tile table above 256 routed rows | 3.143 | 2.453 (P122, P126) | **≤ 0.690** | P126's 4-program table (8 programs: 0.290, ineligible on block noise) |

All values are ms per step.

### (c) The host gap

- **It is the GPU's idle time between replays.** The largest piece is `dispatch`: after every step the engine hands
  each live stream its tokens with one `call_soon_threadsafe`, one cross-thread wakeup per stream. That costs about
  23 µs a stream at 64 slots, 1.2 ms a step, in a segment that is 0.02 ms at one row.
- **Two fixes, nested.**
  - E4B_PAGED_DECODE_LOOKAHEAD (P118, opt-in) issues step t+1 before step t is collected, so it hides all of the gap
    for as long as the host work stays below the device replay. It stays below at every bucket in these traces:
    0.48 vs 3.86 ms at 1 row, 1.24 vs 8.06 at 16, 2.49 vs 15.91 at 64.
  - One wakeup a step would remove most of `dispatch` alone, with nothing changing on the device.
- **P118 read the lookahead at one request**, where the gap was 0.21 ms of a 7.2 ms step: ×1.0198 against its 1.02 bar.
  At 64 slots the gap is about 12 times that.

### (a) `Int4Linear` above 16 rows

- **Already read and on by default.** P124 replaced 1.880 ms of cuBLAS with 1.383 ms of K16's int4 tile at 64 rows, and
  the served step read 0.965.
- **What is left is K16's distance from its bytes.** Fused q/k/v plus o at int4-b32 is 0.333 ms at the 1,532 GB/s copy
  bandwidth. K34 found no better K16 plan of 48, so the rest would need a different kernel shape, not a plan.

### (d) K19 at 512 routed rows

- **Distinct experts per layer at 64 rows,** from #1469's routing trace:
  - 64 consecutive tokens of one sequence: 64.1;
  - 64 tokens across its four sequences: 101.2 (p10–p90 98.8–103.5);
  - uniform routing: 125.9.
- **At 101 experts per layer the copy-bandwidth floor is 8.42 ms, above P119's measured 6.81 ms.** Either P119's 64
  wikitext rows touched fewer experts, or reads beat the copy figure. Either way K19 at 512 rows is not visibly above
  its byte floor.
- **The bound is K20's efficiency at 16 rows (0.89).** Below that, only fewer bytes would move K19: fewer distinct
  experts a step, or fewer bits a weight.

### (b) The tile table

- **Already read twice and on by default.** At 64 rows P126's box measured 3.143 ms for one program and 0.690 ms for 4.
  P122's chunked table had first replaced the chained builder.
- **What is left is the 4-program table.** Eight programs reached 0.290 ms but were ineligible on block noise, not on
  speed.

## 3. The top lever in req/s at the knee (`sc2f_basis`'s model)

**Method:**
- The baseline is SC2f's `on` arm in the model: SC2e's s64a costs with P130's 5.18 ms prefill saving.
- Lever (c) subtracts the host-gap fit from the decode costs:
  - draw 1: 0.506 ms + 0.0353 ms a row;
  - draw 2: 0.514 ms + 0.0349 ms a row.
  These are the same least squares on the bucket as the model's own step fit, so the subtraction is consistent.
- 40 seeds a rate; the crossing of 0.95 mean attainment is interpolated.

| | draw 1 | draw 2 |
|---|---:|---:|
| `on`: 0.95 crossed at | 12.95 req/s | 12.55 req/s |
| `on` + (c): 0.95 crossed at | **14.19 req/s** | **13.94 req/s** |
| gain | **+1.24 req/s** | **+1.39 req/s** |
| knee mean attainment (12–14 req/s), `on` → `on` + (c) | 0.953 → 0.983 | 0.954 → 0.981 |

**This is a ceiling, under three assumptions:**
- the lookahead hides all of the host gap at every bucket;
- it adds no device time;
- the host gap does not grow when the device replay shrinks.

## 4. What changed on the 64-row decode step since SC2e

| commit | change | reaches bucket 64 | read at 64 rows |
|---|---|---|---|
| `0c4da09a` | `E4B_INT4_WIDE_TILES=auto` (P122) | yes | P122: served 0.957 |
| `a3527276` | `E4B_ATTN_INT4_WIDE=auto` (P124) | yes | P124: served 0.965 / 0.967 |
| `2384d2c3` | `E4B_INT4_TILE_PROGRAMS=auto` = 4 (P126) | yes | P126: served 0.844 / 0.846 |
| `dc2221a3` | `E4B_PREFILL_LEAN_DISPATCH=auto` (P130's P2) | **yes**, see below | no: P130 timed the prefill replay |
| `8ea97296` | P127 Phase 2: the MoE residual into the experts' combine, per decode bucket | yes | no (bitwise, no speed claim) |
| `1ddcb0ae` | P127 Phase 2 wiring: the router's cast in-kernel, `rope_norm_qk` in the attention folds | yes (folds run to 64 rows) | no (bitwise, no speed claim) |
| `451a3703` | `E4B_FUSE_PREFILL_GLUE=1` for qwen3_moe (P130's P1) | no (adds prefill rows) | — |

**P2 is not prefill-only.** Bucket 64 routes 512 rows, and the lean dispatch applies to every K19 call above 256
routed rows. The `tests/fixtures/smokeflips` record shows `one_launch|lean|gt256` on a 512-row call under the default.

**So the 64-row step on today's defaults is unread.**
- (a) and (b) were read at 64 rows, each in its own box.
- (d)'s call has two bitwise changes no one has timed at 64 rows.
- (c) has no default-on host change since SC2e: the five later `paged_runner` commits are opt-ins, a build fix and ROCm
  vendor checks. Its share of the step grows as the replay falls.
- SC2f's reading runs s64a servers on 0.54.0 with step traces on, so it re-reads the served bucket-64 step at no extra
  cost. A P119-style profile would re-split the device classes.

## 5. Scope against P132

- P132 owns 2–4 rows of one sequence at B = 1. (b) and (d) here are 256–512 routed rows: no overlap.
- **`_tile_table_r1`'s device time a launch scales with rows.** P119's kernels give 9.3 µs at 128 routed rows (d16)
  and 19.4 µs at 256 (d32), with no fixed per-launch device cost visible.
- This census reads P119's own classes from its receipt (e4b 0.50.0, before `rope_norm_qk`), so it needs no class-map
  revision.

## What this census cannot say

- **Today's step.** See section 4.
- **Whether the lookahead keeps its identical tokens and admission behaviour at 64 slots.** P118 read one request.
- **K19's distinct experts on P119's own rows.** P119 recorded no ids.
- **Anything about other cards or models.**

## Reproduce

```
python bench/h2h-2026-10-02/sc2e/decode64/decode64_census.py --check
```

The check recomputes everything (about 10 s, numpy and the package's `serve_capacity`) and compares the result with
the committed JSON byte for byte.
