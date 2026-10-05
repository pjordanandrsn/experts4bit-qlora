# SC2c: bulk KV bookkeeping cuts e4b's per-prefill stall about 5× and serial TTFT 3.8×, with identical output; the capacity ceiling rises from 2 to 4 req/s, and the knob is licensed as a default (lane SC2c of #846; 2026-10-05)

Pre-registration: [`../../sc2/SC2c-PREREG.md`](../../sc2/SC2c-PREREG.md) (#1132, reviewed by the maintainer before it was
registered; amendment A1, #1154, pins torch 2.8.0). The lever: `E4B_PAGED_BULK_KV=1` (#1131, `ed686648`). It does a
request's KV bookkeeping in bulk:
- the prompt's flush into the FP8 pool;
- the first graphed decode's block claims;
- the slot resets.

Per request on Qwen3-30B-A3B that takes ~13.5k launches issued by the engine thread down to 66, with every pool byte,
block table and length identical. The census behind the lane is
[`../../stall-census-2026-10-05/README.md`](../../stall-census-2026-10-05/README.md).

| run (adertha-receipts commit) | e4b | host (driver; board power limit) | outcome | $ |
|---|---|---|---|---|
| `sc2c-prove-1` (`1f518e20`) | — | — | REFUSED by the launcher before any box (`E4B_RENT_LIVE` unset) | 0.000 |
| `sc2c-prove-2` (`e9e69f0f`) | `1249cc45` | Intel Core Ultra 9 285K, 24 threads (595.91.07) | PROVED box=H, but on torch **2.14.1**+cu130: box H was missing from the harness's torch pin. Per A1 this proves the code on a newer stack, not the registered one | 0.082 |
| `sc2c-prove-3` (`3185e8aa`) | `ecf84a95` | same host class (595.91.07) | **PROVED box=H** on the registered stack: routes (incl. `seen`), both knobs engaged, OFF/ON serial smokes identical, each step trace decomposes | 0.106 |
| `sc2c-5090-1` (`ffa52ab4`) | `ecf84a95` | AMD EPYC 7B13, 256 vCPU (595.84; **575 W**, 3135 MHz), machine 145701 | **OK**: four servers (draw 1 OFF→ON, draw 2 ON→OFF), every gate passed | 1.270 |

SC2c total: **$1.458 across 4 receipts**, all under the registered guards (proof ≤ $1.95, reading ≤ $3.23). Every run
sat under #846's standing no-ask tier.

**Stack.**
- e4b `ecf84a95`, which is #1154's merge on main.
  - The code under test is `serve_paged`, its engines, the KV pool and the step trace. They are byte-identical to the
    reviewed `ed686648` (tree `88452f16`).
  - The package tree differs (`5d55a43c`), in training (`fast.py`, `train.py`, `chunked_lm_loss.py`), `serve_recipe.py`
    and `serve_capacity.py`. The launch record states this, and #846 disclosed it.
- grouped-nf4-gemm v0.41.0 (`dc8f94ab`).
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1.
- **Routes**, read from every server's own `/health`: k19 / k19 / flash, device grouping on, neither pin in the
  environment. `prefill_routes.seen` shows every expert GEMM above 256 rows on `int4_k19`.
- **The arm.** Both arms ran e4b_int4 with SC2's levers on Qwen3-30B-A3B, and both had the first-chunk prefill graph at
  `auto` (SC2b's licensed default, engaged on every server). `E4B_PAGED_BULK_KV` was the only difference.

**The board is not SC2b's.** This 5090 runs uncapped at 575 W. SC2 and SC2b ran on machine 45511, capped at 400 W. So
absolute times here are not comparable across lanes, and **the OFF arm is not a re-read of SC2b's ON arm.** It runs the
same configuration on a faster board.

## The outcome by the registered rule

`sc2c_reduce.py` re-derives `verdict_sc2c.json` byte-identically from the committed run files (see Reproduce).

| gate | result |
|---|---|
| ROUTES | pass on all four servers: k19 / k19 / flash, device grouping on, both pins null, chunking 512 / 512, `seen` on `int4_k19\|gt256` |
| ENGAGED | pass. The prefill graph replayed once per prompt with no eager chunk on every server. OFF: `kv_bookkeeping` 532 / 508 per-layer flushes and as many per-layer readies, 0 bulk. ON: **508 bulk flushes, with the claims taken at the flush (`ready_at_flush` 508)**, 0 per-layer. The early check after warm-up passed on all four |
| PROMPTS | pass: every request reported 512 prompt tokens |
| DETERMINISM | IDENTICAL: OFF draw 1 serial against its repeat, 24 requests, 17,701 characters |
| **IDENTITY** | **IDENTICAL on both draws:** every request's streamed text byte-equal, OFF against ON (24 requests each; 17,701 and 15,204 characters) |

Medians; draw 1 / draw 2. TTFT and TPOT are p50s.

| arm | serial TTFT | serial TPOT | attainment at 1 / 2 / 4 / 8 req/s (d1, d2) | ceiling |
|---|---|---|---|---|
| OFF | 161 / 156 ms | 4.48 / 4.48 ms | 1.00, 1.00 / 0.95, 1.00 / 0.21, 0.65 (UNSTABLE) / 0.08, 0.07 | **2** |
| ON | **42.7 / 41.8 ms** | 4.22 / 4.23 ms | 1.00, 1.00 / 1.00, 1.00 / **1.00, 1.00** / 0.73, 0.38 (UNSTABLE) | **4** |

| prediction | verdict | detail |
|---|---|---|
| P1 bucket-controlled stall per prefill, ON / OFF ≤ 0.6 in both draws | **HOLDS** | 0.190, 0.194 (OFF 0.211 / 0.199 s → ON 0.040 / 0.039 s) |
| P2 serial TTFT OFF / ON ≥ 1.4 in both draws | **HOLDS** | 3.777, 3.736 |
| P3 serial TPOT ON / OFF in [0.95, 1.05] | **REFUTED** | 0.943, 0.944: ON was *faster* than the band allowed (see below) |
| P4 ON's ceiling ≥ 2 req/s | **HOLDS** | 4 |
| P5 ON's ceiling ≥ 4 req/s | **HOLDS** | 4 (registered as a coin flip) |
| P6 no regression | **HOLDS** | ON ≥ OFF − 0.05 at every rate and draw, and ON's ceiling 4 ≥ OFF's 2 |
| **Licence** | **DEFAULT_LICENSED** | every gate passed, P6 held, and TTFT OFF / ON was 3.78 and 3.74, both ≥ 1.10 |

**What the licence means**, as registered. A separate PR makes `E4B_PAGED_BULK_KV` default to `1`, keeping `0` as the
escape. That PR also carries one `/health` `kv_bookkeeping` engagement read each (correctness, not speed) on:
- a hybrid model (a pool-layer subset);
- gpt-oss (attention sinks).

**Scope:** speed was read on Qwen3-30B-A3B int4, `serve_paged`, one uncapped 5090 and 512-token prompts. Equivalence
elsewhere is the tests' job (whole-pool bitwise comparisons, mixed geometry, hybrid layer subsets, partial tail
blocks). Speed elsewhere is unread.

**Why P3 failed.** The registered basis was that the decode step is untouched. That held:

| decode step p50, by bucket | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| OFF (d1) | 4.14 ms | 5.30 | 6.46 | 7.09 | 9.44 |
| ON (d1) | 4.15 ms | 5.29 | 6.44 | 7.16 | 9.11 |

The basis missed where OFF's first-decode block claims land. They are 44–48 ms per prompt, they run in the first
decode step, and so they fall *inside* TPOT. Spread over a serial request's ~170–180 output tokens they add ~0.25 ms per
token, which is the whole 4.48 → 4.22 ms gap. ON takes the claims at the flush (`ready_at_flush`), before the first
token. The prediction is reported as refuted. The licence does not read P3.

## The census: where the stall went (reported, no bar)

From each server's own step trace (`sc2c_census.py`); p50 per prefill step unless noted.

| | OFF d1 | OFF d2 | ON d1 | ON d2 |
|---|---|---|---|---|
| prefill step | 170.5 ms | 162.7 | **45.7** | **45.3** |
| host `pf_flush` | **158.4** | **151.8** | 2.1 | 2.1 |
| host `pf_sync` (waiting on the GPU) | 0.1 | 0.1 | **36.7** | **36.7** |
| the step's decode (`dec_sync`) | 8.3 | 7.7 | 6.3 | 6.0 |
| `plan` | 1.6 | 1.3 | 0.2 | 0.2 |
| forward, device time | 38.5 | 38.5 | 38.6 | 38.6 |
| bookkeeping per prompt: flush / first-decode claims | 161.0 / **47.7** | 153.2 / **44.1** | 2.3 / 0.8 | 2.3 / 0.9 |
| direct stall (step − decode-only step of its bucket + claims) | 210.6 | 198.7 | 40.1 | 40.2 |
| request-level fit, bucket-controlled (s per prefill; R²) | 0.211 (0.998) | 0.199 (0.998) | 0.040 (0.991) | 0.039 (0.991) |
| request-level fit, two-regressor (s; R²) | 0.253 (0.990) | 0.242 (0.991) | 0.083 (0.945) | 0.087 (0.953) |

**The decomposition closes.** In every server the direct stall measured from the step trace agrees with the stall
fitted from the request trace within 1 ms. There is no unexplained residual.

**The registered inference, against the measurement.** The registration called these splits inferred from 5090
numbers, not measured:

| quantity | inferred | measured | |
|---|---|---|---|
| forward's device time | near ~42 ms (P107) | 38.5–38.6 ms | close |
| flush (host) | ≥ ~105 ms | 152–158 ms | larger than the lower bound |
| first-decode claims | ~55 ms | 44–48 ms | somewhat smaller |

- **OFF:** the forward is hidden under a host flush four times its length. The GPU finishes when the host does
  (flush done 152–159 ms).
- **ON:** the prefill step is now the forward. The host waits on it (`pf_sync` 36.7 ms), plus the step's own decode.
- Per host-issued op: the flush is ~18 µs (8,688 ops in ~155 ms) and the claims ~10 µs (4,608 in ~46 ms).

**Memory.** `nvidia-smi` at ready read 23,686 of 32,607 MiB on all four servers. The bulk flush's transient bound
(216 MiB at 2,048 tokens) is counted in the prefill graph's headroom on ON. After the graph, 8,426 MiB stayed free in
both arms. The peak under load was not recorded.

## What it means

**The stall is gone from bookkeeping.** What a prefill now costs the running decodes is its forward: 40 ms, against
200–210 ms.
- **Under load:** at 4 req/s, ON's TTFT p50 was 52–54 ms, against OFF's 0.60–4.9 s. Its TPOT p50 was 8–10 ms, against
  25–29 ms.
- **At 8 req/s:** output throughput was 1,022 / 1,129 tok/s, against 479 / 548.

**The ceiling doubled, not quadrupled toward 8.** At 8 req/s ON still misses the SLO (TTFT ≤ 1 s and TPOT ≤ 100 ms). It
fails on TTFT: p50 0.27 s and 1.23 s, p99 1.8 s and 2.7 s. TPOT stays at 13 ms. Requests queue for one of the 16 slots.
The capacity model below names what binds there.

**Beside SC2b (descriptive, different board).** SC2b's ON arm on a capped 400 W board had a ceiling of 1, with 0.90 /
1.00 attainment at 2 req/s. The same configuration here (OFF) reached 0.95 / 1.00 on the uncapped board, so its
ceiling is 2. The board accounts for that difference. The knob accounts for ON's 4. No position against vLLM, SGLang or
llama.cpp comes from SC2c, and SC2's comparator rows stand.

## Post hoc: the capacity model (descriptive; no rule reads it)

`serve_capacity` (#1134) is run with each server's own step costs on that server's own plans
([`sc2c_capacity_check.py`](sc2c_capacity_check.py) →
[`capacity_check.json`](receipts/sc2c-5090-1/capacity_check.json)). Model / measured attainment:

| server | prefill step / first decode / decode base + per row (ms) | 1 req/s | 2 | 4 | 8 |
|---|---|---|---|---|---|
| OFF d1 | 160.4 / 47.7 / 4.13 + 0.54 | 1.00 / 1.00 | 0.84 / 0.95 | 0.17 / 0.21 | 0.09 / 0.08 |
| OFF d2 | 153.6 / 44.1 / 4.18 + 0.51 | 1.00 / 1.00 | 1.00 / 1.00 | 0.39 / 0.65 | 0.07 / 0.07 |
| ON d1 | 39.5 / 0.8 / 4.13 + 0.35 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.70 / 0.73 |
| ON d2 | 39.5 / 0.9 / 4.12 + 0.37 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.23 / 0.38 |

- **Fit:** 12 of 16 cells within 0.03. ON's ceiling is reproduced (4). OFF's is not: the model gives 1, the
  measurement 2, because of the model's 0.84 at 2 req/s in draw 1.
- **Where it misses:** at the knees, where it is pessimistic: −0.11 (OFF d1, 2 req/s), −0.26 (OFF d2, 4 req/s,
  UNSTABLE), −0.15 (ON d2, 8 req/s, UNSTABLE). On SC2b's traces it was within 0.10, so its error at a knee is
  ~0.1–0.26 in either direction.

**What binds at 8 req/s, under the model.** These rows are exploratory and $0. Each uses ON draw 1's costs on both
draws' seeds. Decode cost above 16 rows is extrapolated linearly, and memory is not modelled.

| change from ON | attainment at 8 req/s (d1, d2) | 12 req/s |
|---|---|---|
| none | 0.70, 0.34 | 0.21, 0.17 |
| `max_seqs` 32 | 0.84, 0.69 | 0.30, 0.28 |
| `max_seqs` 64 | **1.00, 1.00** | 0.63, 0.60 |
| decode step × 0.7 | **1.00, 1.00** | 0.26, 0.23 |
| prefill step × 0.5 | 1.00, 0.87 | 0.23, 0.22 |
| `max_seqs` 32 + decode × 0.7 | 1.00, 1.00 | 0.55, 0.48 |

At 8 req/s, slots and decode-step time bind. The forward comes third. These are model rows, not measurements.

## Next

- **The default flip** (registered as a separate PR). It sets `E4B_PAGED_BULK_KV` default `1`, with `0` as the escape,
  and carries the hybrid and gpt-oss `kv_bookkeeping` engagement reads. No default changes before this read merges.
- **The next lever toward 8 req/s** is slots and decode-step time, not prefill:
  - the model's largest single lever is `max_seqs` 16 → 64;
  - its memory cost (KV arena, decode-graph buckets above 16) is unread;
  - each needs its own registration.
- **The forward's 38.6 ms** is now the whole per-prefill stall. A chunked prefill interleaved with decode would hide it
  behind decode rather than shrink it. It ranks below slots and decode under the model.

## Reproduce

- **Rerun:** `SC1_BOX=H bash bench/sc1/sc1_drive.sh` at `ecf84a95`.
- **Re-reduce:** gunzip the step traces into a scratch copy, then run the reducer:

  ```bash
  cp -r receipts/sc2c-5090-1/sc2 /tmp/sc2c && gunzip /tmp/sc2c/steps_*.gz
  python bench/sc2/sc2c_reduce.py --dir /tmp/sc2c --out verdict_sc2c.json
  ```

  The result is identical to the committed `verdict_sc2c.json`.
- **The post-hoc check:** `python bench/h2h-2026-10-02/sc2c/sc2c_capacity_check.py receipts/sc2c-5090-1/sc2` (it reads
  the `.gz` traces directly).
- **What the receipts leave out:**
  - the driver records drop per-token chunk gaps (`trimmed`);
  - the prompt pool, the logs and the GPU sampler are left out;
  - the step traces are gzipped.

  All of it is complete in the receipt store at `ffa52ab4`.
