# Stall census, part 1: most of `serve_paged`'s per-prefill stall looks like KV bookkeeping, not the forward (exploratory, $0; lane SC2b's open question, #846, 2026-10-05)

**Question.** SC2b left e4b `serve_paged` at a capacity ceiling of 1 req/s. Under load each 512-token prefill stalls
every running decode by ~0.27 s, about 1.6× the graphed serial TTFT. So most of the stall lies outside the graphed
forward (`bench/h2h-2026-10-02/sc2b/README.md`). Where does it go, and what is the largest removable part?

**Answer, with its status stated.**
- Two things are **measured**:
  - **Batch growth.** ~45 ms of SC2b's fitted stall is the batch growing as requests arrive, not the prefill.
  - **The bookkeeping's size.** One request's KV bookkeeping is **~13.5k host-issued launches** at SC2b's geometry.
    That costs **~300 ms** of host time on the NAS A2000's (loaded) host; the same work in bulk costs ~3 ms, bitwise
    identical.
- The split of the rest on SC2b's box is **inferred**, not measured:
  - the prompt's flush into the FP8 pool, ~150 ms, which sets the prefill step;
  - the first graphed decode's block claims, ~55 ms;
  - the graph replay itself, hidden under the flush.

  A registered box with a per-step trace (SC2c, `bench/sc2/SC2c-PREREG.md`) measures it.
- **Projection (a model, not a measurement):** if the bookkeeping goes, the ceiling moves from 1 to 2-4 req/s,
  depending on the 512-token forward's device time, which no receipt holds. 8 req/s stays out of reach at 16 slots.

No claim is drawn from this page. It is not a registered lane.

## 1. A request that arrives while others decode (read from the code at `51639bf`)

The engine is one thread (`serve_paged.PagedEngine._run`) and runs one `ContinuousScheduler.step()` at a time.

**Arrival.**
1. The HTTP thread enqueues the request.
2. The engine picks it up only after the step in flight finishes. A step is never preempted.

**Step *n*: admission, prefill and decode.**
1. `plan()` admits the request into a free slot. `PagedModelRunner.bind` → `Fp8PagedKV.reset`: 48 layers × (`fill_` +
   `zero_`) = **96 launches**, plus a sort of each layer's free list.
2. `run_prefill`, in order:
   - **prepare:** toggles prefill mode on 48 hot-residency tiers (host only), and the prompt ids go host-to-device from
     a Python list (a pageable copy, i.e. a stream sync);
   - **replay:** the first-chunk graph (~5k kernels at 48 layers, one CUDA graph since SC2b);
   - **flush:** per layer, `torch.cat` + `quantize_kv_fp8` (~20 launches) + 32 block claims (one `fill_` each) +
     2 × 64 `narrow().copy_()`. Measured below at **2,448 kernels + 6,240 device copies**. Every one is issued from
     Python in turn, and the GPU waits for each;
   - **first token:** `int(argmax)` (a sync), then the first token is emitted.
3. `run_decode` for every OTHER active slot runs only now: one bucketed graph replay (+ two H2D copies and the step's
   `index_select`), then `.tolist()` (a sync).

**Step *n*+1: the new slot's first decode.**
- `_ensure_graph_ready` claims every block the slot can reach before the replay: 48 × (128 − 32) = **4,608 `fill_`**
  at 2048 tokens per slot (SC2b's `E4B_PAGED_MAX_TOKENS_PER_SEQ`; 10,752 at the default 4096).
- Every resident decode waits for it.

**Finish.** `free_slot` → `reset`: 96 launches.

**The barriers that keep decode from progressing during a prefill:**
- one engine thread, with prefill strictly before decode within a step;
- the graph replay and the decode on one stream;
- the host-issued bookkeeping loops, which are serial and launch-bound;
- the three syncs: the pageable id copy, `int(argmax)` and `.tolist()`.

## 2. What SC2b's own traces say (post hoc; `sc2c_census.py fit`, SC2b's committed `trace_e4b_*.jsonl`)

| server | admission → first token, p50 (serial / r1 / r2 / r4 / r8) | stall per prefill, SC2's two-regressor fit (R²) | stall per prefill, bucket-controlled (R²) |
|---|---|---|---|
| ON d1 | 157 / 158 / 164 / 179 / 162 ms | 0.262 s (0.985) | **0.218 s** (0.993) |
| ON d2 | 168 / 169 / 170 / 174 / 170 ms | 0.269 s (0.990) | **0.224 s** (0.997) |
| OFF d1 | 261 / 220 / 221 / 221 / 270 ms | 0.327 s (0.987) | 0.295 s (0.990) |
| OFF d2 | 218 / 219 / 230 / 214 / 223 ms | 0.313 s (0.988) | 0.275 s (0.991) |

- **The prefill step does not grow under load.** The time from admission to first token is flat from serial to
  8 req/s, so the stall is not contention inside the step.
- **SC2's fit carried a confound.** Each request that arrives during another's decode is also one more row in every
  later decode step. The bucket-controlled fit adds the time-weighted decode bucket as a regressor:
  `decode_s = a·steps + c·steps·bucket + b·prefills`. It moves ~45 ms out of the stall, and R² rises on every server.
- **What is left.** ~0.22 s (ON) against a ~0.165 s prefill step leaves ~55 ms per prefill outside the step itself.

## 3. The bookkeeping, measured alone (`kv_bookkeeping_bench.py`, the NAS RTX A2000)

**The method.**
- An `Fp8PagedKV` at Qwen3-30B-A3B's KV geometry: 48 layers, 4 KV heads × 128, key groups 4, 16 slots, 2048 tokens
  per slot.
- The library's own methods, called in `serve_paged`'s order: admit, flush, ready, free.
- Per phase: host issue time, GPU time (events) and launches (`torch.profiler`).

**Two bulk arms.**
- **`bulk`:** the bench's prototype.
- **`library_bulk`:** the production methods from branch `serve-bulk-kv`, called in the runner's order with decode
  graphs on: `reset_all_layers`, `claim_blocks` for every reachable block at the flush, then `append_prompt`.

Both are checked bitwise against the library path. The check covers:
- every valid token's pool bytes (payload and scales) through the tables;
- the table against its host mirror, `seq_lens` and `_seen`;
- the claimed block count after the first decode.

**The host.** A Xeon W-1250 at load average ~20 (a shared NAS), so absolutes are this host's. The launch counts are
not host-dependent.

**Results, one request at SC2b's geometry (run `a2000-kv4`, e4b `3e7b75a5`; median of 24):**

| phase | library: host ms | library: launches | `library_bulk`: host ms / GPU ms | launches |
|---|---|---|---|---|
| admit (reset) | 1.4 | 96 kernels | 0.09 / 0.08 | 2 |
| flush (512-token prompt) | **216** | 2,448 kernels + 6,240 D2D copies | 2.3 / 5.3 | 62 |
| ready (first graphed decode) | **80** | 4,608 kernels | 0.06 / 0.02 | 0 (claimed at the flush) |
| free (reset) | 2.1 | 96 kernels | 0.7 / 0.7 | 2 |
| **total** | **300** | **~13.5k** | **3.2 / 6.1** | **66** |

**Parity.** Bitwise for both bulk arms at T = 512 and at T = 500, a partial tail block.

**The other runs.**
- `a2000-kv1`: 4096 tokens per slot, 8 slots. The ready phase is 10,752 `fill_` = 211 ms; the library total is 471 ms.
- `a2000-kv2`: SC2b's geometry, bench prototype only.
- `a2000-kv3`: e4b `1300e4cd`. It found a sync in the production flush: host time equalled GPU time. A layer-index
  tensor was built from a Python list straight onto the device, a pageable copy that blocks until every queued kernel
  finishes. In serving that waits out the prefill replay. `3e7b75a5` builds each index once.
- `a2000-kv3` and `a2000-kv4` also ran the bulk-KV tests and the paged-serving suites under CUDA: 189 passed, 5
  skipped.

## 4. The decomposition for SC2b's ON servers (box F: a 400 W RTX 5090, EPYC 7C13)

| component | per prefill | status |
|---|---|---|
| SC2's fitted stall | 262 / 269 ms | measured (SC2b) |
| — of which batch growth | ~45 ms | measured (the bucket-controlled refit) |
| stall, bucket-controlled | **218 / 224 ms** | measured (refit) |
| prefill step (admission → first token, p50) | 157–170 ms | measured (the request trace) |
|   ↳ the prompt's K/V flush, host issue | ~150 ms | **inferred**: the A2000's flush : ready ratio (2.7–2.8) × the 55 ms below |
|   ↳ the graph replay's device time | under the flush, unmeasured | **inferred**: hidden while the host issues the flush |
|   ↳ admission reset | ~1 ms | A2000: 1.4 ms |
| first graphed decode's block claims | ~55 ms | **inferred**: the bucket-controlled stall minus the prefill step |
| residual | ~0 ± 20 ms | — |

**Consistent across both arms, not proven.**
- **OFF arm.** The eager forward's ~5k launches add ~50 ms of host issue ahead of the same flush: 215 − 165.
- **Per-launch cost.** It comes out at ~11–12 µs on box F, against ~17 µs on the A2000's loaded host.

**What would break the inference.** A 512-token forward whose device time is ≥ ~150 ms would put the forward, not the
flush, on the prefill step's critical path. Then the bulk flush buys only the ~55 ms of block claims plus the flush's
GPU tail. SC2c's step trace reads the forward's device time directly (`pf_forward − pf_prep`).

## 5. What it would buy (projection; `capsim.py`)

**The model.** A discrete-event model of the scheduler: FIFO, one prefill chunk per step before that step's decode,
16 slots. It runs on SC2b's exact plans (`sc2_driver.plan`, seeds draw × 100 + rate), with step costs as parameters.

**Calibration** on SC2b's four servers (`capsim-calibrate.txt`). Simulated against measured attainment at
1 / 2 / 4 / 8 req/s:

| server | simulated | measured |
|---|---|---|
| ON d1 | 1.00 / 0.89 / 0.18 / 0.08 | 1.00 / 0.90 / 0.12 / 0.06 |
| ON d2 | 1.00 / 1.00 / 0.37 / 0.06 | 1.00 / 1.00 / 0.29 / 0.06 |
| OFF d1 | 1.00 / 0.47 / 0.09 / 0.04 | 1.00 / 0.57 / 0.10 / 0.04 |
| OFF d2 | 1.00 / 0.99 / 0.17 / 0.04 | 0.98 / 0.83 / 0.19 / 0.03 |

Median TPOT at 2 req/s matches within ~1 ms.

**Projection** (`capsim-project.txt`): minimum attainment over both draws.

| prefill step P | ready R | 1 | 2 | 3 | 4 | 5 | 6 | 8 req/s | ceiling (SC2's rates) |
|---|---|---|---|---|---|---|---|---|---|
| 165 ms (today) | 55 ms | 1.00 | 0.88 | 0.41 | 0.18 | 0.10 | 0.07 | 0.06 | 1 |
| 110 ms | 1 ms | 1.00 | 1.00 | 1.00 | 0.70 | 0.53 | 0.17 | 0.14 | 2 |
| 80 ms | 1 ms | 1.00 | 1.00 | 1.00 | 0.98 | 0.87 | 0.82 | 0.14 | 4 |
| 50 ms | 1 ms | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.16 | 4 |
| 30 ms | 1 ms | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.47 | 4 |

**What it implies.**
- **Removing only the block claims (R → 0) is not enough.** It leaves the ceiling at 2.
- **The flush is the lever.** Without it, P becomes the forward's device time.
- **8 req/s needs more than a cheap prefill.** At 16 slots and ~10 ms per decode step at bucket 16, the slots fill.
  The next levers are decode-step time and slot count.

## 6. A cross-check from an earlier receipt (P107)

P107 (`bench/p107/RESULTS-p107.md`) profiled one 4096-token prefill (eight 512-token chunks) on a 5090 at
`max_seqs` 1, flash:
- **device total:** 378 ms;
- **K19 experts:** 138 ms;
- **flash attention:** 40 ms;
- **device-to-device memcpy:** 38 ms over **50,352 copies**.

It called those copies "the next lead for TTFT ... host-side" and did not attribute them. They are this census's flush:
48 layers × 256 blocks × (2 sides × 2 regions) = **49,152** `narrow().copy_()` for a 4096-token prompt. It also
bounds the forward: (378 − 38) ÷ 8 ≈ **~42 ms of device time per chunk** on that box. That is an average; the first
chunk attends to no history, so it is cheaper. It is a different grouping (host-grouped, `max_seqs` 1) and a box
without SC2b's 400 W cap.

## 7. What this does not show

- The 512-token forward's device time under `serve_paged`'s device grouping on box F's capped board: no receipt
  isolates it. P107 (§6) puts it near 40–50 ms; SC2c's step trace reads it.
- Box F's per-launch host cost. It is inferred from two fits on one board.
- Anything on another model, prompt length or host class. The launch counts scale with layers × blocks: 48 × 128
  here, more at longer slots and deeper models.

## Next

- **Code** (branch `serve-bulk-kv`):
  - `E4B_PAGED_BULK_KV`, opt-in, `0` by default: `Fp8PagedKV.reset_all_layers` / `claim_blocks` / `append_prompt`,
    and the runner's bulk flush that also claims the slot's reachable blocks;
  - `E4B_PAGED_STEP_TRACE`, a per-step timeline;
  - `/health`'s `kv_bookkeeping` engagement counts.
  - Whole-pool bitwise tests, CPU and CUDA.
- **Lane SC2c** (`bench/sc2/SC2c-PREREG.md`): the knob OFF against ON on a 5090, paired, with the step trace in both
  arms. It measures the decomposition above and reads whether the ceiling moves.

## Files

| file | what |
|---|---|
| `kv_bookkeeping_bench.py` | the bench as run in `a2000-kv3` and `a2000-kv4` (three arms) |
| `a2000-kv1/kv_bookkeeping_bench.run1-2.py` | the bench as run in `a2000-kv1` and `a2000-kv2` (two arms) |
| `run_kvbench.sh`, `run_gpucheck.sh` | the NAS drivers. `run_gpucheck.sh` also runs the CUDA tests; a copy sits in `a2000-kv4/` |
| `a2000-kv1/` … `a2000-kv4/` | each run's log and JSON (kv4 also has the pytest tail) |
| `capsim.py`, `capsim-calibrate.txt`, `capsim-project.txt` | the scheduler model and its two outputs |

The refit tool is `bench/sc2/sc2c_census.py` (`fit`), run on SC2b's committed traces.

**Spend: $0.** Four A2000 runs on the NAS, 2026-10-05 08:07–08:48Z, each claimed on the bus.
