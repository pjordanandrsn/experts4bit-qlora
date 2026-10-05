# SC2c: does bulk KV bookkeeping lift e4b `serve_paged`'s request-level capacity, and where does the per-prefill stall go? `E4B_PAGED_BULK_KV` OFF against ON, paired, with a per-step trace in both arms, one RTX 5090 (lane SC2c of #846; drafted 2026-10-05)

**Status: DRAFT.** Registered when this file merges to `main`. That needs another agent's review, the code PR merged
first, and CI green. No SC2c run exists. The census behind it (`bench/stall-census-2026-10-05/`) is exploratory and
$0.

**The code under test** is branch `serve-bulk-kv` at `81d3e994` (`3e7b75a5` merged with `main` at `6c80df42`, which
brings e4b#1129's `seen` routes), whose `experts4bit_qlora/` tree is `61c6057fad30ea3e3a3136e91699bafba022e4f8`:
- `E4B_PAGED_BULK_KV`, opt-in;
- `E4B_PAGED_STEP_TRACE`;
- `/health`'s `kv_bookkeeping` block.

The box runs this registration's merge commit. Its `experts4bit_qlora/` tree must equal the code PR's merged tree
except for the version string, and the launch chain records both. A change to the code PR after this draft is
re-pinned here before any box.

**The `/health` contract** this rule reads, from `serve_paged.kv_bookkeeping_report` and
`PagedModelRunner.kv_bookkeeping_stats` at `81d3e994`:
- `kv_bookkeeping` always carries `requested`, the knob as a bool.
- Once the engine is built it also carries `bulk`, plus per-request counts:
  - `flush_layers` and `flush_bulk`: the prompt's flush into the pool, per path;
  - `ready_layers` and `ready_bulk`: a graphed slot's block claims at its first decode, per path;
  - `ready_at_flush`: the bulk flush made those claims itself.

## Why this lane

SC2b (`bench/h2h-2026-10-02/sc2b/`) licensed the first-chunk prefill graph, but the capacity ceiling stayed at
1 req/s. Under load the stall each prefill imposes on running decodes (~0.27 s) is ~1.6× the graphed serial TTFT, so
most of it lies outside the forward. The stall census (`bench/stall-census-2026-10-05/README.md`) makes four points.

**1. The fitted stall carried batch growth.** With the time-weighted decode bucket as a regressor it is **0.218 /
0.224 s** on SC2b's ON servers, not 0.262 / 0.269. R² rises from 0.985 / 0.990 to 0.993 / 0.997.

**2. The prefill step does not grow under load.** Admission to first token holds at **157–170 ms** from serial to
8 req/s.

**3. Per request, the engine thread host-issues ~13.5k launches of KV bookkeeping before resident decodes continue.**
- **The parts:**
  - the prompt's flush into the FP8 pool: 2,448 kernels + 6,240 copies;
  - the slot's first graphed decode claiming every reachable block: 4,608;
  - two slot resets: 96 each.
- **The cost:** ~300 ms on the NAS A2000's host.
- **In bulk:** ~3 ms, bitwise identical in every pool byte, table and length.

**4. Inferred, not measured.** On SC2b's box the flush (~150 ms) sets the prefill step, the block claims add ~55 ms at
the next decode, and the graph replay hides under the flush. A forward of ≥ ~150 ms device time would refute that.

This lane measures the decomposition directly, with the step trace in both arms. It reads whether removing the
bookkeeping moves the ceiling.

## The box (H) and the stack it pins

- **One box, e4b only.** `SC1_BOX=H` (`bench/sc2/sc2c_box_h.sh`), sourced after `sc2_box_e.sh`; `bench/sc1/sc1_run.sh`
  and `sc1_drive.sh` gain box H and nothing else.
- **e4b.** This registration's merge commit (above).
- **grouped-nf4-gemm.** **v0.41.0** (`dc8f94abfd868f149178623f6eb403dc8b892b02`, tag object `e90a3523`): e4b 0.48.0
  CI's pin, as SC2g's box G.
- **Arm `e4b_int4`.** SC2's and SC2b's levers (`SPEEDENV`, the three folds, `E4B_PAGED_FUSE_QKV=1`, `ROUTEENV`), with
  `max_seqs` 16, `E4B_PAGED_MAX_TOKENS_PER_SEQ=2048` and chunk 512.
- **The prefill graph.** It runs at its default, `auto`: the server is started with `env -u E4B_PAGED_PREFILL_GRAPH`,
  and it must engage in BOTH arms (the stack under test).
- **The only difference between OFF and ON** is `E4B_PAGED_BULK_KV=0|1`.
- **Both arms run** `E4B_PAGED_TRACE` (per request) and `E4B_PAGED_STEP_TRACE` (per step). The step trace costs a few
  `perf_counter` calls and up to six CUDA events a step, the same in both arms.
- **Routes are main's defaults, read from each server's own `/health`:**
  - `prefill_routes`: k19 / k19 / flash, device grouping on, both raw env values null;
  - `prefill_routes.seen` (e4b#1129), what the forward took at the startup captures: every expert GEMM above
    256 rows on K19 (`int4_k19|gt256`), every prefill attention call on flash. The resolved fields alone read
    k19 / flash on gpt-oss while neither ran (sc2g-prove-2);
  - `engine`: chunk 512 and a per-step budget of 512.
  - Otherwise the arm STOPs (rc 47), as in SC2b.
- **Engagement is checked before the paid workload.** Right after each server's 4 warm requests:
  - **prefill graph:** `/health` must read `status` "on", 4 replays and 0 eager chunks;
  - **`kv_bookkeeping`, OFF:** `requested`/`bulk` false, `flush_layers` 4, nothing in bulk;
  - **`kv_bookkeeping`, ON:** `requested`/`bulk` true, `flush_bulk` 4, `ready_at_flush` 4, nothing per layer;
  - every warm response reports 512 prompt tokens;
  - the step trace holds ≥ 64 rows.
  - Otherwise the arm STOPs (rc 48).
- **Records, no gate:** `nvidia-smi` memory used at ready, per arm; the full `/health` at start, after warm and at
  the end.
- **Image and checkpoint.** SC1's `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`; Qwen3-30B-A3B @ `ad44e77`, baked to
  the NF4 arena on the box.

## The instrument

- **Driver and prompts.** SC2's, unchanged. Each request is one 512-token wikitext-2 row, `max_tokens` drawn from
  U[64, 256], greedy, `ignore_eos`, streaming, 16 in flight. SLO: TTFT ≤ 1.0 s and TPOT ≤ 100 ms.
- **Paired arms, as SC2b.** Each (draw, arm) is its own server. Draw 1 runs OFF then ON, draw 2 ON then OFF. Both arms
  of a draw use the same plan seeds: serial = the draw, Poisson = draw × 100 + rate.
- **Per server:**
  1. 4 warm serial requests;
  2. Q1 serial, 24 requests;
  3. Q2 Poisson at 1, 2, 4 and 8 req/s, 120 requests each.
  - `/health` is read at start, after warm and at the end.
  - Draw 1's OFF server repeats the serial plan: the determinism control.

## The rule (`bench/sc2/sc2c_reduce.py`, self-tested on 13 cases; census by `bench/sc2/sc2c_census.py`, 7)

**Gates, in order.**
1. **ROUTES.** Every arm's start `/health` reads the registered routes, chunking and `seen` routes, or the run is VOID.
2. **ENGAGED.** Every arm's end `/health` reads as follows, or the run is VOID.
   - **Prefill graph:** `status` "on", `T` 512, replays equal to the requests the server admitted (warm-up included),
     eager chunks 0.
   - **`kv_bookkeeping`, OFF:** `requested` and `bulk` false; `flush_layers` = `ready_layers` = admitted; `flush_bulk`
     = `ready_bulk` = `ready_at_flush` = 0.
   - **`kv_bookkeeping`, ON:** `requested` and `bulk` true; `flush_bulk` = `ready_at_flush` = admitted;
     `flush_layers` = `ready_layers` = `ready_bulk` = 0.
3. **DETERMINISM.** OFF draw-1 serial against its repeat must be IDENTICAL (`sc2_identity.py`), or the identity gate is
   UNREAD.
4. **IDENTITY.** OFF against ON on each draw's serial plan must be IDENTICAL. Otherwise the knob DIFFERS and cannot be
   a default. The bulk forms leave the same pool bytes, so the expectation is IDENTICAL.
5. **PROMPTS.** Every request in every run reports 512 prompt tokens, or the run is VOID.

**Rows and ceilings** use SC2's rule (`sc2_reduce.row` / `ceiling`), per arm.

**Predictions**, paired per draw. F below is the 512-token forward's device time. No receipt isolates it; the step
trace reads it as `pf_forward − pf_prep`. P107's profile of a 4096-token flash prefill (`bench/p107`) bounds it
indirectly at ~42 ms of device time per 512-token chunk: 378 ms total, minus 38 ms of D2D copies, ÷ 8. That is on
another box, uncapped, at `max_seqs` 1 with host grouping (census §6). Those 50,352 copies are themselves the per-block
flush (48 × 256 × 4 = 49,152).

| # | prediction | basis |
|---|---|---|
| P1 | the stall per prefill, bucket-controlled (`sc2c_census.fit` on the request trace), ON / OFF ≤ 0.6 in both draws | **The mechanism.** OFF ≈ 0.22 s (SC2b's ON refit). With the bookkeeping gone, ON ≈ F + a few ms. P1 holds iff F ≲ 125 ms |
| P2 | serial p50 TTFT, OFF / ON, ≥ 1.4 in both draws | OFF ≈ 0.16–0.17 s (SC2b's ON servers). ON ≈ F + ~5–10 ms. P2 holds iff F ≲ 110 ms |
| P3 | serial p50 TPOT, ON / OFF, in [0.95, 1.05] in both draws | the decode step is untouched (its first-step claims move to the flush) |
| P4 | ON's capacity ceiling ≥ 2 req/s | `capsim.py`: ceiling 2 at a 110 ms prefill step, against today's 1 |
| P5 | ON's capacity ceiling ≥ 4 req/s | `capsim.py`: 4 at ≤ 80 ms (attainment 0.98 at 80 ms, marginal). A coin flip on F, stated as such |
| P6 | no regression: at every rate and draw ON's attainment ≥ OFF's − 0.05, and ON's ceiling ≥ OFF's | |

**Licence, decoupled from P1, P2, P4 and P5.** The result is DEFAULT_LICENSED iff all of these hold:
- ROUTES, ENGAGED, PROMPTS, DETERMINISM and IDENTITY pass;
- P6 holds;
- serial p50 TTFT OFF / ON is ≥ 1.10 in both draws.

Otherwise NOT_LICENSED, with the reason. On DEFAULT_LICENSED, a separate PR makes `E4B_PAGED_BULK_KV` default to `1`,
keeping `0` as the escape.

**The licence's scope.** Speed is read on Qwen3-30B-A3B int4, `serve_paged`, one RTX 5090 and 512-token prompts. The
bookkeeping's equivalence elsewhere is the tests' job (whole-pool bitwise comparisons, mixed geometry, hybrid layer
subsets, partial tail blocks). Speed elsewhere is unread.

## Reported, no bar: the census

Per arm and draw, from the server's own traces (`sc2c_census.py`).
- **The prefill step:** its median length and host segments (`plan`, `pf_prep`, `pf_forward`, `pf_flush`,
  `pf_sync`, …).
- **The forward's device time:** `pf_forward − pf_prep`, and when the GPU finished the flush.
- **Bookkeeping per prompt:** `pf_flush`, `dec_ready`, `plan`, `retire`.
- **Decode steps by bucket:** step time and device time.
- **The direct stall:** a prefill step minus a decode-only step of its bucket, plus the first-decode claims.
- **Fits:** the request-level fit, two-regressor and bucket-controlled.
- **Memory:** VRAM at ready.

These close (or fail to close) the census's decomposition on the box that measures it. A large unexplained residual is
reported as one.

**Post hoc, descriptive.** The read will test e4b#1134's capacity model a second time:
`serve_capacity.StepCosts.from_step_trace` on each arm's step trace, `simulate` on that arm's own plans, set beside its
measured attainment. No rule reads it.

**No position against vLLM, SGLang or llama.cpp comes from SC2c.** It is e4b against itself on one box. SC2's
comparator rows stand.

## Proof and budget

**Proof** (`sc2c-prove-*`, guard 1.0 h):
- self-tests: driver, SC2 reducer, identity, census, SC2c reducer;
- the proof model's prompt pool;
- an OFF server and an ON server on Granite-3.1-3b-a800m (NF4 store, SC1's `GR_ENV` unchanged), each on the registered
  routes, each answering a 6-request serial smoke and a 16-request Poisson smoke at 4 req/s with every request VALID;
- the serial smokes IDENTICAL, OFF against ON;
- both knobs engaged as registered (`sc2c_reduce.py --engagement`);
- each arm's step trace decomposes: ≥ 26 prompts, a forward device time, decode steps.

**Reading** (`sc2c-5090-*`, guard 2.5 h): installs ~10 min, fetch and bake 25–40 min, four servers at ~9 min each.

**Guards**, as SC2b counted them: hours × $0.75, plus download at $0.0078/GB.
- **proof:** 1.0 h ≤ $0.75 + ~$0.10;
- **reading:** 2.5 h ≤ $1.88 + ~$0.55.

Each run sits under #846's standing no-ask tier for a single run under $15. The deadline drops draw 2 first.

## Out of scope

- Prompts over 512 tokens (later chunks stay eager; the bulk flush covers them, the graph does not).
- The default flip itself (a separate PR on the licence).
- Chunked prefill interleaved with decode, decode-step time, and slot count. These are the next levers if 8 req/s is
  the target (census §5).
- Comparators, other models, other GPUs.
