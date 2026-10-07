# SC2e: does a wider batch lift e4b `serve_paged`'s request-level capacity toward 8 req/s? `E4B_PAGED_MAX_SEQS` 16 against 32 and 64, with decode-graph buckets that end at `max_seqs` (`E4B_PAGED_BUCKETS=auto`) against the default list, paired, with a per-step trace in every arm, one RTX 5090 (lane SC2e of #846; registered 2026-10-07)

**Registered** when this file merges to `main`, before any SC2e run. Opened without `ready-to-merge` or auto-merge,
for maintainer review.

**The code under test** is #1319 ("serve_paged: `E4B_PAGED_BUCKETS=auto` captures decode buckets up to
`max_seqs`"), once merged. Its merge commit and `experts4bit_qlora/` tree hash are
recorded here before the proof; the box runs this registration's merge commit, and the launch chain records both.
**Recorded at review** (read from the objects, not recalled): #1319 merged as
`a99b857e2c547f05926ec6fcc603203c49eb60be` (GitHub `mergedAt` 2026-10-07T18:37:46Z); `experts4bit_qlora/` tree at that
commit `4d96736cb34dc0d8077969907cf3c27fa0be66ce`. The box's tripwire should find that tree at the launch commit unless
a later package change lands first, in which case the launch commit's tree is recorded beside it. It
brings:
- `E4B_PAGED_BUCKETS=auto` (`serve_recipe.default_buckets`): every power of two below `max_seqs`, then `max_seqs`;
  opt-in, equal to the default list up to 16;
- `/health`'s `engine.buckets_requested`, `engine.graph_stats` (per bucket: `replays`, `eager_steps`, `rows`,
  `pad_rows`) and `levers.kv.pool_mib`;
- the step trace's `dec_pieces` (replays per decode step).

Nothing else on the path under test changes: no route, kernel or default.

## Why this lane

**Where e4b stands.** SC2 (#1057): vLLM and SGLang hold the SLO (TTFT ≤ 1.0 s and TPOT ≤ 100 ms) to 8 req/s on one
RTX 5090; e4b held to 1. SC2b's prefill graph (#1086) and SC2c's bulk KV bookkeeping (#1166, #1200) brought e4b to
**4 req/s**. Both are defaults now.

**What fails at 8 req/s, from SC2c's own ON servers** (`sc2c-5090-1`, the server this lane's control repeats), draw 1 / draw 2,
from the committed receipts (`python bench/sc2/sc2e_basis.py census bench/h2h-2026-10-02/sc2c/receipts/sc2c-5090-1/sc2`):

| | serial | 4 req/s | 8 req/s |
|---|---|---|---|
| decode-only step p50 at bucket 16 (ms) | – | 8.8 / 8.8 | 9.3 / 9.4 |
| its device time p50, `gpu.dec_issue − gpu.dec_prep` (ms) | – | 7.9 / 7.9 | 8.3 / 8.4 |
| prefill steps' share of the workload's wall time | 5 % / 5 % | 19 % / 15 % | 34 % / 33 % |
| client TPOT p50 (ms) | 4.2 / 4.2 | 9.7 / 8.2 | 13.1 / 13.1 |
| client TTFT p50 (ms) | 40 / 40 | 44 / 43 | 262 / 1,213 |
| queue wait p50 (ms) | 0.9 / 0.6 | 4.7 / 3.3 | 223 / 1,174 |

The decode step is ~90 % device time, and TPOT is far inside its 100 ms bound. TTFT fails because requests wait for one of 16
slots: at 8 req/s and ~170 output tokens a request, 16 slots are saturated.

**The model's ranking.** SC2c's post hoc (`bench/h2h-2026-10-02/sc2c/README.md` §"Post hoc") ran `serve_capacity`
(#1134) on each server's own costs: at 8 req/s `max_seqs` 64 reads 1.00 / 1.00 against 0.70 / 0.34 today, ahead of
decode × 0.7 and far ahead of prefill × 0.5. Its decode cost above 16 rows is a linear extrapolation, its memory cost
is unread, and its error at a knee is 0.10–0.26 (pessimistic).

**Why two bucket lists.** At `max_seqs` above 16 the default list stops at 16, so a 64-row decode step runs as four
16-row replays with a host sync after each. `auto` captures one 64-row graph instead, whose rows above 16 take the
paths prefill chunks take today:
- `Int4Linear` above 16 rows runs cuBLAS on its cached bf16 weight;
- above 256 routed expert rows (bucket 64 at top-k 8), K19 runs over the chained tile table;
- the T=1 folds still apply at 64 rows.

The model separates the two only at 12 req/s: 64 slots on the default list read 0.63 / 0.60 there, on `auto` 0.79 /
0.68. Whether the wide graph beats four narrow ones on this card is the question P1b asks.

**The basis**, reproducible with `bench/sc2/sc2e_basis.py` from SC2c's committed fitted costs
(`receipts/sc2c-5090-1/capacity_check.json`; ON d1: prefill 39.48 ms, first decode 0.81, decode 4.129 + 0.3476 / row;
ON d2: 39.46, 0.86, 4.122 + 0.3681), seeds draw × 100 + rate, 120 requests:

| arm | 8 req/s | 12 req/s | 16 req/s | model ceiling | decode step at full rows |
|---|---|---|---|---|---|
| s16 (today) | 0.70, 0.34 / 0.69, 0.23 | 0.21, 0.17 / 0.20, 0.17 | 0.16, 0.14 / 0.16, 0.14 | 4 / 4 | 9.7 / 10.0 ms |
| s32a | 0.99, 0.98 / 0.98, 0.92 | 0.34, 0.30 / 0.32, 0.30 | 0.28, 0.28 / 0.28, 0.27 | 8 / 4 | 15.3 / 15.9 ms |
| s64c | 1.00, 1.00 / 1.00, 1.00 | 0.63, 0.60 / 0.63, 0.60 | 0.52, 0.53 / 0.51, 0.53 | 8 / 8 | 38.8 / 40.0 ms (4 replays) |
| s64a | 1.00, 1.00 / 1.00, 1.00 | 0.79, 0.68 / 0.77, 0.65 | 0.54, 0.53 / 0.53, 0.53 | 8 / 8 | 26.4 / 27.7 ms |

Cells read "draw 1, draw 2" on ON d1's costs, then on ON d2's. Sensitivity: twice the per-row cost (0.70 ms / row)
puts s32a at 0.59 / 0.31 and s64a at 0.83 / 0.64 at 8 req/s, both ceilings 4.

**Memory**, from `serve_recipe.paged_kv_pool_bytes(48, 4, 128, batch=m, max_tokens_per_seq=2048, scratch_slots=…)`:
1,669.7 / 3,339.4 / 6,678.8 MiB at 16 / 32 / 64 slots (103.5 MiB per slot). SC2c's ON server had 23,686 of 32,607 MiB
used at ready and `prefill_graph.free_after_mib` 8,426, so 64 slots fit with ~3.4 GiB to spare before graph pools.

## The box (L) and the stack it pins

- **One box, e4b only.** `SC1_BOX=L` (`bench/sc2/sc2e_box_l.sh`), sourced after `sc2_box_e.sh` and `sc2c_box_h.sh`
  (whose `routes_ok`, `h_health_end`, `drive`, `gpu_free`, `wait_e4b_ready` and `stop_pid` it uses);
  `bench/sc1/sc1_run.sh` and `sc1_drive.sh` gain box L and nothing else.
- **e4b.** This registration's merge commit. A tripwire refuses the box (rc 9) unless the installed e4b has
  `serve_recipe.default_buckets` and `/health`'s `graph_stats` (the code under test).
- **grouped-nf4-gemm.** **v0.42.0** (`b4f93f1c62d1e3436ed45bec8ccd608c90433737`): e4b CI's pin at registration.
  SC2c ran v0.41.0; the paired control re-reads the baseline on this stack, and no ratio crosses lanes.
- **Arms**, each SC2c's ON server (`SPEEDENV`, the three folds, `E4B_PAGED_FUSE_QKV=1`, `ROUTEENV`, 2,048 tokens a
  slot, chunk 512, bulk KV and the prefill graph at their defaults, `E4B_PAGED_TRACE` and `E4B_PAGED_STEP_TRACE` on),
  differing only in two knobs:

  | arm | `E4B_PAGED_MAX_SEQS` | `E4B_PAGED_BUCKETS` | buckets the server must report |
  |---|---|---|---|
  | `s16` (control) | 16 | unset (the default list) | 1, 2, 4, 8, 16 |
  | `s32a` | 32 | `auto` | 1, 2, 4, 8, 16, 32 |
  | `s64c` | 64 | unset (the default list) | 1, 2, 4, 8, 16 |
  | `s64a` | 64 | `auto` | 1, 2, 4, 8, 16, 32, 64 |

  Every server starts with `env -u E4B_PAGED_PREFILL_GRAPH -u E4B_PAGED_BULK_KV -u E4B_PAGED_GRAPHS` (defaults under
  test) and `env -u E4B_PAGED_BUCKETS` unless the arm sets it.
- **Routes are main's defaults, read from each server's own `/health`** (SC2c's `routes_ok`, unchanged): k19 / k19 /
  flash, device grouping on, both raw env values null, `seen` K19 above 256 rows and flash; chunk 512, budget 512.
  Otherwise the arm STOPs (rc 47).
- **Slots and buckets, read at start** (`slots_ok`): `engine.max_seqs` = `engine.kv_slots` = the arm's;
  `engine.buckets` = the table's; `engine.buckets_requested` as set; every `graph_status` entry "graph";
  `levers.kv.scratch_slots` = the largest bucket. Otherwise the arm STOPs (rc 49).
- **Engagement before the paid workload** (`l_early`), after 4 warm requests:
  - the prefill graph `on` with 4 replays and 0 eager chunks; `kv_bookkeeping` bulk with `flush_bulk` 4;
  - every warm response reports 512 prompt tokens; the step trace holds ≥ 64 rows;
  - then a **burst**: 64 requests arriving at once (Poisson at 1,000 req/s, seed 998), every one VALID with 512 prompt
    tokens, after which `engine.graph_stats` shows at least one replay of the arm's largest bucket and no eager step in
    any bucket. Every request asks ≥ 64 tokens and prefill admits one prompt a step, so the last admission finds the
    earlier ones still decoding: the arm's widest step must run.
  - Otherwise the arm STOPs (rc 48). The burst makes the widest graph run before any run that counts.
- **Records, no gate:** `nvidia-smi` memory used at ready; `/health`'s `prefill_graph.free_after_mib`,
  `bulk_flush_mib` and `levers.kv.pool_mib`; the full `/health` at start, after warm, after the burst and at the end.
- **Image and checkpoint.** SC1's `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, torch 2.8.0+cu128 (A1 of SC2c);
  Qwen3-30B-A3B @ `ad44e77`, baked to the NF4 arena on the box.

## The instrument

- **Driver and prompts.** SC2's, unchanged: each request one 512-token wikitext-2 row, `max_tokens` from U[64, 256],
  greedy, `ignore_eos`, streaming, open-loop (the client caps nothing; the server's slots bound concurrency).
- **Paired arms.** Each (draw, arm) is its own server. Draw 1 runs s16, s32a, s64c, s64a; draw 2 the reverse. Every
  arm of a draw uses the same plan seeds: serial = the draw, Poisson = draw × 100 + rate.
- **Per server:** 4 warm serial requests; the burst; Q1 serial, 24 requests (s16 draw 1 runs it twice: the
  determinism control); Q2 Poisson at **1, 2, 4, 8, 12 and 16 req/s**, 120 requests each. 12 and 16 are new: without
  them a 64-slot server's ceiling is censored at 8.

## The rule (`bench/sc2/sc2e_reduce.py`, self-tested; census by `bench/sc2/sc2e_census.py`)

**Gates.** ROUTES, SLOTS, ENGAGED and PROMPTS are read per server, and a failure names the server. An arm is licensable
only if its own two servers and s16's pass them. DETERMINISM and IDENTITY are read over the lane.
1. **ROUTES.** Every server's start `/health` reads SC2c's registered routes, or VOID.
2. **SLOTS.** Every server's start `/health` reads its arm's slots and buckets, all captured, or VOID.
3. **ENGAGED.** Every server's end `/health` reads: the prefill graph on, T 512, replays = the requests it admitted
   (warm, burst, serial, repeat, every rate), eager chunks 0; `kv_bookkeeping` bulk for every one of them
   (`flush_bulk` = `ready_at_flush` = admitted, nothing per layer); `graph_stats` with ≥ 1 replay of the largest
   bucket and 0 eager steps in every bucket. Otherwise VOID.
4. **PROMPTS.** Every request in every run reports 512 prompt tokens, or VOID.
5. **DETERMINISM.** s16 draw 1's serial against its repeat is IDENTICAL (`sc2_identity`), or IDENTITY is UNREAD.
6. **IDENTITY.** Per draw, s32a, s64c and s64a against s16 on the serial plan are IDENTICAL. A serial request decodes
   at bucket 1 on every server, so the expectation is IDENTICAL; DIFFERS means the slot count changed one sequence's
   arithmetic, and no arm is licensable.

**Rows and ceilings.** `sc2_reduce.row` per arm and rate over both draws; the ceiling is the largest of the six rates
whose row is VALID with attainment ≥ 0.95 in both draws (`sc2_reduce`'s rule over this ladder).

**Predictions**, paired per draw, each with what a miss means:

| # | prediction | basis | if MISSED |
|---|---|---|---|
| P1 | decode-only step p50 at the largest bucket: s32a's bucket 32 in [12, 20] ms; s64a's bucket 64 in [20, 34] ms, both draws | linear fit 15.3–15.9 / 26.4–27.7 ms, plus ≤ ~1 ms for the bf16 attention reads (1.36 GB more per step at ~1.5 TB/s) and the chained tile table above 256 rows | high: the per-row cost is not linear above 16 and the ceilings fall toward 4 (sensitivity above) |
| P1b | s64a's bucket-64 step p50 / s64c's four-replay step p50 (decode-only steps with `dec_pieces` 4) ≤ 0.85, both draws | model 26.4 / 38.8 = 0.68 and 27.7 / 40.0 = 0.69 | the wide graph's prefill-side routes cost more than three syncs and replays: `auto` is not the way to 64 rows |
| P2 | serial p50 TTFT and TPOT of s32a, s64c and s64a within ±5 % of s16's, both draws | a serial request runs bucket 1 and one batch-1 prefill on every server | slots or scratch slots cost a single sequence time: name where in the census |
| P3 | s64a's ceiling ≥ 8 req/s | model 1.00 / 1.00 at 8 on both cost sets; TTFT p99 0.19–0.29 s | the slots are not what binds at 8 on this card |
| P4 | s32a's attainment at 8 req/s ≥ 0.90 in both draws (its ceiling at 8 is a coin flip, reported) | model 0.99 / 0.98 and 0.98 / 0.92 | 32 slots do not reach the knee; the 24 GB tier's fallback needs another lever |
| P5 | s64a at 12 req/s ≥ 0.50 in both draws, and at 16 req/s < 0.95 in at least one | model 0.79 / 0.68, 0.77 / 0.65 at 12; 0.54 / 0.53 at 16 | high at 16: decode is not the next bound; low at 12: the model is optimistic past the knee |
| P6 | no regression at 1, 2 and 4 req/s: each wide arm's attainment ≥ s16's − 0.05 per draw, and each wide arm's ceiling ≥ s16's | a 16-slot server never queues there | wider is worse at low load: name where |
| P7 | VRAM at ready minus s16's: s32a ∈ [1,600, 2,400] MiB; s64c ∈ [4,900, 5,600]; s64a ∈ [4,900, 6,200]; the prefill graph engages on every server | pool +1,670 / +5,009 / +5,009 MiB; s64a's 48 more scratch blocks are ~3 MiB; the graphs for buckets 32 and 64 are unpriced (SV1: +60 MiB for five buckets on OLMoE) | the estimate misses the wide buckets' pools: `estimate_serve_footprint` must price them before any default |
| P8 | s64a's TPOT p50 at 8 req/s ≤ 40 ms, both draws | model 24.3–34.9 ms | the stall per interleaved prefill grows with the batch |

**Licence**, decoupled from P1, P1b, P3, P4, P5 and P8. For an arm X among s64a, s64c and s32a, X is LICENSABLE iff:
- X's two servers and s16's pass ROUTES, SLOTS, ENGAGED and PROMPTS; DETERMINISM passes; IDENTITY reads IDENTICAL for
  X in both draws;
- P6 holds for X and P7's engagement clause holds for X;
- X's ceiling is strictly above s16's.

The verdict is the first LICENSABLE arm in the order s64a, s64c, s32a: `SLOTS_LICENSED(64, auto)`,
`SLOTS_LICENSED(64, default)` or `SLOTS_LICENSED(32, auto)`, else `NOT_LICENSED` with the reasons.

**What a licence licenses**, as a separate PR citing this read:
- `E4B_PAGED_MAX_SEQS=auto` as the default: at startup the largest of {the licensed width, 32, 16} whose
  `estimate_serve_footprint` device total, plus the prefill graph's pool and the bulk flush's bound, fits the free
  memory with 1 GiB to spare; otherwise 16. An integer keeps today's meaning, and `/health` reports the resolution.
  Never a bare 64: on a 24 GB card, at the 4,096-token default, or on a hybrid (~62 MiB of linear state per slot),
  64 slots do not fit.
- `E4B_PAGED_BUCKETS=auto` as the default only under `SLOTS_LICENSED(64, auto)` with P1b HOLDS (the reducer's
  `buckets_auto`); otherwise wider servers keep the default list.
- **Quality is not read here.** Serial IDENTITY reads bucket 1 only, and decode rows above 16 change the bf16
  arithmetic. The default flip also needs P110's teacher-forced read at buckets 32 and 64, registered separately,
  unless the maintainer rules otherwise on this registration.

**The licence's scope.** Qwen3-30B-A3B int4 through `serve_paged`, one RTX 5090, 512-token prompts, 2,048 tokens a
slot. Other models, prompt lengths, cards and memory tiers are unread.

## Reported, no bar: the census

Per server, from its own traces (`sc2e_census.py`, built on `sc2c_census` with the arm's bucket list):
- decode-only steps by bucket, including 32 and 64: step p50, device time p50, host segments p50;
- `dec_pieces` per decode step (how many decode steps chained, and their step time);
- prefill steps: step p50, the forward's device time, and the decode rows riding each;
- inter-step gaps under 20 ms (the engine loop while busy) as a share of busy time; per rate, the mean requests in
  the server and the queue wait p50;
- the request-level fit (`sc2c_census.fit`, two-regressor and bucket-controlled) with the arm's buckets;
- VRAM at ready, `free_after_mib`, `bulk_flush_mib`, `pool_mib`; `graph_stats` with padding rows per bucket.

**Text agreement, no bar.** Per rate and draw, the share of requests whose streamed text is byte-equal between s16 and
each wide arm (same seeds). Under load, batch composition differs between servers, so this is descriptive only.

**Post hoc.** `serve_capacity` on each server's own `StepCosts.from_step_trace` (buckets as captured) and plans, beside
its measured attainment (`bench/h2h-2026-10-02/sc2e/sc2e_capacity_check.py`, as SC2c's): the model's first test above
16 rows. No rule reads it.

## Proof and budget

**Proof** (`sc2e-prove-*`, guard 1.0 h), on Granite-3.1-3b-a800m (NF4 store, SC1's `GR_ENV` unchanged):
- self-tests: driver, SC2 reducer, identity, SC2c census, SC2e census, SC2e reducer;
- the proof model's prompt pool;
- four servers, one per arm: `routes_ok` (SC2c's proof form: a grouped call above 256 rows ran, attention flash),
  `slots_ok`, warm, `l_early` with the burst, a 6-request serial smoke and a 16-request Poisson smoke at 8 req/s, every
  request VALID;
- the serial smokes IDENTICAL, s16 against each wide arm;
- every server engaged as registered (`sc2e_reduce.py --engagement`);
- each step trace decomposes, with decode steps at a bucket above 16 on s32a and s64a, and at least one 4-piece
  decode step on s64c.

The proof exercises capture and replay of buckets 32 and 64 on Granite's NF4 routes. It does not run int4 K19 above 256
rows or `Int4Linear` above 16 rows; those first run on the reading box, where SLOTS and the burst stop the arm before any
run that counts.

**Reading** (`sc2e-5090-*`, guard 3.5 h): installs ~10 min, fetch and bake 25–40 min, eight servers at ~10–11 min each.

**Guards**, as SC2c counted them: hours × $0.75, plus download at $0.0078/GB.
- **proof:** 1.0 h ≤ $0.75 + ~$0.10;
- **reading:** 3.5 h ≤ $2.63 + ~$0.55.

Lane ceiling $8 (two proofs, one reading, one re-run). Each run sits under #846's standing no-ask tier for a single
run under $15. The deadline drops draw 2's servers first.

## Out of scope

- Teacher-forced quality of decode above 16 rows (see the licence).
- Hybrids, gpt-oss, other models and cards; prompts over 512 tokens; 4,096 tokens a slot; `max_seqs` above 64.
- The host work per decode step (the per-piece sync, `_seen`'s mirror loop, pageable index copies): ≤ ~3 % of a
  15–26 ms step; the census prices it.
- Faster decode kernels (K19 at bucket 16 runs ~70 % of its byte floor, P86) and chunked prefill: the next lanes.
- Comparators.

## Amendments

None yet.
