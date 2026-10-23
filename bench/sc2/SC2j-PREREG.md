# SC2j: does the shared KV pool lift e4b `serve_paged`'s request-level capacity at the default 4,096 tokens a slot? `E4B_PAGED_KV_POOL=shared` (levers G1-G3, e4b#2029, #2020, #2034, #2039) against `claim-all`, both at `E4B_PAGED_MAX_SEQS=auto`, paired, one RTX 5090 (lane SC2j of #846; serving roadmap lever G)

**Registered** when this file merges to `main`, before any SC2j run, and only after e4b#2039 has merged (the launch
commit must contain it). Box Q rents through the guarded launcher (`tools/pod-launch.sh`), with the independent deadline
watcher armed at create.

## Why this lane

Under `claim-all` (the default) a decode slot holds every KV block it could reach, its whole
`E4B_PAGED_MAX_TOKENS_PER_SEQ`, so slots × the per-slot ceiling is a hard device floor. `E4B_PAGED_MAX_SEQS=auto`
therefore offers fewer slots as the ceiling grows. On Qwen3-30B-A3B int4 on an RTX 5090, at the server's default 4,096
tokens a slot and with the default `E4B_PAGED_BUCKETS=auto`, the estimate (`serve_recipe.choose_max_seqs`) gives 32 slots.

Under `shared`:
- slots claim blocks as they write (G1);
- the scheduler admits a request only when its prompt and longest answer fit the blocks not yet promised (G2);
- the pool holds the rows free memory leaves, sized as `auto` prices a width (G3, `serve_recipe.size_shared_pool`).

The same estimate gives 64 slots on about 12,000 of 16,448 rows there, about 3,000 tokens a slot on average. These
figures are a design input only: the registered pool is whatever G3 sizes on the box.

Lane SC2e (#846, `bench/h2h-2026-10-02/sc2e/README.md`, 2026-10-07) read the widths at 2,048 tokens a slot, where
claim-all fits 64. The Q2 ceilings were 4 req/s at 16 slots, 8 at 32 slots with buckets `auto`, and 12 at 64 slots with
buckets `auto`, with identical serial output. At 4,096 tokens a slot claim-all cannot fit 64 slots, so the default
server stops at 32. Whether the shared pool turns the width it opens there into capacity is not known; this lane reads
it.

Correctness is G1's contract, read here again on the box (the contract arm, below): at matched width, shared emits the
same tokens as claim-all on one fixed schedule. It is not a speed claim. Explore smokes at matched width on Granite
(`g1-smoke-3`, `g3-smoke-3`, `g4-smoke-1/-2`) are design inputs only and license nothing.

## The launch commit and the stack (bench/common rule 1)

- **Launch commit:** this registration's merge, which must contain e4b#2039. Every SC2j box launches from it; later
  merges change neither the kit nor the stack.
- **Stack under test:** e4b at the launch commit; grouped-nf4-gemm v0.50.0 (`41ef086e6a3e2be5b158ff1bb38e0de67862a949`),
  as boxes N, O and P. A tripwire refuses the box (rc 9) unless the installed e4b has `Fp8PagedKV(pool_rows=...)`,
  `serve_recipe.size_shared_pool` and `/health`'s `engine.kv_pool_sizing`.
- **Image and checkpoint:** as SC2f's: `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, torch 2.8.0+cu128;
  Qwen3-30B-A3B @ `ad44e77`, baked to the int4 arena on the box.
  - The e4b install line also pins `brotli>=1.2.0`. An unpinned `huggingface_hub` on httpx decodes brotli with an API
    older brotli lacks, so a br-encoded file at the fetch raised TypeError on some hosts.
  - A failed model fetch is retried once after 20 s before it is a DL FAIL.
  - A box that still dies at the fetch from that cause is a launch fault, not a strike.

## The box (Q)

- **One box, e4b only.** `SC1_BOX=Q` (`bench/sc2/sc2j_box_q.sh`), sourced after `sc2_box_e.sh`, `sc2c_box_h.sh`,
  `sc2e_box_l.sh` and `sc2f_box_n.sh`, whose server start, early check, burst and tripwire it uses unchanged.
  `bench/sc1/sc1_run.sh` and `sc1_drive.sh` gain box Q and nothing else.
- **Arms**, each SC2f's `on` server (`SPEEDENV`, `E4B_PAGED_FUSE_QKV=1`, `ROUTEENV`, chunk 512, bulk KV and the prefill
  graph at their defaults, the decode lookahead off, both traces on) at `E4B_PAGED_MAX_TOKENS_PER_SEQ=4096`,
  `E4B_PAGED_MAX_SEQS=auto` and `E4B_PAGED_BUCKETS` unset (`auto`, the default). They differ only in the pool:

  | arm | `E4B_PAGED_KV_POOL` | the start `/health` must read |
  |---|---|---|
  | `ca` (control) | unset (`claim-all`) | `kv_pool` claim-all; `kv_pool_sizing` null |
  | `sh` | `shared` | `kv_pool` shared; `kv_pool_sizing.pool_rows`, `claim_all_rows`, `kv_blocks` and the derivation present |

- **Order.** Draw 1: `ca_a sh ca_b`; draw 2: `ca_a sh ca_b`. The two `ca` servers of a draw bracket `sh` and measure the
  drift and the noise. Every server of a draw runs the same plan seeds.
- **The contract arm, in process, before the servers** (`bench/sc2/sc2j_contract.py`, G1's contract on this model and
  stack). The engine is driven step by step through `serve_paged.build_engine` on one fixed schedule, using
  `bench/kv-pool-g/g3-smoke/g3_smoke.py`'s workload and drive (staged beside it) at an EXPLICIT 32 slots
  (`E4B_PAGED_MAX_SEQS=32`):
  - `claim-all` at 32 slots, twice (the floor);
  - `shared` at 32 slots. G3 keeps an explicit width and sizes the pool for it (`resolve_kv_pool`). At 4,096 tokens a
    slot, 32 slots' claim-all size (32 × 256 + 32 scratch = 8,224 rows) fits whole, so the pool holds the claim-all
    size.
  - Every token and finish reason of `shared` must equal claim-all's, and the floor must be IDENTICAL.
  - The fixed schedule never waits on blocks: the record must read `kv_blocks_waits` 0 on `shared`, or the arm is VOID
    as unreadable. The contract therefore covers the claim timing (G1) at matched width, not admission under waits.
    Admission under waits is G2's design test on the CPU (`tests/test_kv_pool_sizing.py`), where every request finishes
    with the tokens claim-all gives it, and it is exercised under load by Q3 below.
- **Per server:** 4 warm serial requests, SC2e's burst, serial 24 (seed = draw), then two ladders:
  - **the SC2 ladder** (Q2, the primary outcome): SC2's driver and 512-token prompts, `max_tokens` U[64, 256], Poisson at
    **1, 2, 4, 8, 12 and 16 req/s**, 120 requests each (SC2e's rates; seed draw × 100 + rate);
  - **the long ladder** (Q3, so reservations bind): 2,048-token prompts (`sc2_prompts.py` at 2,048 tokens from the same
    wikitext-2 split), `max_tokens` U[768, 1,536], Poisson at **0.5, 1, 2 and 4 req/s**, **96 requests each** (seed draw
    × 1,000 + 10 × rate). `/stats` is captured before and after every rung.
    - A request reserves its prompt plus its own `max_tokens`: 200 blocks on average and 224 at most. At the top rungs
      the server holds 64 requests (the slots), and 64 × 200 = 12,800 blocks exceeds the pool's about 12,000 (the
      estimate) by about 7 %: admission is designed to wait.
    - **PRESSURE**, stated before data: Q3 is read only if `sh`'s `kv_blocks_waits` rises over the 4 req/s rung in both
      draws. Otherwise Q3 is UNREAD, the shared pool was never under reservation pressure, and the verdict cannot be
      SHARED_LICENSED.
  - At the end, `/health` and `/stats` are captured.

## The rule (`bench/sc2/sc2j_reduce.py`, self-tested)

**Gates.** A failure names the server or the phase.
1. **ROUTES, ENGAGED, PROMPTS**, on every server, each VOID otherwise.
   - ROUTES as SC2e's (`sc2c_reduce.routes_bad`).
   - ENGAGED is SC2e's engagement at the server's own width, failing closed on every field it reads:
     - the prefill graph on at T 512, with replays = every prompt admitted;
     - `eager_reasons` present, with `later_chunk` the only nonzero reason, equal to `eager_chunks` and to 3 × the Q3
       prompts admitted (a 2,048-token prompt runs its first chunk on the graph and three later chunks eagerly);
     - `kv_bookkeeping` present with every field (bulk for every prompt, nothing per layer);
     - `graph_stats` carrying every bucket of the width, each with `eager_steps` present and 0, and the widest
       replayed.
   - PROMPTS per ladder: 512 tokens on Q2, 2,048 on Q3.
2. **POOL.** Every server's start `/health` reads its arm, as in the table. Otherwise VOID.
3. **WIDTH.** Every `sh` server's `engine.max_seqs` is strictly above both of its draw's `ca` servers'. Otherwise
   UNREAD: the box's memory did not open a wider width, and nothing is read.
4. **SLOTS.** Each server's buckets are `auto`'s for its own width, all captured (`sc2e_reduce.slots_bad` on the
   resolved width). Otherwise VOID.
5. **CONTRACT.** The contract arm: the floor IDENTICAL, and `shared` at 32 IDENTICAL to claim-all at 32. Otherwise VOID:
   the pool mode changed tokens at matched width.
6. **DETERMINISM and IDENTITY.** Draw 1's `ca_a` serial against `ca_b`'s is IDENTICAL (`sc2_identity`). In each draw,
   `sh`'s serial against `ca_a`'s is IDENTICAL: a serial request decodes at bucket 1 on every server. Otherwise
   IDENTITY is UNREAD and nothing is licensed.

**Rows, ceilings and noise.** `sc2_reduce.row` per server, ladder and rate. As SC2e's rule, a ceiling on Q2 is the
largest rate whose row is VALID with attainment (TTFT ≤ 1 s and TPOT ≤ 100 ms) ≥ 0.95 in both draws. For `ca`, that
holds for both `ca` servers of each draw. Per draw and rate, `noise` = |attainment(ca_a) − attainment(ca_b)|.

**Outcome.** The verdict is SHARED_LICENSED iff every gate passes and all of:
- `sh`'s Q2 ceiling is strictly above `ca`'s;
- at every Q2 rate up to and including `ca`'s ceiling, in each draw, `sh`'s attainment ≥ the lower `ca` server's − 0.02;
- at `sh`'s ceiling, `sh`'s completion tokens per second ≥ 1.25 × `ca`'s at `ca`'s ceiling, in each draw: tokens/s at
  the knee;
- Q3 is read (PRESSURE holds), and at every Q3 rate, in each draw, `sh`'s attainment ≥ the lower `ca` server's −
  max(0.05, noise at that rate). The shared pool must not serve worse when reservations bind.

The tokens/s clause is kept on purpose. Any strictly higher Q2 ceiling on these rungs is already ≥ 1.33× the lower
rate, so it adds little beyond attainment. It guards against a ceiling won on attainment while completion tokens per
second lag.

The verdict is REGRESSED if, in both draws, `sh`'s attainment at any Q2 rate at or below `ca`'s ceiling is below the
lower `ca` server's − 0.05, or the same holds on Q3. It is NOT_LICENSED otherwise, with the failed clause named.

**Every gate fails closed.** A missing, None or empty input never passes:
- A server lost to the deadline makes the reading UNREAD, never SHARED_LICENSED. That holds only when the runner recorded
  the deadline skip (`can_run`'s `SKIPPED` line for that server or rung).
- A missing start or end `/health`, or a missing ladder run, WITHOUT a recorded skip is a crash or a lost server: VOID.
- QUALITY needs every plan present: P117's at B = 16, 32 and 64, and each server's at every one of those buckets it
  captures. A missing plan is QUALITY UNREAD.
- CONTRACT needs `shared`'s `kv_blocks_waits` recorded as exactly 0; a missing count is VOID.
- Each of these has a self-test case with its evidence removed.

**Quality.**
- The pool mode changes no token at matched width: the contract arm, gate 5.
- The width (64 against 32 decode rows) is lane P117's question, already read AT_PARITY under P110's bar on SC2e's
  int4 stack (`bench/p117/RESULTS-p117.md`, 2026-10-08): W64 −0.0032 nats, W32 +0.0006, inside the 16-row
  arithmetic's own perturbations. The stack has moved since (release 0.57.0), so the box re-runs P117's teacher-forced
  check at the launch stack as a QUALITY arm (`bench/p117/p117_box.py`, unchanged, buckets 32 and 64 against pieces of
  16), with P117's bar. QUALITY failing makes the verdict NOT_LICENSED (QUALITY), whatever the speed clauses read.
- The QUALITY arm runs the servers' attention plans although its table is narrower. `p117_box.py` builds its pool at
  its own window (P + C + 16 tokens), not 4,096, and the paged decode attention's split count reads the table width.
  It reads it only as an upper bound: grouped-nf4-gemm v0.50.0, `fp8_paged_attn.decode_split_plan` (lines 1292-1310),
  plans min(32, want, ceil(table tokens / 64)), with want = floor(11 × SMs / (5 × B × KV heads)).
  - On an RTX 5090 (170 SMs) with Qwen3's 4 KV heads, want is 5 at B = 16, 2 at 32 and 1 at 64.
  - P117's 656-token window gives a bound of 11 and 4,096 tokens gives 64. Both are above want at every bucket the
    QUALITY arm compares, so the plans are the same. The match holds for B ≥ 8 only (at B = 1 the bounds would give 11
    against 32), which P117 does not exercise, and it is cited for P117's widths alone.
  - The servers record their plans (`/health`'s `levers.attn_split_cap.plans`). The box computes P117's with the same
    library function at P117's table width and the box's SM count (`quality/plans.json`). The reducer asserts they
    match at B = 16, 32 and 64, wherever a server captured that bucket; otherwise QUALITY is UNREAD.
- Reported, no bar: per rate and draw, the share of requests whose streamed text is byte-equal between `sh` and
  `ca_a`. Under load, batch composition differs between servers, so this is descriptive only.

**What a licence licenses**, as a separate PR citing this read, and only what it read: `E4B_PAGED_KV_POOL=shared` as
the default for `E4B_PAGED_MAX_SEQS=auto` for ONE allowlisted pair, **model_type `qwen3_moe` with the int4 expert store
it served, on the sm_120 card class (the RTX 5090)**. It is a default-allowlist entry, like K25's plan allowlist, and
not a global default.
- Every other (family, card class) pair keeps claim-all until its own read.
- A later lane may widen the allowlist by reading a second family (a different architecture, e.g. a dense or GQA-heavy
  model) and a second card class (a smaller-memory card, where the shared pool's advantage should be largest).
- Nothing else is licensed, and never a bare width.

**Scope.** Qwen3-30B-A3B int4 through `serve_paged`, one RTX 5090, 4,096 tokens a slot, the two ladders above. 8,192
tokens a slot (claim-all 16 against shared 64 in the estimate) is a later lane, if this one licenses.

## Predictions (graded in the read, never ruling)

| # | prediction | basis | if MISSED |
|---|---|---|---|
| P1 | `ca`'s width 32 and `sh`'s 64 on the box | the estimate above | G3's arithmetic or the box's free memory differs from the estimate: the pool record names which |
| P2 | `sh`'s Q2 ceiling 12 req/s and `ca`'s 8, both draws | SC2e's measured s64a 12 and s32a 8 at 2,048 tokens a slot: Q2's requests (512 + up to 256 tokens) never approach either ceiling, so the per-slot ceiling should not move the step cost | the width does not carry over to 4,096 tokens a slot on this stack: the census names where |
| P3 | on Q3, `sh` counts block waits (`kv_blocks_waits` rising) at 2 and 4 req/s, both draws | 64 × the 200-block mean exceeds the pool | the long ladder did not exercise the reservation (at 4 req/s that is PRESSURE failing: Q3 UNREAD) |
| P4 | serial p50 TTFT and TPOT of `sh` within ±5 % of `ca`'s | a serial request runs bucket 1 on every server | the pool mode costs a single sequence time: name where in the step trace |
| P5 | VRAM at ready: `sh` − `ca` ≤ 1.5 GiB | both arms are sized to the same free memory less the same reserve and margin | G3 does not leave the margin it prices |

## Reported, no bar

The census per server (`sc2e_census.py` with the arm's buckets): decode-only steps by bucket, `dec_pieces`, prefill
steps, inter-step gaps, the mean requests in the server and the queue wait p50 per rate. From `/health`, the pool rows,
the reserved blocks and the waits per ladder. The pool's derivation as G3 recorded it.

## Shakedown, proof and budget (bench/common rules 2-5)

**$0 shakedown.**
- The self-tests of the reducer (`sc2j_reduce.py`, 33 cases), the contract arm (`sc2j_contract.py`, 10) and the long
  prompt pool (`sc2j_prompts.py`).
- The box's checks on fixtures (`tests/test_sc2j_box.py`): the POOL check executed on every departure, the tripwire,
  the lane's constants held equal across files, Guard Fit's static form, the cap and ids, and the wiring.
- `SC1_BOX=Q SC1_DRIVE_DRYRUN=1` at the launch commit, with and without the proof.

**Proof** (`sc2j-prove-*`, guard 1.0 h), on Granite-3.1-3b-a800m (NF4 store, SC1's `GR_ENV` unchanged):
- the self-tests;
- the contract arm on Granite at 32 slots;
- both arms' servers through POOL, SLOTS and the burst;
- a serial smoke, and a 16-request Poisson smoke on each ladder at its top rate, every request VALID.
WIDTH is not read in the proof: Granite fits 64 slots under both pools. The QUALITY arm is smoked on 8 windows.

**Reading** (`sc2j-5090-*`, guard 5.5 h):
- installs about 10 min; fetch and bake 25-40 min;
- the contract arm about 15 min; the QUALITY arm about 10 min;
- six servers at about 22-26 min each (start, warm and burst about 7; serial about 3; Q2 about 6; Q3 about 10: 360 s
  of arrivals plus the long generations' drain).
- The fit is therefore about 192-231 min. Guard Fit's static form needs boot + 1.25 × fit ≤ guard. With a 540 s boot
  term, 9 + 1.25 × 231 = 298 min, so the reading's guard is **5.5 h** (330 min, 32 min to spare). The proof's fit is
  about 25 min, so 9 + 1.25 × 25 = 40 min, under its 1.0 h guard.

**Run ids** (rule 2): proofs `sc2j-prove-1` and `sc2j-prove-2`; readings `sc2j-5090-1` and `sc2j-5090-2`; and ONE set
of retry ids `sc2j-r1` to `sc2j-r8`, shared across both kinds, for a box that follows one that did not start. That is
12 ids in all.

**Outcome classes** are bench/common rule 3's, decided from the launcher's receipt and the runner's exit.

**The started-box cap, by kind.** At most **4** boxes may START: at most 2 proofs and at most 2 readings, a re-run
counted as a reading. A proof never stands in for a reading.
- Worst cases at $1.05/h plus download at ≤ $0.011/GB: a proof 1.0 h ≤ $1.05 + ~$0.10 = $1.15; a reading 5.5 h ≤ $5.78
  + ~$0.83 = $6.61.
- A box that does not start costs, at most:
  - refused at create: $0;
  - NOT_RUN: about $0.18 (up to 600 s held before the pre-flight gives up, at $1.05/h, no download);
  - refused at the guard fit (rc 41): this fires only after the box has used more than guard − 1.25 × fit, so up to
    330 − 1.25 × 231 = 41 min on a reading, about $0.72, and 60 − 40 = 20 min on a proof, about $0.35.
- The lane's worst is bounded: the cap × each kind's worst, plus every other registered id (12, less the 4 that start)
  as a non-start at the reading's $0.72: 2 × $1.15 + 2 × $6.61 + 8 × $0.72 = **$21.28**.
- Each run is under the standing single-run tier ($15) and inside the launcher's 6 h cap.
- **Strikes.** A strike is a started box that is NOT read (HARNESS_ERROR, VOID, a fault mid-install or mid-run), or two
  consecutive refusals at the guard fit (rc 41). **Two strikes** stop the lane pending a root-cause note. A read box
  never counts toward the stop, and neither does a non-start.

**Boot slack.** Each box's guard satisfies Guard Fit's static form, boot + 1.25 × fit ≤ guard, with a boot term of at
least 540 s (`ssh_ready_s` 420 s plus 120 s of staging). `tests/test_sc2j_box.py` asserts the static form for the
proof and the reading, not a slack over the estimate.

**Frozen pins once closed (rule 6).** When the verdict is recorded, the lane's tests assert its staged hashes and
kernel pin against the read's merge commit (`SC2J_FROZEN`), never the live tree.

## Out of scope

- 8,192 tokens a slot, other models, families and cards.
- The decode lookahead (SC2h) and the prefill knobs (SC2f, SC2i): at their defaults here.
- A default flip: only a separate PR citing a SHARED_LICENSED read.

## Amendments

### Amendment 1 (registered before any SC2j reading, after the proof `sc2j-r3`): QUALITY at box Q's stack constants

**Why.** The proof `sc2j-r3` (machine 52780, 2026-10-23, $0.235) started and was NOT PROVED, on QUALITY alone. Everything
else passed: the installs and tripwires, every self-test, CONTRACT OK (shared IDENTICAL to claim-all, 96 requests and
79,313 tokens at 32 slots, `sh` waits 0, the floor identical), both servers' ROUTES, POOL, EARLY and BURST, the Q3 smokes
VALID, and the serial identity. QUALITY read VOID because `bench/p117/p117_reduce.py` holds two constants that name
P117's registration, not its method:

- the grouped-nf4-gemm commit P117 ran (`b4f93f1c`, v0.42.0). Box Q installs v0.50.0 (`41ef086e`), and `q_quality` did
  not export `GNF4_SHA` to `p117_box.py`, so the record carried none;
- P117's window count per model (Qwen3 64, Granite 40). The proof smokes 8 Granite windows.

As registered, QUALITY could never read on box Q, in the proof or in the reading. That is a kit fault, and the proof did
its job by catching it.

**The change.**
- `q_quality` exports box Q's `GNF4_SHA` to `p117_box.py`, which is unchanged and records `os.environ["GNF4_SHA"]`.
- The verdict comes from `bench/sc2/sc2j_quality.py`, staged beside `p117_reduce.py`. It runs `p117_reduce.reduce` with
  exactly two constants set to box Q's registered values: `GNF4_SHA` = `41ef086e6a3e2be5b158ff1bb38e0de67862a949`, and the
  windows, Qwen3-30B-A3B 64 (P117's own count; the reading) and Granite 8 (the proof's smoke). P117's bar (tol 0.01,
  spread 2x with a 0.005 floor), its floor draws, its engagement and FUNCTION checks and its model revisions are unchanged.
- It fails closed. A record whose grouped-nf4-gemm is not box Q's, a box that installed another commit, or a window
  count other than the registered one for its model is VOID, through P117's own checks. Its self-test (11 cases) refuses
  each of these with its evidence removed.
- `tests/test_sc2j_box.py` runs the change on `sc2j-r3`'s own QUALITY record (`tests/fixtures/sc2j_r3_quality/`). As
  recorded it reads VOID under P117's constants. With the exported `GNF4_SHA` it reads AT_PARITY at box Q's constants:
  FUNCTION 248 positions, 0 differ, under P117's unchanged bar. Each of the four refusals above turns it VOID.
- The kit gains one staged entry, `sc2j_quality.py`, and `sc2j_box_q.sh` changes; `staged.sha256` is regenerated. The
  launch commit is this amendment's merge.

**Strikes, ids and spend.** `sc2j-r3` is a started box that was not read: strike 1 of the two that stop the lane
(bench/common rule 3). Proofs started: 1 of 2. The next proof is `sc2j-prove-2`, the last proof the cap allows: it
follows a box that started, so it takes a proof id, not a retry id. A second strike stops the lane pending a root-cause
note. Spent so far: $0.251. Retry ids used: `sc2j-r1` to `sc2j-r3` (`sc2j-prove-1` and `sc2j-r1`/`r2` did not start: the
320 GB disk floor, then a ghost ask, then a vanished offer, all before any install).

**Unchanged:** every arm, gate, threshold, rate, guard, cap and cost; the outcome classes; the QUALITY plan match; and
QUALITY's place in the rule (anything other than AT_PARITY is NOT_LICENSED (QUALITY), whatever the speed clauses read).
SC2f, SC2h and SC2i are untouched: their kit is held at `7b8f1f69` (their own notes), and this change to the shared kit
means they launch from `7b8f1f69` itself.

### Amendment 2 (registered before any SC2j reading): the launch checkout takes rule 8's fetch fix

**Why.** bench/common rule 8: a fetch never imports links. The launch commit Amendment 1 registered (its merge,
`ea298854`) fetches with `rsync -az`, which keeps symlinks, so no box may be created from it. This change adds
`--no-links` to `sc1_drive.sh`'s `pull_box` rsync line.

**The change.** The launch checkout is a `main` commit at or after this amendment's merge, and the stack under test is
that checkout's `HEAD`, as before. Before the create, `python bench/sc2/sc2_kit_hold.py package-hold --base ea298854
--at HEAD --out <run>/package_hold.json` must read OK, and its record is kept with the run. It refuses unless the package
under test is Amendment 1's: the `experts4bit_qlora/` tree and the install metadata (`pyproject.toml`, `MANIFEST.in`)
are identical, the staged kit (`staged.sha256`) and `make_pin.sh` are byte-identical, and `sc1_drive.sh` differs at most
by rule 8's one token on `pull_box`'s rsync line. Its self-test removes or alters each, and `tests/test_sc2_kit_hold.py`
runs it on registered commits.

**Unchanged:** every arm, gate, threshold, rate, guard, cap and cost; strike 1 of 2; proofs started 1 of 2; the next
proof `sc2j-prove-2`.
