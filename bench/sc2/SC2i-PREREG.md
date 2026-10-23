# SC2i: does spacing or shortening prefill lift e4b `serve_paged`'s served decode under load at 64 slots? `E4B_PAGED_PREFILL_EVERY` and `E4B_PAGED_LOADED_CHUNK_TOKENS` (e4b#2017) against the release's defaults, on one RTX 5090 (lane SC2i of #846; serving roadmap lever A1)

**Registered** when this file merges to `main`, before any SC2i run. Box P rents after SC2f's and SC2h's boxes (their
order is SC2f Amendment 3's), on separate boxes, through the guarded launcher (`tools/pod-launch.sh`). The independent
deadline watcher is armed at create.

**The code under test** is e4b#2017's two scheduler knobs (`engines/scheduler.py`, `ContinuousScheduler(prefill_every,
loaded_chunk_tokens)`, read by `serve_paged` from the environment). Both apply only while a decode row is live:
- `prefill_every=N`: a step carries prefill work only if the previous `N - 1` steps carried none. The next chunk waits at
  most `N - 1` steps; a given prompt waits about `N` steps per chunk owed ahead of it.
- `loaded_chunk_tokens=T`: caps the chunk and the step's prefill budget at `T`. A first chunk shorter than the prefill
  graph's `T` runs eagerly (the runner counts it as `short_chunk`).

Their defaults (1, 0) compose exactly the previous plans, call for call (`tests/test_scheduler_interleave.py`).

## Why this lane

On this server a prefill chunk runs as its own forward inside a step, and every running decode waits it out. SC2 and
SC2c measured the per-chunk stall under load at tens of milliseconds, far above one decode step. The knobs trade a
little TTFT for fewer or shorter stalls on the decodes in flight. Whether the trade lifts the served TPOT at 8 and
12 req/s, without costing attainment or tail TTFT, is not known, and it is what this lane reads. Neither knob is a
default, and this lane licenses none: a FASTER arm goes to a quality read first, because chunking a prompt differently
can change its tokens.

## The box (P) and the stack it pins

- **One box, e4b only.** `SC1_BOX=P` (`bench/sc2/sc2i_box_p.sh`), sourced after `sc2_box_e.sh`, `sc2c_box_h.sh`,
  `sc2e_box_l.sh` and `sc2f_box_n.sh`, whose server start (`l_server_start`: ROUTES rc 47, SLOTS rc 49), early check
  (`h_early`), burst (`l_burst`, rc 48) and P130 tripwire (`n_tripwire`) it uses unchanged. `bench/sc1/sc1_run.sh` and
  `sc1_drive.sh` gain box P and nothing else.
- **e4b.** The launch commit is SC2f Amendment 3's: the first `main` commit containing `v0.57.0` and e4b#2011's merge.
  It must also contain e4b#2017's merge (box P's tripwire refuses without both knobs, rc 9). Before the create, SC2f
  Amendment 3's mechanical default check runs (`bench/sc2/sc2_defaults.py`, with the same two allowance rules).
- **grouped-nf4-gemm.** v0.50.0 (`41ef086e6a3e2be5b158ff1bb38e0de67862a949`), as boxes N and O.
- **Arms**, each SC2h's `base` server, which is SC2f's `on`: SC2e's s64a at the release's defaults (`SPEEDENV`,
  `E4B_PAGED_FUSE_QKV=1`, `ROUTEENV`, 64 slots, `E4B_PAGED_BUCKETS=auto`, 2,048 tokens a slot, chunk 512, bulk KV and the
  prefill graph at their defaults, the decode lookahead off, both traces on). They differ only in the two knobs:

  | arm | `E4B_PAGED_PREFILL_EVERY` | `E4B_PAGED_LOADED_CHUNK_TOKENS` | the start `/health` must read (KNOBS) |
  |---|---|---|---|
  | `base` (the release's defaults) | unset | unset | `prefill_every` 1, `loaded_chunk_tokens` 0 |
  | `pe2` | 2 | unset | 2, 0 |
  | `pe4` | 4 | unset | 4, 0 |
  | `lc256` | unset | 256 | 1, 256 |
  | `lc128` | unset | 128 | 1, 128 |

  The box unsets both knobs when it is sourced, so only an arm that names a knob sets it.
- **Order.** Draw 1: `base_a pe2 pe4 lc256 lc128 base_b`; draw 2: `base_a lc128 lc256 pe4 pe2 base_b`. The two `base`
  servers of a draw bracket its knob arms and measure the drift and the noise.
- **Per server:** 4 warm (seed 999), SC2e's burst, serial 24 (seed = draw), then the ladder at 4, 8 and 12 req/s, 120
  requests each (seed draw × 100 + rate). Every server of a draw runs the same seeds. At the end, `/health` and `/stats`
  are captured.
- **Image and checkpoint.** As SC2f's: `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, torch 2.8.0+cu128; Qwen3-30B-A3B @
  `ad44e77`, baked to the int4 arena on the box.
- **Host.** SC2f Amendment 2's bar and SC2f Amendment 3's host continuity: the same machine as SC2f's and SC2h's boxes
  while it stays listed, else this bar decides; no substitution outside it.

## The rule (`bench/sc2/sc2i_reduce.py`)

**Gates, per server** (a failure names the server and VOIDs the reading):
- ROUTES and SLOTS as SC2h's (`sc2c_reduce.routes_bad`, `sc2e_reduce.slots_bad`, arm s64a);
- ENGAGED: SC2e's check (`sc2e_reduce.engagement`, arm s64a) unchanged on `base`, `pe2` and `pe4`. On `lc256` and
  `lc128`, a first chunk capped below the prefill graph's T runs eagerly by design (`short_chunk`), and a 512-token
  prompt has later chunks (`later_chunk`). There the gate requires replays + `short_chunk` = the requests admitted,
  and every eager reason one of those two; the rest of SC2e's check runs unchanged (`sc2i_reduce.engagement_bad`).
  The adapted gate is tested on a real record, not only on hand-written ones: `tests/fixtures/sc2i_shake1/` holds the
  end `/health` of an `lc128` server from the EXPLORE shakedown `sc2i-shake-1` (box P's proof on Granite, one RTX
  5090), with sha256s against the receipt. That record has 12 replays + 78 `short_chunk` = 90 admitted, and
  `later_chunk` 234 = 78 × 3. The adapted gate passes it, and SC2e's unadapted check fails it;
- DEFAULTS: the start `/health` reads SC2f's `on` (`sc2f_reduce.flips_bad`) and the lookahead off;
- KNOBS: the start `/health`'s `engine.prefill_every` and `engine.loaded_chunk_tokens` are the arm's;
- PROMPTS: every request reports 512 prompt tokens.

**Statistics, per server:** from the driver's summaries, at 4, 8 and 12 req/s, the attainment (TTFT ≤ 1 s and TPOT ≤
100 ms), `tpot_p50_s`, `tpot_p99_s`, `ttft_p50_s` and `ttft_p99_s`; serial `tpot_p50_s` and `ttft_p50_s`; from the end
`/health`, `prefill_graph.eager_reasons.short_chunk`; from the end `/stats`, `scheduler.prefill_deferred`.

**Per draw and knob arm:**
- `ref` = the mean of the draw's two `base` servers;
- `noise` = |base_a − base_b| / `ref`, on TPOT p50 at 12 req/s;
- `bar` = max(0.05, `noise`);
- `ratio` = the arm's TPOT p50 at 12 req/s / `ref`'s.

**Outcome, per knob arm** (each arm read on its own; the reading's verdict lists all four):
- VOID if any server fails a gate. UNREAD if a needed number is missing.
- REGRESSED if, in both draws, `ratio` ≥ 1 + `bar`; or if, in both draws, the arm's attainment at 4, 8 or 12 req/s is
  below the lower `base` server's minus 0.05.
- FASTER if, in both draws, all of:
  - `ratio` ≤ 1 − `bar`;
  - the arm's attainment at each of 4, 8 and 12 req/s ≥ the lower `base` server's minus 0.02;
  - the arm's TTFT p99 at 12 req/s ≤ max(1.0 s, the higher `base` server's).
- FLAT otherwise.

The reducer licenses nothing. A FASTER arm is reported to the maintainer as a candidate for a quality read (K8 / P110's
bar on prompts chunked as the arm chunks them), and only that read can open a default question.

**Predictions** (per draw; HOLDS / MISSED / UNREAD; no outcome reads them):
- P1: `pe2`'s `ratio` in [0.75, 0.97].
- P2: `lc128`'s TTFT p50 at 12 req/s / `base`'s ≥ 1.00 (a shorter first chunk loses the prefill graph). It is read under load,
  not serially: with no decode live the cap never applies.
- P3: every knob arm's attainment at 4 req/s ≥ 0.95.
- P4: `lc256`'s and `lc128`'s `short_chunk` > 0 and `base`'s = 0.

**Reported, no bar:** TPOT p99 and TTFT p50 at 8 and 12 req/s for each arm against `ref`; `prefill_deferred` per server;
per draw the drift `base_b − base_a`; the text agreement of each knob arm with `base_a` at 12 req/s (serially the knobs
never act, so serial texts would agree by construction), beside `base_a` against `base_b` at 12 req/s as its floor:
the same seeds, but a different schedule.

## Shakedown, proof and budget

**$0 shakedown:** the reducer's self-test, the box's checks on fixtures (`tests/test_sc2i_box.py`), and
`SC1_BOX=P SC1_DRIVE_DRYRUN=1` on the controller checkout at the launch commit.

**Proof** (`sc2i-prove-*`, guard 1.0 h), on Granite-3.1-3b-a800m (NF4 store, SC1's `GR_ENV` unchanged): the self-tests,
three servers (`base`, `pe2` and `lc128`) at 64 slots through ROUTES, SLOTS, KNOBS (proof form: Granite engages neither
P130 knob), warm, `h_early` and the burst, a 6-request serial smoke and a 16-request Poisson smoke at 12 req/s, every
request VALID. Each server's end `/health` then goes through ENGAGED as the reducer reads it. `lc128`'s is the real
record the adapted gate is checked on, before any reading box. The step census (`sc2i_reduce.census_bad`) is SC2e's
`--require-wide` on `base` and `pe2`: the 64-row decode step, and at least 26 prompts counted by prefill-graph replay.
On `lc128` most first chunks run eagerly by design, so the replay count cannot reach 26 (the shakedown's `lc128`
counted 12); there only the 64-row decode step is required. A `base` or `pe` arm with that census still fails.

**Reading** (`sc2i-5090-*`, guard 4.5 h): installs about 10 min, fetch and bake 25–40 min, twelve servers at about
12–14 min each: about 179–218 min, so a 4.0 h guard would leave 11 % headroom.

**Worst cases** at $1.05/h plus download at ≤ $0.011/GB:
- a proof: 1.0 h ≤ $1.05 + about $0.10 = $1.15;
- the reading: 4.5 h ≤ $4.73 + about $0.73 = $5.46;
- the lane, as SC2f's and SC2h's (two proofs, one reading, one re-run counted as a reading): 2 × $1.15 + 2 × $5.46 =
  **$13.21**.

Each run is under the standing single-run tier ($15), and every box sits inside the launcher's 6 h cap. Two
consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane's rentals pending a root-cause note. A harness or
infrastructure failure before the lane's first arm starts is not this lane's strike. The deadline drops draw 2's last
servers first; a server lost to the deadline makes the reading UNREAD, never FASTER.

## Out of scope

- The decode lookahead (SC2h) and P130's prefill knobs (SC2f): both at their defaults here.
- Quality: any FASTER arm gets its own quality read before a default question.
- Other models, families and cards; 4,096 tokens a slot; prompts other than 512 tokens.
- A second prefill graph at a smaller chunk: if `short_chunk` costs the TTFT, that is its own lane.

## Amendments

### Amendment 1 (registered before any SC2i run): which commit is the launch commit

As SC2f Amendment 4, which corrects the same sentence. **The box (P)** above names the launch commit as "SC2f Amendment 3's:
the first `main` commit containing `v0.57.0` and e4b#2011's merge". Read literally, that is `75ddc16e`, which predates
e4b#2017's knobs (box P's tripwire would refuse it) and box P itself (e4b#2027). The launch commit is a `main` commit at
or after `1a25b209`. Before the create, SC2f Amendment 3's mechanical default check runs at it against `v0.57.0`, with
the same two allowance rules, and its output is recorded with the run.

**The kit is held at `1a25b209`**, as in SC2f Amendment 4. At the launch commit, every `bench/sc1/staged.sha256`
entry that box P stages, the shared `bench/sc1/sc1_run.sh` included, must be byte-identical to that entry at
`1a25b209`. The comparison is recorded with the run, and any difference is its own pre-run amendment.

**Unchanged:** every constant, threshold and cost; the arms, the order, the gates, the rule, the predictions, the box,
the proof, the guards, the budget and the $0 shakedown.


### Amendment 2 (registered before any SC2i run): the kit is held at e4b#2041's merge, for the harness's brotli pin and fetch retry

As SC2f Amendment 5, for box P. The kit is held at `7b8f1f69` (e4b#2041's merge), replacing `1a25b209` for the KIT only.
At the launch commit, every `bench/sc1/staged.sha256` entry, plus `bench/sc1/sc1_drive.sh` and `bench/sc1/make_pin.sh`,
must be byte-identical to `7b8f1f69`'s. The stack under test stays `36cdd2d4`: the launch command sets `E4B_SHA=36cdd2d4e031e8365ab52e495c40cd79043edd49`.
Against `1a25b209` the kit differs as SC2f Amendment 5 declares. In `sc1_run.sh`: the `"brotli>=1.2.0"` pin and the
single fetch retry, both behavioural, and box Q's wiring, none of it reachable with `SC1_BOX` = P. Box P's own
dispatch line is byte-identical to `1a25b209`'s. Seven new box Q files are staged on box P, and nothing box P runs
sources or calls them. Before the create, `bench/sc2/sc2_kit_hold.py check --at <launch commit>` runs from a `main` checkout that
carries this amendment, with `--e4b-sha "$E4B_SHA"` (its STACK gate refuses any launch whose `E4B_SHA` is not
`36cdd2d4` in full), fails closed,
and its record is kept with the run.


**The stack is a VOID condition of the rule**, as in SC2f Amendment 5. A box P verdict stands only when
`sc2_kit_hold.py verdict-stack` reads OK on the fetched run (`versions.txt`'s installed e4b commit and every
`KNOBS ... e4b=` line of `summary.txt` both `36cdd2d4`); otherwise the read is VOID. `verdict_stack.json` is a
required results file: a results PR or re-derivation without it, or with it not OK, reads VOID.
**Unchanged:** every constant, threshold and cost; the arms, the order, the gates, the rule, the predictions, the box,
the proof, the guards, the budget and the $0 shakedown.
