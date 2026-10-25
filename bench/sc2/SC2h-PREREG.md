# SC2h: does P118's decode lookahead lift e4b `serve_paged`'s capacity at 64 slots? `E4B_PAGED_DECODE_LOOKAHEAD=1` against the release's default (off), on 0.54.0's s64a server, paired on one RTX 5090 with same-seed replicates, the bucket-64 step's saving against its own host gap, and a knee block placed by SC2f's reading (lane SC2h of #846; registered 2026-10-11)

**Registered** when this file merges to `main`, before any SC2h run. Opened without `ready-to-merge` or auto-merge,
for maintainer review. Box O rents **after SC2f's reading** (it needs SC2f's verdict, below), behind P131 and P132 in
the maintainer's queue, on **e4b 0.54.0** as SC2f, and every paid ACK names the host.

**The code under test** is P118's opt-in decode lookahead (#1339, merged `5a1e36e9`;
`bench/p118/RESULTS-p118.md`). Each decode step is issued before the previous one is read back, from the
device-resident newest tokens, so the GPU has a step queued while the host emits, retires, plans and copies inputs in.
The scheduler's contract (`engines/scheduler.py`) is that sequences, tokens and finish reasons are the synchronous
path's; a sequence that stops on a stop id frees its slot one step later. P118 read it at **one request** on the NF4
server: ×1.0198 against a 1.02 bar (SLOWER), with identical tokens. There the gap between steps was 0.21 ms of 7.2.

**Recorded at launch** (read from the objects, not recalled; in the launch note and the read): the release tag, its
commit and `experts4bit_qlora/` tree, the launch commit and its tree, which must equal the tag's; SC2f's verdict file
and its sha256; the knee rates the rule resolved from it.

## Why this lane

**The census** (`bench/h2h-2026-10-02/sc2e/decode64/`, #1639). SC2e's served bucket-64 decode step is 18.392 / 18.273 ms
(two draws). It closes exactly into host before the replay (0.879 / 0.932), the device replay (15.906 / 15.800) and the
host tail (1.607 / 1.541). The exposed host gap, **≤ 2.47 ms a step**, ranked first of four levers by what it can still
save. Its largest piece is the engine's per-stream token dispatch: one `call_soon_threadsafe` per live stream per step,
1.2 ms at 64 slots. The host gap stays below the device replay at every bucket (0.48 vs 3.86 ms at one row, 2.49 vs
15.91 at 64), which is the condition under which the lookahead can hide all of it.

**Why it was not read here before.** At one request the gap is about a twelfth of what it is at 64 slots. P118's
SLOWER says nothing about a 64-slot server.

**What changed on the decode step since SC2e** (the census's section 4): P122, P124 and P126 were read at 64 rows; P2's
lean dispatch and P127 Phase 2 reach the 64-row step too, bitwise and unread there. So **this lane takes `base`'s step
and host gap from its own servers, not from SC2e**.

## The model's answer, before data

`bench/sc2/sc2h_basis.py model`: `serve_capacity` on SC2e's s64a costs with P130's 5.18 ms prefill saving (SC2f's `on`,
this lane's `base`); `la` removes the exposed host gap from every decode step. That gap is the least squares on the
bucket over SC2e's decode-only steps, 0.506 + 0.0353 ms a row (draw 1) and 0.514 + 0.0349 (draw 2): the same fit the
model's own step costs use. This is a ceiling. It assumes the lookahead hides all of the gap at every bucket and adds no
device time.

| | d1 base | d1 la | d2 base | d2 la |
|---|---|---|---|---|
| decode step at 64 rows (model's line) | 19.87 | 17.10 | 19.71 | 16.96 |
| ceiling (ladder, one plan a rate) | 14 | 14 | 12 | 12 |
| knee block 12, 13, 14 | 0.953 | 0.983 (+0.030) | 0.954 | 0.981 (+0.027) |
| knee block 13, 14, 15 | 0.882 | 0.938 (+0.057) | 0.900 | 0.955 (+0.055) |
| knee block 15, 16, 17 | 0.742 | 0.830 (+0.088) | 0.700 | 0.796 (+0.096) |
| knee block 16, 17, 18 | 0.709 | 0.797 (+0.088) | 0.637 | 0.714 (+0.078) |
| knee block 4, 8, 12 | 1.000 | 1.000 (+0.000) | 0.994 | 0.999 (+0.005) |

- **la/base ratios** (both draws): serial TPOT 0.890 / 0.889; serial TTFT 1.000; TPOT p50 at 8 req/s 0.683 / 0.793; at
  12 req/s 0.855 / 0.883.
- **With the decode device 15 % faster** (0.54.0 carries P122, P124 and P126, which SC2e's costs do not; the host gap
  unchanged): the knee gains range +0.002 to +0.125 across the same blocks. `la`'s ceiling reaches 16 / 14 against
  `base`'s 14 / 12.
- **The gain is largest where the block straddles `base`'s crossing of 0.95**, which is why the block is placed from
  SC2f's reading of the same server rather than from the model.

## The box (O) and the stack it pins

- **One box, e4b only.** `SC1_BOX=O` (`bench/sc2/sc2h_box_o.sh`), sourced after `sc2_box_e.sh`, `sc2c_box_h.sh`,
  `sc2e_box_l.sh` and `sc2f_box_n.sh`, whose server start (`l_server_start`: ROUTES rc 47, SLOTS rc 49), early check
  (`h_early`), burst (`l_burst`, rc 48) and P130 tripwire (`n_tripwire`, rc 9) it uses unchanged. `bench/sc1/sc1_run.sh`
  and `sc1_drive.sh` gain box O and nothing else.
- **e4b.** As SC2f: the launch commit is `main` at or after the `v0.54.0` tag, with that tag's `experts4bit_qlora/` tree.
  Box O's tripwires refuse (rc 9) an e4b without P130's two defaults (`n_tripwire`) or without P118's lookahead off by
  default (`o_tripwire`: `PagedServeConfig.decode_lookahead` defaulting False, the runner's `issue_decode` /
  `collect_decode`, the scheduler's `lookahead`).
- **grouped-nf4-gemm.** Box N's pin (v0.45.0, `724ccc45`), and it moves with box N's: an amendment that moves SC2f's pin
  moves this one.
- **Arms**, each SC2f's `on` server (SC2e's s64a at the release's defaults: `SPEEDENV`, the three folds,
  `E4B_PAGED_FUSE_QKV=1`, `ROUTEENV`, 64 slots, `E4B_PAGED_BUCKETS=auto`, 2,048 tokens a slot, chunk 512, bulk KV, the
  prefill graph and P130's two prefill defaults, both traces on), differing only in the lookahead:

  | arm | `E4B_PAGED_DECODE_LOOKAHEAD` | the start `/health` must read (LOOKAHEAD, rc 44) |
  |---|---|---|
  | `la` | 1 | `engine.decode_lookahead` true; SC2f's `on` FLIPS reading (P130's two defaults engaged) |
  | `base` (the release's default) | unset | `engine.decode_lookahead` false; the same FLIPS reading |

  The box unsets the knob when it is sourced, so only `la` names it. The reducer also reads every server's step trace:
  each decode-only step of an `la` server carries the lookahead note (`tr.note(bucket=b, lookahead=1)`), and no step of
  a `base` server does.
- **Image, checkpoint and host**: as SC2f (Qwen3-30B-A3B @ `ad44e77` baked to the int4 arena; the ACK pins the host;
  SC2f's host preferred when offered, so the two lanes' `base` and `on` servers share a host).

## The knee rates: SC2f's reading, by a rule registered now

The knee block is placed where `base` crosses 0.95, read from SC2f's `on` arm (the same server on the same release), by
`sc2h_reduce.knee_rates` on SC2f's verdict (`arms.on.rows`: four servers' attainment at each of its ladder rates, 4, 8,
12, 13, 14, 16):
- **r\*** = the lowest ladder rate whose four-server **mean** attainment is below 0.95; the block is **r\* − 1, r\*,
  r\* + 1**;
- **never below 0.95**: the top rate and the two above it, **16, 17, 18**;
- **below 0.95 at the lowest rate**: the bottom three, **4, 8, 12**;
- **SC2f VOID or UNREAD, or an `on` row not VALID**: no rates, and **SC2h does not launch**.

`sc1_drive.sh` runs the rule on the controller before any rental (`SC2H_SC2F_VERDICT`, the committed `verdict_sc2f.json`)
and stages the file. Box O runs it again (`o_knee`, rc 45 on refusal), writes `knee_rates.json` with the verdict's
sha256, and the reducer re-derives it (gate KNEE). The ladder is SC2f's plus any knee rate it lacks.

## The instrument

As SC2f's: SC2's driver and prompts unchanged (512-token wikitext-2 rows, `max_tokens` U[64, 256], greedy, `ignore_eos`,
streaming, open loop). With `ignore_eos` and fixed `max_tokens` no lookahead token is discarded.
- **Eight servers, each a cold start.** Draw 1 runs `la_a`, `base_a`, `base_b`, `la_b`; draw 2 runs `base_a`, `la_a`,
  `la_b`, `base_b`. Every server of a draw runs the same seeds: serial = the draw; ladder = draw × 100 + rate; knee =
  10,000 × draw + 100 × rate + j.
- **Per server:** 4 warm serial requests; the burst; serial, 24 requests; the ladder (4, 8, 12, 13, 14, 16 and any new
  knee rate), 120 requests each; then the knee block: the three knee rates at four more seeds each, 120 requests each.

## The rule (`bench/sc2/sc2h_reduce.py`, 38 self-test cases)

**Gates, per server** (a failure names the server and VOIDs the reading): ROUTES, SLOTS (s64a), LOOKAHEAD (above,
start /health and step trace), ENGAGED (`sc2e_reduce.engagement` at 64 slots, every admitted request counted; the
lookahead's issued steps count their replays as the synchronous path does), PROMPTS (512 tokens); and once per run,
KNEE (the rates equal the rule's on the staged verdict, whose sha256 they record).

**Statistics.**
- Rows, ceilings and knees are SC2f's (a row meets capacity iff its four servers each attain ≥ 0.95; a server's knee is
  its mean attainment over the knee block's 15 runs).
- **A server's step** = the mean `step_ms` over its decode-only, one-piece bucket-64 steps, at least 100 of them.
- **The ceiling, registered before the bar:** a `base` server's **host gap** = the mean over the same steps of
  `gpu.dec_prep + step_ms − gpu.dec_issue`, the time the GPU is not replaying. SC2e's 2.47 ms is the prior only; this
  lane reads its own. (Under the lookahead those events mark an enqueue, so the host gap is read on `base`.)
- **The null, registered before the bar:** the same-arm replicate spread. Per draw, **step noise** = the larger of
  |la_a − la_b| and |base_a − base_b| of the step; **noise** = the same of the knee.
- **The bars.** Per draw: **saving** = mean `base` step − mean `la` step; **step bar** = max(0.25 ms, step noise), where
  0.25 ms is a tenth of the ceiling's prior. **gain** = mean `la` knee − mean `base` knee; **bar** = max(0.03, noise);
  **headroom** = 1 − mean `base` knee.

**Outcome**, decoupled from Q1–Q6:
- **VOID** if a gate fails; **UNREAD** if a needed number is missing.
- **TOKENS_DIFFER** if, in any draw and replicate, the `la` server's serial texts differ from its `base` partner's.
  Consequence: the lookahead breaks its contract at 64 slots. To the maintainer; the knob stays opt-in, and the next
  candidate is the fallback below.
- **REGRESSED** if `la`'s ceiling is below `base`'s; or, at 4 or 8 req/s, an `la` server attains below the draw's lower
  `base` server minus 0.05; or gain ≤ −bar in both draws; or saving ≤ −step bar in both draws. Consequence: to the
  maintainer; the knob stays opt-in. The read prints the knee and step numbers beside the trigger.
- **FASTER** if saving ≥ step bar **and** gain ≥ bar, in both draws. Consequence: **the reducer licenses nothing.** The
  read recommends the default-on flip for graph servers, and the flip comes to the maintainer as its own small PR with
  the receipts. P118's ×1.0198 at one request stands as the no-regression read there.
- **KNEE_SATURATED** if neither of the above and headroom < bar in either draw: the block could not show a gain. The
  step saving is recorded, no capacity claim.
- **STEP_ONLY** if saving ≥ step bar in both draws (the knee did not clear). **Licenses nothing default-on:** the knob
  stays opt-in and the saving is recorded.
- **FLAT** otherwise. The host gap is then not what stalls the step at 64 slots; the read says which segment the census
  shows instead.

`outcome.above_ceiling` flags a draw whose saving exceeds its ceiling plus the step noise. The lookahead hides host
time; it should not change device time, so a saving above the ceiling is reported to the maintainer as unexplained.

**Reported beside the no-regression rule:** TPOT p99 at 8 and 12 req/s, la/base. The lookahead can trade tail for mean.

**Predictions**, per draw (an arm's value is the mean over its two servers in the draw), each with what a miss means.
No outcome reads them.

| # | prediction | basis | if MISSED |
|---|---|---|---|
| Q1 | saving ∈ [1.2, 2.7] ms | the census's ceiling 2.47; the lookahead hides most of it | low: host work stays on the critical path (the census then names which segment); high: Q2's flag |
| Q2 | saving ≤ ceiling + step noise | the lookahead hides host time only | the lookahead changed device time: reported, unexplained |
| Q3 | serial p50: TPOT la/base ∈ [0.85, 1.00]; TTFT la/base ∈ [0.97, 1.03] | model 0.889–0.890 and 1.000 | TPOT high: one-row steps keep their gap; TTFT out: the lookahead touched prefill |
| Q4 | gain ∈ [+0.01, +0.12] | model +0.027 to +0.096 on the blocks the rule can choose | low: FLAT or STEP_ONLY; high: the model under-prices the host gap at the knee |
| Q5 | `la`'s ceiling ≥ `base`'s (over the reading) | model: equal on SC2e's costs, +2 with a faster device | below: REGRESSED |
| Q6 | TPOT p50 at 8 and 12 req/s, la/base ∈ [0.65, 0.95] | model 0.683–0.883 | high: the saving does not reach loaded TPOT |

## Reported, no bar: the census

Per server, from its own traces (`sc2e_census` with s64a's buckets): decode-only steps by bucket (step; on `base` the
device time and host segments), prefill steps, `dec_pieces`, busy-loop gaps, per-workload queueing, VRAM and
`free_after_mib`. Per draw and replicate, the text agreement between paired servers under load. Batch composition
differs with timing, so agreement below 1 under load is expected. The serial run is TOKENS' gate.

**Post hoc** (the read PR; no rule reads it): `serve_capacity` on each server's own fitted costs beside its measured
attainment.

## The fallback, not bundled: one wakeup a step

The census's dispatch segment, 1.2 ms of the 2.47 at bucket 64, is the engine waking the event loop once per live
stream per step. Handing every stream its tokens in one `call_soon_threadsafe` a step would remove most of it with no
device change and no change to tokens. It is a separate, smaller, code-only candidate. It gets its own registration
only if the lookahead reads TOKENS_DIFFER or REGRESSED or is refused by the maintainer. It is not part of either arm
here.

## Shakedown, proof and budget

**$0 shakedown** before any paid run: the reducer's and basis's self-tests, the box's checks on fixtures
(`tests/test_sc2h_box.py`), and `SC1_BOX=O SC1_DRIVE_DRYRUN=1 SC2H_SC2F_VERDICT=<SC2f's verdict>` on the controller
checkout at the launch commit. The launch is from a commit whose CI is green.

**Proof** (`sc2h-prove-*`, guard 1.0 h), on Granite-3.1-3b-a800m (NF4 store, SC1's `GR_ENV` unchanged; no SC2f verdict
needed):
- self-tests on the box;
- both arms' servers at 64 slots: ROUTES (proof form), SLOTS, LOOKAHEAD in its proof form, warm, `h_early`, the burst
  with bucket 64 replayed, a 6-request serial smoke, and 16-request Poisson smokes at 13 req/s at a ladder seed and a
  knee seed, every request VALID;
- serial IDENTICAL `la` against `base`. Granite's decode graphs run the lookahead, so this is the contract's first check
  at 64 slots;
- engagement as the reducer reads it; step traces with 64-row steps and the lookahead notes on `la` only.

**Reading** (`sc2h-5090-*`, guard 4.0 h): installs ~10 min, fetch and bake 25–40 min, eight servers at ~14–17 min each
(up to two new ladder rates when the block is 16, 17, 18).

**Guards**, at ≤ $0.75 an hour plus download at $0.0078/GB:
- **proof:** 1.0 h ≤ $0.75 + ~$0.10;
- **reading:** 4.0 h ≤ $3.00 + ~$0.55.

Lane ceiling $7.50 (two proofs, one reading, one re-run). Each run sits under #846's standing no-ask tier for a single
run under $15. Two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane's rentals pending a root-cause note. The
deadline drops draw 2's last servers first; a server lost to the deadline makes the reading UNREAD, never FASTER or FLAT.

## Out of scope

- Comparators; other models, families and cards; prompts other than 512 tokens; slots other than 64.
- The dispatch fix (above), and every other lever the census ranked: `Int4Linear` above 16 rows, K19 at 512 rows, the
  tile table.
- The lookahead with speculative decoding: the two refuse to combine.
- Quality beyond the serial TOKENS gate. The lookahead's device inputs are the synchronous path's by construction
  (P118's identical tokens at one request).

## Amendments

None yet.

### Amendment 1 (registered before any SC2h run): the release under test, the kernels pin, the price ceiling, a new owner, and the order of three lanes

**Why.** SC2h was registered on e4b 0.54.0 and never read; no SC2h box was ever created. The lane changed owner on
2026-10-23 and now runs as part of the serving roadmap, where the decode lookahead at 64 slots is lever F2. The release pin moves with SC2f's. **This changes the system under test.**
The 0.54.0 pin was never read.

**The changes**, as SC2f's Amendment 3, which this lane follows step for step:
- **e4b.** The launch commit is the first `main` commit containing both `v0.57.0` and e4b#2011's merge, with the same
  record and the same mechanical default check (`bench/sc2/sc2_defaults.py`: the resolved levers and the source scan, at
  `v0.57.0` and at the launch commit). Any diff refuses unless it passes under SC2f Amendment 3's two allowance rules,
  `--allow LEVER=reason`. It must be the commit SC2f's reading ran, unless SC2f's
  run recorded a refusal that a later commit fixes; a different commit is named in the launch note.
  - e4b#2011 removes most of the per-stream dispatch from the host gap this lane's lookahead hides, so this lane's `base`
    arm has less gap to hide than the registration's model assumed.
  - The model's numbers above are SC2e's and stay a ceiling. No outcome cut moves, and the read reports the `base` arm's
    measured host gap beside the saving.
  - **What moved under the arms.** The serving defaults that changed since 0.54.0 are listed in SC2f Amendment 3. On this
    lane's int4 s64a server the int8 head at one request, the attention split cap and #2011 reach the step; the NF4 routes
    do not. All of them apply to both arms alike, since none is P118's knob. So `la` against `base` still reads the
    lookahead alone, against a host gap that #2011 has shrunk.
- **grouped-nf4-gemm.** v0.50.0 (`41ef086e`), as SC2f's.
- **Price ceiling.** $1.05 an hour.
  - Worst cases at $1.05/h plus download at ≤ $0.011/GB: a proof, 1.0 h ≤ $1.05 + ~$0.10 = $1.15; the reading, 4.0 h ≤
    $4.20 + ~$0.73 = $4.93.
  - Lane total, as registered (two proofs, one reading, one re-run), with the re-run counted as a reading: 2 × $1.15 +
    2 × $4.93 = **$12.16** (was $7.50 at $0.75/h).
  - Each run is under the standing single-run tier ($15).
- **Host and order.** The host continuity rule, the separate boxes, the per-lane verdict and strike accounting, the watcher
  at create and the 48-hour screen deadline are SC2f Amendment 3's.
  - Box O launches only after SC2f's verdict is committed: `sc1_drive.sh` stages it as `SC2H_SC2F_VERDICT`, as
    registered.
  - The box launches through the guarded launcher (offer freshness and the same-pin ghost retry).
- **Run ids.** `sc2h-prove-1` onward and `sc2h-5090-1` onward, as registered.

**Unchanged:** the arms, the knob, the knee rule on SC2f's verdict, the ladder, the statistics, the predictions, the
outcomes, the box, the proof, the guards and the $0 shakedown.

### Amendment 2 (registered before any SC2h run): which commit is the launch commit

As SC2f Amendment 4, which corrects the same sentence. Amendment 1 above names the launch commit as "the first `main`
commit containing both `v0.57.0` and e4b#2011's merge". Read literally, that is `75ddc16e`, which predates box O's
grouped-nf4-gemm v0.50.0 pin and `bench/sc2/sc2_defaults.py` (both e4b#2023). The launch commit is a `main` commit at or after
`1a25b209`, the commit SC2f's reading ran unless SC2f's run recorded a refusal that a later commit fixes, as Amendment 1
says. The launch-time `sc2_defaults.py diff` against `v0.57.0`, under SC2f Amendment 3's two allowance rules, is recorded
with the run.

**The kit is held at `1a25b209`**, as in SC2f Amendment 4. At the launch commit, every `bench/sc1/staged.sha256`
entry that box O stages, the shared `bench/sc1/sc1_run.sh` included, must be byte-identical to that entry at
`1a25b209`. The comparison is recorded with the run, and any difference is its own pre-run amendment.

**Unchanged:** every constant, threshold and cost, and everything Amendment 1 left unchanged.


### Amendment 3 (registered before any SC2h run): the kit is held at e4b#2041's merge, for the harness's brotli pin and fetch retry

As SC2f Amendment 5, for box O. The kit is held at `7b8f1f69` (e4b#2041's merge), replacing `1a25b209` for the KIT only.
At the launch commit, every `bench/sc1/staged.sha256` entry, plus `bench/sc1/sc1_drive.sh` and `bench/sc1/make_pin.sh`,
must be byte-identical to `7b8f1f69`'s. The stack under test stays `36cdd2d4`: the launch command sets `E4B_SHA=36cdd2d4e031e8365ab52e495c40cd79043edd49`.
Against `1a25b209` the kit differs as SC2f Amendment 5 declares. In `sc1_run.sh`: the `"brotli>=1.2.0"` pin and the
single fetch retry, both behavioural, and box Q's wiring, none of it reachable with `SC1_BOX` = O. Seven new box Q
files are staged on box O, and nothing box O runs sources or calls them. Before the create,
`bench/sc2/sc2_kit_hold.py check --at <launch commit>` runs from a `main` checkout that
carries this amendment, with `--e4b-sha "$E4B_SHA"` (its STACK gate refuses any launch whose `E4B_SHA` is not
`36cdd2d4` in full), fails closed, and its record is kept with the run.


**The stack is a VOID condition of the rule**, as in SC2f Amendment 5. A box O verdict stands only when
`sc2_kit_hold.py verdict-stack` reads OK on the fetched run (`versions.txt`'s installed e4b commit and every
`KNOBS ... e4b=` line of `summary.txt` both `36cdd2d4`); otherwise the read is VOID. `verdict_stack.json` is a
required results file: a results PR or re-derivation without it, or with it not OK, reads VOID.
**Unchanged:** every constant, threshold and cost, and everything Amendments 1 and 2 left unchanged.

### Amendment 4 (registered before any SC2h run): the HOLD gate's declared exceptions

As SC2f Amendment 6, for box O, which names the full list in one place. Rule 8's one `--no-links` on
`pull_box`'s rsync line. Box Q's own `sc2j_<name>` files, changed or added and staged by name, with INERT re-run
at the launch commit. A staging checkout that is dirty, or whose `HEAD` is not the launch commit, refuses. Anything else refuses. The staged kit's meaning for box O and
the stack under test (`36cdd2d4`) are unchanged.

**Unchanged:** everything else.

### Amendment 5 (registered before any SC2h run): SC2f Amendment 7's qualify-at-proof gate and release pair

**Why.** SC2f Amendment 7 replaces the receipt bar with qualify-at-proof and moves the release pair to e4b 0.58.1 and
grouped-nf4-gemm 0.51.0. SC2f's committed verdict is read here, so the three lanes (SC2f, SC2h, SC2i) must sit on the same pair and the same gate.

**The change.** SC2h takes SC2f Amendment 7, by reference, in full:
- **(a) the gate.** Box O launches through `bench/sc2/sc2_qualify_drive.sh`. Before any install: the enforced
  power read (575 W, as a float; rc 17) and the `card_clock.py` burn (floor 2700 MHz; rc 18). After a clean proof,
  the proof's host records (rc 18). Admission is `bench/sc2/sc2_qualify.py admit --lane sc2h`: the same walk (52780,
  then 145701, then 55337), the same exclusions (13828, 37367; 26157 until a create there succeeds; the launcher's
  power exclusions at create), the same stability rule (3 consecutive reads, ≤ $1.05/h, ≥ 320 GB). Host continuity
  puts the earlier lanes' PROVED host first.
- **(b) attempts.** Every create consumes one of **3** proof attempts (`sc2h-prove-1` to `-3`), a qualify refusal
  included. A host tried is never revisited unless its proof PROVED. Three creates and no PROVED: the lane parks. The
  worst case per run is as SC2f Amendment 7 counts it (a proof $1.18, a reading $4.44), each under $15.
- **(c) the deadline.** One 72 h deadline for the three lanes, counted from SC2f Amendment 7's merge or the G3 GPU
  check's PASS, whichever is later. With no qualifying proof by then, SC2f, SC2h and SC2i park, with no further change
  to the bar.
- **(d) the release pair.** `E4B_SHA` = `605d21e67958fa94d5e73fc3a95ffcb5c8eae02d` (v0.58.1), and box O on
  grouped-nf4-gemm v0.51.0 (`30adb1bf`) through `sc1_run.sh`. The kit hold's STACK gate, `verdict-stack` and
  HOLD exception 3 are SC2f Amendment 7's. **The three lanes sit on the same pair. If one lane later moves alone, its
  cross-lane read is VOID.**
- **(e) order.** The G3 GPU check (e4b#2133) must PASS before the first SC2f proof creates, and so before any SC2h create.
  The lane order, SC2f then SC2h then SC2i, is unchanged.

**Unchanged:** everything else this lane registered; every constant, threshold and cost not named above.
