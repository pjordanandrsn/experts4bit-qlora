# P234: does `E4B_NF4_SMALLM_BW=1` decode Mixtral-8x7B's DEFAULT NF4 server faster at 2 and 4 requests, at no measurable quality cost, read against a bucket that is actually inert? On one RTX 5090 (serving roadmap lever B, step B3 re-registered; registered before any run)

Lane number claimed by `p234/registration`. Issue: #1313. Follows P231 (the same question, FUNCTION_FAIL:
`bench/p231/RESULTS-p231.md`), P227 (the per-row route against K25 at 2 to 8 tokens, a microbench:
`bench/p227/PREREG-p227.md`) and P141 (K33 at Mixtral's shapes at one request, LICENSED: `bench/p141/RESULTS-p141.md`).
Its kit is P231's (`bench/p231/`) by named edits, listed under "What changes from P231".

## Why this lane

**What P231 read.** P231 asked this lane's question on the same server, card, model and kernels, and read
FUNCTION_FAIL: B1's tokens differed from B0's on W16, a workload it had registered as inert, in both pairs. FUNCTION_FAIL
is read first, so its gain and quality rungs were never graded, and the switch stayed opt-in "until it is explained".

**Why W16 moved.** P231's results name the reading its own records support. The server admits the sixteen requests one
per step as their prefills complete, so W16's first and last decode steps run with fewer than sixteen live rows: P223's
census of the same workload counts two decode calls per pass at each live-row count from 1 to 15 beside the full
steps. Those ramp steps run buckets 1 to 8, and in buckets 2 to 8 B1 routes K33's singleton groups where B0 runs K25.
So W16 is not inert under this switch. Its sixteen-row steps are, but its tokens carry whatever its ramp steps did.
That holds for any lever that acts on small buckets, which is why this lane does not rule on W16's tokens at all.

**What this lane changes.** One rung moves. Token identity between B1 and B0 is ruled on **W1 only**, the one workload
whose every decode step runs bucket 1, where the switch takes no call. W16 keeps P231's 0.99 speed line as **the W16
regression bar** (a default must not slow sixteen requests), and its B1-against-B0 tokens are reported, never ruled.
What decides the streams the switch does change is the quality rung P231 never reached, unchanged, behind a premise
that ties all of buckets 2, 4 and 8 to the per-row arithmetic it reads.

## Subject

As P231's. The default server: `PagedServeConfig.from_env()` + `build_engine`, graphs on, every serving lever unset
(the runner's `unset` list), at P136's Mixtral size, `E4B_PAGED_MAX_SEQS=16 E4B_PAGED_MAX_TOKENS_PER_SEQ=768`. Its
configuration is recorded; the buckets must be 1, 2, 4, 8 and 16, all captured as graphs.
- **Model:** `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61` (FAM Amendment 5's
  pin), NF4 arena baked on the box by P39's `k8_bake.py`.
- **Stack:** e4b at P234's launch commit; grouped-nf4-gemm **v0.50.0** (`41ef086e`), whose K33 auto table lists
  Mixtral's two expert shapes with P141's plans; transformers 5.17.0; bitsandbytes 0.50.2; torch 2.8.0+cu128 (the
  image's).
- **grouped-nf4-gemm's own knobs** (`GNF4_GEMV_BW`, its plan, decode and dot-pad knobs) stay unset in every arm.

## Arms, workloads and timing

Four arms in a palindrome, **B0a B1a B1b B0b**, each a fresh process (`bench/p234/p234_box.py --mode speed`):

| setting | environment |
|---|---|
| B0 | `E4B_NF4_SMALLM_BW` unset: the shipped default (the device tile table and K25 at 2 to 8 tokens) |
| B1 | `E4B_NF4_SMALLM_BW=1`: 2 to 8 tokens take singleton groups, so K33 |

**Workloads** are P109's prompts: W1 is row 0 alone; W2, W4, W8 and W16 are the first 2, 4, 8 and 16 distinct
512-token wikitext prompts at once. Each runs 32 and 160 new tokens, one warm pass and 3 timed passes.
- **Ruled:** W2 and W4 (buckets 2 and 4, where the switch moves the route).
- **The inert check: W1 only.** Every W1 decode step is one token, bucket 1. The switch's guard takes 2 to 8 tokens
  only (`1 < T <= 8`), so a 1-token call runs the singleton route and K33 in both arms. This is pinned on CPU call for
  call (the bucket-1 test, below) and read on the card from every arm's per-bucket engagement (the rule's VOID rung).
- **W16: the regression bar only.** Its sixteen-row steps run bucket 16 (K25 in both arms, read positively per bucket),
  and its ramp steps run buckets 1 to 8. Its rate ratio g16 is held to 0.99; its tokens are reported.
- **Reported only:** W8.

**Decode-only timing** (P112 Amendment 2, P113), as P231's: each pass's wall less its largest ttft, p37's slope over
those times, the whole-pass slope reported beside it.

**Recorded per arm**, as P231's: every pass's wall, ttfts and token digest, and the last pass's tokens per row; the
resolved server configuration and `graph_status`; per decode bucket, tagged at capture, e4b's route tally
(`hot_residency.ROUTE_SEEN`) and grouped-nf4-gemm's dispatch counts (K33's `bw_*`, every counter, zeros kept); memory.

**The fit check** (`--mode fit`, B0, before the arms), as P231's: the default server builds at 16 x 768 and decodes
W16. Not fitting halts the lane (rc 31).

**The quality phase** (`--mode quality`, after the arms, not after a failed one), as P231's: P115's teacher-forced
instrument (`p115_quality.measure_phase`), 24 windows per text (wikitext, c4val1), 128 continuation tokens, the server
built eager (`E4B_PAGED_GRAPHS=0`) at the same size, **at group 4 only**. OFF (B0) runs P115's reference R, its floor
arms (`rep`, `half`, `chunk`) and the mutant (`mutant_scale`); ON (B1) runs once. Each is a fresh process.

## The premise (on the card, before anything is fetched; rc 25)

All three must hold, or the lane HALTS until it is amended:
1. **The graph buckets:** e4b's `tests/test_decode_graph_buckets.py`, **7 passed**, none skipped (read from the junit
   report). P231's.
2. **The kernel premise (new):** e4b's `tests/test_nf4_smallm_bw_gpu.py`, **20 passed**, none skipped (read from its
   junit report). Its property, on a small all-hot NF4 store with device grouping on and K33 forced on:
   - under the switch, every token's output row at T = 2 to 8 is **BITWISE** that token's own T == 1 row, through K33
     only (one `bw_prmt32` launch per projection, no dot-pad, scalar GEMV, K25 or M-tile call), and counted;
   - the T = 4 step captures in a CUDA graph and replays bitwise as eager;
   - T = 9 and the switch off keep the device-grouped route; the switch reads the module's reference forward within
     the served tolerance;
   - each case with the lean glue off and at its default.

   **It last passed** on an RTX 5090 (170 SMs, torch 2.8.0+cu128) at e4b `b7146363` with grouped-nf4-gemm v0.50.0, in
   P227's premise (`p227-5090-1`: 20 of that premise's 25 passes, `logs/premise.log` in its receipt). The file has not
   changed since: this lane stages it byte-identical to P227's copy (sha256 `023f68ee...`, both pins checked by
   `tests/test_p234.py`). The box re-runs it at P234's launch commit, before the timed phase, so the property is read
   on the reading's own card and code.
3. **The rows premise** (`p234_box.py --mode rows`), P231's unchanged: with `E4B_NF4_SMALLM_BW=1` resolved for real
   on the box's device, the expert call exactly as the collapse makes it returns **bitwise** the same output for every
   (token, slot) row of four tokens whether it is called at T = 1, 2 or 4, on random NF4 stores at Mixtral's two shapes
   over six routings, each T launching K33 with the `nf4_singleton` route and no `nf4_k25`.

**Why they bound buckets 2 to 8.** Premise 2 pins the per-row property at every T from 2 to 8; premise 3 pins it at
Mixtral's own shapes and auto-engaged K33 for T = 2 and 4. Together they say ON's arithmetic at buckets 2, 4 and 8 is
one per-row function, the one every one-request step already serves by default. The group-4 quality read reads that
function; buckets 2 and 8 run the same per-row arithmetic. The consequence stays P231's anyway: a licence moves
buckets 2 and 4 only, and 5 to 8 tokens stay opt-in.

### The bucket-1 test (CPU, in CI; the maintainer's condition)

`tests/test_nf4_smallm_bw_bucket1.py` runs the served one-token decode call itself (`_HotResidency._forward_collapsed`
of an all-hot NF4 module at Mixtral's routing, 8 experts top-2, device grouping on, the switch resolved as the engine
build resolves it) with the knob unset and at `0`, against `1`, with the lean glue off and at its default. One recorder
logs, in order, every decision of the switch (`_nf4_smallm_bw_takes`), every grouping (`_collapsed_grouping`), every
kernel call with its arguments (each tensor by dtype, shape and a hash of its bytes), and after each step the route
tally, the switch's tally and the output's bytes. Over six routings the logs must be **equal, entry for entry**. At
T = 2 and 4 the same recorder must see them differ (K25 against the singleton route), so an equal bucket-1 log is not
a recorder that sees nothing.

**Its mutants, each run once** (e4b `8be4af31`, CPU, CI's grouped-nf4-gemm pin; the unmutated source passes 6 of 6):

| mutant of `hot_residency.py` | the test |
|---|---|
| M1: widen the switch's guard to `0 < T <= 8` | **FAILS**: the switch's own decision at T = 1 flips. Its kernel calls, tallies and output bytes stay equal, so the dispatch is the same function either way and the test catches M1 at the decision |
| M2: force the switch's branch at T == 1 (`... or (T == 1 and the knob)` in `_collapsed_grouping`) | **passes: an equivalent mutant.** The branch returns singleton groups without device grouping, exactly what T == 1 takes by default, so nothing at bucket 1 can change; no test can fail on it, and that is the inertness this lane relies on |
| M3: the knob sends T == 1 to the device tile table | **FAILS**: the grouping and every kernel call differ |
| M4: the knob drops T == 1's token-row read (`gather_div`) | **FAILS**: the kernel calls differ |

## The rule (`bench/p234/p234_reduce.py`, self-tested on 91 cases)

The verdict is the first of these that applies.

1. **VOID.** P231's, unchanged. Any of:
   - the fit check failed or is missing;
   - an arm or a quality phase is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, another model revision, or another server size;
   - an arm that ran the wrong switch (B0 must run it unset, B1 at `1`), or quality phases that did; a record without
     its switch entry;
   - different prompts, lengths, reps or buckets across arms; different windows, lengths, layers or buckets across the
     quality phases;
   - a void decode-only slope on any workload; a workload whose record lacks either length's tokens (each exactly its
     batch of rows) or either length's rep digests (each exactly `reps` of them);
   - **wrong engagement, read per bucket at capture and failing closed** (a bucket with no route tally, or whose gnf4
     dispatch record does not name every v0.50.0 counter, is a fault). Every arm reports `graph` for buckets 1, 2, 4, 8
     and 16; **bucket 1 takes `nf4_singleton` and K33 with no K25 in every arm** (the inert check, read positively on
     the card); buckets 2, 4 and 8 take `nf4_singleton` and K33 with no K25 in B1, and `nf4_k25` with no singleton and
     no K33 in B0; bucket 16 takes `nf4_k25` with no K33 in every arm;
   - the quality phases' engagement, each from a complete dispatch record and a route tally: OFF launches no K33 and
     records `nf4_k25` with no `nf4_singleton`; ON launches K33 and records `nf4_singleton` with no `nf4_k25`; every
     pass's eager step and padding statistics match P115's bucket arithmetic at group 4; every pass records its replay
     count, and no graph replays in a `capture=False` pass; every arm has its registered window count;
   - P115's mutant passes the quality bar on either text.
2. **FUNCTION_FAIL.** Any of: B0b's tokens differ from B0a's, or B1b's from B1a's, on any workload at either length;
   **B1's tokens differ from B0's on W1** (either pair, either length), where the switch takes no call; an arm's timed
   reps do not all digest the same. Read before the timing rungs. B1 against B0 on W2, W4, W8 and W16 is **reported**
   by row (`token_diffs`), never a rung: the switch acts in those workloads' steps, W16's included.
3. **NOISY.** A self-pair (B0b/B0a or B1b/B1a, decode tok/s) falls outside **[0.985, 1.015]** on W1, W2 or W4, or
   outside **[0.96, 1.04]** on W16. W8's self-pairs are reported only.
4. **QUALITY_FAIL.** P231's, unchanged. On either text, ON's bias or spread fails P110's bars against P115's floor
   (bias <= B_floor + 0.01; spread <= 2 x max(S_floor, 0.005); the floor is `half` and `chunk`, and `rep` too when R
   was not identical), or wikitext's K8 perplexity moves more than 0.05.
   - **The streams it covers:** P115's teacher-forced scoring at group 4, 24 windows per text, 128 continuation
     tokens. Each scored pass decodes four windows at once, so every scored token is produced in a bucket-4 step: K33's
     singleton route under ON, K25 under OFF (both read positively above).
   - **Why that stands for buckets 2 and 8:** the kernel and rows premises (above) make ON's per-row output at any T
     from 2 to 8 bitwise its T == 1 row.
5. **SLOWER.** g2 or g4 (g_w = min(B1a/B0a, B1b/B0b), decode tok/s) is below **1.03**; g1 (the inert check) is below
   **0.99**; or g16 is below **0.99** (the W16 regression bar).
6. **LICENSED** otherwise.

**Reported:** every pair ratio and g on every workload (W8's g with no bar), the self-pairs, ms per step, the
whole-pass slopes, each bucket's route tally and dispatch counts, `token_diffs`, and the quality table.

**Why these bars (from variance only), as P231's.** The self-pair bands are P141's, on this model, card and timing:
its self-pairs read 0.9955 and 0.9985 at W1, and 0.9978 and 1.0009 at sixteen requests, inside [0.985, 1.015] and
[0.96, 1.04]. The gain bar, 1.03, is twice the narrow band's half-width, and g asks the smaller of the two pairs to
clear it. The 0.99 lines are P141's no-regression line. The quality bars are P110's, as P141 applied them. None moved
after P231's data.

## Predictions

Written before any P234 data, and after P231's (seen and stated below). They are expectations, not the sizing basis.

| # | prediction |
|---|---|
| Q1 | the premise holds on the card (all three parts), and every arm's per-bucket engagement holds |
| Q2 | the self-pairs read inside their bands; B1's tokens equal B0's on W1; B1's tokens differ from B0's on W16 in some rows (reported) |
| Q3 | g2 lies between 1.9 and 2.3 and g4 between 1.55 and 1.9 (P231's ungraded reads: 2.096 and 1.729) |
| Q4 | g1 and g16 lie between 0.99 and 1.01; W8's g is above 1 |
| Q5 | ON passes P110's bars on both texts (P231's ungraded ON draws read bias -0.00144 on wikitext and 0.00014 on c4val1) |
| Q6 | the verdict is LICENSED (about 85 %), VOID or NOISY (about 8 %), QUALITY_FAIL (about 5 %), SLOWER or FUNCTION_FAIL (about 2 %) |

## Consequence, registered now

- **LICENSED.** `E4B_NF4_SMALLM_BW` becomes the default for Mixtral's NF4 store at 2 to 4 tokens (buckets 2 and 4),
  in its own e4b PR citing this read: one family, this card class, the all-resident collapse. Calls of 5 to 8 tokens
  (bucket 8) are not ruled here, so **5 to 8 tokens stay opt-in** until B2 or a later lane reads them. A register row
  `e4b.serve.p234.nf4-smallm-bw.mixtral-nf4.5090.<date>` carries g2 and g4, and the families inventory
  (`bench/fam/INVENTORY-fam.md`) notes the read. The default's PR names the new served tokens at 2 and 4 requests as a
  changed stream within P110's bars, and the default-path diff note that moves later readings forward.
- **SLOWER.** No default moves; the switch stays opt-in.
- **QUALITY_FAIL.** No default moves. The record names the text and the bar it failed.
- **FUNCTION_FAIL.** W1's tokens moved under a switch that takes no 1-token call, or a box is not deterministic:
  reported with the rows; the switch stays opt-in until it is explained.
- **VOID or NOISY.** No consequence; the one rerun (`p234-5090-2`), or an amendment.

**Scope.** Mixtral-8x7B's NF4 store, one RTX 5090, 1, 2, 4, 8 and 16 requests. Nothing for Qwen3, other families,
other cards, or the multi-row kernel (B2).

## Run ids, outcomes and the stop

**Run ids.** `p234-5090-1` is the reading and `p234-5090-2` its one rerun. Every box needs its own id and receipt, and
a box that does not start uses an id while counting toward no started slot. The retry ids
**`p234-5090-r1` to `p234-5090-r8`** follow any box that did not start, at the same terms. The ids are labels, not budget: the caps below
alone govern, and started boxes are counted by outcome, never by id.

**Shakedowns** (P231's Amendments 2 and 3, carried): the ids **`p234-shake-1` to `p234-shake-3`** run this runner
through the refusals, the download probe, the installs, the tripwire, the self-tests and all three premises, then stop
with rc 0 (`SHAKE OK`) before anything is fetched. Their fit is the installs only and their disk floor 40 GB. The
classifier reads them as `SHAKE_OK` (rc 0), `SHAKE_HOST_REFUSED` (a host code 13, 15, 16, 18 or 19, or the fit's 41:
the next shake id) or `SHAKE_FAIL` (any other code, which halts the lane until the kit is fixed). None is a box-started
run, a strike or a reading's host refusal. Before `p234-5090-1`, one shakedown at P234's merge runs on a 5090 as an
exploration, outside the lane's counts and budget, at a worst case under USD 1.

**The start.** The runner writes `STARTED` once every pre-install check has passed (the refusals, the download probe
and the pre-install fit), immediately before the first install command. A box is box-started from that marker.

**The download probe**, P231's: one sustained GET of the checkpoint's first shard from the endpoint the fetch uses,
until 256 MiB and 20 s, or 60 s. Below **40 MB/s**, a non-2xx status or an error refuses the host with rc 19:
**HOST_REFUSED, never a strike**.
- **The floor is tied to the fit.** The fetch's need is 3,000 s. At 40 MB/s, 93 GB takes 2,325 s, which leaves 675 s
  of the need to spare.
- **The stop:** after **3 consecutive rc 19** boxes the lane stops renting and reports the market's state instead; it
  resumes only on the maintainer's word or a later market read. The lane's launcher counts: before every create it
  reads the lane log's last three classified boxes and refuses to create when all three are rc 19.
- **The receipt:** `download_probe.json` is written on EVERY box, refused or not (status, bytes, seconds, MB/s, shard,
  any error by name). The probe's bytes count in the box's download allowance.
- **The token** goes to the resolve endpoint only (an unredirected header), never to the CDN and never on a command
  line.

**The classification is mechanical** (`p234_reduce.py --classify --work W`, run on every fetched box; it fails
closed, exit 2, unless W holds exactly one exit-code file and `RUN_ID`, and a refused classify halts the lane until the
box's files are read by hand):

| runner exit | before `STARTED` | after it |
|---|---|---|
| 0 | (cannot occur) | an ANSWER: LICENSED, SLOWER, QUALITY_FAIL or FUNCTION_FAIL |
| 13 free disk, 15 card / compute capability / SM count / VRAM, 16 host RAM, 18 CUDA unusable, 19 download below the floor | **HOST_REFUSED**: no start; counts toward the refusal cap | (cannot occur) |
| 41 the guard does not fit the deadline (a slow boot) | **FIT_REFUSED**: no start, a retry id; two CONSECUTIVE are one strike | (cannot occur) |
| 9 staging, install or pins, 10 torch does not import, 28 the image's torch, 78 run id / env / frozen knobs, 130 a signal, or any other code | **PRE_INSTALL_HALT**: no start, not capped; the lane halts until it is fixed | a started box WITHOUT an answer: a strike |
| 25 a premise failed on the card; 31 the default server does not fit at 16 x 768 | (cannot occur) | **HALT**: the lane stops until it is amended |
| 27 (VOID or NOISY) and any other code | (cannot occur) | a started box WITHOUT an answer: a strike |

**NOT_RUN** (the launcher's pre-flight tore down a created instance before the runner ran) is decided from the
launcher's receipt, never classified; it counts toward no cap and uses one id. A pre-create refusal (no instance) also
uses one id under the current launcher (bench/common rule 3), and counts toward nothing.

**The stop (nonadaptive):**
- The first ANSWER ends the lane.
- **At most 2 box-started runs** (the reading and one rerun).
- **Two strikes end the lane** with no reading. A strike is a started box without an answer, or two consecutive
  FIT_REFUSED boxes.
- **The download stop:** 3 consecutive rc 19 boxes stop the renting, with a market report.
- **HOST_REFUSED cap:** 4 refusals; then the lane posts the market's state and pauses, and resumes on a later market
  read with no amendment. A second trip (8 in all) ends the lane, UNRUNNABLE.
- No top-up, no change of arms, bars, workloads or lengths after data.

## Budget

As P231's registration.
- **One RTX 5090** (Vast verified/secure, cc 12.0, at least 160 SMs), at most USD 1.05/h including a 250 GB disk (the
  box's floor is 220 GB), host RAM at least 60 GiB, download at most USD 0.011/GB. No power floor: the read is a ratio
  inside one box, and the power limit is recorded. The pin is the cheapest passing offer not pinned by another lane
  (the market rule), with the download screen's registry readings as the tie-breaker.
- **The needs** (the runner's named constants): install and the three premises 600 s (the kernel premise ran in
  under 14 s on P227's card); the fetch 3,000 s, the bake 1,500 s, the fit 600 s, quality OFF 2,400 s and ON 900 s;
  each arm 900 s; and a 600 s reserve. **The fit is 13,200 s**, and with the cross-lane 25 % headroom the pre-install
  check needs **16,500 s** left.
- **Guard 4.8 h (17,280 s).** The boot slack, the guard less the fit's need, is **780 s**: at least the launcher's ssh
  wait (420 s) plus a 300 s margin for staging, the refusals and the download probe, as the test checks.
- **The download allowance:** 93 GB for the fetch plus the probe's bytes, at most one 4.9 GB shard: **97.9 GB** a box
  (the launch's `--download-gb`).
- **Worst per box:** 4.8 h x USD 1.05 + 97.9 GB x USD 0.011 = **USD 6.12**, inside the USD 15 per-box rule.
- **Worst per box that does not start:** 780 s x USD 1.05/h + 4.9 GB x USD 0.011 = **USD 0.29**.
- **Lane worst:** the ids bound the boxes at 10 (`-1`, `-2`, `r1` to `r8`), at most 2 of them started: 2 x USD 6.12 +
  8 x USD 0.29 = **USD 14.56**. The shakedown is an exploration outside it.
- **Expected:** about 2.5 hours and about USD 1.1 for the reading (P231's box cost USD 1.086).

## Launch

- **One exact launch commit:** the merge of this registration, through pod-launch with `--validate` passing at that
  commit, the host pinned, and rent-watch armed as the mini-side alarm. Every pin (staged files, the gnf4 commit, the
  model revision) is checked against that commit: `tests/test_p234.py` checks the staged pins against their sources,
  and the runner's tripwire checks the installed commits on the box.
- **The tripwire**, P231's with its Amendment 2: the installed commits and versions; the switch's routing block
  (unresolved it refuses by the guard's own message; the REAL `resolve_nf4_smallm_bw` on stand-in modules at Mixtral's
  expert shapes on the box's device; then 2, 4 and 8 tokens singleton, 16 device grouping, undone in a `finally`); K33
  engaged at Mixtral's two shapes with the `prmt32` decode; the per-bucket tag's target and `ROUTE_SEEN` present.
- **STOP-4:** the driver refuses a dirty tree, an `E4B_SHA` override, an unregistered run id, or a staged file that
  differs from `staged-p234.sha256`.

## What changes from P231, and what is carried

**Changed (this registration):**
- FUNCTION_FAIL reads B1 against B0 on W1 only; W16's B1-against-B0 tokens are reported (`token_diffs`).
- W16 keeps the 0.99 line as the W16 regression bar; W1's 0.99 line stays as the inert check's.
- The premise gains the kernel premise (`tests/test_nf4_smallm_bw_gpu.py`, staged and re-run on the card); the
  runner's premise gate reads all three parts.
- The bucket-1 CPU test (`tests/test_nf4_smallm_bw_bucket1.py`) joins CI.
- The run ids (`p234-5090-1`, `-2`, `r1` to `r8`; `p234-shake-1` to `-3`); the predictions.

**Carried from P231's pre-data amendments, unchanged:**
- **The fetch imports no links** (P231's Amendment 1): the driver's fetch passes `rsync --no-links`, so a symbolic
  link a rented box leaves is never copied to the controller (bench/common rule 8).
- **The tripwire resolves the switch** (P231's Amendment 2): the routing block makes the real resolve, and CI runs the
  block from the runner itself on CPU, with its mutants (the guard deleted, K33 not engaging, other shapes).
- **The rows premise resolves the switch itself, and every no-checkpoint phase runs in CI** (P231's Amendment 3): the
  tripwire's lines, the three self-tests, the rows premise through `rows_main` at small shapes, and the served premise's
  pytest line and junit read. This registration adds the kernel premise's pytest line: on CPU it collects its 20 cases,
  all skipped, and the box's junit read refuses that report (a skipped premise is never a passed one).

Everything else (the arms, workloads, prompts, lengths, timing, fit check, quality phase, budget, download probe and
classification) is P231's.

## What was seen before this page (stated, not hidden)

- **P231's reading** (`p231-5090-3`, `bench/p231/RESULTS-p231.md`) in full: its FUNCTION_FAIL rows; its ungraded rate
  ratios (g2 2.096, g4 1.729, W1 0.998, W16 1.001, W8 1.207) and self-pairs (0.99746 to 1.00468); its ungraded quality
  draws (ON's bias -0.00144 and 0.00014, argmax agreement 0.975 and 0.973); its records' ttfts and P223's census, the
  basis of the ramp reading. Predictions Q3 and Q5 lean on those numbers. The bars do not: every bar is P231's,
  registered before P231's data.
- **P227's receipt** (`p227-5090-1`): the kernel premise's file passing on a card, and its versions.
- **Locally:** the box's self-test and the reducer's (91 cases), `tests/test_p234.py`, and
  `tests/test_nf4_smallm_bw_bucket1.py` with its four mutants (the table above). Triton's kernels do not run on the
  development hosts, so the kernel premise's 20 cases are counted from the source and from P227's log.

## What this lane cannot say

- Nothing about Qwen3-30B-A3B or any other family, other cards, or the int4 and MXFP4 stores (where the switch is
  refused).
- Nothing about 5 to 8 requests beyond the reported W8 ratio.
- Nothing about B2, the multi-row kernel.
- Nothing about token identity at sixteen requests: that is reported, and no ruling turns on it. The cause of W16's
  divergence stays P231's reading (the admission ramp), consistent with the records and not traced step by step.
- Nothing about quality at group 2 or 8 beyond the premises that tie them to group 4.

## Receipts

Fetched to the run directory's `p234/` and committed to `bench/p234/receipts/<run>/`:
- `download_probe.json`, `fit.json`, `rows.json`, `arm_{B0a,B1a,B1b,B0b}.json`, `quality_{off,on}.json`,
  `verdict.json`, `summary.txt`, `versions.txt`, `prompts.json`, `work/mixtral/bake.json`;
- the logs (`premise.log`, `premise_kernel.log` and their junit reports among them), with `SHA256SUMS` over every file,
  and `STORE-ONLY.sha256` for the files kept only in the private receipt store (`forensics.txt` and `outer.log`, which
  carry the card's identity).

`RESULTS-p234.md` is written from those files.
