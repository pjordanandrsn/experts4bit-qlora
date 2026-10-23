# P231: does `E4B_NF4_SMALLM_BW=1` decode Mixtral-8x7B's DEFAULT NF4 server faster at 2 and 4 requests, at no measurable quality cost? On one RTX 5090 (serving roadmap lever B, step B3; registered before any run)

Lane number claimed by `p231/registration`. Issue: #1313. Follows P227 (the per-row route against K25 at 2 to 8
tokens, a microbench: `bench/p227/PREREG-p227.md`), P141 (K33 at Mixtral's shapes at one request, LICENSED:
`bench/p141/RESULTS-p141.md`) and the B2 design note (`bench/lever-b2/DESIGN.md`, e4b#2050).

## Why this lane

**What P227 read.** P227 timed e4b's MoE expert call on random NF4 stores at 2, 4 and 8 decode tokens, through two
routes: the one that serves those steps today (the device tile table, then K25), and the per-row route that
`E4B_NF4_SMALLM_BW=1` (e4b#2009) opens (one group per (token, slot) row, so grouped-nf4-gemm's K33 bandwidth GEMV, one
row per program). On Mixtral's shapes, over 60 routed steps, the per-row route took **0.356x** K25's time at T = 2 and
**0.461x** at T = 4 (upper bounds 0.360 and 0.465), against P227's bar of 0.7x.

**What P227 could not say.** It timed the expert call alone, eager, in a microbench. It said nothing about the served
decode step (attention, the router, the graphs, the other layers), and nothing about quality: the per-row route is a
different function from K25 within the tolerance contract, so its tokens differ from today's at 2 to 8 tokens.

**This lane reads both, on the served default.** The switch is read on the default `serve_paged` server, graphs on,
at P136's Mixtral size, with P141's ABBA arms and P115's quality instrument. It is the step that decides whether the
switch can become Mixtral's default at 2 and 4 requests.

**Why W8 is reported only.** At T = 8 the per-row route re-reads shared experts most (2.213x the weight floor on
P227's routing table, against 1.132x at T = 2 and 1.455x at T = 4), and B2, a multi-row K33, is designed for exactly
that case. So this lane rules on buckets 2 and 4 alone. Bucket 8 is timed and reported, with no bar, and no
consequence follows from it here.

## Subject

The default server: `PagedServeConfig.from_env()` + `build_engine`, graphs on, every serving lever unset (the runner's
`unset` list), at P136's Mixtral size, `E4B_PAGED_MAX_SEQS=16 E4B_PAGED_MAX_TOKENS_PER_SEQ=768` (P141's). Its
configuration (max_seqs, buckets, placement) is recorded; the buckets must be 1, 2, 4, 8 and 16, all captured as graphs.
- **Model:** `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61` (FAM Amendment 5's
  pin, P141's), NF4 arena baked on the box by P39's `k8_bake.py`.
- **Stack:** e4b at the launch commit; grouped-nf4-gemm **v0.50.0** (`41ef086e`), whose K33 auto table lists Mixtral's
  two expert shapes with P141's plans (gate_up (28672, 4096) at (16, 1024, 4, 1), down (4096, 14336) at
  (16, 512, 4, 2)); transformers 5.17.0; bitsandbytes 0.50.2; torch 2.8.0+cu128 (the image's).
- **grouped-nf4-gemm's own knobs** (`GNF4_GEMV_BW`, its plan, decode and dot-pad knobs) stay unset in both arms: K33
  is what the shipped one-request decode already runs at these shapes.

## Arms, workloads and timing

Four arms in a palindrome, **B0a B1a B1b B0b**, each a fresh process (`bench/p231/p231_box.py --mode speed`):

| setting | environment |
|---|---|
| B0 | `E4B_NF4_SMALLM_BW` unset: the shipped default (the device tile table and K25 at 2 to 8 tokens) |
| B1 | `E4B_NF4_SMALLM_BW=1`: 2 to 8 tokens take singleton groups, so K33 |

**Workloads** are P109's prompts: W1 is row 0 alone; W2, W4, W8 and W16 are the first 2, 4, 8 and 16 distinct
512-token wikitext prompts at once. Each runs 32 and 160 new tokens, one warm pass and 3 timed passes.
- **Ruled:** W2 and W4 (buckets 2 and 4, where the switch moves the route).
- **Controls:** W1 (one token is singleton already) and W16 (above the switch's 8 tokens). The switch is inert on both.
- **Reported only:** W8.

**Decode-only timing** (P112 Amendment 2, P113). Every pass records each row's `Request.ttft`, and the rows arrive
together. The decode-only time is the wall minus the pass's largest ttft, and p37's slope (P109's `slope`) is taken
over those times. The whole-pass slope is reported beside it.

**Recorded per arm:** every pass's wall, ttfts and token digest, and the last pass's tokens per row; the resolved
server configuration and `graph_status`; **per decode bucket**, tagged at capture (the box wraps
`PagedModelRunner._capture_bucket` for the build only), e4b's route tally (`hot_residency.ROUTE_SEEN`: `nf4_singleton`,
`nf4_k25`) and grouped-nf4-gemm's dispatch counts (K33's `bw_*`); memory.

**The fit check** (`--mode fit`, B0, before the arms): the default server builds at 16 x 768 and decodes W16, the
widest workload. Not fitting halts the lane (rc 31).

**The quality phase** (`--mode quality`, after the arms, not after a failed one): P115's teacher-forced instrument
(`p115_quality.measure_phase`), 24 windows per text (wikitext, c4val1), 128 continuation tokens, the server built eager
(`E4B_PAGED_GRAPHS=0`, the instrument's requirement) at the same size, **at group 4 only** (four windows per pass, so
T == 4, the route B1 changes). OFF (B0) runs P115's reference R and its floor arms (`rep`, `half`, `chunk`) and the
mutant (`mutant_scale`); ON (B1) runs once. Each is a fresh process.

## The premise (on the card, before anything is fetched; rc 25)

Both must hold, or the lane HALTS until it is amended:
- **The graph buckets:** e4b's `tests/test_decode_graph_buckets.py`, **7 passed**, none skipped (read from the junit
  report).
- **The rows premise** (`p231_box.py --mode rows`, the maintainer's condition for reading quality at group 4 only): with
  `E4B_NF4_SMALLM_BW=1`, the expert call exactly as the collapse makes it (`hot_residency._fused_over_stack` over the
  token rows, singleton groups) returns **bitwise** the same output for every (token, slot) row of four tokens whether
  it is called one token at a time (T = 1), two at a time (T = 2) or all four at once (T = 4). It runs on random NF4
  stores at Mixtral's two shapes (8 experts, top-2) over six routings: all eight experts distinct, a shared expert,
  all four tokens on one pair, and three seeded draws. Every case must also launch K33.
- **Why it licenses group 4 alone.** If each row's output does not depend on how many rows share the call, then B1's
  arithmetic at 2 tokens is B1's at 4, row for row, and the group-4 read covers group 2. Without it, B1 could differ
  between buckets 2 and 4, and a group-2 read would be owed.

## The rule (`bench/p231/p231_reduce.py`, self-tested on 44 cases)

The verdict is the first of these that applies.

1. **VOID.** Any of:
   - the fit check failed or is missing;
   - an arm or a quality phase is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, another model revision, or another server size;
   - an arm that ran the wrong switch (B0 must run it unset, B1 at `1`), or quality phases that did;
   - different prompts, lengths, reps or buckets across arms; different windows, lengths, layers or buckets across the
     quality phases;
   - a void decode-only slope on any workload;
   - **wrong engagement, read per bucket at capture and failing closed** (a bucket with no route tally is a fault):
     every arm reports `graph` for buckets 1, 2, 4, 8 and 16; bucket 1 takes `nf4_singleton` and K33 with no K25 in
     every arm; buckets 2, 4 and 8 take `nf4_singleton` and K33 with no K25 in B1, and `nf4_k25` with no singleton and
     no K33 in B0; bucket 16 takes `nf4_k25` with no K33 in every arm. So the W1 and W16 controls are read positively,
     never argued from the code;
   - the quality phases' engagement: OFF launches no K33 in its measurement; ON launches K33 and records `nf4_singleton`;
     every pass's eager step and padding statistics match P115's bucket arithmetic at group 4 (half: two passes of 2);
     no graph replay in a `capture=False` pass; every arm has its registered window count;
   - P115's mutant passes the quality bar on either text (the instrument cannot see a known fault).
2. **FUNCTION_FAIL.** Any of: B0b's tokens differ from B0a's, or B1b's from B1a's, on any workload at either length;
   B1's tokens differ from B0's on a control (W1 or W16), where the switch is inert; an arm's timed reps do not all
   digest the same. Read BEFORE the timing rungs: token identity does not depend on timing. (B1's tokens may differ from
   B0's on W2, W4 and W8: a different function, which the quality rung reads.)
3. **NOISY.** A self-pair (B0b/B0a or B1b/B1a, decode tok/s) falls outside **[0.985, 1.015]** on W1, W2 or W4, or
   outside **[0.96, 1.04]** on W16. W8's self-pairs are reported only.
4. **QUALITY_FAIL.** On either text, ON's bias or spread fails P110's bars against P115's floor (bias <= B_floor +
   0.01; spread <= 2 x max(S_floor, 0.005); the floor is `half` and `chunk`, and `rep` too when R was not identical), or
   wikitext's K8 perplexity moves more than 0.05.
5. **SLOWER.** g2 or g4 (g_w = min(B1a/B0a, B1b/B0b), decode tok/s) is below **1.03**, or a control (g1 or g16) is
   below **0.99**.
6. **LICENSED** otherwise.

**Reported:** every pair ratio and g on every workload (W8's g with no bar), the self-pairs, ms per step, the
whole-pass slopes, each bucket's route tally and dispatch counts, and the quality table.

**Why these bars (from variance only).** The self-pair bands are P141's, on this model, card and timing: its self-pairs
read 0.9955 and 0.9985 at W1 and 0.9978 and 1.0009 at W16, inside [0.985, 1.015] and [0.96, 1.04]. W2 and W4 take the
narrow band: like W1, they decode a few rows, and their steps are short. The gain bar, 1.03, is twice the narrow band's
half-width: g asks the SMALLER of the two independent pairs to clear it, so a box whose self-pairs sit at the band's
edge cannot make a null read as a gain. The controls' line, 0.99, is P141's no-regression line. The quality bars are
P110's, as P141 applied them.

## Predictions

Written before any P231 data. They are expectations, not the sizing basis.

| # | prediction |
|---|---|
| Q1 | the premise holds on the card (both parts), and every arm's per-bucket engagement holds |
| Q2 | the self-pairs read inside their bands; B1's tokens equal B0's on W1 and W16 |
| Q3 | g2 lies between 1.6 and 2.6 and g4 between 1.4 and 2.2 (P227's expert-call times, scaled from its 8 layers to 32, against P141's 8.1 ms W1 step) |
| Q4 | g1 and g16 lie between 0.99 and 1.01; W8's g is above 1 |
| Q5 | ON passes P110's bars on both texts (the per-row route is K33 at every row, which P141 read within them at one request) |
| Q6 | the verdict is LICENSED (about 80 %), VOID or NOISY (about 15 %), QUALITY_FAIL (about 4 %), SLOWER or FUNCTION_FAIL (about 1 %) |

## Consequence, registered now

- **LICENSED.** `E4B_NF4_SMALLM_BW` becomes the default for Mixtral's NF4 store at 2 to 4 tokens (buckets 2 and 4),
  in its own e4b PR citing this read: one family, this card class, the all-resident collapse. Calls of 5 to 8 tokens
  (bucket 8) are not ruled here, so **5 to 8 tokens stay opt-in** until B2 or a later lane reads them. A register row `e4b.serve.p231.nf4-smallm-bw.mixtral-nf4.5090.<date>` carries g2 and g4, and the
  families inventory (`bench/fam/INVENTORY-fam.md`) notes the read.
- **SLOWER.** No default moves. The switch stays opt-in, and the record says the microbench's gain does not survive
  the served step on this card.
- **QUALITY_FAIL.** No default moves. The record names the text and the bar it failed; the switch stays opt-in.
- **FUNCTION_FAIL.** A control's tokens moved under a switch that should be inert there, or a box is not deterministic:
  reported with the rows; the switch stays opt-in until it is explained.
- **VOID or NOISY.** No consequence; the one rerun (`p231-5090-2`), or an amendment.

**Scope.** Mixtral-8x7B's NF4 store, one RTX 5090, 1, 2, 4, 8 and 16 requests. Nothing for Qwen3 (its own lane if it
is wanted), other families, other cards, or the multi-row kernel (B2).

## Run ids, outcomes and the stop

**Run ids.** `p231-5090-1` is the reading and `p231-5090-2` its one rerun. Every box needs its own id and receipt (the
ledger holds one row per run, so an id is never reused), and a box that does not start uses an id while counting toward
no started slot. The retry ids **`p231-5090-r1` to `p231-5090-r8`** follow any box that did not start, at the same
terms. The ids are labels, not budget: the caps below alone govern, and started boxes are counted by outcome, never by
id.

**The start.** The runner writes `STARTED` once every pre-install check has passed (the refusals, the download probe
and the pre-install fit), immediately before the first install command. A box is box-started from that marker.

**The download probe** (before the start, P225's form; added with the registration's GO): one sustained GET
 of the checkpoint's first shard from the endpoint the fetch uses, until 256 MiB and 20 s, or 60 s. Below
**40 MB/s**, a non-2xx status or an error refuses the host with rc 19: **HOST_REFUSED, never a strike**.
- **The floor is tied to the fit.** The fetch's need is 3,000 s (`NEED_FETCH`). At 40 MB/s, 93 GB takes 2,325 s, which
  leaves 675 s of the need to spare; the floor is the speed that keeps the fetch inside its need with that slack. (The
  probe reads one stream; the fetch runs eight workers, so the probe errs low.)
- **The stop, as HO3's:** after **3 consecutive rc 19** boxes the lane stops renting and reports the market's state on
  the bus instead of renting on; it resumes only on the maintainer's word or a later market read. (This sits inside the
  HOST_REFUSED cap below, which still governs.)
- **The receipt:** `download_probe.json` is written on EVERY box, refused or not: the HTTP status, the bytes read, the
  seconds, the measured MB/s, the shard and any error by name. It travels with the receipt, so the host registry
  learns the edge speed from a refusal too. The probe's bytes count in the box's download allowance (see Budget).
- **The token** goes to the resolve endpoint only (an unredirected header), never to the CDN and never on a command
  line.

**The classification is mechanical** (`p231_reduce.py --classify --work W`, run on every fetched box; it fails closed,
exit 2, unless W holds exactly one exit-code file and `RUN_ID`, and a refused classify halts the lane until the box's
files are read by hand):

| runner exit | before `STARTED` | after it |
|---|---|---|
| 0 | (cannot occur) | an ANSWER: LICENSED, SLOWER, QUALITY_FAIL or FUNCTION_FAIL |
| 13 free disk, 15 card / compute capability / SM count / VRAM, 16 host RAM, 18 CUDA unusable, 19 download below the floor | **HOST_REFUSED**: no start; counts toward the refusal cap | (cannot occur) |
| 41 the guard does not fit the deadline (a slow boot) | **FIT_REFUSED**: no start, a retry id; two CONSECUTIVE are one strike (bench/common's Guard fit) | (cannot occur) |
| 9 staging, install or pins, 10 torch does not import, 28 the image's torch, 78 run id / env / frozen knobs, 130 a signal, or any other code | **PRE_INSTALL_HALT**: no start, not capped; a kit, image or launch fault, so the lane halts until it is fixed | a started box WITHOUT an answer: a strike |
| 25 the premise failed on the card; 31 the default server does not fit at 16 x 768 | (cannot occur) | **HALT**: the lane stops until it is amended |
| 27 (VOID or NOISY) and any other code | (cannot occur) | a started box WITHOUT an answer: a strike |

**NOT_RUN** (the launcher's pre-flight tore down a created instance before the runner ran: no `RUN_ID`, exit file or
marker) is decided from the launcher's receipt, never classified by `--classify`; it counts toward no cap and uses one
id. A pre-create refusal (no instance) also uses one id under the current launcher (bench/common rule 3), and counts
toward nothing.

**The stop (nonadaptive):**
- The first ANSWER ends the lane.
- **At most 2 box-started runs** (the reading and one rerun).
- **Two strikes end the lane** with no reading. A strike is a started box without an answer, or two consecutive
  FIT_REFUSED boxes.
- **The download stop:** 3 consecutive rc 19 boxes stop the renting, with a market report (the probe, above).
- **HOST_REFUSED cap:** 4 refusals; then the lane posts the market's state and pauses, and resumes on a later market
  read with no amendment. A second trip (8 in all) ends the lane, UNRUNNABLE.
- No top-up, no change of arms, bars, workloads or lengths after data.

## Budget

- **One RTX 5090** (Vast verified/secure, cc 12.0, at least 160 SMs), at most USD 1.05/h including a 250 GB disk (the
  box's floor is 220 GB: the 93 GB checkpoint, its NF4 snapshot and arena, the reference log-probs), host RAM at least
  60 GiB, download at most USD 0.011/GB. No power floor: the read is a ratio inside one box, and the power limit is
  recorded. The pin is the cheapest passing offer not pinned by another lane (the market rule), with the download
  screen's registry readings as the tie-breaker.
- **The needs** (the runner's named constants): install and premise 600 s; the fetch 3,000 s, the bake 1,500 s, the
  fit 600 s and quality OFF 2,400 s and ON 900 s (P141's); each arm 900 s (five workloads, P141's two at 600 s); and a
  600 s reserve. **The fit is 13,200 s**, and with the cross-lane 25 % headroom the pre-install check needs
  **16,500 s** left.
- **Guard 4.8 h (17,280 s).** The boot slack, the guard less the fit's need, is **780 s**: at least the launcher's ssh
  wait (`ssh_ready_s` 420 s) plus a 300 s margin for staging, the refusals and the download probe, as the test checks.
- **The download allowance:** 93 GB for the fetch plus the probe's bytes, at most one 4.9 GB shard: **97.9 GB** a box
  (the launch's `--download-gb`).
- **Worst per box:** 4.8 h x USD 1.05 + 97.9 GB x USD 0.011 = **USD 6.12**, inside the USD 15 per-box rule.
- **Worst per box that does not start:** its boot term at the box rate plus the probe's download, at most one 4.9 GB
  shard: 780 s x USD 1.05/h + 4.9 GB x USD 0.011 = **USD 0.29**.
- **Lane worst:** the ids bound the boxes at 10 (`-1`, `-2`, `r1` to `r8`), at most 2 of them started: 2 x USD 6.12 +
  8 x USD 0.29 = **USD 14.56**.
- **Expected:** about 2.5 hours and about USD 3 for the reading (the fetch at the probe's floor or better, the bake,
  four arms and two quality phases).

## Launch

- **One exact launch commit:** the merge of this registration, through pod-launch with `--validate` passing at that
  commit, the host pinned, and rent-watch armed as the mini-side alarm. Every pin (staged files, the gnf4 commit, the
  model revision) is checked against that commit: `tests/test_p231.py` checks the staged pins against their sources at
  the commit, and the runner's tripwire checks the installed commits on the box.
- **The tripwire** (after the install, before the premise): the installed e4b and grouped-nf4-gemm commits, the
  versions; the switch's boundary as `hot_residency._collapsed_grouping` resolves it (unset: 1 token singleton, 2 to 8
  device grouping; set: 2, 4 and 8 tokens singleton, 16 device grouping); K33 engaged at Mixtral's two shapes with the
  `prmt32` decode; the per-bucket tag's target (`PagedModelRunner._capture_bucket`) and `ROUTE_SEEN` present.
- **STOP-4:** the driver refuses a dirty tree, an `E4B_SHA` override, an unregistered run id, or a staged file that
  differs from `staged-p231.sha256`.

## What was seen before this page (stated, not hidden)

- **P227's reading** on `p227-5090-1`: its Mixtral cells (B over K25 0.356 at T = 2, 0.461 at T = 4; B over the floor
  1.139, 1.326 and 1.997 at T = 2, 4 and 8) and its Qwen3 cells.
- **P141's records** (`bench/p141/RESULTS-p141.md`): its self-pairs, the basis of the bands above, its 8.1 ms W1 and
  53 ms W16 steps, and its quality table.
- **The B2 design note's routing arithmetic** (e4b#2050): the weight bytes over the floor at M = 1, the basis for
  reporting W8 only.
- **The probe, from a development host** (not a box): its code read Mixtral's pinned first shard at 88.7 MB/s over
  20 s, a public repository's smaller shard to its end, and recorded an unknown repository as a 401 with the error by
  name, writing `download_probe.json` each time.
- **Locally:** the box's self-test (24 cases) and the reducer's (44), and `tests/test_p231.py`, pass. Triton does not
  install on the development hosts, so the premise's test counts are read from the sources at the launch commit.

## What this lane cannot say

- Nothing about Qwen3-30B-A3B or any other family, other cards, or the int4 and MXFP4 stores (where the switch is
  refused).
- Nothing about 5 to 8 requests beyond the reported W8 ratio.
- Nothing about B2, the multi-row kernel.
- Nothing about quality at group 2 beyond the rows premise that ties it to group 4.

## Receipts

Fetched to the run directory's `p231/` and committed to `bench/p231/receipts/<run>/`:
- `download_probe.json`, `fit.json`, `rows.json`, `arm_{B0a,B1a,B1b,B0b}.json`, `quality_{off,on}.json`, `verdict.json`, `summary.txt`,
  `forensics.txt`, `versions.txt`, `prompts.json`, `work/mixtral/bake.json`;
- the logs, with `SHA256SUMS` over every file, and `STORE-ONLY.sha256` for the files kept only in the private receipt
  store.

`RESULTS-p231.md` is written from those files.
