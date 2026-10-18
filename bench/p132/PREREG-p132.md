# P132 (e4b#1313): where the 1 → n row verify increment goes, split by kernel class on one RTX 5090

## The question

SD2's read (`sd2-5090-3`, VERIFY_COST_REFUTES, `bench/sd2/RESULTS-sd2.md`) measured the target forward of a speculative
verify against a plain decode step on Qwen3-30B-A3B. A verify of n rows costs I(n) = verify_ms(n) − anchor =
**4.119 / 4.559 / 4.897 ms** more than one row at n = 2 / 3 / 4. A 1 → 2 step that size closes every multi-token B = 1
lever (speculative decode, decode lookahead) on this stack.

Two $0 reads since then frame this lane:
- **The census of distinct experts.** A census-weighted expert-byte model leaves a flat ~3.4 ms after it at every n.
  Distinct experts explain the slope, not the step.
- **The A2000 kernel inventory (SUPPORT).** It shows the expert path changes kernels between T = 1 and T ≥ 2: the
  bandwidth GEMV `_gemv_nf4_bw` at one row; the device-grouped tile table `_tile_table_r1` plus K25
  `_gemm_nf4_grouped_smallm` at 2–4 rows. The B = 1 folds run at every row count.

P132 asks the 5090 the question the inventory could only name: **how much of the increment is that expert-path switch**,
and is the part a better 2–4-row kernel could save large enough to make speculative decoding faster than plain decode?

## The registered prediction (the desk prediction v2, accepted as final before this registration)

`bench/p132/desk_prediction.py` (sha256 `71229f08…`) prints every number below from the receipts it reads:
- SD2's `v.json`;
- P123's B = 1 and B = 16 census terms;
- gnf4's K14 cuBLAS rows;
- the four Qwen3 routing traces.

`bench/p132/desk_out.json` is its output. `tests/test_p132.py` holds the reducer's constants equal to it.

| n | I(n) | distinct-expert bytes | route lo / c / hi | dense lo / c / hi | attn + norm | known lo / c / hi | HOLE lo / c / hi | savable S under H lo / c / hi |
|---|---|---|---|---|---|---|---|---|
| 2 | 4.119 | 0.742 | 0.171 / 0.217 / 0.263 | 0 / 0.393 / 0.702 | 0.034 | 0.947 / 1.386 / 1.741 | 2.379 / 2.733 / 3.172 | 2.641 / 2.950 / 3.343 |
| 3 | 4.559 | 1.342 | same | same | 0.069 | 1.582 / 2.021 / 2.376 | 2.183 / 2.538 / 2.977 | 2.446 / 2.755 / 3.148 |
| 4 | 4.897 | 1.863 | same | same | 0.103 | 2.138 / 2.576 / 2.931 | 1.967 / 2.321 / 2.760 | 2.229 / 2.538 / 2.931 |

**H:** the hole lies in the expert path, meaning K25 + the tile table + the T ≥ 2-only glue at about 13 distinct
experts per layer. No receipt times K25 at 2–4 rows on Qwen3 shapes, so the desk can only bound the hole, not
attribute it.

**H predicts BUILD.** Under H's central values, S(2) = 2.95 ms clears need(2) = 2.358 ms. That makes this read
decision-relevant, not confirmatory.

## The subject

SD2's read server, built by SD2's own code (`sd2_box._build_v`, staged at its bytes and imported, never edited):
- **Packages:** e4b **`88ae1cfd`** (SD2's read target) and grouped-nf4-gemm **v0.45.0** (`724ccc45`).
  `Qwen/Qwen3-30B-A3B` at `ad44e77`, NF4 arena baked on the box.
- **Server knobs:** `E4B_PAGED_SPEC=eagle3` at k = 3 with the pinned head (`6afc5aa2`, sha256 `d2d6e2e6…`),
  `E4B_PAGED_MAX_SEQS=16`, `E4B_INT4_TILE_PROGRAMS=1`, and `sd2_run.sh`'s unset list. So decode bucket 1 and verify
  buckets 2, 3, 4 are captured CUDA graphs.
- **Verification on the box:**
  - SD2's package-diff audit against its proven build `539a2d26` runs unchanged, here and on the box.
  - The target's GPU tests (SD2's list) run on the card before the fetch.

**The read transfers to main.** The A2000 inventory found the T = 1..4 kernel sets identical at `88ae1cfd` and at
`e9ab2ccc`, so a BUILD decision read here applies to main. That inheritance is recorded; it is not re-run.

## The instrument

**Timing (unprofiled), `p132_box.py --mode timing`.** SD2's V1 timed terms (`sd2_box.v1_terms`, unchanged) re-run in
this box: the A/A anchors and, per k = n − 1, verify_ms by CUDA events. I_b(n) = verify_ms(n) − the anchors' mean.

**The census, `p132_box.py --mode census`.** P123's Nsight Systems census, with one condition per process, 32 replays
skipped then exactly **64** bracketed (`cudaProfilerApi`), in graph mode then node mode.
- **n = 1:** the plain decode bucket-1 replay (`runner._run_decode_bucketed`, the anchor's call).
- **n = 2, 3, 4:** the verify bucket-n replay (`runner._spec_verify`, the call V1's verify_ms times) at a fixed base.
  It is fed the T = 1 oracle's own next tokens (a fully accepted verify) and rewound to the base between replays
  (`Gate.rewind`, V0's construction).
- **Reading:** SC1b's `sc1b_census.py arm` (unchanged) with `--unprofiled-ms` = the timing's anchor (n = 1) or
  verify_ms(n).
- **The class map is v1.2:** P123's v1.1 plus one name. `_rope_norm_qk` (gnf4 0.44, #528), which v1.1 leaves unmapped,
  joins `_rope_norm_heads` in `norm_elem`. Every kernel name in P123's own census classifies identically under v1.1 and
  v1.2 (`tests/test_p132.py`), so the change cannot move the P123 terms the desk used.

**The canary, first, in the box (rc 26).** Before the timing and the census, and before any long phase:
1. Every bucket 1..4 must be captured as a graph.
2. One plain replay and one verify replay per n must return.
3. A short census of 12 verify replays at n = 2, in both trace modes, must read cleanly through SC1b's classifier:
   status ok and no blocking label, with at least `CANARY_STEPS - 4` = 8 steps kept. The registration said 8 replays and
   a floor of `CANARY_STEPS - 2`. Amendment 1 changed both.

A failure costs minutes, not the hour. The A2000 cannot capture decode graphs (sm_86), so this in-box canary is the GPU
shakedown. The $0 parts were exercised off-box:
- the driver's dry run;
- the reducer's 14-case self-test;
- the class-map check;
- the staged pins.

## The rule (`bench/p132/p132_reduce.py`, self-tested)

**VOID (the first rung that applies):**
- a record is missing or not ok, or the GPU tests failed;
- the target, kernel package or model revision is off the registration;
- a bucket of 1..4 was not captured, or the canary failed;
- a census arm is void, or labelled CLASS_MAP_INCOMPLETE, CLASS_MAP_SEGMENT_BROKEN, NSYS_DIAGNOSTIC_ERRORS or
  CLOCK_MISMATCH;
- a census window did not bracket exactly 64 replays;
- **the re-run's anchor or a verify_ms(n) is more than 5 % off `sd2-5090-3`'s;**
- **the measured increment I_b(n) is more than 10 % off the target I(n) = 4.119 / 4.559 / 4.897 ms.** This tolerance is
  on the increment itself, because the readings below divide by it.

**NOISY:** the A/A anchors differ by more than 3 % (SD2's ANCHOR_TOL).

**READ** otherwise, with two readings:
- **H:** share(n) = [ΔExp(n) − bytes(n) − route_c] / [I_b(n) − known_c(n)]. ΔExp(n) is the census's
  moe_expert + moe_route increment from n = 1 to n; the terms with subscript c are the central values above.
  - **SUPPORT** if share ≥ 0.70 at every n = 2, 3, 4;
  - **REFUTE** if share < 0.50 at every n;
  - **INCONCLUSIVE** otherwise. A share between 0.50 and 0.70 is not support.
- **BUILD**, on the measured savable part whatever H reads: S(n) = ΔExp(n) − bytes(n), the switch's overhead and not its
  byte cost (even a perfect 2–4-row kernel must read the distinct experts' bytes).
  - **BUILD** a 2–4-row expert kernel iff S(n) ≥ need(n) for some n, with need = **2.358 / 1.683 / 1.558 ms**.
  - need(n) is SD2's S_V at its own FASTER bar (S_V ≥ 1.05 on two of three workloads) solved for full_ms with SD1's τ:
    the second-smallest over R / C-think / C-nothink of full_ms(k) − τ(k, w) × anchor / 1.05.
  - **NO_BUILD** otherwise: multi-token B = 1 levers close for this model on this stack.

**Reported, not ruled:**
- every class's increment;
- the unattributed remainder I_b(n) − ΣΔclass (out-of-graph work: the alias binding, the bucket load, the rewind);
- the census's own gates;
- kernels per replay;
- peak memory.

## Predictions (registered before any data)

- **The increment check passes.** I_b(n) lands within ±10 % of I(n), since the build is SD2's on the same card class.
- **H reads SUPPORT.** The central share is about 0.90 at n = 2.
- **The read is BUILD,** at n = 2 under H's central values (S(2) ≈ 2.95 ms ≥ 2.358).
- **The dense increment is at most 0.70 ms at every n.**

## Budget

- **Hardware:** one RTX 5090, pinned by machine id and offer id under the standing pin rule. The paid ACK names both, and
  the launch passes `--vast-pin-machine` / `--vast-pin-offer`.
- **Guard and cost:** a guard of **1.25 h** (`GUARD_S=4500`), about 62 GB of download as SD2. The ceiling is about $1.40
  at $0.79/h all-in plus the download, with about $0.9 expected.
- **Timing:** install about 10 min, GPU tests 2, fetch 7, bake 1, nsys 2, canary about 6, timing 3, census 8 captures of
  about 2 min each, reduce 1: about 50 min.
- **Gate:** the run waits for DQ11 to close (the stabilize-first rule) and the maintainer's ACK.

## What this lane cannot say

- Nothing about quality: no tokens are judged.
- Nothing about B > 1 batched decode of independent sequences. Its routing (ρ about 1.94 at 2 rows) differs from a
  verify's consecutive-token routing (ρ = 1.61).
- Nothing about cards other than the RTX 5090.
- A BUILD licenses building the kernel, not shipping speculation: SD2's E4B_PAGED_SPEC stays off until a new read.

## The consequence

- **BUILD:** a 2–4-row expert kernel (the bandwidth GEMV's design extended past one row) gets its own registration, with
  a bitwise gate against the current K25 path and a re-read of SD2's verify cost.
- **NO_BUILD:** the B = 1 multi-token lane closes for Qwen3-30B-A3B on this stack, with the measured split as its record.

## The lane number

P132, claimed at registration by the `prereg/p132` push after checking:
- origin's branches (prereg/p130 and p131 exist, no p132);
- bench/ and CHANGELOG on main;
- PR titles (P131 is #1627);
- the receipt store's run ids.

## Amendment 1 (2026-10-16, after `p132-5090-1`; the canary's floor; before any relaunch)

**What happened.** `p132-5090-1` (2026-10-11, one RTX 5090) stopped at the canary with rc 26: HARNESS_ERROR, $0.731,
teardown verified. Every other check in the canary passed:
- buckets 1..4 were captured as graphs;
- one plain and one verify replay per n returned;
- both short captures at n = 2 exited 0, with 8 replays bracketed (10.13 / 10.40 ms per profiled replay) and no period
  dropped.

SC1b read both trace modes as void: `5 steady steps kept < 6 (dropped {})` and `5 steady launches kept < 6 (dropped {})`.

**The cause: the floor was one higher than the reader can keep.**
- SC1b's `graph_mode` and `node_mode` make one row for each period between consecutive steady replays, so N bracketed
  replays give N − 1 rows.
- `_filter` then trims both edges, so at most N − 3 rows are kept.
- The registration set the canary's floor to `CANARY_STEPS - 2`. At 8 replays, that is 5 kept against a floor of 6.

The canary could not pass on any host. The error was the registration's: the class map, the box and the card had no
part in it.

The main census is unaffected. Its 64 bracketed replays keep 61, against SC1b's default floor of 56, and its calls set
no floor of their own.

**The amendment (the maintainer's ACK, 2026-10-16):**
1. `CANARY_STEPS` goes from 8 to **12**, and the floor from `CANARY_STEPS - 2` to **`CANARY_STEPS - 4`** = 8. That is
   N − 1 periods, less the two edge trims, less one long period tolerated. A second long period still voids the canary.
2. **Tests run on the run's real records first.** The records are in `tests/fixtures/p132_canary/`, byte-identical to
   the receipt, with their digests checked; the tests are in `tests/test_p132.py`.
   - (i) **The old rule:** run through the staged `sc1b_census.py` at the record's bracket count, it reproduces both
     committed `why` strings exactly. The canary's census was its only failed check.
   - (ii) **The new floor:** at the record's 8 replays and at the new 12, it is at most what the reader keeps. The
     reader's own count, N − 3, is read, not assumed. At 12 replays, one long period still reads ok and two void.
   - (iii) **The main census:** its 64 replays keep at least SC1b's `MIN_STEPS`.
   - **Mutation checks:** the registration's floor (8, − 2) fails (ii) on `(8 - 2) <= 5`. A floor with no tolerance
     (− 3) fails (ii) because a single long period voids the canary.
3. **Cost:** four more replays in each short capture, a few seconds.

**Unchanged:** the reducer, the box, the class map, the rule, every prediction and the budget. **Re-pinned:**
`p132_run.sh` only.

**The relaunch** is a new packet after this merges: a fresh host poll, the machine and offer set, a dry validation of
the exact launch manifest, and the maintainer's ACK. Nothing paid happens before that. Under the lane's two-strike rule,
a second paid failure simplifies the lane instead of retrying it.

## Amendment 2 (2026-10-16, before any relaunch; the card is the reference card)

**Why.** The rule reads absolute milliseconds against `sd2-5090-3`: verify within 5 %, I_b(n) within 10 % of I(n), and
the BUILD thresholds. `sd2-5090-3` ran on one card, GPU-38c66e15 on machine 55913. A matched pair in the records shows
that another RTX 5090 does not reproduce those milliseconds:
- `p116-5090-1` ran on machine 152440 and `p116-5090-2` on GPU-38c66e15;
- the e4b package bytes, kernel package, torch, model, prompts and box were the same;
- per decode step the other card took 0.847–0.867× the time at B = 1 and 0.933–0.938× at 16 rows, with each run's
  self-pairs within 0.3 %.

On that card the anchor would read about 3.6 ms against 4.1825 ± 5 %: a VOID by host, not by the build.

**Normalizing by the anchor does not rescue it.** The same pair breaks both bands when one B = 1 anchor normalizes a
multi-row step:
- the 16-row / 1-row step ratio moves +7.7 to +10.8 % against the 5 % band;
- the increment ratio moves +15 to +17.7 % against the 10 % band.

The one-row and multi-row phases scale differently between cards. So the registered rule stays in milliseconds on the
reference card, and the box runs only on that card.

**The amendment (the maintainer's ACK, 2026-10-16).**
- Right after the card-class refusal, before anything is installed or fetched, `p132_run.sh` reads the uuid (field 4
  of `forensics.txt`'s first line).
- It refuses with **rc 18** unless the uuid is GPU-38c66e15-7fac-e3bc-9760-a49deb41e4d6.
- It records `card.json` (uuid, reference, match) and a `P132_CARD` line in `summary.txt` either way.
- A refusal costs a few minutes of rental and reads nothing. A card refusal (rc 18), like the class refusal (rc 15),
  reads nothing and is not a strike.
- The launch pins the card itself, beside this check: on machine 55913 the offer id maps to one card. Offer 40421555
  gave GPU-38c66e15 in both `p116-5090-2` and `sd2-5090-3`, and offer 40421556 gave another card in `sd2-5090-1`.

**Tests, real records first** (`tests/test_p132_card.py`):
- `sd2-5090-3`'s committed `forensics.txt` (its digest checked against that receipt's SHA256SUMS) must pass;
- `p132-5090-1`'s `forensics.txt` (`tests/fixtures/p132_canary/`, byte-identical to its receipt, from GPU-280958b0)
  must refuse with rc 18;
- the bash cases run the runner's own block.

**Unchanged:** the subject, the instrument, the rule, the predictions, the budget and Amendment 1. **Re-pinned:**
`p132_run.sh` only.

**If the reference card does not return,** the lane cannot run as registered. It would need a new registration that
takes its reference in the box, not a third amendment.

## Closed (2026-10-18): not run, the reference card was unavailable

The reference card (machine 55913, offer 40421555, GPU-38c66e15) was listed in **0 of about 4,050 reads over 34.5 h**
(2026-10-16 18:52Z to 10-18 05:20Z). The same machine's other card appeared 71 times. P132 as registered cannot run
on any other card (Amendment 2), and P116's matched pair shows that normalizing by the B = 1 anchor cannot carry its
rule to another card. **P132 closes "not run"; that is not a strike.**

The question moves to a new registration, the verify split (`bench/vsplit/PREREG-vsplit.md`). There every number the
rule compares comes from the box that runs it, P132's box is reused byte-identical, and the desk prediction is
restated as fractions of the increment.
