# P225: a trained attention adapter through the paged server, on hosts the screen proves can run it (P224's successor)

**Status:** DRAFT, registered before any data once merged. P224 is CLOSED at its registered stop
([`../p224/PREREG-p224.md`](../p224/PREREG-p224.md), Amendment 6 item 5). Its records and verdicts are unchanged, and
nothing here converts its VOID readings into evidence. This lane is the directed successor (e4b#1998) on the
same scientific question, with its own id, registration, run ids and receipt family. It needs one independent review,
then the maintainer's pass, before any reading.

## Why

P224's question was never answered. Its first reading was too small to judge. Amendment 6 resized it, and the second
reading ran on a host that could not finish training in time: training stopped at 65/80 steps under a fixed 1,800 s
alarm, on a box whose effective CPU quota was 11.52 against 46.08 on the first box. No trained adapter existed to
compare. The registered stop then closed the lane.

Two lessons decide this design:
- **The screen must floor the resource the slow phase depends on, measured on the box.** P224 screened VRAM, disk and
  host RAM, and never the effective CPU.
- **Every fixed alarm must hold on the slowest host the screen admits**, not on one fortunate sample.

Before the freeze, two explore boxes (tier #348; timings are design inputs only, never quoted as findings) tested
the obvious candidate, thread oversubscription: torch sizing its pools from the visible CPUs instead of the cgroup
quota. It was NULL on both boxes (ABBA, pools confirmed from inside the process, almost no throttling). What the
timings track is **single-thread speed**: train and the reference step keep about one CPU busy dispatching GPU work.
The screen therefore floors single-thread speed, measured on the box, and the CPU quota is recorded only.

**The question:** on one RTX 5090 with Qwen3-30B-A3B @ad44e777, does a genuinely trained Loggetta attention adapter
keep its measured learned behaviour on the quantized paged serving path, with no dropped adapter target, no base
substitution and no unreported fallback? The expert adapters (D0) are out of scope: an attention-only result licenses
no claim about expert adapter serving.

## What is reused from P224, and what changes

**Reused, staged by sha256** (`staged-p225.sha256`; every reused file is pinned at P224's bytes, and any edit to one is
listed below):
- the arms HF0, REF, P0 and PA;
- the canary task;
- the estimator: the mean over distinct sequences and its paired SE;
- G1 graded on the assistant positions, against the all-token margin `min(0.05, max(0.01, 2 x |loss(P0) - loss(HF0)|))`;
- the ratio bar 0.95, SE_FRACTION 0.2, FAITHFUL and MIN_REF_GAIN;
- G2 to G5 and the reported lines, as P224 Amendment 5 left them;
- the G3 graded cells (W1 `itl_wall_ms`, `ttft_any_ms`, `tok_s`) under the band rule.

**Changed** (each with its reason):

| | P224 | P225 |
|---|---|---|
| host screen | VRAM, disk, host RAM | **+ single-thread speed and sustained download, measured on the box before any install; the CPU quota recorded** |
| CPU thread pools | torch's default (visible CPUs) | **unchanged (the explore A/B was null on two hosts); each pool's size recorded** |
| alarms | fixed, priced from one host | **each derived from the slowest admitted host, with the invariant tested** |
| "trained" | train exit 0 | **80/80 steps from the trainer's record, and a LEARNED gate on REF vs HF0** |
| adapter check | schema v2 | **plus every registered target present with nonzero B, provenance on the real file** |
| engagement | identity over HTTP | **plus the server's own merge record: all targets merged, never a log grep** |
| SE too large | VOID | **INCONCLUSIVE, a final answer** |
| stop | two box-started VOID / HARNESS_ERROR | **two strikes, as defined in the table below; host refusals capped separately** |

## The held-out set: extended from P224's, sized from variance only

**The set.** P224 Amendment 6's 688 rows (canary pin `f58d2605...`, 86 topics x 8 templates, all distinct) are kept
byte for byte as the PREFIX. The training data and the HTTP triggers are unchanged. P225 appends 76 new topics from a
fixed word list (`p225_canary.py`; each word substring-free against every P224 topic and every other word), so
**K = 162 topics and N = 1,296 rows**. The canary pin is the runner's `CANARY_SHA256`, and the test asserts that the
first 688 rows equal P224's.
- The P224 rows were fixed by construction, as a cross product, before any outcome existed on 432 of them.
- The PA-vs-REF values on the other 256 came from a different trained adapter, and nothing is selected against them.

**The sizing rule** (`p225_size.py`, ruled before the freeze). Each run trains its own adapter, and the per-row SD of
loss(PA) - loss(REF) on the assistant positions varies BETWEEN adapters. The verdict reads the estimated SE against
0.002, so the rule prices both sources of a miss:
- log SD is treated as normal across adapters. A new adapter's log SD is predictive t(n - 1) about the mean of the
  input logs, with scale v x sqrt(1 + 1/n) (v: the sample SD of the logs; n = 5);
- the run's own SD estimate adds relative noise sqrt((kurtosis - 1) / (4N)) on the log scale, with the LARGEST input
  kurtosis (4.6). A bootstrap over rows agrees with that formula to 0.002 on every input run. The two scales are
  combined in quadrature;
- P(INCONCLUSIVE | N) = P(log SD_new + noise > log(0.002 x sqrt(N)));
- **K is the smallest whole number of topics with P(INCONCLUSIVE) <= 0.10.**

**The inputs** (variance and shape only: every SD and kurtosis is computed with the mean removed; no effect value
entered the sizing; the explore runs are design inputs and never readings):

| receipt | role | distinct rows | SD of PA - REF (assistant positions) | per-row kurtosis |
|---|---|---|---|---|
| `p224-5090-1` | P224's first reading | 256 | 0.0481 | 4.21 |
| `p225x-5090-3` | explore | 688 | 0.0663 | 4.45 |
| `p225x-5090-4` | explore | 688 | 0.0470 | 4.60 |
| `p225x-5090-5` | explore | 688 | 0.0621 | 3.07 |
| `p225x-5090-6` | explore | 688 | 0.0566 | 3.35 |

**The result:** K = 162, N = 1,296, P(INCONCLUSIVE) = **0.0991** (0.1013 at K = 161). The check beside it: the 90 %
predictive bound SD_90 = 0.0717 needs 1,285 rows. The estimate's own noise moves the prior by only about 0.002, since
it is small next to the between-adapter spread (log-SD sd 0.152). A stricter target (a 1.06 factor on SD_90: K = 181,
P about 0.067) was considered and not chosen.

**The a-priori chance of INCONCLUSIVE is therefore 0.0991.** An INCONCLUSIVE is a final answer for P225, not a reason
to resize.

## The host screen (mechanical, measured on the box)

Each floor is read by the runner on the rented box, before the first install command. A box under any floor refuses as
**HOST_REFUSED**, with every measured value recorded. The refusal path reads ONLY the floor variables below; a test
asserts that no other condition can refuse there.

| floor | how it is measured | value |
|---|---|---|
| single-thread speed | the runner's own probe, the same fixed single-thread Python loop as the launcher's `host_facts` `cpu_loop_mops`, run for at least 2 s per run, the MEDIAN of 5 runs, in Mops; the launcher's own best-of-3 is recorded beside it | **>= 30 Mops** (the ruling; no fallback tier at this N) |
| download | one timed GET of the checkpoint's first shard from the same HF endpoint the fetch uses: at least 256 MB or 20 s, with its HTTP status (a 429/5xx is a refusal reason, not a number) | >= 60 MB/s (the slowest measured: 62.7) |
| host RAM | `/proc/meminfo` | >= 60 GiB |
| free disk | `df` on the work directory | 100 GB |
| card | RTX 5090, cc 12.0, VRAM >= 30 GB | as P224 |

**Recorded, never refusing:** the cgroup CPU quota (`cpu.max`), visible CPUs, every thread pool's size from inside
each process, `/proc/stat` steal %, and a Mops re-probe at every phase boundary, so a mid-run speed change would show
in the receipt. `p224-5090-2`'s extra slowness stays UNNAMED; this registration offers no explanation of it.

**Pinning, a preference only.** The on-box median-of-5 probe is the ONLY refusal. Every pin names the host.
1. First, hosts the registry measured at >= 30 Mops.
2. Then, unmeasured hosts whose CPU model (the offer's `cpu_name`) is in a family the registry measured FAST (table
   below).
3. Never a family the registry measured SLOW. A family in neither row is not pinned while the family rule stands.

The family table, frozen here. It is a design input from the host registry's 31 machines with a `cpu_loop_mops`
record as of 2026-10-22 (Python 3.11, the launcher's loop), not a finding. The two groups do not overlap. The offer's
listed clock does not separate them (a Zen 5 EPYC listed at 2.6 GHz measured 41 Mops; a Zen 4 EPYC listed at 3.6 GHz
measured 25.9), so the rule reads the model, never the clock.

| group | CPU families | measured Mops |
|---|---|---|
| FAST | AMD EPYC 9xx5 (Zen 5: 9115, 9655, 9755 measured); AMD Ryzen 9000-series desktop (Zen 5: 9950X, 9800X3D measured); Intel Core Ultra 9 2xx (285K measured) | 36.2 to 54.5 |
| SLOW | AMD EPYC 7xx2, 7xx3 and 9xx4 (Zen 2, 3 and 4); Intel Xeon Silver | 13.0 to 25.9 |

A fast-family host that reads under 30 on the box is HOST_REFUSED and recorded. It also RETIRES the fast-family
preference, as a registered event (**FAMILY_RULE_RETIRED**: one log line naming the run and the host's CPU model and
reading; its own row under Outcomes; never a strike). The box's screen records its CPU model (`/proc/cpuinfo`) and
family (`p225_probe.py`'s two regexes are this table), and the renter-side `p225_reduce.py --classify` prints the
`P225_EVENT FAMILY_RULE_RETIRED` line when a fast-family host fails the speed floor. From then on, pinning is step 1,
then any unmeasured host outside a family measured SLOW. The slow-family exclusion stays, since it rests on its own
measurements. A refusal costs about USD 0.10 (the boot only).

**Where strike-eligibility begins:** at the first install command, run after every floor has passed. A refusal before
it is HOST_REFUSED and is never a strike.

## The alarms, from the slowest admitted host

Every fixed phase alarm (fetch, bake, train, ref, each slot, collect) and the total guard are derived as follows:
- **The model form, fixed here:** each CPU-dispatch-bound phase (train; the reference process, REF and HF0 together)
  is interpolated linearly in 1/Mops between the two measured speeds, 36.4 and 18.6. It is used ONLY between them,
  never outside: its intercepts are negative (train -2.8 s, ref -1,104 s), so it would extrapolate absurdly. A power
  fit (exponents 1.31 train, 1.54 ref) is steeper and gives less at 30 Mops; the 1/Mops line is the conservative one.
  Fetch and bake are priced at their slowest measured time.
- **need(phase):** the model time at the floor, times 1.75 for the three phases no screen floor bounds, times 1.5.
  The 1.75 is the maintainer's margin for `p224-5090-2`'s unexplained extra ~1.75x on train, which no probe has
  named. It applies to train, the reference process, and the FETCH: the screen's download probe is ONE stream while
  the fetch runs 8 workers, and the measured pairs do not track (62.7 MB/s gave a 297 s fetch, 71 MB/s gave 461 s;
  the slowest fetch, 806 s, had no probe). A fetch timeout falls after strike-eligibility begins, so the screen must
  not be trusted to bound it.
- **alarm(phase):** at least 1.25 x need(phase).
- **The guard:** (the sum of the needs + install 450 + reserve 600) x 1.25 + 300 s of boot.

**The invariant test** parses every alarm and NEED out of the runner's own text (never restated in the test), computes
need() from the derivation table below, and asserts alarm >= need x 1.25 for every phase at the floors.

**The derivation table** (design inputs only; no number here appears in any result):

| receipt | single-thread Mops | cgroup quota | visible CPUs | threads | fetch s | quantize s | train s / 5 steps | bake s | ref step s (REF + HF0) |
|---|---|---|---|---|---|---|---|---|---|
| `p224-5090-1` | 36.4 | 46.08 | 48 | default | 122 | 12 | 5.9 | 140 | 478 at 256 rows (about 1,247 at 688) |
| `p224-5090-2` | 18.0 | 11.52 | 24 | default | 806 | 101 | ~25 (stopped at 65/80) | 362 | none (no adapter) |
| `p225x-5090-3` (explore) | 18.6 | 11.52 | 48 | A/B null; default | 297 | 15 | 14.25 | 126 | 3,497 at 688 |
| `p225x-5090-4` (explore) | 17.85 | 7.68 | 32 | A/B null | 461 | 18-19 | 15.4 | 151 | over 3,600 at 688 (REF done; the explore alarm cut HF0) |

**At the 30-Mops floor and N = 1,296** (`p225_derive.py` writes the runner's lines; the test parses them back):

| phase | model s at the floor | need s | alarm s (>= x 1.25) |
|---|---|---|---|
| train (80 steps) | 698 (1/Mops between 559 and 1,184) | 1,840 (x 1.75 x 1.5) | 2,300 |
| reference process (REF + HF0) | 3,274 (22 s load, then the 688-row figure 1,749 per row, scaled to 1,296) | 8,600 (x 1.75 x 1.5) | 10,750 |
| fetch | 806 (slowest measured) | 2,120 (x 1.75 x 1.5) | 2,650 |
| bake | 362 (slowest measured) | 550 (x 1.5) | 690 |
| each of 6 server slots | 98 (slowest measured plain slot) | 150 (x 1.5) | 880 (covers the slot's scoring term) |
| scoring term, 2 slots | 363 (0.28 s a row, read from each run's FIRST slot, so it carries the cold start) | 550 (x 1.5) | (inside the slot alarm) |

The 1.75 margin applies to train, the reference process and the fetch only. Every measured first slot (the slowest:
`p224-5090-2`'s 364 s at 688 rows) sits under need(slot) + need(score) at 688 rows, and `p225_derive.py`'s self-test
checks it. The needs plus gate 60 and reduce 60 sum to 15,230 s. The guard is (15,230 + install 450 + reserve 600) x
1.25 + 300 = 20,650 s, **5.74 h**: inside the launcher's 6 h teardown cap (the fill refuses above 5.9 h). The worst
case is 5.74 x USD 1.05 + 66 x USD 0.011 = **USD 6.75**.

**The 30-Mops pricing is CONSERVATIVE.** Every host the screen would admit, measured so far, reads 36.2 or more, so a
need is a bound, not an expected runtime.

**No fallback tier at this N.** An 18-Mops tier would need an 8.9 h guard at 1,296 rows, over the 6 h teardown cap; it
fits only up to 688 rows. It is withdrawn, stated plainly: if the 30-Mops market cannot be reached, the lane ends
UNRUNNABLE (Outcomes).

## The trained adapter is proven, on the real file

Before any scoring, the runner reads the adapter Loggetta's train wrote, and the trainer's own record:
- **80/80 steps**, from the trainer's record. Fewer is **VOID INCOMPLETE_TRAIN**, and it is a strike.
- the provenance Loggetta's train actually writes (its manifest, schema `backend-adapter/1`): the manifest's
  `resolved_model_revision` equal to the registered base revision `ad44e777`, and the manifest's sha256 equal to the
  real `adapter.safetensors` file's; the quantization fingerprint (attention 192 modules, experts 96) equal to the
  registration's. Any mismatch is **VOID ADAPTER_REVISION**, **ADAPTER_SHA** or **ADAPTER_FINGERPRINT**.
- every registered attention target (q, k, v and o in each of 48 layers) present, its B tensor nonzero, and the count
  equal to the schema's. Any failure is **VOID ADAPTER_<reason>**.
- **The LEARNED gate:** on the held-out assistant positions, loss(HF0) - loss(REF) minus 2 SE must be at least
  MIN_REF_GAIN (0.05 nats). Otherwise the outcome is **NOT_LEARNED**, a named outcome that is not PASS, and it is a
  strike.
- **Disjointness by hash, on the box:** no held-out row equals a training row (the full row text's sha256, read from the
  real data files). Otherwise **VOID LEAK**.

## Engagement and identity

The paged server must prove that it serves the adapter:
- **From its own record, never a log grep:** every one of the 192 targets merged. "Some layers but not all" is **VOID
  PARTIAL_ENGAGEMENT**.
- A request naming the base model is refused by name (HTTP 404) while the adapter is served (P224's identity check);
  otherwise **VOID IDENTITY**.
- The adapter served is the trained file: the server's adapter id equals that file's sha256. Otherwise **VOID
  IDENTITY**.
- The paged server ran its graphed path (the slot record's `graphs` is true). Otherwise **VOID PARTIAL_ENGAGEMENT**.
  The paged server has no eager fallback path to count. The held-out losses come from P224's teacher-forced scoring
  pass on the served model, unchanged.

## Outcomes

| outcome | when | strike | counts toward the 3-box cap |
|---|---|---|---|
| **PASS** | G1 within the margin, ratio >= 0.95, SE < 0.002, every gate | ends the lane | yes |
| **FAIL** | G1 or ratio outside, at SE < 0.002 | ends the lane | yes |
| **INCONCLUSIVE** | SE >= 0.002 | ends the lane (final) | yes |
| **NOT_LEARNED** | the LEARNED gate fails | yes | yes |
| **VOID** (INCOMPLETE_TRAIN, ADAPTER_*, LEAK, PARTIAL_ENGAGEMENT, IDENTITY, a missing record) | as named | yes | yes |
| **HARNESS_ERROR** | the runner fails after strike-eligibility began | yes | yes |
| **HOST_REFUSED** | a floor fails before the first install | no | no; capped separately (below) |
| **UNRUNNABLE** | the HOST_REFUSED cap trips a second time (8 refusals in all) | no: a lane end, no reading | no |
| FAMILY_RULE_RETIRED (an event, not an outcome) | a fast-family host reads under 30 on the box | no | no; one log line, and the pin order changes as registered |
| pre-create refusal | no instance existed ($0) | no | no |

**The strike call is mechanical.** The runner writes `run/STRIKE_ELIGIBLE` once every floor has passed, just before the
first install. On the renter side, `p225_reduce.py --classify --work W` runs on EVERY fetched box (W is the box's work
directory as the driver fetched it). It reads the runner's own exit-code file, the marker and the screen line, and
prints the registered event. It FAILS CLOSED: it refuses (exit 2, no class) unless W holds exactly one exit-code file,
`RUN_ID` and `run/`, so a wrong directory level can never read as "no strike". A REFUSED classify halts the lane
until the box's files are read by hand, and the hand reading classifies by this table: if `run/STRIKE_ELIGIBLE`
exists, or nothing could be fetched after the box started, it is a strike (HARNESS_ERROR: a box killed, out of memory
or destroyed after the marker writes no exit file). Only a box with no marker is PRE_INSTALL_HALT (for example one that
exited 78 or 130 before it wrote `RUN_ID`). The refusal message says whether the marker exists. The test runs the
command line on each case:

| runner exit | before `STRIKE_ELIGIBLE` | after it |
|---|---|---|
| 0 | (cannot occur) | an answer (PASS, FAIL or INCONCLUSIVE) |
| 13 free disk, 15 card / compute capability / VRAM, 16 host RAM, 18 CUDA unusable (torch imports; the device does not answer), 19 the host screen | **HOST_REFUSED**: no strike; counts toward the cap of 4 | (cannot occur) |
| 9 staging or hash pins, 10 torch does not import (the image), 28 the image's torch is not the pinned build, 41 the guard does not fit the deadline, 78 run id / env / frozen knobs, 130 a signal, or any other code | **PRE_INSTALL_HALT**: no strike, not capped; a kit, image or launch fault, so the lane halts until it is fixed, never relaunched on the same kit | a strike |
| 27 (VOID or NOT_LEARNED), 22 (reducer failed), any other | (cannot occur) | a strike |

G3 (latency) and the cost are reported with every outcome that reaches the reducer.

**The stop (nonadaptive):**
- Two strikes end the lane at STOP.
- The first PASS, FAIL or INCONCLUSIVE ends it.
- No top-up and no N change after data.
- **The HOST_REFUSED cap:** 4 refusals (about USD 0.10 each, the boot only). After that, the lane posts the
  market's state (a listing census against the registry) and pauses. It resumes on a later market read with no
  amendment. A second trip (8 refusals in all) ends the lane as **UNRUNNABLE**: no fallback tier is registered at
  this N.
- **The lane cap:** at most 3 box-started readings.

## Budget

Worst per box = guard x USD 1.05/h + 66 GB x USD 0.011/GB = **about USD 6.75**, inside the USD 15 per-box rule. Lane
worst: 3 boxes plus 8 host refusals, about 3 x 6.75 + 8 x 0.10 = **about USD 21.1**.

## Sequencing

1. This registration and the kit PR: one independent review, then the maintainer's pass.
2. The explore points are in the derivation and sizing tables, each with its receipt (done before the freeze).
3. The merge.
4. The readings, `p225-5090-1` to `-3`, through pod-launch from the merge commit, or by direct API rental under the
   project's standing rental rules (host pinned, USD 15 per box, verified teardown).

## Amendment 1 (pre-data): refusal-retry run ids

The registration names three run ids, `p225-5090-1` to `-3`, while a HOST_REFUSED box is not box-started and is
capped separately (4 refusals a trip, 8 in all). Every box needs its own id and receipt: the ledger holds one row per
run, so an id is never reused. This amendment adds the refusal-retry ids **`p225-5090-r1` to `p225-5090-r8`**, one for
each refusal the registration allows in all, at the same tier and terms. With `-1` to `-3` they cover the lane's
maximum of 11 boxes. The ids are labels, not budget: the registered caps (3 box-started, 4 refusals a trip, 8 in all)
and the strike rules alone govern. The runner's and the driver's id cases accept them; nothing else changes.
- A box that HOST_REFUSES keeps its own id and receipt, with no strike, and counts toward the refusal cap as
  registered.
- A retry box that passes the screen and starts takes the next unused box-started slot. The 3-box-started cap, the
  strike rules, the stop and the HOST_REFUSED cap are unchanged; nothing is widened.
- It merges only if a box refuses on the screen before any reading has started. A refusal is not measurement data,
  so the amendment is still pre-data. A box launched after it merges launches from its merge commit, since the runner's
  bytes (and their staged pin) change.

## What this lane cannot say

- Nothing about expert adapters (D0) or their serving.
- Nothing for other models, cards or adapter shapes. One RTX 5090, Qwen3-30B-A3B, a rank-r attention LoRA on q, k, v
  and o.
- No default changes.
