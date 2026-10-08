# RA — release anchor

Work item: [#1363](https://github.com/pjordanandrsn/experts4bit-qlora/issues/1363).
Draft for maintainer review, clock read 2026-10-08T19:00:37Z. No rental or
measurement preceded this registration. Owner: Jordan. The maintainer reviews
the registration and merges the PR. Executor: Codex desktop, release-anchor.

RA compares the whole shipped e4b/gnf4 configuration with its predecessor on
one rented RTX 5090. It measures interactions between defaults. It licenses
no default change and makes no comparison of absolute times across boxes.

## Releases and identity

The first baseline is **e4b 0.50.0 + gnf4 0.43.0**, with tag commits read into
`source-pins.json`. There is no successor release yet. Do not substitute main
for a release or invent its version. On each e4b or gnf4 release, compare the
last accepted pair with the new pair. A release of just one package holds the
other package at its accepted version. If both change together, compare the
two pairs; attribute a failure only after bisecting them separately.

Install release wheels into separate venvs. Record wheel SHA256, metadata,
module import paths and installed source-tree digests; `pip freeze` alone is
insufficient. Install both into one locked non-release environment, with no
dependency upgrades by either install. If there is no mutually compatible
environment, report VOID and seek an amendment before measurement.

Each arm's `identity` must contain identical, nonempty values for:

- host boot ID, GPU UUID/name/SM count, driver, CUDA runtime, Python;
- complete non-release wheel lock, torch, triton, transformers, bitsandbytes,
  peft, datasets, allocator configuration, physical core count and thread settings;
- checkpoint and tokenizer revision/file digests, dataset and token bytes,
  decode prompts, serving arrival plan, quality windows, calibration inputs;
- harness digest, this registration digest and `source-pins.json` digest.

Use torch 2.8.0+cu128, transformers 5.18.0, bitsandbytes 0.50.2 and peft 0.21.2
from TC1's field environment for the first proof. Freeze the image's compatible
Triton and every remaining dependency to wheel bytes in the reviewed launch
manifest before renting; if unavailable, amend rather than silently upgrade.
Refuse inherited feature flags. Every `E4B_*`, `GNF4_*` and `NF4_*` key is
removed before each fresh process except the named fixture or telemetry keys
below. Save the cleared environment and each release's resolved configuration.

Each release **bakes its own arena from identical checkpoint/calibration
inputs**. Record its arena, quantization and cache hashes as output evidence.
They need not match across releases: a packing/layout change is the subject.
Same-version runs reuse that version's arena. The new installation must never
import a package, compiled extension or cached graph from the old venv.

## One fixed battery

Reading model: `Qwen/Qwen3-30B-A3B` at the revision read from P115 into
`source-pins.json`. Harness helpers run at the registered SHA256 bytes in that
file, identically against both installations. Do not run old/new copies of the
harness: a changing instrument would confound the comparison.

Each position runs the battery in this order, with fresh processes per item:

1. **TC1 field training.** Reuse `tc1_arm.py` (e4b fused arm) and its
   `tp4_alpaca.py` token builder: Alpaca, unpacked, sequence 2048, microbatch 2,
   accumulation 4, 20 optimizer steps, rank/alpha 16, LR 2e-4, weight decay
   0.001, linear schedule, five warmup steps, adamw_8bit, seed 3407, eval at
   0/20 on eight rows. Attention 4-bit, resident, fp32 adapters,
   `--lora-init matched:3407`, dgrad on. No training feature flag is forced.
   Report median wall s/step over steps 11–20 and peak allocated GB as TC1
   defines them. A **separate** identical profiled arm uses
   `--profile-warm 10 --profile-steps 10`: device ms/step = CUDA self-time
   `profile.device_ms / profile.profiled_steps`. It includes copies, can sum
   overlapping streams, and is not a CUDA-event wall interval. Wall and peak
   verdicts use the unprofiled arm; profiling overhead does not enter them.
2. **P109 W1 and W16.** Reuse `wikitext_rows`, `run_pass` and `slope` from
   `p109_box.py`, through `PagedServeConfig.from_env` + `build_engine`. Sixteen
   distinct 512-token prompts, row 0 alone for W1. SHORT 32, LONG 160, one warm
   pass and three timed passes at each length. Decode tok/s =
   `B * (160 - 32) / (min wall_LONG - min wall_SHORT)`. Invalid/nonpositive
   slopes VOID. Do not use P109's arm wrapper, which fixes slots and graphs.
   Slots, buckets, graphs, folds, grouping and routes resolve independently
   from each release. Preserve P109's request-count/length and digest checks.
3. **One SC2e capacity point.** Reuse `sc2_driver.py` streaming open-loop
   driver: 12 requests/s, 120 requests, 512-token wikitext prompts,
   U[64,256] output tokens, greedy, ignore EOS. One byte-identical Poisson plan
   (seed 112) in all four positions. Four warm serial requests and the SC2e
   64-request burst (seed 998) precede measurement. Context limit 2048 and
   chunk 512 are fixture settings; max_seqs/buckets and speed levers remain
   unset. Goodput = 12 × fraction of VALID requests with TTFT ≤ 1 s and TPOT
   ≤ 0.100 s. Record attainment, p50/p99 TTFT/TPOT, memory and full health/step
   traces, but do not infer a capacity ceiling from one point. No SC2e arm
   environment (SPEEDENV/ROUTEENV/explicit fusion flags) is inherited.
4. **P115 SANE.** Reuse Phase B's `measure_phase`/teacher-forced paged pass
   on **12 wikitext windows, prompt 512, continuation 128**, at group **1 and
   12**, separately. Group 1 exercises the served GEMV; group 12 exercises
   padded multi-row arithmetic. Eager padded scoring is the named measurement
   fixture (`E4B_PAGED_GRAPHS=0` for quality only), with device grouping as
   P115's instrument uses it. All folds and GEMV routes retain release defaults.
   Save NLL and full per-position argmax IDs per window (a wrapper around the
   score helper adds these IDs; it must leave scores unchanged). Old and new
   each score their own R; RA pairs their NLL/argmax arrays. Do not use P115's
   ON/OFF wrapper. Mean NLL difference new−old, absolute bar 0.02 nats, mean
   argmax agreement bar 0.95, at **both group sizes and both ABBA pairs**.
   A negative NLL difference is also subject to the absolute SANE bar.

Fixture-only environment keys: model/revision, arena/calibration path, device
CUDA, all-vram placement, the capacity context/chunk above, quality graphs=0,
and telemetry trace/log destinations. OMP/MKL/thread/allocator settings are
common identity. The executor PR must enumerate exact keys, reject all others,
and validate its normalized records against the reused raw receipts before a box.

## ABBA and the registered reducer

One rental, sequential **old_a, new_a, new_b, old_b**. No concurrent GPU work,
no in-launch retry. One warm-up per helper as above. Give each venv a separate
Triton/extension cache; warm before measuring. Each position loads afresh.
`run.json` orders these tags and names old/new release provenance and the full
common identity; `arm_<tag>.json` holds the native receipts plus RA metadata.
Verify `SHA256SUMS` over all staged instruments and all raw/normalized outputs.
The stdlib `ra_reduce.py --dir DIR --out verdict.json` reads that directory;
`--self-test` exercises the decision boundaries and deliberate invalid records.

Performance metric bounds are fixed now, not fitted after a reading:

| metric | better | same-version and cross-version bound |
|---|---|---|
| train wall s/step | lower | 5% |
| train profiled CUDA self ms/step | lower | 5% |
| train peak allocated GB | lower | 3% |
| decode W1 tok/s | higher | 7% |
| decode W16 tok/s | higher | 7% |
| SC2e point goodput req/s | higher | 10% |

For each metric let cost be the value when lower is better, and its reciprocal
when higher is better. Same-version spread is `max(value_a,value_b) /
min(value_a,value_b) - 1`. Costs are paired new_a/old_a and new_b/old_b.
Use full precision, never rounded display numbers for comparisons.

1. **NOISY** if either same-version spread is strictly above its bound.
2. **REGRESSION** if both cost ratios are strictly above `1 + bound`.
3. **IMPROVED** if both are strictly below `1 / (1 + bound)`.
4. **WITHIN_NOISE** otherwise. Mixed signs stay WITHIN_NOISE.

Zero goodput: both versions zero is WITHIN_NOISE (no capacity demonstrated);
old zero/new positive is IMPROVED; old positive/new zero is REGRESSION.
One zero and one positive within the same version is NOISY. Never emit Infinity
or NaN as a JSON ratio; these zero cases are explicitly tagged.

Quality is read in its own units, not a ratio of signed NLL differences.
At each group size: **NOISY** if either same-version mean NLL drift exceeds
0.005 nats in absolute value or same-version argmax agreement is <0.999;
**REGRESSION** if both cross-pairs fail SANE; **IMPROVED** if both pass SANE,
NLL improves by >0.005 nats in both pairs and argmax remains ≥0.95;
otherwise **WITHIN_NOISE**. One failed SANE pair blocks clearance as
**QUALITY_UNSETTLED** even though it is not a confirmed REGRESSION.
NOISY takes precedence only within its own metric; it must not conceal a
confirmed regression in another metric.

Run disposition: VOID for invalid evidence; FUNCTION_FAIL for broken
same-version decode determinism or frozen training weights; REGRESSION if any
valid metric is REGRESSION; QUALITY_UNSETTLED for one failed SANE pair; NOISY
if any remaining metric is NOISY; otherwise CLEAR. Proofs are labelled PROOF,
never release-clearance evidence. No invalid/noisy/unsettled run clears a release.

## Engagement and sensitivity

The reducer refuses missing/raw non-ok arms; mismatched non-release identities;
wrong versions, wheel/import provenance or input digests; wrong fixture shape;
missing metrics, NaN/Infinity or invalid counts; profiler and wall arms mixed;
missing counters; unexpected environment flags; stand-in attention.

- Training: complete matched init, trainable count equal across arms, matched
  init SHA, all 20 losses finite, C1 frozen-byte equality with byte-flip control,
  nonempty per-step fused-kernel calls and no fallback. The profile covers
  exactly steps 11–20. Read TC1's resolved knobs and path counters, including
  pad/compact/single ladder, checkpoint, absmax and grouping, as evidence.
- Decode: digest every timed pass, verify tokens/request lengths, require each
  requested graph bucket captured and used as its resolved mode requires;
  same-version final tokens and all timed digests must agree. Cross-version
  tokens may differ; quality decides whether that change is SANE.
- Capacity: recompute validity/SLO/goodput from all requests; prompts, arrival
  times, output lengths and request count identical; widest resolved bucket
  exercised by the burst; no accidental eager fallback; reconcile prefill and
  KV counters with admitted requests. A smaller release-selected slot count is
  evidence, not a forced mismatch failure.
- Quality: 12 windows with unique IDs, exact shapes and window hashes; real
  paged attention calls `(cont-1) * layers` per group; padded eager bucket
  stats and per-forward/fusion/GEMV counters. Record default resolution and
  require nonzero calls for each enabled feature, zero vacuous enable. Assert
  the group-1/group-12 dispatch distinction when the release implements it.

Before a rental the executor must pass CPU mutation tests for every engagement
check: wrong input/dependency/version, zero kernel calls, missing graph bucket,
unexercised widest bucket, per-request HTTP/count error, altered window,
stand-in attention, unexpected env flag, and a score mutant beyond SANE.
On-card premise: reused decode graph, KV-step, fused-glue graph and bandwidth
GEMV tests, collected and run, no skips. Version-specific unavailable premise
APIs must be checked with a shared black-box oracle, never called a green skip.

## Proof, budget and consequences

After maintainer approval, implement/review the executor and freeze its launch
manifest. **Proof first:** one RTX 5090, Granite at its read pin, both baseline
installations (old=baseline/new=baseline), ABBA, the same battery/validity logic,
decode 8/24 tokens with one timed rep, quality 12 × 32 positions at groups 1/12.
Training keeps 20 steps to exercise its registered steady/profile window;
Granite changes only the family-derived trainable/module counts. Capacity still
uses 120 requests. A proof may establish harness validity without a performance
licence. No successor release run until a successor exists.

Target reading cost $3–5 including transfers. Proof guard 0.75 h; reading guard
3 h; price and download/storage costs checked against the launcher's current
ledger before buying. Those estimates are targets, not measured feasibility.
Owner limits: **$15 per run without asking, $35 absolute per-run cap,
$100/day across launches**. Above $15 requires Jordan's approval first; never
cross $35 or $100/day. Refuse a plan whose worst-case cost exceeds its authority.
Leave 10 minutes for checksum/fetch-back/teardown; refuse a step that cannot
fit its timeout plus that margin. No retries hidden within one purchase.

Before expensive fetch: RTX 5090, usable CUDA/sm_120, ≥60 GiB host RAM,
≥150 GB free disk, shared lock and all gates proved. A2000 correctness only.
Preserve failed attempts and bills. Fetch/checksum receipts before teardown,
verify rental absence, and retain the provider receipt and cost ledger.

Receipts: `bench/ra/receipts/<run>/` (`git add -f` for logs), with manifest,
native/normalized records, default resolutions, version/import provenance,
SHA256SUMS, reducer verdict, premise/mutation logs, billing and teardown proof.
Code and real receipts ship in separate PRs. No speed claim from synthetic tests.

A confirmed REGRESSION is posted to this work item and the maintainer and
**blocks the next release** until a within-box bisect names the cause. Freeze
all common inputs and the failing metric's rule; vary one package at a time,
repeat ABBA on candidate commits, and confirm last-good/first-bad with the same
mutant-sensitive instrument. Do not excuse the failure by moving the bound.
The maintainer owns release enforcement; Jordan is final authority. Upstream
filings need Jordan's explicit say-so.

**State:** registration/reducer only. Executor, GPU proof, successor reading,
cost feasibility and enforcement integration remain unmeasured work. Review
of this PR authorizes no implicit rental; ask the maintainer for explicit
proof clearance after the executor gates are reviewable.
