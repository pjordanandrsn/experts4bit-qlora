# SC2f: do P130's two prefill defaults lift e4b `serve_paged`'s capacity at 64 slots? `E4B_FUSE_PREFILL_GLUE` and `E4B_PREFILL_LEAN_DISPATCH` at the release's defaults against both at 0, paired on one RTX 5090, with same-seed replicates and a knee block at 12–14 req/s (lane SC2f of #846; registered 2026-10-10)

**Registered** when this file merges to `main`, before any SC2f run. Opened without `ready-to-merge` or auto-merge,
for maintainer review. Box N rents only after DQ11 closes (the standing stabilize-first rule), on **e4b 0.54.0** (the
first release that carries #1608 and #1615), and every paid ACK names the host.

**The code under test** is P130's two flips, both shipped as defaults (`bench/p130/RESULTS-p130.md`, read #1605):
- #1608 (merged `dc2221a3`): `E4B_PREFILL_LEAN_DISPATCH` defaults to `auto`. On the int4 K19 route above 256 rows,
  gate_up reads the token rows itself and down stores in the caller's order: no 4,096-row gather, no unsort.
  Bit-identical on every window; ×0.968 [0.966, 0.970] per 512-token first-chunk prefill graph replay.
- #1615 (merged `451a370353122f5c1ba90acc7653f3ee885c2489`, GitHub `mergedAt` 2026-10-10T21:57:50Z;
  `experts4bit_qlora/` tree `85fcce24cb0cb3fce5cc7858819019d4b0152a72`, the tree its A2000 smoke ran):
  `E4B_FUSE_PREFILL_GLUE` defaults to 1 on
  `qwen3_moe`, so the decode folds take prefill rows. AT_PARITY (bias −0.0035 nats against a 0.0125 bar); ×1.118
  [1.116, 1.119], 4.02 ms of 38.2.
- Together P130 measured **38.17 → 32.99 ms** (−5.18 ms) per 512-token first-chunk replay on Qwen3-30B-A3B int4, one
  RTX 5090.

**Recorded at launch** (read from the objects, not recalled; in the launch note and the read): the release tag, its
commit and `experts4bit_qlora/` tree; the launch commit and its tree, which must equal the tag's.

## Why this lane

**Where e4b stands.** SC2e (`bench/h2h-2026-10-02/sc2e/README.md`): 64 slots with decode buckets up to 64 serve
**12 req/s** on one RTX 5090 at TTFT ≤ 1.0 s and TPOT ≤ 100 ms (s64a: 1.000 / 0.967 at 12 req/s, 0.558 / 0.542 at 16).
That is the number `docs/STATUS.md` and `e4b.serve.sc2e.64-slots-buckets-auto.qwen3.5090.2026-10-07` carry; since 0.50.0
it is the default server at 2,048 tokens a slot.

**Why prefill is the lever at 64 slots.** Past 12 req/s a 64-slot server is bound by its step loop, not by slots: s64a's
queue wait stays under 60 ms at every rate, while each 512-token prompt's prefill (40.9 ms to its first token on s64a,
40.0 ms of it the forward) stalls every decoding row. At 16 req/s prefill alone takes 16 × 40.9 = 654 ms of every second. P130 took
5.18 ms off that forward. This lane reads whether that reaches capacity.

**The model's answer, before data** (`bench/sc2/sc2f_basis.py model`: `serve_capacity` on s64a's own fitted costs from
SC2e's `capacity_check.json`; `on` = the prefill step − 5.18 ms, nothing else moved; this lane's seeds):

| | d1 off | d1 on | d2 off | d2 on |
|---|---|---|---|---|
| serial TTFT / TPOT p50 (ms) | 43.9 / 4.90 | 38.7 / 4.90 | 43.9 / 4.95 | 38.7 / 4.95 |
| 8 req/s: attainment; TTFT, TPOT p50 (ms) | 1.000; 54.8, 18.35 | 1.000; 47.4, 16.90 | 1.000; 55.5, 18.71 | 1.000; 49.7, 17.48 |
| 12 req/s | 1.000; 101.1, 31.83 | 1.000; 76.2, 29.95 | 0.892; 139.5, 34.34 | 1.000; 108.0, 31.62 |
| 13 req/s attainment | 0.808 | 0.992 | 0.658 | 0.783 |
| 14 req/s attainment | 0.717 | 0.992 | 0.717 | 0.783 |
| 16 req/s attainment | 0.542 | 0.558 | 0.542 | 0.567 |
| knee block mean (12–14 req/s, 5 plans each) | 0.901 | 0.953 | 0.869 | 0.954 |

What the model says, and why the instrument is shaped as it is:
- **The flips should show at the knee,** 12–14 req/s: +0.053 and +0.086 mean attainment over the knee block. At 16 req/s
  both arms are past saturation (0.54–0.57).
- **One plan a draw cannot read the knee.** Over 60 seeds of 120 requests (`sc2f_basis.py spread`, draw 1's costs) a
  plan at 13 req/s reaches 0.95 in 67 % of seeds with `on` and 43 % with `off`; at 14, 43 % and 12 %. SC2's ceiling on
  one plan a draw is close to a coin flip there. Hence the **knee block**: 15 plans a server at 12, 13 and 14 req/s.
- **A server's level offset is the same size as the effect.** A captured runner can carry a whole-life offset of
  3–3.6 % (P124, P126). On the model, +3.6 % on every step costs the knee block 0.028–0.061 (`sc2f_basis.py
  sensitivity`), as much as the flips gain. Hence **two servers per arm per draw at the same seeds**: their difference is
  the noise bound the rule reads, rather than an assumed one.
- The model ran pessimistic at SC2e's knee (s64a draw 2 at 12 req/s: model 0.892, measured 0.967). The predictions below
  are paired ratios and differences, not absolute levels.

**What comes after it** (`sc2f_basis.py levers`, the same costs with the flips on; attainment at 16 req/s, d1 / d2):
prefill × 0.75 → 0.933 / 0.683; prefill × 0.5 → 1.000 / 1.000; decode × 0.8 → 0.883 / 0.667. On the model prefill stays
the larger lever per unit at 64 slots. A FLAT reading says the mapping from prefill time to capacity is not what the
model assumes; the next lane is then chosen from this lane's census, not from more prefill glue.

## The box (N) and the stack it pins

- **One box, e4b only.** `SC1_BOX=N` (`bench/sc2/sc2f_box_n.sh`), sourced after `sc2_box_e.sh`, `sc2c_box_h.sh` and
  `sc2e_box_l.sh`, whose server start (`l_server_start`: ROUTES rc 47, SLOTS rc 49), early check (`h_early`) and burst
  (`l_burst`, rc 48) it uses unchanged. `bench/sc1/sc1_run.sh` and `sc1_drive.sh` gain box N and nothing else.
- **e4b.** The launch commit is `main` at or after the `v0.54.0` tag, with that tag's `experts4bit_qlora/` tree (the
  launch wrapper compares the two trees and refuses otherwise). Box N's tripwire refuses
  an e4b without both defaults (rc 9): unset `E4B_FUSE_PREFILL_GLUE` must resolve to (1, default-allowlisted) on
  `qwen3_moe`, and unset `E4B_PREFILL_LEAN_DISPATCH` must read `auto`. It records `__version__`.
- **grouped-nf4-gemm.** **v0.45.0** (`724ccc454f006c1a46836e434e997f31f293747f`), the latest release, installed from git
  as on box L. If the release under test ships beside a newer grouped-nf4-gemm, an amendment moves this pin before the
  launch. SC2e ran v0.42.0; no ratio crosses lanes.
- **Arms**, each SC2e's s64a server (`SPEEDENV`, the three folds, `E4B_PAGED_FUSE_QKV=1`, `ROUTEENV`, 64 slots,
  `E4B_PAGED_BUCKETS=auto`, 2,048 tokens a slot, chunk 512, bulk KV and the prefill graph at their defaults, both traces
  on), differing only in P130's two knobs:

  | arm | `E4B_FUSE_PREFILL_GLUE` | `E4B_PREFILL_LEAN_DISPATCH` | the start `/health` must read (FLIPS) |
  |---|---|---|---|
  | `on` (the release's defaults) | unset | unset | `levers.prefill_glue` = {mode 1, source default-allowlisted}; both raw env values null; `prefill_lean_dispatch` auto; every K19 call above 256 rows on the `lean` dispatch, the prefill capture's `chained\|lean` among them, none on a `split` table (Amendment 1); the norm, layer and attention folds each seen on prefill rows |
  | `off` | 0 | 0 | {mode 0, source explicit}; both raw env values "0"; `prefill_lean_dispatch` 0; every K19 call above 256 rows on the `gather` dispatch, the prefill capture's `chained\|gather` among them, none on a `split` table (Amendment 1); no prefill fold |

  The box unsets both knobs when it is sourced, so only the `off` arm names them. FLIPS (`sc2f_reduce.py --flips`, the
  reducer's own gate) reads what the startup captures took (`prefill_routes.seen`: the first-chunk prefill graph and the
  64-row decode graph both call K19 above 256 rows), so a server whose knobs did not engage stops before any run that
  counts (rc 43).
- **P2 reaches the 64-row decode step too:** 64 rows × top-8 = 512 routed rows, above 256. P1 does not reach decode: the
  folds already take up to 64 rows there.
- **Otherwise box L's s64a server, read the same way:** routes are main's defaults, read from each server's own
  `/health` (rc 47); 64 slots = `kv_slots`, buckets 1–64 all captured, scratch 64 (rc 49); after 4 warm requests the
  prefill graph on and bulk KV engaged, then the 64-request burst replays bucket 64 with no eager step (rc 48).
- **Image and checkpoint.** As SC2e: `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, torch 2.8.0+cu128; Qwen3-30B-A3B @
  `ad44e77`, baked to the int4 arena on the box.
- **Host.** Pinned in the ACK (Vast machine id or offer id; the launch passes `--vast-pin-machine` and/or
  `--vast-pin-offer`; a host gone or over the ceiling means a new ACK, never a substitute). SC2e's machine (58411) is
  preferred when offered, so that `off` re-reads SC2e's server on its own host; otherwise any RTX 5090 at 575 W. The
  paired reading holds on any host; a ceiling compares with SC2e's 12 only on the same host, and the read says which.

## The instrument

- **Driver and prompts.** SC2's, unchanged: each request one 512-token wikitext-2 row, `max_tokens` from U[64, 256],
  greedy, `ignore_eos`, streaming, open loop.
- **Eight servers, each a cold start.** Draw 1 runs `on_a`, `off_a`, `off_b`, `on_b`; draw 2 runs `off_a`, `on_a`,
  `on_b`, `off_b`. Every server of a draw runs the same seeds: serial = the draw; ladder = draw × 100 + rate; knee
  = 10,000 × draw + 100 × rate + j.
- **Per server:** 4 warm serial requests; the burst; serial, 24 requests; the **ladder** at 4, 8, 12, 13, 14 and 16 req/s,
  120 requests each; then the **knee block**: 12, 13 and 14 req/s at four more seeds each, 120 requests each. 13 and 14
  are new, between SC2e's 12 and 16. 1 and 2 req/s are dropped: every SC2e server held 1.000 at both, and the serial
  run reads the unloaded path.

## The rule (`bench/sc2/sc2f_reduce.py`, 34 self-test cases)

**Gates, per server.** A failure names the server and VOIDs the reading: ROUTES (`sc2c_reduce.routes_bad`), SLOTS
(`sc2e_reduce.slots_bad`, s64a), FLIPS (above), ENGAGED (`sc2e_reduce.engagement` at 64 slots, with every request the
server admitted counted: warm, burst, serial, ladder and knee), PROMPTS (512 prompt tokens on every request).

**Statistics.**
- A **row** is one arm at one ladder rate over its four servers: UNREAD if a run is missing, INVALID if a run has an
  invalid request, else VALID; it meets capacity iff every server attains ≥ 0.95. An arm's **ceiling** is the largest
  ladder rate whose row is VALID and meets capacity (unread if any ladder row is UNREAD or INVALID).
- A server's **knee** is its mean attainment over the knee block's 15 runs (12, 13 and 14 req/s; the ladder's seed and
  four more each).
- Per draw: **gain** = the mean knee of its two `on` servers minus that of its two `off` servers; **noise** = the larger
  of |on_a − on_b| and |off_a − off_b|; **bar** = max(0.03, noise).
- Per draw: **headroom** = 1 − the mean knee of its two `off` servers: the most gain the knee block can show, reported
  always. SC2e's s64a measured above the model at the knee (0.967 at 12 req/s in draw 2, against 0.892 modelled), so a
  real `off` near 0.97 would leave less headroom than the bar.

**Outcome**, decoupled from Q1–Q6:
- **VOID** if a server fails a gate; **UNREAD** if a number the outcome needs is missing.
- **REGRESSED** if `on`'s ceiling is below `off`'s; or, at 4 or 8 req/s, an `on` server attains less than the draw's lower
  `off` server minus 0.05; or gain ≤ −bar in both draws. Consequence: to the maintainer, since the release then ships a
  serving regression. `E4B_FUSE_PREFILL_GLUE=0` and `E4B_PREFILL_LEAN_DISPATCH=0` restore the old path; whether to
  revert a default is the maintainer's call. The read prints the knee gains beside the trigger: a ceiling inversion
  alone can come from one server's 3–3.6 % offset flipping a single-seed row at 13 or 14 req/s.
- **FASTER** if gain ≥ bar in both draws. Consequence: **the claim moves.** A serving claim row for the P130 defaults
  records the knee gain and the paired TTFT and TPOT ratios on this box. **The STATUS / README capacity number** moves to
  `on`'s ceiling iff that ceiling is above SC2e's 12 (`readme_number`), stated with its host; the claim attributes to the
  flips only the paired difference against `off`. Otherwise the number stays 12 and the claim states the knee gain.
- **KNEE_SATURATED** if the outcome would be FLAT and headroom < bar in either draw. The instrument could not have shown
  the gain, so this is not a measurement of FLAT: FLAT's consequences do not fire, the capacity number stays 12, `on`'s
  ceiling is reported with no claim, and the next step is the maintainer's.
- **FLAT** otherwise. Consequence: **the next lever is not prefill glue.** P3 (more prefill glue) stays parked; the next
  lane is chosen from this census and the prefill expert-GEMM shape census (the $0 row beside this one); STATUS says the
  P130 defaults did not measurably move serving capacity at 64 slots.

**Predictions**, per draw (an arm's value is the mean over the draw's two servers of that arm), each with what a miss
means. No outcome reads them.

| # | prediction | basis | if MISSED |
|---|---|---|---|
| Q1 | prefill forward device time p50 (`sc2e_census`), on − off ∈ [−7.0, −3.5] ms | P130: −5.18 ms per replay | high (less saved): the serve path does not get P130's saving; the read explains it from the step segments, and no serving line is claimed for the flips. Low: reported |
| Q2 | serial p50: TTFT on/off ∈ [0.83, 0.93]; TPOT on/off ∈ [0.98, 1.02] | model 0.882 and 1.000 | TTFT: Q1's saving does not reach the client. TPOT: a prefill knob moved single-request decode; reported to the maintainer |
| Q3 | TPOT p50 at 8 and 12 req/s on/off ∈ [0.85, 0.99]; TTFT p50 at 8 req/s on/off ∈ [0.78, 0.97] | model 0.921–0.941; 0.865 and 0.895 | the model's prefill-stall term is wrong under load; re-fitted from this census before the next lane leans on it |
| Q4 | gain ∈ [+0.03, +0.12] | model +0.053, +0.086 | low: the outcome is FLAT or worse. High: the model under-prices prefill at the knee |
| Q5 | `on`'s ceiling ≥ `off`'s, and ≥ 12 (over the reading) | model: `on` 12 (14 in draw 1, 12 in draw 2), `off` 8–12 | below `off`'s: REGRESSED. Below 12: this box runs slower than SC2e's, stated with the host |
| Q6 | decode-only step p50 at bucket 64, on/off ∈ [0.95, 1.02] | P2 at 512 routed rows: about 1/8 of its ~1.2 ms at 4,096 | high: P2 slows the 64-row decode step; reported to the maintainer with the census |

## Reported, no bar: the census

Per server, from its own traces (`sc2e_census` with s64a's buckets):
- decode-only steps by bucket (step, device time, host segments);
- prefill steps (step, the forward's device time, the decode rows riding them, the direct stall);
- `dec_pieces`; busy-loop gaps; per workload, the mean requests in the server and the queue wait p50;
- VRAM at ready, `free_after_mib`, `pool_mib`, `bulk_flush_mib`;
- the K19 dispatch and prefill fold counts, and `graph_stats`.

**Text agreement, no bar.** Per draw and replicate, between the paired `on` and `off` servers (same seeds), per workload.
P1 changes the prefill arithmetic (AT_PARITY, not bitwise), so agreement below 1 is expected even on the serial run,
unlike SC2e.

**Post hoc** (the read PR; no rule reads it): `serve_capacity` on each server's own fitted costs beside its measured
attainment, as SC2e's capacity check.

## Shakedown, proof and budget

**$0 shakedown** before any paid run: the reducer's and basis's self-tests, the box's checks on fixtures
(`tests/test_sc2f_box.py`), and `SC1_BOX=N SC1_DRIVE_DRYRUN=1` on the controller checkout at the launch commit. The launch
is from a commit whose CI is green.

**Proof** (`sc2f-prove-*`, guard 1.0 h), on Granite-3.1-3b-a800m (NF4 store, SC1's `GR_ENV` unchanged):
- self-tests on the box: driver, SC2 reducer, identity, SC2c and SC2e census, SC2e reducer, SC2f reducer and basis;
- both arms' servers at 64 slots: ROUTES (the proof form), SLOTS, FLIPS in its proof form, warm, `h_early`, the burst
  with bucket 64 replayed, a 6-request serial smoke, and 16-request Poisson smokes at 13 req/s at a ladder seed and at a
  knee seed, every request VALID;
- serial IDENTICAL, `on` against `off`. Granite is not on P1's allowlist and its NF4 store takes no int4 K19, so both
  knobs resolve as each arm sets them and neither engages;
- engagement as the reducer reads it; step traces with 64-row decode steps.

The proof does not run P1's folds or P2's dispatch on prefill rows. FLIPS stops each reading server before its paid runs
if they did not engage (rc 43).

**Reading** (`sc2f-5090-*`, guard 3.5 h): installs ~10 min, fetch and bake 25–40 min, eight servers at ~13–15 min
each (start ~5 min; ladder ~3 min; knee block ~5 min).

**Guards**, as SC2e counted them, at ≤ $0.75 an hour plus download at $0.0078/GB:
- **proof:** 1.0 h ≤ $0.75 + ~$0.10;
- **reading:** 3.5 h ≤ $2.63 + ~$0.55.

Lane ceiling $7 (two proofs, one reading, one re-run). Each run sits under #846's standing no-ask tier for a single run
under $15. Two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane's rentals pending a root-cause note. The
deadline drops draw 2's last servers first; a server lost to the deadline makes the reading UNREAD, never FASTER or FLAT.

## Out of scope

- Comparators; other models, families and cards; prompts other than 512 tokens; chunked prefill.
- 4,096 tokens a slot: there `E4B_PAGED_MAX_SEQS=auto` picks 32 slots on a 5090 (SC5), a different server.
- Quality: P130's reads license both defaults (P2 bit-identical, P1 AT_PARITY).
- Prefill expert-GEMM tiles and their timing: the shape census beside this lane is $0 and shapes only; any timing gets
  its own registration.

## Amendments

### Amendment 1 (registered 2026-10-11, before any SC2f run): FLIPS reads K19's dispatch above 256 rows, not its table

**Why.** As registered, FLIPS required the K19 calls above 256 rows in the start `/health` to be exactly
`chained|lean|gt256` (on) or `chained|gather|gt256` (off). The registration already said that the 64-row decode graph
calls K19 above 256 rows too. It wrongly assumed those calls take the chained table. Bucket 64's decode capture routes
64 × 8 = 512 rows. On Qwen3-30B-A3B (128 experts) that is exactly the bound of `E4B_INT4_WIDE_TILES=auto` (lane P122:
`next_pow2(E) × next_pow2(R) ≤ 128 × 512`), so those calls take the one-launch table: `one_launch|lean|gt256` on `on`
and `one_launch|gather|gt256` on `off`. FLIPS would have stopped every reading server at rc 43, after the fetch and the
bake. Two real records show it:
- #1625's GPU serve smoke on the project's RTX A2000 (correctness only). At knob 0, its 1,024-row calls on 4 experts
  took `one_launch|lean|gt256`, not the chained table. That smoke's wrapper expected `chained` and failed on its own
  expectation, which is how this was found.
- SC2e's committed start `/health` on the RTX 5090 (`sc2e-5090-1`). At 64 slots with buckets up to 64 (s64a, this
  lane's server shape), the startup captures make 384 K19 calls above 256 rows; with buckets up to 16 (s16, s64c), 240.
  Bucket 64's capture adds 144 = 48 layers × 3 forwards. That run predates the dispatch counter, so it shows the count,
  not the table.

**The change: the gate only.** FLIPS (`sc2f_reduce.k19_gt256_bad`) reads the dispatch of every K19 call above 256 rows:
`lean` on `on`, `gather` on `off`. It requires the prefill capture's `chained|<dispatch>|gt256` among them, and refuses
a `split` table on either arm (`E4B_K19_PREFILL_BIG_M` is not this lane's knob and stays unset). The table of the other
calls is `E4B_INT4_WIDE_TILES`'s choice, not P130's, and is not read. The proof's form is unchanged: Granite NF4 takes
no int4 K19.

**Tested on real records first** (`tests/test_sc2f_real_records.py`, 24 passed):
- the smoke's `prefill_routes` from `tests/fixtures/smoke1625/pass2_summary.json`, byte-identical to its receipt
  (sha256 `1f4960b1…`, checked), with `levers.prefill_glue` from `serve_paged.resolve_prefill_glue`, the function
  `/health` reports it from;
- SC2e's six start `/health` files, digests checked.
The real qwen3_moe record reads as `on` on every knob. Its one departure is the prefill capture's chained call, which a
1,024-row forward never routes. With that one call added, `on` passes, and the rule as registered refuses it. Real
split records are refused on both arms. Synthetic departures from the real record come second. They include the `off`
arm, of which no real record exists: its keys follow `hot_residency`, where above 256 rows the dispatch is `lean` iff
the prefill lean dispatch is on, and `E4B_PREFILL_LEAN_DISPATCH=0` turns that off. Restoring the rule as registered
fails 5 of the 24 tests. The reducer's self-test gains 4 cases (34), and its fixture's start `/health` now carries an
s64a startup's calls: 240 chained and 144 one-launch above 256 rows. `tests/test_sc2f_box.py` reads the same shape,
with three new departures.

**Unchanged:** the arms, both knobs, the ladder, the knee block, the statistics, the predictions, the outcomes, the box,
the budget, and the rental gate (e4b 0.54.0, DQ11 closed, an ACK naming the machine and the offer).
`bench/sc1/staged.sha256` re-pins `sc2f_reduce.py` only.

**Both arms on real records** (added 2026-10-11, before any run; the rule is unchanged). The `off` arm, derived from the
code above, now has a real record: a record-only smoke on the project's RTX A2000, `tests/fixtures/smokeflips/summary.json`
(sha256 `1eeb6e02…`, checked). It ran one fresh process per arm on the smoke's qwen3_moe at top-4, so both of an SC2f
startup's row counts are reached at `E4B_INT4_WIDE_TILES=auto`: the serve path's 2,048-row chunks take the chained table,
and a 512-row forward takes the one-launch table. Each cell's `health` is the two blocks FLIPS reads, built by the
`/health` handler's own expressions. Above 256 rows, `on` read `chained|lean` and `one_launch|lean`; `off` read
`chained|gather` and `one_launch|gather`. With nothing added, `flips_bad` passes each arm on its own record and departs
on all seven keys across arms. The rule as registered refuses both arms' real records for the one-launch call. Decode's
calls at or below 256 rows read `one_launch|lean|le256` on both arms (the knob governs rows above 256), and FLIPS does
not read them. The synthetic departures now start from these records: `tests/test_sc2f_real_records.py` (28 passed).
The rule as registered fails 6 of the 28.

### Amendment 2 (registered 2026-10-19, before any SC2f run): the host bar and how a host is pinned

**Why.** Under the registered selection, two watcher windows found no rentable host:
- envelope v2, 2026-10-17 to 10-18;
- envelope v4, 2026-10-18 08:27Z to 23:55Z: 5,327 machine-filtered reads over 15.4 h.

The receipt bar then was clean runs ≥ 3, every recorded power limit 575 W, no host failure and no capped block. Four of
the five machines that met it were never listed in v4. The fifth, machine 26157, listed one offer (54055665) that
refused every create with `no_such_ask`: `sc2f-prove-1` to `-4`, $0, nothing created.

**The change: host selection and the launch mechanism only** (maintainer ruling, 2026-10-18 23:57Z).
- **Bar.**
  - Clean runs ≥ 2 (was ≥ 3). The measurement-quality terms stay: every recorded power limit 575 W, no host failure, no
    capped block.
  - Machines 13828 (clock-bound) and 37367 (a 400 W cap in its clocks) stay excluded. So does 26157, until a create
    there succeeds.
  - Clean runs are reliability evidence. Lowering the bar raises the chance of a host-fault VOID; it does not bias the
    reading, because the on/off pair runs on one host.
- **Pinning.** As every lane does under the program's market rule:
  - one pin post names up to three bar machines, each present and passing in 6/6 machine-filtered reads;
  - the create walks them in order on `no_such_ask`;
  - spending stays within the standing rule (worst ≤ $15). The proof's worst is $1.07 at ≤ $0.85/h.
  - This replaces "Pinned in the ACK" under **Host** above, and the rental gate's ACK. There is no per-run ACK and no
    watcher envelope. Machine 58411 stays preferred when it is offered and meets the bar.
- **Run ids.** `sc2f-prove-11` onward. `-1` to `-4` are v4's refused attempts; `-5` to `-10` stay unused.

**Unchanged:**
- the science: the arms, both knobs, the ladder, the knee block, the statistics, the predictions, the outcomes and FLIPS
  (Amendment 1);
- the running of it: the box, the proof and its 1.0 h guard, the reading's guard and budget, and the $0 shakedown.

### Amendment 3 (registered before any SC2f run): the release under test, the kernels pin, the price ceiling, a new owner, and the order of three lanes

**Why.** SC2f was registered on e4b 0.54.0 and never read: no reading box was created, and the four proofs
(`sc2f-prove-1` to `-4`, 2026-10-18, $0) were refused at create. The lane changed owner on 2026-10-23 and now runs as
part of the serving roadmap, on the release after 0.56.1, so the release pin moves. **This changes the system under test.** The 0.54.0 pin was never read,
so nothing is lost and no reading crosses releases.

**The four refused proofs, read before this amendment.** Each refused for one cause: a ghost listing. Machine 26157,
the only bar machine listed, offered `54055665`, and the provider answered every create with `no_such_ask`, four times
in 66 seconds. Nothing was created and nothing was charged. Three fixes followed, and one cause is still live:
- Amendment 2 replaced the single ACK pin with a walk over up to three bar machines.
- The launcher now refuses an offer listed for under two minutes before it holds it. It also retries a ghost once, on the
  same pin, 180 s later, inside the same run id. Both live in the guarded launcher (`tools/pod-launch.sh`), so **every
  SC2f box launches through it**; a direct provider create gets neither.
- **Still live: the market half.** On 2026-10-23 at 00:03Z the bar, recomputed from the receipt store (clean runs ≥ 2,
  every recorded power limit 575 W, no host failure, no capped block), named 13 machines. After this lane's exclusions
  and the launcher's power exclusions, 9 remain. A read-only screen of six machine-filtered reads found **no candidate**:
  one bar machine was listed, above the price ceiling and power-excluded. So the rule below is: no candidate, no launch.

**The changes.**
- **e4b.** The launch commit is the first `main` commit that contains both the `v0.57.0` tag and the merge of e4b#2011
  (the paged server's single event-loop wakeup per step). Its `experts4bit_qlora/` tree, its diff against `v0.57.0`'s, and
  every commit between the two are recorded at launch.
  - **The default check is mechanical** (`bench/sc2/sc2_defaults.py`, `tests/test_sc2_defaults.py`). Before the
    create, the controller dumps the serving defaults twice, beside the same grouped-nf4-gemm: at `v0.57.0` (the
    reference) and at the launch commit. Each dump is taken on CPU, with every `E4B_*` / `GNF4_*` variable cleared, for
    `qwen3_moe`, in two parts:
    - **The resolved levers:** the server's config, the decode fusions, the prefill glue, the attention split cap, the
      int8 head at Qwen3-30B-A3B's shape, and the expert-path routes. This part catches resolved logic.
    - **A source scan:** every `E4B_*` / `GNF4_*` environment read in the package with its default literal, including
      reads through a module constant, and every such name the source mentions. This part catches inline defaults with
      no resolver and knobs added after the tag.

    `sc2_defaults.py diff` exits 1 naming every lever whose value differs or that exists in only one tree, and that
    refusal stops the launch. Two kinds of lever may pass, each by `--allow LEVER=reason`:
    - **A knob merged since the tag that is opt-in, defaults off, and is unset in this lane's arms.** Its reason reads
      "opt-in, default off, unset in this lane", checked against its PR and the lane's arm environment.
    - **Named here:** `config.prefill_every`, `config.loaded_chunk_tokens`, `env.E4B_PAGED_PREFILL_EVERY` and
      `env.E4B_PAGED_LOADED_CHUNK_TOKENS` (e4b#2017), should it have merged. They are opt-in, with defaults whose plans
      equal the earlier planner's call for call (tested), and unset in this lane.

    A changed default of a lever that both trees have is never allowed at launch; it needs a further amendment. The two
    dumps, the diff's output and every `--allow` with its reason are committed with the run.
  - The tripwire is unchanged (rc 9 without both of P130's defaults).
- **What moved under the arms, and what that does to the comparison.** Serving defaults that changed between 0.54.0 and
  0.57.0 on `qwen3_moe`:
  - grouped-nf4-gemm's single-request bandwidth GEMV (0.55.0);
  - K25's `prmt` decode, the int8 lm_head at one request, and the 16-split cap on the paged attention decode (0.56.0);
  - K25's `c1` plan (0.56.1);
  - #2011's single wakeup per step (the launch commit).

  This lane's server is SC2e's s64a on the **int4** expert store (K19), so the NF4 routes (the bandwidth GEMV, K25's decode
  and plan) are not on its expert path. The head, the attention cap and the dispatch change are. They apply to **both**
  arms alike, because neither is one of P130's two knobs. So `on` against `off` still reads exactly P130's two prefill
  defaults, now on a base whose decode and host time are smaller. The absolute ceilings are this release's, and no
  number compares with SC2e's (as registered: no ratio crosses lanes).
- **grouped-nf4-gemm.** v0.50.0 (`41ef086e6a3e2be5b158ff1bb38e0de67862a949`), the latest release, installed from git as on
  box L (`bench/sc1/sc1_run.sh`, boxes N and O). If a newer release ships before the launch, a further amendment moves it.
- **Price ceiling.** $1.05 an hour, the program's RTX 5090 policy rate on 2026-10-23, was $0.85. The ceiling is a spend
  rule, not a scientific term.
  - Worst cases at $1.05/h plus download at ≤ $0.011/GB: a proof, 1.0 h ≤ $1.05 + ~$0.10 = $1.15; the reading, 3.5 h ≤
    $3.68 + ~$0.73 = $4.41.
  - Lane total, as registered (two proofs, one reading, one re-run), with the re-run counted as a reading: 2 × $1.15 +
    2 × $4.41 = **$11.11** (was $7 at $0.75/h).
  - Each run sits under the standing single-run tier ($15).
- **Host continuity across three lanes.** SC2f, SC2h and SC2i (A1's interleave knobs) run in that order, on **separate
  boxes**, each its lane's registered box: N here, O for SC2h, and SC2i's own. SC2h then reads SC2f's committed verdict
  as registered.
  - The three pin the **same machine** while it stays listed and passes this lane's bar.
  - If it does not re-list, each lane's own bar decides. There is no substitution outside the bar, and no waiting on the
    earlier host past the screen deadline below.
  - Each box arms the independent deadline watcher at create.
  - The three lanes' running worst total is posted on the bus before the first create.
  - Each lane keeps its own verdict and its own strike count. A harness or infrastructure failure before a lane's first
    arm starts is not that lane's strike.
- **Screen deadline.** After the `v0.57.0` tag, a read-only screen may run at $0 for at most 48 hours. With no candidate
  at the end, the market data goes to the maintainer before any change to the bar is proposed. The bar itself is unchanged.
- **Run ids.** Unchanged from Amendment 2: `sc2f-prove-11` onward and `sc2f-5090-1` onward.

**Unchanged:** the arms, both knobs, FLIPS (Amendment 1), the bar and its exclusions (Amendment 2), the ladder, the knee
block, the statistics, the predictions, the outcomes, the box, the proof, the guards and the $0 shakedown.

### Amendment 4 (registered before any SC2f run): which commit is the launch commit

**Why.** Amendment 3 names the launch commit as "the first `main` commit that contains both the `v0.57.0` tag and the
merge of e4b#2011". Read literally, that is e4b#2011's merge (`75ddc16e`). That commit does not carry the lane's own
kit: box N's grouped-nf4-gemm v0.50.0 pin in `bench/sc1/sc1_run.sh` and `bench/sc2/sc2_defaults.py` both landed later,
in e4b#2023. The default check that Amendment 3 registers compares a launch commit against `v0.57.0` precisely because the
launch commit may sit after the tag. So the sentence is corrected before any SC2f run.

**The change.** The launch commit is a `main` commit at or after `1a25b209` (the merge of e4b#2027). That commit holds
`v0.57.0`, e4b#2011, e4b#2023's box-N kit and e4b#2027's `bench/sc1/sc1_run.sh`. Amendment 3's mechanical default check
runs at the launch commit against `v0.57.0`, with its two allowance rules, before the create. Its two dumps, the diff's
output and every `--allow` with its reason are recorded with the run, as are the launch commit, its `experts4bit_qlora/`
tree, its diff against `v0.57.0`'s tree, and every commit between the two.

**The kit is held at `1a25b209`.** Amendment 3's default check compares package defaults, not the kit, and
`bench/sc1/sc1_run.sh` is shared by every box. So at the launch commit, every `bench/sc1/staged.sha256` entry that box N
stages must be byte-identical to that entry at `1a25b209`: the lane's own pieces and the shared `sc1_run.sh` alike.
The driver refuses to stage a file whose bytes differ from its entry, so the staged kit is then `1a25b209`'s. The
entry-by-entry comparison is recorded with the run. Any difference is its own pre-run amendment.

**Unchanged:** every constant, threshold and cost; the arms, both knobs, FLIPS, the bar and its exclusions, the price
ceiling, the order of the three lanes, the screen deadline, the box, the proof, the guards and the $0 shakedown.


### Amendment 5 (registered before any SC2f run): the kit is held at e4b#2041's merge, for the harness's brotli pin and fetch retry

**Why.** The kit held at `1a25b209` (Amendment 4) installs `huggingface_hub` without a `brotli` floor. An unpinned
`huggingface_hub` on httpx decodes brotli with an API older brotli lacks, so a br-encoded file at the fetch raised
TypeError on some hosts. e4b#2041 fixes the shared harness, `bench/sc1/sc1_run.sh`, before any SC2f run. Amendment 4
makes any difference from the held kit its own pre-run amendment; this is that amendment.

**The kit is held at `7b8f1f69`** (e4b#2041's merge), replacing `1a25b209` for the KIT only. At the launch commit, every
`bench/sc1/staged.sha256` entry must be byte-identical to that entry at `7b8f1f69`, and `bench/sc1/sc1_drive.sh` and
`bench/sc1/make_pin.sh`, which decide what is staged, must be byte-identical too. The stack under test is unchanged:
the launch command sets `E4B_SHA=36cdd2d4e031e8365ab52e495c40cd79043edd49`, which `sc1_drive.sh` uses in place of its checkout's HEAD, so every box installs `36cdd2d4`. This supersedes the instruction
to check out exactly `36cdd2d4` for the KIT only: the kit's checkout is now a `main` commit at or after `7b8f1f69` whose kit
equals `7b8f1f69`'s (the HOLD gate below). If a later commit changes any staged entry, `7b8f1f69` itself is the checkout.

**What changed in the kit, against `1a25b209`.** Boxes N, O and P stage one flat kit, every entry on every box, so the
whole difference is declared:

- `sc1_run.sh` is the only existing entry whose bytes differ, and it differs in three ways. Two are behavioural for
  box N: the e4b install line adds `"brotli>=1.2.0"`, and `fetch()` retries a failed snapshot download once after
  20 s, appending to the same log, with the alarm re-armed from the time left, before it is a DL FAIL (rc 11, as before).
  The third is box Q's wiring (SC2j): Q added as an alternative to nine box lists, Q's dispatch line, and its
  `prove_q` line. With `SC1_BOX` = N, none of the Q wiring is reachable.
- Seven entries are new, all box Q's: `sc2j_box_q.sh`, `sc2j_contract.py`, `sc2j_prompts.py`, `sc2j_reduce.py`,
  `p117_reduce.py`, `test_decode_graph_buckets.py` and `g3_smoke.py`. They are staged on box N, but nothing box N runs
  sources or calls them. An inert file belongs to the kit hold, not to the stack under test.
- `staged.sha256` carries those changes; `sc1_drive.sh` and `make_pin.sh` change only to stage the seven files.

**The check, which fails closed** (`bench/sc2/sc2_kit_hold.py`, its self-test refusing each gate with its evidence removed;
`tests/test_sc2_kit_hold.py`). On the controller, before the create, `python bench/sc2/sc2_kit_hold.py check --at <launch commit>
--e4b-sha "$E4B_SHA" --out <run>/kit_hold.json`, run from a `main` checkout that carries this amendment, runs five
gates. DELTA: the difference `1a25b209` -> `7b8f1f69` is
exactly the set above: one changed entry, every `sc1_run.sh` hunk classified as the brotli line, a fetch-retry line, a
Q alternative (the line with its Q removed is its `1a25b209` line) or a box Q line, each declared hunk present once, and
the seven entries added. PIN: every entry's sha256 is its source at `7b8f1f69`. INERT: no staged entry other than
the seven names any of them or box Q's functions, except `sc1_run.sh` on box Q's own lines, and in `sc1_drive.sh` and
`make_pin.sh` the names appear only on staging lines. HOLD: the launch commit's entries and the two driver files equal
`7b8f1f69`'s. STACK: the launch's `E4B_SHA` is `36cdd2d4e031e8365ab52e495c40cd79043edd49` in full, so a launch that
omits it (and would install the checkout's HEAD) refuses. Any refusal stops the launch before the create. The record holds every hunk and every hit. At
`7b8f1f69` the check reads OK, with 16 classified `sc1_run.sh` lines (1 brotli, 4 fetch-retry, 9 Q alternatives, 2 box Q lines), the seven entries added, all 99 pins matching their sources, and 12 hits on the new names, each on one of box Q's three `sc1_run.sh` lines or a staging line of `sc1_drive.sh` or `make_pin.sh`.

**The stack is a VOID condition of the rule.** After the box, on the fetched run,
`python bench/sc2/sc2_kit_hold.py verdict-stack --run <run>/sc1 --out <run>/verdict_stack.json` reads the installed
evidence first: `versions.txt`'s `e4b <version> @<commit>` line, which the box writes only after its tripwire has
confirmed the install, must name `36cdd2d4`, and so must every `KNOBS ... e4b=` line of `summary.txt` (the launch's
`E4B_SHA`). Either record missing, abbreviated, or disagreeing with the other reads VOID. **A box verdict stands only
when verdict-stack reads OK; otherwise the read is VOID.** `verdict_stack.json` is a required results file: a results PR
or a re-derivation without it, or with it not OK, reads VOID. This also catches a hand launch that bypasses the check
above. The staged reducers are unchanged, so the kit stays `7b8f1f69`'s. Its self-test removes or alters each record,
including the two disagreeing either way, and `tests/test_sc2_kit_hold.py` runs it on a real fetched run's records.

**Unchanged:** every constant, threshold and cost; the arms, both knobs, FLIPS, the bar and its exclusions, the price
ceiling, the order of the three lanes, the screen deadline, the box, the proof, the guards and the $0 shakedown.
Amendment 3's default check still runs at the launch commit against `v0.57.0` and is recorded with the run.

### Amendment 6 (registered before any SC2f run): the HOLD gate's declared exceptions, in one place

**Why.** Amendment 5's HOLD gate requires the launch commit's staged kit, `sc1_drive.sh` and `make_pin.sh` to equal
`7b8f1f69`'s byte for byte. Since then, two changes landed on `main` that boxes N, O and P never run, so no later
`main` commit passed. bench/common rule 8 (a fetch never imports links) put `--no-links` into `sc1_drive.sh`'s fetch.
SC2j's own amendments changed `sc2j_box_q.sh` and added `sc2j_quality.py` to the flat kit and to the drivers'
staging lists. The only checkout that still passed was `7b8f1f69` itself, whose fetch keeps links.

**The change.** The HOLD gate (`bench/sc2/sc2_kit_hold.py`) keeps its byte identity with `7b8f1f69` and tolerates
exactly these exceptions. This list is the whole set; anything outside it refuses.

1. **Rule 8's fetch token.** `sc1_drive.sh` may carry one ` --no-links`, immediately after `rsync -az` on `pull_box`'s
   rsync line (`pull_box()` is the nearest function above that line). A second token, the token on any other line or
   at any other position, or any other changed byte refuses.
2. **Box Q's own files.**
   - An entry whose name matches `sc2j_<name>.(py|sh|json)` exactly may be changed or added.
   - Any other name refuses, including a new file nobody references: a staged `json.py` or `torch.py` would shadow a
     real module on every box, so the exact name is a safety condition. A removed entry refuses.
   - The staging lines of `sc1_drive.sh` and `make_pin.sh` may gain exactly the names of the sc2j entries that were
     added, and nothing else.
   - INERT runs again at the launch commit, over every other staged entry (the Python files included) and both
     drivers. No `sc2j_<name>` may appear outside box Q's own `sc1_run.sh` lines and the drivers' staging lines.

The checkout that stages must be the launch commit, clean. The check runs in that checkout (`--repo <launch
checkout>`) and refuses if its `git status --porcelain` is non-empty or its `HEAD` is not `--at`, however `--at` is
spelled. Fetch flags
and box Q's files change nothing that boxes N, O and P run, so the hold keeps its meaning. The stack is still
`36cdd2d4`, under the STACK gate and `verdict-stack`.

**Evidence.** The self-test refuses each exception beyond its letter (the new cases: a non-sc2j entry changed; an N/O/P
runner gaining `. $W/sc2j_extra.py`; a removed entry; the delta plus another driver byte; an unreferenced staged
`json.py`; a staging line gaining a non-sc2j name, or an sc2j name that is not an added entry; an sc2j name on a
non-staging driver line; a box Q file named outside the pattern, including names only a prefix or substring match
would pass, `sc2j_extra.py.bak` and `foo_sc2j_extra.py`; a dirty staging checkout at a full-sha `--at`; a staging
checkout whose `HEAD` is not `--at`). `tests/test_sc2_kit_hold.py`
checks the registered `main` commit `ac7e6ed1`, which carries exactly exceptions 1 and 2: HOLD reads OK, and
`hold_diff` is `sc2j_box_q.sh` and `sc2j_quality.py`.

**Unchanged:** everything else Amendment 5 registered; every constant, threshold and cost.

### Amendment 7 (registered before any SC2f run): qualify-at-proof replaces the receipt bar, the release pair moves to e4b 0.58.1 and grouped-nf4-gemm 0.51.0, and a new screen deadline

**Why.** Under the receipt bar (Amendment 2: clean runs ≥ 2, every recorded power limit 575 W, no host failure, no capped
block), the SC2f screen ran to its deadline (2026-10-25T05:05Z) and saw none of the bar's machines listed: 502 reads, the
fourth window in a row with no candidate. That screen logged only `offers_total` per read, so it cannot say which
machines were listed. The bar asked for history the market does not offer. Amendment 7 qualifies the card that is
actually rented, at its proof, and over every run; history only orders the list.

**The basis window, recounted (and its limit).** The SC2f screen logged only `offers_total`, so the new gate is
recomputed from the SC2j screens alone: 2026-10-23, 15:12Z to 22:35Z (7.4 h, 40 reads; the record is
`bench/sc2/sc2f-am7/sc2j_screen_reads.json`). The loops logged every under-ceiling offer from 20:43Z on; before that
they logged only the cheapest offer, so presence there is a lower bound. Prices are the launcher's hourly at 320 GB
(GPU plus storage).
- **Presence:** an offer listing 575 W under the $1.05/h ceiling in 16 of 40 reads (any under-ceiling offer in 21);
  52780 in 15; 145701 in 2; 55337, 36213 and 150147 in 1 each; 19317 and 46990 in 2 each.
- **Stability, the rule admission binds on** (an offer, by id, in 3 consecutive reads): 52780 in **6 of 40** reads
  (15 %); **every other machine in 0**. Without 52780 the window shows **no** admissible read: 0 %, not the 5 %
  presence of 145701.
- **Listed power** (the offers' `gpu_max_power`, an upper bound, never the gate): 52780, 145701 and 55337 list
  575 W; 19317, 36213 and 150147 list 450 W; 46990 lists 400 W. An earlier reading of this window took 55337's
  listed power as unknown; its one logged offer lists 575 W.

**(a) Qualify-at-proof** (`bench/sc2/sc2_qualify_drive.sh`, the launcher's `--command` for every SC2f box; the
admission and the checks in `bench/sc2/sc2_qualify.py`). It replaces the receipt bar of Amendment 2.

- **Admission** (`sc2_qualify.py admit`, which reads the receipt store; the code and this text are the same rule). The
  walk is **open**; prior clean runs only order it. In order:
  1. host continuity (Amendment 3): SC2h, then SC2i, put the earlier lanes' PROVED host first;
  2. **52780, then 145701** (ruled first, from the recount);
  3. **any other machine whose offers list 575 W**, by prior clean runs descending (the launcher's receipt ranking,
     passed as `--clean-runs`; a machine absent from it has 0), then machine id;
  4. machines whose listed power is unknown, in the same order, **last**.

  A machine whose offers list any other power is never admitted. Excluded throughout: 13828 (clock-bound) and 37367
  (a 400 W cap in its clocks); 26157 until a create there succeeds. The launcher's own power exclusions apply at
  create; a refusal before a create costs nothing and consumes no attempt (X2).
- **The listing filters**: an offer on the machine at or under **$1.05/h, where the price is the launcher's
  `vast_provider.hourly(offer)` at 320 GB disk (GPU plus the ordered storage, the figure `--usd-per-hour` binds;
  download is billed apart and counted in the worst case)**, with at least 320 GB of disk, present by offer id in
  each of the last 3 consecutive reads, re-read with an exact filter seconds before the launch. The screen writes that
  figure as `usd_h` and the offers' `gpu_max_power` as `listed_power_w` in each read it hands to `admit`.
- **Exit codes, three ways.** A record that was READ and failed is a host fault: rc 17 (power) or 18 (host floor),
  the launcher's host-limited codes, so it excludes the machine from its next draws; never a strike. A record that is
  missing or unreadable is probe_error, a **harness** fault: rc 25, which voids the run and never excludes the machine.
  A failed setup of the burn (apt, venv, the torch download) is rc 9, a harness fault too.
- **On the card, before any install:**
  - `nvidia-smi` `enforced.power.limit` must be 575 W, parsed to a float and compared numerically (`575.00 W` passes).
    Read at another limit (`574.99 W`), or two GPUs: rc 17. Unreadable (`[N/A]`, no row): rc 25. `power.limit` and
    `power.max_limit` are recorded beside it;
  - `bench/common/card_clock.py burn --floor 2700`: a busy SM median ≥ 2700 MHz, no hard reason bit on any sample,
    and SwPowerCap clear on ≥ 95 % of busy samples. Its own refusal field decides: `below_floor` is rc 18,
    `probe_error` is rc 25. **The burn's torch** is torch 2.8.0+cu128, a pip download into a throwaway venv (the image
    ships no torch): about 3 GB and a few minutes, counted in each attempt's cost below. It is not the arms' torch,
    which `sc1_run.sh` installs itself, so the burn qualifies the **card** only.
- **Over every run, proof and reading: "no capped block"** (as ruled 2026-10-25). Caps can arise mid-run, from heat or
  a host power policy. So a controller-side sampler, outside the staged kit (the HOLD is untouched), starts on the
  box before `sc1_drive.sh`: `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,power.draw
  --format=csv -lms 1000`. It is stopped and fetched after the run. After a clean run (the driver exits 0) it is
  judged by `card_clock.py`'s rule over the whole timed run: no hard reason bit on any sample, idle ones included, and
  SwPowerCap clear on ≥ 95 % of the **busy** samples (SM above card_clock's 1000 MHz), so idle install and download
  time cannot dilute the share; the busy SM median is recorded, not ruled. Read and failed: rc 18. Missing or
  unreadable (no header, fewer than 10 readable rows, more than 5 % unreadable, or fewer than 10 busy samples, a
  sampler that never saw the load): rc 25. An SC1 box's `box.json` carries no per-block timestamps, so there is no
  per-block judgment.
- **After a clean proof**, also the proof's own host records, both always passed: `forensics.txt`'s `power.limit` and
  the launcher's `host_facts.json` `gpu_power_limit_w` must each read 575 W. A record read at another limit is rc 18;
  a record missing, unreadable, or without its field is rc 25. (The launcher writes `host_facts.json` right after the
  pre-flight and never fails a run for it, so an unread probe there voids a clean proof as a harness fault.)
- **A run that did not finish clean** keeps its own exit code: the lane's rules classify it, and its records are
  fetched and judged for the record only, never read as a host fault.
- **A reading refused** by any of these is a host fault, not a strike; it stops the lane's rentals and goes to the
  maintainer.
- Every record lands in `<run>/qualify/`, fetched by `rsync --no-links` (bench/common rule 8) before teardown, and is
  committed with the run.

**(b) Attempts and cost.**
- **Every create consumes one of 3 proof attempts** (`sc2f-prove-11`, `-12`, `-13`), whatever its exit. That includes
  a qualify refusal, a harness rc 25, and a failed burn setup (rc 9): a flaky package mirror can burn attempts, each
  costing well under the bound below. The walk never revisits a host it has created on, unless that host's proof
  PROVED: it is then the lane's host, and only it is admitted for the reading and any re-run. Three creates and no
  proof PROVED: the lane **parks**.
- Worst case, at $1.05/h plus downloads at ≤ $0.011/GB:
  - a proof attempt (guard 1.0 h): $1.05 + ~$0.10 (Granite) + ~$0.03 (the burn's torch) = **$1.18**. A refusal or
    a setup failure costs less: the create, the burn and teardown, about 15 minutes, ~$0.30;
  - the reading (guard 3.5 h): $3.68 + ~$0.73 + ~$0.03 = **$4.44**;
  - the lane: 3 × $1.18 + 2 × $4.44 (the reading and one re-run) = **$12.42**. Each run is under the standing $15
    single-run tier. The worst is posted on the bus before the first create.

**(c) The screen and its deadline.**
- A read-only screen applies the admission above at $0.
- **The deadline is 72 h**, counted from Amendment 7's merge or from the G3 GPU check's PASS (below), whichever is
  later.
- Sized on the stability recount. In the basis window only 52780 ever passed (6 of 40 reads); without it, no read
  did. So the window gives no evidence that any other machine passes within 72 h. The open walk is what can change
  that: any new 575 W listing is admissible. The deadline is a time box, and parking is a likely outcome if 52780
  does not relist stably.
- If no proof qualifies a host by the deadline, SC2f, SC2h and SC2i **park**, with no further change to the bar. That
  window's market data is the record.

**(d) The release pair moves: e4b 0.58.1 and grouped-nf4-gemm 0.51.0.** This changes the system under test. Nothing
was read on the old pair, so nothing is lost and no reading crosses releases.
- **e4b.** The stack under test is `v0.58.1`'s commit, `605d21e67958fa94d5e73fc3a95ffcb5c8eae02d`. The launch command
  sets `E4B_SHA` to it in full; `sc2_kit_hold.py`'s STACK gate and `verdict-stack` now name it (was `36cdd2d4`).
  `v0.58.1` contains e4b#2087 (the G3 prefill-headroom change).
- **grouped-nf4-gemm.** `v0.51.0`, commit `30adb1bfa76f95f9ce4a4c29734b8f36de907eca` (tag object `245d8bd4`), e4b
  CI's pin, installed from git as before. `sc1_run.sh` moves boxes N, O and P to it. **Box Q stays at `v0.50.0`**: it
  is SC2j's registered pin, SC2j stopped at two strikes on that stack, and a re-read of SC2j is its own registration.
  Moving Q here would change a stopped lane's kit with no lane to read it.
- **What moved under the arms** (36cdd2d4 → 605d21e6, from `bench/sc2/sc2_defaults.py` CPU dumps of both trees beside
  the same grouped-nf4-gemm 0.51.0, for `qwen3_moe`; recorded in `bench/sc2/sc2f-am7/`, with each arm's dump: `on`
  with nothing set, `off` with `--set E4B_FUSE_PREFILL_GLUE=0 --set E4B_PREFILL_LEAN_DISPATCH=0`):
  - `config.int8_head_rows`: `1` → `default`. P232's K2b default: the int8 lm_head's 2..8-row calls go to K37 on
    qwen3_moe at Qwen3-30B-A3B's head shape, when the head is on and speculation is off. Not one of P130's knobs; **it
    applies to both arms alike**;
  - `E4B_NF4_SMALLM_BW`: the read's `'0'` default literal is gone. Lever B's per-family default covers Mixtral only, so
    on qwen3_moe it stays off. The NF4 collapse is not on this lane's int4 expert path anyway. Not one of P130's knobs;
    **both arms alike**;
  - new levers at 0.58.1, each off or empty by default and unset in both arms: `attn_split_fixed` (None),
    `merged_arena` (False, `E4B_PAGED_MERGED_ARENA` `'0'`), `E4B_DENSE_LT_PIN` (K1, opt-in), `E4B_TRAIN_DETERMINISTIC`
    and `E4B_TRAIN_DET_GATHER` (training only), and `kv_pool_resolution` / `kv_pool_rows` (G3's sizing record; only
    under `E4B_PAGED_KV_POOL=shared`, while the default and both arms are claim-all). **Both arms alike**;
  - **P130's two knobs: neither default moved.** The `on`/`off` contrast is identical in both trees and is exactly
    the two knobs: `prefill_glue` `1` (default-allowlisted) against `0` (explicit), and `_prefill_lean_dispatch_env`
    `auto` against `0`. So `on` against `off` still reads P130's two prefill defaults, as registered.
  - **e4b#2087 moves neither arm's path.** The runtime prefill-graph gate's arithmetic is unchanged: free memory after
    capture ≥ the graph's pool plus the bulk flush, now computed by `serve_recipe.prefill_graph_headroom_bytes`. The
    sizing change applies only under `E4B_PAGED_KV_POOL=shared`, and box N's servers are claim-all at an explicit 64
    slots.
- **The at-launch mechanical check** (Amendment 3, kept). Before the create, the controller dumps the defaults of
  the stack the box installs (a checkout of `605d21e6`) and of `v0.58.1`, beside grouped-nf4-gemm 0.51.0. Both
  trees are the tag's commit, so the diff must be empty: no `--allow` is registered. Both dumps and the diff are
  committed with the run. The kit checkout (the launch commit) may be a later `main` commit, but its package tree
  is never installed (`E4B_SHA` is explicit), so it is not the subject of this check. Its tree, and its diff against
  `v0.58.1`'s, are recorded with the run, as Amendment 4 registered.
- **The kit hold: HOLD exception 3.** HOLD keeps its byte identity with `7b8f1f69`, plus Amendment 6's two
  exceptions, plus exactly one more: `sc1_run.sh` carries the two declared grouped-nf4-gemm lines (boxes N and O, and
  box P apart from box Q), each once, and nothing else in it differs. This exception is **required**, not just
  tolerated: a launch commit whose `sc1_run.sh` still equals the hold's would install v0.50.0, so it refuses (the
  registered commits before this amendment now refuse for that alone). At this amendment's tree the check reads OK.
  `hold_diff` is `sc1_run.sh`, `sc2j_box_q.sh` and `sc2j_quality.py`.
- The tripwire is unchanged: rc 9 unless both of P130's defaults hold.

**(e) Order, and the other two lanes.**
- **The G3 GPU check (e4b#2133) must PASS before SC2f's first proof creates.** SC2f now reads at a release that
  contains #2087, so that check's PASS is the proof that the default rests on: the shared server's prefill graph
  captured and replayed at 64 slots on Qwen3 int4, read from the server's own state. A G3 FAIL holds SC2f, SC2h and SC2i
  until #2087's fix is re-proven, and that time does not count toward the 72 h deadline.
- SC2h (Amendment 5) and SC2i (Amendment 4) take this gate and this release pair. SC2h reads SC2f's committed verdict,
  so the three lanes sit on the **same** pair (e4b 0.58.1, grouped-nf4-gemm 0.51.0). If one lane later moves alone, its
  cross-lane read is VOID.

**Evidence.**
- `sc2_qualify.py --self-test` (51 cases) covers:
  - the open walk's tiers, a machine that appears mid-screen (admitted on its third read, not before), the order by
    clean runs then id, unknown listed power last, another listed power never admitted;
  - the exclusions, attempt accounting, PARK, continuity, and stability by offer id;
  - the power parse's three ways, the post-proof judge's three ways (missing `host_facts.json` is UNREAD, 450 W is
    REFUSE, 575 W is PASS), and the run sampler's (a hard bit, on a busy or an idle sample; SwPowerCap at 5 % and
    6 % of busy samples; half the run idle with SwPowerCap on 10 % of busy samples; GpuIdle; fewer than 10 busy
    samples; missing, no header, too few rows, too many unreadable rows).
- `tests/test_sc2f_am7.py` adds:
  - the admission from a fixture receipt store;
  - the post-proof judge on sc2j-prove-2's own `host_facts.json`;
  - the wrapper on a stubbed box, proofs and readings alike, for each exit code;
  - the stability recount from the window's record;
  - the default-dump records' stated changes.
- `sc2_kit_hold.py --self-test` (70 cases) refuses each violation of exception 3: the move absent, only one line
  moved, box Q moved too, another byte, a line twice, and the old stack.
- `sc2_defaults.py --self-test` (12 cases) covers `--set`.

**Unchanged:** the arms, both knobs, FLIPS (Amendment 1), the ladder, the knee block, the statistics, the predictions,
the outcomes, the box, the proof and reading guards, the $0 shakedown, Amendment 5's DELTA, PIN and INERT gates and
`verdict-stack` (now naming `605d21e6`), and Amendment 6's two exceptions. Run ids: `sc2f-prove-11` to `-13` for the
three attempts, and `sc2f-5090-1` onward.
