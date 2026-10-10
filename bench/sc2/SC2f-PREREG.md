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
  | `on` (the release's defaults) | unset | unset | `levers.prefill_glue` = {mode 1, source default-allowlisted}; both raw env values null; `prefill_lean_dispatch` auto; every K19 call above 256 rows `chained\|lean`; the norm, layer and attention folds each seen on prefill rows |
  | `off` | 0 | 0 | {mode 0, source explicit}; both raw env values "0"; `prefill_lean_dispatch` 0; every K19 call above 256 rows `chained\|gather`; no prefill fold |

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

## The rule (`bench/sc2/sc2f_reduce.py`, 27 self-test cases)

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

**Outcome**, decoupled from Q1–Q6:
- **VOID** if a server fails a gate; **UNREAD** if a number the outcome needs is missing.
- **REGRESSED** if `on`'s ceiling is below `off`'s; or, at 4 or 8 req/s, an `on` server attains less than the draw's lower
  `off` server minus 0.05; or gain ≤ −bar in both draws. Consequence: to the maintainer, since the release then ships a
  serving regression. `E4B_FUSE_PREFILL_GLUE=0` and `E4B_PREFILL_LEAN_DISPATCH=0` restore the old path; whether to
  revert a default is the maintainer's call.
- **FASTER** if gain ≥ bar in both draws. Consequence: **the claim moves.** A serving claim row for the P130 defaults
  records the knee gain and the paired TTFT and TPOT ratios on this box. **The STATUS / README capacity number** moves to
  `on`'s ceiling iff that ceiling is above SC2e's 12 (`readme_number`), stated with its host; the claim attributes to the
  flips only the paired difference against `off`. Otherwise the number stays 12 and the claim states the knee gain.
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

None yet.
