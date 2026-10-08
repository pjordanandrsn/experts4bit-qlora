# P121 — K25 on Qwen3-30B-A3B at the served W16 step: does `E4B_NF4_GROUPED_SMALLM=auto` (the default since P96) cost quality or speed against the NF4 M-tile (`0`) at 128 routed rows? One RTX 5090 (registered 2026-10-08, before any run)

Issue: experts4bit-qlora#846 (the serving campaign; the owner's no-ask tier for a single run under $15). Lane number
claimed by `prereg/p121` (pushed 2026-10-08T19:40:52Z). Asked for by the maintainer from FAM0's inventory (#1368,
`bench/fam/INVENTORY-fam.md`). The instrument is the maintainer's choice (bus, 2026-10-08T19:29Z, option A), and so is
the continuity row (19:53Z, option ii). Box, runner and driver are derived from P115 Phase D's (#1354).

## Why this lane

- **K25 has no family or shape gate.** Under `auto` it takes every device-grouped NF4 decode step with T > 1 and at
  most 256 routed rows (`hot_residency.py`: `R_rows <= 256 and _k25_mode != "0"`). On Qwen3-30B-A3B NF4 that includes
  the default W16 step: 16 rows × top-8 = 128 routed rows.
- **Its licence read other families only.** P93 read speed (Granite's B = 16 step 0.594×, OLMoE's 0.598×) and P96
  read quality (windowed K8, LICENSED), both on Granite-3.1-3B-A800M and OLMoE-1B-7B. P93's first quality read failed
  OLMoE's c4val1 (+0.155 ppl) before P96's windowed gate passed it. So quality on a new family is a live question.
- **Every Qwen3 W16 number since 2026-10-02 ran K25 unread.** FAM0 lists it as the only unread cell in the
  `qwen3_moe` serving row.

**Why not P96's instrument.** P96's windowed K8 decodes one token per forward (T == 1), and `auto` never sends
T == 1 to K25: `k25_t1` needs the env at `1`. Under P96's instrument, `auto` against `0` would score identical
arithmetic. P96 itself forced `1`. The claim to settle is the served W16 condition.

**The question:** on the default server at 16 rows a step, does K25 move the teacher-forced NLL beyond P110's floor,
and is the served W16 decode faster with it?

## Instrument

**The server** is the shipped default (`PagedServeConfig.from_env()` + `build_engine`):
- Qwen3-30B-A3B @ `ad44e77`, its NF4 arena baked on the box by P39's `k8_bake.py`;
- `E4B_PAGED_MAX_SEQS=16` named, buckets 1–16, decode graphs on;
- the fusion knobs and the decode GEMV at their defaults.

The arms differ only in `E4B_NF4_GROUPED_SMALLM`, named explicitly in each:

| arm | value | the decode rows at <= 256 routed rows |
|---|---|---|
| K0 | `0` | the NF4 M-tile (`nf4_grouped.gemm_4bit_grouped_captured`, TF32 on the fp32 dequant) |
| K1 | `auto` | K25 (`nf4_smallm.gemm_nf4_grouped_smallm`), with K23's lean glue, as served |

**Route counter.** `RouteCounter` wraps both kernels in their modules before `build_engine`; e4b reads them from the
modules at call time. Every record carries:
- K25 calls;
- M-tile calls at ≤ 256 routed rows and at > 256 (prefill chunks, M-tile in both arms).

**Speed** (`p121_box.py --mode speed`; arms `K0a`, `K1a`, `K1b`, `K0b` in that order, each a fresh process):
- P109's workloads at its registered bytes: W16 (16 prompts at once) and W1 (row 0 alone);
- 32 and 160 new tokens, 1 warm and 3 timed passes each; decode tok/s from the slope;
- every timed pass's token digest, and the last pass's tokens;
- the route tally after the build. The bucket graphs capture their kernels there; replays never reach Python, so the
  tally is read at capture (P115 Phase D's lesson, Amendment 4).

W1 is a null control: T == 1 runs the decode GEMV in both arms, so K1's W1 tokens must equal K0's.

**Quality** (`p121_box.py --mode quality`; P115 Phase B's instrument, `p115_quality.measure_phase`, at its registered
bytes):
- the default server built eager, as P110 built it;
- **16 windows a pass**: every decode step is 16 rows (128 routed rows), with device grouping, the padded bucketed path
  and lean glue as served;
- wikitext-2 and c4val1, 48 windows each at k × 4096 tokens, 512 prompt tokens and 128 positions.

| phase | arm | scores |
|---|---|---|
| OFF | K0 | R; the floor `rep` (R again, first group), `half` (two 8-row halves) and `chunk` (prefill in 256-token pieces); `mutant_scale` (P108's halved decode scale). R's fp32 log-probs are saved |
| ON | K1 | ON against R's saved log-probs |

**Continuity** (reported, no bar): P96's two arms through the same instrument at one window a pass (T == 1, bucket 1)
on 12 wikitext windows:
- Cm, `0` with `E4B_NF4_T1_DEVICE_GROUPING=1` (the M-tile at T == 1), scores R;
- Ct, `1` with T == 1 device grouping `0` (K25 at T == 1), scores ON.

It reports K25's arithmetic at the shape P96 read, in nats and perplexity, not in K8's windows.

## The rule (`bench/p121/p121_reduce.py`, self-tested on 35 cases)

The first that applies is the verdict.

1. **VOID**, any of:
   - a record is missing; another e4b or grouped-nf4-gemm commit or model revision; the arms read different prompts or
     ran different lengths; a decode slope is void;
   - **speed**: a bucket not captured; `max_seqs` or the buckets not 16 / 1–16; a fusion census not zero; the captures
     took the wrong kernel (K1: K25 and no M-tile call at ≤ 256 rows; K0: the M-tile and no K25); **K1's W1 tokens differ
     from K0's**;
   - **quality**: the group not 16; a text short of its windows; the phases scored different windows; a pass with the
     wrong decode attention calls, grouping flag or bucket statistics (every decode step an eager padded step, no
     replays); a pass that took the wrong kernel (K0 passes: the M-tile at 2 × layers per decode step, no K25; ON: K25 at
     2 × layers per decode step, no M-tile call at ≤ 256 rows); `mutant_scale` passing the bar on either text; ON
     bit-equal to R in every window of both texts.
2. **NOISY**: a self-pair (K0b/K0a or K1b/K1a, decode tok/s, either workload) outside [0.96, 1.04].
3. **FUNCTION_FAIL**: K0b's tokens differ from K0a's, or K1b's from K1a's, on any row of either workload at either
   length, or an arm's timed reps do not all digest the same. K1 differing from K0 at W16 is expected and reported.
4. **QUALITY_FAIL**: on either text, ON fails P110's bar against the floor:
   - mean d_ON ≤ B_floor + 0.01 nats, and
   - mean |d_ON| ≤ 2 × max(S_floor, 0.005).

   d_X(w) is X's mean continuation NLL minus R's. The floor draws are `half` and `chunk`, plus `rep` if R did not repeat
   bit for bit. B_floor is the largest |mean d_f|, S_floor the largest mean |d_f|.
5. **SLOWER**: g16 = min(K1a/K0a, K1b/K0b), decode tok/s on W16, below **1.00**.
6. **LICENSED** otherwise.

The license rests on the main read only: P110's bar and g16 (the maintainer's ruling).

**Reported beside the verdict:**
- the pair ratios on both workloads and their geometric means; ms per step;
- K1-against-K0 token agreement;
- every quality arm's bias, spread, SE, max |d|, KL and argmax agreement per text;
- each text's perplexity move (exp of the mean NLL over its positions). This is the number P115 gated on wikitext at
  0.05; here it gates nothing;
- the continuity row;
- the route tallies.

Floats are summed with `math.fsum`, so the verdict file is byte-identical on any Python.

## Predictions (written before any data; evaluated by the reducer, none gates the verdict)

| # | prediction | if it holds | if it misses |
|---|---|---|---|
| Q1 | g16 in [1.10, 1.70]. Basis: P93's 0.594× / 0.598× B = 16 steps on Granite and OLMoE; Qwen3's wider experts and heavier attention dilute the gain | K25 buys W16 speed on Qwen3 at about P93's scale | below 1.10 but ≥ 1.00: licensed, the gain small. Above 1.70: the M-tile's TF32 path is worse on Qwen3's shapes than on P93's |
| Q2 | W1 pair ratios in [0.98, 1.02] | T == 1 is untouched, as the code says | something besides K25 differs between the arms; the W1 token gate says whether it changes arithmetic |
| Q3 | self-pairs in [0.98, 1.02] | the 4 % noise bound is generous | the box is noisier than P111 / P115's |
| Q4 | ON's bias on wikitext in [−0.004, +0.004] nats and on c4val1 in [−0.006, +0.006]; wikitext spread ≤ 0.015; wikitext ppl move within ±0.03 | K25's TF32 MMA on the NF4 select tree is as neutral on Qwen3 as on P96's families | the bar decides; a miss that still passes says the floor carries it |
| Q5 | `mutant_scale` bias ≥ +0.3 nats on both texts | the gate can fail | the instrument has lost sensitivity; RESULTS says so before any claim |

## Consequence, registered now

- **LICENSED:** K25's `auto` default is read on Qwen3-30B-A3B at the served W16 step. The read PR adds a claims row
  (quality and g16), and STATUS's K25 row cites it. No code change.
- **QUALITY_FAIL or SLOWER:** a separate e4b PR scopes `auto` away from `qwen3_moe`, keeping `=1` as the way to force
  K25. STATUS and the claims row say so.
- **NOISY or FUNCTION_FAIL:** no change; a rerun is an amendment.
- **VOID or NO_READING:** no consequence.

## The proving rental

No local card runs the fp8 paged KV, the bucket graphs or grouped-nf4-gemm's kernels. `tests/test_p121_box.py` covers on
CPU:
- the counter's wiring, through the lookups `hot_residency` makes;
- the arms' refusals;
- the instrument's per-pass route records at 16 windows a pass.

**The proof** runs the whole box end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @ `a02780686e08a03fe0d2679a293b5c74a90efa89`
(NF4; 40 experts, top-8, so its W16 step is 128 routed rows too). It runs at 8 / 24 tokens, 1 rep, and 16 windows of
32 positions per text. It covers:
- the refusals, the install and the tripwire;
- the self-tests and the premise;
- fetch, bake, the four arms, both quality phases, the continuity row and the reducer.

It is **PROVED** iff the lane exits 0 with a verdict other than VOID. Its numbers are not a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 2.0 h at ≤ $0.75/h, estimate $1.50. Expected 45–70 minutes:
  - the 61 GB fetch (4–28 min on recent hosts);
  - four speed arms at about 3 min each;
  - quality OFF about 12 min, ON about 4 min;
  - the continuity row about 8 min.
- **Lane ceiling:** $3.00, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals run before any install: CUDA unusable (18), dud box (10), card class (15), disk < 150 GB
  (13), host RAM < 60 GiB (16). The premise runs before the fetch (25): `tests/test_decode_graph_buckets.py` and
  `tests/test_k25_row_exact_gpu.py` on the card, **11 passed**, none skipped.
- **STOP-2:** every time-left check (fetch, bake, each arm, each quality and continuity phase) fits inside its guard,
  enforced by `tests/test_p121_staged_pin.py`.
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p121/staged.sha256`.

grouped-nf4-gemm is pinned at **`56f90e3`** (0.43.0 main at registration; K25 since #429).

## What this lane cannot say

- **Other row counts.** Only 128 routed rows are read. Today's 64-slot default (#1334, #1346) also sends bucket 32
  (256 rows) through K25, and W2–W8 take it too.
- **Other models, cards or prompt lengths.**
- **T == 1.** The continuity row reports K25's arithmetic there with no bar; `auto` never runs it.
- **Serving attainment.** The step ratio is an input to `serve_capacity`, not a ceiling.

## Receipts

Fetched to the run directory's `p121/` and committed to `bench/p121/receipts/<run>/`:
- `arm_K0a.json`, `arm_K1a.json`, `arm_K1b.json`, `arm_K0b.json`;
- `quality_off.json`, `quality_on.json`, `continuity_off.json`, `continuity_on.json`;
- `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the logs (force-added past the `*.log` ignore) and the teardown proof;
- `SHA256SUMS`.

The reference log-probs stay on the box. `RESULTS-p121.md` is written from those files.
