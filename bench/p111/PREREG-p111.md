# P111 — does one KV-table selection per decode step (`E4B_KV_STEP_SELECT=1`) decode the default `serve_paged` server's tokens exactly, and faster? On one RTX 5090 (registered 2026-10-04, before any run)

Lane number claimed by `prereg/p111` (pushed 2026-10-04T00:00:30Z). Follows SC1b's read (#993) and the switch (#999).

## Why this lane

**What SC1b found.** Its census (`bench/h2h-2026-10-02/sc1b/README.md`) put about 0.8 ms of e4b's B=16 decode step on an
RTX 5090 in `fp8_paged_kv.py`. On Qwen3-30B-A3B, every decode step makes:
- 97 `index_select`, re-selecting the active set's block-table and seq-lens rows per layer;
- 48 `seq_lens.index_add_`, one per layer.

The active set is fixed for the whole step. SC1b named "one active-set selection per step" as a lever, untested.

**What #999 changed.** It ships that lever behind `E4B_KV_STEP_SELECT=1`, opt-in. With a decode-graph bucket bound:
- every layer's rows are selected in one launch each, outside the graph;
- each layer's attention reads its slice;
- the lengths advance once after the step.

The kernels see the same values, so the tokens must be identical. On a tiny Qwen3 they are
(`tests/test_kv_step_select.py`, on sm_89+). Neither the speed nor the identity has been read on the served model.

## Subject

The shipped default server: `PagedServeConfig.from_env()` + `build_engine`, with only the model, arena and calibration
set. Since 0.43.0 that server captures bucketed decode graphs (`E4B_PAGED_GRAPHS=auto`).
- **Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, NF4 arena baked on the box by P39's
  `k8_bake.py` (P109's subject).
- **Stack:** e4b at the launch commit, which carries #999 and this registration; grouped-nf4-gemm v0.35.0 (`51a4916`);
  transformers 5.17.0.

## Arms and workloads

Four arms, in ABBA order, each a fresh process. `bench/p111/p111_box.py` runs each one, importing P109's prompts,
passes and slope at their registered bytes:

| arm | `E4B_KV_STEP_SELECT` |
|---|---|
| S0a, S0b | `0`: the per-layer selection (today's default) |
| S1a, S1b | `1`: one selection per step |

**Workloads** are P109's:
- W16 is 16 distinct 512-token wikitext prompts at once; W1 is row 0 alone;
- 32 and 160 new tokens, one warm pass and 3 timed passes;
- the decode throughput is p37's slope.

**Recorded per arm:** every pass's token digest; the last pass's tokens per row; `graph_status`; the switch as the KV pool
read it; `graph_stats`; memory.

## The rule (`bench/p111/p111_reduce.py`, self-tested on 11 cases)

The verdict is the first of these that applies.

1. **VOID.** Any of:
   - an arm is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - different prompts or lengths across arms;
   - a void slope;
   - wrong engagement: every arm must report `graph` for every bucket (1–16), and its `step_select` must match its arm.
2. **NOISY.** A self-pair, S0b/S0a or S1b/S1a decode tok/s on either workload, falls outside [0.96, 1.04].
3. **FUNCTION_FAIL.** Any of:
   - S1a or S1b emits a token different from S0a's on any row of either workload at either length;
   - S0b differs from S0a;
   - an arm's timed reps do not all digest the same.
4. **SLOWER.** g16 = min(S1a/S0a, S1b/S0b) on W16 is below 1.00, or g1, the same on W1, is below 1.00.
5. **DEFAULT_ON** otherwise.

**Reported:** both pair ratios per workload, their geometric means, and ms per step per arm.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | every arm captures every bucket, and S1's switch is in force |
| Q2 | S1 ≡ S0 bitwise on every row of both workloads at both lengths |
| Q3 | g16 lies between 1.02 and 1.06. SC1b's 0.8 ms on P109's 21.4–21.9 ms NF4 B=16 step is about 3.7 %, less the four launches the step adds |
| Q4 | g1 lies between 1.02 and 1.10: the per-layer selects run at bucket 1 too, against a 10.1 ms step |
| Q5 | the self-pairs read within [0.98, 1.02] (P109's G pairs read 1.023 and 1.001) |
| Q6 | the verdict is DEFAULT_ON |

## Consequence, registered now

- **DEFAULT_ON:**
  - `E4B_KV_STEP_SELECT` defaults on; `0` keeps the per-layer selection;
  - `docs/SERVING.md` and the CHANGELOG say so, with the gain;
  - a register row `e4b.serve.p111.kv-step-select.qwen3.5090.<date>` carries g16, from the verdict file.
  - **Scope:** read on Qwen3-30B-A3B NF4 through the graph default. The selection is the same code on every family; the
    docs say it was read on one.
- **SLOWER:** the switch stays opt-in, and the ratios are recorded.
- **FUNCTION_FAIL:** the switch stays opt-in, and the token difference is a bug to find before anything moves.
- **NOISY or VOID:** no consequence; a rerun or an amendment.

## The proving rental

The whole box runs end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89` at 8 / 24 tokens and 1 rep:
- refusals, install and tripwire, both self-tests;
- the premise: `test_decode_graph_buckets.py` + `test_kv_step_select.py`, **13 passed**, none skipped, on the card;
- fetch, bake, prompts, the four arms and the reducer.

It is **PROVED** iff the lane exits 0 with a verdict other than VOID. The proof's verdict is not a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 1.25 h at ≤ $0.75/h, estimate $0.9375. The expected time is about 25 minutes: fetch
  6, bake 2, four graph arms of about 2.5 minutes each.
- **Lane ceiling:** $2.50, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals run before any install: dud box (10), card class (15), disk < 150 GB (13), host RAM < 60 GiB
  (16), premise (25).
- **STOP-2:** every time-left check is a per-mode variable inside its own guard, enforced by
  `tests/test_p111_staged_pin.py`.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p111/staged.sha256`.

## What this lane cannot say

- Nothing about the int4 stack, where SC1b measured the 0.8 ms on a shorter step, or other families.
- Nothing about eager decode, where the switch does nothing.
- Nothing about the B=1 overlap lever SC1b also named (programmatic dependent launch).

## Receipts

Fetched to the run directory's `p111/` and committed to `bench/p111/receipts/<run>/`:
- `arm_{S0a,S1a,S1b,S0b}.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`,
  `bake.json`;
- the logs and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p111.md` is written from those files.
