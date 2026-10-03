# P109 — should `serve_paged` capture decode graphs by default? Its eager default against bucketed graphs, through the server's own construction, on one RTX 5090 (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#770. Lane number claimed by `prereg/p109` (pushed 2026-10-03T20:00:02Z).

## Why this lane

**Every e4b serving number in the register ran with decode graphs; the shipped server does not.**
- SC1's e4b arm built the engine through `serve_paged.build_engine` with `E4B_PAGED_GRAPHS=1`
  (`bench/sc1/sc1_run.sh`, `sched_env`). So did the serving reads since: P96 through the harness's graph loop, and
  P98, P99, P100 and P101 through the serving stack with `E4B_PAGED_GRAPHS=1`.
- `PagedServeConfig.graphs` defaults to `False`. A user who starts `python -m experts4bit_qlora.serve_paged` with the
  documented example gets eager decode, a configuration no registered number describes.

**What #770 left open, and what has since closed:**
- **Identity.** P82 (`bench/p82/RESULTS-p82.md`) read the graph replay bit-identical to the eager runner on the
  licensed int4 stack, *with the device grouping on in both*. B771b read the same on NF4. Neither read went through
  `build_engine`'s defaults, where an eager server (graphs off) keeps the library's HOST grouping.
- **Memory and capture time.** P81 measured about 0.05 GiB and about 5 s.
- **What stays open:** whether the server's graph path computes its own eager function at the default configuration,
  and how much faster it is there.

**Graphs are not only a decode change.** With graphs on and `max_seqs > 1`, `build_engine` switches on
`hot_residency.DEVICE_GROUPING` before it captures (`_batched_graph_grouping`). That routes every `T > 1` MoE call, the
prefill chunks included, through device grouping (`hot_residency._collapsed_grouping`). So a graph server is NOT
expected to emit the eager default's tokens bit for bit.
- **The function the replay must equal** is eager decode with that same grouping: arm D below.
- **E against G** is a sanity bar, registered here, plus a report.

## Subject

The default server, as `PagedServeConfig.from_env()` reads it with nothing set but the model, the arena and the
calibration:
- `max_seqs` 16, `all-vram`, buckets 1–16;
- chunk 512, 4,096 tokens per sequence;
- NF4 experts, no int4 levers, no fusions;
- every route at its default.

**Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, SC1's pin. This is the documented example
in `docs/SERVING.md`.

**Inputs:**
- the NF4 arena is baked on the box by P39's `k8_bake.py`, as SC1 bakes it;
- the host calibration is P39's `calib.json`. Both are staged at SC1's bytes.

**Stack:**
- e4b at the launch commit, the merge of this registration;
- grouped-nf4-gemm v0.35.0 (`51a4916`), e4b CI's pin;
- transformers 5.17.0.

## Arms and workloads

Five arms, in this order, each a fresh process. `bench/p109/p109_box.py` builds each one with
`PagedServeConfig.from_env()` + `build_engine(cfg)`; only the arm's own switch differs.

| arm | switch | decode | grouping |
|---|---|---|---|
| E1, E2 | none (today's default) | eager | host (the library default) |
| G1, G2 | `E4B_PAGED_GRAPHS=1` | bucketed CUDA graphs | device (set by `build_engine`) |
| D1 | `hot_residency.DEVICE_GROUPING` on, `FORCE_SINGLETON_GROUPS` off, before `build_engine` | eager | device |

Each arm runs two workloads on the same engine:
- **W16** — 16 distinct 512-token prompts added at once;
- **W1** — one request (row 0) on the same 16-slot server.

**Prompts:** row k is tokens `[k·4096, k·4096+512)` of wikitext-2-raw test, joined as P97's `wikitext_windows` joins
it. They are written once to `prompts.json`, and every arm checks its digest.

**Per workload and length** (SHORT 32, LONG 160 new tokens):
- one untimed warm pass, then 3 timed passes;
- every request runs to `max_new_tokens` (`stop_ids` None);
- decode throughput is p37's slope, `B·(LONG−SHORT) / (min wall_LONG − min wall_SHORT)`. The prefill is in both walls
  and cancels.

**Recorded per arm:**
- every timed pass's token digest;
- the last pass's tokens, per row, at both lengths;
- `graph_status`, the grouping flags at run time, load time, peak memory, `graph_stats`, `nvidia-smi`.

## The rule (`bench/p109/p109_reduce.py`, self-tested on 16 cases)

The verdict is the first of these that applies.

1. **VOID.** Any of:
   - an arm is missing or not ok;
   - a receipt names another e4b or grouped-nf4-gemm commit, or another model revision;
   - the arms read different prompts or ran different lengths;
   - a slope is void;
   - engagement is wrong. That means G does not report `graph` for every bucket with device grouping on, E reports
     graphs or device grouping, or D reports graphs or no device grouping.
2. **NOISY.** A self-pair, E2/E1 or G2/G1 decode tok/s on either workload, falls outside [0.93, 1.07].
3. **FUNCTION_FAIL.** Any of:
   - G1 or G2 emits tokens that differ from D1's on any row of either workload at either length;
   - a G arm's timed passes do not all digest the same.
4. **KEEP.** S16 = min(G1/E1, G2/E2) on W16 is below 1.25, or S1, the same on W1, is below 0.97.
5. **DIVERGENT.** Fewer than 12 of W16's 16 rows have G1 agree with E1 for their first 16 LONG tokens. A broken
   function diverges at once; benign reordering of bf16 arithmetic does not, on most rows.
6. **DEFAULT_GRAPHS** otherwise.

**Reported, with no bar:**
- E1 against E2 tokens;
- the E-against-G first-divergence position of every W16 row and of W1;
- G-minus-E peak memory and load time.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | S16 lies between 2 and 15, host-dependent (P80's NF4 run read ×2.3 on a Zen 5 host, and P81/P82 read ×12–14 on the int4 stack on EPYC hosts) |
| Q2 | S1 lies between 1.5 and 8 |
| Q3 | G1 ≡ G2 ≡ D1 on every row (P82 / B771b's identity, through the server) |
| Q4 | E1 ≡ E2 (eager is deterministic) |
| Q5 | E and G differ on some W16 rows, the median first divergence is ≥ 32 tokens, and the sanity bar holds |
| Q6 | G's peak allocated memory exceeds E's by ≤ 0.5 GiB (16 scratch slots and 5 graphs; P81 measured 0.05), and its load time by ≤ 20 s |
| Q7 | the verdict is DEFAULT_GRAPHS |

## Consequence, registered now

- **DEFAULT_GRAPHS:**
  - `serve_paged`'s `E4B_PAGED_GRAPHS` default becomes `auto`. Graphs are on when the device is CUDA and the
    placement is `all-vram`. `0` turns them off. `1` forces them, refusing where it refuses today.
  - `docs/SERVING.md` says graphs are the default and that the registered serving numbers are that path.
  - A register row `e4b.serve.decode-graphs-default.qwen3.5090.<date>` carries S16, from the verdict file.
  - #770 closes.
  - Scope: this was read on Qwen3-30B-A3B. Hybrid models' graphs are P101's (SUPPORTED); other families ride the same
    capture code, with no reading of their own, and the docs say so.
- **KEEP:** graphs stay opt-in, the docs state the measured ratio, and #770 stays open with the reading.
- **DIVERGENT or FUNCTION_FAIL:** graphs stay opt-in. The divergence is the next investigation, and a FUNCTION_FAIL is a
  bug filed against the graph path.
- **NOISY or VOID:** no consequence. A rerun is a new attempt under this registration if the cause is the host, or an
  amendment if it is the harness.

## The proving rental

No local card can run the fp8 paged KV (sm_89+), so the box cannot be rehearsed locally beyond its CPU tests
(`tests/test_p109_box.py`, a scripted runner under a real scheduler, through the reducer). The proving rental therefore
runs the whole box end to end, on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89` (P94's pin, SC1's proof model), at SHORT 8 / LONG 24 and 1 rep:
- the refusals;
- install and tripwire;
- both self-tests;
- the premise;
- fetch, bake, prompts, the five arms and the reducer.

**PROVED** iff the lane exits 0 with a verdict other than VOID. The proof's verdict is not a reading.

**The premise, on the card before anything is fetched:** `tests/test_decode_graph_buckets.py` must report 7 passed,
none skipped. That means a bucket replay decodes exactly as the padded eager step on a tiny model with the real fp8 KV.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 2.5 h at ≤ $0.75/h, estimate $1.875. The expected time is about 75 min: install
  10, fetch 15, bake 10–20, five arms of about 6 min each.
- **Lane ceiling:** $3.50, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals (exit codes) run before any install. They are card class (15), disk < 150 GB (13), host RAM
  < 60 GiB (16), a dud box (10) and the premise (25).
- **STOP-2:** an arm that cannot finish 10 minutes before the deadline is skipped, and the reducer VOIDs on the missing
  arm.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p109/staged.sha256`.

## What this lane cannot say

- Nothing about other families, the int4 stack, or `max_seqs` other than 16.
- Nothing about TTFT. Prefill is in both walls and cancels in the slope; its arithmetic changes under G, as the sanity
  bar records.
- Nothing about quality beyond the sanity bar. The device-grouped path is the one the registered serving numbers ran
  on, and this lane does not re-gate it.
- Nothing about hosts unlike the one drawn: the eager path is host-bound, so S16 moves with the host. The rule's bars
  are set where any measured host clears them.

## Receipts

Fetched to the run directory's `p109/` and committed to `bench/p109/receipts/<run>/`:
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`;
- `arm_{E1,G1,G2,E2,D1}.json`, `verdict.json`;
- `logs/` (install, premise, fetch, bake, prompts, each arm), `work/bake.json`, and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p109.md` is written from those files.
