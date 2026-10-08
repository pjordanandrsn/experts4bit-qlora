# P118 -- does reading each decode step's tokens one step late (the next step issued from the device-resident tokens) decode the default serve_paged server faster at one request, with identical tokens? One RTX 5090

Lane number claimed by `prereg/p118` (pushed 2026-10-08T01:00:39Z). Issue: e4b#1313 (B1 lane 4: the out-of-graph host
step). The switch: e4b#1339 (`E4B_PAGED_DECODE_LOOKAHEAD`, opt-in, off by default). This page merges after #1339 and
after the serve-throughput lane's P117 reading (#1335), as agreed on the bus; no box runs before it merges.

## Why now

**The host gap.** A decode step in `PagedModelRunner._run_decode_bucketed` reads its tokens back
(`buf["tok"][:n].tolist()`) before the step ends. From that sync until the next replay is issued, the GPU idles. The
host meanwhile mirrors the KV lengths, the scheduler emits, retires and plans, and the next step's ids, positions and
slot selection are copied in.
- SC1b's class census of the int4 B=1 step (`bench/h2h-2026-10-02/sc1b/`, one RTX 5090) booked that idle at
  **0.365 ms** of a 4.44 ms step (`idle_out`), against vLLM's 0.014.
- The same runner and scheduler serve the NF4 route, so the gap is route-independent in kind. Its size on the default
  NF4 server has never been measured. This lane's L0 trace is the first.

**The lever** (#1339). With `E4B_PAGED_DECODE_LOOKAHEAD=1`, `ContinuousScheduler(lookahead=True)` issues step t+1
(`PagedModelRunner.issue_decode`) before it collects step t (`collect_decode`).
- Step t+1's input ids are gathered on the device from a per-slot table of each slot's newest token, which step t wrote.
  Its positions count the queued step. Its slot ids and positions go through event-guarded pinned staging.
- The GPU therefore has a step queued while the host works. When the host work is shorter than a step, the decode
  step's period falls to its device time.
- The device inputs are the synchronous step's (ids, positions, slots, padding), so the **tokens must be identical**.
  This lane rules on that, rather than on a quality bar.

## The subject

The shipped default server at the launch commit:
- graphs `auto`, buckets 1–16, all-vram, bulk KV, one KV-table selection per step, the fusion knobs at their
  launch-commit defaults;
- **`max_seqs` 16, named** (`E4B_PAGED_MAX_SEQS=16`), the W16 workload's width, so a later `max_seqs` default does
  not move the subject.

It runs Qwen3-30B-A3B @ `ad44e777bcd1…`, with the NF4 arena baked on the box by P39's `k8_bake.py`. grouped-nf4-gemm
is at **`6ee2e10`** (0.43.0, e4b CI's pin at registration since e4b 0.49.0; a registered constant in `p118_run.sh` and the
reducer), with transformers 5.17.0 and the image's torch.
- 0.43.0 carries P116's consequence: `GNF4_GEMV_BW=auto` is grouped-nf4-gemm's default, so at Qwen3-30B-A3B's two
  expert shapes the subject decodes through the bandwidth GEMV. The registration moved here from 0.42.0 before it merged,
  when CI's pin moved (this page's own rule).
- Whatever the launch commit's defaults are, they are recorded per arm (`fusions`, `fusion_modes`, `levers_env`,
  `grouping`) and must agree across the arms.
- **If e4b CI's grouped-nf4-gemm pin moves before the run, an amendment moves this pin with it, before any box.**

## The arms

Four fresh processes, ABBA: **L0a L1a L1b L0b**. The arms differ only in the switch:
- **L0:** `E4B_PAGED_DECODE_LOOKAHEAD` unset, the shipped default. Every decode step is read back before the next is
  issued.
- **L1:** `E4B_PAGED_DECODE_LOOKAHEAD=1`.

**Workloads** are P109's, from `p109_box.py` at its registered bytes (the bytes P115 and P116 pinned):
- W16: 16 distinct 512-token wikitext prompts at once. W1: row 0 alone.
- 32 and 160 new tokens, 1 warm pass then 3 timed passes, p37's slope.
- No stop ids: every sequence runs to its length. The lookahead therefore issues the synchronous path's rows, step for
  step.

**Engagement**, counted in-process on the runner's two entry points: issues, collects, and collects made with a newer
step already queued behind them (the overlap).
- L1 runs the lookahead scheduler, collects every step it issued, overlaps at least **90 %** of its collects at each
  workload, and discards nothing.
  - At W1 and 160 tokens, 158 of a pass's 159 collects should overlap; only the last step drains.
- L0 runs the synchronous scheduler and never calls the entry points.

**The mechanism.** After its timed passes, each arm runs one more pass per workload at 160 tokens
under the step tracer (`engines/step_trace.py`), stepped as the server's engine loop steps it. Over the decode-only
steps:
- **L0:** the GPU is idle at `dec_prep`, so a step's device time is `gpu.dec_issue - gpu.dec_prep` and the rest of
  `step_ms` is the **host gap**.
- **L1:** the step period should fall to about L0's device time.

L0's W1 host gap is the rule's premise (rung 4 below). Everything else here is reported, not ruled: per workload,
L0's host gap beside the period L1 saved. The traces travel as receipts.

## The rule (`bench/p118/p118_reduce.py`, self-tested on 28 cases)

First rung that applies:
1. **VOID:**
   - a record is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - a bucket was not captured;
   - the arms differ in prompts, lengths or fusion census;
   - a slope is void;
   - the engagement rule fails;
   - either L0 arm's traced W1 pass lacks a host gap, so the premise below cannot be read.
2. **NOISY:** a self-pair, L0b/L0a or L1b/L1a, falls outside [0.96, 1.04] at W16, or outside **[0.99, 1.01] at W1**.
   That is half the W1 gain bar, so instrument drift cannot pass as the gain. P111's self-pairs read 0.997–1.002.
3. **FUNCTION_FAIL:**
   - any two arms decode different tokens on any row, workload or length (L1 ≡ L0 is the contract);
   - one arm's timed reps digest differently;
   - the arms' bucket statistics (`graph_stats` over the timed passes) differ.
4. **UNTESTED (premise unmet):** L0's traced W1 host gap is **< 0.2 ms** per decode step. The gap is the mean of
   L0a's and L0b's median `step_ms − (gpu.dec_issue − gpu.dec_prep)` over the decode-only steps.
   - The lookahead hides host time, so on a host with no gap to hide, a slow or flat reading would be the host's answer
     and not the switch's. Re-ask on a host with a gap.
   - The rung precedes both SLOWER and DEFAULT_ON: a "gain" with no gap behind it is not the switch's either.
   - Added in the maintainer's review of #1340, before any data (TC1 amendment 63's lesson: a host-time remedy read on a
     box that is not host-bound reads FALSIFIED for the wrong reason).
5. **SLOWER:** g1 = min(L1a/L0a, L1b/L0b) at W1 **< 1.02**, or g16 at W16 **< 0.99**. The gap was there and the
   lookahead did not recover it.
6. **DEFAULT_ON:** otherwise.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | engagement exact: L1 overlaps ≥ 99 % of its collects at W1 and ≥ 90 % at W16, discards nothing; L0 makes no lookahead call; every bucket captured in every arm |
| Q2 | the four arms decode identical tokens on every row, workload and length; identical bucket statistics |
| Q3 | L0's traced W1 host gap **0.25–0.60 ms** per decode step; **g1 ∈ [1.02, 1.10]**. The L0 W1 step is 6–8.5 ms: P116's bandwidth GEMV (now the default) read 8.22 ms, and a family-scoped fused-stack default, if it lands before the launch, takes it lower (P115 read the stack 1.43× at W1 on its own) |
| Q4 | **g16 ∈ [1.00, 1.04]**: the gap is a smaller share of a 16-row step, but 16 rows' host mirrors make it larger |
| Q5 | self-pairs within [0.995, 1.005] at W1 and [0.98, 1.02] at W16 |
| Q6 | at W1 the period L1 saved is 0.7–1.1× L0's host gap (the lookahead adds two small gathers and a 1-row D2H a step) |
| Q7 | DEFAULT_ON about 60 %, UNTESTED about 20 % (a host gap under 0.2 ms), SLOWER about 10 %, the rest VOID or NOISY |
| Q8 | each arm ≤ 4 min including its traced passes; peak memory as P116's speed arms (≤ 23 GiB) |

## Consequence, registered now

- **DEFAULT_ON:**
  - a flip PR makes the lookahead the default wherever decode graphs run: `E4B_PAGED_DECODE_LOOKAHEAD` gains `auto`,
    on with graphs and off without them, as its default, with a `tests/test_*_default.py` pin on the
    `test_kv_step_select_default.py` pattern;
  - `docs/SERVING.md` and the changelog follow;
  - the register row is `e4b.serve.p118.decode-lookahead.qwen3.5090.<date>` (g1).
  - **Scope:** read on Qwen3-30B-A3B NF4 at 1 and 16 requests with buckets up to 16. The int4 route and other families
    run the same runner path, but this lane does not read them. Buckets above 16 (P117's) are not read either. The
    flip PR says so.
- **UNTESTED:** the switch stays opt-in, and nothing is concluded about it. The question is re-asked on a host whose
  traced gap meets the premise, under an amendment that names that host class.
- **SLOWER:** the switch stays opt-in. The ratios and the trace are recorded; the trace shows how much of the gap the
  saving reached.
- **FUNCTION_FAIL:** a correctness defect in the lookahead. The switch stays off and the defect is found first.
- **NOISY or VOID:** no consequence. One rerun inside the ceiling, then an amendment.

## The premise and the proving rental

**Premise**, on the card before anything is fetched (rc 25):
- `tests/test_decode_graph_buckets.py`, `tests/test_kv_step_select.py` and `tests/test_decode_lookahead_gpu.py`:
  **16 passed**, none skipped.
  - The bucket graphs replay as the padded eager step.
  - The tiny Qwen3 decodes, through captured graphs and the padded eager step, the synchronous tokens, buckets and KV
    lengths with the lookahead, the overlap engaged; a stop through the lookahead keeps the synchronous streams.
- `tests/test_decode_lookahead.py` without its HTTP case: **18 passed**, none skipped. That is the protocol on the CPU
  and on this card's pinned staging and events.

**Proof** (`p118-prove-<n>`): the whole box on Granite-3.1-3b-a800m at 8 / 24 tokens and 1 rep. A VOID from the reducer
fails the proof (rc 27). Its verdict is not a reading.

**Order on the box:**
1. Refusals, then install and the tripwire. The tripwire checks:
   - the pins;
   - the switch opt-in at this commit and wired into `build_engine`'s scheduler;
   - the runner's two entry points;
   - the step tracer.
2. The two self-tests, then the premise.
3. Fetch, bake and prompts.
4. The four arms, each with its traced passes after its timed ones.
5. The reducer.

## Budget and STOP

- **Proof:** one RTX 5090, **guard 0.75 h**, about $0.5 at the launcher's policy rate.
- **Reading:** one RTX 5090, **guard 1.25 h**, `--download-gb 61`, about $1.6 at the policy rate. Expected about 40
  minutes: install and premise 6, fetch 6–10, bake 2, prompts 1, four arms about 16.
- **Lane ceiling $2.50, hard stop $3.00**, covering the proof, the reading and one rerun. A launch goes ahead only while
  the lane's actual spend plus that run's launcher estimate is ≤ $3.00. Every run sits inside the owner's standing no-ask
  tier for a single run under $15; anything over $15 needs the maintainer lane's approval first.
- **STOP-1:** the refusals run before any install: CUDA unusable 18 (the host floor), dud box 10, card class 15, disk
  < 150 GB 13, host RAM < 60 GiB 16, premise 25.
- **STOP-2:** every time-left check fits inside its guard, enforced by `tests/test_p118_staged_pin.py`.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p118/staged.sha256`.

## What was seen before this page (stated, not hidden)

- **No P118 data exists.** No served run has ever set `E4B_PAGED_DECODE_LOOKAHEAD=1`, and no step trace of the default
  NF4 server's decode has been taken.
- **SC1b's int4 numbers** are quoted above from its receipts.
- **Locally (CPU):** the box's self-test (12 cases) and the reducer's (28) pass. `tests/test_p118_staged_pin.py` passes.
  `tests/test_decode_lookahead.py` passes (14). `tests/test_decode_lookahead_gpu.py` skips without sm_89+.
- **On the RTX A2000** (sm_86, the NAS card; correctness only under #1133, no timing taken), with #1339's code:
  `tests/test_decode_lookahead.py` 15 passed on the CPU and CUDA cases, and the CUDA runner cases passed in 5 of 5
  repeats with a slowed stand-in graph. The sm_89+ file skipped by name.

## What this lane cannot say

- **Nothing about traffic with stop ids.** The workloads run to their lengths. Under the lookahead a stop costs one
  discarded step, and frees its slot one step later; that is documented, not measured.
- **Nothing about buckets above 16, other families, the int4 route or other cards.**
- **Nothing about the HTTP server's own host work** (dispatch, detokenisation, the lock). The box drives the scheduler
  as P109 does, so the served gap can only be larger than the one read here.

## Receipts

The run directory's `p118/` is fetched and committed to `bench/p118/receipts/<run>/`:
- `arm_{L0a,L1a,L1b,L0b}.json`, `traces/trace_<tag>_<workload>.jsonl`, `verdict.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `logs/` and `work/bake.json`;
- `SHA256SUMS`.

The checkpoint and the arena stay on the box. `RESULTS-p118.md` is written from those files.
