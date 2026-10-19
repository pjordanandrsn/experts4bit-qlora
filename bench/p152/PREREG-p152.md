# P152 — where Qwen3-30B-A3B's 16-request served step goes, kernel by kernel (an instrument)

**Status:** registered before any data. This is an instrument registration: it names the W16 decode-step candidates for
stream (B) of e4b#1313, and it changes no default, no gate and no speed claim. Lane number reserved by the maintainer on
2026-10-19 (stream (B), the fresh decode-step candidates; stream (A), P150 and P151, is not consumed here).

## Why

Stream (B) started at $0 from `pername-5090-1` (P142's per-name census of Qwen3 at W1) and its receipts:
- **The 0.472 ms "tail" outside the graph is the profiler's.** The same box's unprofiled arms read W1 at 3.9134 and
  3.9198 ms a step, against a profiled graph span of 3.9087 ms, so the real tail is about 0.1-0.3 % of the step. Closed
  as NO_CANDIDATE; every new registration now divides by the unprofiled step (the maintainer's rule).
- **W1's expert GEMV (K33) runs at 77 % (gate_up) and 63 % (down) of the copy rate.** That is 0.587 and 0.362 ms against
  byte floors of 0.453 and 0.227 ms, a headroom of 0.269 ms (6.9 % of 3.913). It is a real candidate, held until this
  read ranks it against W16.
- **W16 is where serving lives, and it is unnamed for Qwen3.** The same receipt reads the unprofiled W16 step at 16.21
  and 16.28 ms (987 and 983 tok/s). At T > 1 the experts take the grouped small-M GEMM, not K33. Under uniform routing,
  16 tokens x top-8 touch about 82 of 128 experts a layer: about 10.5 GB a step and a copy floor near 7.0 ms. P145's W16
  table for Mixtral put its grouped NF4 GEMM at 91 % of the step. Nothing about Qwen3 at W16 is measured.

## The instrument

P145 (`bench/p145/PREREG-p145.md`, the per-name census at W1 and W16) by counted substitutions:
- **Family:** Qwen3-30B-A3B @ `ad44e777`, 48 MoE layers, at **W1 and W16** (P136's reducer is given Qwen3 at (1, 16), as
  the per-name census gave it W1).
- **Server:** the shipped default at 16 slots x 4096 tokens, graphs on, every knob unset.
- **Pins:**
  - e4b at this registration's merge commit;
  - grouped-nf4-gemm **v0.47.1** at `38b308887688ea5bda7dd449bd3cc3dc1426b995`, e4b CI's pin. The tripwire asserts
    Qwen3's two shapes in `_BW_SHAPES` and the qwen3_moe fusions resolved by default;
  - transformers 5.17.0, torch 2.8.0+cu128, nsys 2025.6.1.190.
- **Unchanged from P145, P136 and P142:**
  - host refusals, tripwire, self-tests and P136's 19 premise tests;
  - two unprofiled speed arms S1a / S1b (the self-pair must sit in [0.97, 1.03]), each reading W1 and W16;
  - graph-mode and node-mode captures of 64 steps after 32 skipped, at each batch;
  - SC1b's `arm` with P142's class map, and `pername_table.py` at its pinned bytes on each node export.

**Added (the maintainer's terms for this lane):**
1. **The denominator** is each batch's **unprofiled** step: the mean of S1a's and S1b's `decode_ms_per_step` at that
   batch, from the same box. The profiled node period P is reported as context only.
2. **The routing census** (`p152_route.py`): the census's own W16 window, run **eager** (`E4B_PAGED_GRAPHS=0`, nothing
   else set). `census_window` gives it the same prompts, ramp, 32 skipped and 64 bracketed full-decode steps, with the
   profiler's start/stop replaced by a forward hook on every layer's router (`mlp.gate`'s third output, the expert ids).
   - It records the unique experts each layer touched, per step, and each layer's expert hit counts.
   - A fault is recorded if a step's routers are called out of layer order, if a call has other than 16 rows x top-8, if
     a row repeats an expert, or if an id is out of range.
3. **The power floor:** `power.limit` >= 575 W before anything is installed (rc 17; unreadable, 15), as P149's. The GPU
   state (SM and memory clocks, power, temperature, clock event reasons) is appended to `gpu_phases.csv` at the start
   and end of every arm, capture and the routing pass. Reported only.

## What each line decides (`p152_reduce.py`, fixed before data)

- **VOID:**
  - anything P136's reducer VOIDs for Qwen3;
  - either batch's table missing, not closing every replay exactly, or under 56 kept steps;
  - at W1, the expert path not at **48 launches per step** for each of gate_up, `_swiglu_rows` and down;
  - an arm's unprofiled step missing at either batch.
- **W16 (the subject):** every line by removable time, with its share of the **unprofiled W16 step**. The top five are
  **named** as candidates. Nothing is closed at W16. The head (one launch a step) is marked as stream (A)'s (P150) and is
  never named.
- **W1 (the re-read):** the same, with dense_gemm **closed** (stream (A): D1 and D2, P151 and P150). The two expert
  GEMVs rank on removable minus K33's isolated time (gate_up 0.6068, down 0.3540 ms for 48 launches), as the per-name
  census reads them.
- **The expert byte floor (REPORTING only):** each expert is 0.5625 B a weight (packed nibbles plus an fp32 absmax per 64:
  the arena `bench/p39/k8_bake.py` bakes), so gate_up (1536 x 2048) plus down (2048 x 768) is 2,654,208 B. The floor is
  the bytes over 1,498.9 GB/s, P141's measured copy rate:
  - W1: 8 experts x 48 layers, a floor of 0.680 ms;
  - W16: the routing census's mean unique experts per step x 2,654,208 B, against the moe_expert class's removable time
    (the headroom).

  Routing missing or faulty reads the W16 floor **UNREAD**; the census stands.
- **Nothing is proposed automatically.** The named candidates and the floors go to the maintainer. Each lever is its own
  registration. The K33 plan sweep (W1, about 0.27 ms) is ranked against W16's candidates after this read.

Profiled node times are never quoted as speed.

**Predictions (graded in the read, never ruling):**
- the unprofiled steps: W16 in 15.5-17.0 ms, W1 in 3.85-3.98 ms (pername-5090-1 read 16.21-16.28 and 3.913-3.920);
- at W16 the moe_expert class holds at least 60 % of the unprofiled step, and its top line is the grouped small-M GEMM;
- the routing census reads a mean of 55-85 unique experts per layer and step (uniform routing: 82.4; skew lowers it);
- the W16 expert floor is 4.5-7.5 ms, and the expert path's headroom over it is at least 3 ms;
- at W1 the top line not owned by stream (A) is the attention decode (`_fp8_paged_decode_split_f8dot`), 0.45-0.65 ms;
- both self-pairs within [0.99, 1.01].

## Budget and sequencing

- **One RTX 5090** at <= $0.85/h, **guard 1.25 h**, 61 GB at <= $0.005/GB: worst 1.25 x $0.85 + 61 x $0.005 = **$1.37**,
  under the standing rule. Disk 200 GB (the box's floor is 150 GB free).
- **Run ids** `p152-5090-1` to `-3`; two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane. A pre-flight host
  stall or a power-floor refusal (rc 17) is not a strike.
- **Every time-left check** fits the guard (`tests/test_p152.py`). The rental waits for this registration's merge and
  the maintainer's final pass, and goes through the shared pre-flight and `--validate`.

## Receipts

Fetched to the run directory's `p152/` and committed to `bench/p152/receipts/<run>/`:
- `arm_S1a.json`, `arm_S1b.json`, `census_drv_b{1,16}_{graph,node}.json`, `census_b{1,16}.json` and
  `pername_b{1,16}.json`;
- `route_b16.json`, `gpu_phases.csv` and `verdict_p152.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, the logs and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p152.md` is written from those files.

## What this lane cannot say

- Nothing about speed: it names candidates; a lever's gain is its own served read.
- Nothing about other cards or families: one RTX 5090, Qwen3-30B-A3B.
- Nothing about loads other than 16 rows in steady decode (W16) and one row (W1); no prefill, no mixed steps.
