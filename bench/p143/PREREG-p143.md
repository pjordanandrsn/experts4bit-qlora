# P143 step 1: the paged FP8 attention decode at B=1, by context and split count (e4b#1313)

**Status:** registered, not run. A microbench: it decides whether step 2 (a split-plan lever, with a P127-style bitwise
and served read) happens. No default, gate or speed claim changes.

## Why, from data already seen

P142 (`pername-5090-1`, store receipts) read the B=1 step's top addressable line as `_fp8_paged_decode_split_f8dot`:
0.556 ms of P = 4.380 ms (12.7 %), 48 launches of about 11.6 µs with nothing billed to a predecessor. The library plans
`n_split` from the block table's **capacity**, not the live length (a host sync avoided; a CUDA graph fixes the grid):
`min(32, 11·SMs / (5·B·H_kv), capacity / ktile)`. That is 32 splits × 4 KV heads = 128 programs at B=1 with 4096-token
slots. At P142's ~640 live positions the KV read is about 0.65 MB per layer, ~0.4 µs at the card's bandwidth. So the
launch is latency-bound. Most splits find no tokens, and each still holds a fused-combine slot. Whether fewer splits help
is not obvious: each working split then walks more tokens in series. This bench measures it.

## The bench (`p143_bench.py`)

e4b's own `Fp8PagedKV` at Qwen3-30B-A3B's attention geometry: 48 layers, 4 KV heads, 32 q heads, head_dim 128, 16
slots of 4096 tokens, `k_groups` auto. Slot 0 is filled to 4096 tokens per layer by one bulk `append`. Each context
{256, 640, 1024, 2048, 4096} is set by `rewind`. Each layer's arguments are `kernel_args(layer, [0])`, exactly what
`Fp8PagedKV.attention` passes to `fp8_paged_decode_attention`, with Qwen3's `sm_scale`, no window and no sinks, as
`paged_attention` serves them. The fused combine is the library's default (on for fp8 compute).

The arms are `n_split` ∈ {library plan, 4, 8, 16, 32}: 25 arms. Each arm is one CUDA graph of the 48 layers' calls,
warmed, then 200 replays, each timed by its own event pair. Two passes, the second in reverse arm order. The KV stays
L2-warm across replays; the anchor below checks the bench still transfers to the served step.

Subject: e4b at this registration's merge commit, grouped-nf4-gemm 0.46.1 @ `70bc03cb` (P142's), torch
2.8.0+cu128. The box is P142's runner by counted edits, with the same refusals, install, tripwire and P136's 19-test
premise. The tripwire also asserts `n_split` on the kernel and the KV class's `append` / `rewind` / `kernel_args`. No
model, no checkpoint, no hf token.

## The rule (`p143_reduce.py`, fixed before data)

Times are per-replay medians for 48 launches, averaged over the two passes. **BAR = 3 % of P142's P = 0.131405 ms.**

- **VOID:**
  - an arm is missing;
  - the library plan's self-pair (pass 2 / pass 1) lies outside [0.97, 1.03] at any context;
  - the library plan is more than 3 % from the 32-split arm at any context (the rule assumes 32 at B=1);
  - the library plan at 640 lies outside 0.5–1.5× P142's served 0.5558 ms (the bench does not transfer).
- **GO_FIXED:** some fixed `n_split` ∈ {4, 8, 16} saves ≥ BAR at 640 against the library plan AND loses no more than
  noise at 4096. Noise is the larger of the two arms' |pass 1 − pass 2|. Step 2 is that plan for bucket 1.
- **GO_ADAPTIVE:** otherwise, the best split *per context* saves ≥ BAR at 640. This is an upper bound for a fixed-grid
  plan that picks its live split count on the device. Step 2 must build that plan and read it bitwise and served.
- **PARK:** otherwise. The lever closes with the table.

Reported, never ruling: the whole grid in ms per step and µs per launch, the best split per context, and the anchor ratio.

**W16.** At B=16 the library already plans 5 splits (`B·H_kv` = 64 programs per split). A bucket-1 plan change leaves the
W16 graphs untouched. Step 2 still measures W16, as asked. The serve-throughput lane is flagged for any SC2f collision.

**Predictions (graded, never ruling):**
- the library's launch at 640 is 9–14 µs (P142: 11.6);
- the best fixed split at 640 is 8 or 16, saving 0.10–0.25 ms per step;
- at 4096 every split below 32 loses beyond noise;
- verdict GO_ADAPTIVE.

## The box

- **Run ids:** `p143-5090-1`, with `-2` as a pre-create spare. `-2` runs only if `-1` is refused before create or stops
  before the bench. A READ or a VOID after the bench is the result.
- **Host:** one verified-secure RTX 5090, direct SSH, at most **$0.80/h**.
- **Guard and downloads:** guard **0.5 h**; about 5 GB of installs at ≤ $0.005/GB.
- **Worst case:** 0.5 × $0.80 + 5 × $0.005 = **$0.43**, under the lane's standing authority.
- **Receipts:** the bench record, the verdict, the summary and the logs travel back.

## Amendment 1 (2026-10-18, after `p143-5090-1`)

**What happened.** `p143-5090-1` (machine 40521) read HARNESS_ERROR, rc 34, $0.008. The refusals, install, tripwire and
premise passed (19 passed). `p143_bench.py` stopped before timing any arm. `Fp8PagedKV.rewind` only moves a sequence
back and refuses a forward move (`rewind forward (layer 0: 256 -> 640)`), and the bench took its contexts ascending
after filling slot 0 to 4096. No arm was timed, so no data was seen. It is the lane's **strike 1**: one more paid
failure stops the lane for a simpler design.

**The fix.**
- The setup is now `fill`, `hold_args` and `check_held` in `p143_bench.py`, and the box runs those same functions.
- `hold_args` takes each context's `kernel_args` **longest first**. Each copy keeps its lengths through the later
  rewinds, and the copies are held for the run, as before.
- `tests/test_p143.py` now runs that setup on CPU, on e4b's own `Fp8PagedKV` with the registered contexts and capacity,
  wherever grouped-nf4-gemm is installed (CI installs it). An ascending walk fails it.
- Nothing else changes: the grid, the timing, the rule and the box terms are as registered.

**Run ids.** `p143-5090-2` is the rerun under this amendment. `p143-5090-3` is its pre-create spare and runs only if
`-2` is refused before create.

## Amendment 2 (2026-10-18): step 2, the served read of the split cap

**Status:** registered, not run. Step 2 under the maintainer's 18:12Z ruling: a kernel tolerance contract, a kernel
timing add-on and a served read on Qwen3-30B-A3B. Together they decide whether `qwen3_moe` takes
`E4B_ATTN_SPLIT_CAP=16` by default. Nothing changes by default before the verdict: the cap is opt-in at e4b#1768, and
`ATTN_SPLIT_CAP_DEFAULT_FAMILIES` is empty.

**Seen data (step 1, `p143-5090-2`, GO_FIXED).** This data was seen before this amendment was written, so it informs the
design and is never the quoted gain. Microseconds per launch (library plan | 4 | 8 | 16 | 32):

| context | plan  | 4     | 8     | 16    | 32    |
|---------|-------|-------|-------|-------|-------|
| 256     | 11.09 | 5.11  | 5.88  | 7.59  | 11.26 |
| 640     | 11.19 | 8.18  | 7.46  | 7.63  | 11.17 |
| 1024    | 11.10 | 9.72  | 7.44  | 7.67  | 11.28 |
| 2048    | 12.53 | 16.92 | 11.31 | 10.12 | 12.45 |
| 4096    | 15.27 | 29.96 | 18.36 | 14.43 | 15.33 |

At 640, 16 splits saves 0.171 ms per step (bar 0.131), and at 4096 it is 0.040 ms faster. That 0.171 ms is a
microbench of the attention launch alone, L2-warm. The quoted gain is the served read's g1.

**The lever (merged before this amendment's data).**
- grouped-nf4-gemm 0.47.1 (grouped-nf4-gemm#558, tag commit `38b30888`):
  - `decode_split_plan` is now a pure function.
  - `fp8_paged_decode_attention(..., split_cap=None)` plans `min(cap, plan)`.
  - The default is unchanged and byte-identical.
- e4b#1768:
  - `E4B_ATTN_SPLIT_CAP` (0..32; 0 = the library's plan) is resolved per family.
  - `Fp8PagedKV.split_cap` is set by `build_engine` before graph capture. The KV hands the kernel `split_cap` only on the
    fp8 decode path, never on speculative verify and never over an explicit `n_split`.
  - `info["attn_split_cap"]` records the cap, its source and, per row count B, the plan and split count the cap met.
- At Qwen3's 4 KV heads and 4096-token slots on the 5090 (170 SMs), the plan is 32 / 32 / 23 / 11 / 5 at buckets
  1 / 2 / 4 / 8 / 16. With the cap it is 16 / 16 / 16 / 11 / 5: only buckets 1, 2 and 4 move.

**Why this read is not bitwise.** Fewer splits regroup the fp32 log-sum-exp combine, so a bf16 output can differ in its
last bit. The gate is a tolerance contract plus P115 Phase D's SANE, not identity.

### (a) The kernel tolerance contract (`p143a2_kernel.py`, 5090, no model)

- **Setup:** e4b's `Fp8PagedKV` at Qwen3's geometry, as in step 1. Slots 0–3 are filled to 4096 tokens per layer and cut
  back to each context, longest first.
- **The two arms:** at each B ∈ {1, 2, 4} and context ∈ {256, 640, 1024, 2048, 4096}, the **served call**
  (`Fp8PagedKV.attention`, Qwen3's `sm_scale`) runs every layer twice: with the KV's cap unset (the library's plan) and
  at `split_cap = 16`.
- **The reference:** fp64, from the KV's own oracle. `reference_kv(..., dtype=float64)` gives the stored e4m3 bytes and
  scales, dequantized; then softmax(q kᵀ·scale) v in fp64, with query head h reading KV head h // 8.
- **The error:** RMS over 48 layers × B rows × 32 heads × 128, for each arm.
- **Rule:** at every cell, the cap's error ≤ **1.05 ×** the plan's.

### (b) The kernel timing add-on (same script)

- **Setup:** slots 4–7. At B ∈ {1, 2, 4} × context ∈ {640, 1024, 2048}, one CUDA graph of the 48 layers' kernel calls
  per arm: plan (no `split_cap`) and cap (`split_cap = 16`).
- **Held arguments:** each context's `kernel_args` copies are held for the run and checked before timing, as in
  Amendment 1.
- **Timing:** 20 warm replays, then 200 replays each timed by its own event pair. Two passes, the second in reverse order.
- **Rule:** at B ∈ {2, 4} (captured graphs in every server), the cap's mean over the two passes may not exceed the plan's
  by more than the noise. Noise is the larger of the two arms' |pass 1 − pass 2|. B = 1 repeats step 1 and is reported
  only.

### (c) The CPU test

`tests/test_attn_split_cap.py` (e4b#1768) runs the real planner. It checks:
- the exact new `n_split` at B 1..5;
- equality with the plan at B ∈ {6, 8, 16, 32, 64};
- both 170 SMs (5090) and 188 SMs (PRO 6000).

`tests/test_p143a2.py` covers this kit:
- the staged pins;
- the runner's order;
- the kernel script's served arguments and held graphs;
- its setup on the KV class on CPU;
- the SANE capacity patch on the real KV class;
- the self-tests, run from one directory as on the box;
- seven reducer mutants, each killed by the self-test.

### (d) The served read (`p143a2_box.py`)

**Engine.** The shipped default server at grouped-nf4-gemm 0.47.1:
- the B=1 fused stack on (auto, qwen3_moe allowlisted);
- the bandwidth decode GEMV on;
- 16 slots of 4096 tokens.

**Arms.** They differ only in `E4B_ATTN_SPLIT_CAP`. S0 = `0`, named explicitly: the library's plan, today's server.
S1 = `16`.

**Speed** (P109's workloads, bytes and timing). The arms run in four processes in ABBA order: S0a, S1a, S1b, S0b. Each
runs W16 and W1 on the graph server.

**SANE** (P115 Phase D's gate, unchanged in windows, positions, grouping and bars). Phase B's instrument runs on the
default server built eager, one window per pass (bucket 1, T == 1), over 12 wikitext windows of 128 positions. S0 scores
R; S1 scores ON against it.

**One change to Phase D's instrument, without which the gate reads nothing.** Its pass (`p110_box.paged_pass`) builds its
own KV of `P + C + 16` = 656 tokens. At that capacity the library plans **11** splits at B = 1, so a cap of 16 cannot
bind, and S0 and S1 would run the same arithmetic.
- Inside the SANE measurement, every KV is built at the server's own capacity: `cfg.max_tokens_per_seq`, 4096 tokens,
  where the plan is 32.
- Each carries the server's resolved cap, as `build_engine` sets its own.
- Recorded per phase: every pass KV's capacity, its cap, what the cap met, and the library's B = 1 plan at that capacity.

### The rule (`p143a2_reduce.py`, fixed before data)

The first rung that applies decides.

1. **VOID.** Any of:
   - **Records:** a record is missing or not ok.
   - **Pins and provenance:**
     - another e4b commit, grouped-nf4-gemm commit or model revision;
     - a card other than the 170-SM RTX 5090;
     - a bucket not captured, or bucket 1 never replayed;
     - the arms read different prompts or lengths;
     - a void slope;
     - the fused stack's census or modes off the default's;
     - the bandwidth GEMV not where the default puts it (Phase D's checks).
   - **Engagement:**
     - S1's build does not record cap 16 (explicit) with buckets 1, 2 and 4 at 16 splits and 8, 16 at the library's
       plan;
     - S0's build records any cap;
     - the SANE phases' pass KVs are not at 4096 tokens (256 blocks, plan 32 at B = 1);
     - S1's pass KVs do not carry cap 16, or S1's SANE phase does not cap B = 1;
     - a SANE phase is not at one window per pass, or is short of 12 windows.
   - **The kernel record:**
     - it is off the pin or the card, or its block table is not the server's 256 columns;
     - its contract or timing plans are off the registered values;
     - a cell is missing;
     - the plan's RMS error exceeds 0.25 × the reference's RMS (the reference does not describe the kernel).
2. **KERNEL_FAIL:** (a) or (b) fails.
3. **FUNCTION_FAIL:**
   - a self-pair (S0b/S0a, S1b/S1a) decodes different tokens on any row, workload or length;
   - one arm's timed reps digest differently.
4. **QUALITY_FAIL:** SANE fails, i.e. |mean NLL(ON) − NLL(R)| > 0.02 nats or argmax agreement < 0.95.
5. **FLIP:** g1 = min(S1a/S0a, S1b/S0b) at W1 ≥ **1.03**, **and** g16 (the same at W16) ≥ **0.99**, W16's
   no-regression line in P141's form. A separate PR then adds `qwen3_moe: 16` to `ATTN_SPLIT_CAP_DEFAULT_FAMILIES`,
   citing this read.
6. **NO_FLIP:** otherwise, with each failed bar named. The cap stays opt-in, with the numbers.

**Reported, never ruling:** the self-pairs, the SANE mean KL, every kernel cell (errors, ratios, bitwise agreement,
timings), B = 1's timing, whether W1 decodes the same tokens in S1 as in S0, and W16's token identity: rows identical
out of 16 and each row's first divergent step, per pair and length.

**W16 is a speed check, not an identity check (adopted before data, 2026-10-18 22:32Z, on serve-throughput's
finding).** `p109_box.run_pass` adds all 16 rows at once, and the scheduler takes one 512-token prompt of prefill per
step while every row already past prefill decodes. So W16's first decode steps run 1, 2, 3 and 4 rows, on buckets 1, 2
and 4, where the cap binds (32/32/23 → 16). A last-bit regrouping of the log-sum-exp there can move one argmax, and that
row then diverges for the rest of its tokens: an identity rule would FAIL on correct behaviour. Small-B quality is
SANE's question at T == 1, and SANE reads the capped B = 1 path.

**Predictions (graded, never ruling):**
- every contract ratio in [0.98, 1.02];
- at B 2 and 4 the cap is no slower and is faster at 640 and 1024;
- SANE |bias| ≤ 0.002 nats, argmax agreement ≥ 0.99;
- W1 tokens differ from S0's somewhere within 160 tokens;
- g1 1.025–1.045; g16 within 0.99–1.01;
- verdict FLIP.

### The box

- **Run ids:**
  - `p143a2-5090-1` is the reading.
  - `p143a2-5090-2` and `-3` are pre-create spares. Each runs only if the one before it is refused before create, or
    stops before the kernel step.
  - A verdict after the kernel step is the result.
- **Card and budget:** one RTX 5090 (compute capability 12.0, ≥ 30 GB, 170 SMs) on torch 2.8.0+cu128, at ≤ $0.80/h.
  - Disk: 170 GB; the runner refuses below 150 GB.
  - Download: 70 GB at ≤ $0.005/GB.
  - Guard: 1.25 h. Worst case $1.35 at the caps.
  - **Fallback tier** (approved 2026-10-18 22:54Z, before merge): if one full launch loop finds no stable offer under
    these terms, the 5090 class rate may be **≤ $0.85/h** including 170 GB. Everything else is unchanged ($/GB
    ≤ $0.005, 70 GB, 1.25 h). Worst case = 1.25 × 0.85 + 70 × 0.005 = **$1.41**. The pin post names the tier it used.
- **Runner** (`p143a2_run.sh`): P143's runner, by counted edits, plus P115 Phase D's served half. Order: install and
  tripwire, self-tests, P136's 19 premise tests, the kernel step (before anything is fetched), fetch, NF4 bake, prompts,
  four arms, two SANE phases, reduce.
  - The tripwire asserts grouped-nf4-gemm 0.47.1, `split_cap` on the kernel, and the planner's values at 4096 and 656
    tokens.
  - It also asserts the knob's parsing and resolution, the empty default map, and the KV class's `reference_kv` /
    `attention` / `_record_split`.
- **Driver** (`p143a2_drive.sh`): P143's, by counted edits. It stages every piece pinned in `staged-p143a2.sha256`:
  - P115 Phase D's served pieces at `staged-d.sha256`'s bytes;
  - the premise tests at P136's bytes;
  - the hf token, as Phase D's driver does.
- **Receipts:** `kernel.json`, `arm_{S0a,S1a,S1b,S0b}.json`, `sane_{off,on}.json`, `verdict_p143a2.json`,
  `summary.txt`, `forensics.txt`, `versions.txt` and `logs/`. The checkpoint, the arena and the reference log-probs stay
  on the box.
- **Strikes:** the lane's strike count carries over (1, from `p143-5090-1`). One more paid harness failure stops the lane
  for a simpler design.
