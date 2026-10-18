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
