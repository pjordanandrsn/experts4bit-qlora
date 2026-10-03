# P99 — Localise P98's replay fault (#913): the same serving-stack graph run repeated, beside arms that take away the hybrid state, K25, and all but one bucket, on one RTX 5090 (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#913 (the fault); lane register #564. Follows P98 (#912, VOID).

**Why.**
- In P98 (`bench/p98/RESULTS-p98.md`), Qwen3.6-35B-A3B served by `serve_paged.build_engine` with
  `E4B_PAGED_GRAPHS=1` captured every decode bucket (1–16).
- A replay then hit a device-side assert: `indexSelectSmallIndex: srcIndex < srcSelectDimSize`, an `index_select` whose
  index exceeds its source's first dimension.
- The same padded steps run eagerly (`capture=False`) were fine, 209 of them across all buckets. The fault is
  replay-only.
- Three candidate causes, none confirmed:
  1. **The hybrid's per-slot state under replay** (#907 / #908).
  2. **K25** (the select-tree small-M kernel), the NF4 default for T > 1 rows since 0.40.0. Bucketed graphs with it on
     a real NF4 MoE had not run together before P98.
  3. **A buffer moved after capture**: a later bucket's capture warm-up, or a prefill between replays, reallocates
     something an earlier graph captured by address.
- **This lane** answers three yes/no questions on one box, after one bake per model. No code changes until it reads.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- receipts and ledger rows;
- proven teardown.

The guard is 1 h, so there is no proving rental. The premise runs first on the box, before anything is fetched.

## The lane

- **Box and software.** P98's: one RTX 5090.
  - e4b at the launch commit; grouped-nf4-gemm `34da93d`; transformers 5.17.0.
  - The engine is built by `serve_paged.build_engine` (all-VRAM, the hybrid expert tier, the batched lane's device
    grouping, the fp8 KV), from NF4 arenas baked on the box by P98's `p98_bake.py`.
- **The measurement** is P98's `p98_box.py` at its registered bytes: the W16 workload (16 staggered requests) and W1
  (one request). It runs through `p99_box.py`, which prints and flushes one `P99_STEP call=<n> rows=<r>` line before
  every decode call. When a replay faults, the process aborts without a record, and its last line names the step and
  bucket.
- **Seven arms, each a fresh engine in its own process, in this order:**
  - **d0g:** P98's arm g repeated (Qwen3.6 @`995ad96`, buckets 1–16, K25 at its default). The fault must reproduce.
  - **d1g / d1e:** OLMoE-1B-7B-0924-Instruct @`7f1c97f`, which has no linear-attention state. Graphs, and their
    padded-eager oracle.
  - **d2g / d2e:** Qwen3.6 with `E4B_NF4_GROUPED_SMALLM=0` (K25 off; the served NF4 M-tile instead). Graphs, and
    oracle.
  - **d3g / d3e:** Qwen3.6 with the single bucket 16 (every step padded to 16 rows). Graphs, and oracle.
- **The premise** (rc 25): P98's three hybrid GPU test files pass on the card, 4 tests, none skipped.
- **The order:** fetch Qwen3.6, then OLMoE; bake both; run the arms above; reduce.

**The reducer** (`p99_reduce.py`, 11-case self-test). Each arm is classified:
- **ran:** it wrote a record;
- **faulted:** no record, and the log holds `device-side assert triggered`;
- **error:** anything else.

- **VOID** if any of these holds:
  - d0g did not fault (then a clean arm says nothing);
  - an oracle arm (d1e, d2e, d3e) did not run;
  - any arm ended in **error**;
  - a record is off the registered shape or model, or ran a rehearsal knob.
- **LOCALISED** otherwise, with three answers:
  - **hybrid_state_necessary** = d1g did not fault;
  - **k25_necessary** = d2g did not fault;
  - **multiple_buckets_necessary** = d3g did not fault.
- **Reported beside them:**
  - every faulted arm's last step and bucket;
  - **SILENT MISMATCH** for any graph arm that ran but whose W16 or W1 tokens differ from its oracle's. That would be a
    correctness finding of its own.

**The registered consequence.**
- **LOCALISED:** the fix is written against the necessary causes, with a GPU test that reproduces the fault on its own
  first. P98's question is then asked again in a new lane.
- **VOID:** #913 records the reading, and nothing changes.

## Predictions (written before the data)

- **d0g faults again**, so the lane reads LOCALISED.
- **k25_necessary: yes.** With K25 off, Qwen3.6's replays do not fault. K25's path under bucketed graphs is the newest
  untested combination here, and it is the only one of the three that P98's dense-hybrid premise test does not touch.
- **hybrid_state_necessary: no.** OLMoE faults too, because the fault lives in the shared expert path.
- **multiple_buckets_necessary:** no prediction.
- **No silent mismatch:** every graph arm that runs decodes exactly as its oracle.

## Box and cost

- **`p99-5090-<n>`:** one RTX 5090 with ≥ 200 GB of disk (the runner refuses below 170 GB). **Guard 1 h at
  ≤ $0.75/h (≤ $0.75).** Estimated:
  - install and premise ~4 min;
  - fetch ~7 (72 + 14 GB);
  - bakes ~3;
  - seven arms ~1–2 each (build ~30 s; a fault ends an arm early).
- **Lane ceiling $1.50; hard stop $2.00.**

## Rehearsal

None on the A2000: every arm needs the fp8 kernel (sm_89+), and Qwen3.6's all-VRAM placement needs about 23 GB. On Qwen3.6,
every piece P99 runs has already run on a 5090 in P98's `p98-5090-2`: the install, the premise, the fetch, the bake,
`p98_box.py`'s arms g and e, and the crash itself.

OLMoE through `p98_bake.py` and `build_engine` in this harness is new. Its checkpoint and `k8_bake.py`'s steps ran in
P96; the bake differs only by the pinned revision. If an OLMoE arm fails for any reason other than the device-side
assert, the reducer reads it as **error**, and the lane is VOID rather than wrongly answered.

The new pieces are the per-step log line (`p99_box.py`, a class-level patch of `DecodeTimer.__call__`), the arm
configurations (an environment variable and a flag of `p98_box.py`'s), and the reducer. CI covers them: the reducer's
self-test, the staged pin and the driver's dry run.

Amendments, dated, go below this line before any data is read.

### Amendment 1 (2026-10-03, before the launch; no P99 data exists)

- **Answer 1 reads as "Qwen3.6-specific", not "the hybrid state".** Arm d1 swaps Qwen3.6 for OLMoE-1B-7B, which
  removes more than the per-slot linear state:
  - 64 routed experts instead of 256;
  - no shared expert;
  - a different attention geometry (16 KV heads × 128 against 2 × 256);
  - a full-size KV pool instead of the compact 10-layer one.

  A d1g that does not fault therefore says that something particular to Qwen3.6 is necessary for the fault, not that
  the linear state is. The reducer's key keeps its name, `hybrid_state_necessary`, because its bytes are staged and
  pinned. The results report it as **qwen36_specific**. A conclusion about the hybrid state needs a further arm.
  `tests/test_hybrid_decode_graphs_gpu.py`, which passed on sm_120 in P98's premise, already shows the per-slot state
  replaying exactly under bucketed fp8 graphs on a dense hybrid with no MoE.
- **One mechanism ruled out before the launch, at $0, on the NAS A2000.** The fused tile-table builder
  (`int4_b32.build_group_tiles_fused`) feeds `sorted_ids = local_ids.index_select(0, order)`: a small-index
  `index_select` over the routed rows, the shape of P98's assert. If its kernel left part of `order` unwritten,
  an eager step could reuse a block that held a valid `order` and pass, while a graph's never-written buffer would
  index past the routed rows.
  - Checked on 64, 128 and 256 experts, 1–16 rows × top-8, 20 random routings each, with the caching allocator handed
    a poisoned block (`0x7FFFFFFF`) before every call.
  - Every output (`order` a full permutation of the rows, `counts` the per-expert histogram, and the three tile
    tables) equals the chained builder's: 0 of 300 calls differ.

  The builder is not the cause.
- **A second mechanism ruled out, at $0 on the A2000: K25 with device grouping under capture.** The expert engine's
  `_fused_over_stack` ran with `E4B_NF4_GROUPED_SMALLM=1`, device grouping, and the lean glue on (the serving
  default) and off. Its shapes were Qwen3.6's experts exactly: hidden 2048, expert intermediate 512, 256 experts,
  top-8; plus 40 experts as a control.
  - One graph was captured per row count (2, 4, 8, 16, ascending, as `enable_decode_graphs` captures). They were then
    replayed in a different order on new inputs and routings, 5 per step of the order 16, 8, 2, 4, 16, 2.
  - Every replay equals the eager call on the same inputs, bit for bit: 0 of 120 replays differ.

  K25 and the device tile table do not fault under capture on their own.
- **Prediction revised accordingly** (from the $0 evidence above, before any P99 data): **k25_necessary: no.** With
  K25 off, Qwen3.6's replays are predicted to fault as well. The remaining suspects are what the A2000 checks did not
  include:
  - the hybrid expert tier's glue around `_fused_over_stack` (routing, combine, the shared expert);
  - the per-slot linear state inside the full model;
  - the compact pool's paged attention;
  - a buffer moved between bucket captures or by a prefill between replays.

  The other predictions stand.
