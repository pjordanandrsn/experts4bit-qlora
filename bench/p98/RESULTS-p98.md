# P98 — results: **VOID**. Every bucket of Qwen3.6-35B-A3B's bucketed decode captured, and the first graph replays then hit a device-side assert (`index_select`: index ≥ the source's size) during W16; arm g wrote no record. The padded-eager and plain-eager arms ran

Registration: `bench/p98/PREREG-p98.md` (#911, `a08df83`). Issue: #564. Code under test: #907, #908; e4b at `a08df83`,
grouped-nf4-gemm at `34da93d`.

**Verdict by `p98_reduce.py`: `VOID`.** The registered rule voids a reading with a missing record, and arm g has none:
its process aborted (rc 134). Nothing in this lane reads as SUPPORTED or NOT_SUPPORTED. What the box did show is
recorded below, and the cause is localised in a separate lane before any change.

## What happened (`p98-5090-2`)

- **The premise held** on the card: the three hybrid GPU test files, 4 passed, none skipped. That includes the dense
  Qwen3.5 hybrid's bucket graphs, replayed exactly as its padded eager step.
- **The bake held:** 40 layers × 256 experts, a 16.88 GiB NF4 snapshot, loaded commit `995ad96`.
- **Arm g** (graphs):
  - `build_engine` reported every bucket captured (`{"1", "2", "4", "8", "16"}: "graph"`), with device grouping on;
  - during W16, a replayed step raised `torch.AcceleratorError: CUDA error: device-side assert triggered`, surfacing at
    `buf["tok"][:n].tolist()` in `_run_decode_bucketed`;
  - the kernel's message: `Indexing.cu:1478: indexSelectSmallIndex ... Assertion 'srcIndex < srcSelectDimSize'
    failed`, an `index_select` whose index exceeds its source's first dimension;
  - the process aborted (rc 134) and wrote no record (`logs/arm_g.log`).
- **Arm e** (the same buckets and grouping, every step eager on the same padded layout) ran both workloads to their
  budgets. Its `graph_stats` show 209 eager bucket steps, W16's 114 and W1's 95, across all five buckets, with 168 padding rows.
- **Arm p** (plain eager) ran both workloads to their budgets.

So every operation arm g captured also runs eagerly, on the same padded layout, through the same per-slot selector
path. The fault appears only under replay.

## Observations from the eager arms (reported; not a reading of this lane's question)

| | arm e (padded eager) | arm p (plain eager) |
|---|---:|---:|
| W16 decode, tok/s (111 timed steps, 1,220 rows) | 128.3 | 107.3 |
| W1 decode, tok/s (92 timed steps) | 11.98 | 11.46 |
| step time at 1 row / 16 rows, ms | 82.4 / 85.7 | 90.9 / 103.2 |
| peak GPU memory | 23.71 GB | 22.74 GB |
| build | 30.8 s | 30.9 s |

**A decode step costs about the same at 1 row as at 16.** Eager hybrid decode is bound by launch and Python overhead,
not arithmetic: the torch Gated DeltaNet path issues many small kernels in each of 30 layers. That is the cost a
captured graph removes, and the reason the replay fault is worth localising.

The two eager arms agree on W1's 96 tokens exactly. On W16 they agree on 67.2 % of positions: padded and unpadded
steps run different row counts per GEMM, and free-running greedy decode diverges for good after one flipped argmax.

## What the replay fault could be (to be localised, not concluded here)

- **Not shown to be hybrid-specific.** K25 (the select-tree small-M kernel) became the default for T > 1 NF4 rows in
  0.40.0 (#880), after the serving lanes that validated bucketed graphs on 128-expert models. Bucketed graphs with K25
  on a real NF4 MoE have, as far as the repo's tests show, not run together. `tests/test_decode_graph_buckets.py` uses
  a dense model.
- **Hybrid-specific candidates:** the per-slot state's gather through the bucket selector under replay, and the warm-up
  and freeze interacting with later prefills.
- **Expert-engine candidates:** the device tile table on 256 experts under replay, or a buffer a later capture's
  warm-up reallocates after an earlier bucket's graph captured its address.

The next lane runs arms that separate these on one box, after one bake: OLMoE (no linear state) through the same graph
stack; Qwen3.6 with K25 off; and Qwen3.6 with a single bucket.

## Box and cost

- **`p98-prove-1`:** OK, PROVED, **$0.018**; destroyed at 2026-10-03T00:06:41Z.
- **`p98-5090-1`:** REFUSED at **$0**. The controller's PUT to Vast timed out in the TLS handshake before any instance
  existed.
- **`p98-5090-2`:** HARNESS_ERROR (lane rc ≠ 0), **$0.135**, 859 s. One RTX 5090 (sm_120, driver 580.95.05), Vast
  machine 142284, instance 53938995, AMD EPYC 7663. Destroyed at 2026-10-03T00:23:29Z, instance absent.
- **The lane: $0.153** of a $2.50 ceiling.

Receipts (`receipts/p98-5090-2/`): the two eager records, `verdict.json`, `summary.txt`, `forensics.txt`,
`versions.txt`, the premise, fetch, bake and arm logs (arm g's crash included), `work/bake.json`, the teardown proof,
and `SHA256SUMS` over every file the box wrote. The launcher's receipts and ledger rows are in the receipt store
(adertha-receipts `ac6e0d3`, `dbbc894`, `2f16f03`).
