# B511 — read: the bucketed decode-graph replay decodes exactly as the padded eager step on an RTX 5090

Pre-registration: [`PREREG-b511.md`](PREREG-b511.md) (merged #758, `c1f7e27`, before the run). Issue #511; code under
test: #757 at `95983b5`. Run `b511-5090-1`, 2026-09-28 20:30–20:37Z.

## Verdict: P1, P2 and P3 all held

| prediction | registered reading | observed |
|---|---|---|
| **P1** | the replay test passes on the 5090, all three buckets captured | `test_a_bucket_replay_decodes_exactly_as_the_padded_eager_step` **PASSED** (5.9 s) |
| **P2** | the same test fails under `mut_b511.py` | **FAILED**: `replay != padded eager` |
| **P3** | every other arm-A test passes, no card-reason skip | arm A **102 passed, 0 skipped, 0 failed** |

By the decision rule, #757 is verified and merges under the usual self-review.

P1's test asserts `graph_status == {1: "graph", 2: "graph", 4: "graph"}` and that every bucket replayed. It also
asserts that bucket 4 carried padding, and that the replayed streams equal the padded-eager streams token for token.
The active set shrinks 4 → 3 → 2 → 1 as requests finish, and a fifth request lands in a recycled slot. The printed
unpadded-eager agreement is captured by pytest (no `-s`), so it is not in the log. It was never part of the rule.

P2's mutation removes `Fp8PagedKV.kernel_args`'s bound-bucket selector. Attention then reads the per-tuple cache,
which a captured graph bakes at capture time (on the scratch slots). The replay diverges from the oracle, so the test
catches the defect the design exists to prevent.

## What arm A adds beyond #511

It is the first sm_120 run of the GPU tests merged on 2026-09-28, which until now had run only on the sm_86 A2000:
#753's quantised-MoE capture (the pipelined engine replays as eager; `enable_fast` refuses by name), #754's attention
pack on the real kernels (bitwise at 1, 4 and 24 rows), #750's direct cold landing (bitwise with copy and DRAM), and
#748's router tests. All passed.

## Box and cost

RTX 5090 (compute capability 12.0, driver 610.57.04, 32 607 MiB), Vast verified/secure, instance `53249558`,
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. torch 2.8.0+cu128 and triton 3.4.0 from the image (held by
`constraints.txt`), grouped-nf4-gemm 0.33.5 (`fb15cf5`), transformers 5.17.0, bitsandbytes 0.50.2. The tripwires held:
the import resolved to the clone, and torch was unchanged. **Cost $0.0666** over a 447 s lifetime, estimate $0.75.
Teardown proven: destroy HTTP 200, the instance absent from a fresh listing. The launcher receipt, teardown proof and
ledger row are in the private receipt store at `receipts/experts4bit-qlora/2026-09-28/b511-5090-1/` (commit `21b2dba`).
The lane spent $0.0666 of its $1.50 ceiling.

## Files

`receipts/b511-5090-1/`: `armA.{xml,txt}`, `armM.{xml,txt}`, `summary.txt`, `versions.txt`, `forensics.txt`,
`constraints.txt`, all as fetched from the box.
