# P99 — results: **LOCALISED**. P98's fault repeated on Qwen3.6 at the first bucket-1 replay, and again with K25 off; it did not occur with a single bucket, or on OLMoE through the same graph stack. Answers: qwen36_specific yes, k25_necessary no, multiple_buckets_necessary yes

Registration: `bench/p99/PREREG-p99.md` (#914, `9ec097e`; amendment 1 in #915, `daef38c`, before the launch). Issues:
#913 and #564. Code under test: e4b at `daef38c`, which predates #918's fix, and grouped-nf4-gemm at `34da93d`.

**Verdict by `p99_reduce.py`: `LOCALISED`.**
- The reducer's three answers:
  - **qwen36_specific** (key `hybrid_state_necessary`; amendment 1 renames the answer) = **true**;
  - **k25_necessary = false**;
  - **multiple_buckets_necessary = true**.
- No silent mismatch: every graph arm that ran decoded exactly as its padded-eager oracle.

## The arms (`p99-5090-1`)

| arm | configuration | result |
|---|---|---|
| d0g | P98's arm g, unchanged: Qwen3.6, buckets 1–16, K25 auto | **faulted** at decode call 110 with 1 row (bucket 1), rc 134 |
| d1g | OLMoE-1B-7B through the same graph stack | ran; bucket 1 replayed 99 times; tokens equal d1e's |
| d1e | OLMoE, padded eager | ran |
| d2g | Qwen3.6 with K25 off (`E4B_NF4_GROUPED_SMALLM=0`), graphs | **faulted** at decode call 110 with 1 row (bucket 1), rc 134 |
| d2e | Qwen3.6 with K25 off, padded eager | ran |
| d3g | Qwen3.6, bucket 16 only, graphs | ran; 209 replays; tokens equal d3e's |
| d3e | Qwen3.6, bucket 16 only, padded eager | ran |

- **The fault is P98's.** It is the same kernel message (`Indexing.cu:1478: indexSelectSmallIndex ... Assertion
  'srcIndex < srcSelectDimSize' failed`), at the same place in W16. W16's calls 0–109 run between 2 and 16 rows;
  call 110 is the first step with one row left, so the first replay of bucket 1's graph. Both faulting arms died on it.
- **The premise and both bakes held:**
  - the three hybrid GPU test files: 4 passed, none skipped;
  - Qwen3.6: 40 layers × 256 experts, a 16.88 GiB snapshot, loaded commit `995ad96`;
  - OLMoE: 16 × 64, 3.38 GiB, `7f1c97f`.
- **OLMoE ran the path that faults on Qwen3.6:** the hybrid expert tier all in VRAM (masses 1.00 / 0.00 / 0.00), with
  device grouping on.

## Against the predictions

**Registered in `PREREG-p99.md`** (written before the cause was known):
- d0g faults again: **yes**.
- k25_necessary yes: **no** (d2g faulted).
- hybrid_state_necessary / qwen36_specific no, i.e. OLMoE faults too: **no** (d1g ran).
- multiple_buckets_necessary: no prediction.
- No silent mismatch: **yes**.

**Posted on #913 after the code reading, before any P99 data**
([comment](https://github.com/pjordanandrsn/experts4bit-qlora/issues/913#issuecomment-5964098327), 01:26:59Z; the
launch was at 02:19:41Z):
- d0g, d1g and d2g fault; d3g runs.
- Therefore k25_necessary no: **right**. multiple_buckets_necessary yes: **right**.
- qwen36_specific no: **wrong**. d1g did not fault.

## What it says about the cause

The cause posted on #913 and fixed by #918: the expert engine kept its row-to-token index (`rt`) in a single-entry
cache, so each bucket's eager capture warm-up freed the index an earlier bucket's graph still read.

**Consistent with it:**
- **The assert is the T == 1 route's gather.** One token's rows are gathered with `x.index_select(0, rt)`, so any `rt`
  value other than 0 is out of range. The fault fired at the first bucket-1 replay.
- **K25 is not involved.** K25 serves T > 1 only, and turning it off moved nothing: same step, same bucket.
- **A single bucket did not fault.** With K25 on, bucket 16's T = 16 graph uses the lean glue, which does not read `rt`.

**Not explained by it:**
- **OLMoE ran the same expert tier and did not fault.** Its bucket 1 replayed 99 times exactly.
- At T == 1, a freed index asserts only once its block has been handed out again and written with a nonzero value.
  `tests/test_rt_cache_graph_gpu.py` (#918) forces that reuse on a non-hybrid NF4 MoE stub and shows the stale read.
- What P99 cannot say is why Qwen3.6's run reached that state by its first bucket-1 replay while OLMoE's never did.
  Amendment 1 lists four ways the two models differ besides the linear state.
- So the answer "Qwen3.6-specific" stands as read. This lane does not attribute it to the hybrid state.

**The registered consequence (LOCALISED):** "the fix is written against the necessary causes, with a GPU test that
reproduces the fault on its own first. P98's question is then asked again in a new lane."

What happened against that:
- **The order differed.** The fix (#918) was written and merged before this lane read. It came from reading the code
  for state that outlives a capture, after three $0 A2000 exclusions.
- **It addresses the localised conditions:** bucket 1's graph and the earlier buckets' warm-ups.
- **The OLMoE answer** neither supports nor undercuts it.
- **The new lane is P101** (#919, `bench/p101/PREREG-p101.md`): P98's kit at its registered bytes, on code that
  includes #918.

## Observations from the arms that ran (reported; not a reading of this lane's question)

| arm | W16 tok/s (111 steps) | W1 tok/s | peak GPU memory | build |
|---|---:|---:|---:|---:|
| d1g OLMoE, graphs | 1515.2 | 254.6 | 4.98 GB | 11.8 s |
| d1e OLMoE, padded eager | 402.1 | 38.2 | 4.93 GB | 5.0 s |
| d2e Qwen3.6, K25 off, padded eager | 122.5 | 12.46 | 23.71 GB | 35.3 s |
| d3g Qwen3.6, bucket 16 only, graphs | 421.0 | 46.4 | 23.72 GB | 35.8 s |
| d3e Qwen3.6, bucket 16 only, padded eager | 115.6 | 10.84 | 23.71 GB | 38.6 s |

**d3g is the first Qwen3.6 decode under a replayed graph, but not P98's measurement.**
- Every step runs bucket 16, so W1's single request carries 15 padding rows. It is 3.6× its padded-eager arm on W16
  and 4.3× on W1.
- P101 measures all five buckets.

## Box and cost

- **`p99-5090-1`:** OK, LOCALISED, **$0.196** of a $0.75 guard, 1,247 s.
  - One RTX 5090 (sm_120, driver 580.95.05), Vast instance 53952525, on an AMD EPYC 7663.
  - Launched 2026-10-03T02:19:41Z by adertha `898c8f6`.
  - Destroyed at 02:40:27Z, instance absent.
- **One launch attempt before it was refused by the controller's own guard, at $0.**
  - At 02:06:49Z, the launch script's `grep` for the registered arm list was written inside a double-quoted `ssh`
    command, so the local shell expanded `${P99_ARMS:-…}` and the pattern could never match.
  - No instance was requested. The guard was corrected and checked before the relaunch.
- **The lane: $0.196** of a $1.50 ceiling.

Receipts (`receipts/p99-5090-1/`):
- the five arm records, `verdict.json`, `summary.txt`, `forensics.txt` and `versions.txt`;
- the premise, fetch, bake and arm logs, including both faulting arms;
- `work_qwen/bake.json` and `work_olmoe/bake.json`;
- the teardown proof;
- `SHA256SUMS` over every file the box wrote.

The launcher's receipt and ledger row are in the receipt store (adertha-receipts `da36945`).
