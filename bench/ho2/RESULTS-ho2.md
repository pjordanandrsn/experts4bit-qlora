# HO2 read: the planner's first held-out training rows

HO2 (`PREREG-ho2.md`) registered two rented boxes that measure rows for the held-out side of Loggetta's frozen planner split. Each box
ran four `loggetta train` rows: `LiquidAI/LFM2-8B-A1B` and `baidu/ERNIE-4.5-21B-A3B-PT`, each at 1,024 and 2,048 tokens. Every row used
the planner's own plan, with no `--fix` and no `--observations`, at the registered pins:
- experts4bit-qlora `f6e66fd6` (0.56.1);
- grouped-nf4-gemm `a83f6bb3` (0.49.0);
- Loggetta `9348f8b8` (0.6.2);
- transformers 5.17.0, bitsandbytes 0.50.2, torch 2.8.0+cu128.

## The read

**COVERED, 8 of 8 rows, on both cards.** The registration predicted exactly this. On two MoE families no planner constant has seen,
the plan's total bounds the sampled driver peak, and its allocator estimate bounds the allocated peak, on every row.

| box | card | row | loss branch (inferred, not counted) | placement | plan total GB | driver peak GB | driver / plan | allocator estimate GB | allocated peak GB | allocated / estimate | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ho2-5090-1` | RTX 5090 (cc 12.0) | lfm2_1024 | stock | device | 9.768 | 8.540 | 0.8742 | 7.693 | 7.338 | 0.9539 | COVERED |
| | | lfm2_2048 | chunked (at the gate) | device | 9.386 | 7.726 | 0.8231 | 7.375 | 6.750 | 0.9153 | COVERED |
| | | ernie_1024 | stock | device | 21.856 | 18.382 | 0.8410 | 17.766 | 17.100 | 0.9625 | COVERED |
| | | ernie_2048 | chunked | device | 20.886 | 17.067 | 0.8171 | 16.958 | 16.135 | 0.9515 | COVERED |
| `ho2-4090-2` | RTX 4090 (cc 8.9) | lfm2_1024 | stock | device | 9.768 | 8.376 | 0.8575 | 7.693 | 7.338 | 0.9539 | COVERED |
| | | lfm2_2048 | chunked (at the gate) | device | 9.386 | 7.562 | 0.8057 | 7.375 | 6.757 | 0.9162 | COVERED |
| | | ernie_1024 | stock | device | 21.856 | 18.218 | 0.8335 | 17.766 | 17.100 | 0.9625 | COVERED |
| | | ernie_2048 | chunked | device | 20.886 | 16.922 | 0.8102 | 16.958 | 16.141 | 0.9518 | COVERED |

GB = 1e9 bytes. Every number is from the boxes' reducer outputs (`receipts/<box>/ho2/run/verdict_ho2.json`). The plan totals and the
allocator estimates are byte-identical across the two cards.

## Two lines that bind every reading of these rows

**The 2,048-token rows ran a different loss branch, so these rows are not a length comparison.**
- The planner chose one setup per family for both lengths: bf16 adapters r 8, the grouped_nf4 kernel, experts on device, micro-batch
  2, grad-accum 1, 20 steps.
- What differs is experts4bit-qlora's chunked-LM-loss auto gate: the chunked loss runs once the stock fp32 logits, tokens × vocabulary
  × 4, reach 1 GiB.
  - The 1,024-token rows (2,048 tokens a micro-batch) sit below the gate and run the stock loss.
  - The 2,048-token rows (4,096 a micro-batch) run the chunked loss.
  - LFM2's sits exactly at the gate: 4,096 × 65,536 × 4 = 1,073,741,824. Its head is 65,536 wide (`lm_head.out_features`, the
    width both the plan and the gate read). The 64,400 in the receipt is the tokenizer's vocabulary.
- The plan prices both branches, which is why both the plan and the measured peaks fall at 2,048 tokens.
- The branch each row ran is **inferred, not counted**. These receipts do not record the chunked-loss counters, so the branch is read
  from the plan and from the measured drop. Loggetta's receipts now record the counters (loggetta#94).

**HO2 tested the device placement only.**
- The registration expected ERNIE at 2,048 tokens to plan host offload on the RTX 4090, "so both placements are exercised". **That did
  not hold.**
- At this experts4bit-qlora, ERNIE's 2,048-token plan totals 20.886 GB. Every row on both cards planned and ran
  `expert_residency=device`.
- Host offload is untested by HO2. The registration's expectation came from an earlier planning read. Why this plan prices lower
  is a hypothesis (the chunked-loss branch at 4,096 tokens a micro-batch), not a measured finding.

## Attempts and cost

- **`ho2-5090-1`:** OK, USD 0.28 (worst USD 3.20). The box ran 920 s from install to the reduce.
- **`ho2-4090-1`:** REFUSED before any instance existed, USD 0. The provider refused the pinned offer at create while it was still
  listed. The run id is spent, and its receipt is only in the private store.
- **`ho2-4090-2`:** the registered spare, which runs only after a refusal that is not the lane's own. OK, USD 0.461 (worst USD 2.60).
  The box ran 1,427 s.
- **Total:** USD 0.741 against the registered USD 5.80 worst.

The maintainer re-derived both verdicts from the private receipt store. `ho2_reduce.py` re-run on each box's run directory reproduces
its `verdict_ho2.json` exactly.

## What the rows are for

Every valid row goes to the held-out side of a new version of Loggetta's planner split (`SPLIT-v2.json`, under its own review): two
rows per family × card, the minimum for a band. The band itself is Loggetta's L2. HO2 computes none (`band_computed: false`).

## Rebuild

Each box's directory under `receipts/` holds:
- `ho2/`: the box's files that carry no host identifier, byte-identical;
- `public/`: the four row receipts with their two GPU-UUID values set to `null` (`ho2_public.py derive`);
- `SHA256SUMS`: every file of the box's receipt, with its sha256 in the private store;
- `STORE-ONLY.sha256`: the files kept only there.

```
python bench/ho2/ho2_public.py rebuild bench/ho2/receipts/ho2-5090-1 --out /tmp/verdict.json
```

This runs this checkout's `ho2_reduce.py` on the committed files with the derived receipts laid in. It rebuilds the box's
`verdict_ho2.json` byte for byte. `tests/test_ho2_public.py` checks both rebuilds, the sums, and that no committed file names the host.
