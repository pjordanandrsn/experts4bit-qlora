# TC1 amendment 78: the field recipe on a 24 GiB RTX 4090, the lane's verdict

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 78 (merged as experts4bit-qlora#2093; launch commit
`576d99e9`, grouped-nf4-gemm `efdda943`). There were two started boxes, `tc1-4090-1` and `tc1-4090-2`, both RTX 4090s, and both read.
Lane spend: USD 1.621 actual, against the registered worst of USD 13.94.

The block below is `bench/tc1/tc1_reduce.py --pos78-lane` over the two runs' `tc1/` directories, as the private receipt store holds
them: each with its launcher receipt beside it, which is where the reducer reads the machine id.
- `tc1_reduce.py` is byte-identical between the launch commit and main.
- The maintainer re-derived the verdict from the store, and it matched.
- Each box's own RESULTS rebuilds from this repository's copy ([`receipts/tc1-4090-1/`](receipts/tc1-4090-1/README.md),
  [`receipts/tc1-4090-2/`](receipts/tc1-4090-2/README.md)). There each draw reads "VOID: the machine id is unread", because the box writes
  its RESULTS before the launcher's receipt exists.


## Amendment 78's verdict: the field recipe on a 24 GiB RTX 4090 (pb1cl256, the started draws in launch order)

| draw | state | role | steps | usable - max allocated | usable - max reserved | total - device used | held-out at 60 |
|---|---|---|---|---|---|---|---|
| 1 | COMPLETE | pairs | 60 | 1,260 MiB (step 32) | 586 MiB (step 29) | 586 MiB (step 29) | 0.75444 |
| 2 | COMPLETE | pairs | 60 | 1,258 MiB (step 22) | 686 MiB (step 34) | 686 MiB (step 34) | 0.75667 |

- **FIT:** draws 1 and 2 trained all 60 steps with no out-of-memory error on two different machines.
- **Two different machines:** yes (the ids stay in the private store).
- **The least headrooms over the draws the verdict used:** usable - max allocated 1,258 MiB (step 22); usable - max reserved 586 MiB (step 29); total - device used 586 MiB (step 29).
- **Scope:** Qwen3-30B-A3B @ ad44e777, the alpaca tokens (tc1-5090-155's), seq 2048, micro-batch 2, accum 4, r 16, the pb1cl256 levers, a 24 GiB card by its listing and its memory.total. The longest padded micro-batch is 565 tokens (step 22); datasets with longer batches are not covered.
- **Held-out** is reported beside tc1-5090-155's cd1cl1 (0.75489), which is not like-for-like; no quality claim follows from this amendment.
