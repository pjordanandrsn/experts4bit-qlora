# The registered batch ladder — P2 refuted, and the reason is not the expert tier

Graded against `PREREG.md` amendment 4, which registered the ladder before any of
this was measured. **The enablement claim is not established.**

## The ladder

Real **NVIDIA RTX A5000, 23.5 GiB, no software cap** — a genuine 24 GB card, so
this is stronger evidence than the amendment-3 attempt, which ran on a 44.39 GiB
L40S capped to 24 GiB in software.

| rung | tokens × seqlen | total tokens | outcome |
|---|---|---|---|
| 1 **(primary)** | 1 × 128 | 128 | **OOM** |
| 2 | 1 × 256 | 256 | OOM |
| 3 | 2 × 256 | 512 | OOM |
| 4 | 4 × 256 | 1024 | OOM |
| 5 | 8 × 64 | 512 | OOM |

**P2a — refuted.** The primary rung does not fit in 24 GiB.
**P2b — the largest rung that fits is *none*.** Amendment 4 registered that
outcome in advance as a refutation of the enablement claim, and that is what it is.
**P3, P4 — still ungraded.** No training step completed.

## Where it actually fails, which is the useful part

Not in the expert path:

```
formats/fp8_blocks.py:99  in dequantize_fp8_blocks
    return (weight.float() * s).to(dtype)
torch.OutOfMemoryError: Tried to allocate 128.00 MiB.
GPU 0 … 23.55 GiB total, 51.31 MiB free, 22.88 GiB allocated by PyTorch
```

That is **DeepSeek-V4's dense half** — the 365 block-scaled FP8 linears the loader
reports as "dense side stays FP8-resident, decoded on use". The weight is
dequantized FP8 → bf16 at the moment it is used, and on a card already at 22.88
GiB there is no room for the temporary.

**This is why every rung fails identically.** A dense weight's dequantization is
batch-independent, so shrinking the batch cannot help: rung 1 (128 tokens) and
rung 4 (1024 tokens) die in the same call trying to allocate the same 128 MiB.
The ladder was worth running precisely because it *established* that — a single
failing configuration could not have distinguished "too much activation memory"
from "a fixed cost that no batch size reaches".

**The arena machinery under test worked at every rung:**

| gate | value |
|---|---|
| G1 modules patched | **43 / 43** |
| G4 (on the lane it claims) | **true** — 43/43 MXFP4-flagged, 43/43 on the MXFP4 dequantize override, `enable_fast_train` returned 0 |
| load | 23.9 s, **10.63 GiB** peak |
| download / bake | 72 s / 69 s (149 GB, 138 GiB arena) |
| `hot_rows` floor | **256** (correctly clamped to `num_experts`) |

So the tier does what this branch claims: it stages a 284B model's experts from
their own MXFP4 bytes and reaches a training step. The model still does not fit on
one 24 GB card, **for a reason outside the tier** — and that distinction is the
result, not a hedge.

## What this does NOT say

- It does not say the MXFP4 forward is wrong. That is separately established on
  two cards (`RESULTS-forward-parity.md`).
- It does not say a 24 GB card is impossible for this model. It says that *with
  V4's dense half kept FP8-resident and decoded on use*, the dense path alone
  exhausts the card. Whether a dense-side change fixes that is a different
  experiment under a different prereg, and no number here forecasts it.
- P2's original **4–8 GB band for the staged stacks** is still ungraded as such:
  the run reports total peak, and the per-layer staged stack is
  `256 × 13,369,344 B = 3.26 GiB` by arithmetic, not by measurement.

## A harness defect that survived its own fix

Amendment 4 fixed the `hot_rows` sizing after the previous receipt reported 18.3
MB of "usable" host memory. **The fix did not run here either:** this pod is
**cgroup v1**, and the code reads `/sys/fs/cgroup/memory.max`, which is v2-only.

The receipt says so outright —
`{"error": "No such file or directory: '/sys/fs/cgroup/memory.max'", "rejected": "cgroup usable=None; falling back to 24 GB"}`
— and that *is* the improvement over last time, when a bad reading was used
silently. But host memory is still not being measured; it needs the v1 path
(`memory/memory.limit_in_bytes` minus `memory.stat`'s cache). The floor fix did
land: 256 rows, not the 3072 that implied 38.2 GiB of pinned DRAM.

Neither figure touched VRAM, so neither affects the verdict above.

## Cost and teardown

This leg: one A5000 at $0.27/hr for ~15 min, **~$0.10**. Session total across all
legs **~$0.75** against the $35/job cap. Pod verified gone (`404`), actors
cleared, account reconciled — the only remaining pod belongs to another lane and
was never touched.

## The wedge correction, recorded here because it changed this run

This is the first A5000 today that was not destroyed by my own launcher. It
reached a live endpoint at **210 s**; the previous wedge threshold was **120 s**.
Four earlier A5000s and at least one A40 were recorded as "wedged" and torn down
when they were most likely healthy and still starting. A controlled A/B/C probe on
one A40 measured healthy containers first reporting at **167–178 s**, and
exonerated the request body (the historically "proven" body wedged; mine came up).
Threshold raised to 300 s. The shape-correlated wedge table in my notes was
substantially an artifact of that check and has been corrected.
