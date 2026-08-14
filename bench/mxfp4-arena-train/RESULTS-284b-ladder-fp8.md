# The ladder with the dense-side fix — P5 holds, the wall moved, the claim still fails

Graded against `PREREG.md` amendment 5, stamped before this ran. Method held
constant: the same five rungs, same order, one process per rung, so any difference
from `RESULTS-284b-ladder.md` is attributable to the single change.

**The enablement claim is still not established.** All five rungs OOM on the same
real 23.5 GiB A5000, no software cap.

## P5 — confirmed, and the wall moved exactly one allocation

| | before the fix | after |
|---|---|---|
| failing call | `fp8_blocks.dequantize_fp8_blocks` | `engines/offload.py:486 _alloc_dest` (4 of 5 rungs) |
| allocation at failure | 128 MiB | **1024 MiB** |

Amendment 5 predicted only that the dense transient would shrink, and said in
advance that whether the ladder then fits would be **reported, not forecast** —
"predicting a pass here would be guessing dressed as a hypothesis". It did shrink,
and the OOM moved one allocation along. Had a pass been predicted, this would read
as a failed prediction rather than as what it is: the fix worked and revealed the
next constraint.

Rung 1 (1×128) fails slightly differently — in `formats/mxfp4.py:95
dequantize_mxfp4` for **16 MiB**. It got *further* and ran out on a much smaller
request, which is the same underlying pressure seen from one step later.

## The new binding constraint is the tier's own staged stack, and it is documented

That 1024 MiB request is one layer's `gate_up_proj` destination at full `[E, …]`
shape:

| staged tensor (per MoE layer, E=256) | size |
|---|---|
| gate_up blocks `[2048, 2048]` | **1.00 GiB** |
| down blocks `[4096, 1024]` | **1.00 GiB** |
| gate_up scales `[2048, 128]` | 0.06 GiB |
| down scales `[4096, 64]` | 0.06 GiB |
| **total resident per layer** | **2.12 GiB** |

`engines/nvme_train.py` states this limitation in its own module docstring, before
any of this was run:

> **What this does NOT bound: VRAM.** The staged destination keeps the full
> `[E, ...]` shape so every consumer keeps indexing by global expert id … at K3's
> 896 experts/layer that is ~15.7 GB, and fitting that on a small card needs a
> *compacted* stack (`[R, ...]` plus an id remap), which splits the kernel's id
> space from the adapter's. **Deliberately not done here:** it changes the kernel
> contract, and this change does not.

So the run is now stopped by a **declared design decision, not a defect**. That is
a materially better place to be stopped than where this started, and it draws a
clean boundary: everything this branch built works, and what remains is a known
trade-off in a dependency's contract.

## What did work, at every rung

| gate | value |
|---|---|
| G1 modules patched | **43 / 43** |
| G4 (on the lane it claims) | **true** — 43/43 MXFP4-flagged, 43/43 on the MXFP4 dequantize override, `enable_fast_train` returned 0 |
| load | 10.63 GiB peak |
| download / bake | 84 s / 57 s |

The MXFP4 expert path — resolve, stage, declare, decode, forward — runs end to end
against a real 284B checkpoint's own bytes. It is the surrounding VRAM budget that
does not fit.

## Verdicts, cumulative

| prediction | verdict |
|---|---|
| **P1** — STOCK fails | **CONFIRMED** twice, on memory both times (23.56 GiB card, and again uncapped on a 44.39 GiB card) |
| **P2a/P2b** — a rung fits in 24 GiB | **REFUTED**; the largest rung that fits is none |
| **P3** — bytes unchanged | **UNGRADED**; no training step completed |
| **P4** — it learns | **UNGRADED** |
| **P5** — dense transient shrinks | **CONFIRMED**; the wall moved from the dense path to the staged stack |

## Next step, stated rather than attempted

A **compacted staged stack** (`[R, …]` plus an id remap) would cut 2.12 GiB/layer
to roughly `routed_experts × 8.3 MiB`. It is not attempted here because it splits
the kernel's expert-id space from the adapter's — a change to
`grouped-nf4-gemm`'s contract, not a fix to this branch, and well outside what
this work set out to do. Any attempt needs its own prereg.

Nothing here forecasts whether that change would make the model fit. The dense
side alone was 10.63 GiB at load on a 23.5 GiB card.

## Cost and teardown

~$0.10 this leg; **~$0.85** across the whole session against the $35/job cap.
Pod verified gone (`404`), actors cleared, account reconciled by id.

The A5000 reached its endpoint at **150 s** — a third pod that the pre-correction
120 s wedge threshold would have destroyed.
