# Prefetching the next layer removes the stall and makes the step 14% slower

### 2026-08-13 · NVIDIA L40S (Ada sm_89), SECURE on-demand · torch 2.8.0+cu128 · e4b 0.17.5 + branch `feat/arena-prefetch` · raw [`prefetch-l40s.jsonl`](prefetch-l40s.jsonl)

**Evidence tier: `measured`.** Pre-registered in [`PREREG-prefetch.md`](PREREG-prefetch.md)
**before the pod was rented**, including the expected direction and magnitude.

**Outcome: the feature is not shipped.** The code stays on its branch, unmerged.

## Why it was tried

Arena staging runs in a forward pre-hook immediately before the layer computes, and
`ColdTier.ensure` returns only once the rows land. Screening put **~16% of the step**
blocked there, in ~47 misses. A worker that warms the next layer during the current one's
compute should hide that.

It does hide it. Measured on the same box, blocked time **~16% → ~4%** and main-thread
misses **47 → 0–1**. The stall is gone.

## And the step got slower anyway

Four arms, paired within each round, 5 scored rounds × 12 scored steps, `off256_self` being
the same configuration timed twice per round:

| arm | `hot_rows` | prefetch | step ratio vs `off256` | verdict |
|---|---|---|---|---|
| `off128` | 128 (routing floor) | off | 0.988 [0.948–1.118] | overlaps control |
| **`on256`** | 256 | **on** | **1.144 [1.096–1.193]** | **RESOLVED — slower** |
| `off256_self` | 256 | off | 1.020 [0.963–1.029] | *control* |

`on256`'s range is **fully disjoint** from the control's, so under the amended gate this is
a resolved **14.4% regression**, not noise.

**`off128` overlaps the control**, which matters: prefetch requires two layers resident, and
this says that extra capacity is free. The regression is the prefetching, not its
prerequisite.

## Why it loses

**The next layer's routed expert IDs are unknowable when the current layer computes.**
Routing is data-dependent and the router runs inside the next layer's own forward. So a
correct prefetch must fetch the **whole** layer — 128 rows against ~67 actually routed on
Qwen3-30B, about **1.9× the bytes**.

The link is **bandwidth-bound, not latency-bound**. Buying overlap with bandwidth loses.

That also explains, from the opposite direction, why the `hot_rows` dial works: raising
`hot_rows` to hold every routed row cuts traffic **5.4×** and helps, while hiding latency at
1.9× the bytes costs 14%. **Fewer bytes beats better-timed bytes on this path.**

## Scoring the pre-registration

| registered | actual | verdict |
|---|---|---|
| `on256/off256` **1.00–1.20× slower**, centre ~1.09 | **1.144** | **hit** — inside the band, worse than centre |
| `off256/off128` 0.95–1.05× | 1.012 (inverse of 0.988) | **hit** |
| blocked-in-`ensure` < 5% with prefetch on | ~4% | **hit** |

The registered decision rule was *"`on256` resolvably >1.0 → do not ship it; the branch is
closed and the `hot_rows` dial stands as the answer."* That is what happened, and it is
being followed rather than reinterpreted.

## Two process failures on the way, both mine

**The first run was a null experiment.** The site-packages path was hardcoded to
`python3.11`; the image ships 3.12. The copy failed, the `on` arm silently ran stock 0.17.5,
and the two arms came back within 0.8% of each other — a result that read exactly like
"prefetch is neutral". A verification line caught it, but it was written as a print rather
than a gate, so the script continued and spent pod time on an invalid experiment. The
runner now **locates** the package and **exits non-zero** if the patch is not present, and
logs `PATCH-VERIFIED` with the resolved path.

**Then `pkill -f run-prefetch.sh` killed the SSH session** whose own command line contained
that string. Kill by PID.

## What this does not say

One card, one model, one sequence length. A prefetch that fetched only the *likely* routed
rows — say from the tier's own frequency counters — is a different design and is not tested
here; it would trade correctness-by-superset for a smaller read and could plausibly win.
What is refuted is prefetching **the whole layer**, which is the only variant that needs no
speculation.

## Cost

$0.99/hr for ~45 min. External teardown backstop armed on a separate host before the first
run; pod verified gone from the account listing, not from the `DELETE` status.
