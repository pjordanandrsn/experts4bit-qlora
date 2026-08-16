# Pre-registration — does prefetching the next layer make the step faster?

**Written 2026-08-13, before any rented run.** Committed ahead of the data.

## What is being tested

Arena staging blocks the step: it runs in a forward pre-hook immediately before the layer
computes and `ColdTier.ensure` returns only once the rows land. A screening run on the
correctness box put that at **~16% of the step**, with ~47 blocking misses.

A prefetch worker (`E4B_ARENA_PREFETCH=1`, branch `feat/arena-prefetch`) warms the next
layer while the current one computes. Screening says it does what it was built to do —
blocked time **~16% → ~4%**, main-thread misses **47 → 0–1** — but that **every step got
slower**: +9.4%, +18.7%, +8.4%, +3.6% paired by index.

That box has ~50% step-time variance and n=4, so the regression is suggestive, not
established. This settles it.

## Why it might genuinely be slower

The next layer's routed IDs are **unknowable** when the current layer computes — routing is
data-dependent and the router runs inside the next layer's own forward. So the prefetch
fetches the **whole** layer: 128 rows against ~67 routed on Qwen3-30B, **~1.9× the bytes**.

If the NVMe link is the binding constraint rather than latency, trading latency for
bandwidth loses. That would also explain why the `hot_rows` dial — which cuts bytes 5.4× —
was the effective lever, while hiding latency is not.

## Arms

Four, so "prefetch" is separated from "more `hot_rows`" (prefetch *requires* two layers'
capacity, so it cannot be tested at the routing floor):

| label | `hot_rows` | prefetch | what it isolates |
|---|---|---|---|
| `off128` | 128 (routing floor) | off | what a user runs today |
| `off256` | 256 (two layers) | off | the cost of the capacity prefetch needs |
| `on256` | 256 | **on** | the feature |
| `off256_self` | 256 | off | **control** — same config timed twice per round |

Qwen3-30B-A3B, seq 384, one seed, **5 scored rounds of 12 scored steps** (the amended
precision), interleaved in fixed order, warmup round dropped.

## Gate

Per [`PREREG-timing-AMENDMENT-1.md`](PREREG-timing-AMENDMENT-1.md): an effect is resolved
when its per-round range **does not overlap** the control's. `off256_self / off256` is the
control.

## Predictions

1. **`on256` vs `off256`: 1.00–1.20× (slower).** Centre ~1.09, from the screening median.
2. **`off256` vs `off128`: 0.95–1.05×** — more capacity should not change step time much,
   since the `hot_rows` sweep found no resolvable difference from 128 to 1024.
3. **Blocked-in-`ensure` falls to <5% with prefetch on**, reproducing the screening.

## What each outcome means

- **`on256` resolvably >1.0** — prefetch trades latency for bandwidth and loses. Do not
  ship it; the branch is closed and the `hot_rows` dial stands as the answer.
- **`on256` resolvably <1.0** — the screening regression was that box's variance, and
  prefetch is a real win worth shipping opt-in.
- **Overlaps the control** — report indistinguishable and do **not** ship: an opt-in flag
  that removes a visible stall without moving wall clock is a trap, because the stall
  metric reads like a win.

**Registered now: I expect outcome 1.** Publishing that in advance is the point — if it
comes back a win instead, that is a prediction I got wrong on the record, not a story
adjusted afterwards.

## Cost

SECURE on-demand, one card, external teardown backstop armed before the first run,
`costPerHr` read off the pod after create. Expected ~70 min, well inside the $35/job cap.
