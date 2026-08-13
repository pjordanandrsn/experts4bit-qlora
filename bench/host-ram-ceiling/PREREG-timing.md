# Pre-registration — what does the arena cost, and what does it save, in TIME?

**Written 2026-08-13 on a rented RTX 3090, before any timed run.** Committed ahead of the
data. Everything measured so far is memory; this is the first timing claim in the arena
line, and [`feedback-benchmark-testbed-policy`] forbids taking it from the QNAP.

## Why a pod, and why these three

The QNAP answered the free questions. Two remain that only a timing box can settle, and
one that the QNAP actively misleads on:

- **The QNAP has 12 cores; this pod has 256.** Quantize-at-load is CPU work that
  parallelizes. The load-time advantage observed there may be largely a core-count
  artifact — which is the single most likely way the headline below turns out to be worth
  nothing.
- **Step time at the `hot_rows` floor is unestablished.** The existing "1.64× at the
  floor" is one sample per arm, and its companion "1.03× fully pinned" was already
  refuted as inside the harness's own noise (0.17.1).

## Instrument

Paired protocol, per [`feedback-perf-measurement-discipline`]: every arm timed once per
round in fixed order, rounds interleaved, warmup round dropped, and a **`host_self`
control that times the identical host configuration twice per round**. A ratio is only
reportable if the control sits near 1.000; whatever spread the control shows is the
resolution limit, and any effect inside it is *unresolvable*, not real.

Both checkpoints and both arenas live on the pod's **local** disk. `/workspace` is a
MooseFS network mount, and the host arm's load includes reading a 61 GB checkpoint — over
the network that would inflate exactly the ratio being measured.

## T1 — time to first step (load)

The host path fuses and quantizes every expert at load; the arena path reads a baked file.
Observed on the QNAP (12 cores, not a timing venue, directional only): OLMoE 43.2 s vs
6.9 s (6.3×), Qwen3-30B-A3B 324.0 s vs 16.5 s (19.6×).

**Predicted here, on 256 cores:** OLMoE **2–6×**, Qwen3 **4–15×** — lower than the QNAP in
both cases, because the host arm's cost is the part that parallelizes and the arena arm's
is not.

- **Falsified if** the ratio is **< 1.5×** on both models: then quantize-at-load is cheap
  on any real machine and the saving is a NAS artifact. **Stop rule: do not publish a
  load-time claim below 1.5×.**
- **The interesting failure** is a ratio *above* the QNAP's, which would mean something
  other than CPU parallelism dominates.

## T2 — steady step time at the `hot_rows` floor

**Predicted: arena 1.1–1.8× host** on OLMoE at `hot_rows=64`, consistent with the
unpaired 1.64×.

- **Stop rule:** if `host_self` spread exceeds **±8%**, report "indistinguishable at this
  resolution" and no ratio, exactly as 0.17.1 did. A number inside the control's spread is
  not a number.

## T3 — step time against `hot_rows` (Qwen3, 128 → 1024)

Prices the optimization the routing-density measurement suggested. Training routes a
median of 67 of 128 experts per forward but the tier **raises** rather than spilling, so
`hot_rows` must cover the worst forward (99 observed, union 93 and climbing). If a miss
could be fetched on demand, `hot_rows` could sit near the median — ~1.9× less pinned host
RAM.

**Predicted: step time falls monotonically as `hot_rows` rises**, because more residency
means less disk traffic.

- **Falsified if flat** (all points within the control's spread): then disk traffic is not
  the steady-state cost, the spill idea buys nothing in time, and its only value is the
  RAM saving. **That result kills a feature before it is built, which is the cheapest
  possible outcome.**

## Not in scope

One card, so nothing here is a portability claim — per
[`feedback-falsified-component-rides-along`], a second architecture is required before any
timing number ships. Absolute times are this pod's; only within-pair ratios travel.

## Cost

$0.50/hr read off `costPerHr` after create (not the listing). 4-hour external backstop
armed on the mini before the first run; worst case $2.00 against a $35/job cap.
