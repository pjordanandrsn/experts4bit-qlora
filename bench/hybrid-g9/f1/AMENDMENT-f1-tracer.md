# AMENDMENT — F1: profiler `with_stack` attributes nothing on this stack

Registered 2026-08-25, after Stage A attempt 1 produced a vacuous
receipt (`receipts-f1/stageA-attempt1-vacuous/`) and before any
treatment exists.

## What attempt 1 established

The run completed with a full 12/12 profiler window, attributed
**4.88 ms/step** of elementwise device time, and reported **0.003
ms/step unclassified** — the op set is complete, so the block is
accounted for. But **100% of it landed in `<no-python-frame>`**: not
one call site resolved.

Direct probe on the box (torch 2.13.0 / triton 3.7.1): with
`with_stack=True`, `key_averages()`, `key_averages(group_by_stack_n=24)`
and `profiler.function_events` ALL return events whose `.stack` is
empty — 0 of 75 aggregated events and 0 of 221 raw events carry a
python frame. The registered mechanism does not exist on this stack.
(The same is true of T5's `--sync-attr-out` nonzero attribution if it
is re-run here; its own receipts predate this torch.)

The instrument did not lie about it — `_py_site`'s fallback is a
visible `<no-python-frame>` label rather than a plausible-looking
file — but a mechanism that attributes nothing cannot close Stage A.

## Amended mechanism

Call sites now come from `_EwSiteTracer`, a `TorchDispatchMode` that
records, for every elementwise aten dispatch, the nearest python frame
that is our code (`sys._getframe` walking, not `traceback`, because it
runs per op). It depends on nothing but python frames, so it cannot be
silently disabled by a torch build.

**It records COUNTS, not time.** Device time per op still comes from
the profiler; each op's time is apportioned across its sites in
proportion to launch counts. That is exact when one op's launches cost
the same, which is this block's regime — 1.21 µs per launch at the
GPU's minimum kernel duration, so cost tracks launch count rather than
tensor size. The receipt carries `site_method`, the raw counts, and
the apportioned time together so the approximation is never invisible.

Also amended: `row_limit` rises from 80 to 400 on attribution runs.
The 80-row table failed the census coverage gate at 81.8% on the fresh
profile (the gate worked as designed — it refused rather than
publishing a partial budget).

## Bars and decision map

Unchanged (PASS ≤ 10.5 ms/step, PARTIAL 10.5–12.0, REFUTED > 12.0).
Stage A closes when the tracer resolves ≥ 90% of the attributed
elementwise time to named call sites.

## Test coverage

The tracer runs on CPU tensors, so `tests/test_ew_attr.py` now
reproduces the failure mode locally: a test asserts the tracer records
real sites and that none is `<no-python-frame>` — it would have caught
attempt 1 before a box was rented. 17 tests total.
