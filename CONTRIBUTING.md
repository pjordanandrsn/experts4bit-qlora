# Contributing

Thanks for looking. This package decides **which expert bytes are where**; the
sibling [`grouped-nf4-gemm`](https://github.com/pjordanandrsn/grouped-nf4-gemm)
makes one expert-stack matmul cheap. Issues about kernel speed usually belong
there; issues about loading, offload, residency, training, or serving belong here.

## Before filing a bug, check the two things that cause most of them

**1. Is the fast path actually on?** `train.main()` does not enable it. The
fused lane is opt-in:

```python
n = enable_fast_train(model, dgrad=True)   # needs the [fast] extra
assert n, "still on the per-expert loop"
```

`0` and "silently still on the per-expert Python loop" look identical from the
caller's side, which is why every enabler returns a count and why you should
assert it. At 256 experts over 40 layers that loop is ~10k sync-gated iterations
per forward.

**2. Did `grouped-nf4-gemm` build for your arch?** It is arch-gated and compiles.
When it will not build, `enable_batched_train(model)` needs no extras and is the
supported fallback — slower at real width, and it says so in its own docstring.

## Running the checks

```bash
pip install -e ".[test]"      # CPU torch is enough; grouped-nf4-gemm comes with the extra
pytest tests/ -q              # GPU-only tests skip with a reason; do not filter by name
```

The full gate list (lint, packaging, claims, capabilities, manifest, links,
discovery, llms bundle) is AGENTS.md §6; how CI is triggered on a pull request
(the `ready-to-merge` label) is AGENTS.md §10.

Much of this package is testable without a GPU because the pieces that matter
are format and placement, not arithmetic: arenas are built from bytes the tests
write, and `formats.mxfp4.dequantize_mxfp4` is pure torch. **A CPU test that
runs is worth more than a GPU test you assert.**

## Claims carry receipts

Any performance or memory claim in a PR quotes a claim id in `docs/claims.json`
whose receipt is committed; a run whose receipt is private is a `measured-private`
row with a non-empty `evidence_private`. A number without a row is not published
(AGENTS.md §7). If it is a timing claim, ship a **self-pair** — the same arm
timed against itself — because a ratio inside the instrument's own spread is not
a measurement. Two devices, or name the single architecture it holds for.

One caution specific to MoE, learned the expensive way: **benchmark on real
text, not random token ids.** Random ids route to fewer experts far more
unevenly, which flatters the per-expert loop and understated this package's own
fused lane by 1.6–1.7x (`gnf4.kernel.e2e-training-real-prose` in the kernel
repository's register: 4.50x on prose against 2.75x on random ids on a 4090,
4.75x against 2.81x on an H100; receipt
[`RESULTS-e2e-training.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase1/results/dequant_forward/RESULTS-e2e-training.md)).

## The rule that outranks the others

**Never let an arm compute a function the frozen base does not.** `ExpertsLoRA`
re-implements the expert math so the low-rank delta lands before the
nonlinearity, which means it owns the choice of nonlinearity — so a clamped or
gated architecture must supply `_apply_gate`. Get this wrong and nothing fails:
the model trains, the loss falls, and it optimises the wrong function. Changes
touching expert math need a parity test against the reference, not a loss curve.

## Scope

Loading, quantisation placement, offload, residency (host RAM, NVMe arena),
LoRA over frozen experts, the paged serving engine, and provenance. New
architectures are welcome and want a keymap plus a loader test.
