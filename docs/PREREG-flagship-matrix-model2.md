# PREREG — the flagship matrix's second model (pre-data, pre-hardware)

Registered **before** any second-model cell runs and **before** the GPU that will
run it is provisioned. This is the completion of
[`PREREG-flagship-matrix.md`](PREREG-flagship-matrix.md)'s remaining 10 of 20
cells, and unlike the first ten it is stamped in advance — so its verdict can
reach this project's `confirmed` tier rather than `measured`.

Written 2026-07-30, before the model is chosen by the rule below, before it is
downloaded, and before any pod exists.

## Why a second document

The original prereg's Amendment 1 registered 20 cells; ten ran and are reported
in [`../bench/flagship-matrix/RESULTS-flagship-matrix.md`](../bench/flagship-matrix/RESULTS-flagship-matrix.md).
Those ten are `measured` — their protocol predates their data in git history but
carries no anchor, and **a stamp applied afterwards cannot fix that**. Rather
than restamp the old document and imply otherwise, the remaining ten get their
own protocol, stamped now.

## Model selection — fixed HERE, before download

The first of the following that (a) downloads cleanly, (b) loads through the
streaming NF4 loader, and (c) fits the offload path on a 24 GB card:

1. `google/gemma-4-26b-a4b`
2. `mistralai/Mixtral-8x7B-v0.1`
3. `Qwen/Qwen3-30B-A3B-Instruct-2507`

If none qualifies, the matrix ships single-model and says so. **The order is
fixed here so a model cannot be chosen after seeing which one flatters the
result.**

## Fixture — identical to the first ten, deliberately

`offload=True` + gradient checkpointing (`use_reentrant=False`), seq 512, r=8,
α=16, AdamW lr 1e-4, batch 1, **200 steps**, the same five datasets at their
registered sha256s (`76fb9036de80`, `e0176e044fb1`, `e90914aaedfc`,
`379d6e521c7f`, `fbc68d228750`). 2 arms × 5 datasets = **10 cells**.

Card: a 24 GB Ada-class GPU (sm_89). If only another class is available, that is
reported as a deviation rather than silently absorbed — the first ten ran on an
RTX 4090 and cross-card comparison would confound B2.

## Registered outcomes

- **C1 — bit-exactness (HARD GATE).** SHA-256 of every frozen expert's packed
  bytes before and after training must be **identical, 100 %**, in both arms.
  Hashes must come from `state_dict()` (offload maps experts to their CPU home),
  with **bytes hashed > 0** and **zero empty tensors skipped** asserted, and a
  byte-flip positive control demonstrating the check can fail. Any mismatch
  voids every performance number in the document.
- **C2 — loss parity, fused vs reference.** Registered band, unchanged from the
  first ten so the two halves are comparable: **|Δ final train loss| ≤ 0.05**
  **and** median step-wise |Δ| ≤ 0.05, per dataset.
- **C3 — cost, reported for every cell, not gated:** s/step, tokens/s, peak
  VRAM, and energy as J/step from `nvidia-smi` at 200 ms across the timed
  window, idle baseline subtracted.
- **C4 — "best result", defined pre-data.** Lowest held-out eval loss on that
  dataset's own split, per dataset. **A margin smaller than 10× the measured
  zero-adapter floor (0.099 %) is reported as NOT SEPARABLE, not as a win** —
  registered now because the first ten produced exactly that situation and the
  temptation to call a winner is easiest to resist before seeing the numbers.
- **C5 — cross-model transfer, the point of a second model.** Whether the
  *topology* of the first ten reproduces: fused faster per step, peak VRAM ratio
  below 1.0 and flat across datasets, energy ratio below 1.0. Reported as a
  qualitative reproduction check, **not** as an expectation that the ratios
  match numerically — a different architecture has no obligation to.

## Rules

Bands do not move after data. Every cell ships whether or not it flatters the
fused path. If a cell OOMs or a pod dies, the shortfall is stated and the cell
reported missing rather than dropped. If C1 fails anywhere, the matrix publishes
the failure and no speed or energy claim.

**Spend:** capped at **$25**, under the standing $35/job ceiling, with a
session-independent cron bill-cap pinned to the pod id literally — never
resolved from "current pods" at fire time, and positive/negative controlled
before the run starts.

## Anchor

This file is OpenTimestamped **before** model download and **before** pod
creation. That is what separates it from the first ten: the bars below cannot be
moved after the data arrives, and the anchor proves it.
