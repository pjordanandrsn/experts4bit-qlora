# PREREG — G9 fix: batched KV append (append_many)

Registered before any measurement. The successor line
[RESULTS-g9-hostbill.md](RESULTS-g9-hostbill.md) licensed: the decode
step's dominant cost is `Fp8PagedKV.append` running per-sequence,
per-layer, per-side — 196,608 single-token quantize calls ≈ 112
ms/step at B=16, essentially the entire 114.5 ms attention bucket,
with SDPA a rounding error.

## The fix (this PR)

* **`Fp8PagedKV.append_many(layer, seqs, k, v)`** — ONE quantize
  kernel per side for the whole batch ([B, T, H, D]), then the same
  per-sequence block writes under append()'s lockstep discipline (all
  fallible allocating work before any shared state; V-first
  publish-last per sequence). **Bit-identical by construction**: the
  FP8 scales are per (token, head) or finer (`amax` over the last dim
  only, verified in `fp8_kv.quantize_kv_fp8`), so batching along the
  token axis cannot change any byte. One batched async `index_add_`
  (cached index/ones) replaces B per-seq `seq_lens` adds.
* The pre-existing Phase-9 convenience `append_batch` now DELEGATES to
  `append_many` (no callers in-tree; any future caller inherits the
  batching).
* Decode call site (`paged_attention.py`) switches on
  `Fp8PagedKV(batched_append=True)`; default False, so every existing
  receipt path is unchanged. `step_decomp --kv-batched` plumbs it.
* **CI guard**: `tests/test_kv_append_batch.py` — batched vs
  sequential bit-equal across pool bytes, seen counters, seq_lens and
  block tables (T=1 and T=5), overflow rejected before any write,
  shape validation. Green on the real class (CPU) pre-registration.

## Baseline (frozen, from receipts-g8g9/g9_buckets.json)

step 207.5 ms | attention_host 114.5 | other 77.7 | dram 14.6 |
device attention 103.2 | ~83 tok/s aggregate at B=16.

## Protocol (one box) and bars

Swap self-test (tier unchanged — context), then on-box
`pytest tests/test_kv_append_batch.py` on CUDA (**void gate**), then
**A/B/A**: step_decomp unbatched → batched (`--kv-batched`) →
unbatched, same config as the baseline.

* **B0 (void)**: the CUDA unit tests pass; all three arms' decode
  uniques totals are EQUAL (bit-identical pages ⇒ identical attention
  ⇒ identical tokens ⇒ identical routing — any drift voids the box).
* **B1**: batched-arm `attention_host` ≤ **57.2 ms/step** (half the
  frozen baseline).
* **B2**: batched-arm `step` ≤ **166 ms** (−20%), scoreable iff the
  step delta exceeds 3× the A/A step spread.
* Stretch (reported, unscored): tok/s ≥ 110, and the residual
  attention_host/device split (what remains for a future line).

**KVAPPEND-CERTIFIED** = B0 ∧ B1 ∧ B2-scoreable-pass;
DECISION… n/a here — B1 is host-bucket arithmetic on medians of ~140
steps and needs no gate beyond the A/A spread reported with it.
REFUTED (either bar) reverts the call-site switch (the method and
tests stay; `batched_append` default False already ships it dark).
One box, one scored A/B/A. ≈ $0.90.
