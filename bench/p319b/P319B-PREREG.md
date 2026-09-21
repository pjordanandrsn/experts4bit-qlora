# P319b — where the sm_120 divergence is, given that it is not the dot

**Lane:** P319b · **Work:** grouped-nf4-gemm#319 · **Registered:** 2026-09-21
**Box:** one RTX 5090, vast verified-secure · **Estimate:** 0.5 h × $0.66/h =
**$0.33** · **Guard:** 0.5 h (under the one-hour line; no proving run)
**Prior spend on this work item today:** p319-f32prec-1, **$0.0669 actual**.

## What round 1 settled, and what it broke

`p319-f32prec-1` (receipt `2026-09-21/p319-f32prec-1/`) established:

* **#319 reproduces at v0.32.1 on sm_120** — 27 failed / 57 passed at the
  release commit `9206352`, wider than the 10-of-35 in the filing because the
  suite has grown since. Not stale.
* **It is not the toolchain.** The lab A2000 (sm_86) passes the same suite on
  the identical torch 2.8.0+cu128 / triton 3.4.0.
* **It is not the dot precision.** Across `tf32`, `tf32x3` and `ieee` the error
  is **byte-identical** — same max, same over-tolerance percentage to ten
  decimal places — while throughput moves **939 → 403 → 38.5 GB/s**. The knob
  demonstrably reaches the kernel and demonstrably does not move the error.
  H2 of P319 is refuted.

And it raised something round 1 was not designed to answer: the error was also
**byte-identical between `split` and `f8dot`, and between `packed` and `pf8`** —
it tracked the kernel *geometry* and ignored the *compute mode*. Two readings,
with very different consequences:

* **R-A: the modes really are numerically identical here.** Then the fp8 path —
  the sm_120 serving default, certified by RESULTS-m3-default-on — carries the
  same divergence and passes the suite only because its tolerance is 1.5e-1
  against the f32 modes' 2e-2. "The fp8 modes pass, so serving is unaffected",
  which #319, `docs/STATUS.md` and `docs/capabilities.json` all currently say,
  would not be supported by measurement.
* **R-B: round 1's probe did not dispatch what it claimed.** Then the f32-only
  framing stands and the probe is the defect.

**Publishing R-A on round 1's evidence alone would be an alarm about a shipped
certification resting on a probe I already have one reason to doubt.** This run
decides between them before anything is said.

## Checks (one box, one arm, five checks, no default changes)

| check | question | how a result reads |
|---|---|---|
| C1 | do the four modes produce different bytes? | `compute_counts()` plus pairwise `torch.equal`. Identical output **with** the fp8 tally incrementing ⇒ R-A. Identical output with no fp8 tally ⇒ R-B. |
| C2 | is the ORACLE the outlier? | `paged_attn_ref` is pure torch and runs on its tensors' device; the suite runs it on CPU. Compare CPU vs CUDA oracle, and the kernel against both. |
| C3 | is it the cross-split combine or the tile loop? | `n_split=1`, `fuse_combine=False`, T=64/16/1. |
| C4 | what SHAPE is the error? | worst element in context, mean/p50/p99/max, relative Frobenius. A mis-folded scale looks nothing like a rounding cloud. |
| C5 | is it the bf16 output cast? | oracle cast to bf16 (as the suite does) vs kept in fp32. |

The identical battery has already been run on the A2000 (sm_86), where every
check is clean: oracle CPU and CUDA agree to **0.000000**, no dependence on
`n_split` or `fuse_combine`, worst element 0.003906, p50 **0.000000**, relative
Frobenius 0.002344. That baseline is what the sm_120 numbers get read against,
and it is why C1 is the only check that could not be dry-run (sm_86 has no fp8
tensor-core dot).

## Registered predictions

* **P1.** C1 shows the fp8 tally incrementing on the fp8 calls. If it does not,
  R-B holds, round 1's mode rows are withdrawn, and the probe is the finding.
* **P2.** C2's CPU and CUDA oracles agree on sm_120 as they do on sm_86. If they
  disagree by ~0.07, the oracle moved and the kernel is not the subject.
* **P3.** The divergence survives `n_split=1` and `fuse_combine=False` (C3), i.e.
  it is in the scoring loop, not the combine. Soft prior.
* **P4.** C4's error is a broad cloud, not a few structural outliers — p50 well
  above zero, unlike sm_86's exact-zero median.

## Stop rules

* **S1.** C1 answering R-B ⇒ stop. The remaining checks characterise an artefact.
* **S2.** Any check raising ⇒ record the traceback and continue; the battery is
  independent by construction.
* **S3.** 80% of the guard ⇒ fetch and finish.

## What this lane will NOT do

Change a default, widen a tolerance, or publish a claim about the fp8 serving
path that rests on round 1's probe rather than on C1.
