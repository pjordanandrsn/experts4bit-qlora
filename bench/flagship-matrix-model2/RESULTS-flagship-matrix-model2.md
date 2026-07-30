# The matrix's second model: all ten cells, C1 real this time — and a C4 winner that does not survive a re-run
### 2026-07-30 · google/gemma-4-26B-A4B · RTX 4090 (sm_89) · grouped-nf4-gemm 0.2.6 · e4b 0.6.7

The protocol ([`docs/PREREG-flagship-matrix-model2.md`](../../docs/PREREG-flagship-matrix-model2.md))
was committed **and OpenTimestamps-stamped before the model was chosen, before it
was downloaded, and before any pod existed**. Unlike the first ten cells, whose
protocol predates their data in git history but carries no anchor, this one can
reach the `confirmed` tier.

**All 10 registered cells ran.** 2 arms × 5 datasets × 200 steps, one model, one
physical host.

Model selection followed the rule as written: `google/gemma-4-26B-A4B` was first
in the registered order and it qualified — public and ungated, loads through the
streaming NF4 loader (30/30 MoE layers × 128 experts), and fits the offload path
on 24 GB. No later candidate was consulted.

## Every cell

Fixed throughout: `offload=True` + gradient checkpointing (`use_reentrant=False`),
seq 512, r=8, α=16, AdamW lr 1e-4, batch 1, 200 steps. Energy is `nvidia-smi` at
200 ms across the timed window, idle baseline subtracted.

| dataset | arm | eval@200 | rel. impr | s/step | tok/s | peak GB | J/step | C1 |
|---|---|---|---|---|---|---|---|---|
| clinical | reference | 0.27061 | 0.09679 | 10.463 | 8.7 | 9.711 | 736.56 | 12.85 GB ✓ |
| clinical | **fused** | **0.25438** | 0.09087 | **6.643** | 13.7 | **7.889** | **633.61** | 12.85 GB ✓ |
| code | reference | 0.14167 | 0.08195 | 9.650 | 9.7 | 9.713 | 673.98 | 12.85 GB ✓ |
| code | **fused** | **0.13963** | 0.08038 | **6.358** | 14.7 | **7.893** | **615.46** | 12.85 GB ✓ |
| finance | reference | 0.40571 | 0.11848 | 9.582 | 8.6 | 9.704 | 673.20 | 12.85 GB ✓ |
| finance | **fused** | **0.38696** | 0.11315 | **6.201** | 13.3 | **7.884** | **590.16** | 12.85 GB ✓ |
| legal | reference | 0.09949 | 0.04764 | 10.205 | 10.7 | 9.720 | 703.74 | 12.85 GB ✓ |
| legal | **fused** | **0.09968** | 0.04772 | **6.631** | 16.5 | **7.896** | **642.91** | 12.85 GB ✓ |
| support | reference | 0.14526 | 0.04923 | 10.616 | 12.1 | 9.777 | 733.34 | 12.85 GB ✓ |
| support | **fused** | **0.14482** | 0.04904 | **6.869** | 18.7 | **7.920** | **675.84** | 12.85 GB ✓ |

## The registered outcomes

### C1 — bit-exactness (HARD GATE): PASSES 10/10, and this time the gate is real

**12,846,366,720 bytes (12.85 GB) hashed per cell, 0 changed, 0 empty tensors
skipped, byte-flip positive control fires** — in every cell of both arms.

That number was derived from `config.json` *before* the first receipt existed —
128 experts × (1408×2816 + 2816×704) elements per layer, NF4-packed at ½ byte
plus fp32 absmax at blocksize 64, × 30 layers = **12,846,366,720**. It matches
the gate's count **to the byte**, so the gate is hashing the packed expert bytes
and nothing else.

This matters because the first ten cells' equivalent gate hashed **zero** bytes:
it read `getattr(module, "gate_up_proj")`, a 0-element placeholder under
`offload=True`, and compared `sha256(b"")` with itself 192 times. That gate is
withdrawn in [`../flagship-matrix/RESULTS-flagship-matrix.md`](../flagship-matrix/RESULTS-flagship-matrix.md).
Here the hashes come from `state_dict()`, and `bytes_hashed > 0`,
`empties_skipped == 0` and the byte-flip control are **asserted before step 1** —
a cell that fails any of them refuses to write a receipt.

### C2 — loss parity: PASSES on all five, on both registered metrics

Registered band: `|Δ final-**train**-loss| ≤ 0.05` **and** median step-wise
`|Δ| ≤ 0.05`.

| dataset | Δ final train | median step-wise \|Δ\| | (Δ eval — *not* the band) |
|---|---|---|---|
| clinical | 0.00063 | 0.00713 | 0.01623 |
| code | 0.00268 | 0.00720 | 0.00204 |
| **finance** | **0.03653** | **0.01499** | 0.01875 |
| legal | 0.00168 | 0.00221 | 0.00019 |
| support | 0.00172 | 0.00328 | 0.00044 |

Worst cell is finance at **0.03653**, 1.4× inside the band — much closer to the
edge than the first model's worst (0.01651, 3× inside). Both criteria still pass
everywhere. The eval column is shown only to make the point that it is a
*different, and here often larger*, quantity than the one registered.

### C3 — cost, reported not gated

**1.52–1.58× faster per step**, at **0.810–0.813×** peak VRAM and
**0.860–0.922×** energy.

The VRAM ratio is flat to three decimals across five structurally different
datasets — the same residency-policy fingerprint the first model showed at
0.754–0.755×. Throttle reasons were sampled every 30 s for the whole run and
never left `0x0` (none) or `0x1` (GPU idle between cells): **the card was not
thermally or power capped at any point**, at 62–65 °C against a 450 W limit, so
these are not throttled numbers.

### C4 — best result: the registered rule says "fused", and a replication says do not believe it

Registered as lowest held-out eval loss, with margins under **0.99 %** (10× the
measured 0.099 % zero-adapter floor) reported as NOT SEPARABLE:

| dataset | reference | fused | margin | registered verdict |
|---|---|---|---|---|
| clinical | 0.27061 | 0.25438 | 5.998 % | fused |
| code | 0.14167 | 0.13963 | 1.440 % | fused |
| finance | 0.40571 | 0.38696 | 4.622 % | fused |
| legal | 0.09949 | 0.09968 | 0.191 % | not separable |
| support | 0.14526 | 0.14482 | 0.303 % | not separable |

Tally as the rule computes it: **fused 3, not separable 2, reference 0.**

**Do not read that as a fused win.** The `clinical` cell was run twice, on two
different RTX 4090s, same seeds and same config — the first pod was deleted
mid-matrix and the pair was re-run rather than resumed:

| | pod A (`oo45mv…`) | pod B (`l4c283…`) |
|---|---|---|
| eval ref / fused | 0.25575 / 0.27066 | 0.27061 / 0.25438 |
| **C4 verdict** | **reference by 5.83 %** | **fused by 6.00 %** |

**The winner flips.** Both margins clear the 0.99 % threshold by ~6×, and the
sign still reverses on a re-run. The threshold was calibrated from Qwen3's
zero-adapter floor and **does not transfer to this model** — it understates real
run-to-run variance here by roughly an order of magnitude.

The band is not moved; C4 is reported exactly as registered, with this beside it.
The honest reading is that C4's per-dataset winner is **not a measurable quantity
at n=1**, and the two paths remain not separable — which is what C2 tests, and
C2 passes.

Why the trajectories diverge at all, from the same receipts: step 1 is
**bit-identical** across both hosts (2.85056), step 2 — the first loss reflecting
an optimizer update — already differs, and the gap stays **bounded and flat**
(~0.019 mean early, ~0.006 late) rather than compounding. That is nondeterministic
accumulation in the MoE combine (`index_add_` atomics, async H2D ordering), not a
configuration difference. The eval delta that flipped the verdict, 0.0149, is the
same size as that per-step jitter.

### C5 — cross-model transfer: the topology REPRODUCES

| criterion | |
|---|---|
| fused faster per step | ✅ 1.52–1.58× (first model: 1.75–1.81×) |
| peak VRAM ratio below 1.0 | ✅ 0.810–0.813× (first model: 0.754–0.755×) |
| VRAM ratio flat across datasets | ✅ spread 0.003 |
| energy ratio below 1.0 | ✅ 0.860–0.922× (first model: 0.797–0.846×) |

All four directions hold on a different architecture. C5 registered a
**direction** check, not a numeric match, and the magnitudes are lower across the
board — consistent with Gemma-4 moving less per step than Qwen3 (30 layers ×
427 MB vs 48 × 340 MB), leaving the kernel's win less room to hide behind
transfer on a transfer-bound workload.

## What this does not license

- **One host for the ten cells.** The C3 ratios are within-pair on one physical
  4090, which is the right design — but see below: `s/step` itself moved 8.6–9.0 %
  between two 4090s, and only the *ratio* survived that (1.568× vs 1.575×, 0.4 %
  apart). Energy did **not**: 0.964× on pod A against 0.860× on pod B.
- **Cross-host energy is confounded.** `idle_w` is subtracted to form J/step and
  it differed 3.4× between hosts (8.9 W vs 30.5 W). Within-host ratios are
  unaffected; cross-host energy comparisons are not meaningful.
- **No variance estimate within a host.** Each cell is a single run. The only
  repetition anywhere is the `clinical` pair across two hosts, and its purpose
  here is to bound confidence, not to establish a mean.
- **Synthetic datasets**, at the sha256s the prereg registered (verified on the
  pod before the first cell). No claim that they proxy real industry data.
- **The receipt's PCIe field is unreliable.** It is sampled at cell start when
  the link has downtrained (it reads gen2); [`telemetry.csv`](telemetry.csv),
  sampled during the run, shows gen4 ×8 and is the authority. Recorded here
  rather than quietly corrected.

## Provenance

Per-cell receipts `gemma-4-26B-A4B__*__{reference,fused}.json`, each carrying a
host fingerprint (GPU uuid, PCIe width, power limit, clock ceilings, CPU, RAM,
driver). Driver log [`matrix.log`](matrix.log), environment gates
[`hostinfo.txt`](hostinfo.txt), 318 rows of 30-second host telemetry
[`telemetry.csv`](telemetry.csv), the pod↔container↔GPU join
[`pod_mapping.txt`](pod_mapping.txt), and the two-cell cross-host replication in
[`podA-replication/`](podA-replication/).

Adjudicated by [`../flagship-matrix/drivers/n17_summarize.py`](../flagship-matrix/drivers/n17_summarize.py),
which encodes C4's separability rule mechanically and was written and
positive-controlled against the *first* matrix's published cells **before these
cells finished** — it reproduces that matrix's numbers to the digit.

**Spend:** one on-demand RTX 4090 at $0.69/hr, ~$6 total against the prereg's $25
cap. Teardown verified: pod probes 404, zero running pods.
