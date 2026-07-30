# Flagship 30B-class matrix: 5 of 20 registered cells ran. Reporting the shortfall, and what the 5 establish
### 2026-07-29 · RTX 4090 (sm_89) · torch 2.8.0+cu128 · grouped-nf4-gemm **0.2.4** · e4b **0.6.4**

The protocol ([`docs/PREREG-flagship-matrix.md`](../../docs/PREREG-flagship-matrix.md),
committed pre-data as `b08747f`, amended pre-data as `b33c553`) registered
**20 cells**: 2 models × 5 synthetic industry datasets × 2 arms (reference vs
`enable_fast_train`), 200 steps each.

**5 cells ran.** One model, five datasets, **reference arm only.**

| registered | ran | missing |
|---|---|---|
| Qwen3-30B-A3B × 5 datasets × reference | **5** | — |
| Qwen3-30B-A3B × 5 datasets × fused | 0 | **5** |
| second 30B-class model × 5 datasets × 2 arms | 0 | **10** |

The prereg's own rule governs this document: *"If a cell OOMs or the pod dies, the
shortfall is stated and the cell is reported missing rather than silently
dropped."* The fused arm and the second model were not run — the second model was
never downloaded, and the fused path's own correctness gate had to settle first
(it did, on separate hardware:
[`../fused-train-gate/RESULTS-fused-train-gate.md`](../fused-train-gate/RESULTS-fused-train-gate.md)).
No cell failed. Ten registered cells simply do not exist, and no claim below
draws on them.

**Evidence tier: `measured`.** The protocol predates the data in public git
history but is not OpenTimestamps-stamped.

## The cells that ran

Fixed for every cell: `offload=True` + gradient checkpointing
(`use_reentrant=False`), seq 512, r=8, α=16, AdamW lr 1e-4, batch 1, 200 steps.
Energy is `nvidia-smi` power sampled at 200 ms across the timed window with the
idle baseline subtracted (idle 23.1–23.5 W measured per cell).

| dataset | s/step | tok/s | peak VRAM | J/step | mean W | eval step 0 → 200 | rel. improvement | B1 |
|---|---|---|---|---|---|---|---|---|
| legal | 14.5643 | 7.1 | 9.142 GB | 818.06 | 79.5 | 2.86930 → 0.10875 | **0.0379** | 192/192 ✓ |
| support | 15.3796 | 8.0 | 9.153 GB | 850.93 | 78.6 | 3.33547 → 0.16339 | **0.04899** | 192/192 ✓ |
| code | 15.8362 | 5.2 | 9.134 GB | 888.73 | 79.8 | 2.59341 → 0.17116 | **0.06600** | 192/192 ✓ |
| clinical | 15.2056 | 5.6 | 9.132 GB | 836.39 | 78.1 | 3.74586 → 0.27220 | **0.07267** | 192/192 ✓ |
| finance | 13.8788 | 5.3 | 9.126 GB | 760.13 | 78.3 | 4.75361 → 0.44563 | **0.09375** | 192/192 ✓ |

Load time 277–293 s per cell. Post-cell VRAM 1 MiB in all five — no leak across
cells.

**B1 — bit-exactness (HARD GATE): PASSES 5/5.** 192 frozen expert tensors hashed
per cell, **0 changed**. A single changed byte would have voided every number
above.

**B3 — cost: complete for this arm.** Peak VRAM is flat at 9.126–9.153 GB across
all five datasets — a 0.3 % spread. That is the point of the offload path: the
memory bill is set by the resident-layer policy, not by the data. Energy tracks
s/step almost exactly (760–889 J/step over a 13.9–15.8 s/step range at a nearly
constant ~78–80 W), which says the workload is **transfer-bound, not
compute-bound** — mean draw sits at roughly 12 % of this card's TDP throughout.

**B4 — "best result": NOT adjudicated.** As registered, best = *lowest held-out
eval loss on that dataset's own held-out split*, which requires at least two arms
per dataset to rank. With one arm there is nothing to rank, so the registered
question is unanswered and the table above is bolded on **relative improvement**
only — the one quantity the prereg permits across datasets. Absolute losses are
**not** comparable between rows: different token distributions make cross-dataset
loss a tokenizer statistic, not a quality measure. `legal` showing the lowest
final eval loss (0.10875) and the *smallest* relative improvement (0.0379) is
exactly that trap in miniature.

## Data provenance

The prereg registered a sha256 per dataset before generation output was inspected.
[`ds_manifest.json`](ds_manifest.json) carries the full hashes; the leading 12
hex digits match the registered table, so these cells demonstrably trained on the
registered bytes:

| dataset | registered | manifest |
|---|---|---|
| clinical | `76fb9036de80` | `76fb9036de80f3bb…` ✓ |
| code | `e0176e044fb1` | `e0176e044fb16738…` ✓ |
| finance | `e90914aaedfc` | `e90914aaedfc5e23…` ✓ |
| legal | `379d6e521c7f` | `379d6e521c7f4d18…` ✓ |
| support | `fbc68d228750` | `fbc68d228750…` ✓ |

1,200 train / 200 held-out each, disjoint by construction, seeded 1000–1004.
**Synthetic, and no claim is made that they proxy real industry data** — the
design goal was disparate *structure* (list-heavy vs prose vs code vs numeric),
not domain realism.

## What would complete this

The 10 fused-arm cells and the 10 second-model cells, at the same config, under
the same unmoved bands. Until they run, the honest summary of this matrix is:
*the reference path trains a 30B-class MoE on a 24 GB card across five
structurally different datasets at a flat ~9.13 GB peak and ~800 J/step, without
touching one byte of the frozen 4-bit experts* — and the fused arm's parity is
established on `clinical` alone, elsewhere.

Raw per-cell receipts: `Qwen3-30B-A3B__*__reference.json`, driver log
[`matrix.log`](matrix.log).
