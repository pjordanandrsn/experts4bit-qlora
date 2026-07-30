# Flagship 30B-class matrix: both arms, 5 datasets. B2 passes everywhere; B4 says the two paths are not separable
### 2026-07-29/30 · RTX 4090 (sm_89) · torch 2.8.0+cu128 · grouped-nf4-gemm **0.2.4** · e4b **0.6.4/0.6.5**

The protocol ([`docs/PREREG-flagship-matrix.md`](../../docs/PREREG-flagship-matrix.md),
committed pre-data as `b08747f`, amended pre-data as `b33c553`) registered
**20 cells**: 2 models × 5 synthetic industry datasets × 2 arms, 200 steps each.

**10 cells ran** — one model, five datasets, **both arms**. The second model's 10
cells did not run; it was never downloaded.

| registered | ran | missing |
|---|---|---|
| Qwen3-30B-A3B × 5 datasets × reference | **5** | — |
| Qwen3-30B-A3B × 5 datasets × fused | **5** | — |
| second 30B-class model × 5 datasets × 2 arms | 0 | **10** |

An earlier revision of this document reported 5 of 20 with the fused arm
missing. That arm was attempted first, hit **CUDA OOM on the 24 GB card**, and
the matrix was relaunched `ARMS=reference`. The OOM produced the backward-pass
fix (stash a closure, not the packed weights, so the single-resident-layer
offload policy survives autograd); the fused arm then ran to completion on
2026-07-30 against that fix.

**Evidence tier: `measured`.** The protocol predates the data in public git
history but carries no OpenTimestamps anchor.

## Every cell

Fixed throughout: `offload=True` + gradient checkpointing (`use_reentrant=False`),
seq 512, r=8, α=16, AdamW lr 1e-4, batch 1, 200 steps. Energy is `nvidia-smi`
sampled at 200 ms across the timed window, idle baseline subtracted.

| dataset | arm | eval @200 | rel. impr | s/step | tok/s | peak GB | J/step | B1 |
|---|---|---|---|---|---|---|---|---|
| clinical | reference | 0.27220 | 0.07267 | 15.206 | 5.6 | 9.132 | 836.39 | 192/192 ✓ |
| clinical | **fused** | **0.27217** | 0.07254 | **8.544** | 10.1 | **6.891** | **707.83** | 192/192 ✓ |
| code | reference | 0.17116 | 0.06600 | 15.836 | 5.2 | 9.134 | 888.73 | 192/192 ✓ |
| code | **fused** | **0.17019** | 0.06574 | **8.813** | 9.3 | **6.890** | **708.01** | 192/192 ✓ |
| finance | reference | **0.44563** | 0.09375 | 13.879 | 5.3 | 9.126 | 760.13 | 192/192 ✓ |
| finance | fused | 0.45286 | 0.09530 | **7.880** | 9.4 | **6.889** | **617.48** | 192/192 ✓ |
| legal | reference | **0.10875** | 0.03790 | 14.564 | 7.1 | 9.142 | 818.06 | 192/192 ✓ |
| legal | fused | 0.10970 | 0.03802 | **8.322** | 12.4 | **6.898** | **683.72** | 192/192 ✓ |
| support | reference | **0.16339** | 0.04899 | 15.380 | 8.0 | 9.153 | 850.93 | 192/192 ✓ |
| support | fused | 0.16473 | 0.04931 | **8.489** | 14.5 | **6.907** | **696.36** | 192/192 ✓ |

## The registered outcomes

**B1 — bit-exactness (HARD GATE): PASSES 10/10.** 192 frozen expert tensors
hashed per cell, **0 changed**, in both arms. QLoRA trains adapters only; a
single changed byte would have voided every number above.

**B2 — loss parity, fused vs reference: PASSES on all five datasets.** Registered
band `|Δ final| ≤ 0.05`:

| dataset | Δ eval | vs band |
|---|---|---|
| clinical | 0.00003 | 1600× inside |
| code | 0.00097 | 51× |
| legal | 0.00095 | 53× |
| support | 0.00134 | 37× |
| **finance** | **0.00723** | **7× inside** — the worst cell |

**B3 — cost, reported not gated.** The fused arm is **1.75–1.81× faster per
step**, at **0.754–0.755×** the peak VRAM and **0.797–0.846×** the energy. The
memory ratio is flat to three decimals across five structurally different
datasets, which is what a residency policy — rather than a data-dependent
effect — looks like. Both arms sit near 12 % of the card's TDP throughout:
transfer-bound, not compute-bound.

**B4 — "best result": adjudicated, and the honest answer is "neither".**
Registered as *lowest held-out eval loss on that dataset's own split*:

| dataset | reference | fused | winner | margin |
|---|---|---|---|---|
| clinical | 0.27220 | 0.27217 | fused | 0.01 % |
| code | 0.17116 | 0.17019 | fused | 0.57 % |
| finance | 0.44563 | 0.45286 | reference | 1.62 % |
| legal | 0.10875 | 0.10970 | reference | 0.87 % |
| support | 0.16339 | 0.16473 | reference | 0.82 % |

Tally: **reference 3, fused 2** — and reporting that as a result would be
overclaiming. A separate measurement put the **zero-adapter floor** (LoRA `B`
still zero, so the only difference is kernel and summation order) at **0.099 %**.
Four of these five margins are within an order of magnitude of that floor, and
the largest is 1.62 %. The two paths are **not separable at this sample size**;
the correct reading is that the fused path reproduces the reference rather than
competing with it — which is what B2 was there to test. Declaring a winner on a
0.00003 difference would be reading noise.

**No cross-dataset ranking.** Absolute losses are not comparable between rows —
different token distributions make that a tokenizer statistic. Only relative
improvement and the B3 costs are quoted across datasets.

## Data provenance

sha256 registered per dataset before generation output was inspected;
[`ds_manifest.json`](ds_manifest.json) carries the full hashes and the leading
12 hex digits match the prereg table for all five (`76fb9036de80`,
`e0176e044fb1`, `e90914aaedfc`, `379d6e521c7f`, `fbc68d228750`). 1,200 train /
200 held-out each, disjoint by construction, seeds 1000–1004. **Synthetic, and
no claim is made that they proxy real industry data** — the design goal was
disparate *structure*, not domain realism.

## What would complete this

The 10 second-model cells, same config, same unmoved bands. Until they run, this
matrix speaks for one 30B-class model across five structurally different
datasets and says nothing about cross-model transfer.

Raw per-cell receipts: `Qwen3-30B-A3B__*__{reference,fused}.json`; driver logs
[`matrix.log`](matrix.log) (reference) and
[`matrix-fused.log`](matrix-fused.log) (fused).
