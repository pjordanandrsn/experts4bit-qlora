# Layer-composed gradient parity for the training lanes — dgrad cleared, the 24x was a toy-shape artifact
### 2026-08-06 · RTX A5000 (16L) + RTX A6000 (48L), both sm_86 · torch 2.8.0+cu128 · **published wheels** e4b 0.11.0 / gnf4 0.7.0 · driver [`dgrad_gate.py`](dgrad_gate.py)

**Evidence tier: `measured`.** No pre-registered protocol; the bands cited are the ones
[`../fused-train-gate/`](../fused-train-gate/RESULTS-fused-train-gate.md) registered and used.
Both runs installed from PyPI, not a working tree — this gates the releases users get.

## Why this exists

0.11.0 shipped `enable_fast_train(dgrad=True)` with per-op accuracy measured (~2.9e-3
against the decode oracle) and layer-composed fidelity **unmeasured**, with a warning to
that effect in three docstrings. This is the measurement that retires the warning.

Perplexity is the wrong instrument for dgrad — it changes only the backward, so the
forward is bit-identical by construction. What compounds across layers is gradient error.
Measured instead: composed gradient parity (every trainable tensor, all layers, step 0),
loss trajectory over 8 steps, frozen-storage exactness, and cost. Four arms from ONE model
load with adapter snapshot/restore, so every arm starts bit-identical on identical data.

## Results

**16 layers** — OLMoE-1B-7B, A5000, `offload=True` ([receipt](gate_16L_olmoe_a5000.json)):

| arm | s/step | speedup | peak GB | grad mean | grad worst | loss med Δ |
|---|---|---|---|---|---|---|
| reference loop | 4.286 | 1.00x | 2.64 | — | — | — |
| `fast_train` | 2.789 | 1.54x | 2.41 | 3.34e-02 | 1.18e-01 | 0.0012 |
| `fast_train_dgrad` | 2.158 | 1.99x | 2.62 | 3.34e-02 | 1.17e-01 | 0.0020 |
| `batched` | 3.891 | 1.10x | 4.13 | 2.81e-03 | 1.21e-02 | 0.0020 |

**48 layers** — Qwen3-30B-A3B, A6000, resident ([receipt](gate_48L_qwen30b_a6000.json);
frozen check: 192 tensors / 15.2 GiB hashed, 0 changed, flipped-byte control detects):

| arm | s/step | speedup | peak GB | grad mean | grad worst | loss med Δ |
|---|---|---|---|---|---|---|
| reference loop | 12.781 | 1.00x | 23.13 | — | — | — |
| `fast_train` | 7.420 | 1.72x | 24.31 | 4.97e-02 | 2.04e-01 | 0.0016 |
| `fast_train_dgrad` | 5.075 | **2.52x** | 25.51 | 4.99e-02 | 2.03e-01 | 0.0011 |
| `batched` | 12.219 | 1.05x | 26.71 | 3.79e-03 | 3.02e-02 | 0.0017 |

## The three findings

**1. dgrad adds nothing to composed gradient error.** 3.34e-02 → 3.34e-02 at 16 layers,
4.97e-02 → 4.99e-02 at 48. The unmeasured-fidelity warning attached to the 0.11.0/0.7.0
releases is retired by this result, not by fiat.

**2. Composed error compounds with depth, and the fused lane carries it.** `fast_train`
mean grows +49% from 16 to 48 layers; the kernel-free path grows +35% from a base an order
of magnitude lower. At 48 layers the fused lane is **13x looser** than the kernel-free one.
Note the scale against the single-module parity contract
(`tests/test_fused_train_parity.py`): the same lane measures 3–5e-3 per-module and
~5e-2 composed — a per-module result is not a composed one, in gradients as in perplexity.
All loss trajectories sit at ≤0.0020 median |Δ|, far inside the 0.05 band the fused-train
gate registered, so nothing here indicts any lane for training use.

Open question this does NOT answer: composed error is measured against the bf16
*reference loop*, not against fp32 truth. Per-module, the fused path measured *closer* to
truth than the reference; whether the composed 5e-2 is "worse than reference" or merely
"different from reference, equally far from truth" needs an fp32-compute truth arm.

**3. The 24x was a toy-shape artifact — the speed ranking inverts at scale.** At the
A2000 microbench shape (hidden 512), `batched` wins at 24x. At 48-layer/30B width it is
**1.05x — no speedup — at the highest peak memory of any arm**, because at real width the
matmuls dominate rather than Python launch overhead, and the whole-stack dequant it pays
on every forward AND backward stops being amortizable. `fast_train(dgrad=True)` is the
fastest real-scale option at 2.52x. The README guidance shipped on 2026-08-06 morning
("VRAM to spare → `enable_batched_train`") was wrong at the scale people actually train,
and was corrected in the same PR that adds this file.

## Cost

Two pods, ~30 min total: A5000 $0.27/hr + A6000 $0.33/hr ≈ **$0.30 all-in**. Both
terminated by the driver's teardown trap; zero orphans
([`pod-orchestration`](../../deploy/) discipline).

## A vacuous check this campaign caught in itself

The first 16-layer run reported "frozen storage OK" having hashed **zero tensors**: under
`offload=True` the module buffers are 0-element placeholders and the real bytes live in
the offload handle's `home` dict, which the walker skipped. The flipped-byte control
passed by flipping a byte in an unrelated staged tensor. Both fixed — the walker reads
`handle.home`, the control flips a byte in the same storage the check reads, and hashing
zero tensors is now a hard abort rather than an OK. The 48-layer numbers above are from
the fixed check; the 16-layer frozen column is vacuously true and labelled as such in its
receipt (`frozen_tensors_hashed: 0`).
