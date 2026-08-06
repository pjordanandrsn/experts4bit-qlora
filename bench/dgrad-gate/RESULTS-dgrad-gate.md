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

**ANSWERED by the truth-arm rerun (same day):** a fifth arm — the reference loop with
`compute_dtype=fp32` over the same NF4 bytes — scored every bf16 arm against fp32 truth
([16L receipt](gate_16L_olmoe_truth_a6000.json), [48L receipt](gate_48L_qwen30b_truth_a6000.json)):

| arm | vs bf16 reference | vs fp32 truth (16L) | vs fp32 truth (48L) |
|---|---|---|---|
| reference loop | — | 3.41e-02 | 5.18e-02 |
| `fast_train` | 5.05e-02 | **2.95e-02** | 5.13e-02 |
| `fast_train_dgrad` | 5.04e-02 | **2.93e-02** | 5.13e-02 |
| `batched` | 3.71e-03 | 3.41e-02 | 5.18e-02 |

**The "13x looser" framing was a metric artifact.** Against truth, the fused lane is
*closer* than the reference at 16 layers and equal within 1% at 48. The vs-reference
metric measures how *similarly an arm rounds to the reference*: the batched path's tiny
vs-reference number reflects that it shares the reference's dequantize-then-matmul
structure — same rounding, not more accuracy — while the fused kernel's fp32-in-register
accumulation rounds differently and lands at least as close to truth. The composed bf16
noise floor is ~3.4e-2 at 16 layers and ~5.2e-2 at 48, and every lane sits on it.
No lane is looser than any other in the sense that matters.

**3. The 24x was a toy-shape artifact — the speed ranking inverts at scale.** At the
A2000 microbench shape (hidden 512), `batched` wins at 24x. At 48-layer/30B width it is
**1.05x — no speedup — at the highest peak memory of any arm**, because at real width the
matmuls dominate rather than Python launch overhead, and the whole-stack dequant it pays
on every forward AND backward stops being amortizable. `fast_train(dgrad=True)` is the
fastest real-scale option at 2.52x. The README guidance shipped on 2026-08-06 morning
("VRAM to spare → `enable_batched_train`") was wrong at the scale people actually train,
and was corrected in the same PR that adds this file.

## Real-data trajectory gate (same day, [receipt](traj_20step_alpaca_a6000.json), driver [`traj_gate.py`](traj_gate.py))

The complementary instrument to step-0 gradient parity: 20 optimizer steps on Alpaca
(the dataset METHODOLOGY's own eval uses), held-out slice the optimizer never sees, one
load, adapter restore between arms. Band: train median |Δ| ≤ 0.05, per the fused-train
gate. Qwen3-30B-A3B, 48 layers, A6000:

| arm | s/step | eval before → after | Δeval vs ref | train med Δ | verdict |
|---|---|---|---|---|---|
| reference | 9.91 | 2.5550 → 1.6097 | — | — | — |
| `fast_train_dgrad` | **3.46** | 2.5437 → 1.5913 | −0.0185 | 0.0159 | **PASS** |
| `batched` | 9.04 | 2.5533 → 1.5872 | −0.0225 | 0.0169 | **PASS** |

Both accelerated lanes train equivalently on real text — held-out eval lands marginally
*better* than the reference (within noise), train-loss deltas sit at a third of the band —
and the dgrad lane does it at **2.87x** the reference's real-data step rate. The
`eval before` spread (±0.011) is the forward-path rounding difference between lanes,
visible before any optimizer step; consistent with the truth-arm finding that each lane is
its own valid bf16 rounding.

## sm_120 (Blackwell) verification — [receipt](sm120_pro4500_sweep.json)

Every number above is sm_86. Verified on an RTX PRO 4500 Blackwell (`capability (12, 0)`
— same arch as the RTX 5090 the #38 contributor runs; B200/B300 are sm_100, a different
one): gnf4's kernel tests at the `v0.7.0` tag **66 passed**
([log](sm120_gnf4_tests.log)), e4b's parity contract + batched + V4 suites at `v0.11.0`
**29 passed** ([log](sm120_e4b_tests.log)), and the full tile sweep:

| shape | dgrad (default cfg) | oracle loop | speedup | fwd ceiling | relerr |
|---|---|---|---|---|---|
| gate_up E=256 | 0.83 ms | 55.7 ms | **67x** | 0.91 ms | 3.21e-03 |
| down E=256 | 0.48 ms | 48.9 ms | **103x** | 0.57 ms | 2.61e-03 |

`_DGRAD_DEFAULT` (tuned on sm_86) **holds on sm_120** — the swept best on both shapes is
within timing noise of the default, and every config produced identical output. The
speedup over the Python loop is far larger than on sm_86 (67–103x vs 10–26x): the faster
the card, the more the loop's launch overhead dominates, which is the whole thesis of the
kernel. dgrad again runs at ~0.9x of the forward kernel's time.

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
