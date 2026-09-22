# Results — P56: Gemma-4's training-parity FAIL is not a defect in the fused path

Pre-registration: [`P56-PREREG.md`](P56-PREREG.md) (merged `d0f8fde`, 2026-09-21; erratum `706942a`).
Run **`p56-gemma4-ladder-3`**, one RTX 5090 (Vast verified-secure, instance ssh7.vast.ai:25378),
e4b `2555cf3`, grouped-nf4-gemm pinned at **`9206352f` (v0.32.1)** — the cut the standing claim row
was measured on. Read by [`p56_reduce.py`](p56_reduce.py); nothing below is hand-transcribed.

## The ladder

Four arms, one reference, one box, one `init_sha`, same tokens, 20 steps each, ordered by the per-op
error each carries against the reference (`bench/dgrad-gate`):

| arm | composed grad err | final train loss | Δ final train | median step \|Δ\| | Δ held-out | s/step |
|---|---|---|---|---|---|---|
| `reference_attn4` | 0 by definition | 1.18187 | — | — | — | 37.968 |
| `batched_attn4` | **3.79e-03** | 1.12762 | **0.05425** | 0.08487 | 0.08595 | 3.809 |
| `fused_attn4_nodgrad` | forward fusion | 1.07913 | **0.10274** | 0.11678 | 0.18265 | 3.397 |
| `fused_attn4` | **4.97e-02** | 1.07964 | **0.10223** | 0.10639 | 0.16378 | 3.260 |

All four VALID: identical `init_sha`, identical tokens sha, 487,280,640 trainable on every arm,
`n_patched` 30 on each accelerated arm, C1 frozen bytes bit-exact, 20 steps each.
**The batched rung engaged on every call — 7680/7680, zero fallbacks, at `pad_waste_limit` 64.**

## What it says

**The gap is not the fused kernel's.** Every accelerated path misses the 0.05 band, including the
**kernel-free** one, which never touches `grouped-nf4-gemm` at all. A defect localised to that kernel
cannot produce that.

**Cutting the arithmetic error does not buy the band.** The batched path carries **13× less** composed
gradient error than the fused path and closes only **1.9×** of the gap — 0.05425 against 0.10223 —
still **1.7× outside** the band on the median step-wise criterion. Extrapolating that scaling, no
achievable reduction in per-op error brings this model inside 0.05. The band is not reachable by
fixing arithmetic.

**The standing row reproduces.** `e4b.parity.gemma4.train-internal` records 0.08257 final / 0.12421
median. This draw reads **0.10223 / 0.10639** on a different box, a different e4b commit and a
third independent measurement. **P3b (`DID-NOT-REPRODUCE`) is refuted** — the FAIL is real and stable.

## What it does NOT say, including against my own reading

**The arithmetic is not irrelevant, and the registered "flat" pattern fired only just.** The deltas
are **monotone in per-op error** (reference → batched → fused), every accelerated arm sits on the
**same side** of the reference (all lower), and the two fused arms — whose per-op errors differ by
0.02e-2 — agree with each other to **0.0005**. That is a systematic, ordered effect, not scatter.
The reducer's flat test (`hi ≤ 2·lo`) passed with **5.6 % headroom**: 0.10274 against 0.10850. Had
the batched arm read 0.0512, or the worst rung 0.109, the verdict would have been `MIXED`.

So the honest statement is narrower than "it is a floor": **the divergence is real, ordered and
strongly sublinear in the arithmetic error, and the band cannot discriminate a defect on this model
because the smallest perturbation available already fails it by 1.7×.**

**Lower training loss is not "better" here.** All three accelerated arms end below the reference.
This project already records why that reading is a trap (`finding_quantisation_noise_flatters_ood_perplexity`);
it is a different trajectory, not a better one.

**Nothing is quoted against another framework.** Unsloth and HF were skipped by registration.

## Consequence for e4b#558

The fused path is **not shown defective**, and refusing it for `gemma4_text` would cost a path
measured here at **×11.65 the reference's step rate** on no demonstrated defect. Neither is it
cleared: it is 2× outside a band that its kernel-free sibling also fails.

What the evidence supports is that **tp4's `parity()` compares against 0.05 and against zero, with no
floor term**, on a model whose own routed branch is uniquely amplified (`P56-PREREG`, and the released
checkpoint's 0.675× routed/dense norm weight). That is the same defect the serving side already fixed
by moving to nats against a measured floor (`finding_moe_router_flips_are_the_parity_noise_floor`).
**The training band needs the same treatment: a per-family floor, measured by the smallest-perturbation
arm, not a constant against zero.** This draw measures that floor for Gemma-4 for the first time:
**≥ 0.054 final / 0.085 median step-wise.**

## Draws and cost

| draw | outcome | cost |
|---|---|---|
| `p56-prove-1` | proving run; 120 GiB effective RAM vs the 49.9 GiB shard → GO-CLASS | $0.0086 |
| `p56-gemma4-ladder-1` | **measured nothing** — all four e4b arms rc=127 on an env-splice bug of mine (e4b#672) | $0.2863 |
| `p56-gemma4-ladder-2` | pre-flight refused the host at 22.0 MB/s < 40 MB/s floor, destroyed before any arm | $0.0513 |
| `p56-gemma4-ladder-3` | **the read above** | $0.4480 |
| | **total** | **$0.7942** |

Against the ≤$3 authorised for the lane. Every instance destroyed with proof; none left live.
