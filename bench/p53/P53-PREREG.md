# P53 — can calibration make Gemma-4's downstream layers absorb early expert error?

Successor lane to [#636](https://github.com/pjordanandrsn/experts4bit-qlora/issues/636). Written before any P53 data exists.

## The question, and why the obvious answers are already dead

P49's activation probe settled what this defect is **not**. Quantising any single Gemma-4 expert layer damages that layer's branch output by the same 8–10 % relative wherever it sits (`rel_err_raw` 0.0793–0.0962, a 1.21× spread), and the following norm does not amplify it — at layer 0 it *attenuates*, 0.79×. Spearman(KL, post-norm error) is **−0.900**: the post-norm error grows with depth while the end-to-end KL collapses, 0.892 → 0.0056.

> the end-to-end effect is not a property of the perturbation's size at all; it is a property of how many layers the perturbation is processed by afterwards

And the branch output is already reproduced to **cosine 0.997** while the model still moves 0.9 nats.

So every approach that makes the early store *more faithful* — activation-aware scaling, per-channel scales, a finer grid — is answering a question that has been measured not to matter. P49's store sweep found this directly: int8 buys 22 %, a 16× finer grid buys 2.6 %, and layer 0 stays 124× worse than layer 27 under the same store.

**The remaining axis is the downstream, not the perturbation.** If the cost is that 29 layers read a perturbed stream, the lever is making those 29 layers expect it.

## The precedent this lane transfers

On Qwen3-30B (c4val1, 2048 steps, same box, same 8 batches, damp 0.01), GPTQ calibration order alone decided the result:

| order | nll | Δppl | verdict |
|---|---|---|---|
| all-at-once (every layer's Hessians against the NF4 prefix, pack after) | 2.81222 | +0.150 | FAIL |
| streamed (passes of ≤10 layers: calibrate chunk, pack it, the next chunk sees int4) | 2.80011 | **−0.050** | PASS, better than NF4 |

The streamed path exists because Mixtral's all-layer Hessians OOM-killed a container; **the quality effect was discovered, not designed**, which is a point in its favour as a mechanism rather than a tuned result.

It attacks the right axis: it does not reduce the error, it re-fits what comes after it.

## Why this must be run on a fully quantised model

Sequential calibration can only help where there is a quantised downstream to adapt. P49's rows quantise **one** layer against 29 bf16 ones — nothing downstream to absorb anything, so that diagnostic cannot test this. The arms below quantise **all 30** expert layers, which is also the configuration P50 measured at **1.0837 nats** and rejected.

## Arms

All on `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, all 30 expert layers quantised, everything else bf16.

| arm | store | order |
|---|---|---|
| `nf4_uniform` | NF4 b64 (RTN) | n/a — the registered baseline, P50's k=0 |
| `int4_allatonce` | GPTQ int4 | `enable_serve_experts_int4_calibrated`, `layers_per_pass = 30` |
| `int4_sequential` | GPTQ int4 | same call, `layers_per_pass = 10` (the Qwen3 recipe's shape: passes of ≤10) |

Calibration batches, damping and `min_rows` are held identical across the two calibrated arms — **order is the only difference**, as in the Qwen3 measurement.

## Instrument

`bench/p44/kl_serve.py`: full-vocabulary fp64 KL against the bf16 checkpoint, scorer chosen by control (i), K0-gated, one child process per arm, per-row census. The 200 committed prompts, for comparability with P47–P52's rows. Gemma-4's decode scorer has been refused by control (i) in every prior lane (self-KL 0.2787) and is expected to be refused again; prefill is the working instrument. **K8's two-text gate cannot be built on this family** (`e4b.parity.gemma4.no-reference`) and is not claimed.

Every arm's expert-stack census must match its tier spec or the row is void (`kl_serve._builder_check`).

## Predictions, registered before the data

- **P1** — `int4_sequential` reads a **lower KL than `int4_allatonce`**. The Qwen3 ordering effect transfers to Gemma-4's experts. Registered alternative: they land within 5 % of each other, which would say the ordering effect is family-specific and does not generalise — a real and publishable negative.
- **P2** — `int4_sequential` reads a **lower KL than `nf4_uniform`'s 1.0837**. If calibration cannot beat round-to-nearest on this family, the axis is closed and #636 needs a different idea.
- **P3** — **the magnitude is deliberately NOT predicted.** Whether sequential calibration reaches the band e4b's shipped configurations occupy (≤ 0.10 nats, top-1 ≥ 0.93) is the question; naming a hoped-for number is what would make the reading worthless. The bar itself is fixed here, before the measurement, and is the same bar P52 applied.

## What this lane cannot say

Nothing about training throughput, nothing about the fused-vs-dense training control (`e4b.parity.gemma4.train-internal`, which is a separate instrument), and no competitive position against any other framework. It is a serving-quality reading on one family.

## Cost and governance

One H100 NVL, est. 2.5 h at the class's observed rate ≈ **$6.50**, inside the $2–20 band and the $35/run cap. Calibration adds Hessian accumulation over the registered batches to the fetch-and-load cost the P47–P52 lanes already measured at 12–32 min.

This document merges to `main` **before** the launch, and the launch cites the commit. Timestamps, not assertions.
