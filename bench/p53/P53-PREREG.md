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

## Amendment 1 (2026-09-20 13:05Z, before any P53 arm has produced a reading): the box class widens to any single ≥80 GB HBM card

**Why.** Seven draws have produced no measurement, and the last two failures are the market rather than the lane. A read-only probe using the launcher's **own** `offer_filter` (imported, not hand-written — my first hand-written probe omitted a constraint the launcher applies and was therefore not evidence about its behaviour) shows the qualifying pool at the registered floors:

| class, disk ≥ 320 GB | machines | cheapest |
|---|---|---|
| **H100 NVL** (registered) | **2** | \$2.87/h |
| H100 SXM | 4 | \$3.14/h |
| H200 | 2 | \$4.74/h |
| A100 80 GB PCIe / RTX 6000 Ada / RTX PRO 6000 | 0 | — |

Both H100 NVL machines are now known-bad from today's draws: **34985** failed ssh authentication for 600 s (draw 2) and **57775** sat in `loading` past 600 s (draw 7). Redrawing the registered class means redrawing those two, so continuing is not patience, it is paying for the same two faults.

**The change.** The class becomes **any single card with ≥ 80 GB HBM** — H100 NVL, H100 SXM or H200 — and the rate ceiling rises **\$3.10 → \$3.30/h** so H100 SXM's \$3.14 is reachable. Estimate becomes ~\$8.25 for 2.5 h, still inside the \$2–20 band and far inside the \$35/run cap.

**Why this does not weaken the reading, stated so it can be checked rather than trusted.** The instrument is KL against the model's own bf16 checkpoint, and **all three arms run on the same box in the same lane**, so every comparison this lane makes is internal. Cross-box absolute comparability is *not* claimed and is known not to hold here (`finding_licensed_pack_does_not_reproduce_across_boxes`) — which is exactly why `nf4_uniform` is **re-measured as an arm** rather than cited from P50's 1.0837. P50's number is context; the comparator is in the run.

**What would make this amendment wrong**, and is therefore worth saying out loud: if the calibration path's numerics differed by card in a way that changed the *ordering* of the three arms, a class change would matter. Nothing measured says it does, and P47–P52 read this family on H100 NVL while the training-parity lanes read it on RTX 5090 without either instrument's ordering moving. That is an argument from adjacent evidence, not a proof, and if the arms land within noise of each other the class is one of the things to suspect.

**Nothing else moves.** Same fixture, same three arms, same single variable (calibration order), same instrument, same prompts, same bar (≤ 0.10 nats, top-1 ≥ 0.93), same predictions P1–P3, and P3 remains deliberately unpredicted.

## Amendment 2 (2026-09-20 14:00Z, before any P53 arm has produced a reading): the egress floor is re-derived for THIS lane's fetch, and the window widens to keep it honest

**The observation.** Sixteen draws, no measurement. The last eight all died on the same box-side check, and the numbers are the point:

| draw | 9 | 10 | 11 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|
| HF CDN MB/s | 15.9 | 15.8 | 19.0 | 18.3 | 18.0 | 17.7 | 18.3 |

Floor: **20 MB/s**. Every host in the pool sits in a tight band just under it. Eight hosts failing the same check by 5–20 % is a mis-calibrated floor, not eight bad hosts.

**Why the floor is wrong HERE.** `P47_MIN_MBPS` defaults to 20 for a family of lanes; tp4's box C fetches ~142 GB of checkpoints, where 20 MB/s is already two hours. **P53 fetches one model**, Gemma-4-26B-A4B at ~52 GB. The floor was inherited, not derived for this lane.

**The derivation, stated so it can be checked rather than trusted.** The floor exists to answer one question: *will the fetch leave enough window to do the work?* Budget the fetch at no more than **40 %** of the window, leaving 60 % for calibration and three arms' scoring:

```
window 3.5 h = 210 min      fetch budget = 84 min
52 000 MB / (84 x 60 s)  =  10.3 MB/s   <- the requirement
```

So the principled floor at a 3.5 h window is ~10 MB/s. **This amendment sets it to 15**, about 1.5x the requirement, so it still refuses anything materially slow. At the worst rate observed (15.8) the fetch is 55 min, **26 % of the window**.

**The window widens 2.5 h → 3.5 h, and that is not incidental.** Lowering a floor without widening the window would trade a refusal for an overrun — the same lane failing later and more expensively, having paid for the download first. The floor protects "can this finish"; if the floor drops, the window must absorb it or the protection is fake. Estimate becomes 3.5 h x \$3.30 = **\$11.55**, still inside the \$2–20 band and the \$35/run cap.

**The obvious objection, and why I think it does not hold.** Lowering a threshold after it repeatedly refuses is the exact shape of fitting a gate to the outcome one wants, and I have no standing to be casual about it — I published a bar earlier this week whose registering commit merged *after* its run. Three things distinguish this: the quantity is **computable rather than judgmental** (bytes over time against a window, not a quality call); the new number is **derived from the lane's own fetch size and window**, and lands at 1.5x the requirement rather than just below the observations (15 vs a worst observation of 15.8 — if the floor were being fitted to admit what I drew, it would sit at 15.7); and **no bar that bears on the result moves** — KL, top-1, the 0.10/0.93 band and P1–P3 are untouched. A slow host makes this lane *take longer*, not read differently.

**What would make this wrong:** if egress rate correlated with anything the instrument measures. It does not — the arms are scored after the fetch, from local weights, and all three share the box.

**Unchanged:** fixture, three arms, the single variable (calibration order), instrument, prompts, bar, and predictions P1–P3, with P3 still deliberately unpredicted.
