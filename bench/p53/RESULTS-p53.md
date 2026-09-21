# Results — P53: calibration does not rescue Gemma-4's experts (H100 SXM, 2026-09-20)

Pre-registration: [`P53-PREREG.md`](P53-PREREG.md) + amendment 1 (box class) + amendment 2 (egress floor, window). Run `p53-gemma4calib-23`, receipt `receipts/experts4bit-qlora/2026-09-20/p53-gemma4calib-23/`, **$3.14**, 68 min. K0 all_passed; control (i) refused the decode scorer again (reference self-KL **0.3765** against a 0.01 threshold) → **prefill on both sides**, as in every prior Gemma-4 lane. 200 committed prompts, 5,879 tokens, full-vocabulary fp64 KL against the bf16 checkpoint.

## The rows

| arm | store | `layers_per_pass` | KL mean | KL median | top-1 |
|---|---|---|---|---|---|
| `nf4_uniform` | NF4 b64 (RTN) | — | **1.0772** | 0.3841 | 0.6438 |
| `int4_allatonce` | GPTQ int4 | 30 | 1.1050 | 0.3683 | 0.6423 |
| `int4_sequential` | GPTQ int4 | 10 | **1.1564** | 0.4081 | 0.6299 |

All 30 expert layers quantised in both calibrated arms. Calibration batches, source and Hessian budget identical; **order was the only difference**, and the receipt confirms it reached the box as such (`E4B_CALIB_LAYERS_PER_PASS` 30 vs 10, `E4B_SERVE_EXP_INT4=1` and `..._CALIB=1` on both, `0`/absent on the baseline).

## Verdicts

- **P1 — REFUTED, and in the wrong direction.** Sequential was predicted to beat all-at-once. It is **worse**, 1.1564 vs 1.1050, a 4.7 % gap — outside the ±5 % band registered as the "no effect" alternative, on the wrong side. The Qwen3 ordering effect does not transfer to Gemma-4's experts; it **inverts**.
- **P2 — REFUTED.** Neither calibrated arm beats round-to-nearest. NF4 at 1.0772 is the best of the three, and the more principled calibration is the worst.
- **P3 — HELD, and it is why this reading is worth anything.** Only *readability* was predicted; the magnitude and direction were explicitly left unregistered, on the stated grounds that naming a hoped-for direction is what makes a reading worthless. Had "sequential wins" been registered, this page would be explaining away a result that is simply clear.

## What it means for #636

**Both axes I could name are now closed.**

- **Reduce the perturbation** — refuted by P49: the branch output is already reproduced to cosine 0.997 and the model still moves 0.9 nats; int8 buys 22 %, a 16× finer grid 2.6 %, and layer 0 stays 124× worse than layer 27 under the same store.
- **Make the downstream absorb it** — refuted here. It is the only lever that attacked P49's actual mechanism, and it made things worse.

So Gemma-4's positional sensitivity is not addressable by anything currently in this toolbox. The only lever that works remains keeping early expert layers in high precision, and that lever is **memory** (P50: 20 of 30 layers bf16 to reach other families' floor, 76 % of the all-bf16 expert store).

**Nothing is licensed and no position is quoted.** The register gains one row recording the refutation.

## Why sequential might be worse, offered as a hypothesis and NOT tested here

Sequential calibration fits each chunk against the already-quantised prefix, so downstream layers absorb upstream error. On Qwen3 that helped. On Gemma-4 the early-layer error is **159× more consequential than late-layer error** (P49), so fitting layers 10–29 to a corrupted layer-0 stream may be teaching them to compensate for a signal that dominates everything after it — trading a correctable error for a baked-in one. That is a story, not a finding. It would need its own lane and its own pre-registration, and this page does not claim it.

## Caveats a reader should carry

1. **One box, one seed, one prompt set.** The three arms are internally comparable — same box, same tokens, same reference — but the gaps here (2.6 % and 4.7 %) are smaller than the gaps P50/P51 worked with, and no repeat was run.
2. **The decode scorer has never been usable on this family**, here or in any prior lane. This is a prefill reading, and Gemma-4's own reference disagrees with itself by 0.3765 nats across scorer shapes.
3. **This says nothing about calibration elsewhere.** The Qwen3 result it was transferring from stands; what failed is the transfer.

## Cost, including what it took to get one measurement

`p53-gemma4calib-23` cost **$3.14**. Twenty-two earlier draws produced **no measurement** and cost **$14.30** between them: ten refused by a mis-calibrated egress floor (fixed in amendment 2), two by ssh-auth host faults, two by my own wrong-exclusion-knob manifests, two by an empty market, one by a 100 GB instance overlay, one stuck loading, one by a staged-digest mismatch I caused, one by a head mismatch I caused, and one — the most expensive single failure at $2.09 — by a family name that `kl_serve` did not know, which failed **after** the 52 GB fetch (e4b#647).
