# Results — P49: which store survives Gemma-4's layer 0 (H100 NVL, 2026-09-19)

Pre-registration: [`P49-PREREG.md`](P49-PREREG.md) (+ amendment 1: the activation probe's own bug, below). Run `p49-gemma4fmt` (H100 NVL, vast 51572740, e4b `e4f25ad`, gnf4 v0.32.0), receipt `receipts/experts4bit-qlora/2026-09-19/p49-gemma4fmt/p47/`, read by `bench/p49/p49_reduce.py`. K0 passed; control (i) refused the decode scorer again (reference self-KL 0.2787) → prefill on both sides, the same cached bf16 reference and 200 prompts / 5,879 tokens as P44-b / P47 / P48. Every row carries exactly 1 quantised / 29 bf16 expert stacks AND the store the builder named (`kl_serve._builder_check`).

## The stores, on layer 0 alone

| arm | store | KL nats/token | top-1 | general / technical / code / longctx |
|---|---|---|---|---|
| `L00_int8` | int8, blockwise 64 | **0.6928** | 0.729 | 0.549 / 0.857 / 0.977 / 0.523 |
| `L00_fp8` | fp8 (e4m3), blockwise 64 | 0.7744 | 0.699 | 0.792 / 0.983 / 0.975 / 0.606 |
| `L00_nf4b32` | NF4, block 32 | 0.8684 | 0.686 | 0.769 / 1.007 / 1.122 / 0.712 |
| `L00_nf4` | NF4, block 64 (the P48 anchor) | 0.8916 | 0.680 | 1.078 / 1.043 / 1.111 / 0.683 |
| `L00_fp4` | FP4, block 64 | 1.0514 | 0.652 | 1.106 / 1.342 / 1.260 / 0.852 |

For scale: the bf16 reference against itself under the other forward shape is 0.279, gpt-oss's NF4 requant on this instrument is 0.022, and **NF4 on layer 27 alone is 0.0056**.

## The same store (NF4 b64) at four depths — P48's profile, re-measured here

| arm | KL | top-1 |
|---|---|---|
| `L07_nf4` | 0.4436 | 0.767 |
| `L14_nf4` | 0.1007 | 0.891 |
| `L21_nf4` | 0.0144 | 0.952 |
| `L27_nf4` | 0.0056 | 0.970 |

Identical to P48's rows to four digits on a different box — the profile reproduces exactly.

## Verdicts

- **P0 (anchor) — HOLDS**: `L00_nf4` 0.8916, P48's 0.892.
- **P1 (int8 survives layer 0) — REFUTED, and this is the lane's result.** `L00_int8` reads **0.6928** against a registered ≤ 0.02 for "survives" and ≥ 0.20 for "does not". **Sixteen times finer quantisation buys a 22 % reduction in KL** (0.892 → 0.693) and leaves the model 124× further from the checkpoint than NF4 on layer 27. Under the registered decision rule: **Gemma-4's early expert layers cannot be quantised at all by any store e4b ships — they stay bf16.**
- **P2 (fp8 ≤ 0.05) — REFUTED**: 0.7744.
- **P3 (fp4 is no better than NF4's class) — HOLDS**: 1.0514, in fact worse than NF4 (its codebook is the wrong shape for these weights, as everywhere else).
- **P4 (block size does not rescue) — HOLDS**: block 32 reads 0.8684 = 0.97× block 64 (rule ≥ 0.55×). Halving the block halves the scale-sharing and buys 2.6 %.
- **P5 (post-norm amplification) — NOT READ**: the activation probe failed before its first forward on a bug of this lane's own (`re.match` against dotted module names — amendment 1). Redrawn as `p49-gemma4fmt-2`, probe only.

## What this says about #597

**The store is not the variable.** Across a 4-bit codebook (NF4), a different 4-bit codebook (FP4), half the block size, an 8-bit integer store and an 8-bit float store, layer 0's KL moves between 0.69 and 1.05 — while the SAME NF4 store at layer 27 reads 0.0056. A 16× finer grid changing the answer by 22 % is the signature of a quantity that does not scale with quantisation error at all. Combined with P48's two weight-space probes (block-64 NF4 does the same ~9 % relative damage in every layer, per expert and per block), the conclusion is forced: **what makes layer 0 special is what happens to its expert outputs downstream, not how its weights are stored.** P5's probe is the instrument for that and is redrawn.

The remedy for e4b is therefore the one the registered rule names for a refuted P1: **keep Gemma-4's first k expert layers in bf16** (`quantize_layers` = all but 0..k−1, which the loader already supports and P47/P48 exercised). The cost is ~1.5 GB per layer against ~0.4 GB at NF4; P48's profile says k ≈ 10 covers 93 % of the gap and k ≈ 15 covers 96 %. That default belongs to a lane with its own K8 + KL gate — **nothing about Gemma-4 is quotable until it passes one**, and no such default ships from this lane.

## Cost

≈ 23 min on an H100 NVL at ≤ $3.10/h ≈ $1.20 against a registered $4.65; the probe redraw is ≈ $1.
