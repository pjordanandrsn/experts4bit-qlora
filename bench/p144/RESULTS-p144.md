# P144 results — layer-major on Mixtral-8x7B under expert offload, RTX 4090: LICENSED

**Verdict: LICENSED, S = 2.118** (`p144-4090-1`, 2026-10-18). On Mixtral-8x7B-Instruct with expert offload and
`enable_fast_train(dgrad=True)`, on an RTX 4090, `layer_major_step` runs the registered training step at 2.118× the
speed of the stock accumulation loop. The bar was S ≥ 1.10. The maintainer re-derived the verdict independently and got
a byte-identical `verdict.json` (sha256 `201b71c6…`).

Layer-major now reads above 2× on two model families and two cards: P137 2.365 (Qwen3-30B-A3B, RTX 4090), P140 2.311
(Qwen3-30B-A3B, RTX 5090), and P144 2.118 (Mixtral-8x7B, RTX 4090).

Registration: [PREREG-p144.md](PREREG-p144.md) (#1760). It is P137's kit at its Amendment 3, by named substitutions;
every STRICT term is P137's.

## What was run

- **Subject:** e4b `a9d140d8` (0.55.0, main with #1757's Mixtral support and #1752's thread cap); grouped-nf4-gemm
  0.46.0 `6b7c2e74`; torch 2.8.0+cu128; transformers 5.18.0; bitsandbytes 0.50.2.
- **Lane source:** e4b `a2f630ee` (#1760's merge).
- **Recipe and arms:** P137's. Mixtral-8x7B-Instruct-v0.1 @ `eba92302`, all 32 MoE layers offloaded and patched.
  Gradient checkpointing is enabled first, then routed by `enable_fast_train` through e4b's reentrant checkpoint on all
  32 layers. Seq 2048, micro-batch 2 × accum 4, r 16 / α 16, AdamW 8-bit, lr 2e-4, seed 3407. ABBA in one process after
  one load and an untimed warm-up of both loops; 2 warm and 32 timed steps per arm.
- **Box:** one rented RTX 4090 (sm_89, 24 GB, driver 615.71.09) on a Vast.ai verified/secure host: machine 11541, AMD
  EPYC 7B13 (256 CPUs, a 61.2-CPU container quota), pinned host-to-device ceiling 17.57 GB/s. The 93 GB
  checkpoint loaded in 64.8 s. Cost $1.167, of which $0.816 was the 94.9 GB download.

## Numbers

| arm | median s/step (32 timed) | stagings, step 0 / steady | peak allocated | held-out at N |
|---|---|---|---|---|
| A0a (stock) | 11.386 | 253 / 252 | 5.72 GiB | 0.9662 |
| L1a (layer-major) | 5.372 | 64 / 63 | 5.64 GiB | 0.9643 |
| L1b (layer-major) | 5.375 | 64 / 63 | 5.64 GiB | 0.9670 |
| A0b (stock) | 11.386 | 253 / 252 | 5.72 GiB | 0.9749 |

- **The gain.** S = min(A0a/L1a, A0b/L1b) = min(2.1195, 2.1183) = **2.118**.
- **Self-pairs:** 1.0000 (stock) and 1.0005 (layer-major), against the [0.97, 1.03] band.
- **STRICT rungs, all clean:**
  - the warm-up record `{stock 64, layer_major 224}` is identical in all four arms;
  - step 0 is cold (253 and 64);
  - steady stagings are exactly 252 and 63;
  - step 0 is bit-identical across the four arms;
  - every arm starts from the snapshot (arm-start memory within 1 KiB of the baseline);
  - the checkpoint is reentrant on 32/32 layers in every arm;
  - layer-major's peak is below the stock loop's.
- No timed step was above 12.03 s (stock) or 6.01 s (layer-major).

## Predictions, scored

- **Q1 HELD.** Every arm fit in 24 GB; the largest peak was 5.72 GiB. An arithmetic estimate made before the run put the
  stock peak at about 8–10 GiB: non-expert bf16 weights about 3.2 GB (attention 1.34 B parameters, embeddings and head
  0.26 B), two staged expert layers about 1.6 GB, the expert LoRA (r 16, 210 M fp32) with its gradients and 8-bit AdamW
  about 2.1 GB, and activations, logits and the CUDA context about 1–2 GB. The measured peak is 3–4 GiB below that. The
  allocator held 3.40 GiB at every arm's start, so a step adds about 2.3 GiB. The estimate is not reconciled term by
  term here; the measured number is the record.
- **Q2 HELD.** Step 0 is bit-identical across the four arms, and stagings are exactly 253 / 252 and 64 / 63.
- **Q3 HELD.** S = 2.118 is inside the a-priori band [1.5, 2.6], near its point of about 2.0.
- **The host-link model:**
  - A0 = C + 4T and L1 = C + T fit at T = 2.004 s and C = 3.369 s, an effective rate of 49.9 GB / T = 24.9 GB/s.
  - That is above the box's 17.57 GB/s pinned ceiling, so some copy overlaps compute, as on P140's 5090.
  - Both parameters landed at the favourable ends of the registered ranges (T 2.0–2.9 s, C 3–8 s). Compute C is 1.9×
    P137's 1.743 s, not the several times the 13 B / 3.3 B active-parameter ratio suggested.
- **Q4 HELD.** L1's peak is 0.08 GiB below A0's, inside the 4.25 GiB term.
- **Q5 HELD (REPORTING).** The held-out losses are L1 0.9657 and A0 0.9705, |Δ| 0.0049 against the 0.01 term. The
  same-loop spreads are 0.0088 (stock) and 0.0027 (layer-major).
- **Q6:** LICENSED, as predicted at about 85 %.

## The memory term, re-derived for Mixtral

P137's 4.25 GiB term was derived from Qwen3's dimensions as (M − 1) × [checkpointed layer inputs + saved fp32
log-probs while the chunked-loss gate is off]. Re-derived for Mixtral at P137's worst case of 2 × 2048 tokens:
- checkpoints: 32 × 4096 × 4096 × 2 B = 1.0 GiB;
- log-probs: 4096 × 32000 × 4 B = 0.49 GiB;
- so 3 × 1.49 = 4.47 GiB, above the term.

On the registered slice with Mixtral's tokenizer at `eba92302`, the widest micro-batch over the 34 steps is 2 × 463 =
926 tokens. The extra is then 3 × (0.226 + 0.110) = 1.01 GiB. The term held by about 3.2 GiB on this data, not by its
Qwen3 derivation; a Mixtral lane on longer sequences needs the term re-derived before it is used as a STRICT bound.

## The thread cap in the field

#1752's cap worked on exactly the host class that cost 2,078 s on P137. On an AMD EPYC 7B13 with 255 CPUs visible under a
61.2-CPU quota, the load log reads `host-side fuse: intra-op threads 255 -> 62`, and the 93 GB Mixtral loaded in 64.8 s.

## Scope of the licence

The licence covers exactly this configuration:
- `layer_major_step` with `enable_fast_train(dgrad=True)` and expert offload;
- Mixtral-8x7B (`MixtralForCausalLM`, router jitter 0);
- the RTX 4090 class (sm_89);
- P137's recipe.

From the CLI trainer, that is `OFFLOAD_EXPERTS=1 LAYER_MAJOR=1`, which implies `FAST_TRAIN=1` (#1744). No default
changes.

## Receipt

[`receipts/p144-4090-1/`](receipts/p144-4090-1/SHA256SUMS) holds the four arm records, `verdict.json`, `summary.txt`,
`versions.txt`, `forensics.txt`, the logs and the launcher's teardown proof, listed in `SHA256SUMS`. The data slice is
identified by its sha256 and not committed.
