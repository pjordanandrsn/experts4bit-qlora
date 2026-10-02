# P97 — results: **SUPPORTED**. On one RTX 5090, through gnf4's fp8 kernel, the paged runner keeps each sequence's Gated DeltaNet state at transformers' own (7.7e-3 relative error before the first attention layer) and tracks transformers' forward on Qwen3.6-35B-A3B at 4.43e-3 nats, agreement 0.972

Registration: `bench/p97/PREREG-p97.md` (#903, `980d056`). Issue: #564. Code under test: #889 and #897, at e4b
`980d056`; grouped-nf4-gemm at `34da93d` (v0.34.1).

**Verdict by `p97_reduce.py`: `SUPPORTED`.** G1 and G2 hold on the subject, every VOID check passed, and the mutant
fails both gates. Under the registered consequence, `docs/SERVING.md`'s hybrid paragraph cites this as the GPU
reading, and decode-graph capture of the state gather and scatter is the next hybrid item.

| | Qwen3.6-35B-A3B (subject) | OLMoE-1B-7B (control) | gate |
|---|---:|---:|---|
| **G1:** pre-attention state, max relative error | **7.67e-3** | — | ≤ 5e-2 |
| **G2:** mean KL(reference ‖ paged), nats | **4.43e-3** | 7.15e-3 | ≤ 0.05 (both) |
| **G2:** argmax agreement | **0.9717** | 0.9736 | ≥ 0.85 (both) |
| prefill-step KL (step 0: chunked prefill, no fp8 read) | 2.66e-3 | 9.94e-4 | reported |
| max KL over the 1,024 steps | 0.152 | 0.0997 | reported |
| mean Δnll (paged − reference) | +0.0065 | +0.0003 | reported |
| subject KL / control KL | 0.62 | | reported |
| peak GPU memory | 23.05 GB | 5.26 GB | reported |
| eager decode, 4 rows (no graphs, torch Gated DeltaNet path) | 830 ms / step | 282 ms / step | reported |

**The mutant** (decode write-back rotated by one slot) read mean KL **4.05 nats**, agreement **0.186**, and
pre-attention state error **0.879**, failing both gates as it must. Its prefill-step KL equals the real pass's
(2.656e-3) exactly, because the rotation leaves single-row prefill untouched.

**Every check passed:**
- **The loaded checkpoints** were the registered commits: Qwen3.6 `995ad96`, OLMoE `7f1c97f`. The shape was 4 windows,
  512 + 256 tokens, 128-token chunks.
- **The layer plans:**
  - the subject read 40 layers, 10 attention on a 10-layer pool (map {3: 0, 7: 1, …, 39: 9}), and 30 linear, with a
    state record for each and pre-attention layers 0, 1, 2;
  - the control read 16 attention layers and no linear state.
- **Engagement, exact:**
  - decode-kernel calls 2,550 / 2,550 (subject) and 4,080 / 4,080 (control), over pool layers 0..L−1;
  - linear-state stores 8,130 / 8,130, over all 30 linear layers.
- **Not self-referential:** the control's mean KL is nonzero.
- **G2's bound is above fp8's own error:** the control passes it.
- **The premise** passed on this card before any fetch: `tests/test_linear_state_gpu.py`, 1 passed through the real
  kernel. The control's worst relative logit error was 2.03e-2 and the hybrid's 2.63e-2, tokens equal.

## The state, layer by layer (relative Frobenius error against transformers' cache, worst window)

| layer | conv window | recurrent state |
|---|---:|---:|
| 0 | 3.65e-4 | 2.36e-3 |
| 1 | 2.60e-3 | 5.93e-3 |
| 2 | 4.82e-3 | **7.67e-3** |
| after the first attention layer (27 layers) | | 1.06e-2 to 9.07e-2 (the largest at layer 38) |

The three layers before the first attention layer see the same tokens on both paths. Their error is bf16 arithmetic:
128-row prefill chunks against one 512-row prompt, and 4-row decode GEMMs against single rows. It grows from layer
0 to layer 2 as each layer's input inherits the one before. From layer 4 on, every linear layer's input carries the
fp8 KV's effect through the attention layers above it, so its state error is larger and is reported, not gated. On
CPU in fp32 (`tests/test_p97_box.py`), the pre-attention state agrees to 1e-7.

## Predictions, scored

| prediction (written before the data) | read | |
|---|---|---|
| SUPPORTED | SUPPORTED | HELD |
| G1 between 2e-3 and 2e-2, growing from layer 0 to 2 | 7.67e-3; 2.36e-3 → 5.93e-3 → 7.67e-3 | HELD |
| G2: subject mean KL between 1e-3 and 1e-2 | 4.43e-3 | HELD |
| both models agree on argmax at ≥ 0.93 | 0.9717 / 0.9736 | HELD |
| the subject's KL is 1–3× the control's | **0.62×** | **FALSIFIED** |
| the subject's prefill-step KL is of the same order as its mean | 2.66e-3 against 4.43e-3 | HELD |
| the mutant: KL > 1 nat, agreement < 0.6, state error > 0.3 | 4.05 / 0.186 / 0.879 | HELD |
| engagement exact; the linear-state pool 0.24–0.25 GiB | exact; 247.5 MB (0.242 GiB) | HELD |

**The falsified prediction.** The rehearsal read the subject at 1.78× the control. At the registered shape the
subject reads 0.62×. The two readings are not a controlled comparison: the rehearsal ran 2 windows of 128 + 16 tokens
in 64-token chunks, with experts offloaded and the stand-in attention in place of the kernel. Within that caveat:
- The control's mean KL grew 4.3× between the two (1.68e-3 → 7.15e-3), the subject's 1.5× (2.98e-3 → 4.43e-3).
- Over 255 decode steps, all 16 of OLMoE's layers read an fp8 KV that grows with every step, against 10 of Qwen3.6's 40.
- At the prefill step, which reads no fp8, the subject sits 2.7× above the control (2.66e-3 against 9.94e-4). A short
  shape is dominated by that arithmetic floor.

This is the reason the registered rule does not compare the two models' KL. A cross-model ratio swings with the shape,
and its direction is set by each model's sensitivity, not by the code under test. The first-draft rule (subject ≤ 2×
the control's KL, agreement ≥ the control's − 0.02) would also have passed this reading. The pre-registration records
why it was replaced before any data.

## What this does and does not show

- **Shown, on the card through the real kernel, at Qwen3.6's scale:**
  - each sequence's pooled Gated DeltaNet state stays transformers' own state for that sequence, through chunked
    prefill and 255 batched decode steps of 4 interleaved sequences;
  - the whole model's error is that of a working paged runner, 900× below the slot-mapping mutant's.
- **Not shown:**
  - **Speed.** Decode is eager (graphs are refused for hybrids), and the Gated DeltaNet layers ran transformers' torch
    path (no `fla` or `causal_conv1d`): 830 ms per 4-row step is not a serving number.
  - **Other hybrids.** Qwen3-Next and the dense Qwen3.5 models share the module class, but only Qwen3.6 was read.
  - **Long contexts beyond 768 tokens**, and slot recycling under load. CPU tests pin slot recycling.

## Box and cost

- **`p97-prove-1`:** one RTX 5090 (sm_120), Vast instance 53923939, **$0.0221**, OK, PROVED; destroyed at
  2026-10-02T21:47:25Z, instance absent.
- **`p97-5090-1`:** one RTX 5090 (sm_120, driver 580.95.05, 32,607 MiB), Vast machine 142284, instance 53924396, AMD
  EPYC 7663; image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. **$0.2489**, 1,583 s, OK.
  - The fetch ran about 240 MB/s (authenticated, 8 workers) against the single-stream probe's 38.6 MB/s.
  - Destroyed at 2026-10-02T22:15:43Z, instance absent.
- **The lane: $0.2710** of a $2.50 ceiling.

Receipts (`receipts/p97-5090-1/`): the two records, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`,
the premise, box and fetch logs, the teardown proof, and `SHA256SUMS` over every file the box wrote. The launcher's
`receipt.json` files and the ledger rows are in the receipt store (adertha-receipts `3574dce`, `b7f7102`).
