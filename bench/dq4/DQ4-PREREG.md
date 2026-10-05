# DQ4 — pre-registration: how much longer a sequence trains on one RTX 5090 when the frozen weights stream

Work item: #1083. Licensed by DQ3's PROTO_PASS (#1188). DQ3 found streaming bitwise identical, at 1.0023× resident step
time, with 15.08 GB of peak **allocated** memory freed. That PR's follow-up (#1190) then found that peak **reserved**
memory falls much less, and on a small shape rises. Owner, 2026-10-05: "register the capacity read and launch it".

## The question

On one RTX 5090 (PCIe gen 5 x16), with DQ3's subject: what is the longest training sequence that runs **resident** (R),
and how much longer is it when the frozen weights **stream** (S, `enable_dense_offload(train_prefetch=True)` with #1183's
late-bound backward)?

This turns DQ3's allocator headroom into the number a user cares about. The boundary is a real out-of-memory boundary,
measured, not inferred.

## Subject (DQ3's, unchanged)

- Qwen3-32B architecture from its config: 64 layers, hidden 5120, intermediate 25600, 64 q / 8 kv heads × 128, vocab
  151936, untied.
- Random NF4 weights (bnb `Linear4bit`, blocksize 64, double-quant, bf16 compute).
- PEFT LoRA r16 / α32 / dropout 0 on all seven projections: 448 `peft.tuners.lora.bnb.Linear4bit`, fp32 adapters.
- Non-reentrant gradient checkpointing. AdamW, lr 2e-4.
- Software: torch 2.8.0+cu128, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2, experts4bit-qlora at the launch
  commit (which must contain #1183 and #1193), grouped-nf4-gemm `a5edec87`.
- Each rung: one micro-batch of exactly L real tokens, with no padding. Token ids come from `torch.randint` with seed
  1234 + L, so R and S see identical work at every L. Default SDPA, as a real run trains. Two training steps per rung.

## Configurations and arms

| config | LM loss | allocator | role |
|---|---|---|---|
| **c_def** | **chunked** (`enable_chunked_lm_loss(chunk=512)`) | default | **graded** |
| c_exp | chunked | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | secondary verdict |
| s_def | stock (Hugging Face's) | default | descriptive |

**Why the chunked loss is graded.** The stock causal-LM loss keeps full-vocabulary logits: bf16, then fp32, then the fp32
log-softmax for backward, roughly 1.4 MiB per token at Qwen3's vocabulary. That term is identical in R and S, so if it
sets the boundary, it pulls G toward 1 and the read measures the loss, not weight streaming. The chunked loss
(#1193 adds dense Qwen3 to it) bounds that term at one 512-token chunk. The stock pair is kept, descriptively, so the
loss's own effect is visible.

Each configuration runs arms **R** (no offload) and **S** (train_prefetch).

## Procedure (`bench/dq4/dq4_run.sh` on the box; one process per measurement, `dq4_cap.py`)

1. **Host gates, before any install:**
   - not an RTX 5090 on gen 5 x16 → rc 13;
   - DQ3's VRAM probe OOMs → rc 18;
   - DQ3's egress probe is below 1 MB/s → rc 14.

   Then a tripwire: the installed e4b commit, the pinned versions, `_bnb_mirror_mismatches() == []` (the late-bound
   backward will engage), and `Qwen3ForCausalLM` in the chunked-loss table. Then the reducer's self-test.
2. **For each configuration, in the order c_def, c_exp, s_def** (a configuration is skipped, and recorded as skipped,
   if the guard has too little time left for it):
   - **Ladder:** one process per arm. L = 2048, 3072, … (step 1024) up to 32768, two steps each, ascending, stopping at
     the first CUDA OOM. The process ends there: an allocator after an OOM is not trusted again. L\*_X is the largest
     L that passed.
   - **Confirmations** (c_def and c_exp only): for each arm, one fresh process at L\*_X, which must pass, and one fresh
     process at the ladder's first OOM, which must OOM.
3. `dq4_reduce.py` reads the receipts.

## The rule (`bench/dq4/dq4_reduce.py` implements it as written)

Per configuration, in this order:

1. **VOID**, for any of the following:
   - a receipt from the wrong device, or a rehearsal receipt;
   - a subject other than 64 layers;
   - engagement other than 448 bnb LoRA wrappers with fp32 adapters and checkpointing on;
   - the wrong loss mode, or a chunked loss that did not engage;
   - the wrong allocator setting;
   - an S arm not offloading 64 layers with `late_bound_4bit` = 448;
   - an unfinished receipt;
   - a missing ladder or confirmation;
   - software that differs across receipts (the E4B_SHA);
   - an OOM at the ladder's first rung;
   - R reaching the ladder's top (no boundary).
2. **FUNCTION_FAIL** if any passing rung has a non-finite loss.
3. **NOISY** if a fresh process at L\* fails, or a fresh process at the first OOM passes.
4. Otherwise, **G = L\*_S / L\*_R**:
   - **CAP_REAL** if G ≥ 1.5;
   - **CAP_MARGINAL** if 1.1 ≤ G < 1.5;
   - **CAP_NONE** if G < 1.1.

   If S reached the ladder's top, L\*_S is a lower bound. G ≥ 1.5 is still CAP_REAL; a lower bound below 1.5 reads
   CAP_MARGINAL_OR_MORE or CAP_UNRESOLVED.

**The lane verdict is c_def's.** c_exp's verdict is reported as secondary, and s_def's G descriptively.

**Always reported, per configuration and arm:** peak **allocated** and peak **reserved** at L\* (from the fresh
confirmation), and T(S)/T(R) at the largest L both ladders passed. These are descriptive.

## Predictions (from DQ3's rented-5090 readings only; no A2000 number seeds them)

The static-memory accounting at DQ3's 2048-token R arm (peak allocated 24.77 GiB):
- NF4 weights: 64 × 243,793,920 B = 14.53 GiB;
- embedding + LM head, bf16: 2.90 GiB;
- LoRA parameters, gradients and AdamW state: about 2.0 GiB;
- quant state: about 0.1 GiB.

That is about **19.5 GiB static**, leaving about 5.3 GiB transient at 2048 tokens with the stock loss, or ~2.65 MiB per
token. About 1.4 MiB per token of that is the stock loss's logits. The chunked loss replaces it with about 0.9 GiB fixed,
leaving **~1.25 MiB per token** of activation transient: the 64 saved layer inputs plus one layer's recompute.

The usable budget is 31.36 GiB total, less about 0.5 GiB of CUDA context:

| | static | predicted L\* (chunked) | band |
|---|---|---|---|
| R | 19.5 + 0.9 GiB | **~8.6k** tokens | [6144, 11264] |
| S | 19.5 − 14.53 + 2 streamed layers (0.45) + 0.9 GiB | **~20k** tokens | [14336, 26624] |
| **G** | | **~2.3** | [1.6, 3.2] |

- **The stated risk** is allocator fragmentation. #1190 measured S's peak reserved above R's at a small shape. If
  reserved memory, not allocated, binds under the default allocator, L\*_S falls short of the prediction, and c_exp is
  expected to recover part of it.
- **Stock loss (descriptive):** R ~4.4k, S ~9.8k, G ~2.2.
- **Step time** at the largest common L: S/R ≈ 1.0, from DQ3. Descriptive only.

## Consequences

- **CAP_REAL:** streaming is a capacity primitive on this card. The next question is a default-on decision for training
  offload, which belongs to the maintainer.
- **CAP_MARGINAL:** it helps, but by less than the bytes say. The allocator, or the transient, sets the limit; c_exp
  says which.
- **CAP_NONE:** the 15 GB is not usable capacity on this subject under this allocator. Write up why from the
  reserved/allocated readings; do not tune and re-read without a new pre-registration.

## Budget, guard, rehearsal

- **Run:** one RTX 5090 on `vast:verified-secure`, restricted to gen ≥ 5 / x16 / ≥ 40 GB/s, with
  `preflight_bandwidth: none`. Guard **1.5 h** at $0.85/h, estimated **≤ $1.28**. This is inside the standing no-ask tier
  for a single run under $15 (#564).
- **Pre-launch gate** (the DQ3 lesson): an A2000 rehearsal, correctness and memory only, never timing, at a reduced
  subject (8 layers at Qwen3-32B width, chunked loss, step 512). `bench/dq4/dq4_rehearsal_check.py` must find L\*_S > L\*_R
  there, plus `late_bound_4bit` = 56 and the chunked loss engaged.

## Not claimed

- A gen 4 or non-5090 boundary.
- A boundary for real weights. Values don't change bytes.
- A micro-batch above 1.
- A model larger than Qwen3-32B: a 72B does not build resident, so it has no R to compare against.
