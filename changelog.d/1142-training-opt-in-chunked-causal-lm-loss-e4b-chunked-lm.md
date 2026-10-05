### Training: opt-in chunked causal-LM loss (`E4B_CHUNKED_LM_LOSS`) -- the `[tokens, vocab]` logits are never materialised

- **Why.** TC1 amendment 39's box `tc1-5090-86` (packed rows of 4,096 real tokens at micro-batch 1 on Qwen3-30B-A3B) ran
  e4b's arm out of memory at step 1: `Tried to allocate 2.32 GiB` with 29.5 of 31.36 GiB in use. 2.32 GiB is 4,096 x 151,936 x 4
  bytes, the fp32 upcast of the full-vocabulary logits in Hugging Face's `ForCausalLMLoss`. That loss also keeps the fp32
  log-probabilities for backward and builds their fp32 gradient and the bf16 logits' gradient.
- **What.** `engines/chunked_lm_loss.py`. A training forward with `labels` runs the decoder as before, takes the hidden states
  `lm_head` would have seen, and computes Hugging Face's loss over chunks of tokens: per chunk `lm_head`, the architecture's own
  post-head transform, fp32, cross-entropy summed. Each chunk runs under non-reentrant `torch.utils.checkpoint`, so its logits are
  dropped after the forward and recomputed one chunk at a time in backward. The semantics are HF's: shift by one inside each row
  (or `shift_labels` as given), `ignore_index`, the mean over supervised tokens or the sum over `num_items_in_batch`, and the router
  auxiliary loss added as the forward adds it. Ignored rows never reach the head (finding them is one host sync per training
  forward, after the decoder). The output is the model's own class with `.loss`
  and `.logits=None`.
- **Where it hooks in.** `enable_fast_train` applies it when `E4B_CHUNKED_LM_LOSS` is set, and `disable_fast_train` unwinds it, as
  it does the rotary, RMSNorm and MoE-keep switches. So the TC1 harness's fused arm takes it from `TC1_E4B_ENV` alone, through its
  unchanged `model(input_ids=ids, labels=labels)`. `python -m experts4bit_qlora.train` applies it itself (and lists it in `--help`).
  `enable_chunked_lm_loss(model, chunk)` is the direct call. `1` means 512-token chunks; a number is a chunk size in tokens.
- **Off by default, and narrow when on.** Unset, nothing is patched. When on, generation, `torch.no_grad` evaluation (both
  held-out paths in the harness and the CLI trainer) and `return_dict=False` run the stock forward, so a held-out loss is the stock
  path's bit for bit.
- **Covers / refuses.** Covered: the HF classes of Qwen3-MoE, Qwen3.5/3.6-MoE (text), Mixtral, OLMoE, gpt-oss, ERNIE-4.5-MoE,
  Granite-MoE / -Shared / -Hybrid (their `/ logits_scaling` reproduced), LFM2-MoE and Nemotron-H (its `.float()`). Anything else is
  refused with a `RuntimeWarning` and keeps the stock loss. That includes Gemma-4 (final-logit softcap), a forward or
  `loss_function` replaced by another library, and a hooked or already-patched `lm_head`. Every training forward also checks the
  transform at run time: `lm_head` returns a one-element probe that must come back exactly as the table's transform of it. Any
  other change disables the switch for that model with a warning, and the call re-runs stock.
- **Equality** (`tests/test_chunked_lm_loss.py`, CPU, and CUDA where present). Every covered family on a tiny config, with the
  aux loss on, micro-batch 2 with right padding and a mask, a masked prompt and 7-token chunks. The loss is within 8 fp32 ulps of
  stock (measured at most 1). Every gradient, the head's included, is within 1e-5 of its tensor's largest element (measured at
  most 1.4e-6). The tests also pin chunk sizes that do not divide the supervised count, a tied head, `num_items_in_batch`,
  `shift_labels`, positional labels, and an all-ignored batch (stock's nan with zero gradients). bf16 and CPU autocast stay within
  2 bf16 ulps. Switching off is byte-identical to stock. Mutating the module (no shift, the mean over ignored positions, no aux
  loss, no Granite scaling, no fp32 upcast, a dropped tail chunk, mis-compacted labels, `num_items_in_batch` or `shift_labels`
  ignored) fails the suite.
- **RTX A2000 12 GB, head + loss alone** (`bench/chunked-lm-loss/bench_lm_head_loss.py`): vocab 151,936, hidden 2048, bf16
  hidden states with grad, frozen bf16 head, torch 2.8.0+cu128. Each cell is the peak allocated above the inputs (GiB); the
  A2000 is a correctness testbed, so no timing is read from it:

  | tokens | stock | chunk 256 | chunk 512 | chunk 1,024 | chunk 2,048 |
  |---|---|---|---|---|---|
  | 1,024 | 2.04 | 0.44 | 0.88 | 1.74 | 1.74 |
  | 2,048 | 4.06 | 0.45 | 0.89 | 1.75 | 3.48 |
  | 4,096 | 8.11 | 0.47 | 0.90 | 1.77 | 3.51 |

  The time cost is structural: one more head matmul and cross-entropy forward per chunk in backward (the recompute). Its size
  on a target card is unread here.
  - The loss matched stock to 1 fp32 ulp. The hidden-state gradient was `torch.equal` to stock wherever the head's backward
    matmul ran at stock's row count. Elsewhere about a third of its elements differed by bf16 rounding (relative L2 3.2e-3 at
    2,048 tokens, 4.6e-3 at 4,096). That is cuBLAS choosing its bf16 split-K reduction by shape: with
    `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` it was `torch.equal` at 2,048 and 4,096.
- **RTX A2000, a training step** (`train_step_ab.py`): Qwen3-30B-A3B's first 4 layers with the real embedding and head, through
  the harness's fused-arm setup (NF4 experts, gradient checkpointing, fp32 attention LoRA r16, `enable_fast_train(dgrad=True)`),
  micro-batch 1. Arms were interleaved in one process, each cell the peak GiB per step:

  | tokens | stock | chunk 512 | chunk 1,024 |
  |---|---|---|---|
  | 512 | 3.91 | 3.77 | 3.77 |
  | 1,024 | 4.94 | 4.55 | 4.65 |
  | 2,048 | 6.99 | 6.10-6.15 | 6.19 |
  | 4,096 | **OOM** (`Tried to allocate 2.32 GiB`, the 5090's allocation, on an 11.62 GiB card) | 7.11 | 7.11 |

  The step's time cost on a 5090 is a TC1 registration's to read; the A2000 is not a speed testbed. `1` means 512-token chunks.
  - At 2,048 tokens the loss was identical in every comparison. LoRA gradients differed from stock by 2.7e-3 relative L2 on the
    fused path (stock against itself: 1.8e-3) and 2.6e-3 on the reference path (6.1e-4). With the reduced-precision flag off:
    1.36e-3 against 1.46e-3, and 5.2e-4 against 5.4e-4, inside run-to-run noise.
  - Receipts are in `bench/chunked-lm-loss/receipts/`. These are A2000 numbers, read as correctness and memory. A position on the
    5090 is a TC1 registration's to measure.
