### Training estimate: the held-out evaluation is priced, and attention adapters can take their own dtype

`estimate_qlora_footprint` now prices the larger of two peaks: the training step's, and a held-out evaluation's (a no-grad
forward of one micro-batch with labels after an optimizer step, with no gradients live). The new item
`held-out evaluation, above the training step` is what the evaluation exceeds the step by, and zero when the step is
larger. A stock evaluation holds the bf16 logits, their fp32 upcast and its log-softmax (`T x V x 10` bytes) plus
`T x H x 10` of hidden states; the chunked held-out loss (`E4B_CHUNKED_EVAL_LOSS`, on by default) holds the bf16 logits
and one chunk's fp32 rows. `estimate_env()` reports `E4B_CHUNKED_EVAL_LOSS`.

**Estimates rise where evaluation is stock**, by about what the evaluation peak exceeds the training step: under
`E4B_CHUNKED_EVAL_LOSS=0` or `E4B_CHUNKED_LM_LOSS=0`, with the reference kernel, or for an architecture the chunked loss
does not support, on long rows at a large vocabulary. Qwen3-30B-A3B at 4,096 tokens per micro-batch (fp32 absmax, no
fused q/k/v projection): +0.89 GiB with fp32 adapters, +2.60 GiB with bf16 expert and fp32 attention adapters. Under the
defaults the item is zero there. Every estimate also gains a 4 MiB `small allocations` line (kernel scratch, lookup
tables, index buffers; 0.4-2.0 MB measured at two census peaks).

`QLoRASetup.attention_adapter_dtype` (default None: the same as `adapter_dtype`) prices and builds the attention adapters
in their own dtype; TC1's `native` arm is `adapter_dtype="bf16", attention_adapter_dtype="fp32"`.

On HO1's 22 held-out 4096 x 1 packed Qwen3-30B-A3B rows, in the configuration they ran (experts4bit-qlora 0.48.0), every
peak is now within the registered bar: peak <= estimate <= 1.05 x peak (estimate / peak 1.0001-1.0033). They pass
because the evaluation term dominates them. **This does not certify the training-step term at 4,096 tokens on main**,
which no receipt here measures, and under the default chunked evaluation that term decides the estimate. #1526 stays
open for it.
