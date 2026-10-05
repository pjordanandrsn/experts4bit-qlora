### Training: the chunked LM loss is on by default as `auto` (TC1 amendments 39, 40, 41, 43, 44)

- **What changes.** `E4B_CHUNKED_LM_LOSS` unset now means `auto`. `enable_fast_train` and the CLI trainer compute the causal-LM loss over
  512-token chunks for a training forward whose stock fp32 logits would reach 1 GiB, and run the stock loss for every smaller forward. `0`
  keeps the stock loss everywhere (nothing patched); `1` or a chunk size chunks every training forward, as before. Under the default, a
  model outside the chunked-loss table keeps its stock loss without a warning; with the variable set, the refusal warns as before.
- **Why, by the registered rule.** On packed 4,096-token Qwen3 rows e4b at its old defaults ran out of memory at step 1 on an RTX 5090
  (amendment 39). With the chunked loss it trains them resident, 1.278× Unsloth's speed on one stack (amendments 40, 43; P98 HELD). At
  the field recipe, chunking every forward cost the shipped arm 4.9 % (amendment 41), while `auto`'s gate never fired and stepped
  0.992 / 0.999 of the stock loss (amendment 44, P99–P103 HELD).
- **Scope.** The evidence is Qwen3-30B-A3B. The gate counts bytes, so on a large vocabulary it fires at ordinary micro-batches
  (about 1,335 positions per forward for gpt-oss, about 1,081 for Qwen3.5-MoE), where chunking's step cost was not measured;
  `E4B_CHUNKED_LM_LOSS=0` restores the stock loss.
- **Not changed.** Evaluation under `torch.no_grad`, generation, `logits_to_keep` and tuple returns run the stock forward; a forward
  under the gate is the stock forward exactly (test-pinned).
