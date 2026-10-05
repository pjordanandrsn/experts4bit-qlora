### Read: TC1 amendment 39 — on packed 4,096-token rows e4b at its defaults runs out of memory where Unsloth trains (P86 FALSIFIED; P84, P85 UNTESTED)

- `tc1-5090-86` ($1.37, EPYC 7B13): Qwen3-30B-A3B's matched set on packed rows of exactly 4,096 real tokens, both frameworks on one
  stack. Every e4b arm OOMed at step 1 allocating 2.32 GiB, the fp32 copy of the full-vocabulary logits in Hugging Face's causal-LM loss.
  Unsloth trained resident at 24.86 GB (its draws 7.7 % apart, so no speed is read).
- Recorded as an e4b loss in that regime (row `e4b.train.h2h.unsloth.qwen3.5090.2026-10-05.packed-4k`). A chunked loss for e4b is its own
  registration.
