### SD1 read (#1313): speculative decoding Phase 0 -- PROCEED_EAGLE3 (measured acceptance; modelled speedup)

- **What is measured.** The draft acceptance of the licensed EAGLE-3 head `RedHatAI/Qwen3-30B-A3B-speculator.eagle3`, on
  `Qwen/Qwen3-30B-A3B` at e4b's shipped default, from the target's captured hidden states. Tokens per verify step at
  k = 1 are 1.72 on chat with reasoning on, 1.65 with reasoning off, and 1.54 on raw wikitext.
- **What is modelled.** The B = 1 speedups are priced with P123's verify-cost model, not timed: ×1.23 (chat, reasoning
  on, k = 2), ×1.18 (chat, reasoning off, k = 1) and ×1.10 (raw text, k = 1). Prompt-lookup drafting reaches ×1.08 on
  raw text and about ×1.0 on chat.
- **The verdict.** PROCEED_EAGLE3, re-derived by the maintainer byte for byte. Phase 1, a measured verify path in
  `serve_paged`, gets its own registration. Every registered prediction held.
- **The capture path.** It reproduced the served path's tokens exactly (160 of 160).
- **Spend.** SD1 spent $1.488 over two runs, one a harness error fixed by Amendment 1.
