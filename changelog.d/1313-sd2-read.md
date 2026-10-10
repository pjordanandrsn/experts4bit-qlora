### SD2 read (#1313): EAGLE-3 speculative decode measured end to end -- VERIFY_COST_REFUTES

- **What is measured.** The `E4B_PAGED_SPEC=eagle3` path in `serve_paged`, on `Qwen/Qwen3-30B-A3B` at B = 1 on one
  RTX 5090. On a single draft token, a verify costs 8.30 ms against 4.18 ms for a plain decode step.
- **The verdict.** VERIFY_COST_REFUTES, re-derived by the maintainer field for field. With SD1's measured acceptance,
  the best speedup is ×0.773 on raw text, ×0.895 on chat with reasoning off and ×0.988 with reasoning on. The measured
  speedup came in 15–34% below SD1's model.
- **The direct check agrees.** Timed end to end, k = 1 runs at ×0.76–0.80 of plain decode and k = 2 at ×0.84–0.92.
- **Drafting is not the problem.** Live acceptance tracks SD1's offline τ within 1.5%.
- **The consequence.** EAGLE-3 at B = 1 is closed for this model, card and stack. `E4B_PAGED_SPEC` stays an opt-in, off
  by default, with no speed claim.
- **Correctness gate.** V0 (rule a4: the logit gate plus the layer-0 key-row gate) passed the real build and caught all
  six mutants out of sample. The key-row gate caught the RoPE shift that the logit gate misses.
- **Spend.** SD2 spent $4.380 over seven rentals. The read itself cost $0.93.
