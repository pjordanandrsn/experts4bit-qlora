### Router epilogue: decisive probe rows judged on fp32 CPU logits over 64 rows (fix; #1385 follow-up)

- **What changed.** `router_epilogue._probe_matches` decides which probe rows are decisive on the selection logits
  computed in fp32 on the CPU (`_selection_logits_fp32`: the same projection, with the select-on-logits kind's bias and
  Gemma-4's pre-norm), with the near-tie threshold at two ulps of the module's own dtype. It probes 64 rows, up from 4.
  The comparison itself (expert set and weights, per expert) is unchanged.
- **Why.** On a rented RTX 5090 (lane FAM's fam-prove-1), Granite's router fold licensed 27 of its 32 routers, where
  the RTX A2000 and P115 Phase C read 32. The near ties had been judged on the device's bf16 logits over 4 rows, so
  which routers licensed depended on the card's rounding. A refusal is safe (that router keeps its own forward),
  but the Qwen3-MoE default would then engage a card-dependent subset of its routers.
- **Tests (`tests/test_router_epilogue.py`):**
  - under six emulated cards' bf16 rounding (every bf16 linear moved by up to half an ulp), the logits that judge the
    decisive rows are fp32 on the CPU and bitwise the same, and transformers' Qwen3-MoE and Mixtral routers and the
    gpt-oss-shaped one license on every card;
  - a router with Granite-scale logits keeps decisive rows and is licensed.

  Both were mutation-checked: judged on device logits, the first fails; with 4 rows, the second fails.
