### Four more launches gone from the T == 1 decode, where the kernel package offers them (#1313, lane P127's Phase 2)

- **What.** Each of these uses a grouped-nf4-gemm option when the installed kernel package has it (detected by
  signature or a capability constant), and otherwise does exactly what it did before:
  - **Router.** A fused router forward that casts its weights to the logits' dtype asks `router_epilogue(...,
    weights_dtype=)` for that dtype (grouped-nf4-gemm#526). The kernel's store rounds to nearest even, bitwise
    torch's cast, so the cast's launch goes.
  - **Attention.** Both attention folds make one `rope_norm_qk` call for q's and k's heads (grouped-nf4-gemm#528)
    instead of two `rope_norm_heads` calls. The outputs are bitwise the two calls'.
  - **Expert ids.** The NF4 route hands the int64 expert ids over uncast when `nf4_grouped.EXPERT_ID_DTYPES` lists
    int64 (grouped-nf4-gemm#529), since every NF4 kernel widens its id itself.
  - **Token rows.** At ONE token, the NF4 singleton route hands gate_up the token row with `gather_div=top_k`
    (grouped-nf4-gemm#530) instead of copying it `top_k` times. With one token every row is token 0, so row order
    cannot matter; at more tokens the copy stays.
- **Why it is value-preserving.** Every change is bitwise by construction, and each has a CPU test on a kernel
  stand-in. Each test is mutation-checked: removing the change, or mis-wiring it, fails it. That includes
  expert-sorted rows at two tokens, which must keep the copy.
- **No speed claim.** P127's read prices these with Phase 1 under the bitwise gate.
