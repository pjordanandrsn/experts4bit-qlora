### DQ11 MLP-adapter BUILD amendment 1: Phase 1's adapter state, before data (#1083)

`bench/dq11/DQ11-MLP-BUILD-AMENDMENT-1.md` registers what the maintainer ruled during the engine's review (#1809):
- **D2's text:** the engine patches the MLP module's forward, not each `LoraLayer`'s.
- **Phase 1's adapter state:** seeded random B (seed 3407). σ is the smallest of 0.01 / 0.02 / 0.05 / 0.1 that puts the
  adapter term at ≥ 1 % of every checked tensor on the unfused path; otherwise `VOID_LORA_SHARE`. Every share is
  recorded, with the smallest adapter-term error the bar can see in each tensor; where a rank-1 error could hide, the
  outcome says `RANK1_UNRESOLVED`.
- **Reported only:** the state after two AdamW updates of the L path.
- **The reference:** the redraws' reference repeats PEFT's `Linear4bit` LoRA path. R5's order comes from a $0 A2000
  instrument probe. The reference must equal PEFT bit for bit before any floor is read (`REF_MISMATCH`).
- **Engagement:** the harness asserts `enable_dense_mlp_lora_fuse` returns 96 (`ENGAGE_COUNT`), and the call counter
  includes checkpoint recomputes.

No code or default changes.
