### DQ2 registered: can a dense layer's frozen NF4 weights stream behind its own QLoRA compute on PCIe 5.0 x16? (bench and tests only)

- **Why** (#1083). DQ1 left the capacity axis at C_MARGINAL. Its link was PCIe 4.0, and its compute counted the linears
  only. DQ2 measures one real Qwen3-32B decoder layer as HF + PEFT + bitsandbytes QLoRA runs it:
  - `Linear4bit` nf4 with double-quant;
  - PEFT's `lora.bnb.Linear4bit` with fp32 adapters, r16 on all seven projections;
  - SDPA and RoPE, under non-reentrant checkpointing.

  It times forward and backward against pinned H2D of the layer's own frozen bytes, on a gen 5 x16 RTX 5090.
- **Rule.** Rmin(2048) = min(T_fwd, T_bwd) / X. C_ALIVE (≥ 1.25) licenses a streaming prototype on a branch. 27 self-test
  cases; 18 rule mutants are killed in CI. Every band's basis is rented-5090 data.
- **Rehearsed on the RTX A2000 through the local pool** (correctness only). The first rehearsal's engagement check caught
  PEFT's generic wrapper with bf16 adapters (a bare layer lacks `is_loaded_in_4bit`); the fixed subject engages PEFT's bnb
  path with fp32 adapters.
- **Launch.** The box refuses a non-gen-5 x16 host (rc 13). The Vast search gains an opt-in PCIe band
  (adertha-agents#162).
