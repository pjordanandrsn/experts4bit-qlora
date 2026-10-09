### P125 registered (#1313): calibrated int4 attention (and the int4 lm_head) on the shipped default's single-stream decode

- **The question.** P123 found the dense bf16 GEMVs are 38 % of the default's one-request step on Qwen3-30B-A3B NF4.
  P125 reads whether calibrated int4 attention (`E4B_SERVE_ATTN_INT4_CALIB=1`) licenses there, alone and with the int4
  lm_head, and what it buys. Neither lever has been read on today's default.
- **The gates.** They sit at the paths the default serves:
  - the one-row path, which quantises activations, at `T == 1`;
  - K16 in 16-row pieces.
- **The bound.** It is K8's +0.05 ppl budget in nats, computed in-box from the default arm's own windows. The windows
  are sized so the standard error is at most a third of the bound (108 and 112), and a wider spread reads UNDERPOWERED.
- **The mutants.** A sure-fail mutant (scales rolled one block) is the VOID rung. RTN attention is the reported
  sensitivity check.
- **Also recorded.** Speed (palindromic arms), memory including the first-prefill transient, the auto slot count per
  arm, and the calibration time. The calibration runs on the box, as a user's build does.
- **The premise.** An A2000 composition test showed `fuse_qkv` fusing the int4 projections bit for bit on the gemv and
  K16 paths, and the calibration deterministic.
