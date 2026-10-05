### Read: DQ2 run 2 — a dense layer's frozen NF4 weights stream behind its own QLoRA compute on PCIe 5.0 x16 (C_ALIVE) (bench only)

- `dq2-5090-7` ($0.032, RTX 5090, gen 5 x16, 48.6 GB/s), under Amendment 1: READ, **C_ALIVE**. One Qwen3-32B layer as
  HF + PEFT + bnb QLoRA runs it gives Rmin(2048) = 2.86 (T_fwd 15.02 ms against a 5.25 ms copy of its 251.5 MB of
  frozen bytes). The forward slows by 1.6% under DMA.
- **Quality of the read.** 0/10 self-pairs out of band, 25/25 draws covered, and an independent recomputation agrees.
  8 of 9 prediction clauses held; T_fwd missed by 0.13%.
- **Break-even and scope.** Break-even is 1024 tokens a micro-batch. Planner rule (measured, layer rate):
  R = 1.25 at M ≈ 4.25e4 / B. This is one layer on gen 5 x16, so whole-model overlap is not measured.
- **Consequence.** A streamed frozen-weight prototype is licensed, on a branch, with gradient-parity and step-time
  gates. No new repository. `bench/dq2/RESULTS-dq2.md`.
