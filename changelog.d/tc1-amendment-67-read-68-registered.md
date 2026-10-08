### TC1 amendment 67 read; amendment 68 registered

- **Amendment 67** (`tc1-5090-135`, RTX 5090, AMD EPYC 7K62, $1.34): at the new defaults, e4b's packed-row training peak is 1.040 GB
  above Unsloth's, and 0.303 GB above with `E4B_CKPT_OFFLOAD=1` (P196, P197 HELD). Held-out agrees (P199 HELD). Unsloth / e4b read
  1.773 (P195 FALSIFIED, high), and no e4b default moved it: Unsloth's packed step is 16.0–16.3 s on that Vast machine against
  11.1–11.7 s on four others. The offload cost 1.041 of the step (P198 FALSIFIED); it stays opt-in. Register row
  `e4b.train.memory.packed-4k-position-defaults.5090.2026-10-07`.
- **Amendment 68** registered: token `qwen3pos68` reads the same position with every arm profiled, to separate GPU work from host work
  (P200-P204). The reducer adds the family, `pos68_why` and `score_pos68` (self-test 134).
