### P155 registered (#1313): K25's plan at Qwen3-30B-A3B's W16 shapes, on S0's real router picks, selection split from confirmation

A microbench premise for stream (B). It sweeps the grouped NF4 small-M expert GEMM's plan, the one knob the kernel's
contract says moves no output bit, against the 6.81 ms of headroom P154 priced.
- **Grid:** 108 plans (BLOCK_N x KC x warps x stages), called exactly as the served route calls K25, over full-size NF4
  stores.
- **Routing:** P153's committed router replay (S0's real picks, sha-pinned and checked before data).
- **Selection:** every plan must keep today's bits; the plans are ranked on 8 selection steps, and three are carried.
- **Confirmation:** the carried three are re-timed against today's plan on the 64 held-out steps, in paired, rotated
  reps, with a self-pair. GO needs median - 2 SE >= 1.15x on both GEMMs, decided on the confirmation only.
- **Reported, never ruling:** a compaction arm (the real-tile prefix) prices tile compaction's ceiling.
- **Box:** one RTX 5090, guard 1.0 h (the fit with 25 % headroom, checked before the install), worst $1.16.

A GO opens a served read as its own registration. Files: `bench/p155/` and `tests/test_p155.py`.
