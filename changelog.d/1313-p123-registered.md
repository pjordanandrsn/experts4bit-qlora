### P123 registered (#1313): a kernel-class census of the shipped default's single-stream decode step

- **The question.** Since #1361 the default `serve_paged` decodes one Qwen3-30B-A3B NF4 sequence at 4.64 ms per token,
  with the B=1 fused stack and grouped-nf4-gemm 0.43.0's bandwidth GEMV both on by default. The last device census of
  this route predates both. Where does the step go now? The answer prices the next lever before any kernel work.
- **The instrument.** SC1b's census, reused unchanged by import:
  - one RTX 5090, the shipped default (every lever unset, 16 slots named);
  - two unprofiled speed arms as the reference;
  - graph-mode and node-mode Nsight Systems captures of 64 steady decode steps at B = 1 and B = 16;
  - `sc1b_census.py` with a v1 class map: SC1b's v0 map plus grouped-nf4-gemm's NF4 expert kernels, named from an
    RTX A2000 kernel-name inventory of the default step (names and counts only).
- **The rule.** VOID / NOISY / READ (`bench/p123/p123_reduce.py`, 19 self-test cases). It reads per batch:
  - each kernel class's share of the step;
  - a RESIDUAL row (the step minus the named classes);
  - the GPU busy fraction;
  - kernels per step and peak memory.

  Each registered prediction is graded. No default changes.
- **Budget.** Proof (guard 0.75 h, ~$0.65) and reading (guard 1.5 h, ~$2.0), lane ceiling $3.00, hard stop $4.00.
- **Files:**
  - `bench/p123/`: PREREG, box, census driver, reducer, run and drive scripts, class map, `staged-p123.sha256`;
  - `tests/test_p123_staged_pin.py`.
