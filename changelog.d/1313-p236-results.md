### P236 results: bitsandbytes is faster than grouped-nf4-gemm's decode GEMV at all 30 cells, op level and eager (READ)

P236 (`bench/p236/RESULTS-p236.md`) timed grouped-nf4-gemm 0.51.0's K33 decode GEMV against bitsandbytes 0.50.2's
`gemm_4bit` on the same NF4 bytes. It covered six MoE expert projections (Mixtral-8x7B, Qwen3-30B-A3B, OLMoE-1B-7B) at M = 1, 2, 4,
8 and 16 on one RTX 5090. **READ.**
- **Every cell held** the premise, correctness and engagement as registered.
- **bitsandbytes is faster at all 30 cells.** The ratio (its time over gnf4's) runs from 0.2993 to 0.9176, and at
  M = 1 from 0.4870 to 0.8654.
- **Graded:**
  - Q1 held. bitsandbytes took `dequant` at Mixtral down and OLMoE gate_up at M = 16, and `custom` everywhere else.
  - Q2 (gnf4 faster at M = 1) was wrong.
  - Q3 was wrong as written. Its M = 16 clause held (bitsandbytes is faster on both Mixtral projections there), but the
    ratio does not fall with M: only Mixtral's gate_up falls, and not at every step; the others are near flat, and
    bitsandbytes is faster at every M.
  - Q4 held in part: 16 self-pair bands fall outside [0.95, 1.05], 3 of them on Mixtral.
- **Scope:** eager calls between CUDA events, each including its library's Python dispatch. This is not the served,
  graph-captured decode path, and not K33's device time. No default moves.
- **Records:** `bench/p236/receipts/p236-5090-2/`.
  - The first box, `p236-5090-1`, read VOID (Amendment 2).
  - The box did not write the `SHA256SUMS` its registration lists. The one committed was computed over the committed files.
- **Next:** a device-time arm is noted for a later registration and not started.
