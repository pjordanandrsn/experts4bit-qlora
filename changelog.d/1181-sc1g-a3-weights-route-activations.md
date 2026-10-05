### SC1g amendment A3 registered (#846): box J again, to split the MXFP4 route's +0.17 nats into weights, route and int8 activations, plus e4b#1175's attention check (bench and tests only)

- **What box J (`sc1g-diag-1`, $0.709) read on `conv1`:**
  - The paged fp8-KV path carries no gap: NF4 served, eager chunk 1 and prefill agree within 0.006.
  - The MXFP4 T == 1 route does: +0.169 nats served and +0.225 under identical attention.
  - The GEMV kernel is exact for its scheme on sm_120 (≤ 1.9e-4).
  - The modelled-fp8 arms are VOID: `--ppl-fq` omits gpt-oss's sinks.
  - `--kv-groups 16` reads +0.130 worse than the default 4 (e4b#1175).
- **$0 on the A2000** (`bench/sc2/sc1g-a2000/a3_*`):
  - e4b's fp8 KV pack is correct at 4, 8 and 16 key groups, with its error falling with finer groups. The kvg16 regression is
    the kernel's, not the pack's.
  - gnf4's NF4 M-tile on 10,244 rows in one call matches 128-token chunks and fp32 to the bf16 floor. Large M does not explain
    the chunk-free full anchor's +0.075.
  - Both mutation arms fire.
- **Box J under A3:**
  - GEMV=0 (the decode rows on bf16 activations), and a true MXFP4-weights prefill at `KEEP_NF4=0`. Box J's "MXFP4 prefill"
    had run the kept NF4 stacks.
  - Both are route-gated.
  - The K2/K5 rows on four conversations, a determinism repeat, folds-off and PDL=0.
  - `sc1g_attn_check.py`: the fp8 decode kernel against a dequantize-then-attend reference at 4, 8 and 16 groups, plus pack
    reconstruction and compute mode.
- **Predictions K1–K5:**
  - The weights cost ≤ 0.05.
  - The int8 activations carry ≥ 0.5 of the gap, on `conv1` and pooled over ≥ 3 windows.
  - No fold or PDL bug.
  - Determinism within 1e-4.
  - The gap is ≥ 0.10 on every window. 0.10 is about 2× the 0.051 path spread box J read on `conv2`.
