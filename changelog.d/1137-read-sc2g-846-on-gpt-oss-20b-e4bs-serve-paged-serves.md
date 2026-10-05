### Read: SC2g (#846) -- on gpt-oss-20b e4b's `serve_paged` serves every row VALID and is prefill-bound under load as registered (Q4, b/a 27); capacity 1 req/s against vLLM's and SGLang's 8

- **What ran.** `sc2g-5090-2` ($0.959, a 575 W 5090) drove four engines on openai/gpt-oss-20b's MXFP4 experts: e4b
  `eaf3e5b4` (MXFP4 decode, NF4 prefill, prefill graph `auto` engaged), vLLM 0.30.0 (Marlin W4A16), SGLang 0.5.20
  (`flashinfer_mxfp4`, W4A8) and llama.cpp (the published GGUF). Every e4b ratio is ARITH_MISMATCH. Lane total $3.052 across
  4 receipts, inside the registered ~$3.4.
- **Verdicts.**
  - Q1 HOLDS: serial TTFT 5.05× vLLM's.
  - Q2 REFUTED: serial TPOT 1.74× vLLM's (6.21 ms against 3.56 ms).
  - Q3 HOLDS: ceilings vLLM 8, SGLang 8, llama.cpp 2, e4b 1.
  - **Q4 HOLDS:** 0.245 s stall per prefill landing during a decode, against 8.94 ms per token; b/a 27.4, R² 0.984.
  - Q5 REFUTED, on one row: llama.cpp's serial TTFT moved 51 → 44 ms between two runs of the same requests. All 16 Poisson
    rows are VALID.
- **Engagement rests on the code path, not on `/health`'s route names.** `prefill_routes` is env-resolved, so its k19 / flash
  are not gpt-oss's path. The read cites the lines: K21 ≤ 256 rows, kept-NF4 M-tile above, explicit-mask attention on every
  sinks layer. e4b#1129's `seen` fields are what a future box G check should assert.
- `tests/test_sc2_trace.py` pins the Q4 fit from the committed trace.
