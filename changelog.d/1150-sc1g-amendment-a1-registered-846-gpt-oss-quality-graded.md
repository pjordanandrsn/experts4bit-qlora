### SC1g amendment A1 registered (#846): gpt-oss quality graded on in-distribution conversations; activation quantization costs every engine on out-of-distribution text (bench and tests only)

- **`sc1g-prove-1` PROVED ($1.051) and refuted the chat-framed wikitext premise.** Every engine scored the same ids, and gpt-oss
  read perplexity ~500–2,400 with top-1 3–8%. On that text every engine lost NLL as its activations coarsened: MXFP8 +0.15, int8
  per-32 +0.24 (e4b's GEMV vs its NF4) to +0.66 (llama.cpp q8), W4A4 +1.45. Two independent int8 implementations paid, so it is
  the text plus the activation scheme. The reading was held before it launched.
- **A1's graded text** is the first two ultrachat_200k `test_sft` conversations (pinned revision) whose single rendering in
  gpt-oss's chat template reaches 2,561 tokens. Their scored targets are 97% and 90% assistant content. Wikitext stays as a
  descriptive control. The floor, G1–G5 and COMPARABLE are read on the conversations only.
- **G6, new:** `gemv_mxfp4_b32` against its exact reference rounded to bf16, on the 5090, with a mutation arm that must disagree.
  The same check on the A2000 (sm_86, correctness only) agrees within 4.4e-5; the mutation reads 1.47. It also measures the int8
  scheme's own error: 0.5% on normal rows, 1.0–1.3% with 100× outliers.
- **e4b diagnostics** (descriptive, meanings registered): served at `GNF4_PDL=0`, with the folds off, and eager `--ppl-chunk 1`
  for MXFP4 and NF4, which separates the paged fp8-KV decode path from the expert route. A registered hypothesis: the paged path
  carries e4b's served-vs-prefill gap.
- `sc1g_k8.py` reads the e4b arms' window from its file and captures GEMV activations by wrapping gnf4's functions. The shared
  harness stays unmodified. New: `sc1g_gemv_check.py`; `sc1g_reduce.py` at 13 self-test cases. Lane ≈ $4.5 with the re-proof.
