### SC1g registered (#846): the quality of each engine's gpt-oss-20b arithmetic on identical tokens, against a routing-flip floor measured on the same windows (bench and tests only)

- **Why.** SC2g read speed on four stacks serving gpt-oss-20b's MXFP4 experts on different arithmetic, and no quality, so no
  position sentence came from it. e4b's distance to bf16 is already measured (P44 / P90 on an H100). SC1g asks whether, on one
  RTX 5090 and the SAME token ids, any engine's arithmetic moves NLL beyond the floor two equally correct forwards already show.
- **Box I** (`SC1_BOX=I`, box G's image). Teacher-forced NLL in SC1's two shapes, on SC1's two texts in gpt-oss's chat frame
  with the template's date pinned. Arms:
  - e4b: the serve env (served = MXFP4 GEMV W4A8; prefill = kept-NF4 host M-tile at chunk 64 and 128, the floor), and an NF4 control;
  - vLLM: Marlin W4A16;
  - SGLang: `flashinfer_mxfp4` and Marlin;
  - llama.cpp: the published GGUF, default and `GGML_CUDA_MMQ_PREC=q8`.
- **Route records gate the e4b rows** (e4b#1129's counters, written at exit). A missing record FAILS the row.
- **The shared harness is untouched.** P39's `step_decomp.py`, SC1's `sc1_prompts.py` and P42's hook are byte-pinned by other
  lanes. The new `bench/sc2/sc1g_k8.py` runs them unmodified, with the chat date pin and the route record in its own process.
- **SGLang's server script** gains `gptoss_q` / `gptoss_qm` modes, whose engagement check requires the resolved MoE runner.
- **Predictions and scope:** G1–G5 in `bench/sc2/SC1g-PREREG.md`; `sc1g_reduce.py`, 9 self-test cases. COMPARABLE, if licensed,
  covers e4b's B = 1 served arithmetic and its host NF4 prefill experts only, not serve_paged's batched K21 / captured paths.
  Lane ≤ about $3.4.
