### SC1g (#846): A6's conv2 lead is not the MXFP4 prompt kernel -- `gemm_mxfp4_grouped` at > 256 rows reads IN_LINE and at its bf16 floor ($0, A2000; bench only)

- **The question.** Is e4b's `KEEP_NF4=0` prompt kernel (`mxfp4_grouped_v1|gt256`) out of line with the other expert
  paths at large M? A6's conv2 excess sat on the decode steps nearest that prompt.
- **What runs.** e4b passes `sizes=[1] * M`, so gnf4 takes the per-row `_gemv_mxfp4_grouped`: an exact e2m1 × e8m0
  decode, fp32 accumulation, and one bf16 rounding.
- **Result.** On real gpt-oss-20b experts (layers 0, 11, 23), its mean error equals the bf16 output floor on every shape.
  It is bit-equal to the correctly rounded exact result for ≥ 99.9955 % of outputs at M = 2048. The decode GEMV's int8
  activations sit about 7× above that floor.
- **Positive control.** Swapping the nibble order in the reference moves every MXFP4 path about 1000× off the floor.
  - A floor gate was added because the relative verdict cannot see a defect shared by all MXFP4 paths.
  - The mutation arm exits 1, as required.
- **Next.** The router-flip instrument (A7), which needs routing data.
