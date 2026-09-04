# Throughput-parity lanes, 2026-09-04 (tpA + tpB)

Raw outputs of the two lanes that produced [`docs/SERVING-THROUGHPUT.md`](../../../docs/SERVING-THROUGHPUT.md). `tpA/` (Ryzen 9 9900X host, RTX 5090, driver 580.159.03): OLMoE, Granite, gpt-oss. `tpB/` (EPYC 9755 host, RTX 5090, driver 580.119.02): Qwen3-30B (reference), Gemma-4, Mixtral. Stack: e4b 0.32.0 + grouped-nf4-gemm 0.26.0 from PyPI, transformers 5.16.1, torch 2.8 cu128. Each arm is one `step_decomp.py` invocation (the lane script's `k8()`/`arm()` helpers; flags in `outer.log`); its JSON is the receipt, its `run_*.log` the provenance, `forensics.txt` the host.

Reduce: `python tp_reduce.py tpA tpB` → [`TABLE.md`](TABLE.md) (regenerated here). tok/s at B=1 is `1000 / step_ms_clean` of the timed graph window; at B=16 it is `aggregate_tok_s`. A missing JSON with a `RuntimeError`/`REFUSED` line in its log is a refused arm, listed under the table.

```
| family | K8 nll nf4 / int4exp / calib (2048) | B=1 nf4 | B=1 int4exp | B=1 calib | B=1 fused (int4+calib+folds) | B=16 nf4 | B=16 int4exp |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B (reference) | 1.8621 / 1.8578 / 1.8448 | 97 | 116 | 112 | 155 | 483 | 944 |
| OLMoE-1B-7B | 1.9380 / 1.9337 / 1.9295 | 248 | 346 | 336 | 452 | 1294 | 2412 |
| Granite-3.1-3B-A800M | 1.6741 / 1.6859 / 1.7003 | 191 | 285 | 218 | ✗ | 1447 | 2210 |
| gpt-oss-20b | 6.3354 / — / — | 124 | ✗ | ✗ | ✗ | 732 | ✗ |
| Gemma-4-26B-A4B | 4.8103 / — / — | 71 | ✗ | ✗ | ✗ | 572 | ✗ |
| Mixtral-8x7B | 1.1805 / 1.1690 / 1.1916 | 48 | 99 | 107 | ✗ | 186 | 371 |

Ratio to the Qwen3-30B reference on the same protocol (B=1 nf4 / B=1 best / B=16 nf4):
  OLMoE-1B-7B: 2.54x / 2.92x / 2.68x
  Granite-3.1-3B-A800M: 1.96x / 1.84x / 3.00x
  gpt-oss-20b: 1.28x / 0.80x / 1.52x
  Gemma-4-26B-A4B: 0.73x / 0.46x / 1.18x
  Mixtral-8x7B: 0.49x / 0.69x / 0.38x

Refusals / failures (the build-out list):
  - granite/b1_fused: RuntimeError: E4B_FUSE_T1_GLUE_R2=1 patched nothing (no structurally matched decoder layer or fused attention passed the pro
  - gptoss/b1_int4exp: RuntimeError: enable_serve_experts_int4: gpt_oss is not served by this lane (interleaved gate/up rows + bias epilogue are ap
  - gptoss/b1_calib: RuntimeError: enable_serve_experts_int4: gpt_oss is not served by this lane (interleaved gate/up rows + bias epilogue are ap
  - gptoss/b1_fused: RuntimeError: enable_serve_experts_int4: gpt_oss is not served by this lane (interleaved gate/up rows + bias epilogue are ap
  - gptoss/b16_int4exp: RuntimeError: enable_serve_experts_int4: gpt_oss is not served by this lane (interleaved gate/up rows + bias epilogue are ap
  - gemma4/b1_int4exp: REFUSED (model_type=gemma4_text): MoEConventionError("no adjudicated MoE convention for model_type 'gemma4_text'. Add
  - gemma4/b1_calib: REFUSED (model_type=gemma4_text): MoEConventionError("no adjudicated MoE convention for model_type 'gemma4_text'. Add
  - gemma4/b1_fused: REFUSED (model_type=gemma4_text): MoEConventionError("no adjudicated MoE convention for model_type 'gemma4_text'. Add
  - gemma4/b16_int4exp: REFUSED (model_type=gemma4_text): MoEConventionError("no adjudicated MoE convention for model_type 'gemma4_text'. Add
  - mixtral/b1_fused: RuntimeError: E4B_FUSE_ROUTER_EPI=1 patched no routers (0 structurally matched but failed the semantic probe) -- refusing a
```
