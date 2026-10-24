# P235 -- vLLM head-to-head, same box, current vs current, two vLLM builds (/root/p235)
Rule: per vLLM build and batch size, the position is the ratio vLLM/e4b (decode tok/s) from the PRIMARY pair (vllm graph_r1 vs e4b int4_r1), quoted only when both sides' self-pairs are inside 1.03x and both receipts carry the same prompts_sha; VOID arms never enter a ratio. Nothing here is licensed; no cross-box number is divided into these.
```
PYTHONPYCACHEPREFIX /root/p235/pycache-ce386ce4e97748987e8f8f3ec0a174b80bccfbb21f40235c57d05a2a98748327
e4b 0.58.0 (PyPI; provenance.json) harness@7be9e6108910d1c3a8296754948bcda0492bff04
gnf4 0.51.0 (PyPI; provenance.json)
torch(e4b) 2.8.0+cu129
triton(e4b) 3.4.0
transformers 5.17.0
bitsandbytes 0.50.1
vllm(primary) 0.32.0
torch(vllm primary) 2.13.0+cu130
triton(vllm primary) 3.7.1
vllm(secondary) 0.30.0
torch(vllm secondary) 2.13.0+cu130
triton(vllm secondary) 3.7.1
```
box: NVIDIA GeForce RTX 5090, 32607 MiB, 595.91.07 | power.limit,clocks.max.sm 470.00 W, 3090 MHz | Model name:                              Intel(R) Core(TM) Ultra 9 285K | total        used        free      shared  buff/cache   available | Mem:             188           2          15           0         170         183 | cgroup memory.max 193871216640 | Build cuda_12.9.r12.9/compiler.36037853_0

## B=1  (prompts_sha a8e6ea1d7d140dbe, rows 1, row step 298426 tokens)
| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |
|---|---|---|---|---|---|---|
| e4b | int4_eager | VALID | 23.8 | 42.07 | 19.88 |  |
| e4b | int4_r1 | VALID | 277.4 | 3.605 | 19.79 |  |
| e4b | int4_r2 | VALID | 277.5 | 3.603 | 19.79 |  |
| e4b | nf4_r1 | VALID | 151.4 | 6.605 | 18.94 |  |
| e4b | nf4_r2 | VALID | 151.7 | 6.594 | 18.89 |  |
| vllm-primary | eager | VALID | 49.1 | 20.361 | 28.73 (policy: reserves 0.90) | vllm 0.32.0; min-of-3; median-of-3 49.0 tok/s; e2e incl. prefill 49.1; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.87 GiB for consumed memory (weights + |
| vllm-primary | graph_r1 | VALID | 287.9 | 3.474 | 31.0 (policy: reserves 0.90) | vllm 0.32.0; min-of-3; median-of-3 287.9 tok/s; e2e incl. prefill 271.7; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.87 GiB for consumed memory (weights + |
| vllm-primary | graph_r2 | VALID | 287.5 | 3.479 | 28.8 (policy: reserves 0.90) | vllm 0.32.0; min-of-3; median-of-3 287.5 tok/s; e2e incl. prefill 272.2; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.87 GiB for consumed memory (weights + |
| vllm-secondary | graph_r1 | VALID | 286.0 | 3.496 | 29.06 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 286.0 tok/s; e2e incl. prefill 270.4; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.65 GiB for consumed memory (weights + |
| vllm-secondary | graph_r2 | VALID | 286.8 | 3.487 | 28.86 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 286.4 tok/s; e2e incl. prefill 270.5; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.65 GiB for consumed memory (weights + |
- self-pairs: e4b nf4 1.0017; e4b int4 1.0006; vllm-primary graph 1.0014; vllm-secondary graph 1.0028  (rule: inside 1.03x or DRIFT)
- e4b three axes: x1.832 over same-box NF4; rental-measured 277.4 tok/s (3.605 ms/step) on this box; anchor-class PROJECTION 292 tok/s = 159.2 x 1.832 (uncertified class). P54's median on its box, cited beside and never divided: 3.693 ms/step (within +-8 %, STOP-1 informational).
- **RATIO vLLM 0.32.0 / e4b int4 at B=1: 1.038  (287.9 vs 277.4 tok/s; vLLM ahead; primary pair, min-of-3 slope vs 127-step graph window, identical prompt ids)**
  - with the median-of-3 slope: 1.038; with e4b int4_r2: 1.037; vLLM / e4b NF4 control: 1.902
- **RATIO vLLM 0.30.0 / e4b int4 at B=1: 1.031  (286.0 vs 277.4 tok/s; vLLM ahead; primary pair, min-of-3 slope vs 127-step graph window, identical prompt ids)**
  - with the median-of-3 slope: 1.031; with e4b int4_r2: 1.030; vLLM / e4b NF4 control: 1.889
- build-to-build (same box, same prompts): vLLM 0.32.0 / vLLM 0.30.0 = 1.007 at B=1 (287.9 vs 286.0 tok/s)
- secondary -- eager pairing: vllm-primary eager 49.1 tok/s (0.171 of its graph arm); e4b int4_eager 23.8 tok/s (0.086 of e4b graph); eager-vs-eager 2.066 (secondary, never the headline)

## B=16  (prompts_sha f67e7e4d592b002b, rows 16, row step 18651 tokens)
| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |
|---|---|---|---|---|---|---|
| e4b | int4_r1 | VALID | 1684.9 | 9.496 | 20.14 |  |
| e4b | int4_r2 | VALID | 1686.1 | 9.49 | 20.07 |  |
| e4b | nf4_r1 | VALID | 772.8 | 20.703 | 19.44 |  |
| e4b | nf4_r2 | VALID | 773.7 | 20.681 | 19.39 |  |
| vllm-primary | graph_r1 | VALID | 2026.9 | 7.894 | 28.8 (policy: reserves 0.90) | vllm 0.32.0; min-of-3; median-of-3 2033.2 tok/s; e2e incl. prefill 1582.9; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.87 GiB for consumed memory (weights + |
| vllm-primary | graph_r2 | VALID | 2035.9 | 7.859 | 30.45 (policy: reserves 0.90) | vllm 0.32.0; min-of-3; median-of-3 2035.7 tok/s; e2e incl. prefill 1584.4; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.87 GiB for consumed memory (weights + |
| vllm-secondary | graph_r1 | VALID | 2000.4 | 7.998 | 30.57 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 2008.6 tok/s; e2e incl. prefill 1562.2; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.65 GiB for consumed memory (weights + |
| vllm-secondary | graph_r2 | VALID | 2012.8 | 7.949 | 30.57 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 2007.1 tok/s; e2e incl. prefill 1566.3; Free memory on device (30.83/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.65 GiB for consumed memory (weights + |
- self-pairs: e4b nf4 1.0011; e4b int4 1.0007; vllm-primary graph 1.0044; vllm-secondary graph 1.0062  (rule: inside 1.03x or DRIFT)
- e4b three axes: x2.180 over same-box NF4; rental-measured 1684.9 tok/s (9.496 ms/step) on this box; no anchor projection at B=16. P54's median on its box, cited beside and never divided: 11.42 ms/step (OUTSIDE +-8 %, STOP-1 informational).
- **RATIO vLLM 0.32.0 / e4b int4 at B=16: 1.203  (2026.9 vs 1684.9 tok/s; vLLM ahead; primary pair, min-of-3 slope vs 70-step graph window, identical prompt ids)**
  - with the median-of-3 slope: 1.207; with e4b int4_r2: 1.202; vLLM / e4b NF4 control: 2.623
- **RATIO vLLM 0.30.0 / e4b int4 at B=16: 1.187  (2000.4 vs 1684.9 tok/s; vLLM ahead; primary pair, min-of-3 slope vs 70-step graph window, identical prompt ids)**
  - with the median-of-3 slope: 1.192; with e4b int4_r2: 1.186; vLLM / e4b NF4 control: 2.588
- build-to-build (same box, same prompts): vLLM 0.32.0 / vLLM 0.30.0 = 1.013 at B=16 (2026.9 vs 2000.4 tok/s)

TTFT: informational only where present in the logs (vLLM: none from offline generate; e4b: scheduled PREFILL wall line) -- no ratio (P37 fixture).
