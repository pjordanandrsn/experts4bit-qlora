# P58 -- vLLM head-to-head, same box, current vs current, two vLLM builds (/root/p58)
Rule: per vLLM build and batch size, the position is the ratio vLLM/e4b (decode tok/s) from the PRIMARY pair (vllm graph_r1 vs e4b int4_r1), quoted only when both sides' self-pairs are inside 1.03x and both receipts carry the same prompts_sha; VOID arms never enter a ratio. Nothing here is licensed; no cross-box number is divided into these.
```
e4b 0.36.4 @4cab05a81dcf419daa8035d1f7142b266a51f37d
gnf4 0.32.1 @65cb10410239120d25307812827a23139e84abe1
torch(e4b) 2.8.0+cu129
triton(e4b) 3.4.0
transformers 5.16.1
bitsandbytes 0.50.1
vllm(primary) 0.30.0
torch(vllm primary) 2.13.0+cu130
triton(vllm primary) 3.7.1
vllm(secondary) 0.29.0
torch(vllm secondary) 2.13.0+cu130
triton(vllm secondary) 3.7.1
```
box: NVIDIA GeForce RTX 5090, 32607 MiB, 580.119.02, GPU-c8b01e63-c0cd-d763-4a5a-161c91dcdaac | power.limit,clocks.max.sm 575.00 W, 3090 MHz | Model name:                              AMD EPYC 9655 96-Core Processor | total        used        free      shared  buff/cache   available | Mem:             108           1          44           0          62         105 | cgroup memory.max 111380791296 | Build cuda_12.9.r12.9/compiler.36037853_0

## B=1  (prompts_sha a8e6ea1d7d140dbe, rows 1, row step 298426 tokens)
| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |
|---|---|---|---|---|---|---|
| e4b | int4_eager | VALID | 35.5 | 28.188 | 19.84 |  |
| e4b | int4_r1 | VALID | 239.4 | 4.177 | 19.75 |  |
| e4b | int4_r2 | VALID | 239.2 | 4.18 | 19.75 |  |
| e4b | nf4_r1 | VALID | 101.6 | 9.842 | 18.88 |  |
| e4b | nf4_r2 | VALID | 101.5 | 9.849 | 18.83 |  |
| vllm-primary | eager | VALID | 50.4 | 19.836 | 29.9 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 50.0 tok/s; e2e incl. prefill 50.4; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.61 GiB for consumed memory (weights + |
| vllm-primary | graph_r1 | VALID | 260.3 | 3.841 | 29.0 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 260.3 tok/s; e2e incl. prefill 248.7; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.63 GiB for consumed memory (weights + |
| vllm-primary | graph_r2 | VALID | 260.4 | 3.841 | 29.9 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 260.4 tok/s; e2e incl. prefill 248.8; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.63 GiB for consumed memory (weights + |
| vllm-secondary | graph_r1 | VALID | 280.4 | 3.566 | 30.64 (policy: reserves 0.90) | vllm 0.29.0; min-of-3; median-of-3 280.5 tok/s; e2e incl. prefill 267.1; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.66 GiB for consumed memory (weights + |
| vllm-secondary | graph_r2 | VALID | 259.9 | 3.847 | 30.95 (policy: reserves 0.90) | vllm 0.29.0; min-of-3; median-of-3 259.9 tok/s; e2e incl. prefill 248.5; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.66 GiB for consumed memory (weights + |
- self-pairs: e4b nf4 1.0007; e4b int4 1.0007; vllm-primary graph 1.0004; vllm-secondary graph 1.0789  (rule: inside 1.03x or DRIFT)
- e4b three axes: x2.356 over same-box NF4; rental-measured 239.4 tok/s (4.177 ms/step) on this box; anchor-class PROJECTION 375 tok/s = 159.2 x 2.356 (uncertified class). P54's median on its box, cited beside and never divided: 3.693 ms/step (OUTSIDE +-8 %, STOP-1 informational).
- **RATIO vLLM 0.30.0 / e4b int4 at B=1: 1.087  (260.3 vs 239.4 tok/s; vLLM ahead; primary pair, min-of-3 slope vs 127-step graph window, identical prompt ids)**
  - with the median-of-3 slope: 1.087; with e4b int4_r2: 1.088; vLLM / e4b NF4 control: 2.562
- **NO RATIO QUOTED for vllm-secondary at B=1 -- DRIFT: vllm-secondary graph** (readings above; a re-run is a new lane)
- build-to-build (same box, same prompts): vLLM 0.30.0 / vLLM 0.29.0 = 0.928 at B=1 (260.3 vs 280.4 tok/s)
- secondary -- eager pairing: vllm-primary eager 50.4 tok/s (0.194 of its graph arm); e4b int4_eager 35.5 tok/s (0.148 of e4b graph); eager-vs-eager 1.421 (secondary, never the headline)

## B=16  (prompts_sha f67e7e4d592b002b, rows 16, row step 18651 tokens)
| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |
|---|---|---|---|---|---|---|
| e4b | int4_r1 | VALID | 1379.2 | 11.601 | 20.1 |  |
| e4b | int4_r2 | VALID | 1379.0 | 11.602 | 20.05 |  |
| e4b | nf4_r1 | VALID | 483.8 | 33.07 | 19.38 |  |
| e4b | nf4_r2 | VALID | 484.0 | 33.058 | 19.38 |  |
| vllm-primary | graph_r1 | VALID | 1925.6 | 8.309 | 29.0 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 1924.0 tok/s; e2e incl. prefill 1536.9; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.63 GiB for consumed memory (weights + |
| vllm-primary | graph_r2 | VALID | 1917.4 | 8.345 | 30.22 (policy: reserves 0.90) | vllm 0.30.0; min-of-3; median-of-3 1911.9 tok/s; e2e incl. prefill 1533.2; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.63 GiB for consumed memory (weights + |
| vllm-secondary | graph_r1 | VALID | 1912.7 | 8.365 | 30.95 (policy: reserves 0.90) | vllm 0.29.0; min-of-3; median-of-3 1912.1 tok/s; e2e incl. prefill 1533.3; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.66 GiB for consumed memory (weights + |
| vllm-secondary | graph_r2 | VALID | 1915.2 | 8.354 | 30.95 (policy: reserves 0.90) | vllm 0.29.0; min-of-3; median-of-3 1911.9 tok/s; e2e incl. prefill 1533.3; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.66 GiB for consumed memory (weights + |
- self-pairs: e4b nf4 1.0004; e4b int4 1.0001; vllm-primary graph 1.0043; vllm-secondary graph 1.0013  (rule: inside 1.03x or DRIFT)
- e4b three axes: x2.851 over same-box NF4; rental-measured 1379.2 tok/s (11.601 ms/step) on this box; no anchor projection at B=16. P54's median on its box, cited beside and never divided: 11.42 ms/step (within +-8 %, STOP-1 informational).
- **RATIO vLLM 0.30.0 / e4b int4 at B=16: 1.396  (1925.6 vs 1379.2 tok/s; vLLM ahead; primary pair, min-of-3 slope vs 70-step graph window, identical prompt ids)**
  - with the median-of-3 slope: 1.395; with e4b int4_r2: 1.396; vLLM / e4b NF4 control: 3.980
- **RATIO vLLM 0.29.0 / e4b int4 at B=16: 1.387  (1912.7 vs 1379.2 tok/s; vLLM ahead; primary pair, min-of-3 slope vs 70-step graph window, identical prompt ids)**
  - with the median-of-3 slope: 1.386; with e4b int4_r2: 1.387; vLLM / e4b NF4 control: 3.953
- build-to-build (same box, same prompts): vLLM 0.30.0 / vLLM 0.29.0 = 1.007 at B=16 (1925.6 vs 1912.7 tok/s)

TTFT: informational only where present in the logs (vLLM: none from offline generate; e4b: scheduled PREFILL wall line) -- no ratio (P37 fixture).
