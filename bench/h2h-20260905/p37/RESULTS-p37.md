# p37 -- vLLM head-to-head, same box (/root/p37)
Rule: the position is the ratio vLLM/e4b (decode tok/s) from the PRIMARY pair (vllm graph_r1 vs e4b lic_r1) per batch size, quoted only when both sides' self-pairs are inside 1.03x and both receipts carry the same prompts_sha; VOID arms never enter a ratio. Nothing here is licensed; no cross-box number is divided into these.
```
e4b 0.35.0 @
gnf4 0.30.0
torch(e4b) 2.8.0+cu129
triton(e4b) 3.4.0
transformers 5.16.1
bitsandbytes 0.50.1
vllm 0.28.0
torch(vllm) 2.13.0+cu130
triton(vllm) 3.7.1
```
box: NVIDIA GeForce RTX 5090, 32607 MiB, 595.84 | Model name:                              AMD EPYC 7Q83 64-Core Processor | NUMA node(s):                            1 | total        used        free      shared  buff/cache   available | Mem:             251          11         107           0         132         237 | cgroup memory.max 183318347776 | Build cuda_12.9.r12.9/compiler.36037853_0

## B=1  (prompts_sha a8e6ea1d7d140dbe, rows 1, row step 298426 tokens)
| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |
|---|---|---|---|---|---|---|
| e4b | lic_eager | VOID | 12.2 | 81.99 | 19.99 | 64k pack counts 11512/776 not in log (different pack or not calibrated) |
| e4b | lic_r1 | VOID | 236.4 | 4.23 | 19.9 | 64k pack counts 11512/776 not in log (different pack or not calibrated) |
| e4b | lic_r2 | VOID | 235.8 | 4.24 | 19.9 | 64k pack counts 11512/776 not in log (different pack or not calibrated) |
| e4b | nf4_r1 | VALID | 113.4 | 8.82 | 18.91 |  |
| e4b | nf4_r2 | VALID | 113.5 | 8.81 | 18.86 |  |
| vllm | eager | VALID | 20.8 | 48.114 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 20.4 tok/s; e2e incl. prefill 20.7; vllm 0.28.0; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.66 GiB for consumed memory (weights +; Available KV cache memory: 11.03 GiB |
| vllm | fp8kv | VALID | 300.9 | 3.323 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 301.1 tok/s; e2e incl. prefill 282.5; vllm 0.28.0; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.7 GiB for consumed memory (weights + ; Available KV cache memory: 10.13 GiB |
| vllm | graph_r1 | VALID | 286.0 | 3.497 | 30.39 (policy: reserves 0.90) | min-of-3; median-of-3 286.1 tok/s; e2e incl. prefill 270.5; vllm 0.28.0; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.7 GiB for consumed memory (weights + ; Available KV cache memory: 10.34 GiB |
| vllm | graph_r2 | VALID | 286.0 | 3.496 | 30.39 (policy: reserves 0.90) | min-of-3; median-of-3 286.1 tok/s; e2e incl. prefill 270.5; vllm 0.28.0; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.7 GiB for consumed memory (weights + ; Available KV cache memory: 10.34 GiB |
- self-pairs: e4b nf4 1.0011; e4b lic 1.0024; vllm graph 1.0  (rule: inside 1.03x or DRIFT)
- **NO RATIO QUOTED at B=1 -- missing/VOID primary arm: e4b lic_r1** (both readings above; a re-run is a new lane)
- secondary -- eager pairing: vllm eager 20.8 tok/s (0.073 of vllm graph); e4b lic_eager 12.2 tok/s (0.052 of e4b graph); eager-vs-eager ratio 1.705 (secondary, never the headline)
- secondary -- fp8 KV vs kv auto (vLLM only): vllm fp8kv 300.9 tok/s (1.052 of vllm graph)

## B=16  (prompts_sha f67e7e4d592b002b, rows 16, row step 18651 tokens)
| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |
|---|---|---|---|---|---|---|
| e4b | lic_r1 | VOID | 1305.3 | 12.26 | 20.11 | 64k pack counts 11512/776 not in log (different pack or not calibrated) |
| e4b | nf4_r1 | VALID | 500.1 | 31.99 | 19.41 |  |
| e4b | nf4_r2 | VALID | 499.9 | 32.0 | 19.41 |  |
| vllm | eager | VALID | 322.5 | 49.616 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 322.4 tok/s; e2e incl. prefill 311.5; vllm 0.28.0; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.66 GiB for consumed memory (weights +; Available KV cache memory: 11.03 GiB |
| vllm | fp8kv | VALID | 2206.5 | 7.251 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 2217.4 tok/s; e2e incl. prefill 1705.1; vllm 0.28.0; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.7 GiB for consumed memory (weights + ; Available KV cache memory: 10.13 GiB |
| vllm | graph_r1 | VALID | 2030.0 | 7.882 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 2023.7 tok/s; e2e incl. prefill 1595.9; vllm 0.28.0; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.7 GiB for consumed memory (weights + ; Available KV cache memory: 10.34 GiB |
| vllm | graph_r2 | VALID | 2022.2 | 7.912 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 2020.5 tok/s; e2e incl. prefill 1593.2; vllm 0.28.0; Free memory on device (30.86/31.36 GiB) on startup. Desired GPU memory utilization is (0.9, 28.22 GiB). Actual usage is 16.7 GiB for consumed memory (weights + ; Available KV cache memory: 10.34 GiB |
- self-pairs: e4b nf4 1.0003; e4b lic n/a at B=16 (not repeated, pre-registered); vllm graph 1.0039  (rule: inside 1.03x or DRIFT)
- **NO RATIO QUOTED at B=16 -- missing/VOID primary arm: e4b lic_r1** (both readings above; a re-run is a new lane)
- secondary -- eager pairing: vllm eager 322.5 tok/s (0.159 of vllm graph)
- secondary -- fp8 KV vs kv auto (vLLM only): vllm fp8kv 2206.5 tok/s (1.087 of vllm graph)

TTFT: informational only where present in the logs (vLLM: none from offline generate; e4b: scheduled PREFILL wall line) -- no ratio (P37 fixture).
