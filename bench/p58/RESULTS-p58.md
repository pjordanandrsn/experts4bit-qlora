# Results — P58: the same-box vLLM head-to-head re-run — e4b's current int4 stack vs vLLM 0.30.0 (latest) and 0.29.0, one RTX 5090, 2026-09-22

Pre-registration: [`P58-PREREG.md`](P58-PREREG.md) (#676, merged 4cab05a before the run). Receipts: [`receipts/`](receipts/) — 18 arm receipts (e4b step receipts with token lists; vLLM slope receipts with every resolved knob and the generated token ids), `prompts_b{1,16}.json` (the identical token ids both engines decoded, digests `a8e6ea1d…` / `f67e7e4d…` — the same rows P37 used on 2026-09-05), the trimmed engine logs (engagement banners, Marlin/graph-capture/KV lines), `summary.txt`, `forensics.txt`, `versions.txt`, `teardown-proof.json`. Every number below is read from them by [`p58_reduce.py`](p58_reduce.py); the tables under "Generated read" are its output, unedited.

Lane `p58-5090-1`: Vast instance 52069847, RTX 5090 (32,607 MiB, driver 580.119.02, 575 W / 3090 MHz) on an **AMD EPYC 9655** host, image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`; rented 14:05Z, `TP_DONE` 14:47Z, destroyed 14:47:09Z (proven); **$0.31** against the $3.00 estimate. e4b **0.36.4 @ 4cab05a** + grouped-nf4-gemm **0.32.1 @ 65cb104** (K17 merge; `GNF4_GEMV_FUSED_REDUCE` unset → two-launch default, asserted by the tripwire), transformers 5.16.1, torch 2.8.0+cu129, triton 3.4.0, in the image python. vLLM **0.30.0** (PRIMARY, the latest PyPI release at launch — released that morning) and **0.29.0** (SECONDARY, pinned) each in its own venv (torch 2.13.0+cu130, triton 3.7.1), both installed and imported first try on the cu12.9 image; both selected `MarlinLinearKernel` + `MarlinExperts` for Qwen's GPTQ-Int4 checkpoint and captured CUDA graphs (`FULL_AND_PIECEWISE`, 51 piecewise + full decode graphs) — asserted in every graph arm's log.

## The position (same box, same session, identical prompt token ids)

| B | e4b NF4 control | **e4b int4 stack** (P54's) | vLLM 0.30.0 (graph) | vLLM 0.29.0 (graph) | **ratio vLLM 0.30.0 / e4b int4** | ratio vLLM 0.29.0 / e4b int4 |
|---|---|---|---|---|---|---|
| **1** | 9.845 ms · 101.6 tok/s | **4.177 ms · 239.4 tok/s** (fused q/k/v) | 3.841 ms · **260.3** (r1 260.3 / r2 260.4) | 280.4 / 259.9 — **DRIFT** (1.079×) | **1.087** (vLLM ahead) | not quoted |
| **16** | 33.07 ms · 484 tok/s | **11.601 ms · 1379.2 tok/s** (unfused) | 8.309 ms · **1925.6** (r2 1917.4) | 8.365 ms · **1912.7** (r2 1915.2) | **1.396** (vLLM ahead) | **1.387** (vLLM ahead) |

Self-pairs: e4b NF4 1.0007 / 1.0004, e4b int4 1.0007 / 1.0001, vLLM 0.30.0 1.0004 / 1.0043, vLLM 0.29.0 **1.0789** / 1.0013 (B=1 / B=16). Build-to-build: vLLM 0.30.0 / 0.29.0 = **1.007 at B=16**; at B=1 the 0.29.0 pair drifted, so the 0.928 figure the reducer prints from the first draws is not a position.

## Predictions, read against the rows

- **P1 (vLLM latest at B=1 ahead, 1.02–1.20, central 1.10) — HOLDS: 1.087.** vLLM 0.30.0 read 260.3 / 260.4 tok/s on this host (P37's box read 286.0 / 286.0 for 0.28.0 — B=1 is host-bound on both engines and this EPYC 9655 host is slower); e4b's int4 stack 239.4 (4.177 ms, the fusion engaged: `fused q/k/v projections on 48 attention modules` in both draws).
- **P2 (vLLM at B=16 ahead, 1.35–1.75, central 1.52) — HOLDS: 1.396 (0.30.0) and 1.387 (0.29.0).** vLLM 1925.6 / 1912.7 aggregate (P37's 0.28.0 read 2030 on its box); e4b 1379.2 (11.601 ms; P57 read 11.578 on another EPYC host the same day, P54 11.420 on its box).
- **P3 (0.30.0 vs 0.29.0 within ±5 %) — HOLDS at B=16: 1.007.** At B=1 it is **NOT READ**: 0.29.0's two engine starts read 280.4 then 259.9 tok/s (walls 0.137 → 0.146 s for 32 tokens, 0.479 → 0.515 s for 128; the three reps inside each start agree to 0.1 %; identical resolved config, identical Marlin selection, identical graph capture, identical KV size) — a **7 % swing between two fresh engine instances of the same build on the same box**, while 0.30.0's two starts agreed to 0.04 %. Three of the four vLLM B=1 starts on this host read 260; the pre-registered rule says DRIFT, prints both, quotes nothing. Recorded as an instrument fact about vLLM B=1 on this host, not explained.
- **P4 (eager pairing, secondary) — vLLM eager 0.194 of its graph arm and e4b eager 0.148 of its graph arm are both OUTSIDE their registered 0.06–0.10 / 0.04–0.08 bands (P37: 0.073 / 0.052); eager-vs-eager 1.421 is inside 1.3–2.2.** Both engines' eager paths cost less on this host than on P37's — consistent with a slower host where launch-bound loops lose less to the graph. Never the headline.
- **P5 (instrument) — HOLDS where it could be read.** Self-pairs inside 1.03× on five of six pairs (the sixth is the 0.29.0 B=1 DRIFT above). e4b int4 / NF4 same-box: **×2.356 at B=1** (band 2.3–2.7) and **×2.851 at B=16** (band 2.4–3.0). STOP-1: B=16 11.601 vs P54's 11.420 = +1.6 % (within ±8 %); B=1 4.177 vs P54's fused 3.693 = **+13.1 %, OUTSIDE** — the third box today (P57's two hosts read 4.24 / 4.18 on the same stack vs P54's 3.69), so the B=1 step is host-bound and P54's box was the fast one; informational, the same-box ratios stand.
- **P6 (tokens diverge between engines) — recorded, not gated:** both receipts carry the generated ids per row; different quantisers, different continuations, as expected.

## Three axes for e4b's arm (rule 4)

- **B=1:** ×2.356 over the same-box NF4 control; rental-measured **239.4 tok/s (4.177 ms/step)** on an EPYC 9655-hosted 5090; anchor-class PROJECTION 159.2 × 2.356 ≈ **375 tok/s** (the class was never certified).
- **B=16:** ×2.851 over the same-box NF4 control; rental-measured **1379.2 tok/s (11.601 ms/step)** on this box; no anchor projection at B=16.

## What this says, and what it does not

- **Position, current vs current, one box, one prompt set:** vLLM 0.30.0 (GPTQ-Int4, Marlin) decodes **1.09× faster at B=1 and 1.40× faster at B=16** than experts4bit-qlora's current int4 stack (RTN int4 experts, uncalibrated int4 attention, K16 route, fused q/k/v at B=1). Against e4b's NF4 control the same vLLM build is 2.56× / 3.98× ahead. The 2026-09-05 position (P37: no ratio quoted against the licensed stack; vLLM / NF4 2.52 / 4.06) is superseded as the current comparator; `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05` stays as measured history.
- **The engine advantage is understated, not inflated:** vLLM's slope includes its scheduler and detokenisation; e4b's replay window has neither.
- **Quality is quoted, never equated:** Qwen's GPTQ g128 checkpoint carries its author's evaluation; e4b's experts-int4 is register-certified (Δppl −0.084 vs NF4) and its uncalibrated attention-int4 passed the one-sided |Δ| ≤ 0.05 rule on the reference box; this lane scored neither.
- **Not measured:** TTFT/prefill, other batch sizes or context lengths, footprint (vLLM reserves 0.90 of VRAM by policy), any other checkpoint, `GNF4_GEMV_FUSED_REDUCE=1` (K17/P57: slower), fp8-KV vLLM.
- **Decision rule outcome:** nothing changes in the defaults. The B=16 ratio (1.40) sits inside its band, below the 1.75 trigger — so the next B=16 lever work is not forced by this lane; the register rows and the serving-throughput position are re-pointed at these numbers with vLLM's version pinned. The build-to-build delta (1.007 at B=16) says pinning 0.29 vs latest does not matter for this comparator; the B=1 engine-start variance says a vLLM B=1 arm needs more than two engine starts before its self-pair rule can be trusted on a slow host.

## Generated read (`p58_reduce.py`, unedited)

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
- build-to-build at B=1: NOT QUOTED -- a vLLM self-pair is outside 1.03x (first draws 260.3 vs 280.4 tok/s, printed, not a position)
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
