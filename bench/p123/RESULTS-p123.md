# P123 results: where the shipped default's single-stream decode step goes (#1313)

## The reading (`p123-5090-1`): **READ**

**Registration:**
- P123 itself: #1394, merged in `34150a5d`.
- Amendment 1, #1405 (`e691dbe7`): the proof checks that every router licenses under `E4B_FUSE_ROUTER_EPI=auto` alone.
- Amendment 2, #1411 (`b371053a`): the class map names `torch.cat`'s `CatArrayBatchedCopy`.

Two proofs ran first. `p123-prove-1` VOIDed on the class map, which led to Amendment 2; `p123-prove-2` PROVED. The
maintainer re-derived the reading from the receipt store (`6dbc2361`) with main's reducer and got a byte-identical
`verdict_p123.json`.

**Code under test:**
- e4b 0.50.0 at `b371053a`;
- grouped-nf4-gemm 0.43.0 at `6ee2e10`, with its bandwidth decode GEMV at its default;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2;
- Nsight Systems 2025.6.1;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, its NF4 arena baked on the box.

**Subject:** the shipped default `serve_paged`, with every lever unset.
- Graphs on, buckets 1–16 captured, all-vram, `E4B_PAGED_MAX_SEQS=16` named.
- The B=1 fused stack resolved `default-allowlisted` on `qwen3_moe` in both speed arms, with census
  **48 / 193 / [48, 48] / 48**. So the router epilogue licensed on all 48 routers on a 5090 with #1398's fp32 probe.
- The build dispatched `bw_prmt32` ×288 and no dot-pad, in both arms.

**The instrument:** SC1b's census (`bench/sc1b/sc1b_census.py` at SC1b's bytes), with P123's v1.1 NF4 class map.
- **Two captures per batch:** graph mode supplies the step P, and node mode attributes it to kernel classes.
- **Batches:** B = 1 and B = 16, 64 steps each.
- **RESIDUAL** = P − Σ of the eight named classes = the in-graph idle, the idle between graph replays, the overlap and
  any unmapped kernel time.
- **Busy fraction** = (P − I_in − idle_out) / P.

### The step, by class

| | B = 1 | share | predicted | | B = 16 | share | predicted | |
|---|---|---|---|---|---|---|---|---|
| P (graph-profiled) | **4.7475 ms** | | | | **17.9429 ms** | | | |
| unprofiled step (the speed arms' W1 / W16 mean) | 4.5599 ms | | | | 18.3211 ms | | | |
| `dense_gemm` | 1.8142 ms | **0.382** | 0.33–0.50, largest | held | 2.5160 ms | 0.140 | 0.12–0.30 | held |
| `moe_expert` | 1.2112 ms | 0.255 | 0.15–0.28 | held | 12.3899 ms | **0.691** | 0.40–0.65, largest | **MISSED** |
| `attn` | 0.7079 ms | 0.149 | 0.06–0.15 | held | 0.8924 ms | 0.050 | 0.04–0.15 | held |
| `moe_route` | 0.5087 ms | **0.107** | 0.02–0.09 | **MISSED** | 0.7714 ms | 0.043 | 0.02–0.10 | held |
| `norm_elem` | 0.3716 ms | 0.078 | 0.04–0.12 | held | 0.7030 ms | 0.039 | 0.02–0.10 | held |
| `sample` + `input_prep` + `memcpy` | 0.0156 ms | 0.003 | | | 0.0336 ms | 0.002 | | |
| RESIDUAL | 0.1183 ms | 0.025 | 0.02–0.15 | held | 0.6366 ms | 0.035 | 0.01–0.12 | held |
| busy fraction | **0.923** (read) | | ≥ 0.85 | held | **0.962** (read) | | ≥ 0.90 | held |
| kernels per step (graph modal) | 1072 | | 900–1600 | held | 1168 | | 900–1700 | held |

- **Unmapped kernels: none.** The map named every kernel at both batches; the census residual fraction is 0.0.
- **SC1b's labels:**
  - B = 1 is `NODE_TRACE_INFLATED`, which is information, as in SC1b. Node tracing slows the step, and the shares are
    read from node mode against graph mode's P.
  - B = 16 carries no label.
  - Profiler inflation was 0.041 at B = 1 and −0.021 at B = 16, both inside SC1b's 5 % gate, so both busy fractions
    were read.
- **The RESIDUAL row at B = 1:**
  - 0.344 ms idle between graph replays (the host gap);
  - 0.020 ms idle inside the graph;
  - −0.245 ms of overlap between in-graph kernels.

### Where the B = 1 step goes

Counts are launches per step, from SC1b's name map over the 63 node-mode steps.
- **`dense_gemm`, 1.81 ms, the largest class.** It is 146 launches: 145 cuBLAS GEMVs (three per layer, plus one) and
  one `gemmk1_kernel`.
  - Registered arithmetic put the bf16 attention projections and lm_head at about 2.43 GB per token, about 1.6 ms at
    1.5 TB/s. The measured 1.81 ms is about 1.34 TB/s across those bytes.
  - This is the class the registration predicted to lead, and it does.
- **`moe_expert`, 1.21 ms.** It is the bandwidth GEMV `_gemv_nf4_bw` (96 launches, two per layer) and `_swiglu_rows`
  (48).
- **`attn`, 0.71 ms.** It is `_fp8_append_bt1_side` (96) and `_fp8_paged_decode_split_f8dot` (48), at context
  positions 35–98.
- **`moe_route`, 0.51 ms: MISSED** (0.107 against a predicted 0.02–0.09).
  - It is 336 launches, seven per layer inside the MoE segment, averaging about 1.5 µs each:
    - `_router_epilogue` and `_combine_rows`;
    - one `indexSelectSmallIndex`;
    - four elementwise kernels.
  - The registration priced routing by its arithmetic. In fact it is launch-bound glue: at B = 1 these small kernels
    cost more than their bytes.
- **`norm_elem`, 0.37 ms.** It is `_rope_norm_heads` (96), the two rmsnorm kernels (97) and elementwise glue.
- **The host gap, 0.34 ms (7 %).** It is the idle between replays. P118's lookahead recovered the whole gap and read
  1.0198×.

### Where the B = 16 step goes

- **`moe_expert`, 12.39 ms, 69 % of the step: MISSED** (above the predicted 0.40–0.65).
  - It is `_gemm_nf4_grouped_smallm` (96 launches, two per layer) and `_swiglu_rows` (48).
  - That is 10.2× B = 1's expert time for 16× the routed rows. At top-8, 16 rows route 128 rows per layer across 128
    experts, so the step reads most of each layer's expert weights where B = 1 reads eight experts' worth.
  - The small-M grouped GEMM, not the dense projections, is the B = 16 step.
- **`dense_gemm`, 2.52 ms (14 %).** It is cuBLAS's `Kernel2` (145) and `splitKreduce_kernel` (48).
- **`moe_route`, 0.77 ms.** It now includes the device tile table `_tile_table_r1` (48).

### Against the predictions

- **B = 1:** dense_gemm, moe_expert, attn, norm_elem, RESIDUAL, busy fraction and kernels per step held.
  **moe_route MISSED:** 0.107 against 0.02–0.09.
- **B = 16:** dense_gemm, attn, moe_route, norm_elem, RESIDUAL, busy fraction and kernels per step held.
  **moe_expert MISSED:** 0.691 against 0.40–0.65. It is still the largest class, as predicted.
- **The verdict:** READ, predicted about 85 %. Held.

A miss is reported here and does not void the read. Neither miss moves anything: the registration has no default
consequence. They price the next lever.

### Speed (the unprofiled arms; reported, not ruled)

| | S1a | S1b | S1b / S1a |
|---|---|---|---|
| W1 decode, ms per step | 4.5392 (220.3 tok/s) | 4.5806 (218.3 tok/s) | 1.0091 |
| W16 decode, ms per step | 18.2412 (877.1 tok/s) | 18.4010 (869.5 tok/s) | 1.0088 |

- **Self-pairs:** inside [0.97, 1.03], so not NOISY.
- **Against P115 Phase D:** D1, the same default on another 450 W board, decoded W1 at 4.64 ms.
- **Peak memory:** 22.56 GiB allocated in the speed arms; 22.81 GiB reserved in the census.

### What it prices (not a choice)

The registration chooses no lever.
- **Next for B = 1:** the dense bf16 GEMVs, 1.81 ms of a 4.75 ms step.
  - A 4-bit route for the attention projections (and possibly the lm_head) would cut those bytes about 3.5×.
  - It changes the arithmetic, so it needs a quality bar as well as speed, and is registered on its own.
- **The second B = 1 lever:** launch-bound glue. moe_route and norm_elem alone are 635 small launches per step, in
  0.88 ms.
- **At B = 16:** the small-M grouped GEMM is the step. That is the serve-throughput program's lever, not this lane's.

### The reading (`p123-5090-1`)

**Host:** one RTX 5090 on an AMD EPYC 7C13 host (256 threads, ~1 TB RAM), 320 GB free. It was Vast instance 54935034.
- The card: sm_120, driver 595.71.05, 170 SMs, **power limit 450 W**, SM clock max 3090 MHz.
- Nsight Systems ran as root. CPU sampling was unavailable (perf paranoid level 4), which the GPU trace does not use.

**Cost:** $0.844, 16 minutes from launch to teardown (00:06:12–00:22:21Z).

**Timeline (the box's own log, UTC):**
- the self-tests and the premise passed by 00:09 (19 passed, none skipped);
- Qwen3-30B-A3B was fetched by 00:15, then baked, with the prompts made by 00:16;
- the speed arms S1a and S1b ran 00:16–00:18;
- Nsight Systems was installed by 00:19, and the four captures ran 00:19–00:22;
- TP_DONE at 00:22.

## The proofs

**`p123-prove-1`: VOID** ($0.189), from `e691dbe7` on Granite-3.1-3b-a800m.
- The B = 1 census left 2.2 % of the step unnamed, against SC1b's 2 % gate.
- All of it was `torch.cat`: `CatArrayBatchedCopy`, 64 per step (`rotate_half` on q and k across Granite's 32 unfused
  layers), and `CatArrayBatchedCopy_alignedK_contig`, one per step.
- The v1 map's `cat` never matched, because SC1b's `_hit` is case-sensitive. Amendment 2 fixed the map.
- Amendment 1's router build licensed **32 of 32** routers (`explicit`, `failed_probe` 0). `fam-prove-1` licensed 27 on
  a 5090 before #1398, so this is #1398's regression check on a real card.

**`p123-prove-2`: PROVED** ($0.069), from `b371053a`.
- The census residual was 0.0 at both batches, and the routers licensed 32 of 32 again.
- B = 1's busy fraction was UNREAD: the capture was `PROFILER_INFLATED` (0.0514 against the 5 % gate). That is the
  registered rule; the gate was not moved.

**Lane spend: $1.102** of the $3.00 ceiling (the hard stop is $4.00):
- `p123-prove-1`: $0.189;
- `p123-prove-2`: $0.069;
- `p123-5090-1`: $0.844.

## Receipts

`bench/p123/receipts/<run>/` holds each run's records:
- the speed arms;
- the census drivers' and arms' records;
- the verdict;
- `router_census.json` (proofs only);
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- `logs/`;
- `SHA256SUMS`.

The raw Nsight exports stayed on the boxes, as SC1b's did. The private receipt store holds each run's `receipt.json`
and `teardown-proof.json` (`receipts/experts4bit-qlora/2026-10-0{8,9}/p123-*`).
