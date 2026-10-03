# SC1b: where each engine's decode step goes. e4b's B=1 loss to llama.cpp is kernel overlap it lacks, not slower kernels (lane SC1b of #846; 2026-10-03)

Pre-registration: [`../../sc1b/SC1b-PREREG.md`](../../sc1b/SC1b-PREREG.md), with amendments A1–A3. Design reviews:
[`../../sc1b/DESIGN-REVIEW.md`](../../sc1b/DESIGN-REVIEW.md). Instrument: Nsight Systems 2025.6.1.190, `-t cuda-sw,nvtx`.
Each (engine, B) gets SC1's own unprofiled arm, a graph-mode capture and a node-mode capture of decode steps 34–97.

| run (receipt) | e4b | host (driver; board power limit, max clock) | outcome | $ |
|---|---|---|---|---|
| `sc1d-prove-1` (adertha-receipts `3cb8581`) | `36e0344` | AMD EPYC 7663, 224 vCPU (580.95.05; 600 W, 3210 MHz) | HARNESS_ERROR on proof item 3 alone → amendment A2 | 0.4124 |
| `sc1d-prove-2` (`b6ff13d`) | `77469c1` (A1, A2) | same machine, 142284 | **PROVED box=D**: the toy plus four captures, each reduced | 0.3738 |
| `sc1d-5090-1` (`659607f`) | `77469c1` | machine 142284 | NOT_RUN: instance stuck `created` for 600 s; pre-flight refused it | 0.0890 |
| `sc1d-5090-2` (`b23f9d6`) | `77469c1` | AMD EPYC 7C13, 256 vCPU (595.71.05; **400 W**, 3090 MHz), machine 45511 | OK, 24/24 passes exit 0; read under A1 (below) | 0.7788 |
| `sc1d-5090-3` (`8b3c6a2`) | `48c5d17` (A3) | same machine, 45511 | OK, 24/24 passes exit 0; read under A3 (below), **confirmatory** | 1.3860 |

SC1b total: **$3.0400 across 5 receipts**, inside the registered $4.13 (proof + box), with every run under the $15 no-ask
tier.

Box D's two runs landed on a board capped at 400 W. That is lower than SC1's boxes (500–600 W), so absolute step times here
are not SC1's. Every gap below compares engines on one box, in one run. The census box's unprofiled ratios are printed
beside SC1's in each `RESULTS` file as a diagnostic.

## The outcome by the registered rules

**`sc1d-5090-2`, under A1** (`receipts/sc1d-5090-2/RESULTS-sc1b-registered.md`; A1's scorer exactly as at `77469c1`):
- **Q2 REFUTED.** e4b runs 1,550 in-graph kernels per B=1 step against llama.cpp's 1,110, a ratio of 1.396. The prediction
  was ≥ 1.5.
- **Q1, Q3, Q4 and Q5 UNREAD.**
- One gap read: **G4 at B=16** (e4b − SGLang, ΔP +1.874 ms/step), named `norm_elem` (+1.007 ms).
- Every other gap is unread:
  - vLLM was VOID at both B (55 kept steps < 56);
  - NODE_TRACE_INFLATED on the B=1 arms of e4b, llama.cpp and SGLang;
  - NSYS_DIAGNOSTIC_ERRORS on llama.cpp B=16.

The causes are in the next section. They became amendment A3, which was **written after this data was seen and says so**.

**`sc1d-5090-3`, under A3: the confirmatory run** (`receipts/sc1d-5090-3/RESULTS-sc1b.md`). Its capture paths are
byte-identical to the proved `77469c1`; A3 changed only the reducer and the read.

| gap | comparator | B | reading | named | ΔP ms/step (e4b − comparator) | census-box ratio | SC1 ratio |
|---|---|---|---|---|---|---|---|
| G1 | llama.cpp | 1 | named | **overlap_in** (+1.380) | +1.199 | 1.358 | 1.484 |
| G2 | llama.cpp | 16 | spread | (dense_gemm −1.447, moe_expert −1.305, norm_elem +0.898) | −5.607 | 0.719 | 0.624 |
| G3 | vLLM | 1 | unread: NODE_TRACE_AMBIGUOUS | (nominal spread: norm_elem +0.388, idle_out +0.351) | +0.908 | 1.212 | 1.203 |
| G3 | vLLM | 16 | named | **norm_elem** (+1.058) | +1.812 | 1.164 | 1.172 |
| G4 | SGLang | 1 | spread | (norm_elem +0.254, attn +0.227, moe_route +0.186) | +1.118 | 1.376 | 1.268 |
| G4 | SGLang | 16 | unread: NODE_TRACE_AMBIGUOUS | (nominal **norm_elem**, +1.014 of a 0.934 threshold, band floor 0.917) | +1.868 | 1.215 | 1.191 |

| prediction | verdict | detail |
|---|---|---|
| Q1 G1's largest term is ΔI_in (A1's terms) | UNREAD | nominal largest is moe_route (+0.245); ΔI_in +0.031, band [−0.005, +0.153] |
| Q2 e4b ≥ 1.5× llama.cpp's in-graph kernels at B=1 | **REFUTED** | 1,550 / 1,110 = 1.396, the same counts as `sc1d-5090-2` |
| Q3 Δmoe_expert < ΔI_in (G1) | UNREAD | the bands overlap ([−0.113, +0.100] vs [−0.005, +0.153]) |
| Q4 Δidle_out ≥ 50 % of G2 | UNREAD | llama.cpp B=16 is PROFILER_INFLATED, so idle_out is not nameable |
| Q5 G3 B=1's largest term is ΔI_in | UNREAD | G3 B=1 is NODE_TRACE_AMBIGUOUS |
| Q6 (A3 replication) G1 names overlap_in | **HOLDS** | +1.380 of ΔP +1.199, robust at every corner of the bands |
| Q7 (A3 replication) G3 and G4 at B=16 name norm_elem | UNREAD | G3 names it; G4 names it nominally, but its band floor sits 17 µs under the threshold |
| Q8 (A3 replication) llama.cpp overlaps ≥ 90 % of B=1 kernel pairs on one stream, e4b ≤ 1 % | **HOLDS** | 95.5 % vs 0.0 % |

Q6–Q8 were written after `sc1d-5090-2` showed these effects. They are **replications** on a fresh run, not blind tests.

## What the census says

**1. e4b's largest loss, B=1 against llama.cpp, is kernel overlap, not kernel work.** Per B=1 step, in-graph:

| run | engine | summed kernel time | union (busy time) | overlap | consecutive pairs overlapping, one stream | graph span S |
|---|---|---|---|---|---|---|
| `sc1d-5090-3` | e4b | 4.087 ms | 4.087 ms | 0.000 | 0 % (of 1,549) | 4.075 ms |
| `sc1d-5090-3` | llama.cpp | 4.135 ms | 2.754 ms | **1.380** | **95.5 %** (of 1,109) | 2.711 ms |
| `sc1d-5090-2` | e4b | 4.034 ms | 4.035 ms | 0.000 | 0 % | 4.032 ms |
| `sc1d-5090-2` | llama.cpp | 4.090 ms | 2.738 ms | **1.352** | **95.6 %** | 2.679 ms |

- The two engines do almost the same kernel work per step: within 1.2 % on `sc1d-5090-3`, 1.4 % on `sc1d-5090-2`.
- llama.cpp starts each kernel before the previous one ends, on one stream. Same-stream overlap is the signature of
  programmatic dependent launch. It hides about 1.35–1.38 ms per step, more than the whole gap.
- e4b's graph has no idle gaps to speak of (I_in ≈ 0), but no overlap either. Each kernel's ramp-down is exposed.
- vLLM overlaps nothing; SGLang overlaps 0.036 ms.

So e4b's B=1 loss is a launch-structure difference, not slow kernels or graph gaps. The lever this points to, which is
not tested here, is programmatic dependent launch across e4b's decode-graph kernels (gnf4's and the Triton glue).
Q1 predicted the loss would sit in in-graph idle. In the registered term set it reads UNREAD. In-graph idle measured ≈ 0,
and the effect it was reaching for shows up as overlap instead.

**2. e4b's B=16 loss to vLLM and SGLang is KV-table glue, booked as `norm_elem`.** The frozen class map books generic
ATen ops inside the graph as `norm_elem`. Listed below is every kernel name the map put mostly in that class, as per-step
mean time (`sc1b_kernels.py`). Names partly booked elsewhere are included, so the listed time can exceed the class's
median term.

| run | engine | `norm_elem` term | top kernels, ms/step × calls/step |
|---|---|---|---|
| `sc1d-5090-3` | e4b | 1.470 | `indexSelectSmallIndex` 0.608 × 97; `elementwise_kernel` 0.222 × 144; `indexFuncSmallIndex` 0.210 × 48; `unrolled_elementwise_kernel` 0.159 × 97 |
| `sc1d-5090-3` | vLLM | 0.412 | `elementwise_kernel` 0.176 × 96; `triton_red_fused_2` 0.126 × 47; `triton_red_fused_fused_add_rms_norm…` 0.105 × 48 |
| `sc1d-5090-3` | SGLang | 0.455 | flashinfer `FusedAddRMSNormKernel` 0.198 × 96; `elementwise_kernel` 0.086 × 48; `fused_rope_kernel` 0.081 × 48; `fused_qknorm_warp` 0.071 × 48 |
| `sc1d-5090-2` | e4b | 1.461 | `indexSelectSmallIndex` 0.599 × 97; `elementwise_kernel` 0.226 × 144; `indexFuncSmallIndex` 0.208 × 48; `unrolled_elementwise_kernel` 0.157 × 97 |

About 0.8 ms of e4b's step is per-layer KV-table bookkeeping in
[`experts4bit_qlora/engines/fp8_paged_kv.py`](../../../experts4bit_qlora/engines/fp8_paged_kv.py):
- every layer's attention re-selects the active-set rows of that layer's own block table and seq-lens
  (`tbl.index_select(0, self._g_sel)` and `lens.index_select(0, self._g_sel)`, lines 814–815): 2 × 48 = the 97
  `indexSelectSmallIndex`;
- every layer's KV append publishes lengths with `seq_lens[layer].index_add_` (line 638): the 48 `indexFuncSmallIndex`.

The selection is the same for every layer in a step. vLLM and SGLang spend their `norm_elem` on fused norm and rope
kernels. The lever this points to, again untested here, is one active-set selection per step instead of one per layer.

**3. B=1 against vLLM and SGLang: no single cause.** The gaps are 0.9 and 1.1 ms/step. SGLang's spreads across
`norm_elem` (+0.25), attention (+0.23) and MoE routing (+0.19). Against vLLM, `norm_elem` (+0.39) and e4b's out-of-graph
host time (+0.35) lead, but the reading is not robust inside the node-trace bands, so it is unread.

**4. B=16 against llama.cpp, e4b's largest win (ΔP −5.6 ms/step): spread.** llama.cpp's dense GEMMs (−1.45) and expert
GEMMs (−1.31) cost more than e4b's, against e4b's own `norm_elem` (+0.90). Its out-of-graph host time cannot be named:
nsys inflates llama.cpp's B=16 step by 14.7 % (PROFILER_INFLATED; 15.1 % on `sc1d-5090-2`), mostly in host-side
   work, so Q4 reads UNREAD.

## What the first run taught the instrument (A3)

1. **vLLM's every-16th decode step is steady work.** It carries one extra out-of-graph `_apply_write_kernel`, a new KV
   block's table write, at a normal period. The "extra out-of-graph kernels" drop rule threw it away and voided vLLM by a
   single step. Such steps are now kept and counted.
2. **CUPTI's graph-id mapping messages are not lost records.** llama.cpp B=16's node capture carried 1,523 each of
   `GetGraphId(data.originalGraph…)` and `GetGraphNodeId(data.originalNode…)` INVALID_PARAMETER, while every kept replay
   held all 1,733 kernels and 48 balanced MoE segments. These two exact messages now label NSYS_GRAPH_ID_MAPPING
   without blocking.
3. **Node tracing adds about 0.04–0.13 µs per kernel**, which is 1.8–4.1 % of a B=1 step across the two runs. A flat 2 %
   gate blocked every B=1 arm on `sc1d-5090-2`. A3 bounds the overhead instead (ω = node span − S) and reads a gap only when its reading holds across the
   bound. That is why G3 B=1 and G4 B=16 stay unread on `sc1d-5090-3`.
4. **G1's remainder was the overlap above.** A3 made it an explicit term. Q1–Q5 keep A1's term set, so the new term
   cannot change what they mean.

`receipts/sc1d-5090-2/a3-exploratory/` is `sc1d-5090-2` re-reduced under A3. It is **exploratory**: A3 and Q6–Q8 were
derived from it.

## Files

- `receipts/<run>/`: each run's arm receipts, census records (`census_*.json`), the box's own reducer output
  (`sc1b_arm_*.json`, `sc1b_gap_*.json`), `summary.txt`, `outer.log`, `forensics.txt`, `versions.txt` and `logs/`.
- Box runs also carry `RESULTS-sc1b*.md` and `verdicts*.json`.
- The nsys reports and sqlite exports (about 150 MB a run), and the launcher's own receipts, stay in the private receipt
  store at the commits in the table. `sc1d-5090-1` never reached the lane, so it has no files here.
