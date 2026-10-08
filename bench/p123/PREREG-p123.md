# P123: where the shipped default's single-stream decode step goes — a kernel-class census of the new default

Lane P123 of #1313, claimed by pushing `prereg/p123` at 2026-10-08T21:57:38Z on `bf26a73d`, the merge commit of #1361.
The maintainer ACKed it on the bus at 2026-10-08T21:47Z, with four conditions that this registration carries:
1. it is registered at #1361's merge commit, on the default server;
2. there are predictions per kernel class, with a RESIDUAL row, so the breakdown cannot hide a hole;
3. the GPU busy fraction is recorded per step, so the read says whether B=1 is device- or host-bound before any
   lever is priced on bytes;
4. ratios are within the box only, peak memory is reported, and a census changes no default.

## The question

Since #1361, the default `serve_paged` decodes one Qwen3-30B-A3B NF4 sequence at 4.64 ms per token (P115 Phase D's D1,
on a 450 W board). The last device census of this route, SV2 (`bench/hybrid-g9/sv2/RESULTS-sv2-device-census.md`),
predates both levers now on by default:
- the B=1 fused stack (P115);
- grouped-nf4-gemm 0.43.0's bandwidth decode GEMV (P116).

**Where does the new default's decode step go?** The answer prices the next lever before any kernel work.
Architecture arithmetic, not a measurement, says the bf16 attention projections (~1.81 GB per token) and the bf16
lm_head (~0.62 GB) now outweigh the NF4 experts.

## The subject

The shipped default `serve_paged` at the launch commit (#1361 or later), on one RTX 5090:
- every lever and engine knob unset except `E4B_PAGED_MAX_SEQS=16`, which is named as P115 Phase D named it;
- decode graphs on, buckets 1–16 captured, the all-vram placement;
- Qwen3-MoE's four fusion knobs resolving to `auto` by default (census `48 / 193 / [48, 48] / 48`);
- grouped-nf4-gemm **`6ee2e10`** (0.43.0), whose bandwidth GEMV is the default at Qwen3's shapes.

Model: Qwen/Qwen3-30B-A3B @ `ad44e777`, NF4 arena baked on the box (P39's `k8_bake.py`). Workloads: P109's W1 (one
sequence) and W16 (16 sequences), 512-token prompts. B = 1 is what a single user gets on the default 16-slot server.

## The instrument

SC1b's census, reused unchanged by import: `bench/sc1b/sc1b_census.py` and `sc1b_e4b_census.py`'s `census_window`,
staged at SC1b's bytes. Nsight Systems 2025.6.1 is installed on the box as SC1b installed it.

**Speed arms** (`p123_box.py`): S1a and S1b, two fresh processes on the default server, each through W16 and W1.
- 32 and 160 new tokens, 1 warm and 3 timed passes, P109's decode-only slope.
- They give the census its unprofiled step time and its noise, with the fusion census, how each knob resolved,
  grouped-nf4-gemm's dispatch tally and the peak memory.

**Captures** (`p123_census.py`): for B = 1 and B = 16, one graph-mode and one node-mode `nsys profile` of exactly 64
steady decode steps.
- Each run is 32 skip steps after the ramp, inside 160-token rows, bracketed by `cudaProfilerStart` / `Stop`.
- Exported to sqlite, the pair is read by `sc1b_census.py arm --moe-layers 48` with **`kernel_classes_nf4.json`**,
  the v1 class map.

**The v1 class map** is SC1b's v0 e4b map with only the NF4 expert kernels added to `expert_names` and
`matmul_names`:
- the decode GEMVs `_gemv_nf4_bw`, `_gemv_nf4_dotpad`, `_gemv_nf4_dotpad_splitk`, `_gemv_nf4_grouped` and
  `_gemv_nf4_grouped_splitk`;
- the grouped GEMMs `_gemm_nf4_grouped` and `_gemm_nf4_grouped_smallm`.

v0 was frozen for the int4 route, so it would have classed the NF4 expert GEMV as routing or dense. Its rules, rule
order, MoE segment anchors (`_router_epilogue` / `topk` → `_combine_rows`) and inherit lists are unchanged.

The v1 names come from a kernel-name inventory of the default decode step on the RTX A2000:
- a two-layer random-weight Qwen3-MoE with Qwen3-30B-A3B's per-layer dimensions;
- the fused stack at `auto`, and the bandwidth GEMV forced;
- host and device grouping, B = 1 and B = 16;
- names and per-step counts only, never a timing (#1133).

It confirmed the segment anchors bracket every expert kernel on the NF4 route.

**Raw exports.** The Nsight reports and their sqlite exports stay on the box, as SC1b's did. The census arm records,
which carry per-class medians and IQRs and kernels per step, are the receipts.

## Read per batch

- **P:** the step (graph mode, delimiter to delimiter).
- **Each class's time and share of P**, over SC1b's classes: `moe_expert`, `moe_route`, `attn`, `dense_gemm`,
  `norm_elem`, `sample`, `input_prep`, `memcpy`.
- **The RESIDUAL row:** P minus the eight named classes. It holds:
  - the unmapped kernels (SC1b's `residual` class, gated ≤ 2 % by the census itself);
  - the in-graph idle `I_in`;
  - the out-of-graph idle `idle_out`;
  - the overlap terms.

  A hole in the class map or a host stall shows up here.
- **The GPU busy fraction** `(P − I_in − idle_out) / P`. It reads UNREAD if SC1b's G-inflate gate fires (the profiled
  step more than 5 % off the unprofiled one), because then `idle_out` is not nameable.
- **Kernels per step** (node mode, modal).
- **Peak memory:** the speed arms' allocated peak and the capture's reserved peak.

All ratios are within this box.

## The rule (`p123_reduce.py`, self-tested on 19 cases)

The first rung that applies:
1. **VOID**, if any of these holds:
   - a record is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - it is not the shipped default: a fusion knob set or resolved other than `default-allowlisted`, the fusion
     census off `48 / 193 / [48, 48] / 48`, or the build's GEMV dispatch not `bw_prmt32` without dot-pad;
   - a speed slope is void;
   - a census arm is void, or labelled `CLASS_MAP_INCOMPLETE`, `CLASS_MAP_SEGMENT_BROKEN`,
     `NSYS_DIAGNOSTIC_ERRORS` or `CLOCK_MISMATCH`;
   - a capture window is not exactly 64 steps at its batch.
2. **NOISY:** the speed self-pair S1b/S1a outside [0.97, 1.03] on W1 or W16.
3. **READ:** otherwise.

There is no default consequence. A census prices the next lever and moves nothing. `PROFILER_INFLATED` leaves the read
standing with the busy fraction UNREAD, and `NODE_TRACE_INFLATED` is information, as in SC1b.

## Predictions (written before any data)

Shares of P on Qwen3-30B-A3B. Each is graded HELD or MISSED; a miss is reported plainly and does not void the read.

| | B = 1 | B = 16 |
|---|---|---|
| `dense_gemm` (bf16 attention projections, router logits, lm_head) | **0.33–0.50**, the largest class | 0.12–0.30 |
| `moe_expert` (bandwidth GEMV / small-M grouped GEMM, swiglu) | 0.15–0.28 | **0.40–0.65**, the largest class |
| `attn` (fp8 paged decode and append) | 0.06–0.15 | 0.04–0.15 |
| `moe_route` | 0.02–0.09 | 0.02–0.10 |
| `norm_elem` | 0.04–0.12 | 0.02–0.10 |
| RESIDUAL | 0.02–0.15 | 0.01–0.12 |
| GPU busy fraction | **≥ 0.85**: B = 1 is device-bound | ≥ 0.90 |
| kernels per step | 900–1600 | 900–1700 |

- **The basis for B = 1's dense share.** ~2.43 GB per token of bf16 weights streams in ~1.6 ms at 1.5 TB/s. The NF4
  experts' ~0.96 GB streams in ~0.65 ms at the byte floor; the bandwidth GEMV runs at roughly 66–80 % of it.
- **The busy fraction.** P118 measured the host gap at 0.207 ms of a ~7 ms step, so the device is busy ≥ 0.85.
- **The verdict:** READ about 85 %.

## The premise and the proving rental

**Premise**, on the card before anything is fetched (rc 25): P115 Phase D's four tests, **19 passed**, none skipped:
- `tests/test_decode_graph_buckets.py`;
- `tests/test_kv_step_select.py`;
- `tests/test_fused_glue_decode_graphs_gpu.py`;
- `tests/test_gemv_bw_served_gpu.py`.

**Proof** (`p123-prove-<n>`): the whole box on Granite-3.1-3b-a800m at 8 / 24 tokens and 1 rep.
- Its default is the unfused stack (`granitemoe` resolves to `0`) and grouped-nf4-gemm's scalar GEMV.
- Its census reads 32 layers (`--moe-layers 32`).
- A VOID fails the proof (rc 27); the proof's verdict is not a reading.

**Order on the box:**
1. Refusals: CUDA 18, card class 15, disk 13, host RAM 16.
2. Install; the tripwire checks the pins and that the default resolves `auto` on `qwen3_moe` and `0` on `granitemoe`,
   with the bandwidth GEMV at `auto` at Qwen3's shapes.
3. The reducer, box and census self-tests, then the premise.
4. Nsight Systems (26 if it will not install), then fetch, bake and prompts.
5. The two speed arms, then the four captures. They are skipped when an arm failed, and the reducer VOIDs.
6. The census arms and the reducer.

## Budget and STOP

- **Proof:** one RTX 5090, **guard 0.75 h**, `--download-gb 7`, about $0.65 at the launcher's policy rate.
- **Reading:** one RTX 5090, **guard 1.5 h**, `--download-gb 61`, about $2.0 at the policy rate.
- **Ceiling:** $3.00 for the lane (the proof, the reading and one rerun), hard stop $4.00. Every run sits inside the
  owner's standing no-ask tier for a single run under $15; anything over $15 needs the maintainer lane's approval.
- **STOP:**
  - the refusals above;
  - every time-left check inside its guard (`NEED_FETCH`, `NEED_BAKE`, `NEED_ARM`, `NEED_CAPTURE`);
  - no in-launch retry;
  - the driver refuses a dirty tree, or a staged file that differs from `bench/p123/staged-p123.sha256`.

**Receipts.** The run directory's `p123/` is committed to `bench/p123/receipts/<run>/`:
- `arm_S1a.json`, `arm_S1b.json`;
- `census_drv_b{1,16}_{graph,node}.json`, `census_b{1,16}.json`;
- `verdict_p123.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- `logs/` (added with `git add -f`) and `SHA256SUMS`.

## What this does not claim

- No speed claim: the profiled walls are never speed readings, and the unprofiled arms are the reference.
- No default change.
- No comparison with another engine (SC1b did that).
- No lever is chosen here. The next lever's lane is registered on its own; a 4-bit attention-projection or lm_head
  lever changes the arithmetic, so it needs a quality bar as well as speed (the maintainer, 21:47Z).
