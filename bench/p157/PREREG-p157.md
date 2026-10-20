# P157: what bounds K25 — its stages knocked out one at a time, at today's plan on Qwen3-30B-A3B's W16 shapes

**Status:** registered before any data. A microbench PREMISE for stream (B) of e4b#1313: it reads where K25's time goes
and whether its DRAM stream overlaps its on-chip work, so the next decode lane is chosen from a measurement rather than
a guess. It changes no default, no gate and no speed claim, and it quotes no gain. The maintainer and loggetta set its
conditions on the bus on 2026-10-19 and 2026-10-20; each is mapped below.

| condition | where |
|---|---|
| ceilings, not a ledger: each arm prices one stage's removal; A12's additivity on paired per-rep differences | [The reading](#the-reading-p157_reducepy-fixed-before-data) |
| r_dram from the ARMS, its bands in order, the small-denominator rule, the larger-of SE | [r_dram](#r_dram-from-the-arms) |
| every difference r uses is taken inside ONE cubin (data-only), else VOID | [The arms](#the-arms-p157_armspy) |
| the residency ladder: a dose series of DRAM bytes, one cubin, its own out-of-sample check of r's band | [The ladder](#the-ladder-check-of-rs-band) |
| the bytes axis: Nsight's DRAM bytes where the host allows them, else the router-replay ESTIMATE, labelled | [The bytes axis](#the-bytes-axis) |
| SASS load parity against A0; A3 on down by whole-function totals; A2b strictly cheaper; the ladder's math parity | [Gates](#per-arm-gates-and-labels) |
| A2a (a PTX knock-out of the MMA) best effort: NOT BUILT, with its arithmetic | [The arms](#a2a-not-built) |
| occupancy pinned to A0's by dynamic shared memory, checked by the driver and read back from the graph nodes; the carveout recorded | [Pins](#pins-occupancy-held-at-a0s) |
| clocks: NVML per arm (SM, memory, averaged and instantaneous power, the enforced limit, event reasons); 2 % either way labels | [Gates and labels](#per-arm-gates-and-labels) |
| a spacer graph before every timed replay; a Williams order over whole squares (70 reps) | [The reps](#the-reps) |
| ACX0 timed (the rotation and the layout priced); a 1 % copy-gap tolerance | [Copy checks](#copy-checks) |
| a $0 A2000 shakedown; the guard from measured walls x1.5 with 25 % headroom | [Shakedown](#before-the-box-the-0-a2000-shakedown), [Budget](#budget-and-sequencing) |

## Why

K25 (grouped-nf4-gemm's `nf4_smallm._gemm_nf4_grouped_smallm`) is the served NF4 store's batched-decode expert GEMM.
At W16 its two calls a layer hold **11.75 ms** of the step (P152's census); P155's microbench read **12.075 ms** for the
same pair on S0's real routing. Two floors frame that time:
- **the bytes:** P154's byte floor, **4.94 ms** for both GEMMs a step;
- **the instructions:** K25's K loop issues **24.1 lane instructions per weight on sm_120** (triton 3.4.0, ptxas
  12.8.93; gnf4 `bench/sass_count`), an issue estimate of **5.2-6.1 ms** at 170 SMs.

The recorded time is above the SUM of the two floors (10.1-11.0 ms). P155 found no plan in K25's 144-plan grid that moves
it (best 1.019x). P156 (registered at e4b#1857) read K39's cheaper decode at 1.1755x, so on-chip instructions are at least
part of it. What none of these says is **how the DRAM stream and the on-chip work combine**: if they overlap, only the
larger of the two matters, and a cheaper decode helps only while on-chip work dominates; if they serialize, every
stage's saving shows up in full, and the lever is to overlap them (deeper pipelining, more resident warps), not only to
shrink them. P157 measures that, and prices each stage's removal, by timing K25 with one stage knocked out at a time.

The method is Micikevicius's (Analysis-Driven Optimization, NVIDIA GTC 2010 / SC10): time memory-only and math-only
versions of the kernel and compare the full time with their sum and their max. Its three rules are kept: the access
pattern and the load count are unchanged (SASS load parity), dead-code elimination is defeated (every accumulator is
stored, and every loaded word feeds it), and occupancy is held (dummy shared memory). Volkov (Understanding Latency
Hiding on GPUs, 2016, sections 4.11-4.12) gives the reading: full near the sum means latency-bound, full near the max
means throughput-bound.

**The question:** with K25's loads, decode, operand staging and MMA knocked out one at a time on the served call form,
(1) what is each stage's ceiling, and (2) does K25's DRAM stream overlap its on-chip work (r_dram near 0), serialize
with it (near 1), or neither?

## The arms (`p157_arms.py`)

`_k25_ablate` is K25 at the pin, copied verbatim except: the K loop is `tl.range(0, NKC, num_stages=3,
loop_unroll_factor=1)` (triton 3.4.0 multi-buffers loads only when they feed a `tl.dot`, so every arm names its
pipelining explicitly); two constexprs (`ARM`, `RESID`) and two runtime arguments (`emask`, `nrot`, compiled with
`do_not_specialize`) after K25's own parameters; and arms without the select tree sum its 16 codebook values before the
loop and add the sum after it, so they load the codebook as K25 does. Each arm stores its accumulator through K25's own
masked store. `launch` refuses unless `P157_ABLATION=1`; nothing in the library can reach it.

| arm | what is removed | what stays | cubin |
|---|---|---|---|
| **A0** | nothing: K25 as served (`gemm_nf4_grouped_smallm`, today's plan) | everything | K25 |
| **T2** | A0 again: the self-pair | everything | K25 |
| **A0p** | nothing: the placebo (only the explicit `tl.range`) | everything; SASS identical to K25's | K25's twin |
| **A1** | the select tree: the nibble is converted to fp32 | loads, nibble extraction, absmax, staging, MMA | own |
| **A2b** | the MMA and its operand staging: per-thread fp32 folds | loads, decode, absmax | own |
| **A12** | both A1's and A2b's | loads, nibble extraction, absmax | own |
| **A3** | everything but the loads: per-thread XOR/add folds of every loaded word | the loads, A0's bytes | **loads** |
| **AL** | A3 on expert 0's slab only (`eid & 0`): ~2 % of A0's DRAM bytes | A3's loads, from L2 after the first touch | **loads** |
| **AC** | DRAM bytes: every program reads expert `eid & 0` (one slab a call) | the whole K loop | **ladder** |
| **AC4** | DRAM bytes: `eid & 3`, at most 4 slabs a call | the whole K loop | **ladder** |
| **AC16** | DRAM bytes: `eid & 15`, at most 16 slabs a call | the whole K loop | **ladder** |
| **ACX** | nothing: `eid & 127` is the identity on [0, 128), A0's own bytes | the whole K loop | **ladder** |
| **ACX0** | ACX at `nrot` = 0: the ladder's cubin unrotated, bit-identical to K25 | everything | **ladder** |
| **AE** | everything: each program returns at once | the grid and its launches | own |

**One cubin per difference.** Under `RESID` the expert id is `eid & emask`, and a program reads its weights from n-tile
`(pid_n + g * nrot) mod NT` (its output store is unrotated). `emask` and `nrot` are data, so the ladder (AC, AC4, AC16,
ACX, ACX0) is ONE cubin and A3 with AL is another: every difference r is built from (below) is taken inside one cubin.
The record VOIDs if either set is not one cubin (instruction-stream hashes, read from the cubins that ran).

**Why `nrot`.** With `eid & 0`, the ~57 programs dispatched together (consecutive `g`, one `pid_n`) read the same 32 rows
at once. On the lab A2000 that arm timed 7 % SLOWER than reading every expert (hot lines, not bytes). With `nrot` = 1
concurrent programs read different n-tiles even when they share an expert; the bytes each rung moves are unchanged.
ACX0 prices the rotation itself (below).

**The dose.** Each layer's slabs are evicted by the other 47 layers' traffic before the layer comes round again (about
7.3 GB of weights a replay against a 96 MiB L2), so every rung pays a compulsory first touch per call, and its DRAM
bytes per graph are the sum over its calls of (distinct `eid & emask` experts) x (slab bytes): 1,769,472 B a gate_up
slab, 884,736 B a down slab. On S0's routing a call touches a median of 57 experts (41-82), so the rungs move about
**2 %, 7 %, 28 % and 100 %** of A0's bytes. `f(rung)` is that per graph, byte-weighted, exact for each rep's own routing.

### A2a: NOT BUILT

A2a would replace each `mma.sync` in A0's PTX by an integer fold and reload it through `TRITON_KERNEL_OVERRIDE`, pricing
the MMA alone. It was registered as best effort, and it is not built. A0's PTX (sm_120, gate_up and down alike) issues 8
`mma.sync.m16n8k8` tf32 an iteration, chained on one accumulator, with **48 distinct operand registers** (32 A operands,
each from its own scalar `cvt.f32.bf16`; 16 B operands from `ld.shared.v4.b32`). A replacement that keeps every operand
alive, and so the staging, needs at least ceil(48 / 2) = **24 three-input `lop3`** an iteration against 8 HMMA. It trades
tensor-pipe time for extra ALU issues, so its delta mixes MMA removal with added issue pressure and cannot be read as
either. A2b stands alone, and its ceiling is read as **the MMA and its operand staging**, never staging alone.

### The static counts (sm_120, triton 3.4.0, ptxas 12.8.93)

Compiled at the served specialization ($0, CPU), counted with the vendored `sass_count.py` (gnf4 `bench/sass_count`):

| arm | lane instructions / weight (gate_up, down) | registers (gate_up, down) |
|---|---|---|
| A0 = K25 | 24.12, 24.12 | 80, 80 |
| A0p | 24.12, 24.12 (SASS identical to K25's) | 80, 80 |
| A1 | 10.69, 10.69 | 67, 67 |
| A2b | 23.81, 23.88 | 74, 72 |
| A12 | 8.50, 8.50 | 64, 64 |
| A3 | 5.44 (down: ptxas unrolls the 12-iteration light loop, read by totals) | 59, 64 |
| the ladder (AC ... ACX0) | 24.12, 24.12 (its loop's opcodes are A0p's) | 80, 80 |
| AE | no loop | 4, 4 |

The box recounts every arm from the cubin it ran (step 2 below); those counts are the ones gated.

## The instrument (`p157_bench.py`)

The box and the call form are P155's: one RTX 5090 (sm_120, power limit >= 575 W), torch 2.8.0+cu128, triton 3.4.0,
grouped-nf4-gemm **v0.50.0** at 41ef086e (gnf4#572; its three kernel files byte-identical to P156's pin, 68bbefc4,
and hash-pinned). Full-size random NF4 stores for all 48 MoE layers
(16.3 GB). A0 calls K25 exactly as e4b's served route does on the NF4 store at 16 rows (gate_up gathers through the lean
tile table's `order` with `gather_div=8`; down stores into the caller's order with `scatter=order`), over
`int4_b32.build_group_tiles_fused(ids, 128, 16, lean=True, sorted_ids=True)`; the arms use the same call form through
`p157_arms.launch`. Routing: P153's router replay (`rr.json.pt`, sha256 `3c59e72f...`), S0's 16 rows (`3020dbce...`),
with P155's checks before any data.

1. **Pins.** Each arm's compiled kernel per projection, read from the driver: registers, local memory, shared memory,
   the preferred carveout. Where `cuOccupancyMaxActiveBlocksPerMultiprocessor` gives an arm more CTAs per SM than A0
   (6 on sm_120 at 80 registers and 14,592 B), dynamic shared memory is added in 128-byte steps until it gives A0's, and
   the same query verifies it. One cubin takes one pad. The pad rides the compiled kernel's launch metadata, which a
   CUDA graph captures as the kernel node's `sharedMemBytes`.
2. **Static.** The SASS of every kernel that ran, from its own cubin: the inner loop (by the MMA, or by the load stream
   with the unroll checked by bytes against A0's stream), loads per weight, the whole function's loads, the loads outside
   the loop, and an instruction-stream hash (addresses stripped).
3. **<a id="the-reps"></a>The reps.** On each of **70 router-replay steps (58-127)**, one paired rep: every arm's graph
   (both GEMMs of all 48 layers), plus the spacer **S** (K25's calls with their own output buffers, never timed). Capture
   and replay follow row r mod 14 of a **Williams design** over the 14 arms: 70 reps are 5 whole squares, so each arm
   takes each position 5 times and every ordered pair of arms is adjacent 5 times. **10 timed replays** per arm,
   interleaved arm by arm, **each right after an untimed replay of S**. While each capture is open, its kernel nodes are
   read back through the driver (`cuStreamGetCaptureInfo`, `cuGraphKernelNodeGetParams`). NVML samples, tagged with the
   arm replaying: SM clock, memory clock, averaged and instantaneous power, the enforced power limit, and clock-event
   reasons. After the replays, T2's, A0p's and ACX0's outputs are compared with A0's, bit for bit.
   - **What each defence removes.** S is every timed replay's IMMEDIATE predecessor, so no arm inherits a light or heavy
     neighbour's power and clock headroom, and none finds its own outputs warm. The Williams order balances each arm's
     position in the rep (the clock and thermal drift across a rep) and its second-order neighbour (the arm before S).
4. **The bytes axis (runner, before the reduce).** Nsight Compute's `dram__bytes_read.sum` for A0p, A3, AL and the four
   ladder rungs: one process per arm, layers 0-3 of step 58 at the arm's pins, the caches flushed before each kernel
   (ncu's default, as the other layers do in a replay). A probe first; a refused counter (ERR_NVGPUCTRPERM, common in
   rented containers) leaves the router-replay ESTIMATE.

## The reading (`p157_reduce.py`, fixed before data)

Every quantity is a paired per-rep difference or ratio (each rep's median of its 10 replays), read as the **median over
the 70 reps with the larger-of SE** (P155's rule: the larger of sqrt(pi/2) sd / sqrt(n) and a seeded 2,000-resample
bootstrap of the median), and an interval of the median +- 2 SE. Never a difference of medians.

### VOID (the instrument)

The router replay or S0's slice is not the pinned bytes; not an RTX 5090 at cc 12.0; not 48 layers and steps 58-127;
not the 14 registered arms; not 10 replays each; no spacer S; a rep's order is not the registered Williams row; any rep
lacks good replays of every arm or the slab counts of every rung; the identity rung's slabs are not the routing's
distinct experts; T2's, A0p's or ACX0's outputs differ from A0's on ANY rep; A0p's SASS is not K25's; the ladder or
the loads arms are not one cubin; the context was poisoned; or the self-pair (T2 / A0) median lies outside 0.99-1.01.

### Per-arm gates and labels

**Gates** (a failing arm's numbers are reported, never read):
- **load parity:** A1, A2b, A12 and A3 issue A0's global loads per weight (count and width) in the K loop. Where ptxas
  unrolls a light loop away (A3 on down), its whole-function loads must equal A0's loop loads x the trip count (12) plus
  A0's loads outside the loop;
- **A2b strictly cheaper** than A0 per weight;
- **math parity:** each ladder rung's loop opcodes are A0p's; AL's are A3's.

**Labels** (beside the arm's numbers, never a reason to drop them): verified CTAs per SM differ from A0's (A2b, if its
registers keep it below A0's, cannot be padded up); the carveout differs from A0's; the graph nodes' `sharedMemBytes` are
not the pinned launch's; median SM clock off A0p's by more than 2 %, either way ("clock-confounded").

### Ceilings and additivity (reported)

For A1, A2b, A12 and A3: the paired `A0p - arm` (ms) and its share of `A0p - AE`, each with what it prices (the decode;
the MMA and its staging; both; everything but the loads), its gate and its labels. A12's additivity: the paired
`A1 + A2b - A0p - A12`, ADDITIVE if its interval contains 0, else SUB_ADDITIVE (the stages overlap) or SUPER_ADDITIVE.

### r_dram, from the arms

Per rep, each inside one cubin: **full = ACX - AE**, **mem_dram = A3 - AL** (the loads-only kernel's DRAM stream, its
L2 leg cancelled), **onchip = AC - AE** (the whole K loop at ~2 % of A0's bytes), and

    r = (full - max(mem_dram, onchip)) / min(mem_dram, onchip)

r near 0: the stream hides under the on-chip work (throughput-bound); near 1: they serialize (latency-bound). Its
interval's **band**, in this order: entirely below 0 or above 1, **NOT_ADDITIVE**; entirely <= 0.25,
**THROUGHPUT_BOUND**; entirely >= 0.75, **LATENCY_BOUND**; inside [0.25, 0.75], **MIXED**; otherwise **UNRESOLVED**.
- **Small denominator:** if min(mem_dram, onchip)'s median is within 2 self-pair SE (of the per-rep T2 - A0, in ms) of
  0, or any rep's is not positive, r is UNRESOLVED.
- **Called** only if A3, AL, AC and ACX pass their gates and carry no label, and the ladder does not disagree (below).
  Otherwise r is reported with what blocked it.
- **The dose offset, stated:** onchip is AC's time at ~2 % of A0's bytes and mem_dram prices the bytes above AL's (the
  same ~2 %), so a truly serial kernel reads r = 1 and an overlapped one below its knee reads r = 0, both exactly. Above
  the knee, an overlapped kernel reads r about f_AC x D / onchip high (D, the full stream's cost), a few hundredths at
  f_AC ~ 0.02: it cannot cross the 0.25 edge unless D / onchip > ~12.

### Copy checks

Reported beside r: A0p / A0 (the explicit `tl.range`), ACX0 / A0p (the ladder cubin's layout alone), ACX / ACX0 (the
rotation at full dose) and ACX / A0p (the total copy gap). **COPY_TOL = 1 %:** inside it, r reads as K25's; outside it,
r carries the label **"about the RESID copy (copy gap X %)"**. If ACX / ACX0's interval excludes 1, r carries "the
rotation itself moves ACX by X %". Neither VOIDs.

### The ladder: check of r's band

The rungs are a dose series inside the ladder cubin. The model, in one parameter:

    T_k - AE = max(onchip, g_k x mem_dram) + r x min(onchip, g_k x mem_dram),  g_k = (f_k - f_AC) / (1 - f_AC)

g_k measures each rung's dose FROM AC's (AC anchors onchip, so it is never a test point: its r is undefined; ACX at
g = 1 is r_dram with ACX in A0p's place). Each **test rung (AC4, AC16)** implies its own r_k per rep, from that rep's own
arms and rung time, read as the median with the larger-of SE.
- **Uninformative:** an r_k interval wider than 0.5 (low doses, where the min term is tiny).
- **Disagree:** an informative r_k interval lies ENTIRELY in a band the arm call is not consistent with. NOT_ADDITIVE
  splits by side for this check: entirely above 1 is consistent with LATENCY_BOUND (serial or worse), entirely below 0
  with THROUGHPUT_BOUND. A perfectly serial kernel sits exactly on the r = 1 edge, so without the split noise alone
  would read it "disagree". UNRESOLVED never disagrees.
- "Arms and ladder disagree" means r is reported, not called.

### The bytes axis

f is the router-replay **ESTIMATE**, labelled so, unless Nsight measured the rungs. Then, at the profiled step: each
rung's weight bytes (its DRAM reads less its cubin's non-weight reads, the full-dose arm's reads less that arm's weight
estimate) must match its estimate within 5 %, AL's likewise, A3 within 2 % of A0p's reads, and AC and AL at most 5 % of
A0p's. If all hold, each rung's per-rep f is scaled by its measured ratio (**NCU-SCALED**). If any fails, the ladder is
reported and not used.

### Reported shape (never a call)

The paired increments between adjacent rungs, in ms per GB of DRAM, and the hinge `T = max(a, b f)` fitted per rep over
the four rungs (its knee and slope, with SEs).

## Predictions (graded in the read, never ruling)

| | prediction | basis | HELD means | MISSED means |
|---|---|---|---|---|
| **P1** | A0 per graph in **11.5-13.5 ms** | P155 read 12.075 ms; P156's box, 12.87 | the harness reproduces K25's time | a different box or harness effect; the record says which |
| **P2** | A0p / A0 within **0.995-1.005**, and the copy gap within 1 % | A0p's SASS is K25's; the ladder's loop is A0p's | r is K25's | r reads "about the RESID copy" |
| **P3** | r_dram's band **LATENCY_BOUND or MIXED** | full (12.1 ms) exceeds the sum of the floors (10.1-11.0 ms) | the stream does not hide: overlap (deeper pipelining, more resident warps) is a lever beside shrinking either stage | THROUGHPUT_BOUND: overlap is already good, and only the larger stage matters; NOT_ADDITIVE: the stages interfere |
| **P4** | the decode ceiling `A0p - A1` in **1.9-6.0 ms** | removing the tree saves at least what K39's cheaper decode did (12.87 -> 10.96 ms in P156) | the decode is a large on-chip lever | below: the ablation or P156 is off; above: the tree dominates on-chip time |
| **P5** | the MMA-and-staging ceiling `A0p - A2b` in **0-1.5 ms** | A2b removes only 0.3 of 24.1 instructions per weight | the tensor pipe is not where the time is | the MMA chain (8 dependent HMMA an iteration) costs latency the count misses |
| **P6** | mem_dram in **2.0-5.5 ms** | P154's byte floor 4.94 ms for all the bytes; A3 streams below peak; AL's L2 leg cancels | the DRAM stream is the size the floor says | the loads-only kernel is far from the floor (or far from AL) |
| **P7** | onchip in **5.0-10.0 ms** | the issue estimate 5.2-6.1 ms, plus latency at 6 CTAs/SM | on-chip work is the larger stage | the instruction count misprices it |
| **P8** | the ladder AGREES with r's band | one model, one cubin | the call stands on two independent reads | r is reported, not called |
| **P9** | no rung is faster than the rung below it: AC <= AC4 <= AC16 <= ACX within 2 SE | the dose only adds bytes | the dose series is clean | AC carries a concentration penalty: onchip is an upper bound, and r's call says so |

## What follows

- **LATENCY_BOUND or MIXED** (P3 HELD): the next decode lane targets overlap: warp-specialized or deeper-staged loads,
  and K39's smaller register file (64 against 80) for more resident warps, beside the decode's instruction count.
- **THROUGHPUT_BOUND:** only the larger stage matters. If onchip > mem_dram, the decode (A1's ceiling) is the lever; if
  not, the bytes are.
- **NOT_ADDITIVE above 1:** the stages slow each other (shared load/store or issue contention); a counter read on a host
  that allows Nsight comes before any kernel change.
- Every ceiling is an upper bound on what optimizing its stage could give at today's plan, labelled with its confounds.

## Before the box: the $0 A2000 shakedown

Done on 2026-10-20 (00:50-00:52Z), **before this registration and before any rental**, under an exclusive
`gpu:a2000` claim: `p157_bench.py --shakedown --layers 4 --reps 14` (one whole Williams square) on the lab's A2000
(sm_86, 70 W), torch 2.8.0+cu128, triton 3.4.0, grouped-nf4-gemm's `kernel/` at 68bbefc4, which v0.50.0 carries
byte for byte (its three pinned files
byte-identical: `ee46969a`, `773eff5d`, `a7090915`), and this registration's arms, driver probes and `sass_count.py`
byte for byte, with its bench and reducer (the log records every sha256). Two changes came after the run, both inside
the self-tests and neither in code the bench or the reducer runs on a record: the bench's source checks no longer match
their own literals (and gained the arms' call form), and the reducer's small-denominator case now keeps every rep's
mem_dram positive (a mutant had passed through the non-positive path). Its record is
[`shakedown/shake_p157.json`](shakedown/shake_p157.json) (sha256 `977a1b18...`, pinned in `tests/test_p157.py`).
- **The harness, end to end:** both self-tests (the bench's 7 cases; the reducer's 36 then, 37 now), the bench (rc 0 in
  74 s), and
  the reducer, which reads **SHAKEDOWN** with every structural check passing and every arm's static gate passing on
  sm_86. A shakedown is never graded.
- **Bits:** T2, A0p and ACX0 kept K25's bits on all 14 reps, both projections, all 4 layers.
- **Pins:** the driver gave A0 6 CTAs per SM (71 registers, 14,592 B on sm_86). A1, A12, A3, AL and AE ran 7-12
  unpinned and were padded to exactly 6 (7,296 B; AE 13,696 B), confirmed by the same query; every captured kernel node
  read back the pinned `sharedMemBytes`. A2b runs 5 on gate_up (89 registers on sm_86), below A0's, so it is labelled;
  the box reads sm_120's count live (74 registers there, 6 CTAs/SM by the static count).
- **Clocks:** NVML read every arm's SM clock, power (averaged and instantaneous) and the enforced limit (70 W); the
  A2000 sat at its cap (event reason 0x4) throughout.
- **Copy checks:** A0p / A0 1.0034, ACX0 / A0p 1.0004, ACX / ACX0 0.9997, ACX / A0p 1.0017; self-pair 0.9999 (SE
  0.0077 on this card).
- **What the dev runs before it found, and this registration fixes** (each a $0 A2000 run on 2026-10-20, none timed for
  a band): a cyclic order fixed each arm's predecessor (self-pair 1.029; S and the Williams order fix it); A0's own graph
  as the spacer left A0's outputs warm (1.008; S has its own buffers); ACX / A0p was 0.947 on the first build (every
  difference r uses is now taken inside one cubin); one slab a call timed 7 % slower than all of them (hot lines;
  `nrot` fixes it); arms without the tree dropped its 16 codebook loads (A3-down failed the totals gate by exactly 16
  `LDG.E`; restored).
- **What it cannot show:** the A2000's L2 is 3 MB, smaller than AC4's 7 MB a call, so it cannot read the ladder's
  physics (its AC still timed slowest; P9 grades that on the 5090). Its times are the A2000's and seed nothing.

## Budget and sequencing

- **One RTX 5090**, its own box:
  - hosts with a recorded power cap of at least 575 W (`vast_min_power_limit_w: 575`, `vast_power_limit_unknown:
    exclude`), read within the last 3 days;
  - declared at $1.05/h, **guard 1.5 h**, at most 10 GB at <= $0.011/GB. Worst case 1.5 x $1.05 + 10 x $0.011 =
    **$1.69**, under the standing rule;
  - disk 60 GB (the box's floor is 20 GB free). Nothing is fetched from a hub.
- **The fit, e4b#1845's rule, with at least 25 % headroom:**
  - **Install, self-tests and premise:** P155's, `INSTALL_TO_GATE_S` = 600 s.
  - **The bench:** `NEED_BENCH` = **1100 s**, from two measurements, the larger taken, x1.5:
    - this lane's shakedown: 1.17 ms of host time per launch beyond the GPU's (its rep walls less its replays), scaled
      to 48 layers (15 graphs a rep, 2 warm calls and a capture, 96 launches each), plus every graph at P155's
      12.075 ms (10 timed and 10 spacer replays an arm, 3 runs a graph), 70 reps, and its setup scaled to 48 layers:
      about 684 s;
    - P155's box: 0.287 s a 48-layer graph with its 10 replays (110.2 s for 384 graphs), 15 graphs a rep plus the
      spacer replays, 70 reps: about 434 s.

    So about 684 s; x1.5 = 1,026 s, rounded up. Its alarm can run to 3600 s if the deadline allows.
  - **Nsight:** `NEED_NCU` = 900 s: an install if absent, a probe, seven 4-layer processes.
  - **The reserve:** `RESERVE` = 600 s.
  - **The runner's check, before the install:** `FIT` = 600 + 1100 + 900 + 600 = 3200 s, and `FIT_NEED` = 3200 x 1.25 =
    4000 s. Anything short of that is a refusal, rc 41.
  - **The static fit,** (boot + fit) x 1.25 <= guard: boot 300 s + 3200 s = 3500 s, and 3500 x 1.25 = 4375 s <= 5400 s
    (1.5 h). `tests/test_p157.py` asserts it. The 1.5 h guard (not 1.25 h, which would also fit) leaves room for a slow
    boot without a fit refusal.
- **Run ids** `p157-5090-1` to `-3`. Two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane. P155's strike
  rules hold: a pre-flight host stall, a power-floor refusal (rc 17) or a fit refusal (rc 41) is not a strike (two
  consecutive fit refusals count as one); a time-left skip of the bench (STOP-2, rc 40) is.
- **Sequencing:** this registration merges after loggetta's review, b1-decode's on the counter and driver parts, and
  the maintainer's pass; then the shared pre-flight and `--validate` on the exact manifest. The vendored `sass_count.py`
  is v0.50.0's `bench/sass_count/sass_count.py` (gnf4#571's, sha256 `d479fad5...`), byte for byte.

## Receipts

Fetched to the run directory's `p157/` and published under `bench/p157/receipts/<run>/` by the 2026-10-19 citation
rules: `bench_p157.json` (every replay time, the pins, the static records, the clocks), `verdict_p157.json`,
`ncu_p157.json` and the per-arm CSVs when the counters were available, `summary.txt`, `forensics.txt`, `versions.txt`,
`gpu_phases.csv` and the logs, with `SHA256SUMS`. The box writes no instance id and no host name. `RESULTS-p157.md` is
written from those files.

## What this lane cannot say

- **Nothing about the served step.** It times the two K25 GEMMs alone, at today's plan.
- **Nothing at another toolchain, plan or decode.** Its arms are K25's at triton 3.4.0; K39's decode, other plans and a
  triton bump are their own reads.
- **Nothing exact about a stage the arms cannot isolate:** A2b prices the MMA with its staging, and every ceiling is an
  upper bound with its labels.
- **Not the served NF4 picks:** bf16 teacher-forced picks over P153's prompts (P155's caveat). Weight bytes are random.
- **Nothing about other cards, families or row counts:** one RTX 5090, Qwen3-30B-A3B's shapes, 16 rows.

**Review:** loggetta (independent), b1-decode on the counter and driver parts (`sass_count`, the pins, the readback),
then the maintainer's pass.
