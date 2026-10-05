# RD1: grouped-nf4-gemm's frozen-expert GEMM routes, per call, at MoE training shapes, on one RTX 5090

*Registered 2026-10-05, before any box. This file is `bench/moegen/rd1/RD1-PREREG.md`. Work item experts4bit-qlora#1049
(the moe-generalize campaign's class-B follow-up).
- The box side is [`rd1_run.sh`](rd1_run.sh), started by `bench/tc1/tc1_drive.sh` as its `TC1_RUNNER`.
- The probe is [`rd_probe.py`](rd_probe.py). The table and the bar are [`rd_table.py`](rd_table.py).
- The $0 A2000 filter that motivated it is [`a2000/`](a2000/).*

## Question

On sm_120, which training route for the frozen-expert GEMMs beats the shipped fused kernels per call by enough to be worth
a full-step A/B, on the many-group families? Three are candidates:
- **decoded:** grouped-nf4-gemm's `nf4_route.dequant_groups` (one launch for every present expert), then ONE Triton grouped
  bf16 GEMM launch;
- **v3:** the fused kernels' own bf16 MMA;
- **dense:** the per-expert dequant + `torch.mm` route that `auto` takes for at most 16 present groups.

The answer may be none of them.

## Evidence before the box

- **TC1 amendment 22 (one RTX 5090, full training step).**
  - `dense`/fused is 0.651 on Mixtral-8x7B (8 groups) and 2.947 on Qwen3-30B-A3B (up to 128 groups).
  - The per-expert loop adds about 295,000 launches a step, which is why `auto` caps `dense` at 16 present groups.
- **The fused kernels' arithmetic.** grouped-nf4-gemm main @`9622144c`:
  - the default forward (variant 1) and `dgrad_4bit_grouped` feed fp32 operands into `tl.dot`, i.e. TF32 tensor cores, and
    decode inside the GEMM loop;
  - variant 3 (opt-in) is the forward's bf16 MMA, and the dgrad has no bf16 variant.
- **torch.** No release through 2.14.1 has a single-launch grouped bf16 GEMM on sm_120.
  - CUTLASS `_grouped_mm` is gated to sm_90 / sm_100, and the cuBLASLt grouped path to compute-capability majors 9–11.
  - Every other card takes `_grouped_mm_fallback`, a per-group `mm_out` loop behind a device-to-host sync.
- **The A2000 filter** ([`a2000/RESULTS-rd-a2000.txt`](a2000/RESULTS-rd-a2000.txt), this probe, 2026-10-05; a filter, not a
  licence):
  - On the skewed draw, decoded_cap / best(v1, v3, dense) is at most 0.85 at both seqs on Granite-H, Qwen3 and Qwen3.6, and
    0.70–1.01 elsewhere among the many-group families.
  - v3 / v1 is 1.02–1.22 everywhere: the bf16 MMA *inside* the fused loop is slower, so the decode in the loop, not TF32,
    is what the decoded route removes.
  - `dense` / v1 on the skewed draw at seq 512 is 2.09 (Granite-H), 1.25 (Qwen3) and 2.83 (Qwen3.6): launch-bound even there.
  - The 256 MiB cap costs 2–5 chunks and about ±15 µs per extra chunk.

## The instrument

**Probe** (`rd_probe.py`): per projection call, forward and dgrad. The arms:
- `v1`: the fused kernels as shipped.
- `v3`: `gemm_4bit_grouped(..., prefill_variant=3)`, plus a probe-local copy of `_dgrad_nf4_grouped` that casts `g` and the
  decoded `w` to bf16 before `tl.dot` (one change; best of three tile configs).
- `dense`: `dense_forward` / `dense_dgrad`.
- `decoded`: uncapped, best of three tile configs.
- `decoded_cap`: the decode transient capped at **`GNF4_DECODED_MAX_BYTES` = 256 MiB**, as chunks of groups with one dequant
  and one GEMM launch each; the chunk plan is built once per call shape.

Per arm it records device ms (profiler kernel time), event ms (CUDA events over 20 back-to-back calls), peak MiB above the
inputs (`max_memory_allocated`), and relative error against `v1`.

**Grid.** Eight families' expert shapes, from each checkpoint's `config.json`:

| family | experts | top-k | hidden | expert intermediate | MLP |
|---|---|---|---|---|---|
| `olmoe` | 64 | 8 | 2048 | 1024 | gated |
| `lfm2` | 32 | 4 | 2048 | 1792 | gated |
| `ernie` | 64 | 6 | 2560 | 1536 | gated |
| `graniteh` | 64 | 6 | 1536 | 512 | gated |
| `qwen3` | 128 | 8 | 2048 | 768 | gated |
| `nemotron` | 128 | 6 | 2688 | 1856 | non-gated |
| `qwen36` | 256 | 8 | 2048 | 512 | gated |
| `mixtral` | 8 | 2 | 4096 | 14336 | gated |

- Sequence lengths 512 and 2048, at batch 1.
- Two router draws, seed 0:
  - **uniform**: k distinct experts per token;
  - **skew**: grouped-nf4-gemm's `bench/host-reuse/moe_host.py` router, i.e. Gaussian logits + `linspace(1.5, -1.5, E)`,
    then top-k. It gives hot and cold experts, and many groups with few rows.
- Weights are random NF4 stacks: packed bytes uniform, absmax uniform in [0.01, 0.06]. The kernels' time does not depend on
  weight values.
- **Box class:** the train anchor (`bench/train-anchor/`), strict as in tp1. A refused box ends the lane (exit 12).
- **Code:** grouped-nf4-gemm pinned at `9622144c734395de4d40332f06dea92135d858de` (the box tripwire refuses any other), torch
  2.8.0 + the image's Triton.

## The bar (scored by `rd_table.py`, read on the skewed draw only)

A layer's cost is the sum of its four expert calls per arm: the up or gate_up and the down projection, forward and dgrad.

- **DECODED:** decoded_cap's event time is **at most 0.85** of min(v1, v3, dense) at **both** seq 512 and 2048, for **at least
  3 of the 7** many-group families: `olmoe`, `lfm2`, `ernie`, `graniteh`, `qwen3`, `nemotron`, `qwen36`. The decode transient
  is at most the cap by construction, and the measured peak is reported.
- **V3:** v3's event time is at most 0.85 of v1's at both seqs, for at least 3 of the 7.

**Decision** (only an RTX 5090 receipt decides; any other card's table says so and decides nothing):
- If DECODED holds, an opt-in `decoded` route goes into grouped-nf4-gemm behind `route_for`, with `GNF4_DECODED_MAX_BYTES`
  and `auto` untouched. Its licence is then a TC1 full-step A/B on the 5090 with matched-set EQUIVALENT: the route
  accumulates in a different order, as the sm_90 `grouped_mm` route did.
- Otherwise, if V3 holds, a v3 training default is its own TC1 A/B.
- Otherwise neither: the result is recorded and no route is built.

## Predictions (each is a row whatever it reads)

- **P1:** DECODED holds, at least on `graniteh`, `qwen3` and `qwen36`, where `dense` is launch-bound.
- **P2:** V3 does not hold: v3 / v1 is above 0.85 on at least 5 of the 7.
- **P3:** decoded_cap's event / device time is at most 1.5 on every many-group skewed cell. If not, the route is host-bound on
  this card, and the full-step A/B must read wall time, not device time.
- **P4:** dense / v1 event time is above 1.2 at seq 512, skewed, on `graniteh`, `qwen3` and `qwen36`: amendment 22's
  launch-bound mechanism at the call level.
- **P5:** on `mixtral`, dense ≤ decoded_cap on event time at both seqs, so the at-most-16-group route stays `dense`.

## Budget, staging and the rehearsal

- **Budget.** One RTX 5090, with no checkpoint fetch: install, the anchor, then the probe (about 18 min of probe on the A2000).
  Guard 0.75 h, about $0.64. The guard is under one hour, so no proving run is required.
- **Controller.** `bench/tc1/tc1_drive.sh` with `TC1_BOX=A`, `TC1_RUNNER=rd1_run.sh`, `GNF4_SHA` as above, and
  `TC1_EXTRA_STAGE` = `bench/moegen/rd1/rd1_run.sh bench/moegen/rd1/rd_probe.py bench/moegen/rd1/rd_table.py
  bench/train-anchor/train_anchor.py bench/train-anchor/train_anchor_gate.py`.
- **`RD1_REHEARSAL=1`** exists only for the $0 A2000 container rehearsal of the runner. It records an anchor refusal and
  continues, and it shrinks the grid to one cell. `tc1_drive.sh` forwards no `RD1_` knob, so a rented box cannot set it.
