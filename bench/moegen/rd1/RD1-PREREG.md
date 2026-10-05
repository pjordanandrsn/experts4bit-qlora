# RD1: grouped-nf4-gemm's frozen-expert GEMM routes, per call, at MoE training shapes, on one RTX 5090

*Registered 2026-10-05, before any box. This file is `bench/moegen/rd1/RD1-PREREG.md`. Work item experts4bit-qlora#1049
(the moe-generalize campaign's class-B follow-up).
- The box side is [`rd1_run.sh`](rd1_run.sh), started by `bench/tc1/tc1_drive.sh` as its `TC1_RUNNER`.
- The probe is [`rd_probe.py`](rd_probe.py). The table, the correctness gate and the bar are [`rd_table.py`](rd_table.py).
- The A2000 correctness rehearsal is [`a2000/`](a2000/). It is correctness only: no timing is read from it.*

## Question

On sm_120, which training route for the frozen-expert GEMMs beats the shipped fused kernels per call by enough to be worth
a full-step A/B, on the many-group families? Three are candidates:
- **decoded:** grouped-nf4-gemm's `nf4_route.dequant_groups` (one launch for every present expert), then ONE Triton grouped
  bf16 GEMM launch;
- **v3:** the fused kernels' own bf16 MMA;
- **dense:** the per-expert dequant + `torch.mm` route that `auto` takes for at most 16 present groups.

The answer may be none of them.

## Evidence before the box (the code and RTX 5090 readings only)

- **TC1 amendment 22 (one RTX 5090, full training step).**
  - `dense`/fused is 0.651 on Mixtral-8x7B (8 groups) and 2.947 on Qwen3-30B-A3B (up to 128 groups).
  - The per-expert loop adds about 295,000 launches a step, which is why `auto` caps `dense` at 16 present groups.
- **The fused kernels' arithmetic.** grouped-nf4-gemm main @`9622144c`:
  - the default forward (variant 1) and `dgrad_4bit_grouped` feed fp32 operands into `tl.dot`, i.e. TF32 tensor cores, and
    decode the NF4 weight inside the GEMM loop, once per M-tile;
  - variant 3 (opt-in) is the forward's bf16 MMA with the same in-loop decode, and the dgrad has no bf16 variant;
  - the sm_120 census (`bench/sm120-census/` in grouped-nf4-gemm) swept v3 only at serving cells (B=16), so v3 is unmeasured
    at training rows per expert on sm_120.
- **The decoded route's shape.** It is two launches per call, whatever the number of groups. It decodes each present expert
  once (not once per M-tile), and runs a bf16 MMA. Its cost is a decode transient of `groups × N × K × 2` bytes, which the
  cap bounds.
- **torch.** No release through 2.14.1 has a single-launch grouped bf16 GEMM on sm_120.
  - CUTLASS `_grouped_mm` is gated to sm_90 / sm_100, and the cuBLASLt grouped path to compute-capability majors 9–11.
  - Every other card takes `_grouped_mm_fallback`, a per-group `mm_out` loop behind a device-to-host sync.
- **No timing from the QNAP A2000 is evidence here** (standing policy, 2026-07-27: a shared production box, a correctness
  testbed only).

## The instrument

**Probe** (`rd_probe.py`): per projection call, forward and dgrad. The arms:
- `v1`: the fused kernels as shipped.
- `v3`: `gemm_4bit_grouped(..., prefill_variant=3)`, plus a probe-local copy of `_dgrad_nf4_grouped` that casts `g` and the
  decoded `w` to bf16 before `tl.dot` (one change; best of three tile configs).
- `dense`: `dense_forward` / `dense_dgrad`.
- `decoded`: uncapped, best of three tile configs.
- `decoded_cap`: the decode transient capped at **`GNF4_DECODED_MAX_BYTES` = 256 MiB**, as chunks of groups with one dequant
  and one GEMM launch each; the chunk plan is built once per call shape.

Per arm it records:
- device ms (profiler kernel time) and event ms (CUDA events over 20 back-to-back calls);
- peak MiB above the inputs (`max_memory_allocated`);
- relative error against `v1`;
- `rel_err32`, the relative error against an **fp32 reference**: grouped-nf4-gemm's `dequant_ref` in fp32 times the
  activations in fp32, per group. `v1` is measured against that reference too, because it runs TF32, so every arm sits in one
  frame.

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

**Correctness gate** (added before the box, on the gnf4 maintainer's review of this registration).
- An arm passes a call when its `rel_err32` is **at most 2 times dense's** `rel_err32` on the same call. It passes a cell when
  all four of its calls pass.
- A cell counts toward DECODED (its `decoded_cap`) or V3 (its `v3`) only if that arm passes the gate there. A cell that fails
  is a FAIL row that never counts, and every failing call is listed in the table.
- A cell whose error fields are missing is unread, and counts for nothing.
- `v3` enters the min that `decoded_cap` must beat only where it passes the gate; `v1` and `dense` always do.
- Why: the decision's first consequence is an opt-in route shipped into grouped-nf4-gemm *before* the full-step A/B, so this
  probe is the only numerics check the route gets before it lands. A fast arm that computes wrong values (a chunk-plan
  offset, a stride slip in the shared forward/dgrad kernel, a mis-cast in the probe-local bf16 dgrad) must not count.

**The bars.**
- **DECODED:** decoded_cap passes the gate, and its event time is **at most 0.85** of min(v1, v3, dense) at **both** seq 512
  and 2048, for at least 3 of the 7 many-group families: `olmoe`, `lfm2`, `ernie`, `graniteh`, `qwen3`, `nemotron`, `qwen36`.
  The decode transient is at most the cap by construction, and the measured peak is reported.
- **V3:** v3 passes the gate, and its event time is at most 0.85 of v1's at both seqs, for at least 3 of the 7.

**Decision** (only an RTX 5090 receipt decides; any other card's table says so and decides nothing):
- If DECODED holds, an opt-in `decoded` route goes into grouped-nf4-gemm behind `route_for`, with `GNF4_DECODED_MAX_BYTES`
  and `auto` untouched. Its licence is then a TC1 full-step A/B on the 5090 with matched-set EQUIVALENT: the route
  accumulates in a different order, as the sm_90 `grouped_mm` route did.
- Otherwise, if V3 holds, a v3 training default is its own TC1 A/B.
- Otherwise neither: the result is recorded and no route is built.

## Predictions (hypotheses from the code and amendment 22; each is a row whatever it reads)

- **P0:** every arm passes the correctness gate on every call. A failure is a defect in that arm and is named, whatever the
  bar reads.
- **P1:** DECODED holds, at least on `graniteh`, `qwen3` and `qwen36`. These are the families where `dense` is launch-bound
  (amendment 22's mechanism), and where v1 pays an in-loop decode per M-tile and TF32.
- **V3: no prediction.** v3 is a full arm, and nothing has measured it at training rows per expert on sm_120.
- **P3:** decoded_cap's event / device time is at most 1.5 on every many-group skewed cell, because the route is two launches
  per call (or two per chunk). If not, the route is host-bound on this card, and the full-step A/B must read wall time, not
  device time.
- **P4:** dense / v1 event time is above 1.2 at seq 512, skewed, on `graniteh`, `qwen3` and `qwen36`: amendment 22's
  launch-bound mechanism at the call level.
- **P5:** on `mixtral`, dense ≤ decoded_cap on event time at both seqs, so the at-most-16-group route stays `dense`. The
  reasons: amendment 22 read `dense` at 0.651 there, and eight groups are not launch-bound. The 256 MiB cap also leaves
  decoded_cap with nearly as many launches: one gate_up expert per chunk (235 MB decoded), and two down-projection experts
  per chunk (117 MB each).

## Amendment 1 (2026-10-05, after three anchor refusals, before the next box): a load-gated anchor

**What happened.** The first three draws were refused by the train anchor, each on `launch.self_pair` alone (FLOPs and H2D in
band). That is the anchor working, and no reading was taken:

| run | machine | launch.self_pair | cost |
|---|---|---|---|
| `rd1-5090-1` | 145701 (EPYC 7B13) | 1.0421 > 1.03 | $0.024 |
| `rd1-5090-2` | 145701 again (the ranking re-picked it) | refused | $0.026 |
| `rd1-5090-4` | 36544 (Xeon Platinum 8347C) | 1.0602 > 1.03 | $0.018 |

`rd1-5090-3` was refused at $0 before any rental: adertha's anchor-exclusion class accepts only P41-layout receipts
(adertha-agents#166).

**Why load.** TC1 amendment 33 measured, on these multi-tenant 5090 hosts, the host's load average following the unstable
draws, with stable readings at a median load1 of about 5 or below. Launch timing is the most host-sensitive thing the anchor
checks.

**The change** (`rd1_run.sh`; the bar, the correctness gate and the probe are unchanged):
- **The anchor runs only at host load1 at or under 5.0.** The runner waits for that for up to 600 s before each attempt, and
  if the wait times out, the anchor runs anyway and the summary says so.
- **There are at most 3 anchor attempts**, and the last attempt stands. A refused attempt's files are kept as
  `anchor.attempt<k>.json` and `logs/anchor_gate.attempt<k>.log`, so every attempt is a row.
- **A sampler records `/proc/loadavg` every 5 s** (`logs/loadavg.log`). The probe's own window is summarised into its receipt
  as `host_load1_probe` (median, max, samples, gate).
- **No decision from a loaded probe.** `rd_table.py` takes no decision unless that summary exists with a median load1 at or
  under 5.0. A loaded probe is NOT A DECISION and needs another draw: re-running the anchor until it passes is a selection,
  so the probe's own load is what licenses the reading.
- **Budget.** The guard goes from 0.75 h to 1.0 h for up to three waits, at about $0.85. That is not over one hour, so no
  proving run is required.

## The A2000 correctness rehearsal (`a2000/`; correctness only, never speed)

Run before the box, on the owned RTX A2000 (sm_86), with the same probe. It checks:
- that every arm compiles and launches;
- the fp32-reference gate on every arm and call;
- the cap's chunk plan, against the uncapped route;
- peak bytes, which come from the allocator and are deterministic, kept as a sanity number;
- the runner end to end in a container (`RD1_REHEARSAL=1`).

Committed receipts carry **no timing field**: `rd_table.py --strip-timing` removes them, and `rd_table.py --gate-only` reads
what remains. The result is recorded in [`a2000/RESULTS-rd-a2000-correctness.txt`](a2000/RESULTS-rd-a2000-correctness.txt).

**Result (2026-10-05):** every arm passed the gate on every call of all 32 cells: 8 families × 2 seqs × 2 routers × 4 calls.
- Error against the fp32 reference: `v1` (TF32) 0.0017, `dense` 0.0023–0.0036. `v3`, `decoded` and `decoded_cap` were at
  most 1.00× dense's on every call.
- No arm or tile config was skipped, and the capped route's chunk plans matched the uncapped route's outputs.
- Peak above the inputs: `decoded` 180–2016 MiB uncapped, `decoded_cap` 180–448 MiB at the 256 MiB cap; `v1` 9–224 MiB.
- The runner rehearsed end to end in a container: install at the pinned commit, tripwire, the anchor (refused on the
  A2000 and continued, rehearsal only), the probe and the table.

## Budget, staging

- **Budget.** One RTX 5090, with no checkpoint fetch: install, the anchor, then the probe. Guard 0.75 h, about $0.64. The
  guard is under one hour, so no proving run is required.
- **Controller.** `bench/tc1/tc1_drive.sh` with `TC1_BOX=A`, `TC1_RUNNER=rd1_run.sh`, `GNF4_SHA` as above, and
  `TC1_EXTRA_STAGE` = `bench/moegen/rd1/rd1_run.sh bench/moegen/rd1/rd_probe.py bench/moegen/rd1/rd_table.py
  bench/train-anchor/train_anchor.py bench/train-anchor/train_anchor_gate.py`.
- **`RD1_REHEARSAL=1`** exists only for the A2000 container rehearsal of the runner. It records an anchor refusal and
  continues, and it shrinks the grid to one cell. `tc1_drive.sh` forwards no `RD1_` knob, so a rented box cannot set it.
