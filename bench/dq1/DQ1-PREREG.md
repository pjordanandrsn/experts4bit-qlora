# DQ1 — the dense low-bit headroom census: could any 4-bit primitive speed up, or enlarge, dense QLoRA on an RTX 5090?

Registered 2026-10-04, before any box. Lane name claimed by `prereg/dq1`. Work item: the experts4bit-qlora issue linked
in this lane's PR, which carries the owner's directive and the research note
([NOTE-dense-qlora-primitive.md](NOTE-dense-qlora-primitive.md)). One RTX 5090 (Vast, verified-secure) through
`bench/tc1/tc1_drive.sh` with `TC1_RUNNER=dq1_run.sh`.

## Question

Is there a dense low-bit execution primitive worth building for dense QLoRA? The census measures the two quantities that
bound any answer, at Qwen3-32B's linear shapes.

- **Speed headroom, H(M).** The share of bitsandbytes' frozen-base linear time that a perfect low-bit kernel could
  remove at training row counts.
- **Streaming feasibility, Rmin(M).** For each phase of a step (the forward; the recompute + dgrad), that phase's
  compute per decoder layer over one layer's NF4 transfer time from pinned host memory, measured under GEMM load.

It also asks whether grouped-nf4-gemm at one group (G=1) is a usable dense route. It reads the route `auto` really takes
(dense) and the packed kernel (fused).

## Subject (fixed)

| | |
|---|---|
| shapes | Qwen3-32B (`Qwen/Qwen3-32B` config: hidden 5120, intermediate 25600, 64 q / 8 kv heads × 128): q [8192, 5120], k/v [1024, 5120] ×2, o [5120, 8192], gate/up [25600, 5120] ×2, down [5120, 25600] |
| weights | random N(0, 1/K), quantized once by bnb `quantize_4bit` (nf4, blocksize 64, double-quant: `Linear4bit`'s defaults and the QLoRA recipe; HF's `BitsAndBytesConfig` defaults to no double-quant). gnf4's tensors are repacked FROM those (`repack_from_bnb`), so every arm multiplies by the same decoded matrix. No checkpoint. |
| rows M | 512, 1024, 2048, 4096, 8192 (tokens per micro-batch) |
| activations | bf16, `torch.randn` seeded per M; the upstream gradient likewise |
| software | torch 2.8 / CUDA 12.8 (image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`), triton 3.4, bitsandbytes 0.50.2, grouped-nf4-gemm v0.39.0 (`a5edec8789735bff1c0da4708ae5fc93260a1410`) |

## Arms (`bench/dq1/dq1_census.py`)

| arm | forward | dgrad |
|---|---|---|
| `bf16` | `F.linear` on the decoded weight, resident in bf16 | autograd |
| `bnb` | `bnb.matmul_4bit` (bnb's own dispatch) | autograd (`MatMul4Bit.backward`) |
| `dq` | `dequantize_4bit` + `F.linear` | `dequantize_4bit` + `g @ W` |
| `gnf4a` | `gemm_4bit_grouped_train`, one group, `GNF4_TRAIN_GEMM=auto` | autograd |
| `gnf4f` | the same, `GNF4_TRAIN_GEMM=fused` | autograd |

Every (shape, M) cell runs the arms in the palindrome `bf16 bnb dq gnf4a gnf4f gnf4f gnf4a dq bnb bf16`.
- Before the cell starts, 1.5 s of wall-clock GPU warm-up.
- Per position: 5 CUDA-event draws of about 60 ms each (up to 5,000 calls a draw), separately for the forward and the dgrad.
- The forward is timed with autograd on, as in training.

**Engagement is read, not inferred.** Each gnf4 arm records:
- the route the call took (`ctx.route` on `FusedGroupedNf4`, read as `y.grad_fn.route`);
- the delta of `nf4_qlora.DGRAD_STATS`, which dgrad served the call;
- the delta of `nf4_route.ROUTE_STATS`.

**Parity:** each arm's forward and dgrad against an fp32 product with the same decoded weight.

**Also per cell:**
- bnb's own forward decision at that shape (`custom` or `dequant`, from its CUDA heuristic);
- each decoder alone (bnb `dequantize_4bit`, gnf4 `dequant_groups`);
- a PEFT-shaped LoRA delta: r 16, α 32, no dropout, A and B trainable bf16;
- the allocator peak.

**Streaming probe**, after the cells, at M = 1024, 2048 and 4096: one layer's NF4 bytes (packed plus nested absmax
state), copied pinned → device on a side stream, 5 draws. Each draw takes two fully-loaded readings:
- **copy under load.** That layer's bnb forward GEMMs are enqueued first on the compute stream, sized to at least twice
  the copy window, so every copy runs while GEMMs run.
- **GEMMs under load.** The copies are enqueued first, sized to at least twice the GEMM window.

Each reading records on the device timeline whether the loaded stream started after, and ended before, the other one.

## The rule (`bench/dq1/dq1_reduce.py`: 44 synthetic-receipt self-test cases; `tests/test_dq1_lane.py` applies 23 rule mutants and requires the self-test to catch each)

Definitions:
- **arm time.** The mean over the two palindrome positions of each position's median draw. Position 2 / position 1 is
  the arm's self-pair.
- **C.** 2 × fwd + dgrad: the forward, the checkpoint recompute and the frozen-weight input gradient.
- **LC(arm, M).** C summed over one layer's seven linears; LCf and LCd are its forward and dgrad parts.

Readings:
- `H = 1 − LC(bf16)/LC(bnb)`
- `G1 = LC(bnb)/LC(gnf4a)`
- `GF = LC(bnb)/LC(gnf4f)`
- `L = LoRA(2 × fwd + bwd)/LC(bnb)`
- Streaming. One layer's transfer is `X = layer bytes / loaded H2D GB/s`. Each phase must hide one transfer behind
  that phase's own compute:
  - `R_fwd = LCf(bnb) / X` for the forward;
  - `R_bwd = (LCf + LCd)(bnb) / X` for the recompute + dgrad, which keep the layer resident;
  - `Rmin = min(R_fwd, R_bwd)`.

  The pooled `R = LC / 2X` is reported but not graded: it lends the backward's slack to the forward.

The verdict is the first of these that applies:

1. **VOID** when any of the following holds:
   - the device is not exactly `NVIDIA GeForce RTX 5090`;
   - the receipt is a rehearsal;
   - the receipt is incomplete: not the registered shapes and rows, any registered cell missing, the streaming probe not
     at exactly 1024 / 2048 / 4096, or no `finished_at` (a receipt cut short by the alarm is flushed per cell);
   - an arm is missing or has no timings;
   - `bf16` or `bnb` fails parity (> 1e-2 relative, or non-finite), or could not run, anywhere.
2. **NOISY** when more than 10% of all self-pairs fall outside [0.95, 1.05]. No verdicts are given.
3. **READ.** Each axis is graded separately:

| axis | grades |
|---|---|
| speed | **S_ALIVE** if H(2048) ≥ 0.15; **S_DEAD** if H(2048) < 0.10 and H(4096) < 0.07; otherwise **S_MARGINAL** |
| G=1 | Graded at M 2048 and 4096 only. **G1_FUNCTION_FAIL** on a parity failure there; **G1_ENGAGEMENT_FAIL** if the route taken is not `dense` or `DGRAD_STATS.dense` / `ROUTE_STATS.dense_fwd` / `dense_dgrad` did not move; **G1_COULD_NOT_RUN** on an error there; **G1_LOSS** if G1 < 0.95 at 2048 or 4096; **G1_WIN** if G1 ≥ 1.05 at both; otherwise **G1_PARITY** |
| fused | **GF_FUNCTION_FAIL** on any parity failure; **GF_ENGAGEMENT_FAIL** if the route taken is not `fused` or the dgrad kernel did not serve; **GF_PARTIAL** if a cell could not run (reported, not graded); **GF_LOSS** if GF < 0.95 at every M; otherwise **GF_NOT_LOSS** |
| LoRA | **L_HIGH** if L(2048) ≥ 0.15, else **L_LOW** |
| stream | **C_UNCOVERED** if any loaded reading was not covered end to end (an instrument result, not a link result); **C_ALIVE** if Rmin(2048) ≥ 1.25, with GEMM slowdown under DMA ≤ 1.05 and loaded/alone H2D ≥ 0.8; **C_DEAD** if Rmin(4096) < 1.0; otherwise **C_MARGINAL** |

## Predictions (stamped before the data)

- **H.** H(512) between 0.15 and 0.40; H(2048) between 0.04 and 0.15; H(4096) between 0.02 and 0.10. Speed verdict:
  S_DEAD or S_MARGINAL. Reasoning: at training M bnb is dequant + cuBLAS in both directions (its fused forward is a
  small-M path; the census records bnb's own decision per cell), and dequant moves ~2.5 B per parameter against 2M FLOPs
  per parameter.
- **G1.** Between 0.97 and 1.08 at M ≥ 2048 (G1_PARITY). Same structure as bnb; gnf4's decoder is somewhat faster.
- **GF.** Below 0.80 at every M (GF_LOSS). This reproduces the 2026-08-15 A6000 result on sm_120.
- **L(2048).** Between 0.03 and 0.12 (L_LOW).
- **Streaming.** Loaded H2D ≥ 40 GB/s on a PCIe 5.0 x16 host, about 22 GB/s on 4.0 x16. Rmin(2048) (= R_fwd) ≥ 1.8 on
  5.0, ≥ 1.0 on 4.0. GEMM slowdown under DMA ≤ 1.03. The link is recorded (`pcie.link.gen.max` and
  `pcie.link.width.current`; the current gen reads downclocked at idle) and the prediction graded against it.

## Consequence (registered)

| outcome | action |
|---|---|
| S_DEAD (or S_MARGINAL with H(2048) < 0.12) and G1_PARITY | The dense speed-kernel line closes. G=1 stays an internal gnf4 route. Nothing is built for speed and no repository is proposed. |
| S_ALIVE | Next is a full-step profile of HF + PEFT + bnb on Qwen3-32B: the linears' share of the step, before any kernel. |
| C_ALIVE | Next is a streamed frozen-weight prototype under autograd, on a branch (no new repo), plus a matched-work lane: Qwen3-32B resident vs streamed, and a model larger than VRAM against FSDP-QLoRA with CPU offload. |
| C_DEAD | The streaming line closes on this link. |
| G1_LOSS | The dense route needs a guard at G=1. That is the gnf4 maintainer's issue, not a new primitive. |
| VOID / NOISY / C_UNCOVERED | One re-run on another host; a second VOID/NOISY is reported as an instrument finding and stops the lane. |

## Validity, engagement, exit codes

- The box (`dq1_run.sh`) installs grouped-nf4-gemm at the pin and bitsandbytes 0.50.2. Its tripwire asserts:
  - the installed commit and version;
  - CUDA;
  - the premise that `train_gemm_route(cuda, n_groups=1) == "dense"` on this card.
- The box also runs the reducer's self-test, then the census (alarm 35 min), then the reducer.
- The launch passes `GNF4_SHA=a5edec8789735bff1c0da4708ae5fc93260a1410` so the controller's log names the gnf4 the box installs.
- Exit codes:
  - 9: install, tripwire or self-test;
  - 11: the census;
  - 12: the reducer.
- An arm error at a cell is recorded and does not stop the census.

## Cost

About 25 min of box time on one RTX 5090 at the policy rate ($0.85/h). There is no download beyond pip. Guard 0.75 h, so
no proving rental is needed. Expected actual is under $0.50; the run sits inside the standing no-ask tier ($15 per run).

## Rehearsal

Before the rental, the census (on reduced rows) and the reducer run on the RTX A2000 through the local pool
(`tools/pool run --gpu cuda`). Its receipt is marked `rehearsal` and can never be graded. It checks only that every arm
runs, engages and passes parity on sm_86.

## Amendment 1 (registered 2026-10-05, after run 1 read NOISY, before run 2)

**Why.** Run 1 (`dq1-5090-1`, [RESULTS-dq1.md](RESULTS-dq1.md)) read NOISY: 37 of 250 self-pairs fell out of band.
The dominant group is diagnosed. The first timed arm of each cell (bf16, position 1) ran inside a boost transient that the
1.5 s warm-up left behind: it read ~10% fast against both its own position 2 and an independent estimate of the same
GEMM. Re-running an unchanged instrument would re-read the same transient.

**The change: the instrument only.**
- `busy(1.5 s)` becomes `warm_until_steady`. It runs sustained 4096³ bf16 matmuls in ~0.25 s blocks until at least 4 s
  have passed and the last four blocks' rates agree within 1%, capped at 30 s.
- It runs before every cell and before the streaming probe.
- Each cell records the warm-up it got (seconds, first and steady TF/s, whether it reached steady) and one block's rate at
  the cell's end (`end_tflops`).
- The receipt carries `"amendment": 1`.

**Unchanged:**
- the subject, arms, rows, palindrome, draws and parity;
- the rule (`dq1_reduce.py` is byte-identical to run 1's);
- the predictions and consequences.

The warm-up and end rates are recorded for the read; they are not graded.

**Run 2** is the registered "one re-run on another host". `avoid_vast_machine_receipts` carries run 1's receipt so the
launcher does not re-buy its machine. A second VOID or NOISY stops the lane as an instrument finding, as registered.
