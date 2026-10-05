# DQ2 — can a dense layer's frozen NF4 weights stream from host memory behind its own QLoRA training compute, on a PCIe 5.0 x16 RTX 5090?

Registered 2026-10-05, before any box. Lane name claimed by `prereg/dq2`. Work item: #1083. The owner's instruction is
"register DQ2 and launch it on a PCIe 5.0 box" (Claude Code desktop chat, 2026-10-05; relayed on #1083). One RTX 5090
on Vast verified-secure, PCIe gen 5 x16, through `bench/tc1/tc1_drive.sh` with `TC1_RUNNER=dq2_run.sh`.

## Why this lane

DQ1 ([RESULTS-dq1.md](../dq1/RESULTS-dq1.md), [SUMMARY-dq1.md](../dq1/SUMMARY-dq1.md)) closed the dense W4A16 speed
line and left one axis open: **capacity**. That means keeping a dense model's frozen NF4 weights in pinned host memory
and copying each layer in while the previous one computes, so a model larger than the card can train.

DQ1 read it as C_MARGINAL, on two limits:
- **The link.** Both DQ1 hosts were PCIe 4.0 x16 (27.9 GB/s under load).
- **The compute.** R counted only the seven linears, not the layer's attention, norms, RoPE or LoRA.

DQ2 removes both limits: the link the question needs, and the real layer as HF + PEFT + bitsandbytes QLoRA runs it.

## Subject (fixed)

| | |
|---|---|
| layer | transformers `Qwen3DecoderLayer` built from `Qwen/Qwen3-32B`'s config (hidden 5120, intermediate 25600, 64 q / 8 kv heads × 128, RoPE θ 1e6), SDPA attention, causal, random weights (no checkpoint) |
| 4-bit | the seven projections as bitsandbytes `Linear4bit`: nf4, blocksize 64, double-quant, bf16 compute |
| LoRA | PEFT `inject_adapter_in_model`, `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0)`, all seven projections, PEFT's default adapter dtype; everything else frozen |
| step | non-reentrant gradient checkpointing (`torch.utils.checkpoint`, `use_reentrant=False`), micro-batch 1 × M tokens, M ∈ {512, 1024, 2048, 4096, 8192} |
| host | RTX 5090, PCIe gen max 5, width 16 (the box refuses anything else, rc 13, before installing) |
| software | image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (torch 2.8, triton 3.4), bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2, grouped-nf4-gemm v0.39.0 `a5edec87` (imported only for DQ1's census module's warm-up and forensics) |

## Measurement (`bench/dq2/dq2_layer.py`)

Each row runs in this order: steady warm-up (DQ1 Amendment 1), then the layer at position 1, then the streaming probe,
then the layer at position 2. That gives one self-pair per phase.

- **T_fwd.** The checkpointed forward of the layer. This is what runs while the next layer's weights would be copied.
- **T_bwd.** Its backward: recompute plus backward through LoRA and the frozen paths. This is what runs during the
  backward-phase copy.
- Each is 5 draws, each draw the median of reps sized to ~300 ms, timed with CUDA events.
- **Streaming probe.** The layer's frozen bytes (packed nibbles plus nested quant state, as `Linear4bit` holds them)
  are copied pinned → device on a side stream. There are two fully-loaded readings per draw, as in DQ1:
  - copy under the layer's forward;
  - the forward under copies.

  Device-timeline coverage is recorded per draw.
- **Integrity**, every row and position:
  - finite output, input gradient and LoRA gradients;
  - all seven LoRA-B gradients nonzero (PEFT zero-initialises B, so A's gradients are zero at step 0 by construction);
  - the frozen 4-bit storage hashed before and after the run.
- **Engagement.** Seven `Linear4bit` modules, wrapped by PEFT's bnb LoRA layer, with 14 LoRA tensors.

## The rule (`bench/dq2/dq2_reduce.py`: 25 self-test cases; `tests/test_dq2_lane.py` applies 17 rule mutants and requires each to be caught)

Readings per row:
- **T_fwd, T_bwd.** The mean of the two positions' medians.
- **X.** Layer frozen bytes / copy-under-load GB/s.
- **R_fwd = T_fwd / X; R_bwd = T_bwd / X; Rmin = their minimum.** Each phase must hide one transfer behind its own
  compute.
- **slowdown** = forward under DMA / forward alone.
- **copy_ratio** = loaded / alone H2D.

The verdict is the first of these that applies:

1. **VOID** when any of the following holds:
   - the device is not exactly `NVIDIA GeForce RTX 5090`;
   - the link is not gen max 5 / width max 16;
   - H2D alone at M=2048 is below 35 GB/s (not a gen 5 x16 link in practice);
   - the receipt is a rehearsal;
   - the rows are not the registered five, a row errored or is missing, or there is no `finished_at`;
   - any integrity or engagement check fails;
   - a warm-up did not reach steady.
2. **NOISY** when more than 1 of the 10 self-pairs (forward and backward × 5 rows) falls outside [0.95, 1.05].
3. **READ.** The stream verdict at M = 2048:
   - **C_UNCOVERED** if any loaded reading was not covered end to end;
   - **C_ALIVE** if Rmin ≥ 1.25, slowdown ≤ 1.05 and copy_ratio ≥ 0.8;
   - **C_DEAD** if Rmin < 1.0;
   - otherwise **C_MARGINAL**.

Also reported, not graded: the smallest registered row at which Rmin ≥ 1.0 and ≥ 1.25 (break-even).

## Predictions (stamped before the data; every basis is rented-5090 data)

The bases:
- **Linear times.** DQ1 run 2 measured the layer's linears with bnb: forward 9.66 ms and dgrad ≈ 9.9 ms at 2048, and
  LoRA (2 × fwd + bwd) ≈ 3.0 ms.
- **Layer bytes.** 251.5 MB.
- **Link bandwidth.** Vast's own `pcie_bw` for gen 5 x16 RTX 5090 offers is 41.6–54.3 GB/s (read-only search,
  2026-10-05).

| reading | band |
|---|---|
| T_fwd(2048) | [11, 15] ms (linears + LoRA + attention + norms/RoPE/activation) |
| T_bwd(2048) | [22, 34] ms (recompute + dgrad + LoRA + attention backward) |
| H2D under load | [40, 56] GB/s; copy_ratio ≥ 0.9; forward slowdown under DMA ≤ 1.05 |
| X | [4.5, 6.3] ms |
| **R_fwd(2048) = Rmin(2048)** | **[1.8, 3.2] → C_ALIVE** (the forward binds; R_bwd ≈ 2× it) |
| R_fwd(1024) | [0.9, 1.6] |
| break-even (Rmin ≥ 1.25) | the 1024 or 2048 row |

## Consequence (registered)

| outcome | action |
|---|---|
| C_ALIVE | The capacity axis is earned at the prototype level. Licensed: a streamed frozen-weight prototype under autograd, on a branch (no new repository). It is e4b's `dense_offload` made to overlap under grad, the stream and the dense layer's schedule kept tensor-level. Then a matched-work lane: one dense model resident vs streamed (same step, same quality), and one larger than VRAM against FSDP-QLoRA with CPU offload. |
| C_MARGINAL | Report the break-even row; no prototype. Streaming is a planner rule for M at or above it, not a backend. |
| C_DEAD | The streaming line closes for single-GPU dense QLoRA on this card class. |
| C_UNCOVERED, VOID, NOISY | One re-run on another gen 5 host; a second stops the lane as an instrument finding. |

## Exit codes and cost

- **rc 13:** host refused (not an RTX 5090 on gen 5 x16), before any install.
- **rc 9:** install, tripwire or self-test.
- **rc 11:** the census.
- **rc 12:** the reducer.

About 15 min of box time, mostly install. The registered rows need under 5 min of GPU. Guard 0.75 h at the policy rate
($0.85/h), so no proving rental is needed; estimated actual under $0.30. The run sits in the standing no-ask tier
($15 a run).

## Launch requirement

The Vast offer search must be restricted to PCIe gen ≥ 5 and x16 lanes. The launcher gains that as an opt-in manifest
field (`vast_pcie`, adertha-agents). Without it, DQ1's runs drew gen 4 hosts twice; with it, the box's rc 13 stays a
backstop rather than the expected path.

## Rehearsal

Before the rental, `dq2_layer.py` (reduced rows) and the reducer run on the RTX A2000 through the local pool. That is
**correctness only**: build, engagement, integrity and probe coverage. No A2000 timing informs any band, per the testbed
policy.
