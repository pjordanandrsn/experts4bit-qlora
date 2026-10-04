# FP1 — `estimate_qlora_footprint` against a measured Qwen3-30B-A3B run

Work item: experts4bit-qlora#1064 (the owner-authorization permalink). Lane registered 2026-10-04 before any box, on one RTX 5090 (Vast, verified-secure) through `bench/tc1/tc1_drive.sh`
(`TC1_RUNNER=fp1_run.sh`).

**Question.** How close is `estimate_qlora_footprint`'s device total to what the allocator holds on a 30B-class MoE?
What does the process hold beyond the allocator's peak? The answer is broken down three ways:
- the allocator's reserved-but-unallocated blocks;
- the CUDA context and library workspaces, measured as the driver's process peak minus the reserved peak;
- host memory: RssAnon + RssShmem, where pinned homes land.

Every reading so far comes from OLMoE-sized runs on an RTX A2000. Carrying those overheads to 30B is extrapolation.

## Shape (fixed)

| | |
|---|---|
| models | `allenai/OLMoE-1B-7B-0924` @ `6d84c48581ece794365f2b8e9cfb043c68ade9c5` (anchor); `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (TC1's pin) |
| arms | `olmoe_device` (resident); `qwen3_device` (resident); `qwen3_host` (experts in pinned host memory, one layer staged) |
| setup | `QLoRASetup()` defaults except residency: NF4 experts, r 8 / α 16, bf16 adapters, attention + expert LoRA, `expert_kernel="grouped_nf4"`, dgrad |
| workload | alpaca packed into fixed blocks, seq 512 × micro-batch 2 (T = 1,024), 12 steps, AdamW 2e-4, clip 1.0, seed 0 |
| software | experts4bit-qlora at the launch head; grouped-nf4-gemm 0.38.0 (`5a887c48acc90207fdd30f2e9b23d21d62102b14`); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2 |

## Readings (per arm, from `receipts/<arm>.json`)

- **R1 (allocator).** `measured.device_peak_bytes` against the estimate's device total. Expected: within ±15% on
  every arm.
- **R2 (slack).** `measured.reserve_slack_fraction` (reserved / allocated peak − 1). The resident vs offload contrast
  is reported. Expected on an RTX A2000 for OLMoE: 0.22 resident, 0.39 offload. No expectation is registered at 30B;
  that is the question.
- **R3 (context).** `measured.cuda_context_bytes`.
- **R4 (host).** `measured.host_required_peak_bytes` against the estimate's host total: pinned homes under offload,
  each rounded to a power of two.
- **Integrity**, every arm:
  - finite losses;
  - frozen expert bytes unchanged (sha256 of the first and last stacks, the host homes under offload);
  - LoRA-B norm moved.

  An arm failing integrity is reported as ALARM and its memory readings are not used.

## Outcomes

- **Success:** three arm receipts with integrity clean; R1–R4 read and reported as numbers. A miss on R1 is a
  result, not a failure: the estimator's error is what is being measured.
- **Failure:**
  - install or tripwire (rc 9);
  - fetch (10);
  - fewer than three receipts (11).

  An OOM arm is recorded as OOM with the peaks it reached.

## Cost

Estimate: about 1 h on one RTX 5090 at the policy rate. That covers the ~71 GB checkpoint download, three loads and
36 steps. Ceiling: wallclock 2 h.
