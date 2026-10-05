# MG1 amendment 2: the ladder ran out of memory at the licensed configuration, so P2 for Qwen3.6 is still unread

*MG1 amendment 2 ([`../../MG1-PREREG.md`](../../MG1-PREREG.md#amendment-2-2026-10-04-after-the-reading-and-before-its-own-box-p2-for-qwen36),
registered in #1075 before its box). Work item experts4bit-qlora#1049. Read 2026-10-04 from the files in this directory and in
[`../mg1-a2-5090-1/`](../mg1-a2-5090-1/).*

**The answer.** Amendment 2 did not read P2.
- The ladder's `fused` rung ran out of GPU memory in AdamW's first step on Qwen3.6-35B-A3B, at the licensed arms'
  configuration (r 8, fp32 adapters, bf16 attention, no profiler).
- The backward had already run, so the dgrad counter existed, but only inside the process that died. No ladder receipt was
  written.
- By the amendment's own rule an OOM is a row and the status does not change: `qwen3_5_moe.fast_train` stays `experimental`.
- Amendment 3 reads P2 on tp1's arm driver itself, the code path the PASS was read on.

## The draws

| run | box | outcome | cost |
|---|---|---|---|
| `mg1-a2-5090-1` | RTX 5090, machine 152169 (Ryzen 9 7900) | **refused at pre-flight before any work.** HF CDN 2.4 MB/s against the 20 MB/s floor, and the generic endpoints read 2.0 MB/s, so the host's network was slow, not only its route to Hugging Face. Nothing was staged ([launch.json](../mg1-a2-5090-1/launch.json)) | $0.069 |
| `mg1-a2-5090-2` | RTX 5090, machine 151350 (Core Ultra 9 285K) | **the reading: ladder OOM.** The install, the tripwire, the recurrent kernels and the 67 GB fetch were all OK, and the runner printed `AMENDMENT 2 SHAPE (registered)`. The ladder then exited 1 with 0 rungs summarised ([summary](summary.txt), [ladder log](logs/ladder_qwen3_5.log), [launch.json](launch.json)) | $0.288 |

Machine 151350 is the box that read MG1's PASS (`mg1-5090-2`), where tp1's fused arm trained Qwen3.6 resident.

## What ran before the OOM

The ladder log shows the engagement up to the optimizer:
- 40/40 MoE layers quantized, with the fused training path on 40 `ExpertsLoRA` modules (`dgrad kernel backward`);
- fused rotary, as on the reading (`[e4b.rope] fused rotary in 1 attention module(s)`);
- 101 frozen RMSNorms fused with the centred fp32 formula, 0 left on the composite.

The traceback is in `torch.optim.AdamW.step()` → `_multi_tensor_adam` → `torch._foreach_sqrt`:
`CUDA out of memory. Tried to allocate 16.00 MiB … 19.69 MiB is free … this process has 31.30 GiB memory in use. Of the
allocated memory 30.28 GiB is allocated`. That is a step after the first backward, so the dgrad kernel had served that
backward, but the count is not a reading until a receipt holds it.

## Why the ladder and not the arm

These are the same family, the same configuration and the same host. tp1's fused arm (`mg1-5090-2`) peaked at 27.73 GB
allocated (25.83 GiB). The ladder reached 30.28 GiB allocated and failed a 16 MiB request, 4.45 GiB above the arm.

The two drivers differ. The difference this read can name: `ladder.py` keeps `init = [p.detach().clone() for p in train]`, a
GPU copy of every trainable adapter, so that each rung starts from the same values. That is 463,093,760 fp32 parameters here
(the arm's own `trainable_params`), or 1.73 GiB, which tp1's driver never allocates. The remaining 2.72 GiB is not attributed;
no measurement in this read separates it.

## Decision (the amendment's rule)

- An OOM is a row, and the status is unchanged. `qwen3_5_moe.fast_train` stays `experimental`, and its reason now names this
  read.
- The ladder is not re-rolled. Amendment 3 (registered with this read, before its box) reads P2 by running tp1's fused arm,
  the driver file unchanged, under `bench/moegen/p2_hook.py`, which writes `DGRAD_STATS` at the driver's exit.

Spend: $0.357 for amendment 2 ($0.069 refused, $0.288 read). The MG1 lane in all is $2.24, under the $15 no-ask tier.
