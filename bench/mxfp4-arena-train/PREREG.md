# PREREG — training a 284B model from its own released MXFP4 bytes, on one 24 GB card

**Pre-data.** Stamped before the arena is baked or any step is timed.

## What is under test

`enable_nvme_train_residency` read only NF4 quantize-at-bake arenas, so a
natively-MXFP4 checkpoint could **serve** from its released bytes but had to be
re-quantized to train from disk. The overlay under test
(`arena_offload_view`, branch `feat/mxfp4-arena-train`, one file, 70 lines)
resolves MXFP4 arenas through `fuse_gate_up_segments` into the same
four-segment staging the NF4 path uses.

**Claim:** DeepSeek-V4-Flash (43 layers x 256 experts, 284B) runs a QLoRA
training step on a single 24 GB card, with its frozen experts served from an
on-disk arena of the checkpoint's **own MXFP4 bytes** — no re-quantization.

**This code has never run against a real MXFP4 arena.** The CPU resolver check
covers NF4 passthrough and the reject path only.

## Arms, both on the same pod and card

| arm | what it is | expected |
|---|---|---|
| **STOCK** | `transformers` + `bitsandbytes` 4-bit + `peft`, same model, same card | **OOM** — 284B in 4-bit is ~142 GB against 24 GB of VRAM |
| **ARENA** | published e4b/gnf4 wheels + this overlay, `arena_train=True` + `enable_nvme_train_residency` | trains |

The STOCK arm is **run, not asserted**: its traceback is captured verbatim into
the receipt. An enablement claim whose control was never executed is an opinion.

## Frozen predictions

- **P1 — STOCK fails.** If STOCK *succeeds*, the enablement claim is refuted and
  that is the headline, not a footnote.
- **P2 — VRAM.** Peak < 24 GB, and predicted **4–8 GB** for the staged stacks
  from the module's own arithmetic (it keeps the full `[E, ...]` shape, quoted
  at ~15.7 GB for K3's 896 experts/layer, so 256/layer scales to ~4.5 GB) plus
  activations. Outside that band, the arithmetic in the header is wrong and gets
  corrected in public.
- **P3 — bytes unchanged.** Every frozen expert tensor hashes identical before
  and after training, and the arena rows match the checkpoint's released bytes.
  A single mismatch STOPS the run.
- **P4 — it learns.** Loss over the registered steps is finite and moves. No bar
  on how much: this is a feasibility run, not a quality result.
- **NO PREDICTION on s/step.** Reading routed rows off NVMe every step is a new
  cost curve; whatever it is, it is reported.

## Gates

- **G1** `enable_nvme_train_residency` returns > 0 modules patched. Zero means
  the overlay did not take and every later number is meaningless.
- **G2** The overlay is verified *in the imported module* before the run, and
  `overlay_provenance.json` records stock and patched sha256.
- **G3** Gradient checkpointing on — the tier requires it (evict fires on
  forward return; without recompute, backward reads 0-element placeholders).

## Stop rules

Whatever it shows is written up in either direction, including "the enabler
refused", "it OOMed", or "the bytes moved". No retry-until-green: a failure is
the result, and the code is only changed with a new stamped prereg. Hard bill
cap $35; teardown is evidence-first.

## Not blind

I read the tier's source and its serving figure (V4-Flash generating at 8.74 GiB
peak from a 147 GB arena) before writing this. P2's band is derived from that
code's own comment, not guessed.
