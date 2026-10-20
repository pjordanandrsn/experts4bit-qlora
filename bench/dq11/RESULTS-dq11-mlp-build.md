# DQ11 MLP-adapter BUILD: the fused MLP adapters are correct and 2.8% faster, so NO_GAIN

**Measured-private, 2026-10-20.** The runs and their receipts exist in the private receipt store. A reader of this
repository cannot independently check them. No claim is registered by this result: a NO_GAIN moves no shipped default.

**NO_GAIN: H = 0.0279 < 0.05.**
- With `E4B_DENSE_MLP_LORA_FUSE=1`, the engine (`experts4bit_qlora/engines/dense_lora_fuse.py`) fuses the DQ11 L
  fixture's 96 MLP LoRA adapters. That makes a training update 2.8% faster on one RTX 5090 at the science lock, with
  every checked tensor judged correct.
- The registered bar is 5%, so this lever closes with its number, as the [registration](DQ11-MLP-BUILD-PREREG.md) says
  a NO_GAIN does.
- The engine stays opt-in.

## What ran

- **The registration:** [the PREREG](DQ11-MLP-BUILD-PREREG.md) with Amendments 1–5a. The Phase 2 kit is
  [Amendment 5](DQ11-MLP-BUILD-AMENDMENT-5.md) and [5a](DQ11-MLP-BUILD-AMENDMENT-5a.md).
- **Phase 1:** `dq11-mlp-phase1-a2000-2`, on the RTX A2000 at the rehearsal lock, read `PHASE1_PASS`. It was re-derived
  and independently reviewed.
- **The shakedown:** `dq11-mlp-phase2-shakedown-a2000-1`, at $0 on the A2000. It fixed the family list, the draws and
  the (b) part's time.
- **Phase 2:** `dq11-mlp-phase2-1`.
  - One RTX 5090 at the science lock: torch 2.12.1+cu130, PEFT 0.21.2, transformers 5.5.0, bitsandbytes 0.50.2.
  - e4b at `80bc64e329bd8d75def837f428b65798f4839e9c`, the kit's merge.
  - One process per part: the (b) read, then F0 (knob off) and F1 (fused) ABBA twice at 40 updates each, then the two
    drift arms and the reduce.
  - It cost $0.385.

## The read

T_arm is the median of updates 6..40, in seconds:

| F0a | F1a | F1b | F0b | F0c | F1c | F1d | F0d |
|---|---|---|---|---|---|---|---|
| 0.73845 | 0.71788 | 0.71772 | 0.73926 | 0.73919 | 0.71817 | 0.71774 | 0.73884 |

- **H per block pair:** 0.02785, 0.02913, 0.02844 and 0.02856. The minimum, which decides, is **0.027854**.
- **Not NOISY:** the self-pairs are 1.00109, 0.99978, 0.99952 and 0.99940, all within [0.985, 1.015].
- **Quality passes:** F1 against F0 is within 0.0033 PPL on both texts, at the start and after 40 updates (band 0.05).
- **(b) at the science lock:**
  - All 33 tensors are judged and pass. Each has at least 12 distinct reference paths, so the false-reject bound is at
    most 1/13.
  - Fused/band runs 0.952–1.025, against the 1.05 margin.
  - RANK1_UNRESOLVED names 0:out_down only (e_min 0.0664).
  - ACCURACY_OVER_REORDER names the 30 tensors other than dh: the fused path differs from today's by more than reorder
    noise, at an FP64 error no worse than today's path and its distinct reorderings.
- **Engagement:** 96 adapters, fallback 0, and no PEFT MLP LoRA forward in the (b) part or any F1 arm.

The H is under the ceiling the census gave (the MLP-adapter class's exclusive time, about 9% of an update).

## The registered predictions, graded

1. **H 0.02–0.05, with NO_GAIN the likelier outcome: RIGHT** (0.0279).
2. **(b) passes, and 0:out_down is RANK1_UNRESOLVED again: RIGHT.**
3. **Every tensor reaches k ≥ 9 with the section-3 families: RIGHT.** k ≥ 12 everywhere, the same counts as the A2000
   shakedown at every layer.
4. **B's drift below 1e-2 at update 40: WRONG.** It is 0.123.
   - The shakedown had already shown 0.063 by update 8, under C on the A2000, so the miss was foreseeable.
   - The prediction stood as registered. Drift is reported only and rules nothing.

## Drift (reported only)

Under C (deterministic settings), not the timed configuration. This is ‖θ_F1 − θ_F0‖ / ‖θ_F0‖ for layer 0's `up_proj`:

| after update | 2 | 4 | 8 | 16 | 32 | 40 |
|---|---|---|---|---|---|---|
| B | 0.026 | 0.048 | 0.065 | 0.087 | 0.116 | 0.123 |
| A | 0.003 | 0.005 | 0.007 | 0.012 | 0.022 | 0.026 |

The largest per-update loss difference is 0.0015.

## How it launched (the maintainer's rulings, 2026-10-20)

- **The pin rule:** one all-in cap, worst = 2 h × rate + 27 GB × download ≤ $2.10, in place of the separate rate and
  download caps.
  - The pinned offer was $1.0089/h and $0.00267/GB, so its worst was $2.09.
  - The pin was posted before create, stable in 6/6 machine-filtered reads, and held 40 s.
- **The disk floor:** the bound manifest declared `lane_record.min_disk_gb` 120, the box script's own floor, which the
  shared launcher's preflight disk check reads. A follow-up makes `bind_manifest` write it.
- **The drift snapshots:** the receipt store's ignore rule for `*.safetensors` first kept them out of the commit. They
  were added afterwards and matched their recorded sha256.

## Decision and evidence

The merged reducer returns `NO_GAIN: H 0.027854 < 0.05`.
- The maintainer re-derived every field from the store at the kit's merge commit, the drift block included.
- `phase2.json` re-derives byte-identically on Linux. On arm64 macOS the drift block's relative distances agree to
  1e-14, which is the platform's FP64 last bits; every other field is byte-identical.
- The untouched receipt SHA-256 is `2edd5715d9622b325a65ad13db1654fb95f01d510a7683b566a784a19610e5ba`, and
  `phase2.json`'s is `a524e32fbf2c6c62bd7bd2e6af15b9e1c160d1f08ae2f40ada5d63f68cb101e6`.

## What this does not say

- **No shipped default changes.** `E4B_DENSE_MLP_LORA_FUSE` stays opt-in.
- **The scope is narrow:** one card class, one fixture (Mistral-7B, the DQ11 L path), one lock and one seed.
- **It is not a GAIN.** A GAIN would only have licensed a separate default review on a second host.
