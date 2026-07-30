# Blackwell (sm_120) run: both cached MoE models train and infer; two defects found and fixed
### 2026-07-29 · RTX 5090 (33.7 GB, sm_120, driver 570.195.03) · torch 2.8.0+cu128 · transformers 5.14.1 · bitsandbytes 0.50.0 · e4b 0.6.4 · grouped-nf4-gemm **0.2.3 (ship candidate)**

Rented GPU in the datacenter that hosts the model-cache volume. **No 24 GB Ada
card was available there at run time**, on-demand or spot — the 32 GB Blackwell
was the only class on offer, and the cache volume is datacenter-locked, so
moving regions was not an option.

## What ran, and what it shows

Every stage below used the package's **own** entry points
(`load_*_4bit_streaming`, `verify_moe_4bit`, `enable_fast`, `encode_alpaca`,
`add_attention_lora`, `timed_decode`), not a reimplementation.

| stage | Qwen3-30B-A3B (48L, 128E, top-8) | OLMoE-1B-7B (16L, 64E, top-8) |
|---|---|---|
| 4-bit streaming load | 21.87 GB peak, 27.6 s | 5.79 GB peak, 5.2 s |
| `verify_moe_4bit` | **48/48 layers `nf4`** | **16/16 layers `nf4`** |
| decode (24 tok) | 3.13–3.53 tok/s | 8.5–10.6 tok/s |
| `enable_fast` | 48 modules patched | 16 modules patched |
| coherence | "red, blue, and yellow" | "red, blue and yellow" |
| QLoRA (8 steps, seq 512, r16) | 192 LoRA mods, 26.42 GB peak | 64 LoRA mods, 6.37 GB peak |

**The core claim holds on a third architecture.** The fused expert stacks that
bitsandbytes' 4-bit walker silently skips are quantized to NF4 on *every* layer
of both models, and both then generate coherent text and train with gradients
reaching the LoRA parameters.

`grouped-nf4-gemm 0.2.3` — the unreleased ship candidate — imports its full
16-module surface here, including the three new `nvme_*` modules, and
`pip check` is clean alongside e4b 0.6.4. That is sm_120 validation for the
pending release, after sm_86/sm_89.

## Expert offload: 2.27x lower training peak, same arithmetic

| Qwen3-30B-A3B QLoRA | after load | **training peak** |
|---|---|---|
| `offload=False` (seq 512, r16) | 28.89 GB | **27.30 GB** |
| `offload=True` + grad ckpt (seq 512, r16) | 5.58 GB | **12.02 GB** |
| `offload=True` + grad ckpt (**seq 192, r8** — package defaults) | 4.99 GB | **9.13 GB** |

Loss sequences are effectively identical across all three
(step 1 = **2.1341** in every case; then ~2.01–2.05 / ~1.32–1.33 / ~1.62–1.63).
Offloading experts to pinned CPU RAM is not changing the math.

**On the published 7.16 GB figure: not reproduced, and not refuted.** The
closest configuration I could match measured **9.13 GB** — 27 % higher. The
qualitative claim survives (it fits a 12 GB card where the unoffloaded run
needs 27 GB), but this is a different GPU, architecture, torch and
transformers than the original, and I did not control batch/accumulation or
`TRAIN_*` flags against the original protocol. **Treat 9.13 GB as an
independent Blackwell datapoint, not a replication.** Closing the gap needs
the original run's exact env.

**The offload path refuses rather than corrupting.** Without gradient
checkpointing it raises a *named* error — "backward re-dequantization read an
offload-evicted expert (0-element placeholder) … requires gradient
checkpointing (use_reentrant=False)" — instead of quietly training on evicted
experts. That is the right failure mode and it is worth keeping.

## `enable_fast`: no resolvable end-to-end decode speedup here

Three measurements, the first two discarded as invalid **by their own
controls**:

1. single-shot, cold Triton kernel — charges JIT compile to the fast path
   (OLMoE read 0.77x). Invalid.
2. block design (eager x10, fast x5, eager x5) — the self-pair control was
   clean (1.0042) but **eager-vs-eager block drift was +9.7 % on Qwen while
   the effect under test was −1.9 %**. Drift five times the signal. Invalid.
3. **interleaved A/B, fast toggled in place, 8 pairs, both paths warmed:**

| | median paired ratio | per-pair range | eager-only spread |
|---|---|---|---|
| Qwen3-30B-A3B | **1.006** | 0.917 – 1.105 | 1.224 |
| OLMoE-1B-7B | **1.026** | 0.978 – 1.105 | 1.203 |

Both medians sit within ~2 % of unity and the per-pair ratios straddle 1.0 in
both directions, against a 20–22 % noise floor. **Conclusion: at batch 1,
24-token decode, on this fixture, `enable_fast` shows no speedup this
instrument can resolve.**

This does **not** contradict the published 3.65x. That figure is frozen-expert
inference through the fused kernel; this is end-to-end `generate()`, where
per-token overhead outside the expert GEMM dominates. A kernel-level speedup
need not survive Amdahl at bs=1. Testing the published claim requires timing
the expert GEMM directly.

## Two defects found and fixed

1. **`python -m experts4bit_qlora.train --help` cost GPU minutes.** There is no
   argparse, so argv was ignored and it fell through into a real run: model
   load, CUDA init, 32 inductor compile workers. Measured **~10 min and 6.2 GB
   VRAM** before the user learns there is no such flag. Now exits rc=0 in 6–7 s
   printing the real env-var surface, **GPU untouched at 2 MiB**. Both `train`
   and `infer`. *Scope, stated honestly:* it exits before the model load and
   CUDA init, but still imports torch/bitsandbytes, because the package
   `__init__` eagerly imports `.lora`/`.offload`/`.fast`/`.cold_engine`. An
   import-free `--help` needs a lazy `__init__` — a wider change than this
   defect warrants.
2. **`__version__` had drifted to 0.6.3 while the distribution shipped 0.6.4.**
   Nothing tested it, so anything logging `__version__` into a receipt recorded
   the wrong version — a provenance bug. Synced, and both defects now carry
   regression tests (`tests/test_version_and_cli_help.py`).

## Method notes, including my own errors

- **The first driver crashed every post-load stage**: `load_*_4bit_streaming`
  returns `(model, config)` and I treated the tuple as a model. Found by
  reading the package's own `train.main` (`model, _ = load_...`) rather than
  guessing a second time.
- **A false green worth recording**: verifying the `--help` fix, I hot-patched
  three files from a newer branch into the installed 0.6.4 tree. `__init__.py`
  referenced `enable_decode_stack`, absent in 0.6.4, so `--help` exited 1 on an
  ImportError — and my checker reported "OK: no model load" *because the import
  crashed*. Re-verified against the complete source tree via `PYTHONPATH`, and
  the clobbered install was restored from a backup taken first.
- Loss curves here are **not** convergence evidence: 4–8 steps at batch 1,
  non-monotonic. They show the training path executes end to end.

**Teardown:** the rented GPU was 404-verified after evidence was pulled, with
zero instances left running and the cache volume intact.
