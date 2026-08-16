# Pre-registration — does the host-RAM ratio scale with total expert bytes?

**Written 2026-08-13, before the Qwen3-30B-A3B checkpoint was downloaded.** Committed
ahead of the data so the prediction cannot be adjusted to fit it. The measurement it
predicts is the one [`RESULTS-host-ram-ceiling.md`](RESULTS-host-ram-ceiling.md) named as
unmeasured: *"the ratio should widen substantially on larger MoEs — but that is the
mechanism's prediction, not a measurement, and nothing here measures it."*

## The claim under test

The host-resident path pins **every** expert of **every** layer in host DRAM, so its
requirement should grow with total expert bytes. The arena path pins `hot_rows ×
row_stride` regardless of how many experts exist, so its requirement should stay roughly
flat. If that is right, the ratio is not a property of the feature — it is a property of
the model, and it grows.

## Geometry, derived from each model's own config

| | OLMoE-1B-7B-0924 | Qwen3-30B-A3B |
|---|---|---|
| layers × experts | 16 × 64 = 1024 | 48 × 128 = **6144** |
| hidden / moe\_intermediate | 2048 / 1024 | 2048 / 768 |
| per-expert nf4 | ~3.54 MB | ~2.51 MB |
| **total expert bytes** | **3.62 GB** (measured arena) | **~15.4 GB** (predicted arena) |
| `hot_rows=64` pinned | ~0.2 GB | **~0.16 GB** |

**4.2× more expert bytes, and slightly fewer pinned bytes on the arena side** — because
Qwen3's experts are individually smaller, so 64 of them is less memory than 64 of OLMoE's.

## Predictions (stated before any run)

Measured for OLMoE: host **5.91–6.17 GB**, arena **2.28–2.42 GB**, ratio **2.56×**.
Non-expert baseline is therefore about `6.17 − 3.83 ≈ 2.3 GB`.

1. **Host requirement ≈ 2.3 + ~16 ≈ 18–21 GB.** Dominated by pinned experts.
2. **Arena requirement stays 2.2–3.0 GB.** It should NOT scale with expert count. Some
   growth is allowed for 48 layers of bookkeeping vs 16, but not 4×.
3. **Ratio ≈ 7–9×**, up from 2.56×.
4. **Peak RSS again overstates the host arm**, since the checkpoint is 61.1 GB of bf16
   read in full to quantize — far more reclaimable page cache than OLMoE's 13.84 GB.

## What would falsify each

- **(1) fails** if the host arm trains under ~10 GB. Then pinned experts are not the
  dominant term and the mechanism's story is wrong.
- **(2) fails** if the arena arm needs more than ~4 GB. Then something in the arena path
  scales with expert *count*, which would be a real defect worth finding — this is the
  prediction I most want to be wrong, because a failure here is a bug, not a disappointment.
- **(3) is a consequence** of (1) and (2); it is not independently informative.
- **Stop rule:** if the arena arm's requirement exceeds 4 GB, stop the ladder and
  investigate the cause before reporting any ratio.

## Method (unchanged from the OLMoE run)

Same drivers, same four steps, same seed, same box, `hot_rows=64`. The cap is
`docker --memory=N --memory-swap=N`, positive-controlled in both directions. Verdict from
`State.OOMKilled`. Descend until two consecutive failures; report the bracket, not a point.

**Known risk:** the A2000 is shared and had 9.2 GB free at pre-flight; prior evidence puts
Qwen3-30B-A3B training at ~7.16 GB peak VRAM. A CUDA OOM is a contention artifact, not a
host-RAM verdict, and will be reported as such rather than folded into the ladder.
