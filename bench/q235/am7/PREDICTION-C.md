# Q235 Amendment 7: the PRO 6000 box's C, predicted before it is read

This is written before the PRO 6000 Shape B box runs Amendment 7's timed pass, so the read can confirm or refute it.

C is the experts module's staging time over its compute time, per micro-batch per layer. Staging is the forward and recompute
stagings of one layer's packed experts. Compute is the forward, recompute and backward compute of the experts module.

Every input is a measured record. The one figure that comes from a specification is marked as such.

## Inputs

| record | what it gives | sha256 |
|---|---|---|
| Q235 box 1 (`q235-pro6000-2`, RTX PRO 6000 Blackwell Workstation, shape A, 94 layers) `hostrate.json` | pinned H2D **21.76 GB/s** (PCIe gen 4 x16) | `2ce01906` |
| the same box's `box/box.json` | 254,124,490,752 B staged in 187 copies per step, i.e. **1.3590 GB per experts call**; step forward 7.203 s, backward 22.116 s | `27a7863e` |
| A2000 leg `9f6e0088` (shape A, 4 layers) `box/box.json`, `hostrate.json` | H2D 6.24 GB/s; step forward 1.443 s, backward 6.226 s | `879e5fe8`, `445d3948` |
| A2000 gate-1 leg `f17f0744` (shape B, 4 micro-batches of 134 / 266 / 662 / 158 positions) `box/box.json` `moe_timing` | per experts call: stage ~222 ms; compute medians below | `dd49de5d` |

## Compute ratio, measured

A training step stages each layer synchronously: *n − 1* stagings in the forward pass (layer 0 stays resident) and *n* in the
recompute. Subtracting those at each card's own H2D gives the compute per layer:

| | forward per layer | backward per layer |
|---|---|---|
| A2000 (`9f6e0088`) | (1.443 − 3 × 0.2178) / 4 = 197.4 ms | (6.226 − 4 × 0.2178) / 4 = 1,338.7 ms |
| PRO 6000 (box 1) | (7.203 − 93 × 0.06245) / 94 = 14.84 ms | (22.116 − 94 × 0.06245) / 94 = 172.8 ms |
| **ratio** | **13.30** | **7.75** |

The A2000 figures carry per-step fixed costs (the embedding, the head and the loss) spread over 4 layers rather than 94. The ratios
therefore lean high, which leans the prediction toward staging-bound.

## Prediction (per micro-batch per layer, at Amendment 7's shape-B micro-batches)

| | stage | compute | C |
|---|---|---|---|
| A2000, measured (`f17f0744`) | 2 × 222 = 444 ms | (fwd 47.0 + recompute 46.0 + bwd 58.7) = 151.7 ms | **2.93** |
| PRO 6000, point (per-phase measured ratios) | 2 × 62.45 = 124.9 ms | 47.0 / 13.30 + 46.0 / 13.30 + 58.7 / 7.75 = 14.6 ms | **8.6** |
| PRO 6000, low end (*assumption*: the memory-bandwidth ratio 1.79 / 0.288 TB/s = 6.2 for every phase) | 124.9 ms | 24.5 ms | 5.1 |
| PRO 6000, high end (the forward ratio for every phase) | 124.9 ms | 11.4 ms | 11.0 |

**Predicted: C = 8.6, interval 5.1 to 11.0, on a host with box 1's H2D (21.76 GB/s).**
- The staging term scales as 1 / H2D. The box's own `hostrate.json` rescales the prediction: C ≈ 8.6 × 21.76 / H2D, for example
  3.7 at 50 GB/s.
- **The read REFUTES the prediction** if its C, rescaled to 21.76 GB/s by its own measured H2D, falls outside 5.1 to 11.0.

## What it implies

- **The box is staging-bound** at any H2D this lane can rent. C stays above 1 unless H2D exceeds about 186 GB/s.
- **For pipelining (prefetch, or overlapping a layer's staging with the previous layer's compute), being staging-bound is the weak case.**
  Overlap can hide at most the smaller side, so the gain per micro-batch-layer is capped at the compute: about 10 % of the experts time
  at the point (between 8 % and 16 % across the interval). Pipelining is worth registering only if the read shows C near 1.
- **The larger lever is the staged bytes.** Shape B stages each layer twice per micro-batch: four micro-batches stage it eight times per
  step. That is outside what Amendment 7 measures, and nothing here claims a speed-up.

The A2000 numbers are inputs only; this note makes no speed claim about the A2000.
