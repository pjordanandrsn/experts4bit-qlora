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
| A2000 gate-1 leg `f17f0744` (shape B, 4 micro-batches of 134 / 266 / 662 / 158 positions) `box/box.json` `moe_timing` | per experts call: stage ~222 ms; compute means below | `dd49de5d` |

## Compute ratio, measured

A training step stages each layer synchronously: *n − 1* stagings in the forward pass (layer 0 stays resident) and *n* in the
recompute. Subtracting those at each card's own H2D gives the compute per layer. A step's `backward_s` holds the recompute forward
as well, so the backward-only figure subtracts the forward:

| | forward per layer | backward per layer (with recompute) | backward only |
|---|---|---|---|
| A2000 (`9f6e0088`) | (1.443 − 3 × 0.2178) / 4 = 197.4 ms | (6.226 − 4 × 0.2178) / 4 = 1,338.7 ms | 1,141.3 ms |
| PRO 6000 (box 1) | (7.203 − 93 × 0.06245) / 94 = 14.84 ms | (22.116 − 94 × 0.06245) / 94 = 172.8 ms | 157.96 ms |
| **ratio** | **13.30** | | **7.23** |

The A2000 figures carry per-step fixed costs (the embedding, the head and the loss) spread over 4 layers rather than 94, so the ratios
lean high. The ratios also come from shape A (2,648 positions) and are applied to shape-B micro-batches (134 to 662 positions), so they may
overstate the PRO 6000's compute advantage. That would push the read's C lower than predicted.

## Prediction (per micro-batch per layer, at Amendment 7's shape-B micro-batches)

| | stage | compute | C |
|---|---|---|---|
| A2000, measured (`f17f0744`, means over every sampled call) | 2 × 222 = 444 ms | (fwd 47.3 + recompute 45.9 + bwd 58.7) = 152.0 ms | **2.92** |
| PRO 6000, point (per-phase measured ratios) | 2 × 62.45 = 124.9 ms | 47.3 / 13.30 + 45.9 / 13.30 + 58.7 / 7.23 = 15.1 ms | **8.25** |
| PRO 6000, low end (*assumption*: the memory-bandwidth ratio 1.79 / 0.288 TB/s = 6.2 for every phase) | 124.9 ms | 24.5 ms | 5.09 |
| PRO 6000, high end (the forward ratio for every phase) | 124.9 ms | 11.4 ms | 10.93 |

**Predicted: C = 8.25, interval 5.1 to 11.0 (the computed 5.09 to 10.93, rounded outward), on a host with box 1's H2D (21.76 GB/s).**
- The staging term scales as 1 / H2D. The box's own `hostrate.json` rescales the prediction: C ≈ 8.25 × 21.76 / H2D, for example
  3.6 at 50 GB/s.
- The Shape B box's pinned host is PCIe gen 5 (listed at 52.9 GB/s), and its H2D is unmeasured until its own host-rate phase runs. At
  an H2D of 40 to 53 GB/s the point is 3.4 to 4.5, and the interval 2.1 to 6.0. That is still staging-bound.
- **The read REFUTES the prediction** if its C, rescaled to 21.76 GB/s by its own measured H2D, falls outside 5.1 to 11.0.

## What it implies

- **The box is staging-bound** at any H2D this lane can rent. C stays above 1 unless H2D exceeds about 180 GB/s.
- **For pipelining (prefetch, or overlapping a layer's staging with the previous layer's compute), being staging-bound is the weak case.**
  Overlap can hide at most the smaller side, so the gain per micro-batch-layer is capped at the compute: about 11 % of the experts time
  at the point (between 8 % and 16 % across the interval). Pipelining is worth registering only if the read shows C near 1.
- **The larger lever is the staged bytes.** Shape B stages each layer twice per micro-batch: four micro-batches stage it eight times per
  step. That is outside what Amendment 7 measures, and nothing here claims a speed-up.

The A2000 numbers are inputs only; this note makes no speed claim about the A2000.
