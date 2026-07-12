# Bare-metal H100: store de-quarantine + three-tier (VRAM/RAM/flash) surface

**Date:** 2026-07-12 · **Box:** Latitude.sh `g3.h100.small` (MEX2), H100 PCIe 80GB
(gen5 x16, measured pinned H2D L = 56.74 GB/s), 2× 3.5 TB NVMe, 188 GB RAM,
Ubuntu ML-in-a-Box (torch 2.13.0+cu130, driver 580.173.02) · **Genuine root, real
`drop_caches`** — the thing no container pod could give us.
**Cost:** 2.27 hr, ≈ $5.1 (cap $33, ceiling $35) · torn down 12:37:34Z, 404-verified.

**Workload (identical across every arm):** OLMoE-1B-7B-0924, 4-bit
(`quantize_moe_experts`), QLoRA r=8, alpaca, `whole_layer` staging, prefetch OFF
(synchronous — exposes raw store cost), gas=1, mbs=1, 40 steps, warmup arm
discarded, cold page cache (`sync; echo 3 > drop_caches`) before every arm.
**Metric:** authoritative HF `train_steps_per_second` (inverted to s/step).
Expert pool: 16 MoE blocks, 3.22 GB total, single-slot residency (~201 MB/block,
2 touches/step with checkpointed backward).

## Environment traps (cost the first run; documented for reproduction)

1. **cuDNN SDPA crashes at step 0** on this cu130/cuDNN build
   (`cuDNN Frontend error: No valid execution plans built`). Every arm of the
   first pass died at 0/20 while the harness kept extracting "s/step" — from the
   **weight-loading progress bar** (`Loading weights: 179/179 [00:14, 11.96it/s]`),
   a perfectly plausible-looking wrong number. Fixes: `sitecustomize.py` on
   PYTHONPATH disabling `torch.backends.cuda.enable_cudnn_sdp` (survives
   accelerate's subprocess spawn), and the metric switched to the end-of-train
   dict, with per-arm assertions that training loss lines + the offload install
   line (`homed .. MoE blocks`) are present.
2. ML-in-a-Box pip needs `--ignore-installed` (debian-owned urllib3 has no
   RECORD); numba caps numpy < 2.5 (pin `numpy>=2.4,<2.5`); no cmake by default.

## Part A — RAMStore vs FileStore across sequence length

| seq | RAM s/step | File s/step | file penalty |
|----:|----------:|-----------:|-------------:|
|  64 | 0.2286 | 1.1737 | **+413%** |
| 128 | 0.2399 | 1.1976 | +399% |
| 256 | 0.2481 | 1.1933 | +381% |
| 512 | 0.2586 | 1.2077 | +367% |
| 2048 | 0.3568 | 1.1990 | **+236%** |

File arms are **flat ≈ 1.19 s/step regardless of seq**: the step is pinned at the
flash-transfer floor (6.44 GB/step ÷ ~1.19 s ≈ 5.4 GB/s effective through the
per-block O_DIRECT path — well above the 1.5 GB/s single-thread fio floor;
per-slot reads in flight overlap). RAM arms grow with compute (0.229 → 0.357),
so the penalty *ratio* shrinks as compute hides more — the crossover-surface
shape, now on metal.

## Part B — FusedStore interior-f (two-tier), degenerate ends on metal

seq64: f=0.0 → 1.1834 · f=0.25 → 1.1390 · f=0.5 → 0.8224 · f=0.75 → 0.4926 ·
f=1.0 → 0.2276 s/step. seq512: 1.1905 / 0.8389 / 0.2577 at f=0/0.5/1.
Per-flash-block marginal cost 60–76 ms/step (402 MB / ~6 GB/s), roughly linear
in flash count. **Degenerate-end contracts hold live:** f=1.0 vs pure RAMStore
−0.4%; f=0.0 vs pure FileStore +0.8%.

## Part C — the VRAM-resident corner (f_v = 1)

| seq | resident | RAM-tier tax over resident | all-flash tax |
|----:|--------:|---------------------------:|--------------:|
|  64 | 0.2075 | +21.1 ms | +966 ms |
| 512 | 0.2232 | +35.4 ms | +985 ms |
| 2048 | 0.2550 | +101.8 ms | +944 ms |

The un-hidden H2D residue *grows* with seq (21 → 102 ms against a fixed
113 ms worst case = 2×3.22 GB / 56.74 GB/s). Mechanism not fully decomposed
(hypothesis: attention-phase overlap window shrinks relative to copy time as
kernels lengthen); flagged for the GEX131 session.

## Part D — SIMULTANEOUS three-tier placement (the new result)

`e4b-ssdtier` grew a **VRAMStore hot tier** (commit `aa8cc1f`): device-resident
homes behind the same public seam; `copy_required=False` makes the consumer's
`src.to(device)` an alias — staging a VRAM block is zero-transfer, zero public-code
change. `make_placement(..., vram_fraction=)` promotes non-flash blocks without
moving the flash set (prefetch-lane spacing preserved); `vram_fraction=0` is
bit-identical (sha256) to the two-tier map. Rationale: whole-layer training visits
blocks in a deterministic palindromic cycle, where LRU is pessimal and
Belady-optimal ≡ **static pinning** — a hot cache reduces to hot placement.

Gate on the box before arms: **37/37 pytest** (placement math, three-tier
bit-exact grads vs resident reference, degenerate ends, midpoints).

Measured (seq64, 40 steps) vs the corner-calibrated waterfall model
`t(f_v,f_r,f_f) = V + f_r·(R−V) + f_f·(F−V)` with V=0.2075, R=0.2286, F=1.1737:

| placement (f_v, f_r, f_f) | measured | predicted | error |
|---|---:|---:|---:|
| centroid (⅓, ⅓, ⅓) | 0.5869 | 0.5366 | +9.4% |
| vram+ram (½, ½, 0) | 0.2041 | 0.2181 | −6.4% |
| vram+flash (½, 0, ½) | 0.6901 | 0.6906 | **−0.1%** |

**Verdicts.** (1) Simultaneous three-tier placement works, is bit-exact, and is
**predictable from its corners to ≤10%** — the flash-dominated leg to 0.1%. No
new machinery beyond a placement map. (2) `vram+ram (½,½)` lands at the
resident floor (0.2041 ≤ V within noise): at f_v=0.5 the remaining RAM-lane H2D
is fully hidden — on fast-PCIe metal, half residency buys the whole RAM-tier tax
back. (3) Waterfall economics on this box: flash→RAM saves ~0.64 s/GB,
RAM→VRAM ~0.018 s/GB (36×) — fill RAM first, promote to VRAM with what's
left. The VRAM tier's real constituency is slow-PCIe consumer boxes (gen1
3090-class, where 1/L rivals 1/S) plus the host-RAM it frees. (4) The knee thesis
carries to metal unchanged: compute-hiding governs everything; store identity
only matters where transfer isn't hidden.

## Evidence

- `bm-final.tgz` (all 24 arm logs, configs, run sentinel, gates, env fingerprint):
  sha256 `9d93472e6d4be7c428fe1e055438a8ede357b51ddadb8694d4d07475c58b1488`
  at mini `~/bm-evidence/` (box shredded + deleted; private code never touched
  persistent disk — `/mnt/ramcode` tmpfs, `PYTHONDONTWRITEBYTECODE=1`,
  shredded before teardown).
- `baremetal_results.json` sha256
  `1eb063bc2b739559fbeeab51dfc38096992b549ad9be000e246bb5421aefb5dc`.
- First run's invalid pass (cuDNN crash) preserved in the same tarball
  (`out_e*_{ram,file}.log`) as the cautionary artifact.
