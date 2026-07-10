# Session 4 runbook — bare-metal crossover surface (Claims 4/5)

**Status: host-independent kit READY; blocked only on a root bare-metal host.** Everything here
runs push-button once a host exists, keeping the run inside the handoff's 2-hour provisioning
time-box. Nothing in this directory needs the host except steps 1–5 below.

## Why a pod cannot do this (settled)

A RunPod (or any container) pod is disqualified for the *absolute* knee: no root `drop_caches`
(container lacks CAP_SYS_ADMIN — verified: the WS pod denied `nvidia-smi -pl`), no host-side
`nvidia-fs`/GDS, and NVMe is post-provisioned (can't build the RAID-0 bandwidth stripe). O_DIRECT
*does* work in a pod (verified 4.2 GB/s on a 3090), which is why the **relative** surface is
pod-runnable and quarantined — but the standing-asset characterization (fio ladder, throttle
curve, measured S/L, located absolute knee) needs the root host.

## Host (D1 — your call; I can't provision it)

Root bare-metal with an NVIDIA GPU ≥12 GB, **≥2 raw NVMe data drives**, ≥64 GB RAM, root.
Options, in cost order:
- **Hetzner GEX / auction** or **Latitude.sh GPU metal** — short-lease, ~$15–30 for this run. *No
  API key for either is on the mini or Mac* — you'd drop a key or provision + hand me SSH.
- **RunPod bare metal** — exists, but not in the REST API my tooling uses (GraphQL introspection
  is 403 with the current key), and it's flat-monthly per the handoff. Provision in the console,
  give me SSH.

## Sequence (once SSH to the root host is available)

| step | script | produces | gate |
|---|---|---|---|
| 1 | `rig/01_rig_check.sh` (BUILD_RAID=1 STRIPE_DRIVES=…) | `receipts/rig.json`, `/mnt/stripe` | root + drop_caches + RAID-0 or ABORT |
| 2 | `rig/02_fio_ladder.sh` (THROTTLE=1) | `receipts/fio.json`, `receipts/throttle.csv` | **standing asset**; S1, S2 |
| 3 | `rig/03_measure_link.py` | `receipts/link.json` | L |
| 4 | **fill `knee_predictions.json` stage 2 with S1/S2/L, commit+push BEFORE the sweep** | pre-registration | the falsifiable numbers |
| 5 | ship axolotl `feature/expert-store` + private `e4b-ssdtier`, stage weights to `/mnt/stripe` | env | (same build recipe as prior sessions) |
| 6 | `rig/04_sweep.sh` | `results/surface.jsonl` | **G1** bit-exact, **G2** f=1.0 ≤0.5% vs RAMStore, **G3** S2≥L |
| 7 | `05_reduce_surface.py` + sync off-host + teardown | the located diagonal | — |

## The three gates (pre-committed in `knee_predictions.json`)

- **G1 bit-exact** — the FusedStore degenerate-ends + byte-parity suite (already green on the 3090);
  measures correctness, not validity.
- **G2 perf-degenerate** — f=1.0 == pure RAMStore to ≤0.5% s/step (the async-H2D lesson compiled
  into procedure; was 15.4% before the fix). Fail → measuring serialization, stop.
- **G3 stripe-saturates-lane** — measured striped read S2 ≥ link L *before* the sweep. Fail → the
  f-axis measures the wrong bottleneck and the crossover location is meaningless.

## What's measured (scoped by Phase 0, not the original grid)

Placement `f ∈ {0,.25,.5,.75,1}` × access `eff_tokens ∈ {64, 256, 2048}` — the below-knee /
at-knee / plateau access slices Phase 0 located (read fraction 0.53 / 0.69 / 0.80). The original
handoff's ≥1024-token batch grid is skipped: Phase 0 showed those are all one saturated regime.

## Minimum vs target (handoff)

- **Minimum (standing asset even with no sweep):** fio ladder + throttle curve + measured L +
  committed stage-2 `knee_predictions.json`. Steps 1–4.
- **Target:** the located crossover diagonal, single-drive interleaved, matching the registered
  `f_knee = 1 − S/L` within tolerance — or missing it with the miss explained. Steps 1–7.
- **Stretch:** RAID-0 stripe arm, contiguous-placement control, GDS-vs-CPU-bounce mini-axis (D3),
  interference matrix.
