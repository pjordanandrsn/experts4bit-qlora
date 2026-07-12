# VRAM tier on slow-PCIe silicon: all four pre-registered predictions PASS

**Date:** 2026-07-12 · **Prereg:** `prereg_vram3090.json` @ `acabdfc` (OTS), frozen
before any arm · **Host:** RunPod SECURE RTX 3090, driver 580.126,
**L = 20.36 GB/s pinned H2D** (gen4 x16 under load; the idle `pcie.link.gen.current=1`
reading is ASPM power-saving — always measure under load). Two community pods
wedged (RUNNING/no-IP) before ever mapping, so the hoped-for true-gen1 draw was
unavailable; 3 provisioning rolls spent per prereg, slowest *measured* link kept.
**Cost:** ≈ $0.40 total (36 min SECURE + two wedge stubs) · pod deleted, 404-verified,
watchdog-autonomous (evidence pulled + private code shredded before teardown).

Same frozen workload as the H100 chain (OLMoE 4-bit whole_layer, seq64, 40 steps,
prefetch off, HF `train_steps_per_second`); V/R corners n=2 interleaved.

## Measured

| arm | s/step |
|---|---:|
| V (resident, n=2) | **0.3735** (spread 0.011) |
| R (all-RAM streamed, n=2) | **0.6278** (spread 0.013) |
| fused fv=0.0 | 0.6277 |
| fused fv=0.25 | 0.5682 |
| fused fv=0.5 | 0.5126 |
| fused fv=0.75 | 0.4219 |
| fused fv=1.0 | 0.3676 |

**RAM tax (R−V) = 254 ms/step = +68% — vs 21 ms (+10%) on the H100.** The tax is
19–23× the n=2 floors: decisive.

## Prereg verdicts (all frozen at `acabdfc`)

- **P1 ends parity (±3%): PASS** — fused fv=1.0 vs V −1.6%; fv=0.0 vs R −0.0%.
  The VRAM-tier machinery is free at both degenerate ends.
- **P2 interior linearity (±10%): PASS** — errors +0.7 / +2.4 / −3.5%. The
  placement→step-time map is *linear* on this host, tighter than the H100's ±10%.
- **P3 frozen tax model: PASS** — `tax_pred = max(0, 6.44/L − 0.092·(V/0.2075))`
  = 151 ms; measured 254 ms = 1.69× (band [0.5, 2.0]×). The V-scaled hideable
  window over-credits this host (sm86 overlaps less per unit compute than the
  model assumed), but the band holds.
- **P4 half residency recovers 40–60% of the tax: PASS** — fv=0.5 recovered 45.3%.
- **Headline directional (≥2× per-GB value at L<30): PASS at 12×** — per-GB
  promotion value (R−V)/6.44 GB = **0.0395 s/GB vs 0.0033 on the H100**.

## What this settles

The dose–response on the link axis, with two measured points: drop L from
56.7 → 20.4 GB/s (2.8×) and the per-byte value of VRAM residency rises **12×**
(super-linear, because slower links also hide a smaller fraction under compute).
On datacenter gen5, the VRAM tier is a rounding error (+10% tax, half residency
recovers it all); on consumer-class links it is the difference between 0.63 and
0.37 s/step — a **1.71× end-to-end speedup dialed linearly by `fused_vram_fraction`**,
bit-exact (gates 21/21 on-box), with zero public-seam changes. Fill order stands:
RAM first, then promote to VRAM — and the promotion budget matters most exactly
where the hardware is cheapest. `e4b-ssdtier@aa8cc1f`.

## Ops notes (each cost an iteration this run)

- **bnb 0.49.1's `libbitsandbytes_cuda130.so` needs `LD_LIBRARY_PATH` =** the pip
  `nvidia/*/lib` dirs on torch-cu130 pods (import succeeds without it; only a real
  quantize call fails: `Native code method attempted to call cquantize_blockwise_bf16_nf4`).
  The H100 masked this via its system CUDA-13.3 toolkit. Add a **real quantize
  smoke** to every build gate — import checks prove nothing about native libs.
- **Nested-ssh stdin:** an inner `ssh` in a heredoc-driven block swallows the rest
  of the script (silent half-run); `ssh -n` fixes that but *breaks* heredoc-to-pod
  payloads. Rule: `-n` when the inner ssh takes inline commands and lines follow;
  no `-n` when the whole stdin IS the pod script.
- `pgrep -f` matched its own probe cmdline through two ssh hops ("build RUNNING"
  while nothing ran) — verify long jobs by sentinel/output files, not pgrep.

## Evidence

`v3090-evidence.tgz` (SENT incl. attempt-1 failures, all arm logs, configs,
results JSON) sha256
`23fb0895269c842bc1b3ff60b9512cfccfdcbc6da2a52ecea948c29d5b8c7612`
at mini `~/v3090-evidence/`. Results JSON: `v3090_results.json` (this dir).
