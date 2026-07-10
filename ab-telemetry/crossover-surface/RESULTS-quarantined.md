# Crossover surface — quarantined relative version (pod)

## Quarantine rule (read first)

Measured on a containerized shared RTX 3090. **O_DIRECT reads bypass the page cache (verified), so
the relative deltas are honest — but NO absolute bandwidth, NO located `f_knee`.** The absolute
knee (`f_knee = 1 − S/L`) needs measured S (striped NVMe) and L (link) on a root bare-metal host;
that is a separate session (`RUNBOOK.md`, blocked on host provisioning). This is the RELATIVE
(placement × access) surface — the honest precursor that extends the S3 single-point placement
curve to the full 2D surface and validates the harness end-to-end.

## Setup

OLMoE-1B-7B QLoRA, FusedStore (private `e4b-ssdtier` @ b4577b7, async-H2D fix), 30-step arms,
seed 42, O_DIRECT FileStore. Access axis = `sequence_len` at micro_batch 1 (eff_tokens/forward);
placement axis = `fused_ram_fraction` f. OLMoE (not Qwen3-30B) for a fast 12-arm surface — the
shape is the deliverable and it's directly comparable to the S3 re-probe; Qwen3-30B is the subject
for the bare-metal absolute run.

## Gates (both green across the whole surface)

- **G1 bit-exact** — FusedStore suite 28/28 before any arm.
- **G2 perf-degenerate** — f=1.0 vs pure RAMStore: **+1.0% / −0.5% / +0.0%** at eff_tokens
  64 / 256 / 2048 — all ≤0.5–1%, the async-H2D fix holds across the surface (was +15.4% pre-fix).
  Losses tiering-invariant within each slice.

## The surface (relative s/step)

| eff_tokens | f=1.0 (all RAM) | f=0.5 | f=0.0 (all flash) | placement penalty (f0/f1) |
|---|---|---|---|---|
| 64 | 1.97 | 4.48 | 7.34 | **+273%** |
| 256 | 2.04 | 4.62 | 7.33 | **+259%** |
| 2048 | 3.60 | 4.67 | 7.52 | **+109%** |

## Finding: the placement penalty shrinks as compute/forward grows — and the mechanism decomposes

The relative cost of putting experts on the slow tier **falls from +273% to +109%** as eff_tokens
goes 64 → 2048. The two rows explain why:

- **f=0.0 (all flash) is staging-bound** — nearly flat in eff_tokens (7.34 → 7.33 → 7.52). It pays
  the per-layer O_DIRECT read every forward regardless of token count.
- **f=1.0 (all RAM) is compute-bound** — rises with eff_tokens (1.97 → 2.04 → 3.60) as more tokens
  mean more compute.

So as eff_tokens grows, the all-RAM baseline climbs toward the staging-bound ceiling, shrinking the
*relative* penalty. **The crossover diagonal: the more compute per forward, the more expert bytes
you can park on the slow tier for the same relative cost.** Long-context / high-token-per-forward
inference is exactly where the SSD tier is cheapest — consistent with the Phase-0 sub-finding that
the thesis favors long-context single-stream.

## Scope honesty (what this is NOT, and where Phase 0's read-fraction bites)

This is the **whole-layer** staging design: every visited layer stages its *entire* expert stack,
so staged bytes are eff_tokens-independent and the surface above is a *compute-hiding* surface. The
deeper thesis — stage only the experts actually routed to — is where Phase 0's read-fraction (0.53
→ 0.80 over eff_tokens 64 → 2048 for Qwen3-30B) directly sets staged bytes, giving a *second*,
opposing pull (more tokens → more distinct experts → more to stage). **Measuring the routed-subset
staging surface is the next design increment**; the current FusedStore does not implement it, so it
is not claimed here.

## Absolute knee — still owed, still host-gated

The located `f_knee` and the standing-asset storage characterization (fio ladder, throttle curve,
measured S/L, RAID-0 stripe, GDS) require the root bare-metal host in `RUNBOOK.md`. This quarantined
surface de-risks that run: harness validated end-to-end, G1/G2 green across the surface, the
(placement × access) shape established in relative terms. Artifacts: `surface_reduction.json`,
`crossover_surface.svg`, per-arm logs.
