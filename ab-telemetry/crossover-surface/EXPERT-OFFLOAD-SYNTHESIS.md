# Expert-offload for 4-bit MoE QLoRA — consolidation synthesis (2026-07-12)

*One clean account of the expert-offload arc: what is validated, the routed-subset negative with its
complete causal chain, what it means for the product, and the host decision for the absolute-knee run.
Every number traces to a committed, OpenTimestamps-anchored reduction under `bench/crossover-surface/`.*

## TL;DR

- **`whole_layer` expert-offload is validated end-to-end for QLoRA training** — bit-identical
  offload-vs-resident forward, convergence preserved at scales the resident model OOMs, FileStore
  (O_DIRECT) byte-identical to RAMStore against a measured noise floor.
- **`routed_subset` staging is validated for forward/decode only, and is *not viable for training* —
  and we now know exactly why.** A five-probe investigation traced the training divergence to its
  root: the un-staged experts' zeroed rows **perturb the fused dequant of the forward** by a tiny,
  coherent, depth-accumulating amount that compounds over training. It is **intrinsic to subsetting**
  (a full-count control is bit-identical to whole), not a fixable staging bug, and it recovers only at
  read-fraction 1.0 — which defeats the bandwidth purpose.
- **Product consequence:** the SSD-tier thesis is an **inference/decode** bandwidth technology. That
  regime is where routed staging pays the most *and* where it is fully validated (forward bit-exact).
  Training-viability was a stretch goal; it now has a clean, evidenced boundary, not a question mark.

---

## 1. Validated: `whole_layer` offload + FileStore

| claim | evidence |
|---|---|
| Offload forward is **BEFORE bit-identical** to resident | step-0 loss 5.30==5.30 (Qwen3-30B), 0.7038==0.7038 (OLMoE) |
| Convergence preserved where resident **OOMs** | Qwen3-30B-A3B trains in 7.16 GB peak; resident OOMs on 12–24 GB |
| FileStore (O_DIRECT) == RAMStore | per-slot byte parity; FileStore≈RAM within the resident-vs-RAM noise floor (0.033 ≤ 0.038), not a fixed epsilon |
| async-H2D perf-degenerate gate | f=1.0 FusedStore == pure RAMStore to ≤0.5% s/step (was +15.4% pre-fix) |
| relative placement×access surface | placement penalty shrinks +273%→+109% as eff_tokens 64→2048 (long-context is where the slow tier is cheapest) |

The **absolute** knee (`f_knee = 1 − S/L`, located with measured striped-NVMe S and link L) is the one
piece still owed — it is host-gated on a root bare-metal box (§5).

---

## 2. The routed-subset negative — complete causal chain

Routed-subset staging streams only the experts a forward routes to (`read_fraction × layer`), the
larger bandwidth win. It is **forward bit-identical** to whole-layer (the sparse expert loop never
reads un-routed rows). But **training diverges**, and the divergence was chased to its root by direct
intervention — each step a pre-registered, OTS-stamped bracket at the decisive seq-256 point
(Qwen3-30B, rf≈0.69, floor 0.001 across **seven** consecutive clean measurements):

| # | probe | result | commit |
|---|---|---|---|
| 0 | **dose-response** (3 points) | gap **scales with fill mass** (1−rf): 0.062 @ rf0.97 → 0.186 @ rf0.69; **convex**, not √ | da6e776, 0173b4c |
| 1 | **content mask** (un-routed rows → exact 0.0) | real fix, **halved** the gap 0.182→0.104; correct but insufficient | 9e981dd |
| 2 | **stale-union top-up** (boundary-block freshness) | real bug (147 escapes, block-0 only, as predicted) but **zero** loss effect | fc929ca |
| 3 | **gradient diff** (whole vs routed, float64) | residual is **sub-atomic-noise** — not a localizable discrete bug | cea7677 |
| 4 | **R128 bisection** (same expert *count* both sides) | R128 = whole **exactly** (gap 0.000); control diverges 104× → **subset-intrinsic**, assembly path innocent | ce059a2 |
| 5 | **activation diff** (signed mean per block, √N-resolution) | zero rows **perturb the forward** — 1e-7→2e-4 growing with depth; **seed = forward dequant numerics** | ce059a2 |

**The mechanism, stated once:** the zero-decode masked rows are never *read* by the sparse expert
loop, but the fused whole-stack dequant still *processes* them, and their content (zero vs real)
perturbs the routed experts' dequantized values by ~5e-5 (signed mean) — coherent and growing through
the residual stream. That perturbation is **below eval print-precision** (so the forward reads
"bit-identical" at 3 decimals) and **below the gradient atomic-noise floor** (so a gradient diff finds
nothing), yet it is real and coherent, so it compounds over 150 steps into the 104× loss gap. The
A4-arm non-attenuation (gap doesn't shrink under gradient averaging) and the dose-response convexity
are the loss-level fingerprints of that same coherent forward bias. **R128 removes it entirely by
staging all real bytes — proving the zeros are the whole story, and that no staging-layer fix helps
short of rf=1.0.**

**Disposition:** `whole_layer` is the only training-validated staging. `routed_subset` stays
forward/decode-only, gated behind `AXOLOTL_EXPERT_OFFLOAD_ROUTED=1` with the mechanism documented. The
`fix/routed-stale-union` branch carries two genuine correctness fixes (zero-decode mask, routing-aware
top-up) with no training benefit — fold them into the decode path or leave as a documented side branch.

---

## 3. What this means for the product / thesis

The negative is **bounded, not fatal.** The SSD-tier value proposition is *inference/decode* bandwidth
for MoE — precisely the regime where (a) routed staging saves the most (low read-fraction), and (b) it
is **fully validated** (forward bit-exact). Training-viability was always the stretch ambition; it now
has a hard, mechanistically-explained boundary. The honest one-line claim for the page: *"routed-subset
staging is an inference-bandwidth technology; QLoRA training requires whole-layer staging, for a
now-understood numerical reason."* That's a stronger, more defensible statement than a hopeful maybe.

Methodological asset worth surfacing in the write-up: the whole arc is a **pre-registration-disciplined
mechanism hunt** — every verdict criterion committed and OTS-stamped *before* the data, a resolution-limit
result honestly reported (the gradient diff), and a √N-resolution instrument (signed-mean activation
diff) built to see the sub-noise coherent bias the max-diff probe couldn't. Seven-arm floor
reproducibility (0.001) across three pods and two GPU SKUs.

---

## 4. Upstream / follow-through

- **axolotl `feature/expert-store`** — whole-layer validated; routed gate wording honest and current.
  Ready for a PR writeup built on §1–2.
- **bitsandbytes #1965** (Experts4bit) — the real-bnb zero-decode byte verification (nf4 code 7 = 0x77,
  fp4 code 0 = 0x00, per-expert exact) is corroborating evidence; gated on the v0.50.0 floor.
- **product/thesis page** — §3 is the copy.

---

## 5. Absolute-knee host decision (GEX131) — pricing

The one perishable item: the *located* `f_knee` needs a **root bare-metal host** (drop_caches, raw
multi-NVMe, mdadm RAID-0, GDS) that a container/pod structurally cannot provide. The kit is committed
and push-button (`RUNBOOK.md`, `rig/01-04`, `knee_predictions.json`); it runs ≤2h once SSH exists, with
**whole-layer** as the validated payload. The choice is *which host*, and it turns on billing model for a
~2-hour one-off, not $/hr.

| provider | cheapest 1-GPU box | GPU | $/hr (≈$/2h) | setup fee | teardown | root+multi-NVMe+RAID | API destroy |
|---|---|---|---|---|---|---|---|
| **Latitude.sh** | `g3.h100.small` | H100 80GB | **$1.68 (~$3.36)** | **none** | delete → billing stops | **yes** (2× 3.8 TB NVMe) | **yes** (REST/Terraform/`lsh`) |
| **Hetzner GEX44** | GEX44 | RTX 4000 Ada 20GB | €0.375 (~$0.41) | **€114 (~$125)** | cancel anytime; setup non-refundable | yes (2× 1.92 TB NVMe) | Robot only (no GPU in Cloud API) |
| **Hetzner GEX131** | GEX131 | RTX PRO 6000 96GB | €1.92 (~$2.11) | **€599 (~$659)** | as above | yes | Robot only |
| RunPod (pods) | RTX 4090 pod | 4090 24GB | $0.69 | none | stop pod | **no host mdadm/GDS** (container root only) | yes |
| Crusoe / DataCrunch | H100 VM | H100 | ~$3.35–3.90 | none | delete stops | root; raw-disk RAID = bare-metal tier | yes |

*(Hetzner is EUR-native at VAT 0%; USD ≈ 1.10×. Latitude H100 RAM not published but ≥64 GB certain;
a third-party-listed Latitude L40S @ $0.74 is NOT on the official page — treat as unavailable.)*

**Recommendation: Latitude.sh `g3.h100.small`, ~$3.36 for the run.** It is the only option that is both
(a) true bare-metal with root + 2× NVMe (RAID-0/GDS-capable) and (b) genuinely hourly with a clean
API destroy — so the teardown discipline (evidence-first, session-independent, hard wall-clock cap,
incremental rsync off-box) maps to it exactly as it does to RunPod. **Hetzner GEX is the wrong economics
for a one-off:** its €114–599 one-time setup fee makes a 2-hour run cost **~$126 (GEX44) to ~$660
(GEX131)** — 40–200× Latitude — and it provisions as a physical Robot box (not the instant hourly Cloud
API; the Cloud token on file cannot start a GPU box at all). Wry note: **GEX131 *is* the RTX PRO 6000
Max-Q** I rented hourly on RunPod all week at ~$1.64/hr — the codename fits, but renting the actual
Hetzner GEX131 for a short run is the single most expensive path.

**Do NOT provision anything until:** (1) the teardown watchdog is built and tested against the chosen
provider's real API *before* a GPU-hour is spent, and (2) `feedback_rented_compute_teardown_discipline`
is honored (session-independent teardown + hard bill cap + evidence streaming). Nothing is billing now.

---

## Provenance

Every reduction and prereg in `bench/crossover-surface/` is OpenTimestamps-anchored; verdict criteria
were committed *before* their data (commit hashes in §2). Arc total ~$50 across ~11 rented pods; zero
pods billing at write-time.
