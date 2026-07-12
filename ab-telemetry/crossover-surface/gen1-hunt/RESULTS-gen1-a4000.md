# Gen1-class link point: L = 4.65 GB/s (A4000 community host) — 4/5 predictions PASS

**Date:** 2026-07-12 · **Prereg:** `prereg_gen1_hunt.json` (OTS, frozen before any
provisioning) · **Host:** RunPod COMMUNITY RTX A4000 16 GB, pod `bfru9rcjbnclo1` —
pinned H2D **4.65–4.82 GB/s under load** across three independent measurements
(qualification probe on a sibling pod `ct0tay59lp1zxl`, qualify roll, and the
arms pod's own gate; idle `pcie.link.gen.current` read 1/x16, and here the LOAD
measurement agrees — a genuine gen1-class link, not ASPM noise). Cost of the whole
hunt ≈ **$0.4**; every pod terminated + 404-verified (survey: `gen1_survey.jsonl`).

## The hunt itself (protocol result)

| roll | SKU / cloud | outcome |
|---|---|---|
| 1 | 3090 COMMUNITY | no stock |
| 2, 3 | A5000 / 3090 Ti / A6000 COMMUNITY | no stock |
| 4 | 4090 COMMUNITY | create error (host resources) |
| 5, 7 | A4500 COMMUNITY ×2 | wedge (RUNNING/no-IP, killed at 10 min) |
| 6 | A4000 COMMUNITY | torch-python miss (probe bug, fixed) |
| 9 | A4000 COMMUNITY | **L = 4.82 GB/s → qualifies** (probe semantics terminated it) |
| 10 | L4 SECURE | L = 13.33 GB/s (x8 gen4) — kept as a slow-gen3 survey point |
| qualify | A4000 COMMUNITY | **L = 4.8 → HELD**, arms ran here (gate re-measured 4.65) |

## Arms (seq64, 40 steps, seed 42 — the acabdfc arm set, NO flash tier)

| arm | steps/s | s/step |
|---|---:|---:|
| V resident (n=2) | 1.625 / 1.709 | **0.615 / 0.585** (mean 0.600, spread 0.030) |
| R all-RAM (n=2) | 0.524 / 0.525 | **1.908 / 1.905** (spread 0.004) |
| fused fv=0.0 | 0.559 | 1.789 |
| fused fv=0.25 | 0.610 | 1.639 |
| fused fv=0.5 | 0.803 | 1.245 |
| fused fv=0.75 | 1.128 | 0.887 |
| fused fv=1.0 | 1.580 | 0.633 |

**RAM tax = R − V = 1.306 s/step (+218%)** — vs +68% on the 3090 (L=20.36) and
+10% on the H100 (L=56.74).

## Frozen predictions (prereg verbatim)

- **P1 ends parity (±3%): FAIL** — fv=1.0 vs V **+5.4%**, fv=0.0 vs R **−6.2%**.
  Caveat, not excuse: this host's own V floor spread is 5.1% (0.030/0.585 —
  community-pod noise an order of magnitude above the H100/3090 floors the ±3%
  bar was calibrated on), which covers the fv=1.0 miss. The fv=0.0 −6.2% miss is
  *outside* the tight R spread (0.2%) and is a real open sub-question on this
  host class (FusedStore's ram tier ran mildly FASTER than the public RAMStore).
- **P2 interior linearity (±10%): PASS** — errors +3.8 / −0.6 / −4.3%.
- **P3 frozen tax model: PASS** — `max(0, 6.44/L − 0.092·(V/0.2075))` = 1.119 s
  predicted, 1.306 measured, **ratio 1.17** (band [0.5, 2.0]) — the model's best
  fit of the three hosts.
- **P4 half residency recovers 40–60%: PASS** — fv=0.5 recovered **50.6%**.
- **P5 dose response: PASS** — per-GB promotion value **0.2028 s/GB = 5.14× the
  3090's 0.0395 = 61× the H100's 0.0033**.

## The three-point link axis

| host | L (GB/s) | per-GB value (s/GB) | RAM tax |
|---|---:|---:|---:|
| H100 PCIe gen5 | 56.74 | 0.0033 | +10% |
| RTX 3090 gen4 | 20.36 | 0.0395 | +68% |
| **A4000 gen1-class** | **4.65** | **0.2028** | **+218%** |

Log-log slope of value vs 1/L: **2.42** between the fast pair, **1.11** between
the slow pair — the super-linearity *flattens toward pure 1/L at the slow end*,
exactly what the tax model implies: value = (transfer − hidden)/GB, and once the
link is slow enough that compute hides almost nothing, value degenerates to
transfer time ∝ 1/L. The three points bracket the whole curve the placement
economics need: on consumer links, `fused_vram_fraction` is worth **3.2×
end-to-end** here (1.908 → 0.60 s/step, dialable linearly per P2).

## Second live validation of the bnb retention fix (unplanned but decisive)

The V (resident) arms **OOM'd this 16 GB card at step 0 with stock bitsandbytes**
(15.42 GiB allocated in the dequant path — the checkpoint-early-stop
parametrize-cache leak, `vram-tier-30b/FINDING-dequant-retention.md`; resident
mode has no offload installer to strip the hooks, so the leak is live). Applying
the pending 2-line upstream fix (`always_call=True` + counter clamp — the exact
`d5713cd` patch awaiting PR approval) to the pod's installed bnb made resident
OLMoE **fit and train with headroom** (0.585–0.615 s/step). With yesterday's
30B-on-24GB unlock this is the second consumer-card capability the fix restores.

## Declared deviations from the prereg

1. **torch 2.11.0+cu128** instead of the chain's usual 2.13.0+cu130: the
   qualifying host runs driver 550 (< 580 needed by cu130 wheels). axolotl's
   `torch>=2.9.1` range accepts 2.11; pre-pinning the matched cu128 triple keeps
   the pins from pulling cu130. The quantize smoke + suite gates passed on this
   stack (21 passed ThreeTier+Placement on-pod).
2. **bnb parametrize 2-line fix applied to the pod env** — required for the V
   arms to run at all (above). Applied AFTER warmup/R/fv ran and it is a no-op
   for them (their hooks are stripped by `install_expert_offload`); V arms are
   post-patch re-runs of the two crashed attempts.
3. **e4b-ssdtier at `4073d01`** vs the prereg'd `0a5eb8b`: tests-only delta (the
   `/root` EACCES portability fix in the path finder); the reader/gate code is
   byte-identical.

## Evidence

`gen1-final.tgz` (all arm logs, configs, gates, SENT) sha256
`e1dcabb8e98f961dfd7edf42d889e9854d74ecbf34d54c9adcaa6301990fa0d5` at mini
`~/gen1-evidence/`; `gen1_results.json` + `gen1_survey.jsonl` alongside (this
dir). Pod deleted, GET → 404; account at 0 pods; sweeper stopped on empty.
