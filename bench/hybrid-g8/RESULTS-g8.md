# G8 — batched hybrid expert dispatch: the amortization law, measured

Instrument: OLMoE-1B-7B-0924 (16 layers, **E = 64**, **k = 8**) on the
in-house dev box (RTX A2000 12 GB, Xeon W-1250 12c **AVX2**), NF4 arena,
hybrid tier engaged on all 16 MoE modules, box's own calibration blob
(`receipts/calib-gpudev.json`: VRAM triad 255.3 GB/s, grouped scatter
25.0 GB/s). Placement solved under the **batched** cost law.

## Verdicts

| clause | result |
|---|---|
| unique-expert amortization within 10% of `factor(B)` | **MISS as written** — up to **73%** below the closed form in the decode regime |
| …of the **general law** the closed form specializes | **PASS** — worst **5.5%** (decode), **1.0%** (token-rich) |
| GPU/CPU completion within 20% at B=8, B=16 | **NOT MEASURED** on this instrument — see below; not claimed either way |
| control: single-token factor = 1.0 exactly | **PASS** (1.0000, computed by the same path as the results) |

## The law is falsified as stated — in the favorable direction

The gate's `factor(B) = E(1-(1-k/E)^B)/(B·k)` assumes **uniform routing**.
Measured against a real router it is wrong by up to 73%, and always in
the same direction: real dispatch amortizes ~3.5× BETTER than uniform
predicts, because routing is concentrated. At 64 tokens OLMoE touches
~18 of 64 experts per layer; uniform predicts ~64.

| B (tokens) | measured | closed form | Δ | general law | Δ |
|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | +0.0% | 1.0000 | +0.0% |
| 2 | 0.5664 | 0.9375 | −39.6% | 0.5755 | −1.6% |
| 4 | 0.3398 | 0.8276 | −58.9% | 0.3389 | +0.3% |
| 8 | 0.1943 | 0.6564 | −70.4% | 0.2004 | −3.0% |
| 16 | 0.1196 | 0.4410 | −72.9% | 0.1160 | +3.2% |
| 32 | 0.0679 | 0.2465 | −72.5% | 0.0643 | +5.5% |
| 64 | 0.0354 | 0.1250 | −71.7% | 0.0341 | +3.7% |

**The general law is `Σ_e [1 − (1−p_e)^B]`** over measured per-expert
routing probabilities; the gate's form is that with every `p_e` forced to
`k/E`. It predicts the measured curve within **5.5% worst case** across
the whole sweep — inside the gate's 10% band — and the fit is computed
from a *separate* profiling pass, never fit to the data it predicts. So
**the dispatch machinery is correct and the routing model was wrong**:
the amortization G8 asks for is present, and more of it than the gate
assumed.

The closed form is not useless — it is accurate where routing skew stops
mattering. At 256 tokens/step it fits within **1.5%** (all experts get
touched, both laws converge to `E/(tokens·k)`). It fails specifically in
the **decode regime**, which is where serving actually lives. That is the
fitted curve the directive asked for, and its shape is the claim: skew
buys the batch multiplier earlier than uniform routing would.

## Why balance is reported as NOT MEASURED

Both buses ran 20–45× below their calibrated bandwidths at decode-shaped
steps (DRAM 0.36–0.59 GB/s against a 25.0 GB/s calibration; VRAM 13–15
against 255). At 8–16 tokens across 16 layers of a 1B-active model, per
call fixed costs — python dispatch, index_select, launch latency —
dominate the bus entirely. This is the same trap G6 documented ("tiny
models make the python constant dominate; G6-class gates must be measured
on serving-class step times"), and a completion-time ratio measured there
is a ratio of overheads, not of buses.

Two further facts make this box the wrong instrument for clause 2
specifically, and both are recorded rather than worked around:

* **Capacity, not balance, decided the split** at plausible budgets. A
  VRAM sweep shows the solver responding correctly as capacity opens —
  balance ratio 0.048 → 0.103 → 0.305 → 0.52 at 1.2 / 2.0 / 2.8 / 3.3 GB
  — with the 3.3 GB split (933 VRAM / 91 DRAM) matching the calibrated
  10.2:1 bandwidth ratio almost exactly. The solver did what it was
  asked; the residual imbalance is not placement.
* **This host is AVX2**, not the many-channel AVX-512 host the DRAM tier
  is designed around (Phase 2 measured 55–82% of triad there).

Clause 2 therefore needs a serving-class model on an AVX-512 host. It is
left explicitly unmeasured rather than reported from this instrument.

## Regime crossover (the directive's requested test)

Sweeping tokens/step at fixed batch shows the DRAM tier leaving the
bandwidth-bound regime exactly as predicted: achieved DRAM GB/s falls
0.51 → 0.11 → 0.06 as tokens-per-expert rises from ~1 to ~32 to ~64,
i.e. arithmetic intensity `tokens_per_expert × 3.6 flop/byte` crosses the
~29 flop/byte machine ceiling near 8 tokens/expert and the path becomes
compute-bound. Decode (1–4 tokens/expert through B≈64) stays
bandwidth-bound, which is the regime the CPU kernels are tuned for. No
compute-side optimization was done, per the directive.

## What shipped with this measurement

- `placement.expected_weight_reads` / `amortization_factor` /
  `routing_probabilities`, and `solve_placement(batch=, top_k=)` — the
  solver now balances on unique weight reads. `batch=1` reproduces
  Phase 3's placement exactly (pinned by test).
- The dispatcher no longer splits oversize groups; the CPU kernel chunks
  a group across its 8-row register blocking internally, so an expert's
  weights cross the DRAM bus once regardless of how many tokens routed
  to it (gnf4 side, bit-identical to the split it replaces).
- Per-tier amortization instrument on the hybrid tier: unique experts,
  activations, post-split group count (the split tax stays visible), a
  per-expert routing histogram, and per-bus wall time from **its own
  probe** — CPU on the host clock, GPU bracketed by CUDA events. Off by
  default, structurally free when off.

## Receipts

`receipts/g8_decode_olmoe.json` (full sweep incl. per-tier uniques,
per-bus times, general-law predictions), `receipts/calib-gpudev.json`,
`receipts/placement-batch16.json`.
