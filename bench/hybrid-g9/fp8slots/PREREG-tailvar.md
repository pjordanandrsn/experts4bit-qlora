# PREREG — the tail-rate uncertainty model (input (a), re-scoped)

Registered before any measurement. This is the re-scoped input (a)
from [RESULTS-coroute.md](RESULTS-coroute.md):

> not a co-routing correction but a **tail-rate uncertainty model** —
> any slot-value claim must carry window-to-window tail variance …
> or estimate rates online.

The claim made precise: cross-window dispersion of bracket touch-mass
has a STRUCTURE (it grows down the mass ranking) and per-bracket
envelopes fitted on one set of content windows COVER held-out windows.
If they do, the envelopes are the citable uncertainty the slot-pricing
line needs; if they don't, offline tail pricing requires online
estimation and no third offline model will be attempted.

## Design

* **Windows:** 10 disjoint wikitext-test slices, `offset = i × 29000`,
  `span = 28000` (1k-token guard gaps), i = 0…9. Chronological split:
  w0–w4 fit, w5–w9 held out — the harder, honest transfer (adjacent
  windows are never split across sets).
* **Runs:** one serving run per window, identical config: Qwen3-30B-A3B,
  B = 16, prompt 512, gen 64 (~75+ decode steps), chunk 512, pool 32 /
  torch-intraop 8, `vram_gb = 10`, `dram_gb = 60`, **uniform placement**
  — the co-routing cycle established that touch counters are
  placement-independent, so no profile and no ladder are needed. Fresh
  subprocess per window ([tailvar_sweep.py](tailvar_sweep.py)).
* **Brackets (frozen data, not recomputed):** the four ladder brackets
  from the committed co-routing receipts
  ([receipts-coroute/](receipts-coroute/) `p1_v*.amort.json` manifest
  sets): b1 = 7→9 GB (809 experts) … b4 = 10.75→11.5 GB (304).
* **Quantity:** `M_b(w) = Σ_{e∈b} touch_e / steps` — the bracket's
  expected ΔU contribution on window w. Binomial noise per cell
  `σ = sqrt(Σ P̂(1−P̂)/S)` is ~3% relative at S ≈ 75, far below the
  ±12–145% cross-window effects the co-routing cycle measured.

## Bars ([tailvar_verdict.py](tailvar_verdict.py), self-tested on three
synthetic worlds: rank-rising dispersion certifies; stationary world is
UNINFORMATIVE; flat dispersion fails H1)

* **H1 (structure):** fit-set `cv(b4) ≥ 2 × cv(b1)`, and the
  argmax-cv and argmin-cv brackets are the SAME brackets on the
  held-out set.
* **H2 (the deliverable):** `|M_b(w) − mean_fit(M_b)| ≤ 2·cv_b·mean +
  3σ_binom` covers **≥ 18 of 20** held-out (window × bracket) cells.
  Only a pass makes the `u_b = 2·cv_b` table citable.
* **H3 (control — could it have failed):** binomial-only envelopes
  must cover **< 70%** of held-out cells; otherwise the windows are not
  dispersed beyond sampling noise and the run is UNINFORMATIVE.
* NVMe tier must stay empty in every run (assert; abort).

**ENVELOPES-CERTIFIED** = H1 ∧ H2 with H3 informative. Anything else is
REFUTED (or UNINFORMATIVE), and per the co-routing disposition the
offline tail-pricing line CLOSES in favor of online rate estimation —
no third offline model.

## Hard stop

One box, one scored run of the 10-window set. Environment pin as
before (accelerate in, torchvision/torchaudio out). Cost ≈ $0.60.
