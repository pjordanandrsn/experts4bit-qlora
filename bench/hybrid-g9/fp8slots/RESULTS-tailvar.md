# RESULTS — the tail-rate uncertainty model: ENVELOPES-CERTIFIED

Scored under [PREREG-tailvar.md](PREREG-tailvar.md) by the committed
[tailvar_verdict.py](tailvar_verdict.py); receipts in
[receipts-tailvar/](receipts-tailvar/). Box: the reference host class
(EPYC 9655 + RTX 5090, G0 142.2% PROCEED). Ten disjoint 28k-token
windows, one identical serving run each (uniform placement — counters
are placement-independent), chronological 5/5 split. NVMe stayed empty
in all runs. Cycle ≈ $0.77, box destroyed, zero instances.

## Verdict — the first CERTIFIED in the fp8slots line

| bracket | experts | fit mean M_b | fit cv | held-out cv |
|---|---|---|---|---|
| b1 (7→9 GB) | 809 | 241.7 | 11.3% | 9.0% |
| b2 (9→10) | 405 | 67.4 | 22.9% | 15.1% |
| b3 (10→10.75) | 303 | 31.8 | 26.1% | 17.9% |
| b4 (10.75→11.5) | 304 | 21.0 | 28.2% | 20.9% |

* **H1 PASS** — dispersion is structured: cv rises monotonically down
  the mass ranking on BOTH sets; tail cv 28.2% ≥ 2× deep 11.3%; the
  argmax- and argmin-cv brackets replicate on held-out.
* **H2 PASS, 20/20** — the `2·cv` envelopes fitted on windows 0–4
  covered every held-out (window × bracket) cell (bar ≥ 18/20).
  Held-out cvs run slightly below fit cvs, so the envelopes are
  conservative rather than lucky.
* **H3 informative** — binomial-only envelopes covered 4/20: the
  dispersion is content variation, ~7× beyond sampling noise.

## The citable deliverable

Any offline slot-value claim priced from a single profile window must
carry, per bracket of the mass ranking:

**u_b = 2·cv_b = [22.5%, 45.7%, 52.3%, 56.3%]** (b1 deep-DRAM → b4
tail, at 28k-token windows, S ≈ 78 decode steps, B = 16,
Qwen3-30B-A3B on wikitext-class text).

Concretely: a bracket's expected ΔU contribution predicted from one
window is trustworthy to ~±22% deep in the DRAM tail and only to
~±56% at the margin where slot-pricing decisions live. This is the
uncertainty the refuted slot-value pricing lacked, and the number any
future registration must carry (or beat with online estimation).

## Reconciliation note (recorded, not scored)

The co-routing cycle's single window pair showed a +145% tail miss —
outside even these envelopes. Two design differences plausibly explain
the excess: its windows were 5× longer (140k tokens, more content
averaging pulls the other way ⇒ not the cause) and — more likely —
its calibration ran gen-128 vs the sweep's gen-48, so decode-position
routing distributions differed between the windows being compared. The
tailvar design holds gen-tokens identical across all ten windows and
still finds 21–28% tail cv: content non-stationarity is real after
controlling the position confound, and the co-routing cycle's
magnitude was probably part position effect. A registration that needs
the position axis (rates vs decode depth) would sweep gen-tokens; none
is queued.

## Disposition

Input (a) of the slot-pricing re-derivation is **closed with a
certified artifact**: the per-bracket uncertainty table above. Inputs
(b) regime-split conversion constants and (c) reference-host
re-derivation remain open; any revived slot-value registration must
propagate u_b through its bars (which at the tail means effects must
exceed ~±56% to certify offline — a strong argument for pairing (b)
with online rate estimation instead).
