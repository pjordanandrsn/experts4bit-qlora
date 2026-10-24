### D0s registered: the served read of the merged NF4 arena, both D0 fixtures on one card (USD 0 to register)

- `bench/d0s/PREREG-d0s.md`, registered before any data. The paged server serves a merged arena with its own adapter
  on D0's 64 held-out rows, for both fixtures and both writer sources, against D0's registered REF.
- Gated by the spec's rule: retention CI lower bound ≥ 0.95, |gap| within an absolute margin fixed from D0's record
  (two-sided: an R above 1.05 fails as BEYOND_REF, an artifact to explain), an argmax canary in pooled exact counts,
  cost slower-only within the self-pair band floored at 3 %, and an engagement check that VOIDs an arm served on the
  wrong path. A shakedown runs first; at most two reading boxes start, lane worst USD 11.58.
