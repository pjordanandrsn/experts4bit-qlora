### Read: TC1 amendment 50 -- under `auto` the field recipe never buckets (P123-P125 HELD); `auto` is licensed as grouped-nf4-gemm's default

- `tc1-5090-107` ($0.79, a quiet Threadripper PRO 3955WX): `NF4_QLORA_PAD_BUCKETS=auto` (grouped-nf4-gemm#491) against `0` at the field
  recipe. Every `auto` arm resolved `auto` and made no bucketed call (49,152 single-block calls each); held-out within 0.0022; matched
  peak -0.012 GB. Speed reported: the same within the draws on both arms.
- With amendment 48's re-ask HELD, amendment 50's rule licenses `auto` as grouped-nf4-gemm's default; the flip is a grouped-nf4-gemm
  PR and release, not yet made (row
  `e4b.train.pad-buckets.auto.default-decision.5090.2026-10-06`); STATUS says so.
