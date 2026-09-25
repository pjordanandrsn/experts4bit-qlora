# Results -- P70: what rounding the fused router's decode weights to bf16 costs (e4b#726)

Pre-registration: [`P70-PREREG.md`](P70-PREREG.md). Every number below is read from the run's receipts by `p70_reduce.py`.

Pack: `sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42` -- the licensed pack (P55x); routers patched 48, hooked 48; MoE layers 48.

## Validity (read first)

**VALIDITY: VALID** -- determinism True, prefill-last untouched True, engagement True, same rows True, every router hooked True


## Passes (census against the registered counts; router columns are calls by row class and returned dtype)

| text | pass | engaged | chunk | cast | le64 fp32 decode | le64 bf16 decode | le64 fp32 prefill | gt64 bf16 prefill | wall s |
|---|---|---|---|---|---|---|---|---|---|
| wikitext | ep32 | True | 128 | False | 98304 | 0 | 0 | 2304 | 91.3 |
| wikitext | ep16 | True | 128 | True | 0 | 98304 | 0 | 2304 | 92.5 |
| wikitext | ep32_rep | True | 128 | False | 98304 | 0 | 0 | 2304 | 91.8 |
| wikitext | ep32_pc96 | True | 96 | False | 98304 | 0 | 0 | 3072 | 98.9 |
| wikitext | ep32_pc384 | True | 384 | False | 98304 | 0 | 0 | 768 | 74.4 |
| wikitext | ep32_pc64 | True | 64 | False | 98304 | 0 | 4608 | 0 | 111.6 |
| c4val1 | ep32 | True | 128 | False | 98304 | 0 | 0 | 2304 | 90.4 |
| c4val1 | ep16 | True | 128 | True | 0 | 98304 | 0 | 2304 | 92.1 |
| c4val1 | ep32_rep | True | 128 | False | 98304 | 0 | 0 | 2304 | 90.6 |
| c4val1 | ep32_pc96 | True | 96 | False | 98304 | 0 | 0 | 3072 | 98.0 |
| c4val1 | ep32_pc384 | True | 384 | False | 98304 | 0 | 0 | 768 | 74.3 |
| c4val1 | ep32_pc64 | True | 64 | False | 98304 | 0 | 4608 | 0 | 109.5 |

## KL pairs (nats/token, fp64, full vocabulary)

| text | pair (ref || test) | positions | kl_mean | median | p95 | max | top-1 | exactly 0 |
|---|---|---|---|---|---|---|---|---|
| wikitext | ep32 || ep16 (G, the cast) | 2048 | 0.003309 | 0.001226 | 0.011383 | 0.4165 | 0.9775 | False |
| wikitext | ep32 || ep32_pc96 (floor) | 2048 | 0.005581 | 0.001600 | 0.018501 | 0.7565 | 0.9722 | False |
| wikitext | ep32 || ep32_pc384 (floor) | 2048 | 0.005465 | 0.001693 | 0.016779 | 1.2148 | 0.9683 | False |
| wikitext | ep32 || ep32_rep (determinism) | 2048 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| wikitext | ep32 || ep32_pc64 (P64's floor sample; informational) | 2048 | 0.005578 | 0.001693 | 0.015880 | 0.8975 | 0.9702 | False |
| c4val1 | ep32 || ep16 (G, the cast) | 2048 | 0.002806 | 0.001607 | 0.008723 | 0.1774 | 0.9761 | False |
| c4val1 | ep32 || ep32_pc96 (floor) | 2048 | 0.003002 | 0.001840 | 0.009630 | 0.0737 | 0.9678 | False |
| c4val1 | ep32 || ep32_pc384 (floor) | 2048 | 0.003089 | 0.001876 | 0.010056 | 0.0884 | 0.9717 | False |
| c4val1 | ep32 || ep32_rep (determinism) | 2048 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| c4val1 | ep32 || ep32_pc64 (P64's floor sample; informational) | 2048 | 0.003300 | 0.001832 | 0.010235 | 0.2026 | 0.9707 | False |

## NLL (nats/token over the scored positions)

| text | ep32 | ep16 | ep32_rep | ep32_pc96 | ep32_pc384 | ep32_pc64 |
|---|---|---|---|---|---|---|
| wikitext | 2.08339 | 2.08436 | 2.08339 | 2.07796 | 2.08223 | 2.08250 |
| c4val1 | 2.73485 | 2.73701 | 2.73485 | 2.73405 | 2.73206 | 2.73474 |

## The read (P64's rule: factor 2 over the measured floor)

| text | G | floor F (source) | G / F | class | dNLL = NLL(ep16) - NLL(ep32) (95% CI) | F_NLL | material | P64 floor sample / F |
|---|---|---|---|---|---|---|---|---|
| wikitext | 0.003309 | 0.005523 (in-lane) | 0.60 | BELOW FLOOR | +0.00096 (-0.00401, +0.00589) | 0.00330 | False | 1.01 |
| c4val1 | 0.002806 | 0.003045 (in-lane) | 0.92 | BELOW FLOOR | +0.00216 (-0.00111, +0.00566) | 0.00180 | False | 1.08 |

**VERDICT (the cast, G): INDISTINGUISHABLE**
- floor sources: wikitext: in-lane (mean of the measured floor pairs); c4val1: in-lane (mean of the measured floor pairs)
- the family's registered K8 |dNLL| floor, quoted beside: 0.0095 nats (METHODOLOGY section 13.1)
