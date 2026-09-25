# Results -- P70: what rounding the fused router's decode weights to bf16 costs (e4b#726) -- REHEARSAL, NOT A READING

Pre-registration: [`P70-PREREG.md`](P70-PREREG.md). Every number below is read from the run's receipts by `p70_reduce.py`.

**REHEARSAL -- NOT a reading.** A runner knob was off its registered default (the run's `summary.txt` KNOBS line).

Pack: `sha256:0fc7bc44c1dd0b5d1471d4dfcbb5281cfc7885727f74d4b9a0592673c363752c` -- NOT the licensed pack: the read is labelled so; routers patched 16, hooked 16; MoE layers 16.

## Validity (read first)

**VALIDITY: VALID** -- determinism True, prefill-last untouched True, engagement True, same rows True, every router hooked True


## Passes (census against the registered counts; router columns are calls by row class and returned dtype)

| text | pass | engaged | chunk | cast | le64 fp32 decode | le64 bf16 decode | le64 fp32 prefill | gt64 bf16 prefill | wall s |
|---|---|---|---|---|---|---|---|---|---|
| wikitext | ep32 | True | 128 | False | 8192 | 0 | 0 | 192 | 29.6 |
| wikitext | ep16 | True | 128 | True | 0 | 8192 | 0 | 192 | 30.3 |
| wikitext | ep32_rep | True | 128 | False | 8192 | 0 | 0 | 192 | 29.5 |
| wikitext | ep32_pc96 | True | 96 | False | 8192 | 0 | 0 | 256 | 33.8 |
| wikitext | ep32_pc384 | True | 384 | False | 8192 | 0 | 0 | 64 | 21.0 |
| wikitext | ep32_pc64 | True | 64 | False | 8192 | 0 | 384 | 0 | 42.8 |
| c4val1 | ep32 | True | 128 | False | 8192 | 0 | 0 | 192 | 31.2 |
| c4val1 | ep16 | True | 128 | True | 0 | 8192 | 0 | 192 | 30.3 |
| c4val1 | ep32_rep | True | 128 | False | 8192 | 0 | 0 | 192 | 30.7 |
| c4val1 | ep32_pc96 | True | 96 | False | 8192 | 0 | 0 | 256 | 34.8 |
| c4val1 | ep32_pc384 | True | 384 | False | 8192 | 0 | 0 | 64 | 21.8 |
| c4val1 | ep32_pc64 | True | 64 | False | 8192 | 0 | 384 | 0 | 41.2 |

## KL pairs (nats/token, fp64, full vocabulary)

| text | pair (ref || test) | positions | kl_mean | median | p95 | max | top-1 | exactly 0 |
|---|---|---|---|---|---|---|---|---|
| wikitext | ep32 || ep16 (G, the cast) | 512 | 0.001307 | 0.000650 | 0.005132 | 0.0215 | 0.9941 | False |
| wikitext | ep32 || ep32_pc96 (floor) | 512 | 0.001774 | 0.000785 | 0.006615 | 0.0443 | 0.9922 | False |
| wikitext | ep32 || ep32_pc384 (floor) | 512 | 0.001865 | 0.000803 | 0.005524 | 0.1479 | 0.9922 | False |
| wikitext | ep32 || ep32_rep (determinism) | 512 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| wikitext | ep32 || ep32_pc64 (P64's floor sample; informational) | 512 | 0.001750 | 0.000779 | 0.006446 | 0.0299 | 0.9863 | False |
| c4val1 | ep32 || ep16 (G, the cast) | 512 | 0.001772 | 0.000938 | 0.006665 | 0.0217 | 0.9785 | False |
| c4val1 | ep32 || ep32_pc96 (floor) | 512 | 0.001730 | 0.000948 | 0.005708 | 0.0374 | 0.9766 | False |
| c4val1 | ep32 || ep32_pc384 (floor) | 512 | 0.002125 | 0.001059 | 0.007366 | 0.0387 | 0.9766 | False |
| c4val1 | ep32 || ep32_rep (determinism) | 512 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| c4val1 | ep32 || ep32_pc64 (P64's floor sample; informational) | 512 | 0.001835 | 0.000943 | 0.006455 | 0.0252 | 0.9707 | False |

## NLL (nats/token over the scored positions)

| text | ep32 | ep16 | ep32_rep | ep32_pc96 | ep32_pc384 | ep32_pc64 |
|---|---|---|---|---|---|---|
| wikitext | 1.95372 | 1.95715 | 1.95372 | 1.95428 | 1.95423 | 1.95488 |
| c4val1 | 3.02416 | 3.02271 | 3.02416 | 3.02455 | 3.02346 | 3.02815 |

## The read (P64's rule: factor 2 over the measured floor)

| text | G | floor F (source) | G / F | class | dNLL = NLL(ep16) - NLL(ep32) (95% CI) | F_NLL | material | P64 floor sample / F |
|---|---|---|---|---|---|---|---|---|
| wikitext | 0.001307 | 0.001819 (in-lane) | 0.72 | BELOW FLOOR | +0.00342 (+0.00170, +0.00578) | 0.00053 | False | 0.96 |
| c4val1 | 0.001772 | 0.001928 (in-lane) | 0.92 | BELOW FLOOR | -0.00145 (-0.00543, +0.00108) | 0.00055 | False | 0.95 |

**VERDICT (the cast, G): INDISTINGUISHABLE** -- REHEARSAL, NOT A READING
- floor sources: wikitext: in-lane (mean of the measured floor pairs); c4val1: in-lane (mean of the measured floor pairs)
- the family's registered K8 |dNLL| floor, quoted beside: 0.0095 nats (METHODOLOGY section 13.1)
- the pack is NOT the licensed one: whatever the verdict, it is a reading on this box's pack, not on the licensed stack
