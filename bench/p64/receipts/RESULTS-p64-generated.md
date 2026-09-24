# Results -- P64: what the int4 serve lane's int8 activation step costs at decode (e4b#709)

Pre-registration: [`P64-PREREG.md`](P64-PREREG.md). Every number below is read from the run's receipts by `p64_reduce.py`.

Pack: `sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42` -- the licensed pack (P55x); attention pack `sha256:1824cbebe3292e0a07752b516ec5dcc136d10b655557b06319bd5c70c3ee23fa`; int4 expert layers 48, Int4Linear 192.

## Validity (read first)

**VALIDITY: VALID** -- determinism True, prefill-last untouched True, engagement True, same rows True


## Passes (census: decode-phase counts against the registered ones)

| text | pass | engaged | expert quant | expert dequant | attn quant | s/step | prefill chunk |
|---|---|---|---|---|---|---|---|
| wikitext | a8 | True | 196608 | 0 | 393216 | 0.11865 | 128 |
| wikitext | a16 | True | 0 | 1572864 | 393216 | 0.30253 | 128 |
| wikitext | a16_all | SKIPPED | | | | | |
| wikitext | a8_rep | True | 196608 | 0 | 393216 | 0.12034 | 128 |
| wikitext | a8_pc64 | True | 196608 | 0 | 393216 | 0.15376 | 64 |
| wikitext | a8_pc384 | SKIPPED | | | | | |
| wikitext | nf4 | MISSING | | | | | |
| c4val1 | a8 | True | 196608 | 0 | 393216 | 0.11752 | 128 |
| c4val1 | a16 | True | 0 | 1572864 | 393216 | 0.29013 | 128 |
| c4val1 | a16_all | SKIPPED | | | | | |
| c4val1 | a8_rep | True | 196608 | 0 | 393216 | 0.11735 | 128 |
| c4val1 | a8_pc64 | True | 196608 | 0 | 393216 | 0.14303 | 64 |
| c4val1 | a8_pc384 | SKIPPED | | | | | |
| c4val1 | nf4 | MISSING | | | | | |

## KL pairs (nats/token, fp64, full vocabulary)

| text | pair (ref || test) | positions | kl_mean | median | p95 | max | top-1 | exactly 0 |
|---|---|---|---|---|---|---|---|---|
| wikitext | a16 || a8 (G_exp, the lever) | 2048 | 0.003392 | 0.001381 | 0.010891 | 0.1641 | 0.9731 | False |
| wikitext | a16_all || a8 (G_all) | -- | MISSING | | | | | |
| wikitext | a16_all || a16 (G_attn) | -- | MISSING | | | | | |
| wikitext | a8 || a8_pc64 (floor) | 2048 | 0.005578 | 0.001693 | 0.015880 | 0.8975 | 0.9702 | False |
| wikitext | a8 || a8_pc384 (floor) | -- | MISSING | | | | | |
| wikitext | a8 || a8_rep (determinism) | 2048 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| wikitext | nf4 || a8 (anchor) | -- | MISSING | | | | | |
| wikitext | nf4 || a16 (anchor) | -- | MISSING | | | | | |
| wikitext | nf4 || a16_all (anchor) | -- | MISSING | | | | | |
| c4val1 | a16 || a8 (G_exp, the lever) | 2048 | 0.002527 | 0.001588 | 0.007673 | 0.0924 | 0.9717 | False |
| c4val1 | a16_all || a8 (G_all) | -- | MISSING | | | | | |
| c4val1 | a16_all || a16 (G_attn) | -- | MISSING | | | | | |
| c4val1 | a8 || a8_pc64 (floor) | 2048 | 0.003300 | 0.001832 | 0.010235 | 0.2026 | 0.9707 | False |
| c4val1 | a8 || a8_pc384 (floor) | -- | MISSING | | | | | |
| c4val1 | a8 || a8_rep (determinism) | 2048 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| c4val1 | nf4 || a8 (anchor) | -- | MISSING | | | | | |
| c4val1 | nf4 || a16 (anchor) | -- | MISSING | | | | | |
| c4val1 | nf4 || a16_all (anchor) | -- | MISSING | | | | | |

## NLL (nats/token over the scored positions; the ground-truth next token)

| text | a8 | a16 | a16_all | a8_rep | a8_pc64 | a8_pc384 | nf4 |
|---|---|---|---|---|---|---|---|
| wikitext | 2.08339 | 2.08446 | -- | 2.08339 | 2.08250 | -- | -- |
| c4val1 | 2.73485 | 2.73665 | -- | 2.73485 | 2.73474 | -- | -- |

## The read (registered rule; factor 2 over the measured floor)

| text | statistic | G | floor F (source) | G / F | class | dNLL (95% CI) | F_NLL | material |
|---|---|---|---|---|---|---|---|---|
| wikitext | G_exp | 0.003392 | 0.005578 (in-lane) | 0.61 | BELOW FLOOR | -0.00107 (-0.00707, +0.00519) | 0.00089 | False |
| wikitext | G_all | UNREAD | | | | | | |
| wikitext | G_attn | UNREAD | | | | | | |
| c4val1 | G_exp | 0.002527 | 0.003300 (in-lane) | 0.77 | BELOW FLOOR | -0.00180 (-0.00610, +0.00181) | 0.00011 | False |
| c4val1 | G_all | UNREAD | | | | | | |
| c4val1 | G_attn | UNREAD | | | | | | |

**VERDICT (expert int8 step, G_exp): INDISTINGUISHABLE**
- **G_all (experts + attention): UNREAD (a text is missing)**
- **G_attn (attention, experts bf16): UNREAD (a text is missing)**
- floor sources: wikitext: in-lane (mean of the measured floor pairs); c4val1: in-lane (mean of the measured floor pairs)
- the family's registered K8 |dNLL| floor, quoted beside: 0.0095 nats (METHODOLOGY section 13.1)
