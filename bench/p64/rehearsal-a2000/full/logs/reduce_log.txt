# Results -- P64: what the int4 serve lane's int8 activation step costs at decode (e4b#709) -- REHEARSAL, NOT A READING

Pre-registration: [`P64-PREREG.md`](P64-PREREG.md). Every number below is read from the run's receipts by `p64_reduce.py`.

**REHEARSAL -- NOT a reading.** A runner knob was off its registered default (the run's `summary.txt` KNOBS line).

Pack: `sha256:0fc7bc44c1dd0b5d1471d4dfcbb5281cfc7885727f74d4b9a0592673c363752c` -- NOT the licensed pack: the read is labelled so; attention pack `sha256:fc85934341c9ce34ae554a70bbd3e078bd2bea7efb15b74edc69b3b78059cfcf`; int4 expert layers 16, Int4Linear 64.

## Validity (read first)

**VALIDITY: VALID** -- determinism True, prefill-last untouched True, engagement True, same rows True


## Passes (census: decode-phase counts against the registered ones)

| text | pass | engaged | expert quant | expert dequant | attn quant | s/step | prefill chunk |
|---|---|---|---|---|---|---|---|
| wikitext | a8 | True | 65536 | 0 | 131072 | 0.06352 | 128 |
| wikitext | a16 | True | 0 | 524288 | 131072 | 0.1743 | 128 |
| wikitext | a16_all | True | 0 | 524288 | 0 | 0.17448 | 128 |
| wikitext | a8_rep | True | 65536 | 0 | 131072 | 0.06023 | 128 |
| wikitext | a8_pc64 | True | 65536 | 0 | 131072 | 0.08193 | 64 |
| wikitext | a8_pc384 | True | 65536 | 0 | 131072 | 0.05429 | 384 |
| wikitext | nf4 | True | 0 | 0 | 0 | 0.04552 | 128 |
| c4val1 | a8 | True | 65536 | 0 | 131072 | 0.06014 | 128 |
| c4val1 | a16 | True | 0 | 524288 | 131072 | 0.1739 | 128 |
| c4val1 | a16_all | True | 0 | 524288 | 0 | 0.17823 | 128 |
| c4val1 | a8_rep | True | 65536 | 0 | 131072 | 0.06102 | 128 |
| c4val1 | a8_pc64 | True | 65536 | 0 | 131072 | 0.08272 | 64 |
| c4val1 | a8_pc384 | True | 65536 | 0 | 131072 | 0.05368 | 384 |
| c4val1 | nf4 | True | 0 | 0 | 0 | 0.04109 | 128 |

## KL pairs (nats/token, fp64, full vocabulary)

| text | pair (ref || test) | positions | kl_mean | median | p95 | max | top-1 | exactly 0 |
|---|---|---|---|---|---|---|---|---|
| wikitext | a16 || a8 (G_exp, the lever) | 2048 | 0.001735 | 0.000682 | 0.006785 | 0.0939 | 0.9844 | False |
| wikitext | a16_all || a8 (G_all) | 2048 | 0.001734 | 0.000658 | 0.006317 | 0.1275 | 0.9834 | False |
| wikitext | a16_all || a16 (G_attn) | 2048 | 0.001522 | 0.000592 | 0.005452 | 0.1166 | 0.9863 | False |
| wikitext | a8 || a8_pc64 (floor) | 2048 | 0.001937 | 0.000876 | 0.006965 | 0.0951 | 0.9829 | False |
| wikitext | a8 || a8_pc384 (floor) | 2048 | 0.002042 | 0.000859 | 0.007562 | 0.1479 | 0.9790 | False |
| wikitext | a8 || a8_rep (determinism) | 2048 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| wikitext | nf4 || a8 (anchor) | 2048 | 0.069743 | 0.039877 | 0.231737 | 1.9124 | 0.8809 | False |
| wikitext | nf4 || a16 (anchor) | 2048 | 0.069007 | 0.039808 | 0.223806 | 1.7948 | 0.8818 | False |
| wikitext | nf4 || a16_all (anchor) | 2048 | 0.068569 | 0.040229 | 0.227053 | 1.6399 | 0.8823 | False |
| c4val1 | a16 || a8 (G_exp, the lever) | 2048 | 0.001550 | 0.000711 | 0.005786 | 0.0431 | 0.9819 | False |
| c4val1 | a16_all || a8 (G_all) | 2048 | 0.001724 | 0.000804 | 0.006555 | 0.0671 | 0.9849 | False |
| c4val1 | a16_all || a16 (G_attn) | 2048 | 0.001365 | 0.000603 | 0.005165 | 0.0404 | 0.9810 | False |
| c4val1 | a8 || a8_pc64 (floor) | 2048 | 0.001781 | 0.000903 | 0.006246 | 0.0630 | 0.9795 | False |
| c4val1 | a8 || a8_pc384 (floor) | 2048 | 0.001941 | 0.000944 | 0.006702 | 0.0608 | 0.9839 | False |
| c4val1 | a8 || a8_rep (determinism) | 2048 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| c4val1 | nf4 || a8 (anchor) | 2048 | 0.061867 | 0.037201 | 0.185168 | 1.8065 | 0.8677 | False |
| c4val1 | nf4 || a16 (anchor) | 2048 | 0.061618 | 0.037187 | 0.188839 | 1.7490 | 0.8682 | False |
| c4val1 | nf4 || a16_all (anchor) | 2048 | 0.061755 | 0.037724 | 0.183856 | 1.8765 | 0.8691 | False |

## NLL (nats/token over the scored positions; the ground-truth next token)

| text | a8 | a16 | a16_all | a8_rep | a8_pc64 | a8_pc384 | nf4 |
|---|---|---|---|---|---|---|---|
| wikitext | 2.04020 | 2.03902 | 2.03782 | 2.04020 | 2.03693 | 2.03848 | 1.99838 |
| c4val1 | 2.61091 | 2.61168 | 2.61108 | 2.61091 | 2.61171 | 2.60913 | 2.59142 |

## The read (registered rule; factor 2 over the measured floor)

| text | statistic | G | floor F (source) | G / F | class | dNLL (95% CI) | F_NLL | material |
|---|---|---|---|---|---|---|---|---|
| wikitext | G_exp | 0.001735 | 0.001990 (in-lane) | 0.87 | BELOW FLOOR | +0.00118 (-0.00209, +0.00433) | 0.00250 | False |
| wikitext | G_all | 0.001734 | 0.001990 (in-lane) | 0.87 | BELOW FLOOR | +0.00238 (-0.00021, +0.00525) | 0.00250 | False |
| wikitext | G_attn | 0.001522 | 0.001990 (in-lane) | 0.76 | BELOW FLOOR | +0.00120 (-0.00076, +0.00317) | 0.00250 | False |
| c4val1 | G_exp | 0.001550 | 0.001861 (in-lane) | 0.83 | BELOW FLOOR | -0.00077 (-0.00357, +0.00185) | 0.00129 | False |
| c4val1 | G_all | 0.001724 | 0.001861 (in-lane) | 0.93 | BELOW FLOOR | -0.00017 (-0.00258, +0.00213) | 0.00129 | False |
| c4val1 | G_attn | 0.001365 | 0.001861 (in-lane) | 0.73 | BELOW FLOOR | +0.00060 (-0.00133, +0.00266) | 0.00129 | False |

**VERDICT (expert int8 step, G_exp): INDISTINGUISHABLE** -- REHEARSAL, NOT A READING
- **G_all (experts + attention): INDISTINGUISHABLE**
- **G_attn (attention, experts bf16): INDISTINGUISHABLE**
- floor sources: wikitext: in-lane (mean of the measured floor pairs); c4val1: in-lane (mean of the measured floor pairs)
- the family's registered K8 |dNLL| floor, quoted beside: 0.0095 nats (METHODOLOGY section 13.1)
- the pack is NOT the licensed one: whatever the verdict, it is a reading on this box's pack, not on the licensed stack
