# P234 results: `E4B_NF4_SMALLM_BW=1` decodes Mixtral-8x7B's default NF4 server 2.12× faster at two requests and 1.76× at four, within the quality bars (LICENSED)

**Registration:** [PREREG-p234.md](PREREG-p234.md) (e4b#2114), read once, as registered. Lever B, step B3 of the decode
roadmap, re-registered after P231.
**Reading:** `p234-5090-1`, one RTX 5090, e4b `47bf1ed2`, grouped-nf4-gemm v0.50.0, $1.023.
Records: [receipts/p234-5090-1/](receipts/p234-5090-1/README.md). The verdict re-derives byte-identically from the
records, by `p234_reduce.py` at `47bf1ed2`.

## The question

`E4B_NF4_SMALLM_BW=1` sends the default NF4 server's 2-to-8-token decode calls through grouped-nf4-gemm's K33 bandwidth
GEMV. Each (token, slot) row gets its own group, instead of going through the device tile table and K25. Does it decode
Mixtral-8x7B faster at 2 and 4 requests, at no measurable quality cost?

P231 asked the same question and read FUNCTION_FAIL. B1's tokens moved on its sixteen-request workload, whose admission
ramp runs the buckets the switch acts on. P234 rules token identity on W1 alone, where every decode step is bucket 1 and
the switch takes no call. Sixteen requests keep P231's 0.99 line as a speed regression bar.

## The answer: LICENSED

| step | result |
|---|---|
| VOID | no. Every bucket (1, 2, 4, 8, 16) captured as a graph in every arm, and engagement read as registered per bucket. All three premises held on the card: the served suite 7 passed, the kernel premise 20 passed, the rows premise 6 cases with 0 mismatches. Every record is present, and P115's mutant failed the quality bar |
| FUNCTION_FAIL | no. **B1's tokens equal B0's on W1** at both lengths. Each arm repeats itself exactly on every workload, and every timed rep digests the same |
| NOISY | no. The self-pairs read 0.99031 to 0.9991, inside [0.985, 1.015] (W1, W2, W4) and [0.96, 1.04] (W16: 0.99956, 0.999) |
| QUALITY_FAIL | no. See the table below |
| SLOWER | no. **g2 = 2.12452, g4 = 1.76481** (bar 1.03); the inert W1 check 0.99182 (bar 0.99); the W16 regression bar 1.00013 (bar 0.99) |
| **LICENSED** | **yes** |

Decode rate (decode-only timing; tok/s per arm in the receipt's README):

| workload | g = min(B1a/B0a, B1b/B0b) | the two pairs |
|---|---:|---|
| W1 (inert check) | 0.99182 | 0.99182, 1.00032 |
| **W2** | **2.12452** | 2.12452, 2.12900 |
| **W4** | **1.76481** | 1.76481, 1.76492 |
| W8 (reported only) | 1.22138 | 1.22138, 1.22226 |
| W16 (regression bar) | 1.00013 | 1.00069, 1.00013 |

At two requests the server decodes 155.7 tok/s with the switch against 73.2 without. At four requests it decodes 189.4
against 107.3.

Quality at group 4 (P115's teacher-forced instrument, 24 windows per text, against P115's floor at P110's bars):

| text | ON bias | bias bar | ON spread | spread bar | K8 perplexity, R to ON |
|---|---:|---:|---:|---:|---|
| wikitext | -0.00144 | 0.01254 | 0.00763 | 0.01593 | 4.4973 to 4.4908 (-0.0065; budget 0.05) |
| c4val1 | 0.00014 | 0.01146 | 0.00621 | 0.01424 | 7.0168 to 7.0178 (reported) |

**Reported, never a rung: B1's tokens against B0's at W2 to W16.** The switch acts in those steps, so the streams part.

| workload | 32 tokens | 160 tokens |
|---|---|---|
| W2 | none | rows 0 and 1 |
| W4 | rows 2 and 3 | rows 0 to 3 |
| W8 | rows 3 and 5 | rows 0, 1 and 3 to 7 |
| W16 | rows 3, 4, 6, 10 and 13 | rows 0 to 4, 6 and 7 |

These rows are identical across the two pairs. W16's are the same rows P231 found.

**Consequence, as registered:**
- `E4B_NF4_SMALLM_BW` becomes the default for Mixtral's NF4 store at 2 to 4 tokens (buckets 2 and 4). That change lands
  in its own e4b PR, which cites this read: one family, this card class, the all-resident collapse.
- Calls of 5 to 8 tokens (bucket 8) are not ruled here and stay opt-in.
- The register row `e4b.serve.p234.nf4-smallm-bw.mixtral-nf4.5090.2026-10-24` carries g2 and g4.
- The families inventory notes the read.

The lane ends at its first answer.

## Predictions

| # | prediction | held? |
|---|---|---|
| Q1 | the premise holds on the card (all three parts), and every arm's per-bucket engagement holds | **yes** |
| Q2 | the self-pairs read inside their bands; B1's tokens equal B0's on W1; B1's tokens differ from B0's on W16 in some rows (reported) | **yes** |
| Q3 | g2 between 1.9 and 2.3 and g4 between 1.55 and 1.9 | **yes** (2.12452, 1.76481) |
| Q4 | g1 and g16 between 0.99 and 1.01; W8's g above 1 | **yes** (0.99182, 1.00013; W8 1.22138) |
| Q5 | ON passes P110's bars on both texts | **yes** |
| Q6 | LICENSED (about 85 %) | LICENSED |

## What this does not say

- Nothing about calls of 5 to 8 tokens beyond the reported W8 ratio. Bucket 8 stays opt-in.
- Nothing about other families, cards or stores, or about the multi-row kernel (B2).
- Nothing about token identity at 2 to 16 requests: the switch changes those streams. The quality rung is what rules
  on them, at group 4. The kernel and rows premises tie buckets 2 and 8 to the same per-row arithmetic.
