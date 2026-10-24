# P231 results: `E4B_NF4_SMALLM_BW=1` changes the W16 control's tokens on Mixtral-8x7B's default NF4 server (FUNCTION_FAIL)

**Registration:** [PREREG-p231.md](PREREG-p231.md) (e4b#2051, Amendments 1 to 3), read once, as registered. Lever B,
step B3, of the decode roadmap.
**Reading:** `p231-5090-3`, one RTX 5090, e4b `fa40c752`, grouped-nf4-gemm v0.50.0, $1.086.
Records: [receipts/p231-5090-3/](receipts/p231-5090-3/README.md).

## The question

`E4B_NF4_SMALLM_BW=1` sends the default NF4 server's 2-to-8-token decode calls through grouped-nf4-gemm's K33
bandwidth GEMV, one group per (token, slot) row, instead of the device tile table and K25. Does it decode Mixtral-8x7B
faster at 2 and 4 requests, with W1 and W16 as inert controls, at no measurable quality cost?

## The answer: FUNCTION_FAIL

| step | result |
|---|---|
| VOID | no. Every bucket captured as a graph in every arm, the premise held on the card (the served suite, 7 passed; the rows premise, 6 cases, 0 mismatches, K33 at T = 1, 2 and 4), and every record is present |
| **FUNCTION_FAIL** | **yes.** B1's tokens differ from B0's on the **W16 control**, in both pairs: rows 3, 4, 6, 10 and 13 at 32 tokens, and rows 0, 1, 2, 3, 4, 6 and 7 at 160 tokens |

The registration reads FUNCTION_FAIL first, so the rest of the rule (the gain, the controls' timing and quality) is
never graded. The box itself is deterministic: on every workload at both lengths, B0b's tokens equal B0a's and B1b's
equal B1a's. W1's tokens are equal across all four arms.

**Consequence, as registered:** "the switch stays opt-in until it is explained." No default moves.

## What was measured, reported and never graded

Decode rate (decode-only timing; the tok/s per arm are in the receipt's README):

| workload | g = min(B1a/B0a, B1b/B0b) | the two pairs |
|---|---:|---|
| W1 (control) | 0.99831 | 0.99952, 0.99831 |
| **W2** | **2.09555** | 2.09642, 2.09555 |
| **W4** | **1.72902** | 1.72902, 1.73347 |
| W8 (reported only) | 1.20748 | 1.21166, 1.20748 |
| W16 (control) | 1.00063 | 1.00232, 1.00063 |

At two requests the server decodes 154 tok/s with the switch against 73.5 without; at four, 186 against 108. The
self-pairs read 0.99746 to 1.00468, inside their bands.

Quality at group 4 (P115's instrument, 24 windows per text), ON (B1) beside the reference draws:

| text | ON bias | ON argmax agreement | the chunk draw's bias / argmax |
|---|---:|---:|---|
| wikitext | -0.00144 | 0.97461 | 0.00254 / 0.97786 |
| c4val1 | 0.00014 | 0.97298 | -0.00146 / 0.97461 |

The tokens also differ between B1 and B0 at W2 to W8, where the switch acts by design. The rule does not read those
streams.

## Predictions

| # | prediction | held? |
|---|---|---|
| Q1 | the premise holds on the card, and every arm's per-bucket engagement holds | **yes** |
| Q2 | the self-pairs read inside their bands; B1's tokens equal B0's on W1 and W16 | **no**: the self-pairs held and W1 held, but W16's tokens differ |
| Q3 | g2 between 1.6 and 2.6, g4 between 1.4 and 2.2 | **yes** (2.096, 1.729), reported only |
| Q4 | g1 and g16 between 0.99 and 1.01; W8's g above 1 | **yes** (0.998, 1.001; W8 1.207), reported only |
| Q5 | ON passes P110's bars on both texts | not graded: FUNCTION_FAIL is read first |
| Q6 | the verdict | FUNCTION_FAIL, which the registration gave about 1 % |

## Why the W16 control moved: one reading refuted, one untested

- **Refuted: rows ending early.** If a W16 row ended before the others, later steps would decode fewer than 16 rows, in
  buckets where the switch acts. The arm records refute this: every W16 row runs its full 32 and 160 tokens in both
  arms.
- **Untested: the prefill ramp.** The first differing step is early for some rows: step 5 for row 3, 15 for row 4,
  then 28 to 49 for the others. The server admits the sixteen requests as their prefills complete, so the first decode
  steps run with fewer than sixteen active rows, in buckets 2 to 8, where B1 routes K33 singleton groups and B0 runs
  K25. The two are different arithmetic within the kernels' tolerance, so the streams can part there and then drift.
  - The ramp itself is in this read's records. In every W16 pass of all four arms, at both lengths, the rows' first
    tokens arrive one at a time, 0.61 to 0.66 s apart: row 0 at 0.61 s, row 15 at 10.2 to 10.3 s (`ttfts` in each
    arm record).
  - P223's census of the same workload, on the same server shape (`decode_calls` in its arm records, at an earlier
    commit), counts exactly two decode calls per pass at each live-row count from 1 to 15, beside the full-batch
    steps. That is one decode step between successive admissions, plus the matching ramp down. Under that schedule,
    row 3's step 5 runs at about 8 or 9 live rows, at the edge of the switched buckets.
  - What stays untested is that the divergence begins in those steps. The arm records count graph replays per bucket
    across all workloads together, so they cannot place W16's steps by bucket. A per-step trace of W16's buckets would
    settle it.

Either way, the registration assumed W16 never reaches the buckets the switch acts on, and in the served schedule it
does (or something else moved its tokens). That premise, not the switch's arithmetic, is what the read found.

## What this does not say

- Nothing about B1's speed as a licensed result. The gains above are reported because the rule never reached them.
- Nothing about other models, cards or stores, or about the multi-row kernel (B2).
- A follow-up, for example B1 read against a control made schedule-invariant (as P223 measured), is a new
  registration, not this lane.
