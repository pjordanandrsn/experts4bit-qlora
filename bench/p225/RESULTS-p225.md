# P225 results: FAIL on G3 alone, the latency control. The graded quality gates hold, and the merged serve path differs measurably from the reference

**Registration:** [PREREG-p225.md](PREREG-p225.md) (e4b#1998's successor lane; Amendments 1 to 3), read once, as
registered. **Reading:** `p225-5090-r3`, one RTX 5090, e4b `0adbba98`, $1.163. Records:
[receipts/p225-5090-r3/](receipts/p225-5090-r3/README.md).

## The question

On one RTX 5090 with Qwen3-30B-A3B @ad44e777, does a genuinely trained Loggetta attention adapter keep its measured
learned behaviour on the quantized paged serving path, with no dropped adapter target, no base substitution and no
unreported fallback?

## The answer: FAIL, as the reducer read it

The staged reducer's verdict (`run/verdict_p225.json`, re-derived byte-identical from the private receipt store by an
independent reviewer) is **FAIL**, with one reason:

> `G3 block 0 ttft_any_ms W1: |B - mean(A)| = 2.748 > band 2.388`

| step | result |
|---|---|
| the host screen | passed on the box: 47.41 Mops (floor 30), 109.2 MB/s download (floor 60), 123 GiB RAM |
| 80/80 steps | **yes**: the trainer's record holds 80 steps; the loss fell from 0.206 to 0.027 (first and last thirds) |
| the adapter gate | **passed**: 192 attention targets present, each B nonzero; provenance and fingerprint on the real file |
| LEARNED | **passed**: on the assistant positions, HF0 - REF = 5.211 nats, lower bound 5.156 ≥ 0.05 (1,296 rows) |
| disjointness | **passed**: 1,296 held-out rows, all distinct, none in the training data |
| engagement and identity | **passed**, in both PA slots: the server's merge record shows 192 of 192 targets merged (scale 2.0, max error 0.0038); `graphs` true; the base name refused (404); the served adapter id equal to the file's sha256 |
| G1 (graded on the assistant positions) | **held**: PA - REF = -0.0059, inside the margin 0.0120 |
| ratio | **held**: 1.0028 ≥ 0.95 |
| SE | **held**: 0.00159 < 0.002 |
| **G3** (W1 `itl_wall_ms`, `ttft_any_ms`, `tok_s`, under the band rule) | **block 0 outside**: `ttft_any_ms` 2.748 ms against a 2.388 ms band. Every other graded cell inside |

**This is not a failure of a graded quality gate.** Every graded quality gate held; the ungraded lines below show the
merged path differs. The FAIL comes from G3, the latency control, and G3
fails because the adapter arm was **faster**: G3 is two-sided as P224 registered it ("a merged adapter changes no shape
and no kernel; the served step is the base's"), so a shift either way fails it.

## G3, both blocks

The band is 1 % of the mean of the four A slots' W1 `ttft_any_ms`: (239.117 + 238.775 + 238.613 + 238.671) / 4 =
238.794 ms, so 2.388 ms. The raw spread between the A slots (0.342 ms at most) is smaller, so the 1 % floor governs.

| block | A1 (P0) | B (PA) | A2 (P0) | \|B - mean(A)\| | B faster by | graded |
|---|---:|---:|---:|---:|---:|---|
| 0 | 239.117 | 236.198 | 238.775 | **2.748** | 1.15 % | **outside** the 2.388 band |
| 1 | 238.613 | 236.282 | 238.671 | 2.360 | 0.99 % | inside, by 0.028 ms |

The adapter arm reached the first token faster in **both** blocks, by about 1 %, about 8 times the A1-A2 spread. The
direction is consistent; block 1 sits just inside the band. One-request decode (`itl_wall_ms`) and throughput (`tok_s`)
were inside the band in both blocks.

## The reported lines (not graded), verbatim from `run/verdict_p225.json`

P224's Amendment 5 grades G1 on the assistant positions. The reducer also reports, without grading:
- **`all_token_g1`:** ratio 1.00681, |PA - REF| 0.014463 > margin 0.011956, `would_fail`: `"|PA - REF| 0.0145 > margin
  0.0120"`. On all tokens the adapter arm's loss is **lower** than the reference's (3.4094 against 3.4239).
- **`did_all`:** -0.020441, SE 0.002307, flag set (about 8.9 SE).
- **`did_assistant`:** -0.014681, SE 0.003849, flag set (about 3.8 SE).
- **`over_application`:** ratio 1.00681 (all tokens) and 1.00282 (assistant), flags not set.
- **`cost_outside_band_not_graded`:** block 0 W1 `ttft_ms` 2.748 > 2.388; block 0 W16 `ttft_ms` and `ttft_any_ms`
  26.32 > 24.81.
- **`decode_vs_prefill`:** P0 8.57, PA 5.75, flag not set. **`merge`:** 0.118 s, max error 0.0038.

Taken together: the merged serve path (PA) scores the held-out set measurably **better** than the reference path (REF)
does, beyond its standard errors, and reaches the first token about 1 % faster. Every shift runs one way.

**An open question.** Something on the serve path differs between the merged and unmerged arms, in function and in
latency, which contradicts G3's premise that a merged adapter changes no shape and no kernel; until the cause is named
(a later $0 read), no text should say that a merged adapter changes nothing on the serve path.

## A finding about the registration (not a correction of the verdict)

P225's outcome table defines **FAIL** as "G1 or ratio outside, at SE < 0.002", and **PASS** as needing "every gate".
The staged reducer (P224's, at the bytes P225 pins) also reads a G3 miss as FAIL, and that is what ended this lane. The
table and the reducer disagree for a G3-only miss. The verdict stands as the reducer read it; the table's wording was
incomplete.

## Run ids and cost

| run | outcome | cost |
|---|---|---:|
| `p225-5090-1` | NOT_RUN: the instance stuck loading; the launcher's pre-flight tore it down | $0.004 |
| `p225-5090-2` | HOST_REFUSED: download 20.3 MB/s (floor 60) | $0.031 |
| `p225-5090-r1` | PRE_INSTALL_HALT: the boot allowance (rc 41), fixed by Amendment 2 | $0.014 |
| `p225-5090-r2` | HOST_REFUSED: download 29.6 MB/s | $0.043 |
| `p225-5090-r3` | **ANSWER: FAIL** | $1.163 |

Lane total $1.255, against a registered worst of about $21.6. One box started; no strike; two host refusals of the cap
of four. The lane ends at its first answer.

## What this does not say

- Nothing about expert adapters (D0) or their serving.
- Nothing for other models, cards or adapter shapes. One RTX 5090, Qwen3-30B-A3B, a rank-8 attention LoRA on q, k, v
  and o.
- Nothing about why the merged path differs: the cause is the open question above.
- No default changes.
