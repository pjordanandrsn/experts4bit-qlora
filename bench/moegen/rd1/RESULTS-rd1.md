# RD1 read: grouped-nf4-gemm's frozen-expert GEMM routes, per call, on one RTX 5090

*Work item experts4bit-qlora#1049. The registration is [`RD1-PREREG.md`](RD1-PREREG.md), with amendments 1–3. Read on
2026-10-05 from `rd1-rp-5090-2`. The full table, the bar and the predictions are in [`RESULTS-rd1.txt`](RESULTS-rd1.txt).*

**Reproduce:**

```
python bench/moegen/rd1/rd_table.py bench/moegen/rd1/rd1-rp-5090-2/receipts/rd1.json
```

That command prints `RESULTS-rd1.txt` byte for byte. Every line the box's own reducer wrote
([`rd1-rp-5090-2/RESULTS-rd1.txt`](rd1-rp-5090-2/RESULTS-rd1.txt)) is in it, in order.
`tests/test_rd1_lane.py::test_the_committed_reading_rederives_from_its_receipt` holds both.

## Verdict

**DECODED HELD (4/7). V3 NOT HELD (0/7). DECISION: an opt-in decoded route in grouped-nf4-gemm, then a TC1 full-step A/B.**

- **The box.**
  - RunPod Secure RTX 5090, machine `t1upvpp9ld3m`, pod `mtjtv4yp67jjlg`, AMD EPYC 7543, driver 570.195.03.
  - e4b `298224fd`, grouped-nf4-gemm `9622144c` (tripwire OK), torch 2.8.0+cu128, Triton 3.4.0.
  - $0.12. That is adertha's estimate: RunPod returned no billing record for the pod.
- **The licence (amendment 3).**
  - The pre-probe anchor was refused on attempt 1: `launch.self_pair` 1.148 > 1.03, at load1 11.1.
  - It passed on attempt 2 (`pcie-full/launch-fast`).
  - **The post-probe anchor passed (rc 0, `pcie-full/launch-fast`, load1 18.2)**, so the reading is licensed by the box holding
    its class through the probe.
  - Host load is informational: median 16.3 over the probe, max 25.8, and median 11.0 in the 60 s before it.
- **The gate.** Every arm passed the fp32-reference gate on every call of all 32 cells (P0).

### The bar, at both lengths

On the skewed draw, decoded_cap's event time over min(v1, v3, dense), gated:

| family | rows per expert at 512 (seq × k / E) | seq 512 | seq 2048 | both (the bar) |
|---|---|---|---|---|
| `olmoe` | 64 | 0.79 | 0.39 | pass |
| `lfm2` | 64 | 0.74 | 0.37 | pass |
| `ernie` | 48 | 0.81 | 0.40 | pass |
| `graniteh` | 48 | 0.79 | 0.40 | pass |
| `qwen3` | 32 | 1.05 | 0.53 | fail at 512 |
| `nemotron` | 24 | 1.60 | 0.82 | fail at 512 |
| `qwen36` | 16 | 1.38 | 0.81 | fail at 512 |

The 4/7 is a seq-512 count. **At seq 2048 decoded_cap wins on all seven** (0.37–0.82). Each of the three families that fail the
bar fails at 512 only, and wins at 2048. Read the bar as registered, needing both lengths, not as "the route loses on three
families everywhere".

**V3:** v3 / v1 is 0.89–1.13 on the skewed draw, and 0.88–1.23 over every cell. It is never at or under 0.85, so a v3 training
default is not taken up.

## The registered predictions, each scored on its own text

`rd_table.py` prints these lines, and reads them from the receipt.

| | registered text | verdict | read |
|---|---|---|---|
| P0 | every arm passes the correctness gate on every call | **HELD** | 160 arm-cells: 0 failed, 0 unread |
| P1 | DECODED holds, at least on `graniteh`, `qwen3` and `qwen36` | **REFUTED** | DECODED held (4/7), but of the three families P1 names only graniteh passes. qwen3 fails (1.05 at 512) and qwen36 fails (1.38 at 512) |
| V3 | no prediction registered | — | — |
| P3 | decoded_cap event / device ≤ 1.5 on every many-group skewed cell | **HELD** | max 1.03 (graniteh, 512), over 14 cells |
| P4 | dense / v1 event > 1.2 at seq 512, skewed, on graniteh, qwen3, qwen36 | **HELD** | 20.12, 12.22, 24.11 |
| P5 | on mixtral, dense ≤ decoded_cap on event time at both seqs | **REFUTED** | skew holds (dense 0.53 vs cap 0.57 × v1 at 512; 0.32 vs 0.33 at 2048). Uniform fails (0.58 vs 0.56; 0.34 vs 0.32) |

- **P1 is refuted although the bar it predicted held.** It named the three families it expected to pass, and two of them are
  the families that fail.
  - Its mechanism was launch-bound dense plus v1's per-M-tile decode. P4 confirms that dense is launch-bound there.
  - But launch-bound dense is not what decoded_cap has to beat. At seq 512, v1 is the best arm on every many-group cell, and
    decoded_cap beats v1 only where there are 48 or more rows per expert.
- **P5 names no draw, so it is scored on both.**
  - On the skewed draw alone, the draw the bar reads, it holds.
  - On the uniform draw, decoded_cap beats dense on mixtral by 3–6%. `auto`'s at-most-16-group route stays `dense` regardless,
    because the decoded route ships opt-in and leaves `auto` untouched.

## Observations (not registered lines)

- **Rows per expert separates the seq-512 result.**
  - Measured as seq × top-k / experts, on the skewed draw at seq 512: decoded_cap passes at 48 and 64 rows per expert (ernie,
    graniteh, olmoe, lfm2), and fails at 16–32 (qwen36, nemotron, qwen3).
  - At seq 2048 every family has at least 64, and every family passes.
  - This is one card and one draw, and it was found after the fact. It is a hypothesis for any dispatch rule, not a rule.
- **The cap costs nothing measurable on the many-group families.**
  - decoded_cap minus uncapped decoded comes to −177 to +36 µs per extra chunk.
  - Peak above the inputs at the 256 MiB cap is 180–448 MiB, against 180–2016 MiB uncapped.
  - On mixtral it is +68 to +116 µs per extra chunk, over 8 chunks.
- **The uniform draw keeps the same order.**
  - At 512, decoded_cap / best is 0.72–0.98 on olmoe, lfm2, ernie and graniteh. It is 1.35 on qwen3, 2.24 on nemotron and
    1.78 on qwen36, which is worse than skewed for all three.
  - At 2048 it is 0.37–0.79, except on nemotron at 1.01, where v3 (0.88 × v1) is the best arm.

## Attempts

Eight draws for one reading. They total **$0.85**: Vast costs are receipt actuals, and RunPod costs are adertha estimates
because RunPod returned no billing records. Each draw's adertha receipt fields are copied verbatim into
[`attempts/<run>/launch.json`](attempts/) and [`rd1-rp-5090-2/launch.json`](rd1-rp-5090-2/launch.json).

| run | provider · machine | outcome | lane exit | $ |
|---|---|---|---|---|
| `rd1-5090-1` | Vast · 145701 | anchor REFUSED (rc 3) | 12 | 0.024 |
| `rd1-5090-2` | Vast · 145701 | anchor REFUSED (rc 3) | 12 | 0.026 |
| `rd1-5090-3` | Vast · — | refused before launch: the anchor-exclusion evidence was not tracked at the receipt store's HEAD | — | 0 |
| `rd1-5090-4` | Vast · 36544 | anchor REFUSED (rc 3) | 12 | 0.018 |
| `rd1-5090-5` | Vast · 145701 | anchor REFUSED on all 3 attempts (amendment 1), load1 31.2 / 31.3 / 12.0 | 12 | 0.362 |
| `rd1-rp-prove-1` | RunPod · cs1jece2h1bz | proving run (amendment 2): the fetch worked, and the anchor passed on attempt 1 | 0 | 0.232 |
| `rd1-rp-5090-1` | RunPod · 8wlfk5lvja1v | **anchor CRASHED, a harness or host error, not a refusal**: see below | 12 (now 9) | 0.064 |
| `rd1-rp-5090-2` | RunPod · t1upvpp9ld3m | **the reading** | 0 | 0.120 |

**`rd1-rp-5090-1` was a crash, not a refusal.**
- What happened:
  - On all three attempts, `train_anchor.py`'s H2D probe raised `CUDA error: invalid argument` at
    `torch.empty(n, dtype=torch.float32, pin_memory=True)`, a 256 MB pinned allocation
    ([`attempts/rd1-rp-5090-1/anchor.log`](attempts/rd1-rp-5090-1/anchor.log)).
  - No class was read, so the box was never measured.
  - The runner wrote exit 12, the strict-anchor refusal, because it treated every non-zero gate rc as a refusal.
- **Fixed in this change.** `train_anchor_gate.py` exits 0 (accepted) or 3 (REFUSED), and `rd1_run.sh` now finishes any other
  rc as exit 9, a harness error, before the refusal branch can see it.
- **It could not have excluded the machine.**
  - adertha's lane-refusal exclusion admits only the registered host-limited exits: 13 disk, 14 egress, 17 power cap and 18
    host floor.
  - Exit 12 is admitted only by the strict-anchor validator, and that reads the P41 layout (`P41_EXIT_CODE`, `BOX_REFUSED`)
    on `vast:verified-secure`.
  - A RunPod tc1-layout receipt, under either exit, names no machine.

## Next

- **grouped-nf4-gemm.** An opt-in `decoded` route behind `route_for`, with `GNF4_DECODED_MAX_BYTES`:
  - `auto` stays untouched;
  - this probe's fp32-reference gate goes into gnf4's tests;
  - the 448 MiB peak at the 256 MiB cap is recorded as a measured bound, not a guarantee;
  - **no speed claim** goes into gnf4's docs from this read.
- **e4b.** A TC1 full-step A/B on the 5090 with matched-set EQUIVALENT, registered separately before its box. It is what
  licenses any speed claim, because a per-call win does not license a step-time claim.
