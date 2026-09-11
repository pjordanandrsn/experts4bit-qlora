# P39 — results

Pre-registration: [`P39-PREREG.md`](P39-PREREG.md), including Amendments 1–5.
Every number below is read from a committed receipt; nothing is recomputed from
memory. Receipts live in the private record under
`receipts/experts4bit-qlora/2026-09-10/p39-*`.

Two hypotheses. **H1 refuted.** **H2 answered, but only on the third attempt,
and the answer is narrower than the mechanism it was meant to confirm.**

---

## H1 — does R-aware split-K help at the step level on sm_120?

`gnf4#357` made `_plan` take the row count `R`, because the launch grid is
`(cdiv(N, BLOCK_N), R, sk)` and the split-K rule was targeting a block count
without the factor `R` in it. A 48-cell sweep on an A2000 (sm_86) found the
R-aware plan never slower and 1.10–1.31× faster at `R >= 16`.

The registered prediction was a step-level `NEW/OLD` of 0.96–0.99 at B=16 on
the serve path. Measured on an RTX 5090 (sm_120, 128 SMs), ABAB on identical
bytes, CUDA-graph replay:

| pair | OLD step | NEW step | NEW/OLD |
|---|---|---|---|
| r1 | 12.6202 ms | 12.7006 ms | **1.00637** |
| r2 | 12.6141 ms | 12.6939 ms | **1.00632** |
| OLD vs OLD | 12.6202 | 12.6141 | 1.00048 |
| NEW vs NEW | 12.7006 | 12.6939 | 1.00053 |
| B=1 | 4.8208 ms | 4.8199 ms | 0.99983 |

The two pairs agree to within 5e-5, and the NEW/OLD gap is about **twelve times
the within-config spread**. So the 0.6 % is not noise: on sm_120 the R-aware
plan is reproducibly, slightly *slower*, and B=1 does not move.

**Cause.** `SPLITK_TARGET_BLOCKS_PER_SM = 8` was fitted on sm_86. It does not
transfer to sm_120. `gnf4#358` gates the R term to parts with
`sm_count <= SPLITK_R_TERM_MAX_SMS` (64), which keeps the measured A2000 win
and restores sm_120 to the old plan. A 5090-specific constant is unfitted and
is left unfitted rather than guessed.

---

## H2 — does a recorded gptq/rtn assignment reproduce the licensed pack on another box?

Background: `#530`. The pack's identity is decided by `routed_rows >= min_rows`
(32), a threshold on a quantity that sits at the router-flip noise floor, so
P37 re-derived 11522/766 where three earlier boxes read 11512/776 — ten experts
of 12,288 on the other side, which under the lane's fingerprint rule is not the
licensed bytes. `#531` added a recorded assignment: persist the decision, and
honour it instead of re-deriving.

### Box 2 — passed, and proved nothing

The pack reproduced **byte-for-byte** across boxes
(`sha256:0c9955a9…` on both) and the registered K8 gate passed both texts:

| text | nf4 ppl | honoured ppl | delta | budget +0.05 |
|---|---|---|---|---|
| wikitext | 6.4198 | 6.3624 | **−0.0574** | PASS |
| c4val1 | 16.4970 | 16.5370 | **+0.0399** | PASS |

against P37's +0.1093 FAIL on c4val1. That refuted my own prediction that the
gate would fail.

But the run reported **zero disagreements**: the second box's routing agreed
with the record on all 12,288 expert matrices, so the honour-the-record path
never had a decision to override. A control with no assignment produced the
same fingerprint. The pack matched because nothing disagreed, not because
honouring did any work, and the causal attribution was withdrawn. (The
receipt's own `GATE` line still carries the pre-`#549` wording
`PASS (split was the cause)`. Receipts are never edited; `#549` made that
verdict conditional on disagreements, and the clause does not hold here.)

### Box 3 — forced a disagreement, and found the precondition

Amendment 4 stopped hunting for a box that would naturally disagree and made
one deterministically: calibrate with 32 sequences instead of 128.

| arm | counts |
|---|---|
| `recipe32` (no record) | **10820 gptq / 1468 rtn** |
| box 1's rich record | 11512 / 776 |

So the forcing works: the two records genuinely differ. Honouring box 1's
**richer** record on the thin box then **refused**:

```
layer 13 expert 60 gu: the assignment says gptq but this box's calibration
never routed to it (no Hessian) -- refusing; a silent RTN here would not be
the licensed pack
```

That refusal is correct, and it is the finding. **A record is honourable only
on a box whose calibration routed to at least the experts the record calls
`gptq`.** Honouring fixes a classification; it cannot conjure a Hessian that
was never computed. My write-up on `#530` proposed "persist the assignment and
honour it" as though that were sufficient. It is not — the mechanism carries a
precondition, now documented on `enable_serve_experts_int4` itself.

### Box 4 — the reverse direction, confirmed

Amendment 5 inverts it. The **record** is box 3's weak one; the
**calibration** is the rich 128-sequence one that locally splits 11512/776.
Every expert the record calls `gptq` is certainly routed here, so nothing can
refuse and the override has somewhere to land.

| pre-registered | measured |
|---|---|
| honoured counts equal the record | **10820 / 1468** = the record |
| dumped `method_map` equals the record | **true** |
| disagreements > 0 | **714 expert-roles overridden** |

714 across five layer chunks (140 + 148 + 154 + 150 + 122). The net of 692
(11512 − 10820) decomposes as **703 dragged `gptq` → `rtn`** and **11 pulled
`rtn` → `gptq`**.

The run's own summary line says 140 — the first chunk alone, a reducer bug of
the same class `#536` fixed for the counts, left behind on the field next door
and fixed in `#553`. The verdict is unaffected (the gate is `> 0`); the number
is not.

**So the mechanism is demonstrated rather than assumed.** A recorded
assignment, honoured, produces the recorded classification on a box that would
have chosen differently, and reports what it overrode.

---

## What is still open

- The pack that box 4 built is byte-different from the licensed one
  (`sha256:8a7a6ffd…`), as expected and as `#531`'s docstring already says:
  honouring makes the *classification* reproducible, not the bytes, because
  GPTQ output also depends on the Hessian, which routing perturbs. The
  licensed bytes still come only from the artifact.
- Box 4's honoured pack was **not** put through the K8 gate. Whether a pack
  built from a deliberately thin record scores as well as the licensed one is a
  quality question this lane did not ask.
- The **c4val1-specific** gate failures across bo5, bo6 and P37 remain
  unexplained. Three lanes have now passed wikitext and failed c4val1 on the
  same budget. A gate one text passes and another fails, repeatedly, is either
  finding something real or is mis-specified, and it deserves its own look.

## Cost

Fifteen ledger rows, $6.02 against a $6 estimate, a $12 lane ceiling and a $20
hard stop. Two of those rows are the lane's own bugs: `p39-box2-2` ($1.30) was
a watchdog that covered one of two calibrating arms, and `p39-box1b-4` ($1.04)
a build alarm fixed at 5400 s on a host that needed longer. Both are fixed
(`#541`, `#539`).
