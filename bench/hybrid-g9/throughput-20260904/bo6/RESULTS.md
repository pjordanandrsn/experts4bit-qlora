# bo6 — results (2026-09-04, box 49861751, one RTX 5090)

The tables below are the verbatim output of `python buildout_reduce.py .` over the JSON receipts in this directory (cuts, box, protocol, the two calibration methods and the layout: [`README.md`](README.md)). Gate cells and the registered verdict are the registered rule in perplexity on every text scored; the nats beside them are read against the family's arithmetic-order floor and never change the verdict. Every delta is against the NF4 arm re-scored on this lane — nothing on this page is compared across lanes, and there is no speed ratio because this lane has no NF4 speed arm (bo7 measures it). The `method` column is read from each arm's own run log (`logs/run_<family>_ppl_<arm>.log`): `INT4EXP hessians:` = all-at-once, `INT4EXP calibrating (streamed):` = streamed. Rows marked **pending (arrives before merge)** are arms the lane script runs whose receipts were still being produced when this snapshot was taken (Mixtral `lic_calibexp` on wikitext, its B=1 / B=16 arms, and `lic_calibexp_n128`). The wikitext reading itself has since landed on the lane console and is read below; its receipt file fills the row on the final snapshot.

### Qwen3-30B-A3B
arithmetic-order floor 0.0095 nats = wikitext: base ppl 6.420 → ±0.061 ppl; c4val1: base ppl 16.497 → ±0.157 ppl. Every delta is against `nf4` re-scored on this lane (same box, same window sha). Speed ratio: not measured on this lane (no NF4 speed arm; bo7 measures it).
| arm | configuration | method (from the run log) | e4b cut | K8 wikitext nll | Δppl (Δnats) | gate | K8 c4val1 nll | Δppl (Δnats) | gate | registered verdict (ppl) | B=1 ms | B=1 tok/s | B=16 tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention -- re-scored on this box and cut (baseline, both texts) | — | f42924d | 1.85939 | — | baseline | 2.80318 | — | baseline | baseline | — | — | — |
| `calibexp` | GPTQ-calibrated int4 experts alone (attempt 3; the text that failed with RTN experts on bo5) | all-at-once · 16k tok · damp 0.01 · 10826 gptq / 1462 rtn | db2a070 | — | — | — | 2.81222 | +0.1498 (+0.0090 sub-floor) | FAIL | FAIL as registered (c4val1 +0.1498) | — | — | — |
| `all_calibexp` | calibrated int4 experts + C4-calibrated int4 attention + r1 + r2 (rope-only fold) + router epilogue + #385 glue (attempt 3) | all-at-once · 16k tok · damp 0.01 · 10826 gptq / 1462 rtn | db2a070 | 1.85005 | -0.0597 (-0.0093 sub-floor) | pass | 2.80531 | +0.0352 (+0.0021 sub-floor) | pass | pass on 2 texts (within +0.05; no same-sign improvement to claim) | 6.33 | 158.0 | 993.6 |
| `calibexp_c4val_rep1` | calibrated int4 experts alone, queued as a repeat of `calibexp` -- the cut had changed method (bo6b) | streamed · 16k tok · damp 0.01 · 24 GiB budget, 5 passes · 10820 gptq / 1468 rtn | ae9dc12 | — | — | — | 2.80011 | -0.0505 (-0.0031 sub-floor) | pass | pass (1 text; the improvement is not claimable until wikitext agrees) | — | — | — |
| `calibexp_d01` | calibrated int4 experts alone, damping 0.1 (E4B_INT4_GPTQ_DAMP=0.1; bo6b sweep) | streamed · 16k tok · damp 0.1 · 24 GiB budget, 5 passes · 10820 gptq / 1468 rtn | ae9dc12 | — | — | — | 2.80647 | +0.0544 (+0.0033 sub-floor) | FAIL | FAIL as registered (c4val1 +0.0544) | — | — | — |
| `calibexp_n128` | calibrated int4 experts alone, 4x the calibration set (E4B_CALIB_NSEQ=128; bo6b sweep) | streamed · 64k tok · damp 0.01 · 24 GiB budget, 5 passes · 11512 gptq / 776 rtn | ae9dc12 | — | — | — | 2.79031 | -0.2109 (-0.0129 1.4× floor) | pass | pass (1 text; the improvement is not claimable until wikitext agrees) | — | — | — |
| `calibexp_n512` | calibrated int4 experts alone, 16x the calibration set (E4B_CALIB_NSEQ=512; bo6b sweep) | streamed · 256k tok · damp 0.01 · 24 GiB budget, 5 passes · 12008 gptq / 280 rtn | ae9dc12 | — | — | — | 2.79459 | -0.1410 (-0.0086 sub-floor) | pass | pass (1 text; the improvement is not claimable until wikitext agrees) | — | — | — |

Repeat controls (Qwen3-30B-A3B; the released checkpoint re-baked to NF4 by bo6b after attempt 3 freed the arena, then re-scored on the same box and window; `mean_nll` compared at full float precision):
| text | original arm | nll | repeats | nll | spread (nats) | reading |
|---|---|---|---|---|---|---|
| wikitext | `nf4` | 1.85939315385821 | `nf4_rep1` | 1.85939315385821 | 0.0e+00 | bit-identical → the K8 instrument is run-to-run deterministic on one box + cut |
| c4val1 | `nf4_c4val` | 2.803180268104794 | `nf4_c4val_rep1`, `nf4_c4val_rep2` | 2.803180268104794 / 2.803180268104794 | 0.0e+00 | bit-identical → the K8 instrument is run-to-run deterministic on one box + cut |

Calibration method / size / damping sweep (Qwen3-30B-A3B, c4val1, calibrated int4 experts alone; NF4 2.80318 / ppl 16.49703 on this lane):
| arm | method | calibration tokens | damping | gptq / rtn packs | nll | ppl | Δppl (Δnats) | gate |
|---|---|---|---|---|---|---|---|---|
| `calibexp` | all-at-once | 16k | 0.01 | 10826 / 1462 | 2.81222 | 16.64678 | +0.1498 (+0.0090 sub-floor) | FAIL |
| `calibexp_c4val_rep1` | streamed | 16k | 0.01 | 10820 / 1468 | 2.80011 | 16.44653 | -0.0505 (-0.0031 sub-floor) | pass |
| `calibexp_d01` | streamed | 16k | 0.1 | 10820 / 1468 | 2.80647 | 16.55145 | +0.0544 (+0.0033 sub-floor) | FAIL |
| `calibexp_n128` | streamed | 64k | 0.01 | 11512 / 776 | 2.79031 | 16.28610 | -0.2109 (-0.0129 1.4× floor) | pass |
| `calibexp_n512` | streamed | 256k | 0.01 | 12008 / 280 | 2.79459 | 16.35598 | -0.1410 (-0.0086 sub-floor) | pass |

Method effect at 16k tokens, damping 0.01, the same 8 calibration batches, the same box: all-at-once 16.64678 → streamed 16.44653 = -0.2002 ppl (-0.0121 nats (1.3× the 0.0095-nat floor)); registered verdict FAIL → pass.

Speed on this lane (Qwen3-30B-A3B; rental-measured tok/s on this box, one RTX 5090 on a Threadripper PRO 7975WX host; B=1 = 1000 / timed graph step, B=16 = aggregate over 70 graph steps). Ratio: not measured on this lane (bo7 measures it).
- `all_calibexp`: B=1 6.33 ms = 158.0 tok/s (127 timed steps); B=16 993.6 tok/s (16.10 ms/step, 70 steps)

### Mixtral-8x7B-Instruct
arithmetic-order floor unmeasured. Every delta is against `nf4` re-scored on this lane (same box, same window sha). Speed ratio: not measured on this lane (no NF4 speed arm; bo7 measures it).
| arm | configuration | method (from the run log) | e4b cut | K8 wikitext nll | Δppl (Δnats) | gate | K8 c4val1 nll | Δppl (Δnats) | gate | registered verdict (ppl) | B=1 ms | B=1 tok/s | B=16 tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention -- re-scored on this box and cut (baseline, both texts) | — | db2a070 | 1.18214 | — | baseline | 2.10973 | — | baseline | baseline | — | — | — |
| `lic_calibexp` | GPTQ-calibrated int4 experts + r1 + r2 (rope-only fold) + router epilogue, no calibrated attention -- bo5's `lic` with calibrated experts (bo6b, 8 GiB Hessian budget) | streamed · 16k tok · damp 0.01 · 8 GiB budget, 32 passes · 512 gptq / 0 rtn | ae9dc12 | pending (arrives before merge) | — | — | 2.11447 | +0.0391 (+0.0047) | pass | pending (wikitext arrives before merge) | pending | pending | pending |
| `lic_calibexp_n128` | lic_calibexp with 4x the calibration set (E4B_CALIB_NSEQ=128; bo6b) | — | ae9dc12 | — | — | — | pending (arrives before merge) | — | — | pending (arrives before merge) | — | — | — |

Speed on this lane (Mixtral-8x7B-Instruct; rental-measured tok/s on this box, one RTX 5090 on a Threadripper PRO 7975WX host; B=1 = 1000 / timed graph step, B=16 = aggregate over 70 graph steps). Ratio: not measured on this lane (bo7 measures it).
- `lic_calibexp`: B=1 pending (arrives before merge); B=16 pending (arrives before merge)

Gate cells and the registered verdict are in perplexity, the registered unit; the nats beside them are read against the family's arithmetic-order floor and never change the verdict. `pass (1 text; ...)` = a calibrated pack within +0.05 ppl on the one text scored; an improvement is claimable only with the same sign on wikitext (outside the calibration domain).
`pending (arrives before merge)` = an arm the lane script runs whose receipt is not in this snapshot yet.

## Reading it

- **The instrument, on this box.** Qwen3's NF4 reference re-scored here reads wikitext 1.85939 (ppl 6.41984)
  and c4val1 2.80318 (16.49703). bo6b freed the arena, re-baked the checkpoint and re-scored it: c4val1 twice
  and wikitext once, all **bit-identical** to the attempt-2 readings (spread 0.0 nats at full float precision).
  The K8 instrument is run-to-run deterministic on one box and cut, bake included. That settles what the 0.006-nat
  difference between this reference and bo5's on the identical window (2.80923 on box 49841214, integration-6 +
  grouped-nf4-gemm @587eb7a) is *not*: it is not run noise. It stays OPEN (README, instrument notes); no
  sub-0.01-nat reading on this page is compared with bo5's.
- **Attempt 3, all-at-once calibration (integration-8 @db2a070).** Calibrated int4 experts alone on the text that
  failed with RTN experts on bo5: c4val1 2.81222 (16.64678) = **+0.1498 ppl (+0.0090 nats, sub-floor) — FAIL as
  registered**, three times the +0.05 budget, and no better than bo5's RTN reading on its own lane (+0.0709; cited
  beside, not divided). Calibration: 6050 of 6144 (layer, expert) pairs saw rows; 10826 packs GPTQ, 1462 RTN under
  `min_rows=32`. The full stack `all_calibexp` (calibrated experts + C4-calibrated int4 attention + round-1/2 folds +
  router epilogue + #385 glue) under the same method: **wikitext 1.85005 (6.36015) = −0.0597 ppl, c4val1 2.80531
  (16.53226) = +0.0352 ppl — pass on both texts as registered**; the wikitext improvement is not claimed because
  c4val1 carries the other sign. Its speed on this box: **6.33 ms = 158.0 tok/s at B=1, 993.6 tok/s at B=16** —
  rental-measured, no ratio (bo7). The pack it carries was calibrated all-at-once, the two-step API; that is not
  the method 0.35.0 ships.
- **bo6b, streamed (sequential) calibration (@ae9dc122).** The same arm — the same 8 calibration batches, 16k
  tokens, damping 0.01, `min_rows=32` — with the layers packed in five chunks, each chunk's Hessians accumulated
  with the earlier chunks already on int4: c4val1 2.80011 (16.44653) = **−0.0505 ppl (−0.0031 nats) — pass on
  this text**; the improvement is not claimable until wikitext agrees (bo6c). The two methods differ only in the
  *order* of calibration and packing; on the same box that reordering moves c4val1 by **0.200 ppl (0.0121 nats,
  1.3× the family's floor) and flips the registered verdict**. The RTN-fallback count is the same under both
  (1462 vs 1468 of 12288 packs), so "too few rows per expert" is not what separated them. Heavier damping
  (0.1) reads +0.0544 — FAIL as registered, by 0.004. More calibration text helps and cuts the fallbacks: 64k
  tokens −0.2109 (−0.0129 nats, 1.4× floor; 776 RTN packs), 256k tokens −0.1410 (−0.0086; 280 RTN packs) —
  non-monotonic at one measurement per point, so 64k is the candidate size on evidence, not a law.
- **What the nats say beside the verdicts.** Every Qwen3 reading in the table sits between 0.3× and 1.4× the
  family's 0.0095-nat arithmetic-order floor (±0.157 ppl at c4val1's perplexity). The gate is registered in
  perplexity by decision (13:10Z: close the gap, not the gate) and the verdicts stand as printed; the nats say
  the method effect is deterministic (repeats bit-identical) and of the same order as reordering the arithmetic.
  That is why nothing here is claimed as an *improvement* over NF4 until the out-of-domain text agrees.
- **Mixtral-8x7B-Instruct.** NF4 re-scored on this box: wikitext 1.18214 (3.26135), c4val1 2.10973 (8.24604) —
  within 0.001 nats of bo5's (1.18105 / 2.11031; cited beside, not divided). Attempt 3's all-at-once arms never
  produced a reading: the container's 170 GiB cgroup killed each of them (README: the arithmetic). bo6b's streamed
  arm at an 8 GiB Hessian budget (32 passes of one layer; 512 GPTQ packs, 0 RTN — top-2-of-8 routing gives every
  expert its rows): c4val1 2.11447 (8.28518) = **+0.0391 ppl (+0.0047 nats; this family's floor is unmeasured)
  — pass on the in-domain text**, where bo5's RTN `lic` read +0.0575 on its own lane. **wikitext (from the lane
  console, K8 20:33–21:54Z; the receipt lands with the final snapshot): 1.20549 (3.33841) = +0.0771 ppl
  (+0.0234 nats) — FAIL as registered**, over the +0.05 budget on the out-of-domain text. Full-gate verdict for the
  calibrated Mixtral stack: **FAIL as registered; measured, not licensed.** It is the mirror image of bo5's RTN
  `lic` (wikitext −0.046 pass / c4val1 +0.0575 FAIL): the calibrated stack passes the text inside the calibration
  domain and fails the one outside it. Not a reference shift — the window sha `31fd7d408809` matches bo5c's and the
  NF4 references agree to 0.001 nats. B=1, B=16 and the 64k-token arm are pending (arrive before merge) and will be
  measured speed of a configuration that is not licensed.

## Consequences

1. **The gap closes by method, not by moving the gate.** The registered rule was applied as written — perplexity,
   +0.05, every text scored — and the calibrated-expert arms were re-run with GPTQ calibration in place of RTN.
2. **Attempt 3's "calibrated experts alone FAIL +0.150" is a verdict on the all-at-once method.** Under the
   sequential (streamed) method — the mechanism #384 shipped in 0.35.0 — the same arm reads −0.050 on the same
   text, same box, same batches, same damping. The full stack under the streamed method on both texts is bo6c, a
   separate follow-up lane; nothing here waits for it.
3. **Qwen3's full calibrated stack passes both texts as registered** (−0.060 / +0.035, all-at-once pack):
   158.0 / 993.6 tok/s on this box, quoted with its box and without a ratio.
4. **Granite stays NF4** (bo5: calibrated experts +0.387 ppl on c4val1, 8× the budget — not closable with this
   lever; not re-run).
5. **Sequential calibration closed Qwen3's gap and did not close Mixtral's:** c4val1 +0.039 (pass) but wikitext
   +0.077 ppl (+0.0234 nats; floor unmeasured) — FAIL as registered; measured, not licensed. Next levers are not
   gate changes: a per-expert NF4 fallback for the experts with the largest GPTQ residual, or the 64k-token
   calibration set scored on wikitext (the queued `lic_calibexp_n128` arm is c4val1-only).
6. **Instrument:** deterministic on one box + cut; Qwen3's 0.006-nat NF4 shift against bo5 stays OPEN; no
   sub-0.01-nat comparison is made against bo5 or any lane installed from a commit before grouped-nf4-gemm#336.
