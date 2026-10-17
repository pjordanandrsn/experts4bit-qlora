# Q235 Amendment 8: the rehearsal records

Amendment 8's mechanics rehearsal: `q235_box.py --micro-batch 2 --heldout` on the RTX A2000 ($0), at e4b `1c12341c` (the fixed probe).
It ran in the A2000 leg's venv, with the step's tokens asserted equal to the leg's (`input_ids` `0047ba73`) and the registered held-out
rows (`heldout.pt.json`: rows 8–15 of the split, 1,432 real tokens). Each JSON is the box record's registered fields, without host
paths (`SHA256SUMS` covers these files). The full box records stay on the controller: `box.json` sha256 `0eaafa85` (slice) and
`7c64148a` (30B); the step's token record `tokens.pt.json` `640b8164`.

| file | subject | rc | dX (both steps) | held-out: before → after step 0 → after step 1 | step's rows (reported) |
|---|---|---|---|---|---|
| `rehearsal-slice4.json` | ac9c66cc, a 4-layer cut | 0 | finite, nonzero: 4 layers × 4 passes | 12.951 → 12.485 → 11.950 (−3.6 %, −7.7 %) | 13.379 → 11.844 (−11.5 %) |
| `rehearsal-30b.json` | Qwen3-30B-A3B, 48 layers | 0 | finite, nonzero: 48 layers × 4 passes, min 0.0068 | 1.7656 → 1.2280 → 1.0851 (−30.4 %, −38.5 %) | 1.8592 → 0.6075 (−67.3 %) |

Amendment 2's nine rules and Amendment 8's (`step0/step1_dx_finite_nonzero`, `heldout_finite`) all held in both runs, and the frozen
base was unchanged across both steps. Under rule 10 as ruled (one-sided: every read finite, each post-update read at most
before × 1.10), both read PASS: every held-out read fell. The 30B leg sizes nothing; it is the rehearsal of record.

The first 30B run, at `7942eb39` with the first probe, is recorded in the PREREG: dX empty on the reentrant fast path (the bug the
fixed probe closes), held-out 1.7656 → 1.2301 → 1.0838.
