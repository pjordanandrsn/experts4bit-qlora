# TC1 amendments 17–18 read: e4b's steady-state lead over axolotl's scattermoe replicates on a second host — P29 HELD (1.146); a same-host repeat reads 1.253

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 17 and 18. Both boxes are byte-identical to
`tc1-5090-46`'s ([amendment 16's read](RESULTS-tc1-steady16.md), P27 1.238): the `qwen3nativebest200` token, e4b `e5859e9`, grouped-nf4-gemm
`00929a4`, no environment set.

## The boxes

| box | machine | host | how the machine was chosen | cost |
|---|---|---|---|---|
| `tc1-5090-47` (amendment 17) | 142284, **the same as -46** | AMD EPYC 7663, instance 54037643 | launcher preference off; 142284 was still the cheapest RTX 5090 | $0.98 |
| `tc1-5090-48` (amendment 18) | **150527** | AMD EPYC 7C13, instance 54051454, driver 580.95.05 | 142284 left out by evidence: `avoid_vast_machine_receipts` cites -46's receipt (adertha-agents#139); the receipt records the exclusion | $1.23 |

## Readings, steps 101..200 (s/step, late-window medians)

| box | e4b shipped d1 / d2 | axolotl scattermoe best d1 / d2 | axolotl / e4b | summed step time, 200 steps (e4b / axolotl) |
|---|---|---|---|---|
| -46 (P27, for reference) | 3.223 / 3.242 | 4.014 / 3.991 | 1.238 [1.231, 1.246] | 650–654 / 1,373–1,389 s |
| -47, same machine | 3.164 / 3.184 | 3.981 / 3.972 | **1.253** [1.248, 1.258] | 642–644 / 1,364–1,371 s |
| -48, second host | 3.826 / 3.740 | 4.319 / 4.355 | **1.146** [1.129, 1.164] | 763–780 / 1,524–1,653 s |

- **P29 HELD.** On machine 150527, axolotl / e4b shipped is 1.146. Its whole cross-draw interval, 1.129–1.164, lies inside [1.05, 1.50]
  and above 1.0. Both pairs are stable: e4b's draws 2.3 % apart, axolotl's 0.8 %.
- **e4b as shipped steps faster than axolotl's scattermoe at steady state on both hosts.** The lead is host-dependent in size: 15 % on the
  EPYC 7C13 host, 24–25 % on the EPYC 7663.
- **P28 UNTESTED by its own rule** (`tc1-5090-47` re-bought machine 142284). As a same-host repeat it reproduces -46 within 1.2 %
  (1.253 against 1.238).
- **Reported beside, as before:**
  - e4b finishes 200 steps first on every box;
  - axolotl spends ×1.8–1.9 the energy per step;
  - axolotl reaches the matched held-out curve (0.7687 on -48), while e4b as shipped sits 0.025–0.027 above it.

## What follows

By amendment 18's decision rule, the register row `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state` records the second-host
replication, and STATUS drops "one host". The 2026-10-02 row stands for the code before #945.
