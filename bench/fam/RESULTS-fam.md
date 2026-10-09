# FAM — results: the B=1 fused stack at T == 1 beyond Qwen3. Granite ON_auto **FAIL**; gpt-oss-20b **FAIL** on every knob (its own floor sits below the backstop; Phase C's 0.924 was the gate, not the knob); Qwen3.6-35B-A3B ON_auto **PASS**, and its router epilogue **FASTER** at one row (ratio 0.973). One RTX 5090 per family

Registration: `bench/fam/PREREG-fam.md` (#1380, `9629874f`; Amendments 1 in #1403, 2 in #1420, 3 in #1432 and 4 in
#1436). Issue: #1362.

Code under test, on every box:
- e4b 0.50.0 at `9766fb4c` (#1403's merge, carrying #1395 and #1398); Qwen3.6's rerun at `98d59921` (#1420's merge);
- grouped-nf4-gemm 0.43.0 at `6ee2e10`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0.

The instrument is the same on every family. Each ON config is scored against its family's own OFF, teacher-forced in
fp32. The comparison uses four cells (wikitext and c4val1, at shapes 1 and 12), each read on three disjoint window
sets. The gate is relative to the family's worst neutral draw (chunk, half, `split1`), with an absolute backstop. A
config passes only if every (cell, set) entry passes. `mut090` (a ×0.90 softmax temperature) must fail every gated
cell, or the family is UNRESOLVED; `mutant_scale` (×0.5) must fail, or the reading is VOID.

| family | run | host | cost | verdict |
|---|---|---|---:|---|
| Granite-3.1-3b-a800m | `fam-granite-1` | EPYC 7C13, 256 CPUs | $0.828 | ON_auto **FAIL** |
| gpt-oss-20b | `fam-gptoss-1` | AMD engineering sample, 64 CPUs | $1.007 | ON_glue, ON_r2, ON_epi, ON_auto **FAIL**; anchor: 0.924 inside the floor |
| Qwen3.6-35B-A3B | `fam-qw36-1` | EPYC 7763, 256 CPUs | $1.639 | VOID (a harness bug; Amendment 2) |
| Qwen3.6-35B-A3B | `fam-qw36-2` | EPYC 7C13, 256 CPUs | $2.345 | ON_auto **PASS** (Amendment 3's re-reduction) |
| Qwen3.6, speed proof | `fam-speed-prove-1` | EPYC 7713, 128 CPUs | $0.632 | PROVED (Amendment 4) |
| Qwen3.6, speed | `fam-speed-1` | EPYC 7713, 128 CPUs | $0.632 | the router epilogue **FASTER** (Amendment 4) |

The lane spent $7.720 of its $18 ceiling.

## Granite-3.1-3b-a800m (`fam-granite-1`)

**ON_auto FAIL**, on one gated entry of 12: c4val1, shape 1, set C. Argmax agreement was **0.95117 (1461 of 1536)**
against the gate A_f − 0.005 = **0.95195**, where A_f = 0.95703 is chunk's draw on set B. In that entry, |bias| was 0.0026
(gate 0.0144), spread 0.0084 (gate 0.0248) and KL 0.0081, all inside the floor draws' range (KL 0.0071–0.0093). Every
other entry passed. As registered, Granite's knobs stay off by default. The maintainer re-derived the verdict from the
store and ruled no re-read under this registration. Any later re-read gets its own registration, with a window count
from a power calculation on these floor draws.

| cell | floor B / S / A | gate: abs bias / spread / agree | ON_auto worst set: abs bias / spread / agree | |
|---|---|---|---|---|
| wikitext, 1 | 0.0065 / 0.0124 / 0.9577 | ≤ 0.0165 / ≤ 0.0248 / ≥ 0.9527 | 0.0033 / 0.0124 / 0.9603 | pass |
| wikitext, 12 | 0.0053 / 0.0121 / 0.9544 | ≤ 0.0153 / ≤ 0.0243 / ≥ 0.9494 | 0.0034 / 0.0119 / 0.9596 | pass |
| c4val1, 1 | 0.0044 / 0.0124 / 0.9570 | ≤ 0.0144 / ≤ 0.0248 / ≥ 0.9520 | 0.0026 / 0.0119 / **0.9512** | **fail (set C)** |
| c4val1, 12 | 0.0083 / 0.0152 / 0.9505 | ≤ 0.0183 / ≤ 0.0304 / ≥ 0.9455 | 0.0069 / 0.0144 / 0.9564 | pass |

- **The instrument held.** The census read ON_auto `0 / 65 / [32, 32] / 32` exactly and OFF all zero. Every engagement
  count held: decode attention, decode-shaped forwards, and glue calls per step. `mut090` failed all four cells: its
  worst-set agreement was at most 0.9408 and its |bias| at least 0.0261. `mutant_scale` failed all four. `rep` was
  bit-identical; `split1` was not.
- **Ladder** (set A only, reported): `mut095` passed 3 of 4 cells, failing c4val1 at shape 1; `mut098` passed 4 of 4.
  At these window counts, the gate resolves a ×0.90 temperature change but not reliably a ×0.95 one.
- **Process walls:** OFF 2891 s, ON_auto 483 s. Peak memory 4.89 GB.

## gpt-oss-20b (`fam-gptoss-1`)

**ON_glue, ON_r2, ON_epi and ON_auto all FAIL.** Every failing entry fails only the absolute backstop (agreement ≥
0.90, |bias| ≤ 0.020), never a gate relative to the family's floor. The reason is the floor itself: at T == 1,
gpt-oss's neutral draws (chunk, half, `split1`) disagree with R on 7 to 13 % of positions. A_f is 0.874 to 0.884,
and about half the draws read under 0.90. So the backstop cannot be cleared by anything that touches gpt-oss's
arithmetic, and a gate relative to the family alone would license on noise. That is the case the backstop exists for.
As registered, gpt-oss's knobs stay off by default. The maintainer re-derived the verdicts from the store, and ruled
no new registration now.

| cell | floor A_f / B_f | ON_glue: min agree, max abs bias | ON_r2 | ON_epi | ON_auto |
|---|---|---|---|---|---|
| wikitext, 1 | 0.8841 / 0.0229 | 0.8848, 0.0177 ✗ B | 0.9095, 0.0119 | 0.8984, 0.0150 ✗ B | 0.8945, 0.0120 ✗ B |
| wikitext, 12 | 0.8737 / 0.0347 | 0.9023, 0.0175 | 0.9121, **0.0201** ✗ B | 0.8939, 0.0069 ✗ B | 0.8978, 0.0156 ✗ B |
| c4val1, 1 | 0.8841 / 0.0265 | 0.8997, 0.0152 ✗ A | 0.9036, 0.0140 | 0.8926, 0.0151 ✗ A, B | 0.8952, 0.0169 ✗ A, B |
| c4val1, 12 | 0.8743 / 0.0175 | 0.8887, 0.0110 ✗ A | 0.9010, 0.0122 | 0.8848, 0.0057 ✗ A, B | 0.8900, 0.0129 ✗ A, B |

✗ names the failing sets. Every ✗ is agreement under 0.90, except ON_r2's, which is |bias| 0.0201 over 0.020.

- **The anchor answers Phase C.** On SC2g's path (an MXFP4 store on 24 layers; wikitext, shape 12, set A), gpt-oss's
  own floor is A_f = 0.9036, and ON_auto agrees **0.9238** with bias +0.0022. Phase C read 0.924. That lies inside the
  family's own floor, so **Phase C's miss came from the gate, not the knob**.
- **The instrument held.** Census exactly as registered: glue `0 / 49 / [0, 0] / 0`, r2 `0 / 0 / [24, 0] / 0`, epi
  `0 / 0 / [0, 0] / 24`, auto `0 / 49 / [24, 0] / 24`. Every engagement count held. `mut090` and `mutant_scale` failed
  all four cells, `rep` was bit-identical, and `split1` was not.
- **Ladder** (set A, reported): `mut095` passed 2 of 4 cells and `mut098` 1 of 4. Both fail on the backstop too.
- **Process walls:** OFF 2025 s, each ON 321 to 338 s. Peak 15.77 GB on the default path and 25.22 GB on SC2g.
- **The route, if gpt-oss at B=1 ever matters:** a registration with a different quality statistic (teacher-forced
  NLL, with a window count sized by a power calculation), written before any data.

## Qwen3.6-35B-A3B (`fam-qw36-1`, `fam-qw36-2`)

**ON_auto PASS.** This verdict is Amendment 3's re-reduction of `fam-qw36-2`'s records. The box's own reduction, a
VOID, is kept beside it as `verdict.json`. Only the router epilogue engages on this family (census `0 / 0 / [0, 0] /
40`), and it sits far inside the floor: worst-set |bias| at most 0.0012, agreement at least 0.9876, KL at most 0.0010.
The maintainer re-derived the verdict from the store independently.

| cell | floor B / S / A | gate: abs bias / spread / agree | ON_auto worst set: abs bias / spread / agree |
|---|---|---|---|
| wikitext, 1 | 0.0046 / 0.0123 / 0.9622 | ≤ 0.0146 / ≤ 0.0246 / ≥ 0.9572 | 0.0005 / 0.0018 / 0.9954 |
| wikitext, 12 | 0.0036 / 0.0124 / 0.9577 | ≤ 0.0136 / ≤ 0.0247 / ≥ 0.9527 | 0.0007 / 0.0029 / 0.9876 |
| c4val1, 1 | 0.0066 / 0.0084 / 0.9635 | ≤ 0.0166 / ≤ 0.0169 / ≥ 0.9585 | 0.0007 / 0.0009 / 0.9941 |
| c4val1, 12 | 0.0028 / 0.0095 / 0.9661 | ≤ 0.0128 / ≤ 0.0189 / ≥ 0.9611 | 0.0012 / 0.0012 / 0.9941 |

- **Two VOIDs came first, both mine, and neither came from e4b or the data.**
  - `fam-qw36-1` ($1.639) crashed. The box ran shape 1 first, and that runner froze the hybrid's linear-state pool at
    17 slots, which shape 12 cannot grow. Amendment 2 runs the largest shape first, with bit-identity tests.
  - `fam-qw36-2` ($2.345) ran every cell, then VOIDed on a count the registration got wrong. It expected the
    linear-state warm-up forward on every padded pass, but e4b runs it once per process. Amendment 3 corrected the
    count. The records were re-reduced with no new rental, and nobody computed a gate statistic before it merged.
- **The instrument held.** `mut090` and `mutant_scale` failed all four cells. `rep` was bit-identical and `split1` was
  not. On the ladder (set A), `mut095` and `mut098` passed 4 of 4: at these window counts the gate does not resolve
  a ×0.95 temperature change on Qwen3.6, and ON sits well inside even that.
- **Process walls:** OFF 5966 s, ON_auto 922 s. Peak 24.38 GB.
- **What PASS licenses:** a separate default pull request adding the family to the knobs' allowlist. On this family
  that turns on only the router epilogue, so Amendment 4 first read the speed it buys (next section).

## Qwen3.6: what the router epilogue buys at one row (`fam-speed-1`, Amendment 4)

**FASTER.** On the default graph server, one request decodes in 12.30 ms a step with the router epilogue fused, against
12.64 ms without it: ratios 0.9727 and 0.9732 in the two interleaved blocks, about 2.7 % faster. The maintainer
re-derived it from the store and, on both reads, licensed the allowlist flip.

| block | OFF median step | ON median step | ON / OFF | per-pair median |
|---|---:|---:|---:|---:|
| a (OFF first) | 12.643 ms | 12.298 ms | **0.9727** | 0.9728 |
| b (ON first) | 12.627 ms | 12.289 ms | **0.9732** | 0.9732 |

- **The method** was P124 Amendment 1's. One model served both settings (the box swaps each fused router's `forward`
  for its original). Each block held a live runner per setting, with graphs captured under that setting, decoding in
  strict lockstep: 256 timed steps each, 0.06 % between the blocks. Each runner bound its own slot of the hybrid's one
  linear-state pool.
- **The instrument held.** The census read `0 / 0 / [0, 0] / 40`, and each setting swapped 40 routers. Every runner
  replayed bucket 1 exactly 293 times, with no eager step. Each setting was deterministic across blocks, and ON's tokens
  matched OFF's, 293 of 293.
- **The prediction:** FASTER (about 35 %) hit; the magnitude (ratio about 0.99) missed, as the gain was larger.
- **The default install.** The image has no `causal_conv1d` or `flash-linear-attention`, so Qwen3.6's linear layers ran
  transformers' reference path in both settings. This is the default server as e4b installs it. With those kernels the
  step would be shorter, and the epilogue's share possibly larger; that is a hypothesis, not a finding.
- **Peak** 23.1 GiB allocated; SM clock 2542–2917 MHz.

## Predictions, graded

| # | prediction | outcome |
|---|---|---|
| F1 | census and every engagement count exact on all three families; no VOID | **MISSED**: census exact everywhere, but Qwen3.6 VOIDed twice (a harness crash, then a mis-registered count) |
| F2 | `split1` not bit-identical to R in any cell | **HIT** on all three families |
| F3 | gpt-oss A_f at (wikitext, 12) in [0.88, 0.94] | **MISSED**: 0.8737 |
| F4 | the anchor: Phase C's 0.924 within gpt-oss's own SC2g floor, about 60 % | **HIT**: A_f 0.9036; ON_auto 0.9238 |
| F5 | `mut090` fails every gated cell on every family (about 85 %); `mut098` passes everywhere it runs (about 80 %) | **MISSED**: `mut090` failed every cell on all three, but `mut098` failed 3 of 4 on gpt-oss (the backstop) |
| F6 | Granite ON_auto PASS, about 70 % | **MISSED**: FAIL on one entry's agreement by 2 positions of 1536 |
| F7 | Qwen3.6 ON_auto PASS, about 85 % | **HIT** |
| F8 | gpt-oss: ON_epi PASS about 75 %; ON_glue and ON_r2 about 65 % each; ON_auto about 55 % | **MISSED** on all four |
| F9 | peak memory ≤ 26 GiB on every process | **HIT**: 25.22 GiB at most (gpt-oss on SC2g) |

## Receipts

`bench/fam/receipts/fam-granite-1/`, `fam-gptoss-1/`, `fam-qw36-1/` and `fam-qw36-2/` each hold the process records,
`verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, the bake, `logs/`, the teardown proof, a README and
`SHA256SUMS`. `fam-qw36-2/` adds `verdict-amendment-3.json`. `fam-speed-prove-1/` and `fam-speed-1/` hold the speed
record (`speed_qw36.json`), `speed_verdict.json` and the same companions. The launcher receipts and ledger rows are in
the receipt store. Claims: `e4b.serve.fam.fused-stack-t1.granite.5090.2026-10-09`,
`e4b.serve.fam.fused-stack-t1.gptoss.5090.2026-10-09`, `e4b.serve.fam.fused-stack-t1.qw36.5090.2026-10-09`,
`e4b.serve.fam.router-epilogue-speed.qw36.5090.2026-10-09`.
