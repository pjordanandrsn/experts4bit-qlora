# FAM — results: the B=1 fused stack at T == 1 beyond Qwen3. Granite ON_auto **FAIL**; gpt-oss-20b **FAIL** on every knob (its own floor sits below the backstop; Phase C's 0.924 was the gate, not the knob); Qwen3.6-35B-A3B ON_auto **PASS**, and its router epilogue **FASTER** at one row (ratio 0.973); Mixtral-8x7B **PASS** on every knob, resolved at × 0.90; Gemma-4-26B-A4B closed at its proof: its own floor sits below the backstop, so no knob is licensable on its NF4 stack. One RTX 5090 per family

Registration: `bench/fam/PREREG-fam.md` (#1380, `9629874f`; Amendments 1 in #1403, 2 in #1420, 3 in #1432, 4 in
#1436, and Mixtral's 5 in #1457, 7 in #1468, 8 in #1479 and 9 in #1487; Gemma-4's 6 in #1460 and 10 in #1529). Issue: #1362.

Code under test, on every box:
- e4b 0.50.0 at `9766fb4c` (#1403's merge, carrying #1395 and #1398); Qwen3.6's rerun at `98d59921` (#1420's merge);
  Mixtral's reading on e4b 0.51.0 at `68845bfc` (#1487's merge, past #1482);
- grouped-nf4-gemm 0.43.0 at `6ee2e10`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0.

The instrument is the same on every family. Each ON config is scored against its family's own OFF, teacher-forced in
fp32. The comparison uses four cells (wikitext and c4val1, at shapes 1 and 12), each read on three disjoint window
sets. The gate is relative to the family's worst neutral draw (chunk, half, `split1`), with an absolute backstop. A
config passes only if every (cell, set) entry passes. `mut090` (a ×0.90 softmax temperature) must fail every gated
cell, or the family is UNRESOLVED; `mutant_scale` (×0.5) must fail, or the reading is VOID. Mixtral gates on ×0.90,
then ×0.80, and claims the weakest rung that fails every gated cell (Amendment 8).

| family | run | host | cost | verdict |
|---|---|---|---:|---|
| Granite-3.1-3b-a800m | `fam-granite-1` | EPYC 7C13, 256 CPUs | $0.828 | ON_auto **FAIL** |
| gpt-oss-20b | `fam-gptoss-1` | AMD engineering sample, 64 CPUs | $1.007 | ON_glue, ON_r2, ON_epi, ON_auto **FAIL**; anchor: 0.924 inside the floor |
| Qwen3.6-35B-A3B | `fam-qw36-1` | EPYC 7763, 256 CPUs | $1.639 | VOID (a harness bug; Amendment 2) |
| Qwen3.6-35B-A3B | `fam-qw36-2` | EPYC 7C13, 256 CPUs | $2.345 | ON_auto **PASS** (Amendment 3's re-reduction) |
| Qwen3.6, speed proof | `fam-speed-prove-1` | EPYC 7713, 128 CPUs | $0.632 | PROVED (Amendment 4) |
| Qwen3.6, speed | `fam-speed-1` | EPYC 7713, 128 CPUs | $0.632 | the router epilogue **FASTER** (Amendment 4) |
| Mixtral-8x7B, proof | `fam-mixtral-prove-1` | Xeon E5-2699 v3, 72 CPUs | $1.115 | NOT PROVED (server KV pool out of memory; Amendment 7) |
| Mixtral-8x7B, proof | `fam-mixtral-prove-2` | EPYC 7763, 256 CPUs | $1.775 | PROVED; UNRESOLVED at proof scale (Amendment 8) |
| Mixtral-8x7B | `fam-mixtral-1`, `-2` | none | $0 | refused at the provider before any instance existed |
| Mixtral-8x7B | `fam-mixtral-3` | EPYC 9454, 96 CPUs | $0.87 | VOID (#1477's serving regression; Amendment 9) |
| Mixtral-8x7B | `fam-mixtral-4` | EPYC 7C13, 256 CPUs | $2.59 | ON_glue, ON_r2, ON_epi, ON_auto **PASS**, resolved at ×0.90 |
| Gemma-4-26B-A4B, proof | `fam-gemma4-prove-1` | EPYC 7C13, 256 CPUs | $1.13 | VOID on Amendment 6's count; PROVED under Amendment 10 with ON_epi and ON_auto **FAIL** on the backstop; closed, no reading |

The lane closed at $15.20 of its $26 ceiling (raised from $18 in Amendment 5).

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

## Mixtral-8x7B-Instruct (`fam-mixtral-4`)

**PASS on every knob, resolved at × 0.90.** `mut090` failed the gate in every gated cell, so the read is the stronger
null read Amendment 8 allowed: no effect on Mixtral as large as a × 0.90 change of the decode softmax scale. The
maintainer re-derived the verdict from the store independently.

| cell | floor B / S / A | gate: abs bias / spread / agree | ON_auto worst set: abs bias / spread / agree |
|---|---|---|---|
| wikitext, 1 | 0.0039 / 0.0090 / 0.9694 | ≤ 0.0139 / ≤ 0.0181 / ≥ 0.9644 | 0.0025 / 0.0080 / 0.9727 |
| wikitext, 12 | 0.0062 / 0.0109 / 0.9694 | ≤ 0.0162 / ≤ 0.0218 / ≥ 0.9644 | 0.0054 / 0.0092 / 0.9707 |
| c4val1, 1 | 0.0043 / 0.0110 / 0.9661 | ≤ 0.0143 / ≤ 0.0221 / ≥ 0.9611 | 0.0021 / 0.0112 / 0.9661 |
| c4val1, 12 | 0.0039 / 0.0091 / 0.9661 | ≤ 0.0139 / ≤ 0.0182 / ≥ 0.9611 | 0.0029 / 0.0100 / 0.9694 |

| config | census (q/k/v / glue / r2 / router) | worst abs bias | worst agree | max KL |
|---|---|---:|---:|---:|
| ON_glue | `0 / 65 / [0, 0] / 0` | 0.0041 | 0.9668 | 0.0051 |
| ON_r2 | `0 / 0 / [32, 32] / 0` | 0.0069 | 0.9694 | 0.0061 |
| ON_epi | `0 / 0 / [0, 0] / 32` | 0.0094 | 0.9674 | 0.0049 |
| ON_auto | `0 / 65 / [32, 32] / 32` | 0.0054 | 0.9661 | 0.0062 |

- **The resolution.** ×0.90's worst-set agreement was 0.951–0.958, below every cell's gate (0.961–0.964). At proof
  scale (32 positions) it had passed on c4val1, which triggered Amendment 8's ×0.80 rung. The reading's 128 positions
  resolved ×0.90. ×0.80 also failed everywhere (agreement 0.921–0.932).
- **The instrument held.** The census was exact on all five processes, and every glue-kernel count matched Amendment
  5's table. `mutant_scale` failed every cell. `rep` was bit-identical; `split1` was not. On the ladder (set A), ×0.95
  passed 3 of 4 cells (it failed c4val1 at shape 1) and ×0.98 passed all four.
- **Three tries came first.**
  - `fam-mixtral-prove-1` ran out of GPU memory building the server's own KV pool (Amendment 7).
  - `fam-mixtral-1` and `-2` were refused at the provider ($0). No verified 5090 was offered at the policy's $0.85/h.
    The reading then launched through a read-only price waiter.
  - `fam-mixtral-3` VOIDed on its first forward: #1477's MoE residual fold raised a TypeError in the hybrid tier the
    default server installs. #1482 fixed it, and Amendment 9 moved the reading past it. On a tiny Mixtral, the
    glue-kernel counts per decode step were identical before and after that change.
- **Walls:** about 830 s from launch to the bake's end; OFF 4314 s; each ON 555–603 s. Peak 28.19 GiB.
- **What PASS licenses** (Amendment 5): an allowlist pull request, but only together with a speed read of what the
  passing knobs buy on Mixtral (Amendment 4's interleaved method). None is proposed here; the four knobs stay off on
  Mixtral by default.

## Gemma-4-26B-A4B (`fam-gemma4-prove-1`): closed at the proof

**No knob is licensable on Gemma-4's NF4 stack with this instrument.** The family's own neutral floor sits below the
absolute backstop. Each ON knob passes the relative gate and fails only the backstop, so nothing here shows the knobs
harm Gemma-4. The maintainer closed the family at its proof; there is no reading.

| cell | floor B / S / A | ON_epi worst set: abs bias / agree | ON_auto worst set: abs bias / agree | ×0.90 worst agree |
|---|---|---|---|---:|
| wikitext, 1 | 0.3633 / 0.4486 / 0.7370 | 0.2126 / 0.8464 | 0.0872 / 0.8594 | 0.7865 |
| wikitext, 12 | 0.5069 / 0.5486 / 0.7500 | 0.0298 / 0.8516 | 0.0469 / 0.8568 | 0.7604 |
| c4val1, 1 | 0.1095 / 0.3202 / 0.7656 | 0.0409 / 0.8281 | 0.0562 / 0.8125 | 0.7448 |
| c4val1, 12 | 0.0967 / 0.2199 / 0.7448 | 0.0671 / 0.8464 | 0.0501 / 0.8229 | 0.7266 |

- **The box's own reduction VOIDed**, on a count Amendment 6 got wrong. At T == 12 the glue fold sends Gemma-4's
  per-head q and k norms (12 × 16 and 12 × 8 rows) to their own torch forward above 64 rows. So ON_auto calls
  `rmsnorm_rows` 216 times per step, not 271. That is a re-route, not a skip. On the seat's A2000, every norm module
  ran, and the logits stayed within 0–0.07 relative of OFF's, against 1.15 for a skipped norm. Amendment 10 counts per
  shape, and the records were re-reduced under it (`verdict-amendment-10.json`). The maintainer re-derived it
  byte-identical.
- **The floor is the finding.** Neutral redraws alone (chunked prefill, halved batches, one KV split) move R's NLL by
  up to 0.51 nats, with agreement as low as 0.737 and KL up to 0.68. For Mixtral the same arms read about 0.006 nats
  and 0.97. R's mean NLL is 4.96 nats on wikitext and 5.69 on c4val1. This matches the register: Gemma-4's NF4 stack is
  1.077 nats of KL from bf16 (`e4b.serve.p44.gemma4.nf4-vs-bf16`), and its arithmetic-order floor is 5–25 times other
  families' (`e4b.parity.gemma4.chunk-free`).
- **The relative gate passes; the backstop fails.** ON_epi and ON_auto agree with R at 0.81–0.86, above every cell's
  A_f − 0.005 and with |bias| inside B_f + 0.010. Both fail only the 0.90 backstop. ON sits 0.04–0.09 below it, and
  the neutral floor 0.13–0.16 below it. At reading scale the standard error of agreement is about 0.01–0.02, and a PASS
  needs all 12 entries of a config at or above 0.90. So a reading would have bought the same structural FAIL for up to
  $3.95.
- **There is no resolution in substance.** The reducer reports `mut090` failing every cell, so formally the resolution
  is ×0.90. On the wikitext cells, though, ×0.90 passes the relative gate and fails only the backstop. The floor is
  wider than a ×0.90 softmax-scale change there.
- **Walls:** about 680 s from launch to OFF; OFF 1535 s; ON_epi 254 s; ON_auto 240 s, at 32 positions. Peak 26.18 GiB.

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
| X1 | Mixtral: census, every engagement count and the fp32 router path exact; no VOID, about 85 % | **MISSED**: `fam-mixtral-3` VOIDed (a serving regression on main, not the instrument); the reading itself was exact, with no VOID |
| X2 | Mixtral's floor A_f at (wikitext, 12) in [0.93, 0.97] | **HIT**: 0.9694 |
| X3a | `mut090` fails every gated cell at reading scale (resolution ×0.90), about 25 % | **HIT** |
| X3b | `mut080` fails every gated cell, about 75 % | **HIT** |
| X4 | ON_epi PASS about 70 %; ON_glue and ON_r2 about 60 % each; ON_auto about 50 % | **HIT** on all four |
| X5 | peak memory ≤ 30 GiB on every Mixtral process | **HIT**: 28.19 GiB |
| G1 | Gemma-4: census and every engagement count exact; no VOID, about 80 % | **MISSED**: the proof VOIDed on Amendment 6's per-step count (a re-route at T == 12; Amendment 10) |
| G2 | Gemma-4's floor A_f at (wikitext, 12) in [0.90, 0.96] | **MISSED**: 0.750 at proof scale |
| G3 | `mut090` fails every gated cell, about 80 % | **HIT only by the backstop**: on wikitext ×0.90 passes the relative gate |
| G4 | ON_epi PASS about 65 %; ON_glue about 50 %; ON_auto about 45 % | **not read**: closed at the proof, where ON_epi and ON_auto FAIL on the backstop |
| G5 | peak memory ≤ 24 GiB on every Gemma-4 process | **MISSED**: 26.18 GiB |

## Receipts

`bench/fam/receipts/fam-granite-1/`, `fam-gptoss-1/`, `fam-qw36-1/` and `fam-qw36-2/` each hold the process records,
`verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, the bake, `logs/`, the teardown proof, a README and
`SHA256SUMS`. `fam-qw36-2/` adds `verdict-amendment-3.json`. `fam-speed-prove-1/` and `fam-speed-1/` hold the speed
record (`speed_qw36.json`), `speed_verdict.json` and the same companions. The launcher receipts and ledger rows are in
the receipt store. `fam-mixtral-prove-1/`, `fam-mixtral-prove-2/`, `fam-mixtral-3/` and `fam-mixtral-4/` hold the same
companions for each Mixtral run that rented an instance. `fam-gemma4-prove-1/` holds Gemma-4's proof, with its box
verdict and Amendment 10's re-reduction (`verdict-amendment-10.json`). Claims: `e4b.serve.fam.fused-stack-t1.granite.5090.2026-10-09`,
`e4b.serve.fam.fused-stack-t1.gptoss.5090.2026-10-09`, `e4b.serve.fam.fused-stack-t1.qw36.5090.2026-10-09`,
`e4b.serve.fam.router-epilogue-speed.qw36.5090.2026-10-09`, `e4b.serve.fam.fused-stack-t1.mixtral.5090.2026-10-09`,
`e4b.serve.fam.fused-stack-t1.gemma4.5090.2026-10-10`.
