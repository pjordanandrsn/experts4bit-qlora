# bo5 — results (2026-09-04, box 49841214, one RTX 5090)

The tables below are the verbatim output of `python buildout_reduce.py . --ref ../bo3` over the JSON receipts in this directory (cuts, box, protocol and layout: [`README.md`](README.md)). Gate cells and the registered verdict are the registered rule in perplexity on every text scored; the nats beside them are read against the family's arithmetic-order floor and never change the verdict. `i6` = integration-6 @0535930 (bo5), `i7` = integration-7 @d090940 (bo5b/bo5c).

### Qwen3-30B-A3B
arithmetic-order floor 0.0095 nats = wikitext: base ppl 6.437 → ±0.061 ppl; c4val1: base ppl 16.597 → ±0.158 ppl. No NF4 speed arm on this lane, so no `× base`.
| arm | configuration | cut | K8 wikitext nll | Δppl (Δnats) | gate | K8 c4val1 nll | Δppl (Δnats) | gate | registered verdict (ppl) | B=1 ms | B=1 tok/s | × base | B=16 tok/s | × base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention (baseline; wikitext row from bo3, same window sha) | i6 | — | — | — | 2.80923 | — | baseline | baseline | — | — | — | — | — |
| `all` | int4 experts + C4-calibrated int4 attention + r1 + r2 (rope-only fold) + router epilogue + #385 glue | i6 | 1.84571 | -0.1048 (-0.0164 1.7× floor) ᵇ | pass | 2.81299 | +0.0627 (+0.0038 sub-floor) | FAIL | FAIL as registered (c4val1 +0.0627) | 4.90 | 204.1 | — | 1251.6 | — |
| `all_noglue` | all with E4B_FUSE_SWIGLU=0 E4B_FUSE_COMBINE=0 (the #385 A/B arm) | i6 | — | — | — | — | — | — | — | 5.18 | 193.1 | — | — | — |
| `all_cen` | all, kernel census run | i6 | — | — | — | — | — | — | — | 4.89 | 204.5 | — | — | — |
| `all_noglue_cen` | all_noglue, kernel census run | i6 | — | — | — | — | — | — | — | 5.18 | 193.1 | — | — | — |
| `int4exp` | int4 experts only (attribution, second text) | i7 | — | — | — | 2.81349 | +0.0709 (+0.0043 sub-floor) | FAIL | FAIL as registered (c4val1 +0.0709) | — | — | — | — | — |
| `calib` | int4 experts + C4-calibrated int4 attention (attribution, second text) | i7 | — | — | — | 2.81209 | +0.0476 (+0.0029 sub-floor) | pass | one text (needs 2) | — | — | — | — | — |
| `folds` | r1 + r2 + epilogue only, NF4 experts (attribution, second text; exact arithmetic) | i7 | — | — | — | 2.80482 | -0.0730 (-0.0044 sub-floor) | FAIL | FAIL as registered (c4val1 -0.0730 by improving) | — | — | — | — | — |
| `int4folds` | int4 experts + r1 + r2 + epilogue, no calibrated attention (attribution, second text) | i7 | — | — | — | 2.80812 | -0.0183 (-0.0011 sub-floor) | pass | pass (1 text) | — | — | — | — | — |

Pairs on this lane (Qwen3-30B-A3B; B=1 ratio = ms(den)/ms(num), B=16 ratio = tok/s(num)/tok/s(den)):
- `all` vs `all_noglue` — #385 swiglu_rows + combine_rows glue on vs off: B=1 5.18 → 4.90 ms = ×1.057
- `all_cen` vs `all_noglue_cen` — same pair, census runs: B=1 5.18 → 4.89 ms = ×1.059

Attribution on c4val1 (nats vs NF4): int4 experts +0.0043; calibrated attention on top -0.0014; exact folds -0.0044; sum -0.0015 vs measured `all` +0.0038 — non-additive by 0.0053 nats (floor 0.0095).

Kernel census (Qwen3-30B-A3B, `--replay-profile-out`, Self CUDA over the profiled replay steps; launches = CUDA-kernel rows' calls per step):
- `census_qwen3_all_cen.txt`: 8 steps, Self CUDA total 34.516 ms (4.314 ms/step), 1793 launches/step; timed wall 4.89 ms/step
- `census_qwen3_all_noglue_cen.txt`: 8 steps, Self CUDA total 36.457 ms (4.557 ms/step), 1937 launches/step; timed wall 5.18 ms/step

### Granite-3.1-3B-A800M
arithmetic-order floor 0.0033 nats = wikitext: base ppl 5.328 → ±0.018 ppl; c4val1: base ppl 11.572 → ±0.038 ppl. `× base` = ratio to `nf4_r12epi` on this lane.
| arm | configuration | cut | K8 wikitext nll | Δppl (Δnats) | gate | K8 c4val1 nll | Δppl (Δnats) | gate | registered verdict (ppl) | B=1 ms | B=1 tok/s | × base | B=16 tok/s | × base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nf4_r12epi` | NF4 experts + r1 + r2 (rotary-only fold, #379) + epilogue — the licensed stack (baseline) | i6 | 1.67304 | — | baseline | 2.44857 | — | baseline | baseline | 3.40 | 294.1 | — | 1736.1 | — |
| `calibexp_r12epi` | licensed stack with C4-calibrated (GPTQ) int4 experts, #384: 2524 gptq / 36 rtn | i6 | 1.67567 | +0.0140 (+0.0026 sub-floor) | pass | 2.48148 | +0.3872 (+0.0329 10.0× floor) | FAIL | FAIL as registered (c4val1 +0.3872) | 2.44 | 409.8 | 1.393 | 3050.1 | 1.757 |
| `rtnexp_r12epi` | licensed stack with uncalibrated (RTN) int4 experts | i6 | 1.68487 | +0.0634 (+0.0118 3.6× floor) | FAIL | — | — | — | FAIL as registered (wikitext +0.0634) | — | — | — | — | — |
| `nf4_r12epi_fq` | licensed stack without --no-fuse-qkv on integration-6 (no-op control) | i6 | 1.67304 | +0.0000 (+0.0000 sub-floor) | pass | — | — | — | pass (1 text) | 3.40 | 294.1 | 1.000 | — | — |
| `nf4_r12epi_i7` | licensed stack on integration-7 (unfused control for #387) | i7 | — | — | — | — | — | — | — | 3.38 | 295.9 | 1.006 | 1724.6 | 0.993 |
| `nf4_r12epi_fq2` | licensed stack + #387 fused qkv (32 modules) + fused rope-only fold, integration-7 | i7 | 1.67198 | -0.0056 (-0.0011 sub-floor) | pass | — | — | — | pass (1 text) | 3.49 | 286.5 | 0.974 | 1748.4 | 1.007 |
| `nf4_r12epi_i7b` | second unfused control | i7 | — | — | — | — | — | — | — | 3.48 | 287.4 | 0.977 | — | — |
| `nf4_r12epi_fq2b` | second #387 arm | i7 | — | — | — | — | — | — | — | 3.50 | 285.7 | 0.971 | — | — |
| `nf4_r12epi_i7_cen` | unfused, kernel census run | i7 | — | — | — | — | — | — | — | 3.40 | 294.1 | 1.000 | — | — |
| `nf4_r12epi_fq2_cen` | #387 fused, kernel census run | i7 | — | — | — | — | — | — | — | 3.50 | 285.7 | 0.971 | — | — |

Pairs on this lane (Granite-3.1-3B-A800M; B=1 ratio = ms(den)/ms(num), B=16 ratio = tok/s(num)/tok/s(den)):
- `calibexp_r12epi` vs `nf4_r12epi` — C4-calibrated int4 experts (#384) vs NF4 experts — speed only, the K8 verdict is FAIL: B=1 3.40 → 2.44 ms = ×1.393; B=16 1736.1 → 3050.1 tok/s = ×1.757; K8 wikitext +0.0026 nats (+0.0140 ppl)
- `nf4_r12epi_fq` vs `nf4_r12epi` — fused-qkv flag on integration-6: no-op control: B=1 3.40 → 3.40 ms = ×1.000; K8 wikitext +0.0000 nats (+0.0000 ppl)
- `nf4_r12epi_fq2` vs `nf4_r12epi_i7` — #387 fused qkv + fused rope-only fold vs unfused, integration-7: B=1 3.38 → 3.49 ms = ×0.968; B=16 1724.6 → 1748.4 tok/s = ×1.014
- `nf4_r12epi_fq2b` vs `nf4_r12epi_i7b` — second pair: B=1 3.48 → 3.50 ms = ×0.994
- `nf4_r12epi_fq2_cen` vs `nf4_r12epi_i7_cen` — census pair: B=1 3.40 → 3.50 ms = ×0.971

Kernel census (Granite-3.1-3B-A800M, `--replay-profile-out`, Self CUDA over the profiled replay steps; launches = CUDA-kernel rows' calls per step):
- `census_granite_nf4_r12epi_fq2_cen.txt`: 8 steps, Self CUDA total 26.031 ms (3.254 ms/step), 786 launches/step; timed wall 3.50 ms/step
- `census_granite_nf4_r12epi_i7_cen.txt`: 8 steps, Self CUDA total 27.099 ms (3.387 ms/step), 850 launches/step; timed wall 3.40 ms/step

### gpt-oss-20b
arithmetic-order floor 0.0176 nats. `× base` = ratio to `r12` on this lane.
| arm | configuration | cut | K8 wikitext nll | Δppl (Δnats) | gate | K8 c4val1 nll | Δppl (Δnats) | gate | registered verdict (ppl) | B=1 ms | B=1 tok/s | × base | B=16 tok/s | × base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r12` | NF4 experts + r1 + r2 folds (baseline; bo3's licensed row) | i6 | — | — | — | — | — | — | — | 7.33 | 136.4 | — | 741.4 | — |
| `store_r12` | native MXFP4 store: gemv_mxfp4_b32 for single rows, NF4 kept for batched rows (E4B_INT4_KEEP_NF4=1) + r1 + r2 | i6 | — | — | — | — | — | — | — | 5.77 | 173.3 | 1.270 | 719.7 | 0.971 |
| `store_r12_cen` | store_r12, kernel census run | i6 | — | — | — | — | — | — | — | 5.77 | 173.3 | 1.270 | — | — |

Pairs on this lane (gpt-oss-20b; B=1 ratio = ms(den)/ms(num), B=16 ratio = tok/s(num)/tok/s(den)):
- `store_r12` vs `r12` — MXFP4 store route (GEMV single rows, NF4 batched rows) vs NF4 experts: B=1 7.33 → 5.77 ms = ×1.270; B=16 741.4 → 719.7 tok/s = ×0.971

Kernel census (gpt-oss-20b, `--replay-profile-out`, Self CUDA over the profiled replay steps; launches = CUDA-kernel rows' calls per step):
- `census_gptoss_store_r12_cen.txt`: 8 steps, Self CUDA total 42.989 ms (5.374 ms/step), 1288 launches/step; timed wall 5.77 ms/step

### Mixtral-8x7B-Instruct
arithmetic-order floor unmeasured. `× base` = ratio to `all` on this lane.
| arm | configuration | cut | K8 wikitext nll | Δppl (Δnats) | gate | K8 c4val1 nll | Δppl (Δnats) | gate | registered verdict (ppl) | B=1 ms | B=1 tok/s | × base | B=16 tok/s | × base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nf4` | NF4 experts, bf16 attention (baseline; wikitext row = P30 NF4 from bo3, same window sha) | i6 | — | — | — | 2.11031 | — | baseline | baseline | — | — | — | — | — |
| `nf4_i7` | NF4 experts, bf16 attention, re-baked on integration-7 (wikitext control for the i7 arms) | i7 | 1.18105 | — | baseline | — | — | — | baseline | — | — | — | — | — |
| `all` | int4 experts + C4-calibrated int4 attention + r1 + r2 (rope-only fold) + epilogue | i6 | 1.19051 | +0.0328 (+0.0100) ᵇ | pass | 2.12424 | +0.1157 (+0.0139) | FAIL | FAIL as registered (c4val1 +0.1157) | 8.11 | 123.3 | — | 377.3 | — |
| `all_nor2` | all without the round-2 fold (E4B_FUSE_T1_GLUE_R2=0) | i6 | — | — | — | — | — | — | — | 8.61 | 116.1 | 0.942 | 374.5 | 0.992 |
| `all_i7` | all on integration-7 (control for #387) | i7 | — | — | — | — | — | — | — | 8.11 | 123.3 | 1.000 | — | — |
| `all_fq` | all + #387 fused qkv: 0 modules fused (calibrated attention children are not Linear) | i7 | 1.19051 | +0.0310 (+0.0095) | pass | — | — | — | one text (needs 2) | 8.11 | 123.3 | 1.000 | 377.1 | 0.999 |
| `lic` | int4 experts (RTN) + r1 + r2 + epilogue, no calibrated pack — the P30 'licensed stack' | i7 | 1.16684 | -0.0460 (-0.0142) | pass | 2.11726 | +0.0575 (+0.0070) | FAIL | FAIL as registered (c4val1 +0.0575) | 8.93 | 112.0 | 0.908 | 376.5 | 0.998 |
| `lic_fq` | lic + #387 fused qkv (32 modules) + fused rope-only fold | i7 | 1.16702 | -0.0454 (-0.0140) | pass | — | — | — | pass (1 text) | 8.65 | 115.6 | 0.938 | 378.3 | 1.003 |

Pairs on this lane (Mixtral-8x7B-Instruct; B=1 ratio = ms(den)/ms(num), B=16 ratio = tok/s(num)/tok/s(den)):
- `all` vs `all_nor2` — rope-only round-2 fold (#379) on vs off, calibrated stack: B=1 8.61 → 8.11 ms = ×1.062; B=16 374.5 → 377.3 tok/s = ×1.008
- `all_fq` vs `all_i7` — #387 on the calibrated stack (0 modules fused): B=1 8.11 → 8.11 ms = ×1.000
- `all_fq` vs `all` — the same #387 arm's K8 against integration-6 `all` (0 modules fused: identical numerics expected): B=1 8.11 → 8.11 ms = ×1.000; B=16 377.3 → 377.1 tok/s = ×0.999; K8 wikitext +0.0000 nats (+0.0000 ppl)
- `lic_fq` vs `lic` — #387 on the licensed int4-expert stack (32 modules fused): B=1 8.93 → 8.65 ms = ×1.032; B=16 376.5 → 378.3 tok/s = ×1.005; K8 wikitext +0.0002 nats (+0.0006 ppl)
- `all` vs `lic` — calibrated attention on top of the int4-expert stack (integration-6 vs -7): B=1 8.93 → 8.11 ms = ×1.101; B=16 376.5 → 377.3 tok/s = ×1.002; K8 wikitext +0.0237 nats (+0.0769 ppl)

ᵇ = baseline row taken from the `--ref` bundle (bo3) on the same window sha; every other delta is against an arm on this lane.
Gate cells and the registered verdict are in perplexity, the registered unit; the nats beside them are read against the family's arithmetic-order floor and never change the verdict. `one text (needs 2)` = a calibrated pack within +0.05 ppl on the one text scored.

## Reading it

- **Qwen3-30B-A3B (the reference), integration-6.** `all` (int4 experts + C4-calibrated int4 attention +
  round-1/2 folds + router epilogue, now with #385's `swiglu_rows` / `combine_rows` glue) runs **4.90 ms =
  204.1 tok/s at B=1 and 1251.6 tok/s at B=16** on this box; the glue is ×1.057 at B=1 (`all_noglue`,
  `E4B_FUSE_SWIGLU=0 E4B_FUSE_COMBINE=0`, 5.18 ms; the census pair agrees at ×1.059 — 1937 → 1793 launches
  per step, Self CUDA 36.457 → 34.516 ms per 8 replayed steps). wikitext K8 1.84571 (ppl 6.33260; −0.105 ppl
  against bo3's NF4 on the same window sha). **The second text says FAIL as registered:** c4val1 NF4 2.80923
  (ppl 16.59705), `all` 2.81299 = **+0.0627 ppl, +0.0038 nats**, over the +0.05 budget. Attribution on the
  same text (bo5b, integration-7): int4 experts alone +0.0709 ppl (FAIL), int4 experts + calibrated attention
  +0.0476 (pass), the exact folds alone −0.0730 (FAIL as registered — by *improving*, on arithmetic that moves
  no weight), int4 experts + folds −0.0183 (pass). Every one of those is inside the family's 0.0095-nat floor
  (±0.158 ppl at this text's perplexity) and they are non-additive by 0.005 nats: the +0.063 attributes to
  noise, not to a component. The verdict stands as registered and the gate is not retuned to fit; the `all`
  stack is NOT licensed under the rule as written, and 204.1 / 1251.6 are measured speed of an unlicensed
  configuration until the user decides whether to re-register the gate in nats against the measured floor.
- **Granite-3.1-3B-A800M.** The licensed stack `nf4_r12epi` (NF4 experts + round-1/2 folds with the
  rotary-only fold + router epilogue) re-measures at **3.40 ms = 294.1 tok/s B=1, 1736.1 B=16** (wikitext
  1.67304, ppl 5.32834; bo3 on its box: 259.1 / 1689.6). **C4-calibrated int4 experts (#384 draft; 2524 gptq /
  36 rtn) FAIL the second text:** wikitext +0.0140 ppl (pass, sub-floor) but c4val1 2.48148 (ppl 11.95897)
  against NF4 2.44857 (11.57180) = **+0.3872 ppl, 10× the floor** — FAIL. Its speed is measured and refused:
  2.44 ms = 409.8 tok/s B=1, 3050.1 B=16. Uncalibrated int4 experts (`rtnexp_r12epi`) reproduce bo3's
  retraction: +0.0634 ppl, FAIL. `nf4_r12epi_fq` (the `--no-fuse-qkv` flag dropped on integration-6, before
  #387) is a bit-identical no-op control (3.40 ms, K8 identical). **#387 (bo5b/bo5c, integration-7):** fused
  q/k/v on 32 modules + the fused rope-only fold is quality-clean (K8 1.67198, −0.0011 nats, pass) and buys
  nothing: three B=1 pairs ×0.968 (3.38 → 3.49 ms), ×0.994 (3.48 → 3.50), ×0.971 (3.40 → 3.50); B=16
  1724.6 → 1748.4 (×1.014, one pair). The census pair explains the direction of the GPU time, not the sign
  of the wall: Self CUDA falls 27.099 → 26.031 ms per 8 replayed steps (−4%, 850 → 786 launches per step)
  while the timed wall rises 3.40 → 3.50 ms (+3%) — unexplained non-GPU time on the fused path. #387 stays a
  draft.
- **gpt-oss-20b.** `r12` (NF4 experts + round-1/2 folds, bo3's licensed row) 7.33 ms = 136.4 tok/s B=1,
  741.4 B=16. The route rule `store_r12` — the native MXFP4 store through `gemv_mxfp4_b32` for single rows
  with NF4 kept for batched rows (`E4B_INT4_KEEP_NF4=1`) — is **5.77 ms = 173.3 tok/s (×1.270) at B=1 and
  719.7 (×0.971) at B=16**: the B=16 penalty of the GEMV-everywhere route (bo3: 589.7 against 726.2, ×0.81)
  is recovered to ×0.971. No K8 arm: the store is exact against the checkpoint's own bytes (bo3q), and this
  family has no raw-text instrument.
- **Mixtral-8x7B-Instruct.** `all` (int4 experts + calibrated attention + r1 + r2 + epilogue), integration-6:
  wikitext 1.19051 (ppl 3.28876), +0.0328 ppl against P30's NF4 1.18048 / 3.25594 (pass); **c4val1 2.12424
  (8.36651) against NF4 2.11031 (8.25079) = +0.1157 ppl — FAIL as registered.** 8.11 ms = 123.3 tok/s B=1,
  377.3 B=16; without the round-2 fold (`all_nor2`) 8.61 ms = 116.1 / 374.5 — the rope-only fold is ×1.062
  at B=1 on this family. bo5b: `all_i7` 8.11 ms (the control on integration-7); `all_fq` 8.11 / 377.1 with
  **0 modules fused** — the calibrated attention's children are not `Linear`, so #387's guard is correct —
  and K8 identical to `all` (1.19051). bo5c, integration-7: NF4 control 1.18105 (3.25778); `lic` = int4
  experts (RTN) + r1 + r2 + epilogue with no calibrated pack — the stack P30 and bo3 called "licensed"
  (bo3's uncalibrated `stack`, −0.037 ppl on wikitext): wikitext 1.16684 (3.21181, −0.046 ppl, pass),
  **c4val1 2.11726 (8.30834) = +0.0575 ppl — FAIL as registered, by 0.008** (+0.0070 nats; this family's
  arithmetic-order floor is unmeasured, so the nats cannot be read against one). **The P30 "licensed stack"
  label is WITHDRAWN under the registered rule**, pending the user's decision on re-registering the gate in
  nats against a measured floor; the gate is not retuned to fit. Its speed: 8.93 ms = 112.0 tok/s B=1,
  376.5 B=16; with #387 (`lic_fq`, 32 modules fused) 8.65 ms = 115.6 (×1.032), 378.3, K8 1.16702
  (+0.0002 nats against `lic`).

## Consequences

1. **Granite's calibrated int4 experts route (e4b#384, draft) FAILS the second text** (+0.387 ppl on C4
   validation at 10× the family's floor, after +0.014 on wikitext). The licensed row stays NF4 experts +
   folds + epilogue: 294.1 / 1736.1 on this box.
2. **Mixtral's int4-expert stack is one-text pass / one-text FAIL-as-registered** (−0.046 on wikitext,
   +0.0575 on C4 validation against a +0.05 budget). The "licensed" label from P30/bo3 is withdrawn pending
   the user's decision on re-registering the gate in nats against a measured floor; the gate is not retuned.
   112.0 / 376.5 is measured speed of a configuration that is NOT licensed under the rule as written.
3. **Qwen3's `all` second text is +0.063 ppl — FAIL as registered — and attributes to noise:** every
   attribution arm on that text is inside the 0.0095-nat floor, the exact folds alone read −0.073, and the
   components are non-additive by 0.005 nats. Same disposition as Mixtral: 204.1 / 1251.6 is measured speed
   of a configuration NOT licensed under the rule as written, user decision pending.
4. **gpt-oss's route rule** (MXFP4 GEMV for single rows, NF4 kept for batched rows) **recovers B=16 from
   ×0.81 to ×0.971** while holding ×1.270 at B=1 (173.3 / 719.7); quality gate open on this family as before.
5. **#387's fused q/k/v is quality-clean and buys nothing** (Granite ×0.968 B=1 across three pairs with
   GPU time −4% and wall +3%; Mixtral ×1.032 on the licensed stack; ×1.000 on the calibrated stack, where its
   guard correctly fuses 0 modules) — it stays a draft.
