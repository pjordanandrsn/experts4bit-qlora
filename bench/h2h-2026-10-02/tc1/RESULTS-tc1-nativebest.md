# TC1 native-best against native-best (amendments 5-9: P13, P14 and P15)

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendments 5 (the token and P13), 6 (the scattermoe second
draw's Hub rule) and 7 (the reducer registration, the clock sampler, and the whole token again). Receipts:
[`receipts/tc1-5090-33/`](receipts/tc1-5090-33/) and [`receipts/tc1-5090-34/`](receipts/tc1-5090-34/). Every verdict, ratio and
interval below is the reducer's ([`../../tc1/tc1_reduce.py`](../../tc1/tc1_reduce.py)). `tc1-5090-34`'s box-side
`RESULTS-tc1.md` already carries the registration. `tc1-5090-33` ran before it, so its re-reduction is
[`RESULTS-tc1-5090-33-rereduced-2026-10-02.md`](RESULTS-tc1-5090-33-rereduced-2026-10-02.md).

Each framework runs as its users run it, which is **not matched work**. e4b as shipped uses bf16 expert adapters and its own
N(0, 1/r) init. axolotl 0.20.0 uses its scattermoe native-best (KernelsPlugin) with its own init and fp32 adapters. Unsloth 2026.9.14
uses grouped_mm, the speed tilt and its own init. All three train the same 642,514,944 parameters on the same tokens (sha
`bfc742f67e37`) at the field recipe for N = 20, on one RTX 5090 per box. e4b's matched fused arm runs last on each box as the anchor for
the validity predicates. Held-out losses are reported beside the times and never judged across different inits.

## P14 on `tc1-5090-35`: at steady state axolotl's scattermoe steps 9 % faster than e4b as shipped

Amendment 8 re-asked P13's axolotl half over 200 steps of the field recipe, read on steps 101..200. The box was instance 53935033
(Intel Xeon E5-2698 v4, driver 595.71.05), e4b `7c31b88`, $1.76, with receipts in [`receipts/tc1-5090-35/`](receipts/tc1-5090-35/).
Every arm is VALID, and the reducer's box-side `RESULTS-tc1.md` scores P14.

| arm | late window, median of steps 101..200 (s) | median of steps 11..200 (s) | sum of all 200 steps (s) | peak VRAM | held-out at 200 |
|---|---|---|---|---|---|
| e4b shipped, draw 1 / 2 | 6.637 / 6.757 | 6.739 / 6.675 | 1,372 / 1,353 | 25.51 GB | 0.7926 / 0.7958 |
| axolotl scattermoe native-best, draw 1 / 2 | 6.114 / 6.092 | 6.169 / 6.026 | 2,023 / 1,934 | 26.77 GB | 0.7696 / 0.7695 |
| e4b matched fused (anchor, one draw) | 7.556 | 8.048 | 1,649 | 29.15 GB | 0.7691 |

- **P14 is HELD.** axolotl scattermoe / e4b shipped is **0.911 [0.902, 0.921]** on the late window, inside [0.90, 1.10]. The late
  medians agree within 1.8 % (e4b) and 0.4 % (axolotl).
- **The ordering reading favours axolotl.** The whole interval lies below 1.0, so axolotl's scattermoe native-best steps about 9 %
  faster than e4b as shipped at steady state on this card and host. The 11..200 medians read the same: 0.909 [0.894, 0.924]. Under
  amendment 8's decision rule this is the finding, and the claims register says so (`e4b.train.h2h.axolotl.qwen3.5090.2026-10-02.native-steady-state`).
- **Over the whole run, e4b finishes first.** axolotl's first steps take 359 / 324, 57 / 53 and 103 / 102 s (steps 1-3). Counting a spike as a step
  above 1.5× the draw's late median, draw 1 spikes on 20 steps and draw 2 on 19. Eighteen of those steps are the same in both draws (1, 2,
  3, 5, 6, 11, 13, 18, 20, 22, 32, 52, 63, 68, 72, 117, 144 and 191), and three fall inside the late window. Summed over 200 steps, its step time is 1,934-2,023 s against e4b shipped's 1,353-1,372 s.
- **Energy and memory.** axolotl spends ×1.32 the energy per step (1,242 against 940 J). This is net-of-idle power times the whole
  run's mean step time, so it includes the warm-up. Its peak VRAM is 1.25 GB higher.
- **Held-out at step 200.** axolotl's native configuration reaches the box's matched e4b curve (0.7696 / 0.7695 against 0.7691). e4b as
  shipped sits 0.024-0.027 above it, which reproduces TC1b's as-shipped plateau (+0.0258, `.plateau-shipped-n200`). These are reported,
  not judged: each framework uses its own init.
- **Clocks.** The SM clock averages 2,400-2,406 MHz on every arm. A software power-cap flag (`0x4`) appears in some axolotl samples
  only, and one transient sample dips to 1,327 MHz. Dynamo records no recompile on any arm.

The spikes landing on the same steps in both draws fit the warm-up hypothesis: a per-process cost keyed on batch shape, such as Triton
autotuning in the scattermoe kernels. The cause is still not measured. This is one host. On `tc1-5090-34`'s faster host, the warm steps
14-16 put axolotl at 0.94-0.99 of e4b shipped, the same direction.

## P15 on `tc1-5090-36`: the steady-state ordering replicates on a second host

Amendment 9 asked P14 again on a different machine, with the harness and package byte-identical to `tc1-5090-35`'s (e4b pinned at
`7c31b88`). The box was instance 53971676 on Vast machine 45501, an AMD Ryzen 9 3900X with driver 595.71.05; `tc1-5090-35` ran on
machine 96642, an Intel Xeon E5-2698 v4. It cost $0.93, with receipts in [`receipts/tc1-5090-36/`](receipts/tc1-5090-36/). Every arm is
VALID.

| arm | late window, median of steps 101..200 (s) | median of steps 11..200 (s) | sum of all 200 steps (s) | held-out at 200 |
|---|---|---|---|---|
| e4b shipped, draw 1 / 2 | 3.705 / 3.760 | 3.701 / 3.765 | 743 / 752 | 0.7930 / 0.7936 |
| axolotl scattermoe native-best, draw 1 / 2 | 3.371 / 3.356 | 3.371 / 3.351 | 1,231 / 1,219 | 0.7706 / 0.7703 |
| e4b matched fused (anchor, one draw) | 4.465 | 4.617 | 938 | 0.7675 |

- **P15 is HELD.** The box's P14 line reads **0.901 [0.892, 0.910]**. That is inside [0.90, 1.10], the whole interval lies below 1.0,
  and the machine differs from 96642. The late medians agree within 1.5 % (e4b) and 0.5 % (axolotl).
- **The ordering holds on two hosts, and the size sits at the band's edge.** axolotl's scattermoe steps about 10 % faster than e4b as
  shipped here, against 9 % on the Xeon. The point ratio is 0.001 above the band's lower bound: a slightly larger axolotl lead would have
  left the band and FALSIFIED P15, with axolotl still faster. The 11..200 medians read 0.900 [0.890, 0.911].
- **Over the whole run, e4b finishes first again.** axolotl's first steps take 231 s, 39 s and 67-68 s, and it spikes on 20-21 of 200
  steps, mostly the same step numbers as on the Xeon. Its summed step time is 1,219-1,231 s against e4b shipped's 743-752 s.
- **Energy and memory.** axolotl spends ×1.39 the energy per step (1,279 against 922 J, whole-run, net of idle) at 1.25 GB more peak.
- **Held-out at step 200.** axolotl's native configuration sits within 0.003 of the box's matched e4b curve. e4b as shipped sits
  0.026 above it, the TC1b plateau again.
- **Clocks.** The SM clock averages 2,751-2,832 MHz, and the software power-cap flag again appears only in the axolotl arms.

Both hosts' steps are far apart in absolute terms: e4b shipped takes 3.7 s here against 6.6-6.8 s on the Xeon. The ratio is close on
both: 0.901 and 0.911. Per amendment 9, the two boxes' numbers are not divided into each other.

## P13 is scored on `tc1-5090-34` and is UNTESTED

`tc1-5090-34`: instance 53927504, AMD EPYC 9334, driver 590.48.01, e4b `2ad3187`, $0.44. Every arm is VALID.

| arm | s/step, draws 1 / 2 (median 11..20) | stability | peak VRAM | held-out at 20, draws 1 / 2 |
|---|---|---|---|---|
| e4b shipped | 3.277 / 3.279 | STABLE, 0.1 % | 24.58 GB | 0.8137 / 0.8153 |
| axolotl scattermoe native-best | 4.574 / 4.107 | **UNSTABLE, 10.8 %** | 25.84 GB | 0.8304 / 0.8331 |
| Unsloth native-best | 5.869 / 5.889 | STABLE, 0.3 % | 24.84 GB | 0.8442 / 0.8466 |
| e4b matched fused (anchor, one draw) | 4.225 | — | 27.82 GB | 0.8479 |

- **The Unsloth half is HELD.** Unsloth native-best / e4b shipped is 1.794, with a cross-draw interval of [1.790, 1.797]. e4b as
  shipped steps 1.79× faster than Unsloth's native-best configuration on this card, at 0.25 GB lower peak VRAM.
- **The axolotl half is UNTESTED.** The scattermoe draws differ by 10.8 % against the 5 % rule. Every cross-draw ratio lies above 1.0
  (1.25-1.40), but an unstable pair is reported, never quoted.
- **P13 is therefore UNTESTED.** No "fastest native configuration" statement is made for e4b. Amendment 7 makes this reading final
  for P13.

## Why the scattermoe draws disagree (`tc1-5090-34`, per step on identical tokens)

| step | 1 | 3 | 6 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| axolotl draw 1 (s) | 216.6 | 60.8 | 81.4 | 5.39 | 4.23 | 4.92 | 3.46 | 2.99 | 2.99 | 3.77 | 5.85 | 5.06 | 7.34 |
| axolotl draw 2 (s) | 215.4 | 65.0 | 76.2 | 5.03 | 4.70 | 4.72 | 3.02 | 2.99 | 2.98 | 3.76 | 5.42 | 4.40 | 3.81 |
| e4b shipped, faster draw (s) | 3.09 | 2.96 | 3.22 | 3.24 | 3.14 | 3.31 | 3.11 | 3.19 | 3.02 | 3.50 | 3.48 | 3.44 | 3.51 |

The large steps fall on the same steps in both draws, and Dynamo records no recompile and no graph break in either. On steps 14-16
the scattermoe arm runs at 0.94-0.99 of e4b shipped. This is consistent with a per-process kernel warm-up keyed on batch shape, such
as Triton autotuning in the scattermoe kernels. **That cause is not measured.** A 20-step window cannot separate such a warm-up from
the steady step, so amendment 8 asks the question again over steps 101..200 of a 200-step run (P14).

The new per-arm clock record (`gpuclk_*.txt`) shows the card at about 2.68 GHz SM clock, at most 65 °C, with no clock-event
(throttle) reason set during the e4b-shipped arm.

## The first box, `tc1-5090-33`

Instance 53916526, AMD EPYC 7663, driver 580.95.05, e4b `bb32ea8`, $0.51. The box ran before the reducer registered the token, and
its own `RESULTS-tc1.md` reads the anchor VOID for that reason alone. Re-reduced, every arm that trained is VALID, and P13 is UNTESTED
on both halves:

- **axolotl.** The second scattermoe draw refused with the Hub offline, a harness defect fixed by amendment 6.
- **e4b shipped.** The two draws ran 4.306 and 4.578 s/step, 6.1 % apart. The second draw was 5.5-8.5 % slower on every one of steps
  11-20, on identical tokens and at lower mean power, and the box recorded no clocks to say why. Unsloth's pair on the same box agreed
  within 0.5 % (7.938 / 7.897).

The same arms took 24-28 % less time per step on `tc1-5090-34`'s host (EPYC 9334) than on this one (EPYC 7663): e4b shipped 3.28
against 4.31-4.58 s, and Unsloth 5.88 against 7.92 s. The lead has the same direction on both hosts. This host spread is why every TC1 ratio is
formed within one box, and why no number from one box divides a number from another.
