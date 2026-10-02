# TC1 native-best against native-best (amendments 5-7, P13): two boxes, one scored

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
