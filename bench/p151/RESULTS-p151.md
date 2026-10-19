# P151 results: D1's served read, K35's dense GEMV on Qwen3-30B-A3B's attention projections (e4b#1313)

**Status:** closed at **NOISY** (a VOID and the lane's strike one), by the maintainer's ruling: no re-run, and
`E4B_DENSE_GEMV_BW` stays opt-in. The rule takes the minimum of four W1 pairs. The three clean pairs top out at 1.0247,
so that minimum cannot reach 1.03 whatever the noisy arm read, and a re-run would spend a strike to read a NO_FLIP.

[The registration](PREREG-p151.md) ran P143 Amendment 3's served form with the knob in place of the split cap.
[Amendment 1](PREREG-p151.md) (#1819) added spares for refusals before the install.

| run | what spent it | cost | outcome |
|---|---|---|---|
| [`p151-5090-1`](receipts/p151-5090-1/README.md) | the box's power floor: an unrecorded host read 520 W (rc 17) | $0.015 | host refusal before the install; not a strike |
| [`p151-5090-2`](receipts/p151-5090-2/README.md) | the provider: the offer was taken before the create (`no_such_ask`) | $0 | refusal before any instance; not a strike |
| [`p151-5090-3`](receipts/p151-5090-3/README.md) | the box's power floor: an unrecorded host read 520 W (rc 17) | $0.020 | host refusal before the install; not a strike |
| [`p151-5090-4`](receipts/p151-5090-4/README.md) | the served read: eight arms and SANE | $0.429 | **NOISY**: a VOID and strike one |

Total $0.464. The maintainer re-derived `verdict_p151.json` byte for byte from the private receipt store (the reducer
at `63b21b13`, equal to the staged pin and the box's copy; self-test 47/47).

## The read (`p151-5090-4`)

- **The box:** one RTX 5090 at a 600 W power limit, e4b `63b21b13`, grouped-nf4-gemm `9b824155` (K35's merge). The
  premise read 31 passed, K35's compiled contract on the 5090 among them.
- **On shape:** all eight arms and both SANE phases completed. S1 routed all 96 fused `qkv_proj` / `o_proj` modules at
  one row with no one-row fallback; S0 swapped nothing. The split cap read its qwen3_moe default in every record.
  Every timed block read the 600 W limit either side of it.
- **Why NOISY:** the W1 self-pair S0b/S0a read 1.0345, outside [0.985, 1.015]. S0a, the box's first process, ran slow:
  - W1 at 252.86 tok/s, against 261.59 / 261.51 / 261.59 for S0b, S0c and S0d;
  - its model load took 50.5 s, against about 31 s for every later arm;
  - its timed blocks show no throttle (SM clocks 2917-2925 MHz, 51-55 C).

  Every other W1 self-pair agrees within 0.06 % (the W16 self-pairs within 0.11 %). P143 A3 ran the same form clean on another host. P131 met the same kind
  of first-process effect with the opposite sign (its first process 4.7 % fast) and answered it with an unscored
  warm-up process 0 (P131 Amendment 2).

**Context, never a reading:** about **+2.45 % W1 decode** in three clean ABBA pairs on one host (S1b/S0b 1.0244,
S1c/S0c 1.0246, S1d/S0d 1.0247), under the 1.03 bar. P148 projected D1 at 0.131137 ms, 3.35 % of the unprofiled
3.913 ms step, so the clean pairs realize about **0.73** of the microbench projection. Also reported:
- W16 unchanged (pairs 0.9999-1.0013); decode at 2+ rows routes nothing, so only the W16 ramp's one-row steps can move.
- SANE inside A2's band: bias +0.000457, argmax 0.9681, KL 0.0081, computed from `sane_off.json` and `sane_on.json`
  with the reducer's own formula (the reducer stops at NOISY before it reaches SANE).
- W1 tokens: S1 identical to S0 at both lengths.

## Predictions (graded, never ruling)

| prediction | read | grade |
|---|---|---|
| every W1 pair in [1.015, 1.040] | three clean pairs 1.0244-1.0247; S1a/S0a 1.0603 (the slow first arm) | **missed** on one pair |
| g16 in [0.99, 1.01] | 0.9999-1.0013 | hit |
| every W1 self-pair within [0.99, 1.01] | S0b/S0a 1.0345; the others within 0.0006 | **missed** |
| SANE \|bias\| <= 0.002 and argmax >= 0.99 | bias +0.000457; argmax 0.9681 | **missed** on argmax |
| S1's W1 tokens differ from S0's within 160 | identical at both lengths | **missed** |
| verdict NO_FLIP | NOISY, no reading | not reached |

## What it starts

- **`E4B_DENSE_GEMV_BW` stays opt-in.** P151 is closed; there is no re-run.
- **P153** (D2's served read, the int8 lm_head through K36) carries P131 A2's warm-up process 0, written into its
  registration before any data: an S0 build and speed pass before S0a, reported only, never in a pair, a self-pair or a
  bar.
- **Hosts:** both 5090s drawn without a recorded power cap (`-1` and `-3`) read 520 W; `-4`'s host had a recorded cap.
  The launcher now records each host's cap (adertha-agents#268), and a manifest can walk only hosts with a recorded
  cap of at least the floor (`vast_power_limit_unknown: exclude`, #269), which `p151-5090-4` used.
