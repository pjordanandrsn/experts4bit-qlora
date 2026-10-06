### TC1 amendment 53 registered: where torch 2.8's extra time goes at e4b's defaults on packed rows (P134-P136)

- Token `qwen3prof28` profiles e4b's matched arm on packed rows at its defaults. It runs in torch 2.12 and torch 2.8, plus torch 2.8 with
  the single block, two draws each. Steps 3-5 are profiled; the timed window stays steps 11..40.
- P134: environment ratio <= 0.92. P135: at least half of torch 2.8's added profiled wall is not device time. P136: in torch 2.8 the
  buckets lower the device busy fraction by >= 0.03.
- The reducer adds the family, its engagement predicate, the profile table, the family deltas and the scorer. Self-test 118.
