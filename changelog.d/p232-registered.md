### P232 registered: the served read of the int8 lm_head at 2, 4 and 8 rows (lever K, K2b)

- `bench/p232/PREREG-p232.md`: on Qwen3-30B-A3B's default server, `E4B_INT8_HEAD_ROWS=8` against unset at W2, W4 and W8
  (speed, ABBA twice) and at T = 2, 4 and 8 (quality, P153's bars per T). The default already serves the int8-rounded
  head, so the ruling pair is K37 against `F.linear` on the identical grid; the bf16 head is reported only. The verdict
  (FLIP_ROWS_N / NO_FLIP) decides whether a separate PR may make the knob a default. The kit follows as its own PR.
