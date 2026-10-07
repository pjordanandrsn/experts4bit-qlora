### P115 Phase C's scripts: does `auto` on the fusion knobs engage by structure on gpt-oss-20b and Qwen3.6, and compute nothing grossly wrong?

- `bench/p115/` gains Phase C's box (`p115c_run.sh`, `p115c_box.py`), its rule (`p115c_reduce.py`), its controller
  (`p115c_drive.sh`) and their pins (`staged-c.sha256`). PREREG-p115 amendment 2 registers them before any Phase C box.
- One RTX 5090 runs gpt-oss-20b on SC2g's e4b path, then Qwen3.6-35B-A3B on P98's arena. Each model gets five processes
  on the default server: off (`0`), on (`auto`), explicit (`1`, which must refuse), and SANE off/on. SANE is the
  maintainer's gross-error gate: Phase B's teacher-forced instrument on 12 wikitext windows, |bias| ≤ 0.02 nats and
  argmax agreement ≥ 0.95. FLIP_LICENSED needs every gate on both models.
- One prediction is corrected before data. On Qwen3.6, all four knobs at `1` hit glue round 1's own vacuous-enable
  refusal before the q/k/v check. The explicit gate now accepts any of the four knobs' own refusals, provided the
  message names the knob at `=1`.
- Budget: a Granite proof (guard 0.75 h), then the reading (guard 2.0 h). P115's actual spend over all phases stays at or
  below the owner's $4.00 per-lane limit.
- Tests: `tests/test_p115c_staged_pin.py` (the pins, self-tests, order, subject, guards and exit codes, plus the
  explicit gate against the code's real refusals on tiny gpt-oss and Qwen3.5-MoE models) and
  `tests/test_p115c_sane_families.py` (SANE's instrument builds and scores both families on CPU).
- Nothing in the package changes; every default is as before.
