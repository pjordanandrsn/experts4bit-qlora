### P115 Phase C's scripts: does `auto` on the fusion knobs engage by structure on gpt-oss-20b and Qwen3.6, compute nothing grossly wrong, and pass Phase B's quality read on Granite?

- `bench/p115/` gains Phase C's box (`p115c_run.sh`, `p115c_box.py`), its rule (`p115c_reduce.py`), its controller
  (`p115c_drive.sh`) and their pins (`staged-c.sha256`). PREREG-p115 amendment 2 registers them before any Phase C box.
- One RTX 5090 runs Granite-3.1-3b-a800m, gpt-oss-20b (on SC2g's e4b path) and Qwen3.6-35B-A3B (on P98's arena), in
  that order. Each model gets serve off (`0`), on (`auto`) and explicit (`1`, which must refuse) on the default server.
  - gpt-oss and Qwen3.6 then get SANE, the maintainer's gross-error gate: Phase B's teacher-forced instrument on 12
    wikitext windows, |bias| ≤ 0.02 nats and argmax agreement ≥ 0.95. FLIP_LICENSED needs every gate on both models.
  - Granite gets Phase B's full quality read instead, under Phase B's rule unchanged. It was asked for in review after
    the proof hinted at a c4val1 cost. Its own verdict, GRANITE_LICENSED or GRANITE_HELD, decides whether the flip's
    `auto` allowlist takes Granite.
- One prediction is corrected before data. On Qwen3.6, all four knobs at `1` hit glue round 1's own vacuous-enable
  refusal before the q/k/v check. The explicit gate now accepts any of the four knobs' own refusals, provided the
  message names the knob at `=1`.
- Budget: a Granite proof (guard 0.75 h, every process kind), then the reading (guard 2.0 h). Phase C's ceiling is $5.00
  inside the lane's registered $10 hard stop; anything over $15 needs the maintainer lane's approval.
- Tests: `tests/test_p115c_staged_pin.py` (the pins, self-tests, order, subject, guards and exit codes, plus the
  explicit gate against the code's real refusals on tiny gpt-oss, Qwen3.5-MoE and Granite models) and
  `tests/test_p115c_sane_families.py` (SANE's instrument builds and scores gpt-oss and the Qwen3.5 hybrid on CPU).
- Nothing in the package changes; every default is as before.
