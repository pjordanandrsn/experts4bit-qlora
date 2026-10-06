### SC1g A6 continuation read (#846): P3 HELD -- bf16 decode activations do not move e4b's median KL; A6 complete (bench and tests only)

- **The run.** `sc1g-diag-a6-2` (box J, $0.753, Vast 145701) ran the bf16-activation arms the first A6 box's deadline
  dropped.
- **P3 HELD.** median(c) / median(b) is 1.019, 0.920 and 0.912 on conv1–conv3. conv4's (c) was dropped by `can_run`'s
  600 s reserve. The decode GEMV's int8 activations are cleared at the median, though the conv2 and conv3 margins are
  thin (8–9 %).
- **Determinism.** (b) is bit-identical to its repeat, to A5's reading and to A6-1's: three runs on two hosts.
- **Per-arm times** are recorded beside each arm. conv1's three arms took 3.6–5.2 min on a shared host whose host-wide
  load was 17–33; the rest took about 2 min. Values are unaffected.
- **Descriptive.** On conv1, bf16 activations leave the median but double the mean KL: a tail effect.
- **A6 is complete:** P1 PARTIAL, P2 FALSIFIED, P3 HELD. The conv2 lead is with A7 (`sc1g-diag-a7-1`, running).
- **Spend.** The lane is at $10.135.
