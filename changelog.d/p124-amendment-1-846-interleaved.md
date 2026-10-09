### P124 Amendment 1 (#846): attempt 1 read NOISY; the served arms are interleaved step by step (bench and tests only)

- **Attempt 1** (`p124-5090-1`, $1.003) read NOISY: at 64 rows one route-off arm sat 3.1 % above the other, a level
  shift of the whole arm on the device, while every pair at 32 rows agreed within 0.05 %. Quality passed (ON64 +0.0011,
  ON32 +0.0014 nats), and both mutants failed. Nothing from attempt 1 decides anything. Its receipts and a RESULTS
  section are committed.
- **Amendment 1:** at each depth, two blocks with one runner per setting, both alive and captured under their
  settings, decoding in strict alternation (OFF first in block a, ON first in block b), so a shift in the GPU's state
  lands on both. NOISY when the blocks' ON/OFF ratios differ by more than 1.5 %. The per-pair ratio median and a GPU
  clock, power and temperature log are reported, never gated. Everything else is as registered. The reducer is
  self-tested on 42 cases.
