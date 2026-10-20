### P149 Amendment 5: a dry mode on a one-layer cut, and early gates before any scored arm

The lab A2000 (sm_86) cannot build box Q+S's registered server. Its bucketed decode graphs need the fused FP8 KV
append, which needs sm_89 or newer. So until now, the bucket capture census, Q's re-capture and forcing checks, T1, T2
and SANE would first have run for real on the box itself.
- **`P149_DRY=1`** (run ids `p149-dry-2` to `-4`) runs the whole rotation on one RTX 5090 against a one-layer cut of the
  checkpoint (`p149_slice.py`: shards 1 and 19, the embedding, layer 0, the final norm and the head).
  - The reducer's `--dry` checks the same closed forms at one layer and lists every instrument fault it finds.
  - The verdict is hard-coded DRY: an instrument run, never a reading and never a strike.
  - Guard 3.0 h, worst $3.31.
- **Early gates, in P158's form, on the box too.** An unscored S1 process, S1w, runs before any scored arm and stops the
  box on an instrument refusal (never a strike):
  - rc 43, its build: the census, the int8 census, and every captured bucket at exactly 3 × 128 int8 calls;
  - rc 44, one teacher-forced Q pass per gate, with the forcing checks;
  - rc 45, S0a's build, checked before S1a runs.
- The capture count also joins the VOID rule. The box's guard table gains S1w's 600 s, and each speed arm's
  allowance drops from 600 s to 480 s, three times the measured arm. That gives a 16,290 s total in the 6.0 h guard, and
  boot keeps 1,612 s of tolerance. The worst stays $7.38.
