### Read: TC1 amendment 40 — with its chunked loss e4b trains packed 4,096-token rows (no OOM), but the box reads UNTESTED (P87–P89)

- `tc1-5090-91` ($1.43, EPYC 7K62, machine 152440): amendment 39's packed box with `E4B_CHUNKED_LM_LOSS=1` on every e4b arm. Every e4b
  arm trained to step 40 at a 32.4–32.6 GB peak (amendment 39's OOMed at step 1); Unsloth 16.03 / 16.03 s/step at 24.86 GB.
- Every e4b arm is VOID under TC1's registered no-loop rule: grouped-nf4-gemm's `auto` route took the per-expert LoRA loop on ~1.5 % of
  delta calls (padded blocks over its 2 GiB limit). P87–P89 UNTESTED. Read as if VALID, Unsloth/e4b would be 1.43 and the environment
  0.892 -- unquotable under this instrument.
- Three earlier launches died on their hosts ($0.24): offline, ssh never authenticated, a driver below the floor.
