### Read: TC1 amendment 49 -- bucketed padding at the field recipe reads UNTESTED (the single block's draws 12-21 % apart)

- `tc1-5090-103` ($1.11, Threadripper 3990X): `NF4_QLORA_PAD_BUCKETS=0` vs `=1` at the field recipe. The single block's draws were unstable
  on both arms (matched 3.524 / 4.342, shipped 2.970 / 3.358 s/step), so P119-P122 are UNTESTED. The bucketed draws (4.218 / 4.159, 3.685 /
  3.709) were slower than every single-block draw; recorded, not read. Buckets stay opt-in pending a re-ask that records each call's
  single block.
