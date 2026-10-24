### TC1 amendment 78 read: TC1's field recipe fits a 24 GiB RTX 4090 (FIT, two draws on two machines)

- Two rented RTX 4090s on two machines, one draw each, ran e4b's matched arm with the opt-in pb1cl256 levers:
  `NF4_QLORA_COMPACT_DELTA=1`, `NF4_QLORA_PAD_BUCKETS=1` and `E4B_CHUNKED_LM_LOSS=256`.
- **FIT.** Both draws trained the field recipe's 60 steps with no out-of-memory error, and their steps 1-2 losses are bitwise equal.
- **The least headrooms over both draws:** 1,258 MiB usable minus max allocated, 586 MiB usable minus max reserved, and 586 MiB of
  the card's total unused at the fullest step. It fits; it does not fit with room.
- **Scope.** FIT covers this fixture only: Qwen3-30B-A3B, TC1's alpaca tokens, sequence 2048, micro-batch 2 × accumulation 4, r 16.
  The levers stay opt-in; a default change needs its own PR.
- Claim `e4b.train.4090-fit.field.4090.2026-10-24`, with a README section and STATUS lines. The receipts' public copies are under
  `bench/h2h-2026-10-02/tc1/receipts/tc1-4090-1/` and `tc1-4090-2/`.
- The amendment's tests now read its pins at its frozen launch commit (bench/common rule 6).
