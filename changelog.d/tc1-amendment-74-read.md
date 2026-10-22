### TC1 amendment 74 read: no padded-delta switch is a candidate; removing the delta moves the peak to the LM-loss backward

- One rented RTX 5090 at TC1's field recipe with the memory census, e4b's matched arm at today's pins, one draw per arm. Today's
  defaults peak at 26.500 GB with 2.348 GB of grouped-nf4-gemm's padded LoRA delta at the peak. `NF4_QLORA_SINGLE_LADDER=0` removes
  14.5 % of it. `NF4_QLORA_COMPACT_DELTA=1` and `NF4_QLORA_PAD_BUCKETS=1` each remove all of it, but the peak falls only 0.639 GB, to
  25.861 GB, so by the registered 50 % peak credit no switch is a candidate for a default change.
- The census shows why: at those arms' peaks the stock full-vocab LM-loss backward is live instead, and both arms agree to 2,048
  bytes. The rule's "Moved, not saved" label is its wording; the delta's bytes were removed and the peak relocated.
- Claim `e4b.train.padded-delta-switches.field.5090.2026-10-22`. README section, STATUS lines, and the receipt's public copy under
  `bench/h2h-2026-10-02/tc1/receipts/tc1-5090-154/`. The reducer self-test pins the peak credit at its 0.50 boundary.
