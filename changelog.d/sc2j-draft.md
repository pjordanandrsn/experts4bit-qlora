### SC2j (draft): the shared KV pool's capacity read at the default 4,096 tokens a slot

- A draft registration, for review: `bench/sc2/SC2j-PREREG.md`. Claim-all against shared, both at
  `E4B_PAGED_MAX_SEQS=auto`, on Qwen3-30B-A3B int4 on one RTX 5090, with a contract arm (shared at 32 slots
  token-identical to claim-all at 32 on one fixed schedule), SC2e's ladder, a long-context ladder on which reservations
  bind, and a re-run of P117's teacher-forced width check at the launch stack. The kit follows once the design is
  settled; nothing runs until it registers.
