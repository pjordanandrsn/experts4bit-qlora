### TC1 amendment 43 registered: the packed 4,096-token regime with e4b's chunked loss again, the LoRA loop read as a recorded route (bench and tests only)

- **Why.** Amendment 40's e4b arms trained the packed rows with the chunked loss (32.4–32.6 GB, no OOM) but read VOID under TC1's
  no-loop rule: grouped-nf4-gemm's `auto` route took its per-expert LoRA loop for ~1.5 % of delta calls (padded blocks over its 2 GiB
  limit), its default behaviour at 4,096 tokens.
- **The change** (this family only): the loop is a recorded route up to 5 % of a step's delta calls; the share is printed per arm.
- **The box** (token `qwen3samestack4kce2`): amendment 40's box on another host. P96 Unsloth/e4b in [1.25, 1.65]; P97 the environment
  in [0.84, 0.95]; P98 every e4b arm completes resident. The bands were set with amendment 40's unquotable 1.43 / 0.892 in view, and the
  registration says so. Amendment 41's default decision reads P98 in place of amendment 40's P89.
- One new self-test case; amendment 40's receipts reduce as before.
