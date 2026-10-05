### DQ3 read (#1083): PROTO_PASS — streamed frozen-weight QLoRA on a PCIe gen 5 x16 RTX 5090 is bitwise-identical to resident, at 1.0023× its step time, and frees 15.08 GB (bench only; #1188)

- **The run.** `dq3-5090-5` ($0.282): Qwen3-32B architecture, 64 layers, PEFT + bnb QLoRA, 2048 tokens, run as the
  palindrome R S S0 S0 S R. Every gate passed:
  - step T(S)/T(R) = 1.0023;
  - coverage: 62 + 62 prefetches per steady step, 0 blocking, high-water 2;
  - capacity: 15.08 GB saved, 99.75 % of the slot prediction;
  - parity: bitwise.

  S0/R = 1.18 (descriptive).
- **What it took.** #1183's late-bound backward was required: bnb 0.50.2 keeps the frozen weight on ctx (not reported
  upstream).
- **Cost.** Five attempts, $0.625 in all.
- **Evidence.** The receipts re-derive the verdict byte for byte. `bench/dq3/RESULTS-dq3.md` has the scope (one card,
  gen 5 only), the reserved-memory note and an independent review.
