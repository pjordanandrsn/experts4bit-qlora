### DQ4 read (#1083): CAP_REAL — streamed frozen weights train a 2.00× longer sequence on one RTX 5090 (graded c_def; bench only; #1206)

- **The run.** `dq4-5090-2` ($0.689): Qwen3-32B architecture, 64 layers, PEFT + bnb QLoRA, chunked loss, default
  allocator.
- **The boundaries.** The longest training sequence is 7,168 tokens resident and 14,336 streamed, both confirmed in
  fresh processes. G = 2.00, bracket [1.75, 2.14], at unchanged step time.
- **The measured cause of the shortfall against the ~20k predicted** is default-allocator fragmentation: about 6.2 GiB
  reserved but unallocated. The secondary `expandable_segments` configuration reaches 19,456 tokens (G 2.375).
- **Evidence.** The receipts re-derive the verdict byte for byte. `bench/dq4/RESULTS-dq4.md` carries the scope and an
  independent review.
