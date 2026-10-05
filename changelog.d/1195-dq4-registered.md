### DQ4 registered (#1083): the sequence-length capacity of streamed against resident QLoRA on one RTX 5090 (bench only; #1195)

- **The question.** With DQ3's subject (Qwen3-32B architecture, 64 layers), what is the longest training sequence
  resident (R) against streamed (S)? Each boundary is a real out-of-memory boundary: an ascending ladder to the first
  OOM, then fresh-process confirmations.
- **The configurations.** The graded one is the chunked loss with the default allocator. The secondary is
  `expandable_segments`. The stock loss is descriptive.
- **The rule.** G = L\*_S / L\*_R: CAP_REAL if G ≥ 1.5. Predicted G ~2.3, from DQ3's 5090 numbers.
- **The files.** `bench/dq4/` (harness, reducer with a 25-case self-test, A2000 rehearsal gate) and
  `tests/test_dq4_lane.py`.
