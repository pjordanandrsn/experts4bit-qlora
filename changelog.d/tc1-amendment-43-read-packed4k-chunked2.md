### Read: TC1 amendment 43 — on packed 4,096-token rows, with its chunked loss, e4b is 1.278× Unsloth's speed on one stack (P96, P97, P98 HELD; labelled)

- `tc1-5090-95` ($2.65, EPYC 7B13, machine 145701, 40-step load-gated draws): amendment 40's box with the per-expert LoRA loop read
  as a recorded route (max 2.6 % of delta calls). e4b 11.188 / 11.220 s/step against Unsloth's 14.266 / 14.366: **1.278** [1.271, 1.284];
  environment 0.915; every e4b arm resident (peak 32.5 GB against Unsloth's 24.86).
- Recorded as the LABELLED packed position (row `….packed-4k-chunked`, `E4B_CHUNKED_LM_LOSS=1` opt-in) beside amendment 39's
  out-of-memory row; STATUS says so. P98 is amendment 44's packed side.
