### Added

- **P154's record** (#1313), stream (B)'s W16 NF4 expert byte floor: `bench/p154/RESULTS-p154.md` and the public receipt
  copy under `bench/p154/receipts/`.
  - **The read** is READ, on the replay's own route: both route witnesses exact (6,144 K25 calls; 3,072 `nf4_k25|le256`).
    Bytes per expert were read off the loaded NF4 store (1,769,472 / 884,736), on P152's bf16-head server.
  - **The floor:** at 16 requests the grouped NF4 small-M expert GEMM reads its weights at 0.42 of the copy rate. That is
    6.81 ms of headroom over its byte floor, 41 % of the unprofiled step, against P152's recorded GEMM times.
  - **Total:** $0.399. **Default:** this record changes none, and names no lever.
