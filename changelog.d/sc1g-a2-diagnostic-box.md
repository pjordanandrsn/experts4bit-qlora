### SC1g amendment A2 registered (#846): an e4b-only diagnostic box first, because e4b's served path reads +0.185 nats over its own prefill on in-distribution text (bench and tests only)

- **`sc1g-prove-2` ($1.306) ran out of its guard** on a slow host: the vLLM install took 23 min and the first arm started at
  72 min. On conversation `conv1` (ppl ≈ 2) it still read e4b served **0.905** vs e4b prefill **0.720**, +0.185 nats. prove-1's
  wikitext rows split a similar gap into the paged path (+0.25, both bf16 activations) and the int8 GEMV (+0.24). vLLM's served −
  prefill is +0.016.
- **The kernel is correct on sm_86:** on prove-2's real captured decode activations (160 calls) it agrees with its bf16-rounded
  exact reference to ≤ 1.9e-4 ($0, A2000, correctness only). The int8 per-32 scheme itself costs 0.45–0.94% per GEMV output.
- **Box J** (`SC1_BOX=J`, guard 1.0 h, no comparators, no proof): 20 e4b arms in priority order plus G6 on the 5090. The anchor is
  the chunk-free full forward. The paged kernel's fp8 K/V roundings are modelled inside it (`kv`, `k`, `v`, 16 key groups), and it
  adds eager chunk 1, the real kernel at `--kv-groups 16`, PDL=0 and folds-off, on `conv1` with `conv2` replication.
- **Six predictions, J1–J6, are registered per arm** (e.g. J1: chunk 1 closes ≥ 0.5 of the gap; J2: fp8 K/V rounding reproduces
  ≥ 0.5 of it). Graphs and step-select are already excluded, because K8 runs eager. The cross-engine reading waits for J: after an
  e4b fix if the path is at fault, otherwise with a 2.0 h proof guard.
