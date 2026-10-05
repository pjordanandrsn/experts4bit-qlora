### RD1 read: on one RTX 5090 the decoded frozen-expert route beats every shipped route per call, at both lengths, on 4 of 7 many-group families -> an opt-in route, then a TC1 A/B (bench and tests only; `bench/moegen/rd1/RESULTS-rd1.md`)

- **Licensed reading:** `rd1-rp-5090-2`, RunPod Secure, $0.12. The post-probe anchor passed (rc 0, `pcie-full/launch-fast`), and
  every arm passed the fp32-reference gate on all 32 cells.
- **DECODED HELD, 4/7.** decoded_cap (dequant_groups + one grouped bf16 GEMM, 256 MiB cap) / best of v1, v3 and dense, skewed
  router, at seq 512 / 2048: olmoe 0.79 / 0.39, lfm2 0.74 / 0.37, ernie 0.81 / 0.40, graniteh 0.79 / 0.40.
  - At seq 2048 it wins on all seven families (0.37–0.82).
  - qwen3, nemotron and qwen36 fail at seq 512 only (1.05, 1.60, 1.38), where they have 16–32 rows per expert.
- **V3 NOT HELD, 0/7:** v3 / v1 is 0.88–1.23.
- **Predictions, scored on their registered text by `rd_table.py`:** P0, P3 and P4 HELD. P1 REFUTED: the bar held, but qwen3
  and qwen36, two of the three families it named, fail. P5 REFUTED: on mixtral, dense beats decoded_cap on the skewed draw but
  not on the uniform one.
- **Runner fix:** an anchor that crashes now exits 9 (a harness error), not 12 (a refusal). `rd1-rp-5090-1`'s pinned-memory
  `CUDA error: invalid argument` had been written as a refusal.
- **No speed claim follows for training.** That waits on the TC1 full-step A/B, registered separately before its box.
