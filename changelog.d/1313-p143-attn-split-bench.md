### P143 registered (#1313): the B=1 paged FP8 attention decode by context and split count

P142's per-name census put the paged FP8 attention decode at 12.7 % of the B=1 step. That is 48 launches of about
11.6 µs, latency-bound at ~640 live positions. The library plans its 32 splits from the block table's capacity, not the
live length. `bench/p143/` times the served call on e4b's own KV class at Qwen3-30B-A3B's geometry. The grid is context
{256, 640, 1024, 2048, 4096} × split {library plan, 4, 8, 16, 32}, in CUDA graphs, with no model. The rule is fixed
before data: GO_FIXED if a fixed split saves ≥ 3 % of the step at 640 and loses nothing beyond noise at 4096;
GO_ADAPTIVE if only a per-context choice clears; otherwise PARK. A microbench: no default or speed claim changes.
