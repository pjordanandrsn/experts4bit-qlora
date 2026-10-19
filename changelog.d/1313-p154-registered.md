### P154 registered (#1313): the W16 NF4 expert byte floor of Qwen3-30B-A3B's served step, on the replay's own route

A REPORTING instrument for stream (B): it prices the headroom of the grouped NF4 small-M expert GEMM that P152 named at
71.2 % of the 16-request step, and names no lever. P152's runner is cut to its routing pass, which now runs on the
replay's route: the pass sets `DEVICE_GROUPING` itself, and must count exactly two K25 calls a MoE layer a step on an NF4
store on all 48 layers, else VOID. The store's bytes per expert are read off the loaded tensors, and the arena must be
the box's own bake. The floor is compared with P152's GEMM times as recorded, from its committed verdict, sha-pinned.
One RTX 5090, guard 1.0 h (the time-left fit with 25 % headroom), worst $1.72. Files: `bench/p154/` and `tests/test_p154.py`.
