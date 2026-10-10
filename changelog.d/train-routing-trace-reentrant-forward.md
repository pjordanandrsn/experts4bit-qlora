### Training routing trace: record the training forward under reentrant checkpointing

The recorder skipped a router call as an eval forward whenever grad mode was off. Reentrant gradient checkpointing,
which the `OFFLOAD_EXPERTS` arm uses, runs the training forward under `no_grad`. Every training forward was dropped, and
only the recomputes were kept, all under micro-batch -1. The completeness guard voided that capture, so no number came
from it. An eval forward is now recognised by the router module's `eval()` mode, which the harness's `eval_loss` sets.
A test trains a toy through reentrant checkpointing and requires a complete trace; it fails on the previous predicate.
