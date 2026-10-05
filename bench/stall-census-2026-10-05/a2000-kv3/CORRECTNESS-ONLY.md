# Correctness and counts only

These files are the NAS RTX A2000's raw run output, kept as the run wrote it. The A2000 is a correctness testbed
(owner policy since 2026-07-27, restated in e4b#1133): **no timing in these files is speed evidence**, seeds a
prediction, or is read anywhere in this census or in SC2c. What they evidence is:
- bitwise parity of the bulk KV forms against the per-layer path;
- launch counts (`torch.profiler`);
- test pass counts and, in `a2000-kv5`, the flush's memory bound against the allocator (bytes).

Every `host_ms`, `gpu_ms` and `wall_ms` field is out of scope.
