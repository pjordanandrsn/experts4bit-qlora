### P161 read (#1313): K25 at plan c1 in the served step, FLIP (g16 1.4654)

On one RTX 5090, `E4B_K25_PLAN=c1` decoded Qwen3-30B-A3B's sixteen concurrent requests 1.4654x as fast as today's plan
(the smallest of four ABBA pairs), both under K39's decode at qwen3_moe's default, with one request unchanged (g1 0.9966)
and every token identical. Its three engagement witnesses (the decode and plan records, and the kernel's block shapes)
read exactly in all nine processes. The registered prediction band [1.20, 1.42] was missed high, and the carry over the
projection is reported, not attributed. Record: `bench/p161/RESULTS-p161.md`, receipts under
`bench/p161/receipts/p161-5090-1/`, and the claim `e4b.serve.p161.k25-plan-c1.qwen3.5090.2026-10-20`. A FLIP opens a
separate PR to make c1 qwen3_moe's K25 plan by default.
