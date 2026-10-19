### P156 registered (#1313): K39's decode against today's select tree at Qwen3-30B-A3B's W16 shapes, on S0's real router picks

A microbench premise for stream (B). P155 found that K25's plan is not the W16 lever. P156 times what P155 could not
move, the instructions per weight, at the same plan.
- **Arms:** T, today's select-tree decode, and P, K39's `lut="prmt"` (grouped-nf4-gemm#569, bit-identical to the tree),
  both at today's plan, called exactly as the served route calls K25, over full-size NF4 stores; T2 is the self-pair.
- **Routing:** P153's committed router replay (S0's real picks), with P155's checks before data.
- **The question is fixed before data:** GO needs P's median - 2 SE >= 1.05x on both GEMMs (the SE of the median), on
  the 64 held-out steps, in paired, rotated reps. A P that moves a bit is a contract finding, not a timing.
- **Reported, never ruling:** a prmt plan sweep (the 144 plans on 8 selection steps; three carried and confirmed)
  prices decode x plan; both decodes' compiled registers are recorded; Nsight Compute captures T's and P's calls if the
  host allows counters.
- **Predictions:** against the issue/occupancy model: M1 overlap 1.02-1.11, M3 serial 1.15-1.18, and M2 occupancy
  1.19-1.28, the least likely after P155. The basis is triton 3.4.0's count, 17.0 against 24.1 instructions per weight
  (grouped-nf4-gemm#570 has the newer toolchain's).
- **Box:** one RTX 5090, grouped-nf4-gemm at #569's merge, guard 1.25 h ((boot + fit) x 1.25, checked before the
  install; the bench allowance is the larger of P155's box walls and this lane's $0 A2000 shakedown, x 1.5), worst
  $1.42. The shakedown's record is a sha-pinned test fixture.

A GO opens a served read as its own registration. Files: `bench/p156/` and `tests/test_p156.py`.
