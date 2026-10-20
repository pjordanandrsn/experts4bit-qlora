### P161 registered (#1313): K25 at plan c1 in the served step, under K39's decode, on Qwen3-30B-A3B

P160 read plan c1 GO in a microbench (1.6598x on the two K25 GEMMs under `lut="prmt"`, lower bound 1.6568,
clock-adjusted 1.6525). P161 reads it in the served step, by P158's served form.
- **Arms:** S0, `E4B_K25_PLAN=today` (named), and S1, `c1` (e4b#1877), on the shipped default server with K25's decode
  at its qwen3_moe default (`prmt`, unset in both arms); a warm-up process, then ABBA twice through W16 and W1 at five
  reps.
- **The rule is fixed before data:** P158's. FLIP needs g16, the min of four W16 pairs, >= 1.03 and g1 >= 0.99, on
  unrounded minima; the token-identity gate with its null first ("today's plan not token-deterministic" VOIDs the read
  before any S1 difference can be a contract finding against c1).
- **Two witnesses for the plan:** the plan record (`K25_PLAN_SEEN`, exactly 960 under the arm's plan after the build)
  and a spy on the kernel's block shapes (exactly 1920 calls: today's 32/64/4/3 for both projections, or c1's 64/256/4/2
  and 128/128/8/3, 960 each), tied to the decode record's total. Both were confirmed in their licence half on a $0
  layer-cut build on the A2000, with the box's own spy.
- **Box:** P158's: one RTX 5090 at >= 575 W, grouped-nf4-gemm v0.50.0, guard 1.5 h, worst $2.35; the premise at 62
  tests (e4b's served-route contract frozen with c1's cases).
- **Projection, never a bar:** g16 1.378 if P160's ratio carried 1:1, 1.280 at P158's measured carry.

A FLIP opens a separate PR to make c1 qwen3_moe's K25 plan by default. Files: `bench/p161/` and `tests/test_p161.py`.
