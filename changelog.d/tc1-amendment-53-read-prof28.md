### Read: TC1 amendment 53 -- 60 % of torch 2.8's extra time at e4b's defaults is not device time, its largest host-side increase in the bucketed delta's batched matmuls (P134-P136 HELD)

- `tc1-5090-110` ($1.30, a quiet Threadripper PRO 3955WX) profiled e4b's matched arm on packed rows. Environment ratio **0.790** [0.789,
  0.792]; of the 2.63 s per step torch 2.8 adds, 59.7 % is not device time. In torch 2.8 the buckets drop the device's share of the step
  from 0.965 to 0.850. Row `e4b.train.env-gap.torch28.profile.qwen3.5090.2026-10-06`.
- The largest host-side increase is `aten::bmm`: the same ~26,750 calls a step in both torches, about three times the CPU self time per
  call in torch 2.8. That is an upper bound on host work, since self time also counts waits on a full launch queue.
  The device time is grouped-nf4-gemm's Triton kernels (+22 % forward, +52 % data gradient, buckets or not).
- By the registered rule, the next registration is a shape-stable bucket ladder in grouped-nf4-gemm. STATUS says so.
