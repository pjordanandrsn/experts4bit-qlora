### Two opt-in prefill-glue knobs, both off by default: `E4B_FUSE_PREFILL_GLUE` and `E4B_PREFILL_LEAN_DISPATCH`

The design is `bench/prefill-glue/DESIGN.md`, P1 and P2. Neither knob changes anything at its default, and neither is
licensed yet.

- **`E4B_FUSE_PREFILL_GLUE=1`** lifts the decode folds' 64-row gate in `glue_fuse` and `glue_r2`, resolved at patch
  time. A prefill's norm, residual-and-norm and q/k-norm-and-rotary calls then run the fused kernels decode uses.
  - The router epilogue is out of this round, so it keeps its gate.
  - Its licence will be a teacher-forced prefill read.
- **`E4B_PREFILL_LEAN_DISPATCH=1`** changes K19's prefill rows:
  - gate_up reads token rows itself (`gather_div`), and down stores in the caller's order (`scatter=order`);
  - the chained 16-row tile table is kept;
  - it is bit-identical by construction there;
  - a K19 without the options is refused.
- **Evidence in `/health`:** `prefill_routes.seen` gains `moe_k19_dispatch` (the K19 tile table and dispatch) and
  `prefill_folds` (fused fold calls above 64 rows), and `prefill_routes` echoes both knobs' raw values.
- **Tests:**
  - the glue gates engage above 64 rows only under the knob, and stay as today without it;
  - the lean dispatch is `torch.equal` to the gather route on CPU with a K19 stand-in, and is recorded as
    `chained|lean`;
  - the P115 tiny Qwen3-MoE's served-shape prefill is fused except the router.
