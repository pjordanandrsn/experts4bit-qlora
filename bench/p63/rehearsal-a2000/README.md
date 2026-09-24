# P63 rehearsal on the NAS RTX A2000 — NOT a reading

This directory is the plumbing rehearsal for lane P63 ([`../P63-PREREG.md`](../P63-PREREG.md)), run for free on the
home-lab NAS before any rental. **It is not the lane's reading and nothing in it is claimed.** The registered reading
is Qwen3-30B-A3B on an RTX 5090 (sm_120, 170 SMs); this is OLMoE-1B-7B-0924 on an RTX A2000 12 GB (sm_86, 26 SMs).
Lane B393 found that neither `combine_rows`' bits nor torch's own chain are the same on sm_86 and sm_120, and several
routes here plan from the SM count, so the exactness pattern below need not carry to the 5090.

What it shows: the probe, its comparison module and the reducer run end to end on real CUDA kernels, the instrument
gate passes, and it gives a first look at which routes differ. What it cannot show: anything about the 5090, Qwen3's
shapes (the dot-pad NF4 decode kernel only engages at Qwen3's census shapes on ≥ 160-SM parts, so it never ran here),
or fused q/k/v (OLMoE's attention is not Qwen3-MoE's and does not fuse).

## Setup

- Box: the NAS RTX A2000 12 GB, driver 575.64.05, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, shared with
  production services and other agents' jobs; every GPU job took `/share/Container/gnf4-interp/a2000.lock` first and
  released it on exit ([`scripts/gpu_job.sh`](scripts/gpu_job.sh)). Peak allocation 5.7–6.5 GiB.
- Software: torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1, bitsandbytes 0.50.1; grouped-nf4-gemm `66d41c8`
  (0.33.2; its probed kernels are the ones `f88df1e` ships, which changes only two docstrings in them); this branch's
  experts4bit-qlora working tree, with `bench/p63/p63_{probe,compare,reduce}.py`, `fixture.txt` and
  `engines/hot_residency.py` byte-identical to the committed files (sha256 checked on the NAS after the run).
- Model: `/share/models/OLMoE-1B-7B-0924` from the LAN model store, read-only, baked to an NF4 arena by P39's
  `k8_bake.py`. 16 layers, 64 experts, top-8, hidden 2048. Fixture: `fixture.txt`, the first 160 OLMoE tokens
  (ids sha256 `8b378ed0…`).
- Scripts: [`scripts/rehearse.sh`](scripts/rehearse.sh), one stack per GPU job via [`scripts/chain.sh`](scripts/chain.sh)
  so the lock was released between stacks. Wall: `int4` 7.3 min (of it 4.7 min packing int4 on a contended CPU),
  `nf4` 2.3 min, `int4nf` 5.1 min.

## Receipts

- `out/<stack>/p63_arm.json` — the probe's receipts (the per-position `p63_detail.pt` tensors are not committed).
- `RESULTS-rehearsal-generated.md`, `p63_rep.json` — `p63_reduce.py` over the three stacks; the NAS's run of the
  reducer and a local rerun produced byte-identical JSON.
- `job_<stack>.log`, `run_<stack>.log`, `gpu.txt`, `bake.log` — logs.
- `part_oob.json` + [`scripts/part_oob_check.py`](scripts/part_oob_check.py) — the memory-safety check of the int4
  singleton GEMV's preallocated buffer (below).
- `run1-int4-errored/p63_arm.json` — the first `int4` run, every sub-arm an error (below).
- `runner_dryrun.txt` — `p63_drive.sh`'s dry run and refusals, and `p63_run.sh` in a GPU-less container (no rental).

## What it read (A2000, OLMoE: not a reading)

- **G0:** all three stacks' controls and verify repeats bit-identical; every module replay at n = 1 reproduced its
  recording.
- **Kernel census, layer-0 gate_up (2048 × 2048), T ∈ {16, 17, 160} against T = 1:**
  - int4 GEMV → int4 GEMV: EXACT (1,280 / 1,280 rows at T = 160). On 26 SMs the planner's row term DOES change the
    split count here (16 splits at 8 rows, 1 at 128), and the result is still bit-identical: each of the 16 splits
    owns exactly one KU iteration, and the reduce adds the partials from zero in split order, which is the same
    sequential sum one program computes. This mechanism was not in the first draft of the predictions; it is now.
  - int4 GEMV → int4 dequant + bf16 matmul (the default T > 1 route): PRECISION, 0 / 1,280 rows equal, max row
    rel L2 1.0e-2.
  - int4 GEMV → grouped int4 GEMM: REORDER, 1,035 / 1,280 rows equal, max rel L2 2.5e-4.
  - NF4 decode GEMV (the scalar path; the dot-pad kernel never engages here) → itself: EXACT; → the M-tile:
    PRECISION, 0 / 1,280 rows equal, max rel L2 2.5e-3.
  - `combine_rows` against itself across T: EXACT at every T; against the torch chain, bit-equal at T ≤ 17 and not at
    64 / 160 on these random rows (B393's reorder-class difference). Both inside B393's bound (max ratio 0.995).
- **Module replay (n-row call vs the T = 1 calls):** the experts module is EXACT at every n under
  `FORCE_SINGLETON_GROUPS` (int4 with this branch's fix, and NF4), EXACT at 16 / 17 and not at 160 under
  `DEVICE_GROUPING` on the int4 store, and not exact on the default, `combine0` and NF4-device routes. The attention
  projections, the router and the lm_head are never exact above one row (cuBLAS picks its kernel by M; K16 at 2–16,
  cuBLAS above). The torch RMSNorm IS row-invariant (exact at every n); on the `int4` stack the fold kernel is exact
  at 16 / 17 and differs at 160, where the layer falls back to the torch chain.
- **End to end:** no sub-arm is exact in any mode, and every first difference is at layer 0 — `attn_core` (the
  attention core: SDPA by mask) in every verify and in the `nf4` and `int4nf` prefills, `attn_in` (the input norm,
  fold kernel vs torch chain) in every `int4` prefill — the one stack where the norm fold is on, so the `int4nf` stack
  localises the difference to the fold as designed. Sizes over the 45 (stack, sub-arm, mode) rows: KL mean 5.2e-4 to
  4.3e-3 nats/token, KL max 6.3e-2, argmax flips ≤ 5 per mode (top-1 ≥ 0.9688).
- **`E4B_FUSE_COMBINE=0` vs the default, T = 1 controls:** NF4 29 / 160 positions bit-equal, KL mean 4.6e-4, 1 flip;
  int4 11 / 160, KL mean 1.2e-3, 1 flip. First non-equal layer 0.
- **Defect scan:** 227 records against fp64, none outside its model's bound once cuBLAS runs with fp32 split-K
  reduction.

The reducer's three REFUTED rows are all `GNF4_GEMV_DOTPAD=0` (P1's scalar-GEMV census pair, P2's
`hf.singleton.dotpad0` experts replay, P4's `hf.singleton.dotpad0` control). Those predictions are for the 5090, where
the default NF4 decode is the dot-pad kernel and the scalar path's K1 plan splits gate_up 16 ways at 8 rows. On 26 SMs
the default decode already IS the scalar path, and it splits at no row count, so the switch changes nothing. Expected,
and not a reading either way.

## What it changed in the design (disclosed in the pre-registration)

1. **A memory-safety defect in e4b, fixed on this branch.** `part_oob.json`: the int4 singleton GEMV at T = 17 wrote
   16,384 / 12,288 / 180,224 fp32 elements past the store's preallocated buffer (OLMoE gate_up; Qwen3 gate_up / down
   shapes); T = 1 and T = 2 wrote nothing past it. `hot_residency._int4_part_or_none` fixes it; the probe refuses the
   int4 singleton sub-arm on a cut without the fix.
2. **The router epilogue changes dtype with the row count.** The first `int4` run (`run1-int4-errored/`) lost all six
   sub-arms to one crash in the module replay: the fused router (`E4B_FUSE_ROUTER_EPI=1`) returns fp32 routing weights
   at ≤ 64 rows and the upstream router bf16 above, so the replay's bit view of the two did not line up. That is a
   PRECISION route #708 did not list; it is now in the route table and registered as P7 (held here: the dtype differs
   at n = 160 only, max |d| 4.8e-4 on layer 0). The comparison records a dtype difference instead of crashing, and a
   replay error now costs one module, not the sub-arm.
3. **cuBLAS needs its own accuracy model.** Under torch's default `allow_bf16_reduced_precision_reduction = True`,
   cuBLAS bf16 GEMMs exceeded the fp32-accumulation bound on 35 records, by up to 7.35× (NF4's bf16 attention at 16
   and 160 rows). Rerun with the flag off, every one is inside it (max 0.88), and the flag changed the result's bits
   in 54 of 116 cuBLAS records. Without this, the defect line would have flagged torch's documented default as a
   kernel defect. cuBLAS paths are now rerun with the flag off for the DEFECT line; the readings stay on the defaults.
