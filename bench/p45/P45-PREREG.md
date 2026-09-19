# P45 — WHERE A TRAINING STEP'S TIME GOES: the host-side census of e4b's Qwen3 field-recipe step, beside Unsloth's (pre-registered 2026-09-19, before any box is rented)

Work item: adertha-agents#110 (throughput + training campaign). Lineage: P43 T1 (`bench/p43/P43-PREREG.md`, read of 2026-09-19): at the field fixture (Qwen3-30B-A3B, alpaca, seq 2048, mb2 × accum 4, r 16, adamw_8bit) e4b's `fused_attn4` arm trains at **29.30 s/step with the GPU 16 % busy at 113 W from step 0 to 19** (`p43-t1-qwen3-2`); the int4 attention is not the tier (P2), nothing autotunes (P3), and neither the RAMP nor the CLIFF the T1 rule was written for appeared. The curve asks a question the rule did not: **what is the host doing for the other 84 % of the step?** This lane instruments that and nothing else. No kernel, default, gate or registered number changes.

## Instrument

`bench/tp4/tp4_arm.py --profile-steps K --profile-warm W` wraps optimizer steps `W .. W+K−1` of the SAME loop every framework runs (`run_arm`) in `torch.profiler` (CPU + CUDA activities, no shapes, no stacks) and writes `<receipt>_profile.json`: per profiled window, the wall, the summed **device self time** (kernels + memcpys; over overlapping streams it can exceed the wall and is reported, not clipped), **device events per step** (launches + copies), **CPU op-level self time by op family** (`PROFILE_FAMILIES` in `tp4_arm.py`: memcpy, fused_kernel, routing, optimizer, autograd, matmul, norm_act, other — first match wins, the list is the registration), and the top-80 CPU and device rows. The profiled steps carry the profiler's overhead and are flagged in the receipt; **the timed s/step of the lane is never read from them** (the arm's `s_per_step_median_11plus` is over an 8-step arm here and is quoted only as context). Beside each arm `nvidia-smi dmon -s ut -d 1` samples sm %, memory %, and PCIe rx/tx MB/s every second (`logs/dmon_<arm>.txt`). Proven on the CPU self-test (`tp4_arm.py --selftest --profile-steps 1`: every arm emits a `PROFILE` line and its receipt).

Box E of `bench/tp4/tp4_run.sh` (`TP4_BOX=E`, family `qwen3prof`): one RTX 5090, steps 8 (3 warm, 3 profiled, 2 after), two arms at the field recipe from the same tokens file:
1. `e4b/fused_attn4` — T1's arm (`enable_fast_train(dgrad=True)` + int4 attention).
2. `unsloth/ckpt_unsloth` — the comparator at the same fixture (latest PyPI Unsloth at launch, recorded; its own venv; `install_failed` row if it does not import).

## Registered predictions (falsifiable)

- **P1 (the GPU is idle, not slow):** e4b's profiled window has `device_busy_fraction ≤ 0.30` (T1's 16 % utilisation was the GPU, not a sampler artefact). Refuted if ≥ 0.60 — then the step IS device time and the nvidia-smi reading was misleading, and the lane's question dissolves into "which kernels".
- **P2 (where the host time is):** in e4b's window, `routing + autograd` ≥ 50 % of CPU op self time — the HF MoE block's per-expert dispatch (index/gather/scatter/where/nonzero/topk …) and the autograd engine over thousands of small ops, not the fused kernels' own launches. Registered alternatives: `fused_kernel` ≥ 50 % → the launch path of the fused kernels themselves; `memcpy` ≥ 30 % → data movement; `optimizer` ≥ 30 % → the 8-bit optimizer step.
- **P3 (dispatch-bound structure):** e4b's `device_events_per_step ≥ 20,000`. Refuted if < 5,000 (then few launches, each long — not a dispatch story).
- **P4 (the comparator):** Unsloth's `device_busy_fraction` is ≥ 2× e4b's on the same box and fixture (its step is less host-bound). Registered alternative: within 1.3× → the host-boundness is the shared HF MoE training structure, not e4b's.
- **P5 (not the bus):** PCIe rx + tx averaged over e4b's arm < 2 GB/s (experts are VRAM-resident, `offload=0`; nothing streams). Refuted if ≥ 8 GB/s.

## Decision rules

- P1 ∧ P2 (routing/autograd) → register the training-side lever: a **dispatch-free MoE training forward** (sorted-rows grouping, the structure the serving hot-residency path already has) for `enable_fast_train`, with its own pre-registration and a parity gate (tp1's B2/C2 rule) before any speed is quoted. P2 = fused_kernel → a launch-count lever (CUDA-graph the fused step, or fewer launches per layer) — kernel side. P2 = memcpy → placement. P2 = optimizer → the optimizer, not e4b. ¬P1 → the census is of kernels: read the device table, register from it.
- P4 holding says Unsloth is not paying this cost — the tp4 field-recipe comparison (`e4b slower at the field recipe`) is then a HOST-structure gap, and the lever above is what closes it. P4's alternative says both pay it and the comparison stands on other legs.
- Nothing is fixed in this lane. The read goes into this file; the next lane is registered from it.

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p45-qwen3prof` | RTX 5090 | 0.65 | 1.5 h | $0.98 |

Under the $35 cap. STOP: a box not of the class (refused, exit 15); egress below the floor; a fetch alarm → `not_run`; the Unsloth venv failing to install/import → its row is `install_failed` and e4b's arm still runs (the lane's primary question needs only e4b's window). A second host-limited draw → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/p45-qwen3prof/tp4/` — `qwen3_e4b_fused_attn4.json` + `_profile.json`, `qwen3_unsloth_ckpt_unsloth.json` + `_profile.json`, `logs/dmon_*.txt`; read by `bench/p45/p45_reduce.py` (P1–P5 against the numbers above, the top tables quoted). Amendments dated below, before the data they touch.

## Amendments

### Amendment 1 (2026-09-19 ~05:40Z, after run 2, before the comparator redraw) — the Unsloth arm alarmed; it is redrawn alone

Run `p45-qwen3prof-2`: the e4b arm profiled cleanly (P1, P3, P5 hold; P2 names the adapter-dispatch family — `RESULTS-p45.md`);
the Unsloth arm was killed by its alarm (1,388 s) before its first step. The arm alarm is derived from the run's remaining time,
and the e4b arm's profiling overhead — three 38.6 s profiled steps plus ~30 min of `torch.profiler` trace processing at 0 % GPU
(94 GB RSS for ~6 M op records per step) — had consumed it. `TP4_PROF_ARMS` (default `e4b,unsloth`) lets a redraw run one arm;
`p45-unsloth` (`TP4_PROF_ARMS=unsloth`, one 5090, 1.5 h, ≤ $0.98) reads P4 against run 2's e4b window. P4's thresholds are
unchanged; P2's read stands as written (the registered families did not include the adapter matmuls; that omission is
recorded, not repaired after the fact).
