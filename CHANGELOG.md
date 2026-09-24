# Changelog

## 0.37.3 — 2026-09-24 — a memory-safety fix: the int4 singleton GEMV no longer writes past its preallocated split-K buffer at T > 1 (S2 verify); plus default-off lane switches and pre-registrations for P63, P65 and P67

### The int4 singleton GEMV no longer writes past its preallocated buffer at T > 1 (a fix; T = 1 unchanged)

- **The defect.** `enable_serve_experts_int4` sizes each store's split-K partials buffer `st["part"]` for one token's
  `top_k` rows. The singleton branch of `hot_residency._fused_over_stack` passed it to `gemv_int4_b32` at every T,
  and under `FORCE_SINGLETON_GROUPS` at T > 1 (the S2 verify default, `--moe-grouping singleton`) the GEMV has
  `T * top_k` rows and plans its split count from them. Measured on the NAS A2000 with the buffer a view into a
  sentinel-filled one: at T = 17 the call wrote 16,384, 12,288 and 180,224 fp32 elements past it at the OLMoE gate_up
  and Qwen3 gate_up/down shapes; T = 1 and T = 2 wrote nothing past it
  (`bench/p63/rehearsal-a2000/part_oob.json`). Found while mapping lane P63's routes (#708).
- **The fix.** `_int4_part_or_none` keeps the buffer when it has the rows the call's own plan needs and passes None
  otherwise, so the wrapper allocates its own (through the graph pool under capture, as the device-grouping GEMV
  branch already does). The arithmetic is unchanged: the split count was always the call's own plan. The fit is
  memoised per (rows, device) on the store, because this runs on every decode expert call and the eager B=1 step is
  host-bound. Each shape is planned once, then costs a dict lookup.
- **Tests.** `tests/test_int4_singleton_part_fits.py`, CPU with the kernel stubbed: one token keeps the store's
  buffer, a 17-token call gets None, a buffer the call's smaller plan fits is kept, and the shipped planner (where
  triton imports) decides the same way at 26 and 170 SMs.

### P65 Amendment 2: the egress floor measures the fetch's own four-stream path (#710; a lane change, no library change)

- **The single-stream egress probe under-read the fetch by 2.3×.** It read 36.1 MB/s on a box whose 4-worker Granite fetch then ran at ~83.5 MB/s. It refused every box the lane drew: 97.8, 36.1 and 21.8 MB/s, the last on the first reading box.
- **Amended.** `p65_run.sh` probes four parallel 50 MB ranges in Python, since the image has no `curl`. The floor is 80 MB/s aggregate (Mixtral ~19.5 min); the deadline logic still skips Mixtral rather than cut it. Registered in `bench/p65/P65-PREREG.md` "Amendment 2", before any reading data.

## Unreleased

### P67 Amendment 1: the proof and draw rate ceilings follow the verified-5090 market (#713; a lane change, no library change)

- **Why.** The registration priced P67 at $0.69/h; the verified-5090 floor has moved between $0.66 and $0.81/h in a day, and P66 burned three proof launches on provider refusals before amending. Dated `2026-09-24 16:21Z`, before any rental in this lane.
- **Amended.** Proof ≤ $0.75/h at its 600 s guard (≤ $0.125, at most three attempts); draw ≤ $0.75/h at its 3 h guard ($2.25 line); lane line $2.625. Rates and dollar lines only — arms, fixture, knobs, predictions and decision rule are unchanged, and the pinned tree is the registration's (`tests/test_p67_staged_pin.py`).

### Lane P66 read (#711): residency's launch cost is a fixed per-layer count; the bytes model under-predicts the gather on a gen 4 link (docs, register; no library behaviour change)

- **The reading.** `p66-5090-2` ran on one RTX 5090 for $0.3752; the lane cost $0.5285 over nine launches (three $0 price refusals, one Vast API NOT_RUN, one burned id, the proof, an rc-14 egress refusal, the reading), every teardown proven. All six instrument gates held. P1–P6 held, P7 was refuted, P8 and P9 are informational. Full read: `bench/p66/RESULTS-p66.md`; the box's own reducer output and rows under `bench/p66/receipts/`.
- **The count is fixed (P1, P2).** The pipelined engine adds **+10 launches, +2 copies, 0 syncs per MoE layer per token** with `n_hot > 0` (+7 / +2 / 0 at `n_hot = 0`), spread exactly 0 across cold fraction 0 / 0.5 / 1.0, eager and captured. Level M agrees by API name to the launch. Registered as `e4b.serve.p66.qwen3.pipelined-residency-fixed-count.5090.2026-09-24`.
- **The fixed tax (P3).** 1.184 ms/token of added device time captured (576 rows at 2.06 µs); 8.05 ms/token of eager wall for the same 576 submissions on a host-bound step. `…pipelined-fixed-tax.captured…`. At the RFC's 17.19 % cold the tax is 8.9 % of residency's captured cost (P8): launches cannot explain an RFC-size gap in this engine.
- **P7 REFUTED.** cold_deadline's bytes-over-link prediction is under the measured gather by **1.57–2.02×**; the gather runs at the box's *single-copy* link rate (14.4 GB/s implied against a 14.72 GB/s single-copy probe), not the 40-deep back-to-back 23.07 GB/s the calibration blob records. On the A2000 rehearsal the two probes agreed and the ratio was ~1.0. Registered as `…pipelined-gather-over-cold-deadline…`; the cause is not attributed.
- **MXFP4 and the hybrid tier (P4, P5, P6)** read as registered: NVMe engine 4 syncs/layer fixed (−1 / +3 / +3 against all-resident), pinned = all-resident, hybrid tier 4 / 10 / 8 syncs by layer composition. `…gptoss.mxfp4-and-hybrid-sync-census…`. The pinned-staging follow-up is not licensed (its licensing measurement was not taken).
- **Follow-up filed in grouped-nf4-gemm** on `kernel/cold_deadline.py`: a per-step fixed term, a measured UVA-read efficiency factor before any RFC comparison, and the mixed-layer dispatch term the deadline destination omits. No default moves.

### Lane P68 read (#725): a verify built from T = 1 calls is bit-identical to decode; the served T > 1 paths are inside the bar (docs, register; no library behaviour change)

- **The reading.** `p68-5090-2` ran on one RTX 5090 for $0.2386. The whole lane cost $0.2799, including the proof and one NOT_RUN on a stale host key, and every teardown is proven. G0 held on both stacks: 27 predictions held and 1 was refuted. Full read: `bench/p68/RESULTS-p68.md`.
- **Which component.**
  - The projections alone or the attention core alone leave layer-0 attention differing, and both together move the first difference out of it.
  - Forcing projections, core, router, LM head and norms to their T = 1 calls makes a verify **bit-identical to decode on both stacks** (264 / 264 positions). Registered as `e4b.serve.p68.qwen3.verify-from-t1-calls.row-exact.5090.2026-09-24`.
- **The refuted cell: int4 prefill under full forcing.** 159 of 160 positions differ as predicted, but position 0 is exact, because rotation there is the identity.
- **The size.** Every served T > 1 cell is WITHIN the shipped bar over 952 / 896 / 2,048 positions: KL ≤ 0.017, top-1 ≥ 0.946 (lower CI ≥ 0.9355). Registered as `e4b.serve.p68.qwen3.t-gt-1-vs-t1.within-bar.5090.2026-09-24`. P63's over-bar reads were small-sample top-1, and #725 closes.

### Lane P68 registered (#725): which part of the attention makes a verify or a prefill differ from T = 1 decode, and is it over the bar once enough positions are read? (bench only; nothing in the wheel changes)

- **The question.** P63 found every first difference between T = 1 decode and a verify or prefill in layer-0 attention. 16 of its 45 cells were under the shipped top-1 bar, all on top-1 over 64–160 positions and none on KL. P68 asks which component makes the difference, and whether the bar is crossed once enough positions are read. Prereg: `bench/p68/P68-PREREG.md`.
- **The instrument.** `bench/p68/p68_probe.py` reuses P63's probe and comparison module unchanged, and adds forcing arms.
  - A forcing makes one component (projections, attention core, router, LM head, norms) compute a multi-row call one row at a time, as T = 1 decode does.
  - The prediction is that forcing all of them makes a verify bit-exact on both stacks.
  - A size reading of the served configuration runs over P64's committed wikitext rows: ~950 verify and 2,048 prefill positions per cell, with bootstrap intervals.
  - `bench/p68/p68_reduce.py` applies the registration.
- **Rental lessons applied.**
  - The proof's guard is 0.23 h.
  - Egress is measured the way the fetch runs: four parallel ranges, in Python. The reading refuses below 80 MB/s (rc 13).
- **Tests.**
  - `tests/test_p68_force.py`: the forcings on CPU (row order, inert at one row, restoration, the core's prefix).
  - `tests/test_p68_reduce.py`: every gate and verdict branch.
  - `tests/test_p68_staged_pin.py`: the staged pin, the egress probe, the size rows.

### P66 Amendment 1: the proof and reading rate ceilings follow the verified-5090 market (#711; a lane change, no library change)

- **Why.** Two proof launches were refused at the provider before any instance existed, at $0: the cheapest verified RTX 5090 was $0.7237/h, then $0.6604/h, against the registered $0.65/h.
- **Amended.** The proof is ≤ $0.75/h with a 0.2 h guard (≤ $0.15), and the reading ≤ $0.75/h with its 2 h guard (≤ $1.50). The $2 lane ceiling holds (≤ $1.95). Registered in `bench/p66/P66-PREREG.md` "Amendment 1", before any data.

### Lane P65 read (#710): Granite's per-expert selector is TWO_ARMS; Colla-Q's cross-domain claim replicates on OLMoE and Mixtral (docs, register; no library behaviour change)

- **The reading.** `p65-5090-4` ran on one RTX 5090 for $0.5491. The whole lane cost $0.9369 over ten runs: six proofs, three boxes refused before any data, and the read. Teardown is proven for every one.
- **Completeness.** All three censuses are complete and self-checked. The reducer reproduces its output byte for byte from the committed receipts. Full read: `bench/p65/RESULTS-p65.md`.
- **Granite: TWO_ARMS.** Every ranking survives the wikitext → C4 change: `rel_act` 0.932, ρ·error 0.910, entropy 0.882 and routing frequency 0.891. Entropy is not redundant with `rel_act` (ρ ≤ 0.055).
  - The selector is written in the new `docs/SPECULATIVE_LANES_ADDENDUM_4.md`: rank by `rel_act` and by ρ·error, with frequency as the baseline, compared at matched bytes on K8. The OpenTimestamps-anchored plan documents are not edited.
  - Register row: `e4b.quality.p65.granite.selector-two-arms.5090.2026-09-24`.
- **OLMoE and Mixtral: NOT_WRITTEN.** Neither has a per-expert premise.
  - Colla-Q's comparative claim replicates on both: +0.421 and +0.479, recorded as `e4b.quality.p65.collaq-stability.olmoe-mixtral.5090.2026-09-24`.
  - Routing-frequency hot sets barely transfer between texts: Jaccard 0.141, and 0.125 (chance) on Mixtral.
- **One prediction refuted.** P1 on Mixtral: `rel_act`'s domain penalty there is 0.159 against the 0.15 survival line.
- **P44-a's census row now names its Mixtral layers.** They are {0, 1, 2, 10–22}, verified from its receipt.
- **STATUS.** Its P44 paragraph no longer calls Gemma-4's gap a served-model defect: #597 closed on P47–P51.

### Lane P64 read (#709): the int4 experts' int8 activation step at B = 1 decode is below Qwen3's arithmetic-order floor (docs, register; no library behaviour change)

- **The reading.** `p64-5090-1` ran on one RTX 5090 for $1.4193, after the proof `p64-prove-1` ($0.0201). Teardown is proven, and the validity checks were VALID.
- **What it ran on.** The licensed pack (`sha256:0c9955a9…`), rebuilt on the box and verified by `verify_artifact`. The build's K8 read ppl 6.36709, as P55x's did.
- **G_exp was INDISTINGUISHABLE.** KL(bf16 activations ‖ int8) was 0.003392 / 0.002527 nats/token on wikitext / c4val1, which is 0.61× / 0.77× of the in-lane floor. dNLL's intervals include 0. Recorded as `e4b.serve.p64.qwen3.b1.expert-int8-step.5090.2026-09-24`. W4A8 expert decode stays, and #709 closes for the experts.
- **G_attn was UNREAD.** The `a16_all` passes, the second floor pair and the NF4 anchor were skipped for time: a host-limited 67-minute pack build left too little deadline. The attention half is #728. Full read: `bench/p64/RESULTS-p64.md`; receipts: `bench/p64/receipts/`.
- **P59's B = 16 KL row now states its scope.** Its scorer leaves `DEVICE_GROUPING` off, so its decode experts took dequant + bf16 and never ran the int8 step. The fusion licence is unaffected.

### Lane P63 read (#708): which expert routes give a token the same bits alone and inside a verify or prefill (docs, register, tests; no library behaviour change)

- **The reading.** `p63-5090-1` ran on one RTX 5090 with Qwen3-30B-A3B for $0.2881, after the proof `p63-prove-1` ($0.0115) at the same commit. Teardown is proven. **Every registered prediction held**, and no path was outside its fp64 accuracy bound in 179 records. Full read: `bench/p63/RESULTS-p63.md`. Receipts: `bench/p63/receipts/`, and the reducer reproduces its JSON from them byte for byte.
- **Row-exact routes, registered.** Every experts module (48/48) returns each token's rows bit-equal to its T = 1 decode call on three routes:
  - the int4 store under `FORCE_SINGLETON_GROUPS` (`e4b.serve.p63.qwen3.int4-singleton.row-exact.5090.2026-09-24`);
  - the int4 store under `DEVICE_GROUPING` up to 256 routed rows (`…int4-device-gemv.row-exact…`);
  - the NF4 stack under `FORCE_SINGLETON_GROUPS` on the dot-pad GEMV (`…nf4-singleton.row-exact…`).
  The kernels' own row-invariance is registered in grouped-nf4-gemm (#399).
- **Different functions by row count, recorded in STATUS** (each path keeps its own quality licence; P63 moves no default):
  - the default T > 1 routes: the int4 store's dequant + bf16 matmul, and NF4's M-tile;
  - the router epilogue's fp32 weights at ≤ 64 rows against bf16 above (#726).
- **End to end, no position is exact.** Every first difference is at layer 0 in attention, never in the experts. KL mean is 0.008–0.033 nats/token. 16 of 45 cells are under the shipped top-1 bar of 0.93 (all on top-1, none on KL). Filed as #725.
- **New test.** `tests/test_p63_row_exact_gpu.py` asserts the three routes with `torch.equal` on the plan and dispatch a 5090 takes, with a control that must fail. It skips on CPU, and ran 9/9 on the NAS RTX A2000.
- **Corrected.**
  - `hot_residency.py`'s singleton comment. It said the singleton path is bitwise-equal to the grouped path at every T, "pinned in CI". That holds at T = 1 on NF4 only; at T > 1 the grouped path is a different function.
  - `tests/test_singleton_groups.py`'s docstring. It pins the dispatch algebra through a mocked GEMM, not the arithmetic.

### Lane P66 registered: the residency launch census (`bench/p66/`, for #711; nothing in the wheel changes)

- **The question.** What residency adds per token, in CUDA launches, copies and host syncs, against the
  all-resident step on the same box. Whether that moves with cold fraction. Whether grouped-nf4-gemm's cold cost
  model (`kernel/cold_deadline.py`, bytes over the link plus bytes over VRAM) predicts the measured transfer.
  vLLM #57794 left all three open, with an unexplained 2–4× cold-cost gap.
  - Paths: pipelined NF4 at five hot fractions including 0 and the RFC's 17.19 % cold; the hybrid tier (VRAM +
    NVMe); grouped-nf4-gemm's MXFP4 pinned and NVMe engines.
  - Each path is measured against its family's all-resident step, captured where capture works.
  - Families: Qwen3-30B-A3B (NF4) and gpt-oss-20b (native MXFP4) on an RTX 5090. The prereg is
    `bench/p66/P66-PREREG.md`.
- **The instrument.**
  - `p66_census.py` drives one MoE token (every layer's real expert bytes, routing set per window) through each
    engine. It counts CUDA API calls inside record_function regions, attributed to the engine's own phases, plus
    device rows, a sync-debug pass, timing, and a graph replay checked against eager.
  - `p66_reduce.py` is pure Python: the gates, the attribution against the reference, the cold-fraction test and
    the registered verdicts.
  - `p66_step.py` runs `bench/hybrid-g9/step_decomp.py` unchanged, swapping only the pipelined arm's hot sets, for
    the served-step context.
  - Runner and driver are the P60/B374 pattern (`p66_run.sh`, `p66_drive.sh`, `staged.sha256`).
- **Rehearsed on the NAS RTX A2000**, disclosed in the prereg and in `bench/p66/rehearsal-a2000/`, which is not a
  reading. The last round ran `p66_run.sh` itself end to end (rc 0) under documented rehearsal-only overrides.
  - The pipelined engine added a fixed +10 launches and +2 copies per MoE layer (+7 and +2 with no hot experts),
    and 0 syncs, identical at cold fraction 0, 0.5 and 1, eager and captured (replays bitwise equal to eager).
  - The v0-dispatch hybrid tier moved with layer composition. The MXFP4 NVMe engine held a fixed 4 syncs per
    layer.
  - The served step through step_decomp reproduced the MoE token's API deltas exactly.
  - cold_deadline's bytes model predicted the pipelined gather at 0.99–1.05× (MXFP4 engines 0.95–1.01×).
  - The rehearsals changed the instrument seventeen times before registration, each listed in the prereg. Three
    examples:
    - The profiler's device view drops records, eager and captured, while host counts stay exact. So eager
      counts come from host API calls, and captured counts from the graph's own node list: the CUDA runtime's
      `cudaGraphGetNodes`, cross-checked against torch's dot dump. That dump writes nothing, silently, on torch
      2.8 unless the graph is kept.
    - A +1.7 ms "GEMV slowdown" in round 1 was the shared card's state (+0.04 and +1.37 on later rounds). So the
      tax is split into added kernels and shared-kernel shift, and time bands are graded only on the registered
      box.
    - The runner's time guard skipped late arms by using their alarm caps as their expected times.
- **Two rentals.** The reading's guard is 2 h, so under the compute rule a **proving rental** comes first
  (`P66_MODE=prove`, 0.23 h guard, ≤ $0.15 per attempt, ≤ $0.45 over all).
  - It does the class and dud checks, an egress probe, the pinned install, the tripwire and the pin, and it times
    one checkpoint shard.
  - Following P65 Amendment 1, it records the reading-only floors (VRAM, disk, RAM, egress, pin) instead of
    enforcing them, because the proving box is not the reading's box.
  - Following P65 Amendment 2, egress is measured the way the fetch runs: eight parallel 50 MB ranges in Python,
    since the reading's `snapshot_download` uses eight workers. The reading refuses below 80 MB/s (rc 14).
  - G3 records the sha256 of the eager and replayed outputs, so a bit statement across boxes rests on hashes, not
    on difference counts.
  - It was rehearsed through the real install path, with those floors genuinely failing on the A2000.
- **Tests.**
  - `tests/test_p66_reduce.py`: counting rules, gates and every verdict branch, on synthetic event lists.
  - `tests/test_p66_census_helpers.py`: the profiler-tree walk, routing and schedule, on CPU.
  - `tests/test_p66_staged_pin.py`: the staged pin, the driver's mapping, that the driver never forwards the
    runner's rehearsal-only overrides, that the proving mode stops before any measurement, and that every arm's
    expected time is below its alarm.

### `E4B_INT4_DECODE_A16`: bf16 activations at T = 1 on the int4 expert store, as a quality instrument; lane P64 registered (#709)

- **The flag.** `E4B_INT4_DECODE_A16=1` (default off; `hot_residency.DECODE_A16`, the `FORCE_SINGLETON_GROUPS`
  pattern) routes the int4-b32 store's singleton-groups calls through the prefill branch. That is T = 1 decode in
  every default configuration. Each routed expert is dequantised and bf16-matmul'd over the same int4 bytes, and
  `quant_x_rows` is not called. There is no new kernel.
  - **Not covered:** the device-grouped batched-decode routes and the MXFP4 store.
  - **Eager only:** under CUDA-graph capture it refuses with a sentence, because the dequant loop reads the expert
    ids on the host.
  - **Off:** behaviour is unchanged.
- **Tests.** `tests/test_int4_decode_a16.py` pins both states on CPU with the int4 kernels stubbed: T = 1 is the
  GEMV on int8 activations when off, and the prefill branch bit for bit when on. It also pins that T > 1 is
  untouched, that the device-grouped route is not covered, and the capture refusal.
- **Lane P64** (`bench/p64/`: pre-registration, `kl_a16.py`, `p64_reduce.py`, `p64_run.sh`, `p64_drive.sh`,
  `staged.sha256`; `tests/test_p64_staged_pin.py`, `tests/test_p64_reduce.py`).
  - **What it reads:** the decode-scored KL A/B of that step on Qwen3-30B-A3B's licensed int4 stack, at B = 1, on
    wikitext and c4val1. The scorer is P59's, one row at a time. The floor is the arithmetic-order floor measured in
    the same instrument.
  - **The proving rental:** `P64_PROVE=1` is the proving mode the compute rule asks for before a guard over 1 h.
    It runs the install, tripwire, self-test, K0, `kl_a16.py --prove-flag` (the flag on the card's real kernels)
    and an egress probe, with no model and no pack.
  - **What was run:** the whole runner was rehearsed on the NAS RTX A2000 on OLMoE, rc 0, validity VALID, with
    every pass's counts exactly as registered. It is in `bench/p64/rehearsal-a2000/`, NOT a reading.
  - **What the rehearsal fixed:** the runner gates the pack on `verify_artifact`, not on the build process's exit
    code (a failed K8 cross-check had discarded a complete pack), and the proving run's egress probe is python (the
    image has no `curl`). No box was rented.
- **A correction to #709's premise, in the pre-registration.** P59's B = 16 KL never ran the int8 step on the
  experts. Its scorer leaves `DEVICE_GROUPING` off, so T = 16 decode takes the host-grouped dequant + bf16 branch.
  The timed B = 16 arms run the device-grouped GEMV on int8 activations.

### P65 Amendment 1: a proof records the reading-only host floors instead of refusing on them (#710; a lane change, no library change)

- **Three proving draws, none reached the install.** The first box missed the host scale-and-add floor, 0.152 s against 0.15 s. The second was a launcher refusal: I named the first receipt in the wrong exclusion class. The third missed the egress floor, 97.8 MB/s against 100. That spent $0.1009 of the registration's $0.15 proof budget. Those floors exist for the reading's Mixtral census. The proof's box is not the reading's box, and the proof never runs Mixtral.
- **Amended.** In a proof, `p65_run.sh`'s new `floor()` records the RAM, scale-and-add and egress floors (`floor_would_refuse_reading`) and continues. Class, dud and disk still refuse, and a reading refuses on every floor exactly as before.
- **Guard and budget.** The proof guard is 0.23 h, ≤ $0.15 each; launcher boot had taken 3–5 min of the old 10 min guard. The proof budget is ≤ $0.45 over all attempts, and the lane ceiling is $2.05. Registered in `bench/p65/P65-PREREG.md` "Amendment 1", before any reading.

### P67 registered: training parity read against a per-family floor, not against zero (#713) (one default-off switch in the wheel; bench tooling)

- **`bench/p67/P67-PREREG.md`, registered before its draw.** tp1's band compares an accelerated arm with the
  reference against a constant 0.05 **and against zero**, and on Gemma-4 every accelerated path fails it, the
  kernel-free one included. The lane replaces "against zero" with a measured floor: the SAME reference path, same
  box and session, run again (a plain repeat, and four runs whose per-expert loop sums the experts in a fixed
  permuted order, correct by construction). The band is `max(0.05, 3 × F_hi)` on both of tp1's quantities, on the
  TRAIN loss, with at least 3 admissible floor draws; fewer is NO-FLOOR, never FAIL. The batched arm is judged
  against that floor, not used as it: it shares code with the fused path and could not judge itself.
- **The existing receipts, re-read on CPU (`bench/p67/reread-existing/`): no session ever ran the reference twice,
  so no family has a floor draw.** Every Gemma-4 attention-4-bit row reads NO-FLOOR; every other row is a
  constant-band PASS, which the rule keeps as a PASS. The rental therefore covers Gemma-4 only: one RTX 5090,
  ≈ 2.2 h, ≈ $1.5, guard 3 h (approval line $2.07), preceded by a ≤ 10-min proving rental (`p56_prove.sh`, $0.115
  line). **Not launched.** `claims.json`, `capabilities.json` and STATUS do not change until it is read.
- **Rehearsed free on the NAS A2000** (`bench/p67/rehearsal-a2000/`, not a reading): on a small MoE the switch
  reached the loop on every call, a same-box repeat was not bit-identical, and every permuted run moved the
  trajectory (two of three already at step 0, which revised one prediction before registration).
- **In the wheel: `E4B_REFERENCE_EXPERT_ORDER` (`experts4bit_qlora/lora.py`), default off.** Unset, the reference
  loop is the shipped one and the helper is never called. `descending` or `perm:<seed>` visits the same experts in
  another order, which moves only the rounding of the sums. Anything else raises. `reference_order_stats()` reports
  what the loop did. Tests: `tests/test_reference_expert_order.py`.
- **Harness.** `tp4_arm.py` records `reference_order` on every e4b reference receipt and refuses the switch on any
  other arm (exit 19). `tp4_run.sh` gains an opt-in `TP4_P67=1` block (the floor arms, run last). `tp4_drive.sh` gains
  `TP4_RUNNER` / `TP4_EXTRA_STAGE`, which default to its own behaviour. `bench/p67/` adds the reducer, the box guard
  `p67_run.sh` (pins, registered knobs, host RAM ≥ 96 GiB), the controller `p67_drive.sh`, `registered.knobs` and
  `staged.sha256`. Tests: `tests/test_p67_reduce.py`, `tests/test_p67_harness.py` (which executes the new shell
  paths rather than parsing them) and `tests/test_p67_staged_pin.py`.

### P65 registered: per-expert activation entropy beside `rel_act`, and which ranking survives wikitext → c4val1 (#710; not yet run)

- **The calibration tap keeps a first moment, on request.**
  - `calibrate_expert_hessians(..., activation_means=)` fills a dict with each expert's fp64 mean gate/up input and
    down input, from the same tap, over the same rows. It uses `_ExpertHessianSink(means=)`.
  - Off by default. The Hessians are bitwise the same tensors either way, and the return value does not change
    (`tests/test_p65_entropy.py`).
  - The Hessians are uncentred (`2 X X^T`), so no statistic that needs per-channel means could be read from them
    alone. Nothing else in the wheel changes.
- **`bench/p65/`** (P65-PREREG.md):
  - `expert_entropy.py`: Colla-Q's activation-entropy ratio `rho = sigma2_within / sigma2_total` of each expert's
    output, read exactly from the tap's moments.
  - `p65_census.py`: P44-a's `census_row`, unchanged and RTN only, plus the entropy field, on wikitext-2 train and
    K8's c4val1 text. Each text runs in two disjoint halves; the full rows are combined exactly.
  - `p65_reduce.py`: within-layer rank stability (a split-half ceiling, and the domain shift at the same sample size),
    overlap with `rel_act` and with routing frequency (through `hot_sets_from_profile`), Colla-Q's cosine for
    replication, and the registered rule for S-C's selector.
  - `p65_run.sh` / `p65_drive.sh` / `staged.sha256`: the box/controller pair, with the pin, nonce, heartbeat and HF
    token staging. The host floors are refusals, not stalls:
    - RAM and disk;
    - HF egress ≥ 100 MB/s, because Mixtral is 93 GB;
    - a host-side Hessian update ≤ 0.15 s at Mixtral's 14336² shape. That update runs about 4,100 times, and the
      rehearsal measured 0.33 s for it on the NAS Xeon.
  - **`P65_PROVE=1`** is the proving rental the compute rule requires in front of a guard over 1 h (≤ 10 min,
    ≤ $0.15). It runs every refusal, the install and the tripwire, then Granite's first layer at `nseq 8` end to end.
    It writes `prove_census_granite.json` and `MODE=prove`, which the reducer never reads. The same cut-down census
    passed on the A2000 (`rehearsal-a2000/prove_config_*`).
- **Rehearsed on the NAS RTX A2000**, not a reading (`bench/p65/rehearsal-a2000/`). It found that the census walks
  layers in checkpoint-key order (0, 1, 10, 11, …). P44-a's "16 of 32" Mixtral layers were therefore most probably
  not 0–15; unverified here. P65 takes the first 16 of the same order (`--first-layers`).
- **Tests:**
  - `tests/test_p65_entropy.py`: known distributions, moments against raw rows, the sink, the census row keeping P44's
    `rel_act`, halves, texts, and layer order;
  - `tests/test_p65_reduce.py`: every decision branch on synthetic rows;
  - `tests/test_p65_staged_pin.py`.

### Lane P63 registered: does a token's output depend on how many rows share its forward? (#708; bench only)

- **What.** `bench/p63/`: the pre-registration, a GPU probe, its comparison module and reducer, and the rental runner.
  Nothing is claimed yet and no default moves.
- **The routes, mapped against the code.** The int4 store (T = 1 GEMV on int8 activations vs a dequant + bf16 matmul
  above it; `DEVICE_GROUPING`'s GEMV up to 256 rows), NF4 (dot-pad or scalar decode GEMV vs the TF32 M-tile), the
  three int4 attention buckets, `combine_rows`, and routes #708 did not list: the decode folds switch to the torch
  chains above 64 rows, the paged prefill attends bf16 K/V where decode and verify read fp8, and cuBLAS picks its dense
  kernel by M.
- **The probe.** Qwen3-30B-A3B on an RTX 5090, three stacks and fifteen sub-arms, each a route chosen by
  configuration. Each sub-arm is read three ways:
  - end to end: per token, layer and site, the T = 1 incremental-decode control against the same token in 17- and
    16-row verifies and a 160-row prefill;
  - a module replay of the recorded T = 1 inputs as one n-row call;
  - a kernel census against fp64.
- **The design moved before registration, and the prereg says where.**
  - Per-layer ULPs are reported only over significant elements, the B393 lesson.
  - The fp64 accuracy bound is the defect line and not a classifier, because at served K it cannot see a bf16-rounded
    weight. The REORDER vs PRECISION classes come from the kernels' declared operand models.
  - cuBLAS paths are rerun with torch's `allow_bf16_reduced_precision_reduction` off for that line.
- **Rehearsed on the NAS A2000** (OLMoE, not a reading; `bench/p63/rehearsal-a2000/`). It found a route #708 did not
  list: the fused router epilogue returns fp32 routing weights up to 64 rows and the upstream router bf16 above. That is
  registered as P7.
- **B393's side diagnostic** (`fma_attribution.py`, grouped-nf4-gemm #397, present from `f88df1e`) rides along,
  guarded and non-fatal.
- **The rental plan** is a 1.5 h guard, preceded by the ≤ $0.15, ≤ 10 min proving rental the standing rule requires,
  ≤ $1.15 in all.
- **Tests.** `tests/test_p63_compare.py`, `tests/test_p63_reduce.py`, `tests/test_p63_probe_plumbing.py` (a tiny random
  Qwen3-MoE on CPU) and `tests/test_p63_staged_pin.py`.

### The fused MoE combine's call-site comment says what lane B393 measured (a comment; no behaviour change)

- **`engines/hot_residency.py`'s call site said `combine_rows` takes "the same order and roundings as the chain
  below".** grouped-nf4-gemm's lane B393 measured otherwise on an RTX 5090 (grouped-nf4-gemm#397, #393), over 144
  census cases at every served family's shape:
  - **Not bitwise.** The kernel differs from the chain in 95 cases, 392 of 9.07 M elements. It sums in slot order,
    with a fused multiply-add exactly so on sm_86; the chain rounds each product and sums in torch's order.
  - **Both correct.** Both are within the error bound of a correct fp32 summation in every case, at the same max
    ratio (claim `gnf4.kernel.combine-rows-accuracy.5090.2026-09-23`).
- **The comment now says so.** The default stays `E4B_FUSE_COMBINE=1`, because an accuracy-equal kernel is not a defect.
  The end-to-end size of the difference is #708's probe, with `E4B_FUSE_COMBINE=0` as its control arm.

### Lane B393's runner (`bench/b393/`, for grouped-nf4-gemm#393)

- **What it runs.** The box side of grouped-nf4-gemm's pre-registered lane B393: are `combine_rows` and
  `reduce_partials` bitwise equal to the torch chains they replaced? This repository runs `combine_rows` on every MoE
  layer by default, and its call site says the fused path takes "the same order and roundings as the chain below".
- **How.** `b393_run.sh` installs grouped-nf4-gemm at `GNF4_SHA`, clones the same sha for the census script (it is not
  in the wheel), and proves from a work dir holding only the census that `int4_b32` resolves to the installed
  package. It then runs the 414-case census once. The verdict is the census JSON, read against grouped-nf4-gemm's
  pre-registration.
- **Driver and pin.** The driver and the staged pin follow B374's pattern. `tests/test_b393_staged_pin.py` mirrors the
  pin test.

### Correction: Gemma-4 attention-4-bit training HAS receipts; it is unlicensed for a different reason (documentation; nothing in the wheel changes)

- **Two entries below said the arms had not run and had "no receipt", and pointed at a re-run (#703).** That was wrong,
  and I wrote it without checking P56. Since #435 the Gemma-4 attention-4-bit arms have run cleanly on three boxes:
  `tp4-c-4`, `tp4-c-parity-2` and `p56-gemma4-ladder-3`. Each converts all 115 structural projections (25 sliding
  layers × 4, and 5 full-attention layers × 3, which have no `v_proj`), and every arm is VALID.
- **Why it is still not supported.** tp1's internal training-parity band, a constant 0.05 against zero, fails for
  every accelerated path on this model, the kernel-free batched one included (0.054 final against the fused path's
  0.102; `e4b.parity.gemma4.train-floor`, `bench/p56/RESULTS-p56.md`). The band cannot tell a defect from the model's
  own sensitivity. P56 recommends a per-family floor measured by the smallest-perturbation arm, now filed as #713.
- **What changed.** STATUS, `capabilities.json` (four strings), ARCHITECTURE_SUPPORT, the QLoRA solution page and the
  two HARNESS_ERROR register notes now say this, and point at #713 instead of #703.

### `check_change_impact.py` and `check_dependency_floor.py` are one file each, shared with grouped-nf4-gemm (repository tooling; nothing in the wheel changes)

- **Byte-copied from grouped-nf4-gemm** (pjordanandrsn/grouped-nf4-gemm#394). With these, `SHARED` lists 14 files:
  every CI script both repositories carry except the per-package `wheel_smoke.py`. Each reads its role (`runtime` here)
  through `check_system_manifest.system_role` and keeps this repository's settings in its `PROFILES` entry.
  - **Change impact diffs against `git merge-base BASE HEAD`**, where this copy diffed against the base itself. On a
    branch behind its base, the old reading blamed the PR for a claim change the base made, and it passed a PR whose
    missing companion the moved base happened to supply. CI's discoverability checkout therefore now uses
    `fetch-depth: 0`, and the depth-1 base fetch is gone, since it would make the clone shallow again. A shallow
    checkout exits 2 with that hint.
  - **Change impact now also checks, here:**
    - a version bump needs CHANGELOG;
    - a dependencies change warns without README / capabilities;
    - a capabilities entrypoint change needs CHANGELOG;
    - a claim's `unit` change is a measured-result trigger;
    - every class a trigger reports must be named in `docs/change-impact.json`. That contract now describes these
      triggers.
  - **Dependency floor now also checks, here,** the kernel copy's version statements: this package's own version and
    tag links, the torch floor, Python/CI, requires-python and the licence, across the current documents (24
    statements, all agreeing). The floor still comes from pyproject's `fast` extra, and this copy's historical markers
    and anchored-document exemption are kept.
  - **Tests.** `tests/test_readability_checks.py` fixture data only: the fixture manifest names both packages, the
    pyproject gains a Source URL, and the fixture writes the current documents and a minimal contract. No assertion
    changed.

### `check_readme_claims.py` is one file, shared with grouped-nf4-gemm (repository tooling; nothing in the wheel changes)

- **Byte-copied from grouped-nf4-gemm** (pjordanandrsn/grouped-nf4-gemm#392) and now in `SHARED` (12 files). The two
  copies had forked by 399 diff lines. The unified file reads its role through `check_system_manifest.system_role`
  and reads the claim-id namespace from the register. This repository is `runtime`, so it keeps this copy's
  settings:
  - README is the results document, headed `status`;
  - a row's result is its last non-status cell;
  - the generated release block (`--write-release-block` still works, and is a no-op on the current README);
  - anchored documents are exempt;
  - a status word in backticks counts.
- **Now also checked here.** A missing README is reported as a finding (exit 1) instead of a traceback. A register
  with no dotted id is exit 2. The manifest is required.
- **Tests.** `tests/test_check_readme_claims.py` needed four edits, all mechanical: the unified functions take the
  kernel copy's signatures (document name, column tuple, pattern and profile arguments). No assertion was loosened.

### Gemma-4's tp2 attention-4-bit void was the harness's count check, not the converter's (documentation; nothing in the wheel changes)

- **The docs said both e4b attention-4-bit arms "died on the converter's own count check".** Before #435 the converter
  had no count check. The check that voided both arms is the tp2 harness's: `bench/h2h-20260906/tp2/tp2_arm.py:526-527`
  asserts `n_attn4 == 4 · n_layers` on the converter's return value.
  - **Corrected in** STATUS, ARCHITECTURE_SUPPORT, `capabilities.json` and the QLoRA solution page. The two
    HARNESS_ERROR rows keep their headline and gain a precision note.
  - **Why it matters.** A re-run under that harness as written would void again, because Gemma-4's `k_eq_v` layers put
    the library's census (`len(detect_attention_projections)`) below `4 · n_layers` by design. tp4's harness, which
    checks the structural census, is the one that has run these arms since (see the correction below).

### `check_system_manifest.py` is one file, shared with grouped-nf4-gemm (repository tooling; nothing in the wheel changes)

- **Byte-copied from grouped-nf4-gemm** (pjordanandrsn/grouped-nf4-gemm#391), and now listed in `SHARED`. The two
  copies had forked by 734 diff lines: this repository's enforced 39 rules and the kernel's 31. The unified file reads
  its role from the manifest (`runtime` here) and runs that role's rules.
  - **Runtime role, as before:** the `fast` floor, every kernel-pinning extra, and the CI kernel pin as a release-tag
    commit (still the only network read).
  - **Now also checked here:** an ssh-form Source URL; the `Kernel:` URL against `packages.kernels.pypi`; the shape of
    every compatibility record, including the non-current 0.34.x one; every clause of a range, not just the first;
    ownership duplicates or overlap; unique invariant ids; the router's size. A `fast` pin with an upper bound
    (`>=0.30.0,<1`) now fails locally too.
  - **Output.** The check no longer stops at the first finding, so later findings print as well.
  - **Parity.** On `main` its verdicts match the old copy's in both CI forms, and the existing tests
    (`test_check_system_manifest_pin.py`, `test_readability_checks.py`) pass unchanged. No importer here needed an
    edit.

### Gemma-4 attention-4-bit: the docs point at the open re-run, not the closed count-check bug (documentation; nothing in the wheel changes)

- **Six places said Gemma-4 attention-4-bit training is "NOT supported pending #412"**, and STATUS listed #412 under
  "What is open": `docs/STATUS.md`, `docs/capabilities.json` (four strings), `docs/ARCHITECTURE_SUPPORT.md` and the
  QLoRA solution page. #412 was closed on 2026-09-23 as fixed by #435, so the docs pointed at a closed issue.
  - **What #412 fixed.** It was the converter's count check (100 of 120 projections), which killed tp2/P40's two e4b
    attn4 arms. #435 now detects projections by structure, with `v_proj` optional on `k_eq_v` layers.
  - **What stays open.** The configuration stays **not supported**. This entry first said nothing had re-run those arms
    and pointed at a re-run, #703. That was wrong: the arms have run since, as the correction below records.
  - **Register.** The two HARNESS_ERROR rows keep #412 as their evidence and gain a note: fixed by #435, not re-run,
    #703.
  - The dated tp2 receipt README is unchanged, because a receipt is a record.
  - The site's `qlora-fused-moe-experts` FLAG still fires. It is a true family-scoped caveat: the capability is
    supported, and one family's attention-4-bit cell is not.

### The serving-position WARN reads STATUS's position section (repository tooling; nothing in the wheel changes)

- **`scripts/check_capabilities.py` (shared; copied byte-for-byte from grouped-nf4-gemm, pjordanandrsn/grouped-nf4-gemm#389).**
  The WARN takes "the position" to be the newest `area: serve` claims `docs/STATUS.md` quotes. It read the whole file,
  so P61's diagnostic read counted as the position: that expert-GEMV cost split, measured 2026-09-23, is quoted under
  "What is open". The capability then warned on every CI run that it "headlines an older lane".
  - **The fix.** The rule now reads only the text before STATUS's first `## What changed` heading. The position here is
    P58's 2026-09-22 rows, which `serve-moe-on-consumer-gpu` cites, so the warning clears with no data edit.
  - **Tests.** Two new tests in `tests/test_check_capabilities_serving_position.py` use the real STATUS layout:
    - a newer lane quoted only after "What changed" is not the position;
    - a newer lane in the position section still warns.
  - **Mutation check.** With the old whole-file read restored, the first test fails.

## 0.37.2 — 2026-09-23 — documentation, register data and repository tooling only (under the package only `__version__` changes): the claims register has ONE schema, shared with grouped-nf4-gemm; lane B374's runner, whose read holds both predictions

### The claims register has ONE schema, shared with grouped-nf4-gemm, and this register is migrated to it (repository tooling and data; nothing in the wheel changes)

- **`scripts/check_claims_register.py` and `docs/claims-schema.md` are one file each, byte-identical in both
  repositories** (both now in `SHARED`). The two copies had drifted into two schemas that refused each other's data (14
  findings one way, 38 the other). The converged rules, in `docs/claims-schema.md`:
  - **Locations.** An evidence path or a `quoted_in` entry is `path` or `path#anchor`. The path is a file in the git
    tree at HEAD, never a directory, annotation or glob. The anchor is `L<n>` / `L<n>-L<m>`, or the anchor
    **github.com renders** for a Markdown heading, so every location is a working link. The checker's anchors match
    github.com's on all 371 headings of both repositories' READMEs, CHANGELOGs and docs.
  - **Evidence** is a location, `{"url"}` (an issue or pull request of a system repository), or
    `{"repository": <package>, "path": <location>}` (resolved in a `--sibling` checkout). A public-run row's FIRST
    entry is a location, because the consumer site links it.
  - **Successors** are direct and named back. `superseded_by` names the active row itself, with no chains, and that
    row lists it in `supersedes`. A retired row may name its restatement.
  - **The file is closed.** Unknown row fields and top-level keys are findings, and `area` / `tier` / lane fields take
    only their values.
  - The licence, fingerprint, date and placeholder rules are the union of both repositories' old rules.
- **Tests.** One test file, identical in both repositories, produces every rule's failure. A mutation sweep disabling
  each of 32 rules in turn was caught every time.
- **This register's migration.**
  - 20 free-text `quoted_in` entries became locations. Entries whose file no longer quotes the id were dropped, and
    every original entry is kept verbatim in the row's `notes`.
  - The cross-repository entry names the package (`grouped-nf4-gemm`), not the GitHub slug.
  - The successor graph was made direct and bidirectional. `e4b.parity.gemma4.behaves` now points at
    `…no-reference`, not through `…chunk-free`, and three successors gained the `supersedes` back-links they lacked.
- **A defect the migration found.** The featured claim `e4b.train.energy-honest.scoped-a2000` linked
  `docs/METHODOLOGY.md#10-energy--measured-benchupstream…`. GitHub's anchor keeps the underscore of `bench/_upstream`,
  so the receipt link on the consumer site has never scrolled to its section. It is fixed.

### Lane B374's runner (`bench/b374/`, for grouped-nf4-gemm#374)

- **What it runs.** The box side of grouped-nf4-gemm's pre-registered lane B374: the word-addressed NF4 decode routes
  (wide loads, and dot-pad, the default on >= 160-SM parts at its census shapes) put past THEIR 2^31 boundary on an
  RTX 5090.
- **Two passes.** `b374_run.sh` installs grouped-nf4-gemm at `GNF4_SHA` and runs its
  `kernel/test_offset_boundary_words_gpu.py` twice, from two work dirs (the test puts its own directory first on
  `sys.path`):
  - against the installed kernels, where all four cases must pass;
  - against a copy with all six eid promotions removed, where all four must fail. Tripwires record where each pass
    resolved `nf4_grouped`.
- **One process per case.** Each GPU case runs in its own pytest process, so a fault cannot poison the verdicts of the
  cases after it.
- **Driver and pin.** The driver and the staged pin are K18's, re-pointed. `tests/test_b374_staged_pin.py` mirrors the
  K18 pin test, and the driver's dry run stages, starts and fetches as written.
- **The read (pjordanandrsn/grouped-nf4-gemm#385, 2026-09-23).** Run `b374-5090-1` on one RTX 5090 cost $0.0354, and its teardown
  is proven. P1 held: all 4 cases pass on the shipped kernels. P2 held: all 4 read the decoy at the int32-wrapped
  address once the six promotions are stripped. The runner ran unchanged at `cf80b0f`. The results and the claim
  (`gnf4.kernel.word-boundary-wide-dotpad.5090.2026-09-23`) live in grouped-nf4-gemm, the repository that owns the
  kernels.

## 0.37.1 — 2026-09-23 — documentation and repository tooling only: the same-box vLLM comparison is P58's everywhere, the cross-host pack limitation is P55x's, CI runs the census cross-check, and the CI scripts shared with grouped-nf4-gemm start to become one file

**0.37.1.** Nothing a user imports changed: the package code is identical to 0.37.0's (under `experts4bit_qlora/`, `git diff v0.37.0` changes only the `__version__` literal). The documentation that ships with it is corrected. The README (which is also the PyPI description), the serving capability and the serving solution page led with the 2026-09-05 comparison against vLLM 0.28.0; they now quote the current same-box one (P58: vLLM 0.30.0 decodes 1.087× faster than this package's current int4 stack at B=1 and 1.396× at B=16, bounded to one box and prompt set). The capability's statement that the streamed calibration does not reproduce across hosts is replaced by what P55x measured. CI additionally runs the census cross-check against the kernel package's shape census (grouped-nf4-gemm#353) and fails if the CI scripts shared with grouped-nf4-gemm stop being byte-identical to its `main`. No action is needed if you are on 0.37.0; the `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`.

### The CI scripts both repositories carry start to become one file (repository tooling; nothing in the wheel changes)

- **`scripts/check_shared_tooling.py`** (new, byte-identical in grouped-nf4-gemm): `SHARED` lists the scripts that are one
  file in both repositories. Nine of the thirteen same-named scripts had forked, so one check name enforced two rules.
  With `--sibling` it fails on any differing byte, a shared file missing there, a self-comparison or a sibling outside the
  system, and lists the scripts still forked as NOTEs. grouped-nf4-gemm is upstream for shared tooling (kernel-first, as
  for the manifest); a new `ci.yml` step clones its `main` and runs `--sibling`. `tests/test_check_shared_tooling.py`
  produces every failure the check claims and asserts it is detected.
- **`scripts/check_discovery_contract.py`** adopts grouped-nf4-gemm's copy, which adds `page_kind: orientation` records
  ranked over their own corpus. This repository has none, so every query keeps its kind and corpus: the 45 rankings are
  identical before and after (37/45 top-1 against the floor of 30), compared line by line.

### Docs: the same-box vLLM comparison is P58's everywhere, and the cross-host pack limitation is P55x's

- The README ("Do not use this when" and the results table), `serve-moe-on-consumer-gpu`'s limitation in
  `docs/capabilities.json` and `docs/solutions/serve-large-moe-on-a-consumer-gpu.md` still led with lane p37's 2026-09-05
  comparison against vLLM 0.28.0 (2.52× / 4.06× over the NF4 control). P58 re-measured current against current on
  2026-09-22 and re-pointed `docs/STATUS.md` and `docs/SERVING-THROUGHPUT.md`, but not these four. They now quote P58
  (vLLM 0.30.0 / e4b's current int4 stack 1.087 at B=1, 1.396 at B=16) and keep p37 as history. No register check could
  catch the omission: p37's rows are still `measured`, correctly, because they are history rather than retracted.
- `docs/STATUS.md` cited `e4b.serve.buildout.bo6.qwen3.calibexp-streamed-*`, a glob that breaks mid-segment; the
  consumer site's claim-reference check reads it as the id `…calibexp-streamed-`, which is not in the register, and
  refused the 0.37.0 re-pin on it. The three rows it stood for are now named.
- The same capability said the streamed 64k calibration "does not reproduce its licence across hosts". P55x found the
  pack byte-reproducible on one box and across boxes and a release boundary, and P37's divergent pack an outlier; the
  limitation now says that, and what stays open (the calibrated attention half is not in the fingerprint, #674).
### CI runs the census cross-check (grouped-nf4-gemm#353)

- `bench/support/census_cross_check.py --check` (#522) fails when a claimed family that the kernel's shape census covered
  stops being covered, but nothing ran it. It now runs in `lint-and-test` against `census/shape_census.json` read from the
  exact kernel commit pip installed (`direct_url.json`), so it is never a second pin. Calibrated: a census with OLMoE's
  down-proj shape removed exits 1 (`census coverage REGRESSED for: olmoe`); the v0.33.0 census exits 0 (4 of 11 probed
  claimed families covered). `--check` writes a missing baseline and passes, so the step asserts the baseline exists first.
### `check_capabilities.py` joins the shared set; the two discovery corpora stop answering one question two ways

- **`scripts/check_capabilities.py` is one file in both repositories** (added to `SHARED`). grouped-nf4-gemm's copy was a
  strict subset of this one. Adopting this copy there as it stood would have run cleanly, but only because the
  serving-position rule's id pattern hard-coded `e4b.` and so could never match a kernel id: deriving the prefix instead
  made it fire falsely in the kernel repository, whose serving capabilities are separate kernels, each on its own lane
  (two false warnings measured). The rule is now gated on the repository's role in `docs/system-manifest.json`, and the
  id namespace is the register's own. Behaviour here is unchanged (the same single WARN on `main` before and after); new
  tests pin the namespace derivation, the mixed-register refusal and the role gate.
- **One routing question, one answer per phrasing.** Both repositories' `docs/discovery-queries.json` carried "Where does
  an NVMe primitive belong versus model-level NVMe integration?", each routed to its own page, so the consumer site's
  merged corpus recorded it as unmapped with a WARN ("to be settled upstream"). The kernel repository keeps that
  phrasing (the primitive's side); this repository now asks "Where does model-level NVMe integration live versus the
  NVMe primitives?" (the integration's side), which still ranks its page first.

## 0.37.0 — 2026-09-23 — five more families admitted (`nemotron_h`, `granitemoehybrid`, `granitemoeshared`, `jamba`, `lfm2_moe`) and the five still staged say why, as tests; fused q/k/v licensed on the int4 serving lanes at B=1 and B=16 (P54, P59); a licensed int4 expert pack again, as bytes (P55x); the same-box vLLM comparator re-run on 0.30.0 (P58); two expert-GEMV levers built and refused (K17, K18), and the B=16 expert GEMV's remaining headroom bounded with no lever to follow (P60, P61)

**0.37.0.** The loader admits five more MoE families. `nemotron_h`, `granitemoehybrid`, `granitemoeshared` and
`lfm2_moe` each load a real published checkpoint (`reference-ok`); `jamba` is so far exercised only on a toy
checkpoint (`toy-ok`). The five families still refused (`axk1`, `dbrx`, `jetmoe`, `qwen3_vl_moe`,
`qwen3_vl_moe_text`) now each name a measured blocker, asserted by a test that fails the day the blocker lifts. On the
int4 serving stack, fusing q/k/v is licensed at B=1 (12.4 % per step, token-identical) and at B=16 (0.0044 nats/token).
A calibrated int4 expert pack for Qwen3-30B-A3B is licensed again, as retained bytes loaded back by fingerprint. A load
that faults on CUDA can now be re-run with `E4B_LOAD_SYNC_DEBUG=1`, which synchronises after each load stage so the
failure is reported at the stage that caused it, and a failed shard read now prints the host's memory facts. Everything
else is measurement: the same-box comparison with vLLM 0.30.0 was re-run (vLLM decodes 1.09× faster at B=1 and 1.40× at
B=16). Two expert-GEMV levers were built and refused (K17's fused reduce and K18's grouped GEMV), and the B=16
expert GEMV's remaining headroom was measured and has no lever to follow (P60, P61). Affected: loading of the five newly admitted families
(CPU and CUDA) and the int4 serving lanes on NVIDIA GPUs. Nothing changes for NF4 training or for any package default, and families
admitted before 0.37.0 load as they did (the new prefix-rename pass is a no-op for them, asserted by test; the only
difference they can see is the diagnosis a failed shard read now prints). Upgrade if you load one of the five families or want the
load-fault diagnosis; no action otherwise. The `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`; CI now runs against
grouped-nf4-gemm 0.33.0.

### Loader and engine changes not filed under a lane below

- **`E4B_LOAD_SYNC_DEBUG=1` — staged synchronisation for CUDA load faults** (#660, #344). CUDA reports asynchronously, so
  the traceback of a fault raised during a streamed load need not name the kernel or the stage that caused it. With the
  flag set, the loader synchronises at each stage boundary and logs `[sync] <stage>: clean`, so the first stage that
  does not come back clean is the one that faulted. It also logs the host's `MemTotal` / `MemAvailable` / cgroup limit
  and the largest shard it will map. The flag is ignored with a log line on a non-CUDA device, and off by default
  because the synchronisations serialise the load. Separately, and always on, a shard read that raises now logs a
  diagnosis (shard, size, device, host memory facts) before re-raising unchanged. Nothing branches on the memory
  numbers: they describe a host, they do not refuse one. `tests/test_load_sync_debug.py`. #344 stays open; the
  Gemma-4 load fault it tracks is bounded to a stage by this, not fixed.
- **`E4B_BATCHED_PAD_WASTE_LIMIT`** (#662) overrides the batched engine's 4× pad-waste fallback guard (default
  unchanged). The guard only protects speed, since falling back and batching compute the same function. A parity arm
  wants the batched arithmetic on every call, and a run that falls back is measuring the reference against itself.
  `batched_fallback_stats` reports the limit in force, so a receipt says which one it ran.

### P56 read: Gemma-4's training-parity FAIL is not the fused path's, and the band has no floor term (#662, #671, #672, #675)

- **The kernel-free batched path fails the same 0.05 band** as the fused path on the same box, init and tokens:
  0.05425 final / 0.08487 median against the fused path's 0.10223 / 0.10639. It never touches grouped-nf4-gemm and has
  13× less composed gradient error, yet it closes only 1.9× of the gap, so no achievable change to the arithmetic
  reaches the band. The CPU half (#662) found e4b's expert composition exact and family-blind: every family, Gemma-4
  included, sits at the same bf16 floor against an fp32 arm over identical quantized weights.
- `e4b.parity.gemma4.train-internal` is superseded by `e4b.parity.gemma4.train-floor`, the first measurement of this
  family's training-parity floor (≥ 0.054 final / 0.085 median). The superseded row also mislabelled its unit: it said
  "held-out" while the number was `loss_last`, the final TRAIN loss. **#558 stays open as a gate defect**:
  `tp4_reduce.parity()` compares against 0.05 and against zero, with no floor term. Refusing the fused path for
  `gemma4_text` is not supported by this read.
- Two harness defects, both fixed. An unquoted expansion in the arm's assignment prefix made the next `VAR=value` the
  command, so all four arms of draw 1 died with rc 127 after measuring nothing ($0.29). They are now routed through
  `env`, and a regression test executes the real prefix from the real file (#672). The pre-registration's micro-batch
  was misstated as 1: the run used the field recipe's micro-batch 2, so the draw matches the standing claim row's
  fixture, and the erratum is filed rather than edited (#671).

### Lane tooling

- `p47_drive.sh` / `p54_drive.sh` gain the heartbeat, stall verdict and lane-dead exit (25) from `tp4_drive.sh` (#655,
  fixes #641). `bench/p319/` and `bench/p319b/` carry the pre-registrations that three committed receipts already
  cited (#669).

### llms-full.txt headroom: 399,971 → 379,460 bytes against the 400,000 cap (docs only, no number or status changed)

- `docs/STATUS.md` (−11.9 KB): serving-history paragraphs (p37, bo3, bo5, bo6) now give the verdict and point at their
  results files. Training paragraphs (tp1, p38, tp2) and restated numbers elsewhere cite the ACTIVE claim rows that the
  bundle already projects. Numbers with no projected claim row (Gemma-4's P47–P52 store map, P56's first readings) stay.
- `docs/claims.json` (−8.6 KB projected): 128 ACTIVE `claim` sentences shortened by wording. Hashes, revision pins,
  library versions and verbatim log quotes move to the same row's `notes`, and the moved text is appended verbatim. Every
  measured number, id, status and value is unchanged, checked mechanically against `main`.

### P61 read: the B=16 expert GEMV's row work and expert bytes overlap -- no lever lane (lanes `p61-5090-1`..`-6`, $0.28)

- On a 575 W RTX 5090 (`bench/p61/RESULTS-p61.md`), the served int4-b32 expert GEMV reproduces P60's replay: 6.548 vs 6.479
  ms/step. On per-layer weight stores the recorded B=16 routing's repeated rows cost **0.905 ms/step** in total.
- The pre-registered additive model (fixed + per-row + per-distinct-expert, fit to a rows x experts grid) predicts 2.42, so P1
  is refuted. The grid shows why: at fixed rows, time is flat in distinct experts up to ~32, then climbs with them. Row work
  and expert bytes overlap rather than add. By the registered rule, no lever lane follows; nothing changes a default.
- Host lesson: run 1's 5090 was power-capped at **450 W** and read the served arm 12 % slow while the light arm matched.
  Amendment 1 refuses cards below 575 W; adertha-agents #128 lets that refusal (exit 17) exclude its machine. Row
  `e4b.serve.p61.qwen3.b16.expert-gemv-cost-split.5090.2026-09-23`.

### K18 read (grouped-nf4-gemm): a grouped expert GEMV is exact and slower — P60's 0.92 ms is not reachable by sharing loads; its re-streaming reading is withdrawn (lane `k18-5090-1`, $0.13)

- **grouped-nf4-gemm lane K18** built the kernel P60 licensed: an int4-b32 split-K GEMV that loads each expert's slice
  once for up to four of its rows. It was run from `bench/k18/` (#687) against P60's recorded ids on an RTX 5090.
  - It is **bitwise the served GEMV** (0 of 256 replay checks differ) and **1.49× slower**: 9.689 against 6.520 ms/step.
    The dedup arm reads 5.564, reproducing P60 on a second host.
  - At R = 8/16 it is 1.00–1.65× the served call, worst where nothing can be shared.
  - Decision: not a lever. It stays dormant in grouped-nf4-gemm, and nothing here routes to it (row
    `gnf4.kernel.k18-grouped-expert-gemv.5090.2026-09-22`).
- **Correction to the P60 entry below.** The 0.92 ms/step dedup gap was read as "the cost of re-streaming each expert per
  row". The dedup arm also drops the repeated rows' arithmetic and programs, locality was already ruled out (P60's P3),
  and sharing the loads did not recover it. That reading is withdrawn in `bench/p60/RESULTS-p60.md` (dated correction),
  `docs/STATUS.md` and the P60 claims row. The number stands.

### P60 read: the expert GEMV's B=16 headroom is repeated rows — 0.92 ms/step, the ceiling a grouped kernel can recover (lane `p60-5090-1`, $0.23)

- **Recorded B=16 routing replayed through the shipped int4-b32 expert GEMV on one RTX 5090** (`bench/p60/RESULTS-p60.md`):
  the replay reproduces the served kernel row (**6.155 ms/step vs the census's 6.340**, −2.9 %), and one row per distinct expert
  instead of one per routed row runs **5.560 vs 6.479 ms/step** — **0.92 ms/step** (~8 % of the B=16
  step) is the cost of the repeated rows [first read as re-streaming each expert per row — withdrawn by the K18 read above]. Ordering rows by expert changes nothing (-0.55%): L2
  already serves the repeats. One row per expert sits 1.21× above the 1,512 GB/s byte floor (between bands).
- Decision: a **grouped expert GEMV** (grouped-nf4-gemm lane K18) is licensed to build, read against the recorded ids committed in
  `bench/p60/receipts/eids_b16.int16.bin`. Row `e4b.serve.p60.qwen3.b16.expert-gemv-repeat-cost.5090.2026-09-22`. No default changes.

### P59 amendment 1: the fused q/k/v path is the default at B=16 too — 0.0044 nats/token through a 128-row prefill, determinism control bit-identical (lane `p59b-5090-1`)

- **Through the harness's 128-row prefill, `KL(int4 unfused ‖ int4 fused q/k/v)` at B=16 = 0.0044 nats/token, top-1 0.9775**
  over 2,048 teacher-forced decode positions (one RTX 5090, $0.18; `bench/p59/RESULTS-p59.md`); the unfused stack rebuilt in a
  fresh process is **bit-identical** (KL exactly 0), so the number is the fusion's alone. The distance from the NF4 anchor moves
  by +0.0034 (band ± 0.005). All four registered conditions hold → **`--fuse-qkv` is the default on the int4 serving lanes at
  B=16 as well as B=1**; P54's fused B=16 row (0.223 ms/step, 1401 → 1429 tok/s on its box) is `licensed_by` this read.
- The prefill-last control (P4) is refuted as the amendment anticipated: the >16-row cuBLAS path on the cached bf16 weight is
  where fusing changes bits; the K16 decode path is bitwise invariant (run 1). Register rows
  `e4b.serve.p59b.qwen3.b16.{fqkv-kl,nf4-int4-kl,nf4-fqkv-kl}.5090.2026-09-22`.

### P59 read: fusing q/k/v changes nothing at 16-row decode (KL exactly 0) — the K16 kernel is bitwise invariant to it; P57's kernel attribution withdrawn (#682, lane `p59-5090-1`)

- **KL(int4 unfused ‖ int4 fused q/k/v) at B=16 = 0.000000 nats/token, top-1 1.0000, bit-identical on every one of 2,048
  teacher-forced decode positions** (one RTX 5090, $0.34; `bench/p59/RESULTS-p59.md`). The fusion was installed (48 modules,
  `Int4Linear` 192 → 96). A free A2000 probe (`bench/p59/probe/`) explains it: the K16 small-M GEMM is **bitwise invariant**
  to fusing at 2–16 rows (the same plan for every N, independent columns); only the >16-row cuBLAS path on the cached bf16
  weight differs (≤ 1 bf16 ulp). The anchors: the int4 stack sits 0.0717 nats/token (top-1 0.905) from the NF4 control.
- **P57's reading that P54's B=16 token divergence "is the K16 GEMM's" is withdrawn** (P57's RESULTS, STATUS and the P57 entry
  below corrected; also corrected: P57's first-divergence indices are not P54's). The leading hypothesis is now the harness's
  128-token prefill steps, which take the cuBLAS path.
- The determinism arm was skipped by the deadline guard, so the registered decision rule is not met and `--fuse-qkv` stays
  opt-in at B=16 on run 1 (amendment 1 flipped it — entry above). **Amendment 1** re-runs the gate with a 128-row chunked prefill (`kl_b16.py --prefill-chunk 8`),
  a K16-route census, and room for the determinism arm. Register rows `e4b.serve.p59.qwen3.b16.{fqkv-kl,nf4-int4-kl,nf4-fqkv-kl}.5090.2026-09-22`.

### P58 read: same box, current vs current — vLLM 0.30.0 decodes 1.09× (B=1) / 1.40× (B=16) faster than e4b's int4 stack; 0.29.0 vs 0.30.0 within 1 % at B=16 (#676 lane, run `p58-5090-1`)

- **The register's current-vs-current comparator against a production engine is re-pointed** (`bench/p58/RESULTS-p58.md`,
  one RTX 5090 on an EPYC 9655 host, identical prompt token ids for both engines, $0.31): e4b's current int4 stack (RTN
  int4 experts + uncalibrated int4 attention + K16 route, fused q/k/v at B=1) at **239.4 tok/s (B=1) / 1379.2 tok/s
  (B=16)** vs vLLM 0.30.0 serving Qwen's GPTQ-Int4 via Marlin at **260.3 / 1925.6** → ratios **1.087 / 1.396**, vLLM
  ahead, inside the pre-registered bands (P1 1.02–1.20, P2 1.35–1.75). vLLM 0.29.0: 1912.7 at B=16 (ratio 1.387;
  build-to-build 1.007, P3 holds); at B=1 its two engine starts read 280.4 and 259.9 (self-pair 1.079 > 1.03 →
  **DRIFT, no ratio quoted**) while 0.30.0's two starts agreed to 0.04 % — a vLLM B=1 engine-start variance on this
  host, recorded not explained. Same-box e4b int4/NF4: ×2.356 (B=1), ×2.851 (B=16). P37's 2026-09-05 rows stay as
  history; the engine advantage is understated (vLLM's number includes its serving loop). Quality quoted, never equated.
- **B=1 is host-bound across three 5090 hosts today**: the same fused-q/k/v int4 stack read 4.24 (P57, EPYC), 4.18
  (P58, EPYC 9655) and 3.69 ms/step (P54's box, 2026-09-21); STOP-1 informational, same-box ratios only.
- Register rows `e4b.serve.h2h.vllm-0.30.0.p58.qwen3.{b1,b16}.5090.2026-09-22`, `…vllm-0.29.0.p58.qwen3.b16…`, the
  build-to-build row and 20 per-arm rows (`bench/p58/p58_register_rows.py`); `docs/SERVING-THROUGHPUT.md` and
  `docs/STATUS.md` re-pointed. Receipts in `bench/p58/receipts/` (trimmed engine logs; full run private).

### P57 read: K17's fused split-K reduce is exact and SLOWER in the consumer at both batches; P54's B=16 divergence is not the glue's (#666, lane p57-5090-2)

- **`GNF4_GEMV_FUSED_REDUCE=1` costs 0.038 ms/step at B=1 and 0.217 ms/step at B=16** on the int4 serving stack
  (Qwen3-30B-A3B, one RTX 5090, A/A spreads ≤ 0.009 ms; `bench/p57/RESULTS-p57.md`). The route engaged exactly as
  asked (`_reduce_partials` gone from both fused censuses) and the tokens are identical, but `_gemv_int4_b32`'s own
  duration grows by more than the reduce it absorbs (1.415 → 1.642 ms at B=1; +7.0 % at R=128 rows, B=16): the
  separate reduce launches were overlapped in the graph, the fused epilogue sits on the critical path. K17's kernel
  stays opt-in in grouped-nf4-gemm; nothing in e4b changes. P1 and P2 refuted as registered.
- **With the round-2 glue forced OFF on both legs, fused q/k/v at B=16 still diverges from unfused on 15 of 16
  sequences** while the control's two draws are bit-identical: the glue path is exonerated (P3). *(Corrected by P59:
  the indices are not P54's, and the divergence is not the K16 GEMM's — see the P59 entry above.)*
  `--fuse-qkv` stays the default at B=1 and opt-in at B=16 until a KL/K8 read bounds the difference *(it did: P59
  amendment 1 above licenses it at B=16)*.
- **The distinct-expert count is READ (amendment 3, run `p57d-5090-1`, $0.07): 58.7 distinct experts per layer per
  decode step at B=16** (layer means 50.9–71.7, 73 steps counted on device; uniform expectation 82.4;
  7–41 of 128 experts per layer never touched). The first two attempts counted 3 warm-up steps because the
  counter's `torch.unique` synchronised under CUDA-graph capture (amendments 1–2 in `P57-PREREG.md`); v2 of
  `bench/p57/distinct_experts.py` accumulates on device and was verified under capture+replay on an A2000. Against
  #564's expert-tier byte roofline the floor at 58.7 is 4.89 ms/step vs the measured 6.34 ms `_gemv_int4_b32`
  row: the expert GEMV runs at ~77 % of roofline at its real routing. Row
  `e4b.serve.p57.qwen3.b16.distinct-experts.5090.2026-09-22`.
- Register rows `e4b.serve.p57.qwen3.{b1,b16}.{control,fr}.5090.2026-09-22` and
  `e4b.serve.p57.qwen3.b16.{nor2_control,nor2_fqkv}.5090.2026-09-22` (measured, same-box). Receipts in
  `bench/p57/receipts/`; `bench/p57/p57_register_rows.py`. Cost $0.09 (a dud-box bake failure, `p57-5090-1`) + $0.22.

### There is a licensed pack again, and its bytes exist (#658, #405)

Lane P55x built Qwen3-30B-A3B's streamed 64k calibrated int4 expert pack, dumped it as a hash-pinned
artifact, and ran the registered two-text K8 gate **on those bytes loaded back by fingerprint** — the
path a loader takes, not the live stores they were built from. It **passes both texts**: wikitext
−0.05275 ppl, c4val1 −0.06622, against an NF4 reference on that box bit-identical to bo6c's on both.

`pack_fingerprint sha256:0c9955a9f06d8326…` is in the register. The bytes are retained and were verified
after transfer by two independent implementations on two machines. A loader given that fingerprint
refuses anything else and never rebuilds from the recipe, so **the licence travels as bytes** — which is
what [#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405) had been open on since
2026-09-06, when the machinery existed but no pack had been built, gated and kept.

**Three findings beyond the licence.**

- **The recipe reproduces across boxes, and #405's framing was drawn from an outlier.** This pack is
  byte-identical to the one lane P39 built on 2026-09-10 on a different rented 5090 under **e4b 0.35.3**,
  where this ran under **0.36.4** — the same 64 hex characters across a box and a release boundary. With
  P39's own box-1/box-2 agreement and this lane's two same-box builds, four builds across at least three
  boxes agree on every byte. P37's 11522/766 divergence is the outlier, not the rule, and the open
  question becomes what was different about that host. The remedy is unchanged: a licence still travels
  as bytes, because nothing here predicts which box is the next P37.
- **The half that cannot be pinned is the half carrying the quality.** `engines/int4_attn_calib.py` has
  no serialisation of any kind, so the 192 calibrated attention projections are re-derived on every load.
  The same pinned expert bytes with RTN attention instead **fail** c4val1 at +0.13237 — a +0.19858 swing
  and a budget failure. So `pack_fingerprint` names the experts, the unpinnable component is load-bearing,
  and that residual is now measured rather than asserted.
- **The recipe is byte-deterministic on one box.** Two builds in one session produced the identical
  fingerprint, method-map hash and row-count-vector hash. bo6c's determinism row compared mean_nll and
  counts for the 16k arm a day before fingerprints existed; this is the 64k recipe at byte level.

bo6c's verdict row stays **active and unsuperseded**, deliberately: three bo7 census rows take their
licence from it, and repointing them at this verdict would assert their pack is this pack — unverifiable,
since neither bo6c's nor bo7's bytes were kept. What the new row replaces is bo6c's *role*, not its
measurement. No gate, threshold, `min_rows`, damping or K8 budget moved.

### The five families still staged now say WHY, and each reason is a test (#648, #509)

`STAGED_NOT_WIRED` recorded which families no loader admits. For four of them it recorded no reason
at all — *"none was found; that absence is itself the thing to resolve."* Resolved: two of those four
(`jamba`, `lfm2_moe`) simply had not been tried and are now wired; the rest have a measured blocker.

`tests/test_staged_blockers.py` asserts each blocker as the CURRENT upstream or checkpoint fact, so it
**fails the day the blocker lifts**. A blocker kept as a comment rots into a stale excuse; one kept as
a test says when the family became wirable. Nothing here claims a family should stay unwired — only
what would have to change first.

- **`qwen3_vl_moe` / `qwen3_vl_moe_text` — transformers publishes no `ForCausalLM` class.** Only
  `Qwen3VLMoeForConditionalGeneration` exists and neither config is in `MODEL_FOR_CAUSAL_LM_MAPPING`,
  so admission would clear the architecture gate and then raise while BUILDING the tree, before a
  weight is read. This is also why #637/#639 could adjudicate the expert layout while the loader still
  refuses the family: the int4 planner builds its own tree and does not need a CausalLM class.
- **`jetmoe` — its tree has no `experts` submodule to replace.** The stacks sit directly on the MoE
  block as `mlp.input_linear` [E, 2I, H] / `mlp.output_linear`. That is granitemoe's *shape*, but
  granitemoe's tree declares `block_sparse_moe.experts.gate_up_proj`, so a checkpoint-side rename is
  enough there and cannot be here, where both sides say `input_linear`. The convention's `fused_prefix`
  names a path that does not exist — left as-is rather than changed to another wrong value. It is also
  a DUAL MoE (`self_attention.experts`), so a naive admission would quantize the MLP experts, leave the
  attention experts in bf16, and report success.
- **`dbrx` — flat 2-D stacks.** Each projection is one `[E * ffn_hidden, hidden]` tensor and the module
  declares it the same way; `Experts4bit.from_float` requires `[E, out, in]` and refuses anything not
  3-D. Reshaping is a real change with a real orientation decision — which is why the convention pins
  w1=gate / v1=up / w2=down in advance.
- **`axk1` — the keymap does not FIT the registry that would hold it.** #509 said admission and the
  rewriter must land together; measured now, they cannot yet. `CKPT_KEY_REWRITERS` holds callables the
  loader applies one key at a time (`rewrite(k) -> str | None`), while `rewrite_axk1_keys` takes
  `(checkpoint_keys, first_k_dense_replace)` and returns `(kept, dropped)` — because the
  `post_mlp_layernorm` rename is layer-CONDITIONAL: the released checkpoint ships that key on the dense
  layer 0 *and* on the MoE layers, and it must be dropped on one and renamed on the other. So wiring is
  "close over `first_k_dense_replace` and adapt the list-shaped keymap to the per-key contract", not an
  admission row plus a dict entry. Separately, the only released checkpoint is 1.04 TB, so no support
  row is obtainable here and an admission would be a claim the coverage gate has no evidence for.
- A final test requires every member of `STAGED_NOT_WIRED` to be named by a blocker in that file, so
  "no reason recorded" cannot reopen for a family added later.

### `jamba` and `lfm2_moe` are wired — the "no reason recorded" was that nobody had tried (#648)

`STAGED_NOT_WIRED` said of these two only that there was *"no reason recorded here because none was
found; that absence is itself the thing to resolve."* Resolved by trying them: both load unchanged.

- **Both are hybrid towers** — Jamba is Mamba/attention, LFM2-MoE uses short convolutions — and that
  was the stated caution for keeping them out (`READ_COMPATIBLE_CONVENTIONS`: *"hybrid Mamba towers
  whose NON-expert surface this loader has never placed"*). The caution was reasonable and the answer
  is that the non-expert surface is ordinary passthrough: it goes through `_assign` like any other
  dense tensor, no meta tensors remain, and the forward is finite.
- **Admitted through `SUPPORTED_ARCHITECTURES`, not by adding their conventions to
  `READ_COMPATIBLE_CONVENTIONS`.** Membership there would have carried them in on their STORAGE
  convention, and storage was never the open question — so each came in on its own row instead.
- **A dense layer must never be read as an expert, and that is now pinned.** Both families alternate
  MoE and dense layers, and the dense MLP sits in the SAME container under the SAME projection names
  (`feed_forward.{gate,up,down}_proj` / `feed_forward.w{1,2,3}`), differing only by the absent
  `experts.{e}.`. Matching one would build a one-expert stack for a layer the router never routes
  through. `expert_re` requires the index; `test_a_hybrids_dense_layer_is_never_read_as_an_expert`
  asserts it against the released key spellings.
- **Admitting `lfm2_moe` does not reopen the #648 activation hazard**: its config declares none of
  `_ACTIVATION_FIELDS`, so #650's rule would REFUSE it — it is carried by
  `_ACTIVATION_UNDECLARED_OK` with its evidence (upstream `Lfm2Moe` applies SiLU unconditionally),
  and a test pins that.
- **Evidence**, CPU/bf16, transformers 5.17.0 / torch 2.14.0 / bitsandbytes 0.50.2:
  `bench/support/rows/lfm2_moe.json` — `LiquidAI/LFM2-8B-A1B` (4.5 B params, 32 experts on 22 of 24
  layers): load 13.7 s, **22 quantized / 0 unquantized** (nf4), forward finite → **`reference-ok`**.
  `bench/support/rows/jamba.json` — `ai21labs/Jamba-tiny-dev` (222 M params, 8 experts on 8 of 16
  layers): load 0.9 s, **8 quantized / 0 unquantized** (nf4), forward finite → **`toy-ok`**, and the
  row says so: every larger published jamba MoE is 52 B+, gated, or (the 3 B "Jamba2"/"Reasoning"
  releases) ships `num_experts: 1` and is not MoE at all.
- `STAGED_NOT_WIRED` drops both; the derived counts go to 5 across 4.

### GraniteMoe's two aliases are wired — it was an omission, and the tree now says so (#648)

`granitemoehybrid` and `granitemoeshared` had a convention that claimed them, while plain `granitemoe`
was admitted and they were not. This confirms that was an omission rather than a decision, and closes it
with real-checkpoint rows for both.

- **The expert surface is the same, checked rather than assumed.** Upstream's `conversion_mapping` entry
  is the SAME three `WeightRenaming`s for all three model_types, compared entry by entry against
  transformers (`test_the_granite_aliases_share_granitemoes_converter_exactly`), so one rename tuple
  legitimately serves all three and the test fails if upstream ever splits them. They differ only
  OUTSIDE the experts: `granitemoeshared` adds a per-layer dense `shared_mlp`, `granitemoehybrid`
  additionally replaces most attention layers with Mamba — both pure passthrough here.
- **Admission alone would have loaded nothing.** `LEGACY_KEY_RENAMES` is keyed on **model_type**, so
  without an entry each, `block_sparse_moe.input_linear` never becomes `experts.gate_up_proj`, every
  layer looks dense, and the load ends in the zero-expert-stacks guard. Admission and the rename land
  together, and `test_an_admitted_prefused_family_cannot_be_missing_its_legacy_renames` now asserts
  that mechanically for any family admitted on this convention. Same shape as #509's axk1 point.
- **Evidence — both `reference-ok`**, CPU, bf16, transformers 5.17.0 / torch 2.14.0 / bitsandbytes 0.50.2:
  `bench/support/rows/granitemoehybrid.json` — `ibm-granite/granite-4.0-h-tiny` (4.0 B params, 64 experts,
  40 MoE layers, Mamba/attention hybrid): load ok 23.0 s, **40 quantized / 0 unquantized** (nf4), forward
  finite. `bench/support/rows/granitemoeshared.json` — `ibm-research/moe-7b-1b-active-shared-experts`
  (3.5 B params, 62 experts, 40 MoE layers): load ok 23.7 s, **40 quantized / 0 unquantized** (nf4),
  forward finite.
- `STAGED_NOT_WIRED` drops both; the registry's derived counts go to 7 across 6.

### `nemotron_h` is wired: the loader admits it, and its renames now reach the expert path (#648, #509)

The wire-or-remove decision #648 exists for, taken for the first family, with a real published
checkpoint behind it rather than a fixture.

- **Admitted.** `SUPPORTED_ARCHITECTURES` carries `nemotron_h -> mixer.experts`. `has_gate` comes
  from the convention, so the read stacks `up_proj` alone (`down(act(up(x)))`, no SwiGLU gate) instead
  of fusing a gate that does not exist, and #650's `_expert_activation_name` resolves the activation
  from the family's own `mlp_hidden_act` (`relu2`) rather than the old `"silu"` default.
- **Admission alone loaded ZERO experts, and that is the substance of this change.** Nemotron-H ships
  every tensor under `backbone.` while the tree declares `model.`. The loader anchors expert indexing
  on `^model\.layers\.` and builds every fused-target lookup from `model.layers.{i}.{expert_rel}.`, so
  with the family merely admitted `_index_per_expert_keys` returned `{}` against the real checkpoint:
  every MoE layer read as dense and the load died in the zero-expert-stacks guard — while its
  NON-expert keys mapped fine, because the `_assign` pass applies `conv.rename` and the expert path
  never did. Renames meant two different things depending on which branch a key took. That is exactly
  the defect #643/#644 fixed in the PLANNER; this is the same fix on the loader side.
  `loader._rename_ckpt_prefixes` applies a convention's renames to the part of a key BEFORE
  `layers.N.`, which is the planner's own narrow rule — a blanket rename of the whole key would rewrite
  the CONTAINER too, and mixtral's `.block_sparse_moe.` -> `.mlp.` would then destroy the substring
  `MIXTRAL.expert_re` matches on, un-recognising every mixtral expert key. Asserted to be a no-op for
  all twelve families admitted before it.
- **Evidence — two rows, both on real published checkpoints.**
  `bench/support/rows/nemotron_h.json` is `inference-optimization/NemotronH-0.3B-A0.3B` (32 routed
  experts, 2 of 5 layers MoE): load ok, `verify_moe_4bit(strict=True)` 2 quantized / 0 unquantized
  (nf4), forward finite — graded **`toy-ok`**, because at 323 M parameters it is below the probe's 1 B
  reference bar and the row says so rather than overstating it.
  `bench/support/rows/nemotron_h_30b.json` is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
  (128 routed experts, 23 of 52 layers MoE, 14 shards): integrity clean (every shard the length its own
  header declares), load ok 155.2 s, **23 quantized / 0 unquantized** (nf4), forward finite —
  **`reference-ok`**, which is the grade `coverage-baseline.json` records. Both CPU, bf16, transformers
  5.17.0 / torch 2.14.0 / bitsandbytes 0.50.2. The small row is kept rather than replaced: it is the
  one whose two-MoE-layer shape the prefix-rename regression test is written against.
- `STAGED_NOT_WIRED` drops `nemotron_h` in the same change — `tests/test_staged_not_wired.py` fails in
  both directions, so wiring without delisting could not have merged. The registry's own count is now
  derived from the set and asserted (`test_the_stated_counts_match_the_set`): it read "ten model_types
  across seven conventions" when the conventions were **eight** — wrong the day it was written, and
  invisible because nothing read it. It is 9 across 7 now, checked.

### Measured (no default changes)

- **Lane P54 — fusing q/k/v on the int4 attention store** (`bench/p54/RESULTS-p54.md`, receipts in
  `bench/p54/receipts/`, one RTX 5090, $0.47). `Int4Linear.fuse` (0.36.4, #651) measured on Qwen3-30B-A3B,
  two interleaved draws per arm: **B=1 4.210 → 3.693 ms/step (0.516 ms, 12.4 %; 237.5 → 270.8 tok/s) with
  token-IDENTICAL output**, and B=16 11.420 → 11.197 ms (0.223 ms, 2.0 %) where the fused arm's tokens
  **diverge** from the control's on 14 of 16 sequences, reproducibly, against a bit-identical A/A. The K16
  small-M GEMM's accumulation order is a function of N, so the fused N=5120 launch rounds differently from the
  three it replaces *(explanation withdrawn by the P59 read above: the K16 decode path is bitwise invariant to
  fusing; the >16-row cuBLAS prefill path is where the bits change)*. **`--fuse-qkv` stays opt-in at B=16 and no B=16
  position is quoted** *(superseded by P59 amendment 1 above: licensed at B=16)* until a
  KL-from-checkpoint or K8 read bounds the divergence (bar: ≤ 0.10 nats, top-1 ≥ 0.93); B=1 is a licensed
  lever. Four register rows `e4b.serve.p54.qwen3.*.5090.2026-09-21`. The lane's distinct-expert arm (#564's
  unmeasured number) was **unbuildable as registered** — `--series-out` requires `--amort on`, which the
  captured B>1 stage refuses — and is re-registered in the lane's amendment 1.

## 0.36.4 — 2026-09-21 — the loader's three arrows measured against upstream (fused layout, conditional transpose, rename reach) and an activation that no longer defaults; the int4 attention stack can fuse q/k/v (P54 measures it); P52 and P53 read on Gemma-4

### The loader: three arrows measured against upstream's own code, and each one found a defect

None of these came from a wrong number in the field. Each came from executing upstream's code beside e4b's on the same
tensors and reading the disagreement. Nothing here changes which families load.

- **The expert activation is read from the family's own field, or refused** (#650, #648). The lookup was
  `hidden_activation`, then `hidden_act`, then a `"silu"` DEFAULT. `nemotron_h` declares neither and names
  `mlp_hidden_act: relu2`; it is non-gated, so `down(act(up(x)))` with SiLU instead of ReLU² is a different function with
  every shape agreeing, and the existing guard (unknown activation NAME) could not see a field it never read.
  `_ACTIVATION_FIELDS` is the ordered list (adds `mlp_hidden_act`, `activation_function`); a config that declares none of
  them raises `MoEConventionError` unless the model_type is in `_ACTIVATION_UNDECLARED_OK` with its evidence
  (`qwen3_omni_moe`, `lfm2_moe`, `dbrx`). Every admitted family's transformers default config resolves under the new rule
  (39 of 39, asserted by test); the load log names the field the activation came from.
- **`STAGED_NOT_WIRED`** (`arch/moe_conventions.py`): the ten model_types that have a convention here and that no loader
  path admits — `axk1`, `nemotron_h`, `granitemoehybrid`, `granitemoeshared`, `qwen3_vl_moe`, `qwen3_vl_moe_text`,
  `jamba`, `lfm2_moe`, `jetmoe`, `dbrx` — asserted equal to the loader's own refusal in both directions, so wiring one
  without delisting it fails and adding a convention nothing admits without listing it fails too. (`axk2` has no
  `SUPPORTED_ARCHITECTURES` row either but aliases onto `qwen2_moe` and IS admitted.) The wire-or-remove decision on #648
  and #509 stays open.
- **The fused gate/up layout is verified against upstream's own expert forward** (#630, closes #515).
  `arch/fused_layout_probe.py`: a gated expert's output is affine in `up` and nonlinear in `gate`, so scaling one index
  set of the fused axis and taking the second difference finds the layout on CPU, with no checkpoint and no GPU, for any
  activation; no affine set or more than one raises `FusedLayoutUndetermined` rather than guessing. All 13 gated
  conventions measured: contiguous gate-first everywhere except `gpt_oss`, which is interleaved. Three pre-existing
  defects: (a) `gptoss` DECLARED contiguous gate-first and is interleaved — `fused_order` had two values for a layout space
  of at least three; new field `ckpt_gate_up_packing`, and gptoss declares it; (b) the #518 refusal in `expert_layout_for`
  was raised inside a `try` whose `except MoEConventionError` fell back to `SUPPORTED_ARCHITECTURES`, so it was INERT for
  every natively pre-fused family — the `try` now wraps only the lookup; (c) the exposed set is SEVEN (`NATIVELY_PREFUSED`:
  granitemoe, gptoss, qwen3_vl_moe, gemma4, jetmoe, qwen3_5_moe, axk1), not the six or nine counted before.
- **The pre-fused transpose is conditional, because upstream's is** (#639, closes #637). Upstream's converter is
  `Transpose(check_dims=True)`: transpose only if the checkpoint tensor and the module parameter differ. e4b's
  `transpose_last2` was unconditional, so on a SQUARE expert stack (`2 * moe_intermediate_size == hidden_size`) e4b loaded a
  transposed tensor upstream would not — same shape, different values, nothing raised. `make_plan_reader(param_shape=)`
  replicates the condition; `execute_moe_plan` supplies the shape from the model and the int4 serve lane from its meta
  twin; a square stack with no expected shape is REFUSED. `tests/test_converter_arrow.py` (43 arms) runs upstream's real
  `ConversionOps` beside e4b's read path over the same tensors for eight conventions.
- **A convention's renames reach the expert fused target too** (#644, closes #643). A passthrough key went through
  `conv.rename`; a per-expert key had its fused target built from the RAW checkpoint prefix, so a family shipping
  `backbone.layers.N.` where the tree declares `model.layers.N.` (nemotron_h) mapped the two branches inconsistently.

### Serving: the int4 attention stack can fuse q/k/v (lane P54 measures it)

- **`fuse_qkv` takes the int4 store** (#651; #652). Every int4 serving lane applies `enable_serve_attn_int4` at load and
  `fuse_qkv` after it, and `fuse_qkv` read `.weight`, which an `Int4Linear` lacks -- so the two were exclusive and every
  quoted int4 census (P42, K16 P5, bo7) ran `--no-fuse-qkv`, paying q, k and v as THREE attention launches per layer
  where the bf16 stack pays one (at M=16 `k_proj`/`v_proj` sit within 1 us of the 4.6 us launch floor). New
  `Int4Linear.fuse(mods)` / `Int4Linear.from_packed(...)`: the parts' packed rows and scales concatenated along N,
  byte-identical, so the fused projection computes the parts' function on one GEMV (rows == 1) or one K16 small-M GEMM
  (rows 2..16, N = 5120 on Qwen3-30B). A mix of int4 and dense q/k/v is refused, never half-fused. **No default
  moves**; `bench/p54/P54-PREREG.md` registers the measurement (B=16 saving 0.25-0.50 ms/step predicted, B=1
  0.20-0.45, K16 calls 192 -> 96, plus the distinct-expert count #564 names as the unmeasured number) before the lane runs.

### Measured this release (no default changes; nothing licensed)

- **P52 — the Gemma-4 graded store map's gate ran properly, on held-out prompts, and DID NOT PASS** (#621, #622, #623;
  `bench/p51/RESULTS-p51.md`). The K8 two-text gate 0.36.3 promised cannot be built on this family: Gemma-4's own NLL
  moves 0.4 nats with batch shape against K8's 0.05 budget (`e4b.parity.gemma4.no-reference`), and the K8 runner needs
  the arena path, which refuses a per-layer map. Replacement bar, registered on `main` four minutes before the run: KL from
  the bf16 checkpoint ≤ 0.10 nats AND top-1 ≥ 0.93. `bench/kl_prompts_heldout.py` holds 100 NEW prompts (same strata,
  disjointness from the committed 200 asserted — the first build had 6 overlaps). Result: graded map **0.1319 nats /
  top-1 0.874**, failing both axes on every stratum, while the bar's own provenance point (gpt-oss NF4) moved 2 % between
  prompt sets and Gemma moved 20 % — the instability is the family. The map stays a documented option; **no Gemma-4
  default ships and no position is quoted**.
- **P53 — calibration does not rescue Gemma-4's experts, and sequential is WORSE** (#638, #640, #645, #646, #649; closes
  #636; `bench/p53/RESULTS-p53.md`, one H100, $3.14). All 30 expert layers quantised: NF4 RTN **1.0772** nats KL vs the
  bf16 checkpoint, GPTQ int4 all-at-once 1.1050, GPTQ int4 sequential **1.1564** (top-1 0.644 / 0.642 / 0.630). Order was
  the only difference between the calibrated arms. Both axes on #636 — reduce the perturbation (P49) and make the
  downstream absorb it — are now closed; the one lever that works is keeping the early expert layers in high precision,
  and that lever is memory. Register row `e4b.quality.gemma4.calibration-refuted`.
- **The Gemma-4 TRAINING-path parity failure is in the register before #558 closed** (#635, `e4b.parity.gemma4.train-internal`):
  e4b's fused expert path against e4b's own dense per-expert reference, same box and tokens, ends **0.08257 nats** apart on
  held-out loss against the 0.05 band on the 0.32.1 kernel cut (0.09037 on the previous cut) — the failure survives the
  kernel change, so it is the family, not the kernel. The `fused` arm is not quoted for Gemma-4.

### Tests and bench harness

- The paged-attention end-to-end test's tolerance follows the compute mode that ran (fp8 default on sm_89+ gets the
  kernel package's own 1.5e-1; f32 keeps 2e-2) and its draw is seeded (#626, closes #341).
- p41 driver: a run whose registered failure criteria fired can no longer be published as a pass (#627, closes #495).
- tp4 HF arm selects expert parameters by STRUCTURE and refuses an empty selection (#628, closes #542).
- tp4 harness: the stall alarm tells a model fetch from a hang (#625, closes #624); the prologue names where its time went
  and its residual cannot round negative (#629, #633, closes #548); "any byte" is not a stall threshold and a dead lane is
  not a slow one (#634); amendments 5 and 6 register a parity-only box C and an axolotl arm with a proof-of-work predicate
  (#631, #632).
- `bench/p47/staged.sha256` is checked in CI rather than on the controller after a box is rented (#642); `kl_serve
  --family` has the P53 family's REFERENCE entry, found only after a rented box had fetched 52 GB (#647).

## 0.36.3 — 2026-09-19 — a per-layer expert STORE MAP; the field-recipe training position on the new kernel cut (Qwen3-30B-A3B 4.49x Unsloth, parity-gated); Gemma-4 explained end to end

### The loader takes a per-layer store map

- `load_moe_4bit_streaming(quantize_layers=...)` accepts a **mapping** `{layer: spec}` beside the existing `None` and set forms.
  A spec is a scheme name (`"int8"`), a `(scheme, blocksize)` pair, a `{"quant_type": ..., "blocksize": ...}` dict, or `None` for
  the base dtype; layers the mapping does not name stay in the base dtype. `layer_store_spec()` is the documented selector, and a
  map is **refused** on the arena and dedicated-quant paths, which carry one store for the whole model. `blocksize` is also a
  first-class argument of the direct path now. Why: expert layers are not equally sensitive to quantisation — on Gemma-4 the span
  is **159x** — so one store for the whole model is the wrong shape for some families (#597).

### Measured this release (no default changes, nothing silently applied)

- **Training, the field recipe, on grouped-nf4-gemm 0.32.1** (`bench/tp4/RESULTS-tp4-p46cut.md`, 21 register rows
  `e4b.train.h2h.unsloth.*.2026-09-19`): Qwen3-30B-A3B on one RTX 5090, both frameworks training the same 642,514,944
  parameters — e4b `fused_attn4` **6.4707 s/step against Unsloth's 29.0547 (x4.490)**, 237 vs 48 tok/s, 1186 vs 3341 J/step,
  the same peak VRAM, held-out delta 0.0283 nats (COMPARABLE). On the previous cut this arm did not finish. e4b's fused path
  passes its own parity control against its dense reference on **every** family measured (0.00063 / 0.00203 / 0.00140 /
  0.00159 against a 0.05 band), which was the registered condition for this release to proceed. Granite 2.853 -> 2.366 and
  OLMoE 2.739 -> 1.395 on the same cut. Four comparisons are deliberately NOT quoted, each with its reason in the coverage row.
- **Gemma-4 (#597), explained end to end across four lanes** (`bench/p47`-`bench/p51`): the serving stack is innocent; e4b's
  modelling is faithful (0.0056 nats with 29 of 30 expert stacks bf16); the cost is NF4 on the EARLY expert layers and the
  sensitivity is **positional** — the same ~8 % expert-branch damage costs 159x more at layer 0 than at layer 27, while an 8.5x
  smaller damage at layer 0 buys 22 %; no store rescues layer 0; the tail cannot be crushed further (704 = 64 x 11). What works
  is a **graded map** — `{0..9: None, 10..19: "int8", 20..29: ("nf4", 64)}` — which at matched bytes is **1.44x better than a
  uniform high-precision head** (0.1695 nats at 25.70 GB vs 0.2448 at 25.21 GB). **No Gemma-4 position is quoted and no default
  ships**: the K8 two-text gate on the served stack is still owed.

### Also

- **CI pins grouped-nf4-gemm at the v0.32.1 release commit** (`9206352f`; the `[fast]` floor `>=0.30.0` is unchanged) and the
  system manifest is the v0.32.1 copy (`consumer_ci_pin` prose now names v0.32.1). What 0.32.1 changes for e4b training: the
  grouped-LoRA delta's `auto` path now pads unless the padded block would not fit (P46, `bench/p46/RESULTS-p46.md`: 4.22 vs
  24.46 s/step at Qwen3-30B-A3B's field recipe, same loss, same peak VRAM). **No training position moves from this entry** --
  the tp4 box-B head-to-head re-runs on the 0.32.1 commit and the position, if any, is quoted from that receipt.
- `bench/tp4/tp4_run.sh`: refuses before any fetch when the instance overlay has under `TP4_MIN_DISK_GB` (200) GB free (P48
  run 1 died ENOSPC on a 32 GB overlay; the launcher orders machine disk, the instance overlay is what the box gets).

## 0.36.2 — 2026-09-19 — the K16 small-M int4 attention route ships and defaults to `auto` (−1.06 ms/step at B=16 on the 5090, P5 read); the P43 read (T1: no collapse, host-bound; T2b: #558 is a per-family band); the P44 and P45 instruments

### K16 route: `Int4Linear` serves 2..16 rows with grouped-nf4-gemm's small-M int4 GEMM (#578, #587; lane K16, #561)

- `Int4Linear(smallm=True)` routes `1 < rows <= 16` to `int4_smallm.gemm_int4_b32_smallm` (grouped-nf4-gemm ≥ 0.32.0) on the
  SAME packed bytes, split-K workspace preallocated at construction (capture-legal), **no cached bf16 copy** for those rows
  (#561). One row keeps the int4 GEMV; more than 16 rows keep the cached bf16 matmul.
- **Default `auto`** (`resolve_smallm`, applied to BOTH `enable_serve_attn_int4` and `enable_serve_attn_int4_calib`): the route
  is ON when the installed kernel package carries `int4_smallm`, OFF with a one-line banner when it does not — never a silent
  fallback, never a refusal on an older cut. `E4B_ATTN_INT4_SMALLM=1` requires the kernel (refuses without it at enable time),
  `=0` keeps the cached-bf16 path. The `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`; CI pins the kernel at the v0.32.0
  commit (`8b1acc9e…`).
- **Why the default moved — the K16 P5 read** (`bench/k16/RESULTS-k16-p5.md`, lane `k16-p5`, receipt `2026-09-19/k16-p5/`,
  P42's census protocol on one RTX 5090): `int4_b16` 12.25 ms/step → `int4_b16_smallm` **11.19 ms/step** (−1.06 ms, −8.6 %;
  1,306 → 1,429 tok/s); the bf16 GEMM family that carried the attention projections falls by 2.35 ms/step and the K16 kernel
  costs 1.30 in its place (net −1.05; the microbench predicted −0.96); the route appears in the smallm arm's census (192
  calls/step) and in no other arm's. **P5 HOLDS** (≥ 0.4 registered). B=1 is untouched, so every K8 row on record is
  unaffected; at 2..16 rows the route computes the same `x_bf16 @ dequant(W)` with fp32 accumulation, within bf16 rounding.
  Kernel side: grouped-nf4-gemm 0.32.0 (`gnf4.kernel.k16-smallm-int4-gemm.5090.2026-09-19`, measured; P4 untested there).
- `bench/k16/`: the K16 5090 runner (installs pytest on the box, #579), the P5 census lane (`k16p5_run.sh`, `k16p5_drive.sh`,
  `k16p5_reduce.py`, #584).

### P43 read (#582): T1 — no collapse, a host-bound step; T2b — #558 is a per-family parity band, not a kernel defect

- T1 (`p43-t1-qwen3-2`, 5090, three arms at the field fixture): `fused_attn4` **29.30 s/step**, `fused_bf16attn` 29.47,
  `fused_attn4_mb1` 50.65; ρ 0.70 / 0.67 / 0.16 — a +15 % drift over 20 steps at mb2, no RAMP, no CLIFF; tp4's collapse
  (24.7 s → > 200 s) did not reproduce. Int4 attention is not the tier (P2). **P4 refuted: the card sat at 16 % utilisation /
  113 W from step 0** — host-bound the whole run. The follow-up is P45 (below), not the RAMP/CLIFF lanes the rule named.
- T2b (`p43-t2b-g4sweep-2`, H100 NVL; run 1 walked the oracle's vision tower, #580): 30 text-decoder layers, 23,314 positions,
  rms(fused − reference) 0.0044 → 0.42 (**P7 holds**); the fused/reference-to-oracle ratio stays in [0.988, 1.007] on every
  layer (**P8 holds**: neither path is closer to bf16 anywhere). Under the registered rule **#558's remedy is a re-derived
  per-family parity band, not a kernel fix**; no Gemma-4 training position is quoted until that step closes.

### P44 instruments (#581, #583, #585): OLMoE two-text K8 + per-expert census, KL-from-bf16 for the families K8 cannot read

- `bench/p44/serve_stack.py` (ONE table of the arms bo7 timed + `build_served_model()` mirroring `step_decomp`'s all-VRAM
  assembly with an engagement census), `kl_serve.py` (K0-gated; reference scored once decode-shaped, cached, freed; **one child
  process per arm** because the lane hook arms only when a lever flag is set at interpreter start; a set lever that engaged
  nothing refuses the row; gpt-oss's dequant reference proven by its weights), `expert_residuals.py` (per-expert
  `sqrt(tr(DHDᵀ)/tr(WHWᵀ))` for RTN and GPTQ from the recipe's own Hessians), `p44_reduce.py` (k8_gate two-text rule, the KL
  reading rule, P3 tail statistic; a missing row is NOT_READ), runners/drivers, amendments 1–2 in `P44-PREREG.md` (2048-step K8
  window; decode-shaped scorer; the 1.1-nat Gemma-4 rows of run 2 disclosed as unexplained with two controls added).
- First OLMoE rows (`p44-a-olmoe-2`, quoted here as observations; the register rows follow the reducer): `int4all` c4val1
  **+0.255 ppl** (FAIL two-sided), `calibexp_all` (streamed 64k on wikitext-train) wikitext −0.055 / c4val1 **+0.443** (FAIL
  one-sided) — the Qwen3 recipe does not transfer to OLMoE; its licensed position stays NF4.

### P45 instrument (#586, #588): where a training step's host time goes

- `bench/tp4/tp4_arm.py --profile-steps K --profile-warm W` wraps K optimizer steps of the shared loop in `torch.profiler`
  (device-busy fraction, device events/step, CPU op self time by op family, top rows → `<receipt>_profile.json`; profiled steps
  are flagged, the timed number never comes from them); `tp4_run.sh` box E (`qwen3prof`: e4b + Unsloth at the field recipe with a
  1 s `nvidia-smi dmon` sampler); `bench/p45/P45-PREREG.md` + `p45_reduce.py`.

## 0.36.1 — 2026-09-18 — ernie4_5_moe loads its released checkpoint: tensors the text model does not build are skipped when the modeling class declares them, refused by name otherwise (#529)

**A released checkpoint that ships a speculative-decoding block now loads.**
ERNIE-4.5-21B-A3B-PT carries 12 multi-token-prediction tensors its text model
does not build; the loader died on the first of them with a bare
`AttributeError`. It now honours the modeling class's own
`_keys_to_ignore_on_load_unexpected` — what transformers' `from_pretrained`
does with those keys — and skips them before reading a byte, and any other
tensor with no module is refused by name with the two declared-drop routes.
Affects every family whose class declares such patterns (ERNIE-4.5 MoE and
DeepSeek-V4 today) and, only as a clearer error, any checkpoint carrying a
tensor the model does not build; nothing changes for a checkpoint whose every
tensor has a home, and no gate, threshold, floor or registered number moves.
**Verified on the released ERNIE-4.5-21B-A3B-PT** off the LAN (CPU, bf16):
9 shards integrity-clean, load 81.6 s, 12 tensors skipped, 27 keys renamed,
27 of 27 MoE layers quantised (nf4), forward finite — a `reference-ok` row in
`docs/ARCHITECTURE_SUPPORT.md`. Upgrade if you load ERNIE-4.5 MoE or any
checkpoint with an MTP / next-token block; no action otherwise. A patch
release: the `[fast]` extra stays at `grouped-nf4-gemm>=0.30.0` and the CI
kernel pin stays at v0.31.0.

### A checkpoint tensor the text model does not build: skipped when the modeling class declares it, refused by name otherwise (#529)

- ERNIE-4.5-21B-A3B-PT's released index carries a multi-token-prediction block — 12 tensors
  under `model.mtp_block.0.*`, `model.mtp_emb_norm.*`, `model.mtp_hidden_norm.*` and
  `model.mtp_linear_proj.*` — that `Ernie4_5_MoeModel` does not build. The loader walked to the
  first of them and died with `Ernie4_5_MoeModel has no attribute `mtp_block``, a bare
  `AttributeError` naming neither the key nor a remedy. `docs/ARCHITECTURE_SUPPORT.md` read
  `validated` for the family on fixture evidence; the fixture had no MTP weights, which is exactly
  the gap that row's own caveat warns about.
- The loader now honours the modeling class's own `_keys_to_ignore_on_load_unexpected` at the
  non-expert assignment pass — what transformers' `from_pretrained` does with those keys (ERNIE-4.5
  MoE declares `["mtp"]`, its modeling file saying "Not supporting multi-token prediction (MTP)
  atm"; DeepSeek-V4 declares `["(^|\.)mtp\..*"]`). A key with no module that matches is skipped
  before its shard is read and counted in the log (`skipped N checkpoint tensor(s) the text model
  does not build (<Class>._keys_to_ignore_on_load_unexpected)`); a key with no module that matches
  nothing raises `loader.UnplaceableTensorError` — an `AttributeError` subclass, so what caught the
  old exception still catches it — naming the key and the two declared-drop routes
  (`CKPT_KEY_REWRITERS`, the convention's `drop_re`). Family-agnostic: no per-family table, and
  DeepSeek-V4's bespoke `mtp.` rewriter stays as the first line.
- `tests/test_unbuilt_checkpoint_tensors.py`: the family's own tiny model plus tensors under the
  released MTP prefixes loads, counts the skipped tensors, and forwards; an *undeclared*
  unplaceable tensor is refused by name (the false-accept probe — the skip is gated on the
  declaration, not on "anything with no module"); the patterns are read from the class (ERNIE
  yes, Qwen3-MoE none); `_assign` refuses by name.
- **Verified on the released checkpoint** (2026-09-18, `bench/support/rows/ernie4_5_moe.json`,
  `bench/support/support_probe.py` on CPU, bf16, transformers 5.17.0 / torch 2.14.0 / bitsandbytes
  0.50.2): ERNIE-4.5-21B-A3B-PT off the LAN — 9 shards each matching the length its own header
  declares; load 81.6 s with `skipped 12 checkpoint tensor(s) the text model does not build
  (Ernie4_5_MoeForCausalLM._keys_to_ignore_on_load_unexpected)` and 27 transformers checkpoint-key
  renamings (`moe_statics`); `verify_moe_4bit(strict)` 27 quantised / 0 unquantised (nf4); one
  forward finite (loss 5.17 on synthetic ids — a finiteness smoke, not a quality measure); grade
  **`reference-ok`**. CUDA-graph capture not tested (CPU probe). The family loads by convention
  (`QWEN2_MOE`), not through `SUPPORTED_ARCHITECTURES`, so the row sits under "probed but not in the
  claimed list" by design; `docs/ARCHITECTURE_SUPPORT.md`'s hand-written row is now real-checkpoint
  evidence and its Notes say so.

## 0.36.0 — 2026-09-18 — the licensed int4 expert pack is bytes with its gptq/rtn decision recorded (#405, #530, #537); the fused gate/up order and the attention projections are declared by structure (#518, #519, #426)

**Two things stop being recipes and become records.** The calibrated int4
expert pack is an artifact with a root fingerprint, and the per-expert
gptq/rtn decision travels inside it, so a licensed pack loads byte-for-byte
on another box and a re-pack honours the recorded split instead of
re-deriving it from routing counts at the noise floor (#405, #530); a
streamed build's record now covers every chunk, not the last one (#537). The
fused `gate_up_proj` orientation is a declared, validated field on every
convention and an up-first family is refused at the loader instead of
computing `up * act(gate)` silently (#518, #519); attention projections are
found by module structure, so Gemma-4's `k_eq_v` layers are counted rather
than skipped (#426). Affects the int4 serving lanes
(`enable_serve_experts_int4*`, `dump_calibrated_artifact`,
`E4B_INT4_ASSIGNMENT`), attention-4-bit and attention-LoRA on every family,
and — only as a refusal that no shipped convention triggers — every fused MoE
load. Upgrade if you build or consume calibrated int4 packs, target
attention on Gemma-4, or pin this package by version: the 0.35.3 wheel on
PyPI predates `detect_attention_projections`, `pack_manifest` and the
assignment API, so that version string named two code states (#488) and this
release closes it. No gate, threshold, floor or existing claim value moved.
The `[fast]` extra stays at `grouped-nf4-gemm>=0.30.0`; CI pins the kernel
package at its v0.31.0 release commit.

### The gptq/rtn decision travels with the pack; a re-pack honours it (#530)

Mechanism only. No gate, threshold, floor, `min_rows`, damping, or existing claim value moved; no licence granted or withdrawn. The P37 VOID stands until a pack built from a recorded assignment passes the two-text gate on a second box.

- **Why**: the per-expert gptq/rtn choice is `routed_rows >= min_rows`, a threshold on a quantity at the router-flip noise floor, so it does not reproduce across boxes (P37: 10 of 12,288 experts flipped, `11522/766` vs the licensed `11512/776`; every e4b arm VOID). `#405` made the licensed **bytes** reproducible by artifact; it recorded only a *hash* of the decision, so a re-pack elsewhere could verify a mismatch but never avoid one.
- **`pack_manifest`**: the decision (`method_map` per `(layer, expert, role)` + `row_counts` + the `min_rows` that created it) is written as a **hashed payload** `payloads/assignment.json`, so the root `pack_fingerprint` covers it; the manifest's `method_map_hash` is derived from that payload and `verify_artifact` refuses a manifest copy that disagrees. `read_assignment(artifact_dir | assignment.json)`, `assignment_index` (duplicate keys with different methods refuse). `#405` artifacts without the payload still verify and load.
- **`enable_serve_experts_int4(assignment=...)`**: honours the record and does **not** consult `min_rows`. Refusals, never silent fallbacks: an expert the record names `gptq` that this box's calibration never routed to (no Hessian) refuses; an expert the record does not name refuses; an assignment without Hessians refuses. Where the record disagrees with what `min_rows` would have picked locally, that is **counted and reported** (`INT4EXP assignment honoured <hash>: N expert-roles where local routing disagrees`; provenance `assignment_honoured`), never applied.
- **`enable_serve_experts_int4_calibrated(assignment=...)`** and env **`E4B_INT4_ASSIGNMENT`** (file or artifact dir) so a lane hook can pin the licensed split without a code change; the argument wins over the env; a bad path refuses. Absent both, the recipe decides — unchanged.
- **Provenance and dump**: the live record now carries `method_map` and `row_counts` as lists (not only their hashes); `dump_calibrated_artifact` writes them as the hashed payload; a licensed `enable_serve_experts_int4_from_artifact` load carries them back, so dump → load → dump reproduces the same `pack_fingerprint`.
- **What it does not do, said plainly**: it fixes the *classification*. GPTQ output also depends on the Hessian, which routing flips also perturb, so bytes still reproduce only via the artifact. The experiment this enables is the one the issue asks for: re-pack on a second box from the recorded assignment and run the K8 two-text gate — if c4val1 still fails, the split was not the cause.

### Pack-manifest: licensed int4 experts are bytes, not a recipe (#405)

Code and register contract; no gate, threshold, floor, `min_rows`, damping, or existing claim value moved. Qwen3 licensed 238.1 / 1327.5 stay. No `pack_fingerprint` hashes invented for existing licence rows (the bo6c bytes were not retained).

- **`experts4bit_qlora.engines.pack_manifest`**: canonical JSON manifest, per-payload sha256/size, root `pack_fingerprint = sha256:<64 hex>` over ordered `(path, size, sha256)`. Serialize/load packed tensors + scales. Verify refuses corruption, missing files, wrong model revision / layout / fingerprint.
- **`enable_serve_experts_int4_calibrated`**: an `expected_fingerprint` loads the artifact and **refuses** a mismatch — no recipe fallback. Recipe builds remain observations; `dump_artifact_dir` writes them. Live calibrated packs attach observed provenance (`pack_fingerprint`, component hashes, method-map hash, row-count-vector hash, calibration-token SHA, toolchain) onto the model for decode receipts (`step_decomp.py`).
- **Review fixes (2026-09-06)**: the manifest's identity fields (`schema_version`, `layout`, `model_id`, `model_revision`, `layers[]`) are written a second time as a hashed payload `payloads/identity.json`, so the root `pack_fingerprint` covers which checkpoint and layout the bytes belong to; `verify_artifact` refuses a manifest whose top-level copy disagrees with the hashed one. The licensed loader takes N/K from the hashed payload bytes (`packed [E, N, K//2] uint8`, `scales [E, N, K//32]`) and treats `layers[]` only as a cross-check that refuses on disagreement. `dump_calibrated_artifact` refuses an unknown `model_revision` unless `allow_unknown_revision=True` (then the pack carries `model_revision_missing: true`).
- **p37 reducer**: when a lane names `expected_pack_fingerprint` / `E4B_EXPECTED_PACK_FINGERPRINT`, licensed arms VOID on exact fingerprint mismatch; legacy count-banner VOID stays for receipts without an expected hash. Anchored P37/bo6/bo7 receipts untouched.
- **Claims contract**: optional `pack_fingerprint` in `docs/claims-schema.md`; `scripts/check_claims_register.py` regex-checks the format and requires the field to match across an ACTIVE `licensed_by` pair once either side carries it. Step (4) determinism lane is not this PR.

### Serving census rows name the comparator (#418)

Register wording only; no value, status, gate, threshold or floor moved. Jordan's ruling: the Qwen3 licensed 238.1 / 1327.5 rows stay.

- Every `e4b.serve.census.bo7.*` speed row's `unit` and `claim` name the comparator as **vs e4b's own NF4 control on the same box**, never a bare ×N speedup. Granite, OLMoE, gpt-oss, Gemma-4 and Mixtral notes carry **no field comparator measured**. Qwen3 names the P37 vLLM 0.28.0 GPTQ-Int4 / MarlinExperts comparator (footprint not recorded) and scopes the licensed position to the bo6c pack artifact (11512 gptq / 776 rtn); #405 is a notes reproduction item, not a licence withdrawal. The P37 root row is bounded to graph decode at B=1 and B=16 on one box and one prompt set.
- No structured `comparator` field (the register validator does not check one). `docs/STATUS.md` and `docs/SERVING-THROUGHPUT.md` hand-edited; README results table updated to the same wording.
- `llms-full.txt` regenerated.

### Attention 4-bit + LoRA: detect projections by STRUCTURE (#426, #412)

`quantize_attention_projections_4bit` and `add_attention_lora` now share one
detector (`detect_attention_projections`). Admission is by module STRUCTURE,
never family name: `q_proj`, `k_proj` and `o_proj` must be supported linears;
`v_proj` may be a supported linear or absent/`None` (Gemma-4 `attention_k_eq_v`
layers, where transformers sets `v_proj=None` and reuses `key_states` as V).
The expected count is `len(candidates)` on the snapshot, never `4 * n_layers`.
A layer with `q_proj`/`o_proj` but no `k_proj` is refused rather than guessed.
The bias refusal still fires after admission, so gpt-oss's "96 of 96 attention
projections carry a bias" REFUSED holds. `docs/capabilities.json` `gemma4_text`
attention-4-bit stays "not supported pending #412"; the tp2 VOID rows stay VOID.
No gate, threshold, floor or registered claim moved.

### The fused gate/up order is a declared, validated field; up-first is refused at the loader (#515 → #518, #519; #509 → #514)

- `MoEConvention.fused_order` (default `("gate", "up")`) records which half of a fused `gate_up_proj`
  `[E, 2*inter, hidden]` is the gate. It is an adjudicated fact, never inferred: gate and up are
  shape-identical, so a swap computes `up * act(gate)` with every structural gate still passing.
  `__post_init__` refuses any value but the two orders and refuses a non-default order on a non-gated
  convention; `gate_first` is the predicate consumers read. Nine conventions have an unmatchable
  `expert_re` (natively pre-fused or nested — dense, granitemoe, gptoss, qwen3_vl_moe, gemma4, axk1,
  jetmoe, dbrx, qwen3_5_moe) and all nine are gated, so for each an order exists to get wrong; the
  field is data rather than a comment because those families have no key names left to recover it from.
- `loader.expert_layout_for`, the single funnel from the convention system into the loader, raises
  `MoEConventionError` for a convention that declares up-first, until the `chunk(2, dim=-1)` consumers
  (the vendored expert forward, deepseek_v4's dense path, the hybrid and hot-residency engines,
  ExpertsLoRA) are parameterised on the order. No shipped convention declares up-first, so nothing that
  loaded before is refused now.
- `tests/test_moe_conventions.py` adds a numerical detector: pack with the real `fuse_experts`, split
  with the arithmetic the consumers apply, and compare against a reference built from the *separate*
  gate/up tensors (no fused layout to inherit a mistake from) to 1e-12 in float64 on every expert; a
  false-accept probe shows a pack with the halves exchanged fails the same comparison on 3 of 3
  experts. The three pre-existing orientation tests are regexes over upstream source text and could
  not see this pipeline disagreeing with itself.
- Still open on #515: the field states the order and the detector checks this package's packing
  against its own split; neither can tell a reader that a natively pre-fused checkpoint's own order
  differs from the adjudication.
- `axk1` is annotated **staged, not wired** (#509 → #514; comments and one doc line): `loader.py`
  admits it by neither route (`SUPPORTED_ARCHITECTURES`, or a read-compatible convention —
  `{qwen2_moe, mixtral, phimoe}`), and `rewrite_axk1_keys` is absent from `CKPT_KEY_REWRITERS`, so
  admitting it alone would load with the wrong key mapping. `axk2` maps onto `QWEN2_MOE`, which is
  read-compatible, and is admitted; `mixtral` loads by the same second route, which is why it carries
  real-weight PASS receipts without a `SUPPORTED_ARCHITECTURES` entry.

### Streamed calibration merges each chunk's provenance (#537)

- `enable_serve_experts_int4_calibrated` enables the pack one layer chunk at a time, and each chunk's
  enable called `_attach_live_pack_provenance`, which **replaced** the live record — so after a
  48-layer streamed build the record, and the hashed assignment payload `dump_calibrated_artifact`
  wrote from it, described the last chunk only. P39 box 1 dumped an assignment naming 8 of 48 layers;
  box 2's honoured build refused `layer 0 expert 0 gu: not named by the assignment` — the refusal #530
  promised, and correct. The single-chunk tests could not see it.
- A live record already on the model for the same checkpoint is now merged: component hashes by path
  (the chunk's bytes win), the decision by `(layer, expert, role)`, row counts by `(layer, expert)`,
  counts summed, honoured-disagreements summed under the same hash, and every hash and the
  `pack_fingerprint` recomputed over the union — so the record after the last chunk equals what one
  all-at-once enable writes. Test: a two-layer checkpoint enabled as `layers=[0]` then `layers=[1]`
  carries both layers' decision, summed counts, all eight payloads and the SAME `pack_fingerprint` as a
  single enable, and the artifact it dumps names both layers.

### Documentation, and what is repository-only

- `engines/int4_experts.py`'s module note describes the dispatch, not the design (#496 → #520):
  batched decode stays on the NF4 M-tile path in every default configuration (`DEVICE_GROUPING =
  [False]`, assigned nowhere in the package), and the `gemm_int4_b32_grouped_captured` branch the
  published `e4b.serve.b16.qwen3-30b.int4.5090` row was measured in is reached only when a bench flips
  that flag. `tests/test_int4_docstring_matches_dispatch.py` holds the note to the code.
- `docs/ARCHITECTURE_SUPPORT.md` is regenerated from `bench/support/` rows (#521, #522; PR title:
  "1 of 9 evidenced becomes 6 of 9"): real-checkpoint load / verify / forward rows on current versions,
  `tests/test_support_doc_matches_rows.py` fails when regenerating is not a no-op, and the
  architecture claim is joined to grouped-nf4-gemm's shape census (#353's missing artifact). The
  `SUPPORTED_ARCHITECTURES` comment in `engines/offload.py` no longer states a family count.
- `train.py` logs the structurally expected attention-projection count beside the converted one.
- Repository-only, not in the wheel: the pre-registered lanes and their box-side runners under `bench/`
  (P39, P41, P42, tp3, tp4, and the K14/K15 runners whose pre-registrations live in grouped-nf4-gemm),
  the compute-governance policy and run-receipt ledger, and the Vast rent launcher fixes. Lane results
  are registered where they moved a number; see `docs/claims.json` and the receipts they name.

## 0.35.3 — 2026-09-06 — the loader honours a pinned checkpoint revision (#404); tp2 / P40 into the register (#415)

One behaviour change (the loader threads `revision` into both hub lookups, pins remote modeling code to the same commit,
records the commit actually loaded and refuses mismatches; #404 → #410) and the claims half of the tp2 bundle (31 rows,
`training_support` per family × path from the receipts, companion docs; #415 → #419, receipts in #414). No gate, threshold,
floor or existing claim value moved; the `fast` extra's floor (grouped-nf4-gemm >= 0.30.0) and the CI pin are unchanged.

### tp2 / P40 into the register: the per-family Unsloth head-to-head (claims + docs; the claims half of the tp2 bundle, receipts merged in #414)

Documentation and register only; no code, no version bump, no gate, threshold, floor or existing claim value moved.
The receipts landed separately as `bench/h2h-20260906/tp2/` (#414); this entry registers them (#415).

- **31 rows under `e4b.train.h2h.unsloth.<family>.5090.2026-09-06`** (lane tp2 / P40, 2026-09-06, one rented RTX
  5090, Vast 50005568 on a Ryzen 7 5700X3D host; the pre-registration verbatim as the bundle's `P40-PREREG.md`;
  every number copied from `RESULTS-tp2.md` and the receipt JSONs; the `.coverage` and `.footprint` rows per the
  spec amendment on #415): one row per attempt (`….arm.<framework>.<arm>` — including Granite's Unsloth VOID,
  OLMoE's Unsloth HARNESS_ERROR, gpt-oss's three REFUSED rows and Gemma-4's two `void_attn4` rows, #412),
  position + `.quality-n60` rows on Qwen3 and Mixtral, a `.footprint` row on Mixtral, `.coverage` rows on Granite
  and OLMoE (the comparator could not train the experts there — attention-only LoRA / a crash at MoE-LoRA engage
  — its own log lines quoted; no speed ratio in those rows), `.e4b-internal-parity` PASS rows on Granite, OLMoE,
  Qwen3 and Mixtral. The positions: Qwen3-30B-A3B s/step Unsloth/e4b **1.457** (e4b faster per step; held-out
  COMPARABLE, Δ +0.0152) — the cross-lane anchor, **+3.1% from P38's 1.413, inside the pre-registered ±10%**,
  noted on the P38 row (neither supersedes the other); Mixtral-8x7B **0.361**, whose row leads with the footprint
  trade: e4b trained under its registered expert-offload design at a **3.223 GB** peak (the
  trainable-on-smaller-cards result, `….footprint`) against Unsloth resident at **29.163 GB**, its only mode, and
  what that VRAM buys it is speed per step — a footprint-vs-speed trade, not a kernel deficit (held-out
  COMPARABLE, Δ −0.0087; P4's OOM prediction falsified).
- **`training_support` updated per family × path from these receipts only** (`docs/capabilities.json`, inside the
  per-path structure — never a flat boolean): the attention-4-bit configuration (`TRAIN_ATTN_4BIT`,
  `reference_attn4`/`fused_attn4`) now has receipts — supported on `granitemoe` / `olmoe` / `qwen3_moe` /
  `mixtral` (the new claim ids cited per path); **not supported on `gemma4_text` pending #412**
  (`quantize_attention_projections_4bit converted 100 projections, expected 120`; the bf16-attention `fast_train`
  path stays exactly as tp1 left it); refused on `gpt_oss` (96 of 96 attention projections carry a bias).
  `model_families` is unchanged.
- `docs/STATUS.md` (a dated "tp2 / P40 (2026-09-06)" section and a #412 open item),
  `docs/ARCHITECTURE_SUPPORT.md` (a dated per-family head-to-head section: what trained, what refused, the
  competitor observation quoted from the receipt — statuses, never a flat flag),
  `docs/solutions/qlora-fused-moe-experts.md` (the per-family measured-result paragraph, the attention-4-bit
  scope note, evidence rows), two routing queries in `docs/discovery-queries.json`, `llms-full.txt` regenerated.
  The README results table is unchanged: no existing claim's value or status moved.
- **Serving-comparison rule applied** (the #415 spec amendment's third clause): prose touched by this change may
  not quote a serving ×N against e4b's own NF4 control without the same-box vLLM figure, its weight precision and
  its resident footprint in the same sentence (P37: vLLM 0.28.0 serving `Qwen3-30B-A3B-GPTQ-Int4`, footprint not
  recorded in the receipts). No prose touched by this change quotes such a ×N, so no sentence needed the
  annotation; the rule is recorded here for the next edit that does.

### The loader honours a pinned checkpoint revision (#404)

- `load_moe_4bit_streaming(..., revision=<commit sha | branch | tag>)` threads the revision into both hub lookups the
  loader makes (`AutoConfig.from_pretrained` and `snapshot_download`), and pins a trust-remote-code checkpoint's modeling
  module to the same commit (transformers' `code_revision`); remote code hosted in a different upstream repository cannot
  be pinned by the weights' sha and is logged as unpinned. A snapshot staged with
  `snapshot_download(model_id, revision=<sha>)` writes no `refs/main`, so before this the unpinned `main` lookup had
  nothing to resolve offline (`LocalEntryNotFoundError` at load -- five training arms in a row on 2026-09-05) and
  online the loader streamed whatever `main` pointed to that day; every lane since P38 wrote `refs/main` by hand to
  work around it (P38 amendment 2, tp2). The commit actually loaded is recorded on `config._commit_hash`
  (transformers' own receipt slot, filled from the snapshot folder when transformers left it empty) and in the log
  line `checkpoint: <id> @ <sha> (requested ...)`; a full-sha `revision` whose snapshot resolves to a different
  commit -- or a config and a snapshot from two different commits -- is refused with `ValueError` rather than
  loading other bytes. A local directory has no hub revision: noted, not verified. Default `None` still means
  `main`; for existing callers the resolved commit is now logged, and `config._commit_hash` may now be populated
  where transformers left it empty (filled from the snapshot folder's basename). Nine loader tests cover the
  threading, both refusal arms (a pinned sha that resolved elsewhere; a config and a snapshot from two different
  commits), the unpinned receipt, the basename fallback, the local-directory case, the remote-code pinning
  (same-repo and upstream-repo `auto_map`), and a real offline hub-cache regression -- `HF_HUB_OFFLINE` forced over
  a staged `snapshots/<sha>/` cache with no `refs/`, both hub lookups untouched, the unpinned load shown to die
  there and the pinned one to resolve; no kernel, gate or threshold changes.

## 0.35.2 — 2026-09-05 — documentation: head-to-head receipts (same-box vLLM; Unsloth QLoRA end-to-end)

Documentation and tooling only; no runtime change. The `fast` extra's floor (grouped-nf4-gemm >= 0.30.0), the CI
`--requires` assertion and the `>=0.35.0` compatibility record in `docs/system-manifest.json` are unchanged (the
manifest is byte-identical to 0.35.1's). This release exists so the first end-to-end head-to-head against Unsloth is
in the repository with its pre-registration, every amendment and every attempt, and so the machine-readable surfaces
the 2026-09-05 audit found unchecked -- claim sentences, evidence paths, successors, licence labels -- are held by a
check from here on.

### vLLM 0.28.0 head-to-head, same box, identical prompt token ids (lane p37, receipt `bench/h2h-20260905/p37/`)

- One rented RTX 5090 (Vast 49975016, EPYC 7Q83 host), one session, decode-vs-decode on the same 512-token prompt ids
  (dumped once from the e4b harness's own window function and fed to vLLM verbatim; the reducer refuses to divide
  receipts that disagree on the prompt sha): vLLM 0.28.0 (torch 2.13.0+cu130, triton 3.7.1) serving Qwen's
  `Qwen3-30B-A3B-GPTQ-Int4` (Marlin, default CUDA graphs, kv auto; eager and fp8-KV arms beside) against
  experts4bit-qlora 0.35.0 + grouped-nf4-gemm 0.30.0 -- the NF4 control and the licensed recipe (bo7b's
  `calibexp_all_n128`, the pack rebuilt on that box), bo7's harness pieces byte-identical, sixteen arms with
  non-adjacent self-pairs, every knob in the receipts. Pre-registered before the box was rented (`PREREG.md`); three
  amendments (a staging refusal; the comparator fetch hang left to its alarm and recovered in-lane, the guarded follow-up
  ran as `NOT NEEDED`; the registered K8 gate on that box's pack, below); no threshold, arm, prompt set or knob changed.
- **What is quoted** (`e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05`, measured): vLLM 286.0 tok/s at B=1 (3.497
  ms/step; fp8-KV 300.9; eager 20.8) and 2030.0 aggregate at B=16 (7.882 ms; fp8-KV 2206.5; eager 322.5) against the NF4
  control's 113.4 / 500.1 -- **vLLM / e4b-NF4 2.52 and 4.06**, the only licence-free ratio the lane can produce (self-pairs
  inside 1.03×). **What is not quoted:** the ratio against the licensed stack. Every licensed e4b arm on that box is VOID
  under the pre-registered pack-fingerprint rule -- the streamed calibration there packed 11522 gptq / 766 rtn expert
  matrices where the licensed pack reads 11512 / 776 (the same recipe, ten of 12,288 matrices across the `min_rows`
  threshold, not the licensed bytes; speed cannot inherit a licence). The recipe's speed on that box (236.4 / 1305.3
  tok/s, ×2.08 / ×2.61 over its NF4 control -- bo7's ×2.067 / ×2.602 reproduced within 1%) is registered per arm as
  measured and unlicensed. **Amendment 3** (pre-registered after `TP_DONE`, the registered K8 gate on that box's own
  pack, bo6c's `k8()` verbatim, `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05.gate`): wikitext Δ −0.0230 ppl PASS, C4
  validation **Δ +0.1093 ppl FAIL** against the +0.05 budget (the licensed pack read −0.0662 on the same window; the two
  boxes' NF4 references agree to 0.0002 ppl) -- **NOT LICENSED, VOID stands**: no ratio against the licensed stack exists
  on that lane, and the streamed calibration recipe is shown not to reproduce its licence across hosts (an open item in
  `docs/STATUS.md`, filed as #405; bo6c's licence stands on its box). `e4b.serve.h2h.vllm.same-box` (2026-09-03, ×1.47 / ×1.55,
  a different box, the RTN stack, vLLM version unrecorded) is superseded for current-position use and stays as measured.
  One row per arm (`…arm.<engine>.b<B>.<arm>`, sixteen). Quality quoted, never equated.
- `docs/capabilities.json` (`serve-moe-on-consumer-gpu`: the claim and the limitation reworded), `docs/STATUS.md`
  ("Serving speed", "What changed", an open item: the licensed pack does not reproduce bit-for-bit across boxes), the
  README results table and "Do not use this when", `docs/solutions/serve-large-moe-on-a-consumer-gpu.md`, a routing
  query, `llms-full.txt` regenerated; the capability carries the host-dependence as a limitation.

### e4b vs Unsloth, QLoRA end-to-end, one identical training problem (lane p38, receipt `bench/h2h-20260905/p38/`)

- One rented RTX 5090 (Vast 49975389, train-anchor class `pcie-full/launch-fast` recorded): Qwen3-30B-A3B at one pinned
  revision, the registered `clinical` fixture tokenised ONCE into a sha-asserted file both frameworks train on, seq 512,
  r 8 / α 16 on attention q/k/v/o and every expert (321,257,472 trainable parameters asserted in every arm, router
  frozen), the same AdamW call, batch 1, loss over all tokens, the same held-out eval -- experts4bit-qlora 0.35.0 +
  grouped-nf4-gemm 0.30.0 in the image python (the fused `dgrad` path with NF4 attention, the shipped `TRAIN_ATTN_4BIT`
  mechanism; transformers 5.16.1, bitsandbytes 0.50.1) against Unsloth 2026.9.2 + unsloth_zoo 2026.9.1 in its own
  venv (its 4-bit MoE path, `native_torch` backend; transformers 5.5.0, bitsandbytes 0.50.2, peft 0.20.0 -- the
  transformers/peft difference between the two pythons is a recorded environment difference). Pre-registered before
  the box was rented (`PREREG.md` in the bundle, verbatim); every failed attempt a row; four amendments, all
  environment or instrument (the comparator venv's pins; the loader's `refs/main` and a `torchao` import; the Unsloth
  branch's snapshot-directory resolution; the U8 predicate evaluated on PEFT's wrapper, proven on the innermost module
  and re-reduced) -- no workload, fixture, threshold or knob changed.
- **The position at 60 steps** (`e4b.train.h2h.unsloth.qwen3.5090.2026-09-05`, measured): s/step ratio Unsloth/e4b
  **1.413** (2.151 vs 1.522 s -- e4b faster per step at this workload), peak VRAM 21.371 vs 23.141 GB, 157.1 vs 224.7
  J/step, time to a held-out loss of 0.32 92.5 vs 130.3 s; held-out loss comparable, 0.2923 vs 0.2975
  (`…quality-n60`, |Δ| 0.0052 ≤ the pre-registered 0.05 reading threshold). **At 200 steps the curves separate in
  Unsloth's favour: 0.2713 vs 0.2881** (`…curve-n200`) -- a measured row in Unsloth's favour, quoted beside the
  position wherever it is quoted; candidate causes (eval schedule, checkpointing mode, the two stacks' transformers /
  peft versions, the expert adapter's precision -- bf16 on this side because the loader passes the model dtype to
  `ExpertsLoRA`, fp32 on Unsloth's) are not established. e4b's fused-vs-reference pair passes its band on that box
  (`…e4b-internal-parity`: 0.00131 / 0.01138, ×2.92 per step; informational, tp1 owns the licence). One row per arm
  (`…arm.<framework>.<arm>`, eight, all VALID). The pre-registration predicted the opposite sign at this workload
  (Unsloth faster per step, lower peak) and shipped the finding either way. Nothing is superseded or licensed by this
  lane; the 2026-08-26 "1.17× ahead" memory (never a claim) is disqualified as a comparison. Found on the way:
  the loader does not honour a pinned revision (e4b#404).
- `docs/capabilities.json` (`qlora-fused-moe-experts`: the four claims and a limitation that says the 200-step curve
  favours Unsloth), `docs/STATUS.md` ("What you get today"), `docs/solutions/qlora-fused-moe-experts.md` (measured
  result, limitation, evidence), the README results table, three routing queries in `docs/discovery-queries.json`,
  `llms-full.txt` regenerated.

### The register, after the 2026-09-05 machine-readable audit

- **Superseded:** `e4b.serve.tp.qwen3.b1.5090.2026-09-04` and `.b16` -- their "best licensed" configuration class
  (round-to-nearest int4 experts + calibrated attention) failed its second text on lane bo5 -- now point at the census
  rows of the licensed stack, `e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05` / `.b16` (which name them under
  `supersedes`); their values stand as measured. The other four families' 2026-09-04 rows keep their numbers with the
  "best licensed" label withdrawn in the sentence (measured, not licensed), the NF4-position rows of gpt-oss and
  Gemma-4 say what they are, and the Gemma-4 / gpt-oss build-out rows no longer call themselves licensed under a gate
  their own notes say does not exist for the family. No number, gate, threshold or verdict changed.
- **`licensed_by`** (new field, `docs/claims-schema.md`): every active claim whose sentence asserts a licence names the
  claim whose receipt holds the K8 verdict -- the bo6c Qwen3 verdict row for Qwen3's census and census-row claims,
  the bo3 Granite row (its own receipt carries the +0.019 ppl pass) for Granite's; a row whose own receipt carries the
  verdict names itself.
- **Retired by id:** the 2026-08 "vLLM 6.31× ahead" figure (`e4b.retired.vllm-6.31x-ahead`; template prompts, a
  different box) so that `e4b.serve.h2h.vllm.same-box`'s supersession resolves; that head-to-head now carries full
  `conditions` (the vLLM version is not recorded in its receipt; the e4b arm is the 2026-09-03 RTN-int4 class of
  `e4b.serve.b16.qwen3-30b.int4.5090`, not the licensed stack) and a populated `quoted_in`.
  `e4b.retired.13.47x-training-speedup` is `retired` (its "about 7.2×" restatement has no receipt and was never a
  claim); `e4b.retired.inference-md-decode-grid` names its successor; two dangling `supersedes` pointers resolve or
  are dropped with the reason in notes.
- **Evidence resolves:** the three bo3 `measured` rows that named run logs never committed now name the K8 receipts
  beside their speed receipts (`bo3/*_ppl_*.json`); annotated paths are bare paths with the annotation moved to
  notes; the scratch probe of `e4b.serve.gptoss.loader-faithful` is dropped and said so; cross-repository and
  issue evidence use the structured forms the schema now defines (`{"repository", "path"}`, `{"url"}`).
- **`measured_on`** on the ten tp1 rows that lacked it and, from their receipts' own dates (stated in notes), on the
  twelve older measured rows.
- **Stale notes:** `e4b.train.flagship-matrix`, `e4b.serve.tp.granite.*` and nine other rows no longer say "pending"
  for things measured since; each states what is measured, with ids.
- Prose: `docs/solutions/serve-large-moe-on-a-consumer-gpu.md`'s "licensed configuration per family" is rewritten
  from STATUS's current positions (Qwen3 = the bo6c streamed stack; OLMoE and Mixtral = NF4; Gemma-4 = `r1epi` on NF4
  with no instrument; Granite = `r12epi`; gpt-oss = its NF4 reference arm), its Evidence lists the census rows first;
  `docs/STATUS.md` "Serving speed" leads with the licensed position and says the 2026-09-03 RTN class failed its
  second text and is not the licensed stack; `serve-moe-on-consumer-gpu` cites the bo7 census rows and the bo6c
  licence; `docs/SERVING-THROUGHPUT.md` is dated by section; abbreviated ids in STATUS and SERVING-THROUGHPUT are
  written in full; the document count in `docs/INDEX.md` is the number it links (held by a test) and the README no
  longer repeats it.

### Checks (tooling, not the package)

- New `scripts/check_claims_register.py` (CI, discoverability job; `tests/test_check_claims_register.py`): every
  `evidence[]` entry is a path that exists at HEAD or a structured cross-repository / issue entry (cross-repository
  entries verified against `--sibling`); `measured_on` required and ISO on every measured row; a `superseded` row's
  successor chain reaches an active row; `retired` rows carry `retired_reason`; `supersedes` and `quoted_in` resolve;
  no "pending"/"TBD" on an active row; a licence label names its verdict row.
- `scripts/check_readme_claims.py` applies its id rules to the position documents (`docs/STATUS.md`,
  `docs/SOLUTIONS.md`, `docs/solutions/*.md`, `docs/SERVING-THROUGHPUT.md`, `docs/SERVING-PARITY.md`,
  `docs/METHODOLOGY.md`, `docs/ARCHITECTURE_SUPPORT.md`, `docs/CHOOSING.md`): every backticked id exists, an inactive
  one only on a line that says superseded / retired / historical.
- `scripts/check_system_manifest.py`: the CI kernel pin (`git+…@<sha>` in `ci.yml`) is the commit of a release tag at
  or above the tag `consumer_ci_pin` names (`git ls-remote --tags`; without network a visible NOTE, never a silent
  pass), and every kernel-pinning extra (`fast`, `test`) floors at or above the current record's; CI now also clones
  the kernel package at its latest release tag and runs `--sibling` (byte-identical manifest) and the register check's
  cross-repository evidence.
- `scripts/check_capabilities.py` warns (never fails) when a capability whose primary mode is `serving` cites none of
  the newest serving lane `docs/STATUS.md` quotes as the position.
- CI installs grouped-nf4-gemm from the v0.30.1 release commit (`d58b39fb…`; the comment now states the pin's real
  reason); `pyproject.toml`'s `[test]` floor is `grouped-nf4-gemm>=0.30.0`, equal to `fast`'s.
- Checks hardened after review: `check_claims_register.py` reads evidence from the git tree (`git ls-files` -- an
  untracked or gitignored receipt log, a directory, an absolute path or `..` is a finding; outside a checkout the
  working tree stands in and the output says so), applies the licence rule per occurrence (one "unlicensed" no longer
  excuses a bare "licensed stack" beside it; the citation form ``licensed by `<id>` `` refers to another row's licence
  and is resolved -- the id must carry `licensed_by` -- rather than labelled: eleven sentences reworded with no number
  changed, the p37 rows citing the bo6c verdict), follows every `superseded_by` to an active row and refuses one on an
  active row, allows `retired_reason` on retired rows only (`e4b.retired.inference-md-decode-grid` drops the field it
  carried beside `superseded`; `e4b.retired.vllm-6.31x-ahead` names the current successor), and exits 2 for a
  `--sibling` whose slug cannot be resolved instead of skipping its entries; `check_readme_claims.py` reads the
  superseded / retired / historical words from the line's prose alone (links reduced to their text, HTML comments and
  every claim id in any form stripped) and treats ids in `<code>` or link text as citations; `check_system_manifest.py`
  gains `--require-tags` (CI: an unreadable tag list is exit 2, not a NOTE on a green run), reads `consumer_ci_pin` as
  the last `vX.Y.Z` in its prose and ignores comment and non-`pip install` lines; `check_capabilities.py` warnings are
  also `::warning::` annotations; `docs/claims-schema.md` documents `validity`, `row_status`, `parity_verdict`;
  `tests/test_docs_index_count.py` counts links through `md_links`.

## 0.35.1 — 2026-09-05 — documentation (the tp1 training parity matrix) and one behaviour change (#397 → #402)

The training parity matrix on real weights (lane tp1) with its documentation, plus **one behaviour change** from the
parallel #397 fix (#402, below) — **this release is not docs-only**. The `fast` extra's floor (grouped-nf4-gemm >=
0.30.0), the CI `--requires` assertion and the `>=0.35.0` compatibility record in `docs/system-manifest.json` are
unchanged. The release exists so that the repository, the PyPI project page and the site state the training matrix
from evidence, and so that the refusals that matrix argued for ship with it.

### The stock-epilogue contract: `ExpertsLoRA` refuses what it cannot represent (#397)

- `ExpertsLoRA` re-implements the expert forward inline (the low-rank delta lands before the nonlinearity), so it owns the epilogue and can only represent the stock `down(act_fn(gate) * up)` or a module that hands its epilogue over through `_apply_gate`. gpt-oss's stack (per-expert biases, clamped sigmoid GLU, de-interleaved at load, no hook) was wrapped anyway by the arena loader under `arena_train=True` and trained against a plain SwiGLU with nothing raised. New: `experts4bit_qlora.assert_stock_epilogue(module)` decides on the module's STRUCTURE (bias buffers/parameters/tensor attributes, a non-stock `forward` without a hook, `alpha`/`limit`/`swiglu_*` scalars no hook consumes, a forward body that clamps or adds a bias itself, an interleaved-layout marker, a two-argument `act_fn`, a declared shape that is not `[2I, H]`/`[H, I]`) and raises `EpilogueContractError` (a `TypeError`) naming every offending attribute and the faithful route (grouped-nf4-gemm's `mxfp4_qlora.ExpertsMxfp4LoRA`, the `mxfp4-moe-training-and-residency` capability). Applied by `ExpertsLoRA.__init__`, the loader's `arena_train=True` branch, `enable_nvme_train_residency`'s pre-flight (before the tier opens), `enable_hybrid_train` (which used to refuse gpt-oss through the tier's own flag), and `enable_fast` / `enable_fast_train` / `enable_batched_train`, which now REFUSE a wrapper whose base violates the contract instead of skipping it (a skipped wrapper would have trained the same unfaithful reference forward). Stock SiLU, Gemma-4's gelu_tanh, non-gated stacks and DeepSeek-V4's hooked clamp are unaffected. Both new symbols are exported; `docs/capabilities.json` and the solution pages are updated by the release bundle that carries this entry.
- `enable_mxfp4_nvme_residency` refuses a module that carries per-expert bias tensors: the binding passes no biases to the engine and defaults to the V4 epilogue, so gpt-oss would have been served through the wrong GLU with its biases dropped. The MXFP4-arena fused lane (`mxfp4_experts_forward`) keeps the module's own forward for any module the contract refuses.
- `enable_batched_train`'s per-call fallbacks (pad waste past `_PAD_WASTE_LIMIT`, evicted storage, an empty batch) are counted on each patched module; `batched_fallback_stats(model)` (exported) reports `calls` / `batched` / `fallback_calls` / `by_reason` per module and in total, so an arm claiming the batched path can assert `fallback_calls == 0` (the tp1 lane read OLMoE's batched arm as void because layers fell back with nothing counting them).
- The loader logs a one-time NOTE when a family's experts are built bare (no expert adapter; `r`/`alpha` do not apply), and `python -m experts4bit_qlora.train` refuses `TRAIN_EXPERTS=1` on a model with no expert adapter instead of training attention/router alone under that flag.
- Audit of the public enable/load/train/serve entry points for the class "unsupported behaviour, plausible output, no refusal": `docs/audits/no-silent-fallback-2026-09-05.md`.
- Folded under this release from `## Unreleased` at the bundle's rebase; the capability contract and the solution page carry the refusal (`EpilogueContractError`), the MXFP4 NVMe-residency refusal of bias-carrying modules and `batched_fallback_stats` as limitations, and `training_support.gpt_oss.nvme_train` is `refused` with that code reference.

### Training on real weights, per family, under the shipped code (lane tp1, receipt `bench/train-parity-20260905/tp1/`)

- The six serving families through the shipped training path on one rented RTX 5090 (train-anchor class recorded):
  the direct `load_moe_4bit_streaming` + `verify_moe_4bit(strict=True)` path on the real checkpoints, then
  `reference` / `enable_fast_train(dgrad=True)` / `enable_batched_train` for 60 steps on the registered `clinical`
  text, verdicts by `tp1_reduce.py` in the registered B2/C2 units (`|Δ final train loss| ≤ 0.05` and median
  step-wise `|Δ| ≤ 0.05` against the family's own reference; VOID when the arm cannot be read), cost reported and
  never gated. Every log, the amended lane script beside its pre-amendment copy, the patched harness and its patch,
  the reducer, `RESULTS-tp1.md` (the reducer's output verbatim plus the reading) and a README with the three
  amendments — the first box's 10 MB/s link, the fetch-by-short-sha false start and the ≈ 4.4 h it idled, the
  harness's closure bug fixed in flight — and predictions P1–P7 scored.
- Rows (`TP_DONE` 2026-09-05T15:22Z, all six families through the registered arms): the fused path **PASSES** on every
  family that has one — OLMoE-1B-7B-Instruct (`e4b.train.parity.tp1.olmoe.fused.2026-09-05`, the first reading on a
  registered text with real weights for that family), Qwen3-30B-A3B resident on the 32 GB card (`…qwen3.fused…`),
  Gemma-4-26B-A4B-it — the `-it` checkpoint, loaded without #344 on that host — with the step-wise median inside the
  band by a small margin (`…gemma4.fused…`), and Mixtral-8x7B-Instruct under `offload=True` at half the reference
  loop's peak VRAM (`…mixtral.fused…`; the family enters `model_families` on it); the batched path **PASSES** on
  Granite-3.1-3B-A800M (`…granite.batched…`; the family's first direct real-weight load) and Mixtral, and is **VOID** on
  OLMoE, Qwen3 and Gemma-4 — `enable_batched_train` falls back to the reference forward per call above
  `_PAD_WASTE_LIMIT` with no counter, and the kernel was not reached on every layer; gpt-oss fused / batched
  **REFUSED** (0 patched: the loader builds its experts bare), attention-only QLoRA trains with the frozen stacks
  bit-exact, and grouped-nf4-gemm's experimental MXFP4 route trains its experts on its own text with the canary passing
  — experimental, never licensed. Granite's fused arm **PASSES** on its corrected-counter re-run (`…granite.fused…`,
  `TP2_DONE` 15:33Z; attempt 1 is a kept HARNESS_ERROR row, `…granite.fused.attempt1…` — a closure bug in the harness's
  kernel counter, not the shipped code; the follow-up script's own abort between them is amendment 5). Every attempt a
  row, 18 result lines, nothing pending.
- `docs/claims.json`: one claim per family × arm (`e4b.train.parity.tp1.<family>.<arm>.2026-09-05`) plus a per-family
  matrix claim; notes on `e4b.train.fast-train-dgrad` (the batched "no speed-up at real width" is a Qwen3-30B-width
  reading — at Granite's width the batched path is the faster one measured), `e4b.train.flagship-matrix` and
  `e4b.train.olmoe-converges`. Nothing superseded or retired.
- `docs/capabilities.json`: `qlora-fused-moe-experts.model_families` is evidence-gated —
  `olmoe`, `qwen3_moe` and `gemma4_text` confirmed on tp1 rows, **`mixtral` and `granitemoe` added** (Mixtral's fused
  PASS under offload; Granite's on the corrected-counter re-run), `gpt_oss` stays out with the refusal named and the
  experimental route pointed at; the batched-fallback VOID (three families), the `EpilogueContractError` refusal and
  `batched_fallback_stats` are limitations; `training_support.gpt_oss.nvme_train` is `refused` with a code reference; the
  `mxfp4-moe-training-and-residency` capability carries the tp1 canary row and stays `experimental`.
- Phase directive 2026-09-05 14:45Z, applied to the bundle's shape: every row of the receipt is exactly one of OK /
  REFUSED / HARNESS_ERROR / ALARM / OOM / NOT_RUN / EXPERIMENTAL, classified mechanically by the bundle's reducer (v2;
  the box's v1 copy kept) with the parity verdict as a separate column, every attempt a row (Granite's first fused
  attempt stays a HARNESS_ERROR row beside its re-run) and every amendment referenced from the rows it touched; the
  `e4b.train.parity.tp1.*` claims carry `row_status` / `parity_verdict`. `docs/capabilities.json` states training
  support **per path** — a `training_support` object (`headline_path`; `by_model_type` keyed by model_type: quantize / reference_train / fast_train /
  batched_train / nvme_train / native_mxfp4_train, each `supported` / `refused` / `void` / `harness_error` /
  `not_tested` / `experimental` / `n/a` with its claim ids) on `qlora-fused-moe-experts` and
  `mxfp4-moe-training-and-residency`; `model_families` is exactly the families whose `fast_train` is `supported`.
  Tooling, not the package: `docs/capabilities.schema.json` admits the object and `scripts/check_capabilities.py`
  validates it (allowed values; `supported` / `void` / `refused` cite existing claim ids, `supported` ones active;
  `refused` says why; the `model_families` rule), with `tests/test_check_capabilities_training_support.py`.
- `docs/STATUS.md` (the training position and three open items), `docs/ARCHITECTURE_SUPPORT.md` (a new dated section
  "Training on real weights (tp1, 2026-09-05)"; the existing tables are untouched), `docs/SOLUTIONS.md` and
  `docs/solutions/qlora-fused-moe-experts.md` (families from evidence; the refusal and the VOID as "what to check"),
  the README's results table and scope, two routing queries in `docs/discovery-queries.json`, `llms-full.txt`
  regenerated.

## 0.35.0 — 2026-09-04

### Calibrated int4 experts, calibrated sequentially (#384)

- Per-expert GPTQ packing for the int4-b32 expert store from the fused forward's Hessian tap: `calibrate_expert_hessians`, `enable_serve_experts_int4(..., expert_hessians=)`, and the driver that decides the result, `enable_serve_experts_int4_calibrated`, which calibrates and packs layer chunk by layer chunk so every chunk is calibrated against the already-int4 prefix (GPTQ's sequential convention) and the host never holds more than one budget of Hessians. Knobs: `E4B_INT4_GPTQ_DEVICE=cuda` (GPU solve), `E4B_INT4_GPTQ_DAMP`, `E4B_INT4_HESSIAN_BUDGET_GB`. Off by default; the serve hook enables it with `E4B_SERVE_EXP_INT4_CALIB=1`.
- Verdicts stay in the register: Granite's calibrated experts fail the registered gate on their second text (`e4b.serve.buildout.bo5.granite.b1.5090.2026-09-04`, notes); the Qwen3 and Mixtral readings under the sequential method are in the bo6 receipt and are registered with its bundle, not here. Calibrating all layers at once against the unquantised prefix is the two-step API only and is not the method that ships.

### Decode glue through the kernel side (#385)

- `silu(gate) * up` through `swiglu_rows` and the top-k combine through `combine_rows` when the kernel side has them (`E4B_FUSE_SWIGLU=0` / `E4B_FUSE_COMBINE=0` are the A/B arms); the split-K reduce through `reduce_partials`. Qwen3's licensed single-stream and batched positions after this cut: `e4b.serve.buildout.bo5.qwen3.b1.5090.2026-09-04`, `e4b.serve.buildout.bo5.qwen3.b16.5090.2026-09-04`.

### gpt-oss: the native MXFP4 expert store (#372)

- gpt-oss experts are served from their released MXFP4 blocks and scales through the grouped MXFP4 GEMM, never re-quantised onto the int4 grid; single rows take the MXFP4 decode GEMV and batched rows keep the NF4 stack (`E4B_INT4_KEEP_NF4=1`): `e4b.serve.buildout.bo5.gptoss.b1.5090.2026-09-04`, `e4b.serve.buildout.bo5.gptoss.b16.5090.2026-09-04`.

### Receipts: the second text, in the gate's own units (#386, #390)

- The bo3 and bo5 build-out bundles are in-repo. Every calibrated and int4 arm now has a second text; three read FAIL as registered, and Mixtral's licensed label is withdrawn (`e4b.serve.buildout.mixtral.*`, notes). The gate was not retuned.

### Documentation and packaging (#388, #393, #389, #394, #391)

- The agent discoverability layer: routing sections in the README, `docs/SOLUTIONS.md` and six problem-first pages, `docs/capabilities.json` under a schema, `AGENTS.md`, `llms.txt` and the generated bundle, PyPI metadata with labelled project URLs, and a CPU-only CI job for all of it.
- The bitsandbytes position is version-, workload- and shape-aware (upstream 5453368 is in 0.50.0, not in 0.49.2); `e4b.train.energy-honest` is superseded by `e4b.train.energy-honest.scoped-a2000`, scoped to its measured comparator and build. Solution pages for capacity, QLoRA on fused experts, offload and MXFP4 carry their decision structure.
- LICENSE is the verbatim MIT text; the vendored bitsandbytes-derived file names `THIRD_PARTY_NOTICES.md` in its header. Homepage, Documentation, Status and Solutions point at the cerinamroth.com routing pages. CI pins grouped-nf4-gemm at a commit that no longer tracks a stale `build/lib/` (the previous pin could ship an old `nvme_reader.py`).
- Requires grouped-nf4-gemm >= 0.30.0 for the `fast` extra.

## 0.34.0 — 2026-09-04

Optimisation pass, day one: the round-2 fold reaches the stacks it never
engaged on, the calibrated int4 set is complete (attention, output head,
dense MLP, biased projections), an op-level census instrument, and a
retraction. Every lane number below is **measured-private** until the
receipt bundle lands in-repo; the numbers are on the merged PRs. Floors
grouped-nf4-gemm >= 0.28.0 (`rope_heads`).

### Round-2 glue reaches the calibrated int4 stack (#375) and norm-less attention (#379)

- The norm + rotary fold had only licensed the FUSED-qkv attention module.
  The calibrated int4 attention lane packs q/k/v/o separately and is
  exclusive with qkv fusion, so on every family's best stack the fold
  never engaged. #375 adds the same fold for the standard
  separate-projection shape (q/k/v/o + per-head q/k norms), licensed on
  structure. Qwen3-30B-A3B, full stack, one RTX 5090: K8 1.8505 → 1.84944
  (−0.0011 nats, inside the 0.0095 floor), B=1 6.41 → 5.62 ms —
  **156.1 → 177.9 tok/s (×1.14)**, ×1.82 over the NF4 baseline.
- #379 (grouped-nf4-gemm `rope_heads`, 0.28.0): the rotary chain folded
  for attention WITHOUT a head norm (GraniteMoe, Mixtral), licensed on
  exactly `{q_proj, k_proj, v_proj, o_proj}`; gpt-oss's `sinks` refuses
  it. Granite, same box: ΔK8 −0.0026 nats (inside the 0.0033 floor),
  **×1.156 at B=1, ×1.093 at B=16**, control arm flat. Review caught a
  vacuous license (the fold required `sliding_window`, which neither
  family's attention sets) before the lane ran.

### The calibrated int4 set is complete: output head (#373), biased projections (#377), dense MLP (#378)

- `E4B_SERVE_LMHEAD_INT4_CALIB=1` (#373): the calibrated int4 output head,
  opt-in. Qwen3 +0.0085 nats in-stack (below its floor), ×1.04 at B=1;
  Gemma-4 (262k vocabulary) ×1.078 on top of its calibrated stack.
- `Int4Linear` carries a projection bias (#377): gpt-oss's q/k/v/o no
  longer refuse the calibrated lane. Speed on gpt-oss ×1.005 alone,
  ×1.015 in-stack (head_dim 64 — a small attention slice); its best
  licensed B=1 moves 133.3 → 137.4 tok/s with the head. Quality is NOT
  readable for this family on raw text (K8 falls 0.13–0.17 nats under
  every int4-attention arm — the documented OOD-regime flattery); the
  flag stays opt-in for gpt-oss until a harmony-text or KL gate exists.
- `E4B_SERVE_DENSE_INT4_CALIB=1` (#378): the dense MLP beside a routed
  block as an opt-in calibrated target (Gemma-4's shape). Measured
  ×1.010 at B=1 on Gemma-4 — not a lever there; correct by test, shipped
  for completeness, default off.

### Instrument: `--op-profile-out` (#380)

- `bench/hybrid-g9/step_decomp.py --b1d-loop eager --b1d-timed
  --op-profile-out PATH`: an op-level census behind the kernel census —
  by call site (the dispatch-mode tracer; torch 2.13 returns empty
  profiler stacks), by op launch count, by op + input shape — from two
  uncaptured steps after the timed window, reserved so the window stays
  byte-identical to a no-flag run.

### Correction: Granite's int4-expert rows fail the registered K8 gate

- The 0.33.0 notes quoted Granite-3.1-3B-A800M at 302 tok/s (×1.59) with
  int4 experts + round-1 norms + router epilogue as reaching the Qwen3-30B
  ratio. Its int4 experts cost +0.0118 nats = **+0.063 ppl** against NF4
  on the same 2048-step window, over the registered 0.05-ppl uncalibrated
  gate (`experts4bit_qlora.k8_gate`). The lane table carried the family's
  0.0033-nat noise floor and no budget verdict, so the row passed unread.
  Retracted as a parity claim in `docs/STATUS.md`; the 0.32.0 throughput
  table's Granite int4 rows (same delta) are re-labelled in
  `docs/SERVING-THROUGHPUT.md` and `docs/claims.json`. Granite's licensed
  stack keeps NF4 experts (round-1 + round-2 folds + epilogue); its
  combined number is on the validation lane.

### Round-2 glue: rotary fold for attention without a head norm

- `E4B_FUSE_T1_GLUE_R2=1` now also folds the rotary chain of the
  Llama-shaped q/k/v/o attention GraniteMoe and Mixtral use (no q/k norm),
  through the kernel side's `rope_heads` (grouped-nf4-gemm >= 0.28).
  Licensed on structure — exactly the four projections and nothing of the
  module's own — so gpt-oss's attention (`sinks`) is refused, and a kernel
  cut without `rope_heads` refuses loudly rather than silently skipping.
  Lane numbers gate the merge.

## 0.33.0 — 2026-09-04

### Throughput parity across families: the build-out, measured

Six families were run under the Qwen3-30B campaign's serving protocol on
one rented RTX 5090 class (`docs/SERVING-THROUGHPUT.md`, receipt in
`bench/hybrid-g9/throughput-20260904/`, 12 claims at tier **measured**).
Every refused arm became a change; each is listed with the lane number
that gates it.

- **Round-2 layer fold licenses on structure** (#366): exactly the four
  pre-norm children and no parameters or buffers of the layer's own.
  Gemma-4's decoder body (two more norms, a routed branch, a layer
  scalar) and GraniteMoe's (scaled residuals) were being silently
  replaced by the Qwen3-shaped body. Any pre-#366 non-Qwen fused
  number is invalid; none was published.
- **GraniteMoe-shaped layer fold** (#371, grouped-nf4-gemm #328): the
  scaled-residual body folds with the kernel's `rmsnorm_resid_rows(
  scale=)` and a one-launch `scaled_resid_add_rows`, both carrying
  upstream's two bf16 roundings. The plain fold now mirrors a
  tuple-returning MoE block (gpt-oss raised `TypeError` on the lane).
  LANE (bo3, one RTX 5090): Granite K8 round-1 1.67927 → round-1 +
  round-2 1.67927 (bit-identical) vs NF4 1.67407; B=1 216 → 222 tok/s
  (×1.027); gpt-oss bit-identical too (6.38437), 132.8 → 133.3.
- **Router epilogue kinds** (#370, grouped-nf4-gemm #327):
  `topk_softmax` (select on the logits, optional bias: gpt-oss,
  GraniteMoe), `gemma4` (normed/scaled router with a per-expert scale),
  and Mixtral's renormalising router without `norm_topk_prob`; probe-
  chosen among candidates; output order by dtype; the first slot is
  the module's own (probabilities or raw logits), recorded by the probe.
  LANE: Granite +0.0009 nats, ×1.046 at B=1; gpt-oss −0.017 nats (inside
  its floor), ×1.009; Gemma-4 ×1.011 (`gemma4` kind); Mixtral −0.003 nats, ×1.01 (its
  step is expert-bandwidth-bound; int4 experts are ×2.07 there).
  Granite's licensed stack (int4 experts + round-1 norms + epilogue)
  reaches 302 tok/s at B=1 = ×1.59 over NF4, the Qwen3-30B reference's
  own ratio, and ×1.80 at B=16 (2,582 tok/s).
- **Gemma-4 MoE convention** (#369): `gemma4` / `gemma4_text` adjudicated
  pre-fused (`experts.gate_up_proj [E, 2I, H]` gate rows first,
  `down_proj [E, H, I]`), so the int4 expert lane can plan it; the
  released multimodal index maps onto the text-only tree (the
  `model.language_model.` prefix is stripped by the loader's own rule,
  the vision tower dropped deliberately and recorded). The loader's
  dedicated path is unchanged. LANE: int4 experts plan 30 layers,
  calibrated attention 115 projections; Gemma-4 int4 experts 71.7 → 86.0 tok/s at B=1 (×1.20), 574 → 797
  at B=16 (×1.39); int4 + round-1 norms + router epilogue 121.1 tok/s
  (×1.69 — above the reference's ×1.59).
- **Matched routing** (#365/#368): `--ppl-route record|replay` pins the
  router's choices across arms (consumption counters, order-agnostic
  by dtype). On Gemma-4 it removes 0.014 of a 0.21-nat remainder —
  routing is not the mechanism.
- **Refused, with its number** (#367): 2 key groups at head_dim 64 so
  Granite and gpt-oss leave the f32 attention path. Paired A/B on one
  box, twice: the released cut on the f32 path times identically to the
  fp8 path (4.10 / 5.19 ms on Granite; gpt-oss flat), for +0.0136 nats
  on Granite (4× its floor) and +0.108 on gpt-oss (6×). The power-of-two
  rounding of the group count (head_dim 96 asked for 3) is kept; the
  floor of 4 groups is restored.
- The K8 harness gains per-layer diff (`--ppl-layer-diff`), the fp8
  kernel precision model inside a one-shot forward (`--ppl-fq`), and an
  `upstream-full` oracle (#361). `--ppl-layer-diff` must not be combined
  with route replay (not inert: +0.46 nats on the same replay).

Floors: `grouped-nf4-gemm>=0.27.0` for `[fast]` and `[test]`.


## 0.32.0 — 2026-09-04

### fp8 paged KV: key scale groups per layer, 32-wide at every head_dim

`Fp8PagedKV(k_groups=None)` (the new default) sizes each layer's key
scale groups to keep 32-wide scales — 4 at head_dim 128 (unchanged), 8
at 256, 16 at 512 — when the installed grouped-nf4-gemm unrolls that
many (a capability probe of `fp8_compute_unsupported`, never a version
string), and falls back to 4 otherwise. Gemma-4's five 512-dim layers
measured 0.046 nats of fp8 cost with 128-wide groups and 0.017 with
32-wide (P27, #359). Small heads are never coarser than before. An int
still broadcasts to every layer; `kv.kgs[layer]` is the per-layer value
and `kv.k_groups` is the uniform value or `None` under mixed geometry.

Measured end to end on a rented RTX 5090 with grouped-nf4-gemm 0.26.0
(which unrolls 8 and 16 groups): Gemma-4-26B-A4B-it's paged decode on
the P26b window moves from 3.59239 to **3.57228 nats (−0.020)** with
16 groups on its five 512-dim layers and 8 on the sliding layers; the
fake-quant instrument had predicted about −0.029. Qwen3-30B-A3B (head
dim 128, groups unchanged at 4) is bit-exact at 1.61067. The K8 harness
gains `--kv-groups` (default `auto`). The `[fast]` and `[test]` floors
move to `grouped-nf4-gemm>=0.26.0`.

Also: the #344 Gemma-4 load fault did not reproduce on a third host
running driver 580.159.03 (every copy path and the bake succeeded under
`CUDA_LAUNCH_BLOCKING=1`), so the driver-version lead is refuted and the
fault stays confined to two specific, currently unrentable machines.


## 0.31.2 — 2026-09-03

### Correction: Gemma-4 has no parity reference at 512-token resolution

No code changes. 0.31.1 said Gemma-4-26B-A4B-it's paged decode was "not
at parity: 0.247 nats, three times its floor". Three windows and a
three-forward test in plain transformers (no e4b code) say something
different: the paged path is +0.093 / +0.114 / +0.247 nats from a
one-shot forward, and transformers' *own* cached forward is −0.107 /
+0.271 / +0.081 from the same one-shot forwards. The cache is
bit-exact; what moves is the model — bf16 batch-shape variance in the
expert gathers (0.2% at layer 1) that Gemma-4's router amplifies to a
0.4-nat swing on identical tokens. Qwen3 shows the mechanism at a tenth
of the amplitude and loses 0.001. So this family has no reference at
that resolution, and no parity verdict is quoted for it. What survives
as a measured, path-specific cost is the fp8 cache and dot: 0.046 nats,
on the five 512-dim layers, 0.017 with 32-wide K groups. Register:
`e4b.parity.gemma4.chunk-free` → superseded by
`e4b.parity.gemma4.no-reference`; new `e4b.parity.gemma4.fp8-share`.
METHODOLOGY §13.2 describes the three-forward test. [#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)
stays open, re-scoped to the kernel's K groups and a batch-variance-proof
instrument.

The harness gained `--ppl-fq` (the fp8 kernel's precision model inside
the one-shot forward), `--ppl-chunk`, `--ppl-layer-diff` and
`--ppl-oracle upstream-full` (#361).

## 0.31.1 — 2026-09-03

### Correction: Gemma-4 is not at parity through the paged decode path

No code changes. Against a chunk-free reference (one full forward, no
chunk boundaries) on a 512-step window, Gemma-4-26B-A4B-it's paged
decode is 0.247 nats from the model's own attention — three times a
floor (0.081 nats) that is itself five to twenty-five times any other
family's. The 0.31.0 README, `docs/STATUS.md`, `docs/SERVING-PARITY.md`
and `docs/claims.json` carried this family as "behaves" on the strength
of −0.0078 nats against the *chunked* oracle over 8192 steps; that was a
comparison with an instrument, not with the model, and it is superseded
(`e4b.parity.gemma4.behaves` → `e4b.parity.gemma4.chunk-free`). Tracked
as [#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)
with the tests in order: the paged arm with a bf16 KV cache on the same
window first, because the fp8 K-cache scale groups are 128-wide at
`head_dim` 512.

Also in this release: Qwen3-30B-A3B's chunk-free row (0.00173 nats,
floor 0.00641) — three of four families indistinguishable from their
own attention, one not — and the #344 host tally is 3 load / 2 fail.

## 0.31.0 — 2026-09-03

### Documentation release: the README says what is measured, and every number has a register entry

No code changes. This release exists so that what PyPI renders matches
the repository: the README is distilled from 494 lines of benchmark
prose (whose serving story stopped at 0.22 tok/s, v0 figures that
`docs/INFERENCE.md` itself marks superseded) to one page of what the
package is, the doors, install, quickstart, **one table of what is
measured with each row's evidence status**, the caveats that change how
the table reads, what was retired, and where the receipts are.

- `docs/claims.json` — a machine-readable register of 31 claims, each
  with value, unit, model, hardware, conditions, date, status and
  evidence path (`docs/claims-schema.md`). Every number in the README's
  measured section maps to an entry, checked mechanically.
- **`measured-private`** is a status, not a footnote: the serving
  speeds, the parity numbers and the vLLM head-to-head come from a
  private audit tree. Real runs, real receipts, not checkable from
  this repository — and now labelled as such in the README table.
- `docs/STATUS.md` — one page: what you get today, what was retired
  (each with the measurement that retired it), what is open.
- `docs/INDEX.md` — what each of the 42 documents is for and whether it
  is current; the anchored July research record is indexed as such.
- `docs/SERVING-PARITY.md` — the per-family parity table, moved out of
  the anchored `docs/support_matrix.md`, which three same-day PRs had
  appended to after its OpenTimestamps footer. The anchored file is
  restored byte-for-byte to its 2026-07-05 anchored bytes.
- The 0.30.0 corrections (below) are carried in the register as
  `retired` entries so the retractions stay findable.

## Corrections to 0.30.0 — 2026-09-03 (same day)

The 0.30.0 entry below is left as written; this records what the same
day's measurements retired. Full detail: `docs/STATUS.md`,
`docs/METHODOLOGY.md` §13–13.1, `docs/claims.json`.

- **"The fp8 paged KV cache costs +0.047 ppl on Qwen3-30B and +0.022 on
  Granite" — RETIRED.** Those deltas are +0.0058 and +0.0028 nats, and
  the models' measured arithmetic-order floors (two correct forwards
  differing only in the order of the arithmetic) are 0.0095 and 0.0033
  nats. Both deltas sit BELOW their floor: indistinguishable from
  reordering the maths, not a cost of the cache. The rule derived from
  it — "buy headroom back from the cache first" — is retired with it.
- **"gpt-oss's +0.078 nats is 10–20× every other family, a real signal
  about the sinks and sliding-window path" — RETIRED.** Against a
  chunk-free reference (one full forward, `--ppl-oracle full`) the paged
  path sits at 0.00288 nats, below its 0.01758-nat floor. The chunked
  oracle it had been compared against is 6× further from the reference
  than the path it was judging; the 8192-step gap tracked the oracle's
  32 chunk boundaries. Established on a 512-step window; a full forward
  is quadratic in the window and cannot run at 8192.
- **"Chunked teacher forcing is not equivalent to one full forward on a
  family whose layers alternate sliding and full attention" — the
  MECHANISM is retired, the measurement stands.** Widening the window
  past the context leaves the gap (KL 0.0178); every cache class
  reproduces it. The cause is MoE router flips under rounding (4.52% of
  layer-token top-k choices on gpt-oss, 6.77% on Qwen3; flipped tokens
  carry 39× the KL), which applies to every MoE model. Every parity
  delta must be read against a per-model measured floor.
- **The pre-registered KL gate in METHODOLOGY §13 is FALSIFIED** by its
  first measurement: it rejects shipped NF4 experts (0.029 nats and
  93.6% top-1 against 0.01 / 99%). Its 0.01 was calibrated from a signed
  NLL difference and applied to a full-vocabulary KL. Left textually
  unchanged and marked falsified; not retuned.
- **The serving-parity table appended to `docs/support_matrix.md` broke
  that document's OpenTimestamps anchor** (three PRs appended after the
  attestation footer). The anchored file is restored to its anchored
  bytes; the section lives in the new, unanchored
  `docs/SERVING-PARITY.md`.
- **Gemma-4's parity row reads "behaves", not PASS**: −0.0078 nats, the
  same order as the passing families, with the absolute |Δppl| ≤ 0.05
  bar inapplicable at oracle ppl 752. Its load still fails on 2 of 4
  rented hosts (#344); a 2 GiB host-hop fix was merged and reverted the
  same day because the model's largest tensor is 1.375 GiB.

## 0.30.0 — 2026-09-03

### The paged decode path is valid beyond plain-causal attention

Three changes and a floor bump. Until this release the paged B=1 decode
path (the `paged_attention` shim over `Fp8PagedKV` and the kernel
package's split-K attention) computed one attention: full causal,
scaled by `head_dim**-0.5`, one KV geometry for every layer. It served
Qwen3, Mixtral, OLMoE and Granite correctly once #336 forwarded the
attention scale; it could not serve a sliding-window family, an
attention-sink family, or a family whose KV geometry changes per layer.

- **Oracle arm** (#338). `step_decomp.py --ppl-oracle eager` scores the
  same K8 window through transformers' own eager attention with the HF
  cache, shim not registered, in 256-token chunks with explicit
  `position_ids`. It is the reference every paged verdict below is
  measured against; a family's paged perplexity must sit within the K8
  gate of its oracle.
- **Windows and sinks** (#339). The shim reads each layer's sliding
  window (`sliding_window` kwarg, else the module attribute) and
  attention sinks (`s_aux`, else `module.sinks`) and passes them to the
  kernel for decode and verify; prefill attends through SDPA, or
  through a sink-aware manual path when sinks are present. On a kernel
  wheel older than 0.24 the options are dropped with one `PARITY
  WARNING` (the K8 gate then catches the wrong attention); the fallback
  never compares the sinks tensor against a number.
- **Per-layer KV geometry** (#340). `Fp8PagedKV` accepts per-layer KV
  head counts and head dims (Gemma-4: sliding layers at 256/8 beside
  full layers at 512/2), sizes one pool at the widest row and addresses
  each layer at its natural row; the harness reads
  `config.per_layer_config` where transformers 5.16 refuses a global
  attribute.
- **Floor**: `grouped-nf4-gemm >= 0.24.0` (windows, sinks, scale and
  stride overrides in the decode kernels).
- **Instrument fixes found by running the lane**: the K8 record now
  reports the attention compute mode that RAN, from the kernel's own
  tally, instead of the environment request whose default string is
  `f32` (#348); `--ppl-chat` builds the scored window inside the
  tokenizer's chat template for chat-only families (#343);
  `--ppl-oracle upstream` scores the same window through the model as
  transformers loads it, with no e4b loader in the process (#342).
  Known limitation recorded rather than papered over: chunked
  teacher-forced scoring is NOT equivalent to one full forward on a
  family whose layers alternate sliding and full attention (gpt-oss:
  KL 0.0165 nats, top-1 93.9%), so an oracle for such a family must be
  a single full forward.

**Parity verdicts** (paged minus oracle, 8192 sha-matched steps, RTX
5090). Quoted in nats as well as perplexity, because the `|dppl| <=
0.05` bar is only meaningful where perplexity is around 8:

| Family | oracle ppl | dppl | dnats | verdict |
|---|---|---|---|---|
| Granite-3.1-3B-A800M | 7.696 | +0.0219 | +0.0028 | PASS |
| Qwen3-30B-A3B | 8.015 | +0.0467 | +0.0058 | PASS (0.003 ppl of headroom) |
| Gemma-4-26B-A4B-it | 752.5 | −5.839 | −0.0078 | behaves; the absolute bar is inapplicable at ppl 752 |
| gpt-oss-20b | 1336.0 | +108.45 | +0.0781 | no gate exists for this family |

Two things this release makes explicit. **The fp8 paged KV cache costs
+0.047 ppl on Qwen3-30B and +0.022 on Granite against a bf16 cache.**
Every earlier K8 compared paged against paged, so error the two arms
shared cancelled and that cost was invisible; attention-side and
KV-side changes are gated against `--ppl-oracle eager` from here (see
`docs/METHODOLOGY.md` section 13). **And two families have no usable
perplexity gate**: gpt-oss-20b scores 2361 on bare wikitext through
plain transformers, Gemma-4 752, so an absolute 0.05 bar means nothing
for either. A full-vocabulary KL gate is pre-registered in METHODOLOGY
13, thresholds fixed before any KL number was computed.

For gpt-oss specifically: the loader and the served expert tier are
validated faithful (MXFP4 dequant bit-identical against an independent
decode; expert forward cosine 0.991-0.993 against the reference math;
the served layer 0 inside the running model cosine 0.998 per token),
and its +0.078 nats is 10-20x every other family, which is a real
signal about the sinks and sliding-window path rather than a regime
artefact. Serve it if you want; do not read a perplexity from this
harness as evidence about it.

## 0.29.0 — 2026-09-03

### The serving stack outside Qwen: what the first five-model sweep fixed

Four changes, all from one campaign (receipts `INT4B16/P24-GEN-*`): the
same instrument as the Qwen lanes — NF4 bake, K8 perplexity on wikitext
and an out-of-domain C4 shard, graph-timed decode at B=1 and B=16, a
fusions arm, a greedy sample — run on Mixtral-8x7B, OLMoE-1B-7B,
Granite-3.1-3B-A800M, Gemma-4-26B-A4B and gpt-oss-20b.

- **The module's attention scale reaches the decode and verify kernels**
  (#336). The paged `decode` and `verify` branches passed only q/k/v
  and slots, so the fp8 decode kernel always ran at `head_dim**-0.5`.
  GraniteMoe's `attention_multiplier` is 0.015625: NF4 perplexity 4142
  and word salad through the paged loop, **7.72** with the scale threaded
  (validated on a 5090, receipts P24-GEN-E).
  Gemma-4 (scale 1.0, folded into q_norm) is the same class. Families
  whose scale *is* `head_dim**-0.5` (Qwen3-MoE, OLMoE, Mixtral) are
  unchanged. Sliding windows (Gemma-4, gpt-oss) and attention sinks
  (gpt-oss) are still not honoured: those models remain numerically
  invalid on the paged path, and the harness says so rather than
  quoting them.
- **Hessians accumulate on the CPU** (#334, with `grouped-nf4-gemm`
  0.23.0's storage-aware accumulator; the floor moves to 0.23.0).
  Mixtral's 128 attention projections at K=4096 held 8 GB of fp32
  Hessians beside a 23 GB model and ran a 32 GB card out of memory.
  Each batch's Gram is computed on the model's device and only the
  K×K result moves. With that, the Mixtral calibration completes — and
  the one-sided gate refuses the pack (+0.09 ppl on both texts, ×1.01):
  **calibrated int4 attention is a Qwen3-30B-A3B win, not a general
  lever.** OLMoE says the same (+0.60 out of domain, no speed).
- **gpt-oss per-expert biases** (#333): the hot-residency state gathered
  their rows with a CPU index against a CUDA bias; only gpt-oss carries
  them, so the branch had never run.
- **Fusion flags on non-Qwen families** (#333): `E4B_FUSE_T1_GLUE`,
  `_R2` and `_ROUTER_EPI` were consulted only inside the Qwen3-MoE
  serve assembly; elsewhere a set flag did nothing and said nothing.
  The harness now calls the three fusions directly, and each engages
  on matching modules or refuses with a sentence (validated on Granite
  and Mixtral).
- **Gemma-4 config spellings** (#335, #336): `text_config.top_k_experts`
  for the routed top-k; per-layer attributes read uniformly on
  transformers 5.16 heterogeneous configs instead of a global read that
  raises.

What the sweep established beyond the fixes: the generic loader bakes
every one of these layouts; the int4 expert lane is quality-neutral and
**doubles decode on Mixtral** (×2.04 B=1, ×1.97 B=16 over NF4) while a
1B-active model (OLMoE) pays 1.8 % perplexity for it — the K8 gate is a
per-model verdict, not a property of the lane.

## 0.28.0 — 2026-09-03

### Calibrated int4 attention for serving, and a gate that knows what calibration does

One serving lane (#331), gated behind `E4B_SERVE_ATTN_INT4_CALIB=1`.
The uncalibrated int4 attention lane was refused on quality at +0.056
perplexity, and the fp8 lane that followed showed the obvious fix is
not a fix: 4.6× lower weight error bought almost nothing, because
weight error is not what the gate measures. This lane keeps the same
grid, bytes and kernel and changes only *which* grid point each weight
lands on: `int4_attn_calib` records `H = 2·XXᵀ` for every attention
projection through forward hooks over a short calibration text, and
`Int4Linear` takes a `packer` closed over that Hessian
(`grouped-nf4-gemm` 0.22.0's `gptq_pack_int4_b32`). A projection
without a Hessian is refused, never silently packed uncalibrated under
the calibrated banner.

What it measures (RTX 5090, receipts INT4B16/P21–P22c). Calibrated on a
C4 validation shard, the pack scores **−0.042** against bf16 attention
on the wikitext gate and **−0.115** on an out-of-domain C4 text — an
improvement on both, with the same sign. Calibrated on wikitext-2
*train*, two window choices scored −0.017 and −0.078: an improvement
that moves with the calibration windows is fitting the scored text, and
that pack is refused. Speed: **×1.06 at B=1** (5.185 → 4.888 ms/step,
204.6 tok/s on that box). At batch the int4 GEMV *loses* — each row
re-streams the projection, ×0.90 at B=16 on its row axis and ×0.52
through a per-call dequant — so `Int4Linear` serves one row on the
GEMV and every larger row count on a bf16 weight dequantised once and
cached (+≈1.8 GB for 96 projections); batched decode costs exactly what
bf16 attention costs. A batched int4 attention that wins needs a
small-M int4 GEMM with weight-tile reuse; that is a kernel, not this
release.

`experts4bit_qlora.k8_gate` is the perplexity gate as one function.
Uncalibrated formats keep the symmetric `|Δ| ≤ 0.05`. Calibrated packs
are gated one-sided, `Δ ≤ +0.05`, and an improvement is trusted only
with the same sign on two scoring texts, one outside the calibration
domain; `bench/hybrid-g9/step_decomp.py --ppl-source c4val1` scores the
K8 instrument on such a text, and `ppl_source` travels in the output
beside `text_sha`. `grouped-nf4-gemm >= 0.22.0` is now the floor.

## 0.27.0 — 2026-09-02

### Glue round 3: the router epilogue, one launch per layer

One feature (#329), gated behind `E4B_FUSE_ROUTER_EPI=1`. Rounds one
and two folded the norms, the residual add and the rotary chain; the
census's largest remaining cluster at decode was what the router does
after its GEMM — a softmax over every expert, a top-k that torch
serves with a gather plus a bitonic sort, a sum and a divide. Five
launches per layer become one. The GEMM itself does not move, which is
what separates this from the router fusion refused earlier on
occupancy: a program reads 512 bytes of logits rather than the router
weight matrix.

Patching is licensed by a semantic probe against the module's own
forward, requiring both the selected expert **set** and the weights to
match the reference epilogue — a routing change is not a rounding
change. Routers that bias logits before selection, or that softmax
only the selected logits with renormalisation off, are refused by it
despite sharing every structural attribute. A test also pins the
algebraic fact that with `norm_topk_prob` on, top-k-then-softmax *is*
the reference function (the partition function cancels), so the probe
neither can nor needs to separate those.

Measured against the round-two tip: **1.0735x at B=1** (5.657 → 5.270
ms, 176.8 → 189.8 tok/s) and **1.0464x at B=16** (13.823 → 13.210 ms,
1,157.5 → 1,211.2 tok/s aggregate), engagement confirmed by kernel
census in every treated arm and absent in every control. The paired
quality gate PASSES at **-0.01968 ppl** over 8,192 sha-matched steps.

## 0.26.0 — 2026-09-01

### Glue round 2: two more decode folds, both lanes paid

One feature (#326), gated behind `E4B_FUSE_T1_GLUE_R2=1`. Where round
one fused the RMSNorm call itself, round two folds what the censuses
showed around it:

- the decoder layer's `residual + attn_out` disappears into the
  post-attention norm, one call returning both the new residual and
  the normed activation;
- each of q/k's per-head norm folds together with the rotary chain
  into a single launch per projection.

Licensing follows the round-one lesson exactly: every structural
attribute is checked before patching, the norms must pass the semantic
probe that rejects centered variants, and the attention fold only
touches an attention this package already fused — it replaces that
forward rather than half-patching an unfused one. A cos/sin tensor
that upstream broadcasts across the batch is materialised per row, and
any other layout keeps the upstream chain rather than rotating with
the wrong positions. Off decode shapes everything falls through, and a
zero-match enable refuses instead of running as a quiet no-op.

Measured on one box against the round-one tip: **1.1557x at B=1**
(6.575 -> 5.689 ms, 152.1 -> 175.8 tok/s) and **1.0916x at B=16**
(15.305 -> 14.021 ms, 1044.7 -> 1141.1 tok/s aggregate), engagement
confirmed by kernel census in every treated arm and absent in every
control. The paired quality gate PASSES at delta -0.00105 ppl over
8192 sha-matched steps.

## 0.25.0 — 2026-09-01

### Opt-in fused RMSNorm glue for single-stream decode

One feature (#324). `fuse_t1_glue(model)` — env-gated behind
`E4B_FUSE_T1_GLUE=1` and hooked from the qkv fusion pass — swaps
structural RMSNorm sites on the decode path for the kernel package's
single-launch row-parallel norm (its #306), with decode-shape and
bf16 gates. Patching is licensed per module by a semantic probe: a
deterministic probe tensor through the module's OWN forward must
match the non-centered reference formula at rtol 2^-5, which excludes
centered variants (`x_norm * (1 + w)`) that a structural name match
cannot distinguish; a vacuous enable (zero sites patched) refuses and
reports the probe-skip count. Composed on the B=1 serve lane: 8.344 →
6.469 ms per step on the rental class, quality gate PASS (Δppl
+0.0136 @ 8192 paired sha-matched steps).

## 0.24.2 — 2026-09-01

### The fused-SwiGLU wiring is retired — it never fired and would not pay

One change (#322), a removal. Census absence across every composed
receipt showed the epilogue-fusion path added in the tail-fusion round
never executed: its activation identity gate never matched the live
activation-registry object. Two dedicated A/Bs with the gate widened
then bounded the fusion's value below A/A noise at BOTH batch sizes
(1.0001× at B=1, 1.0000× at B=16, A/A ≤ 1.0005) — under graph replay
the three-launch epilogue chain is effectively free. Dead code that
measured null twice comes out rather than being re-gated a third time;
the kernel-side helper stays in the kernel package, tested and
documented as unused by this consumer.

The same measurement pass re-baselined B=1 on the released stack:
**int4 over NF4 is now 1.197×** (9.969 → 8.327 ms) — up from 1.098× at
certification, the quantise-grid fix having compounded at B=1
unannounced. The prior tail-fusion release-note attribution is
corrected accordingly: its composed 1.104× belongs to the one-launch
tile table and the gather-folded quantise alone.


## 0.24.1 — 2026-08-31

### Batched int4 decode routes through the split-K GEMV

One change (#320), no new API: decode shapes (R ≤ 256) on the int4
expert store now take the per-row split-K GEMV instead of the M-tile
GEMM. The engine probe showed the M-tile's binding constraint at B=16
top-8 routing is padded MMA lanes (~1–2 live rows per 16-row tile), not
occupancy — a split-K M-tile barely moved, while the shipped GEMV wins
gate_up 1.92× / down 1.28× and serves rows in input order, deleting the
tile table, gather, and unsort on that path. Prefill keeps the M-tile,
where tile reuse is real.

Composed on the B=16 serving step: 21.81 → **16.61 ms** (734 → 963
tok/s aggregate), A/A ≤ 1.0016, base reproduced cross-box to 0.03 ms.


## 0.24.0 — 2026-08-31

### The int4 serve lanes, the coverage matrix, and the batched-step campaign

Everything merged since 0.23.0, receipts in the audit tree:

- **Opt-in uniform-int4 expert serving** (`enable_serve_experts_int4`):
  repacks the hot expert stacks to int4-b32 and serves decode from
  them — single-stream ×1.098 with Δppl −0.084 on the certified
  family; quality gates PASS on three families (qwen3_moe −0.084,
  olmoe −0.477, qwen15moe +0.030 over 8,192 sha-matched steps).
- **Coverage via load plans**: the enabler routes through
  `plan_moe_checkpoint` and the loader's own reader/fusion helpers, so
  it inherits every family keymap and source quant format; pre-fused
  families pack off the plan's passthrough; split-K sizing follows the
  config's routed-expert count. Named refusals for what the engine
  cannot host.
- **gpt-oss on the arena path**: per-expert biases now carried
  resident and de-interleaved to the baked layout, with the residency
  gather indexed on the biases' own device.
- **The batched (B=16) campaign**: device-grouped decode routes
  through the grouped int4 GEMM; one-launch-per-side batched KV
  append; fused tile table / gathered quantise / fused SwiGLU wiring;
  a kernel census for the batched graph replay (Stage-A budget
  contract). Composed: 39.5 → 21.8 ms per step (405 → 734 tok/s
  aggregate) against the pre-campaign graph lane.
### Corrections

- **The `>275` refutation's basis is corrected; the verdict is not.**
  0.23.0 argued `>275 tok/s REFUTED-AS-COMPOSED` from "device work alone is
  8.43 ms/step". That census is an **eager-path** Self-CUDA sum (basis disclosed
  in RESULTS-sv1), and both it and the 7.25 ms graphed-default wall sit *above*
  the certified opt-in's own **6.476 ms** wall — which, since device work cannot
  exceed wall-clock on a single stream, is the correct bound for the certified
  path. (Box provenance disclosed: 8.43 is box 48728047 / anchor 7.27 ms, 6.476 is
  48709950 / anchor 7.25 ms — same class to 0.3%; and the load-bearing inequality
  is within-run on 48709950, so it does not depend on the cross-box step.) The refutation stands: 6.476 → 3.636 ms is a **1.78×** device-work
  reduction (not 2.32×), still not available from orchestration. The 0.23.0 entry
  below is left unedited so the record of what was claimed survives; the full
  reconciliation is appended to `bench/hybrid-g9/sv1/RESULTS-sv1-census.md`.
  Do not re-derive the 250 verdict from 8.43 — against the certified wall that
  frame needs 1.62×, and it stays OPEN per `RESULTS-250-closing.md` (#283).
- **RESOLVED by measurement (2026-08-26): TR2's tokens/step is 3,086.**
  Re-ran the TR1/TR2 recipe on a rented RTX 5090 (driver 595.84, transformers
  5.5.0). The trainer's own log prints `116 tok/s` at `26.6 s/step`, i.e.
  **3,086 tokens/step** — confirming the ~3,072 that had to be inferred, and
  closing the gap recorded below. Emit `tokens_per_step` explicitly anyway; a
  rate whose denominator must be back-derived from a second printed rate is not
  a receipt. Recipe validity confirmed by held-out eval reproducing TR2's
  published figures: base 1.010 -> measured **1.0093**, grouped 1.009 ->
  measured **1.0118**.
- **The 13.47x needs restating: it is ~7.2x against a current baseline.**
  Same box, same recipe, identical 321,257,472 trainable params:

      e4b base (bnb)      26.6  s/step   116 tok/s   24.36 GB   eval 1.0093
      e4b grouped          3.7  s/step   846 tok/s   24.37 GB   eval 1.0118
      -> same-box speedup 7.19x, against the registered 13.47x

  **The grouped arm did not regress** — it reproduces TR2's 3.77 s/step at 3.7.
  What moved is the BASELINE: 50.86 -> 26.6 s/step, because transformers v5
  ships `grouped_mm_experts_forward` and fused the per-expert loop upstream.
  Roughly half the published multiple is now upstream's work, not ours. Anyone
  who reruns 13.47x on a current stack gets half of it. Restate as: *7.2x over a
  current-transformers bnb baseline, same-box, with held-out eval parity.*
- **Reproducibility gap: no shipped tool bakes the arena.** `TRAIN_ARENA` takes a
  PATH to a pre-baked arena, is undocumented in `train.py --help`, and
  `nvme_arena.bake_expert_tensors` only RELOCATES pre-quantized tensors. The HF
  checkpoint ships bf16 experts, so reproducing TR2 from published artifacts
  requires writing a quantize -> emit-nf4-snapshot step yourself (~60 lines;
  15.19 GiB snapshot in 13.3 s, 16.3 GB arena bake in 7.7 s). Ship it as
  `experts4bit_qlora.bake`. Also: passing a non-path truthy value releases
  16.31 GB of expert storage BEFORE discovering the arena is missing, then dies
  on `FileNotFoundError`. Check the path exists before `_release_expert_storage`.
- **Recorded gap (now resolved above): `tokens_per_step` is not in the TR2 receipts.**
  `tr2_report.json` records `seq: 192`, `grad_accum: 4`,
  `token_budget_effective: 1024` — but not the per-step token count that the
  published "~59 → ~800 tok/s class" figure divides by. The rate is reproducible
  only by inferring ~3,000 tokens/step from the two s/step values. The s/step
  numbers (50.86 → 3.77) and the 13.47× are unaffected and remain the primary
  quantities. Recorded rather than back-filled: the receipts are committed and a
  denominator reconstructed after the fact is not a measurement. Emit
  `tokens_per_step` explicitly in the next training run.
- **Attention int4 stays fit-only** and the lm_head stays bf16 — the
  measured refusals (chain cost at B=1, occupancy at M=16, lm_head
  quality) are recorded with receipts.


## 0.23.0 — 2026-08-26

Minor release: the batched-serving certification and the honest
closure of the single-stream arc.

### Performance (serving)

- **The B>1 CUDA-graph decode loop is certified** (RESULTS-bv3b:
  PASS 3.10×): `--b1d-loop graph --batch 16` serves **419 aggregate
  tok/s** on the reference class versus 135 for the eager scheduler
  at equal compile coverage — with device-vs-eager MoE grouping
  proven BITWISE on all 48 layers, 11/16 rows token-identical, and
  the residual divergence deep inside the registered floor. Uniform
  dynamo-limit envs (`E4B_RECOMPILE_LIMIT`/`E4B_ACCUM_RECOMPILE_LIMIT`)
  ship as the comparability mechanism — equalizing compile coverage
  also made the eager baseline ~9% faster.
- **>275 tok/s single-stream is REFUTED-AS-COMPOSED** per the SV1
  pre-commitment: device work alone is 8.43 ms/step against the
  3.64 ms the target requires. The certified single-stream ladder
  closes at ~138 default / 154.4 with the dot-pad knob; the
  throughput lane is the batched loop.

### Corrections and instruments

- RESULTS-bv3's grouping-numerics attribution is corrected in
  public: the parity probe (three review rounds of vacuity binds)
  measured the paths bitwise-identical; the real confound was
  dynamo compile coverage.
- Degeneracy handling amendment: workload rows that loop identically
  in BOTH arms are excluded with disclosure (75%-clean floor;
  treatment-induced degeneration still refuses).

## 0.22.0 — 2026-08-26

Minor release: the training arc. Headline: **QLoRA training of the
30B reference recipe is 13.47× faster** — `TRAIN_ARENA` routes expert
forward/dgrad through the grouped gnf4 kernels instead of the
per-expert bitsandbytes chain, adjudicated PASS with learning
identical to the third decimal (RESULTS-tr2, receipts committed).

### Performance (training)

- **`TRAIN_ARENA=<arena>` is the documented default path for
  arena-holding models** (RESULTS-tr2: PASS). 50.86 → 3.77 s/step on
  the reference recipe (Qwen3-30B-A3B QLoRA, RTX 5090 class): ~59 →
  **~800 tok/s class**; a training epoch of the TR1 recipe drops from
  ~17 minutes of stepping to ~75 seconds. Kernel launches per step:
  2.92M → 126k (23.1×). Final held-out evals 1.010 (bnb) vs 1.009
  (grouped). Engagement releases the loader's bnb expert storage via
  shape-preserved meta twins BEFORE the tier build (the build peak
  OOMed a 32 GB card otherwise) and refuses partial engagement.
- Census instruments behind `TR1_CENSUS=1`: CUDA-event phase
  brackets in the shipped trainer loop, loss-in-receipt, effective
  token budget recorded, profiler window (CUDA-only) in its own run.
  The TR1 census that found the launch-storm (GPU ~8–11% busy on the
  bnb path) ships with receipts; its 9.1× bound is recorded as
  falsified-conservative by the TR2 receipts.

### Performance (serving)

- **Dot-pad × F2 composition certified** (SV1, K6-B PARTIAL):
  `GNF4_GEMV_DOTPAD=1` on the 0.15.1/0.21.0 defaults measures
  **154.4 tok/s** single-stream on the reference class (6.476 vs
  7.25 ms; 127 tokens identical). Knob remains opt-in, now with a
  composed receipt.
- **B>1 CUDA-graph decode loop implemented; verdict REFUSE,
  standing** (BV3): 38.0 ms/step at B=16 — a would-be 3.44× over the
  eager scheduler (421 aggregate tok/s), reproducible to 0.01 ms —
  but one row diverged from the eager stream at step 3 and the
  registered identity gate refused. Receipts attribute the
  divergence to device-vs-eager grouping numerics; BV3b (logit-parity
  probe + a kernel-swap identity frame) is registered before any
  re-adjudication, and no BV3 wall number is citable until it lands.
  The machinery ships: `--b1d-loop graph --batch B`, slot-list graph
  KV init, batched capture-safe appends.

### Instruments and gates

- The census composer's same-workload A/A gate carries a measured
  absolute-noise floor (per-step box jitter is ~250 ms regardless of
  step duration; a purely relative gate mis-refuses fast-step runs).
  All amendments disclosed in the preregs with self-tests proving the
  refusal directions survive.

## 0.21.0 — 2026-08-25

Minor release: the serving-harness half of the K/F campaign, the
speculative-verification machinery, and two default flips. With gnf4
0.15.0, single-stream decode on the reference class moves from ~66 to
**~139–140 tok/s** at defaults. Every default cites an adjudicated
verdict.

### Performance (defaults changed)

- **`--compile-layers` compiles the dense layer bodies in `default`
  mode** (F1-B1, PARTIAL ships): the 74.3 → 94.2 tok/s rung. Paged
  attention and the MoE tier stay dynamo-disabled; the disable is
  re-asserted after any shim unwrap (the wrapper-cancellation bug is
  tested against).
- **Fused T=1 KV append is the default** (F1-B2, PASS): construction-
  time kernel resolution with graceful degrade, loud refuse on an
  explicit `E4B_FUSED_KV_APPEND=1` without triton. The 94.2 → 133.4
  rung. Rollback: `E4B_FUSED_KV_APPEND=0`.
- **Fused QKV projection is the default at load** (F2 T2,
  RESULTS-f2-tail PARTIAL ships): one matmul replaces the three
  per-layer q/k/v GEMVs; +0.120 ms/step under a 0.001 ms A/A,
  token-identical over the 127-step receipt, per-projection numerics
  inside `max|ref|·2⁻⁷` on all 48 layers. Rollback: `--no-fuse-qkv`.
  NOT claimed bitwise — the prereg records the falsification of that
  claim by its own CPU gate.

### Correctness

- **Fused KV append resolves by the KV's DEVICE, not kernel
  presence.** gnf4 0.15.0 ships `fp8_kv_append_t1`, so the old
  presence-only check enabled the fused path on CPU-device KVs and
  the first append died inside triton's driver ("0 active drivers" —
  caught by this release's own CI the hour 0.15.0 hit PyPI). A
  non-cuda KV now degrades to the eager append; an explicit
  `E4B_FUSED_KV_APPEND=1` on a non-cuda KV refuses loudly. Resolution
  factored into `_resolve_fused_append` with the full cell table
  under test.

### New capabilities

- **Speculative-verification machinery** (S2-lite/S3), shipped as
  capability with a negative headline: verify-mode paged attention
  (K+1 rows against one slot with length-staggered causality),
  `rewind`/`rewind_nosync`/`seen_device` (device-truth forwardness
  under graph mode), and device-side expert grouping
  (`DEVICE_GROUPING`, capture-legal via gnf4's grouped API).
  **The S3 verdict REFUTED grouped verification at this scale**
  (0.66× vs the decode anchor at K=16–64) — and with it the 425 tok/s
  single-stream target on the registered path. The machinery ships
  because the measurement required it and future models may invert it.
- **Batch-decode curve harness** (BV2): the eager scheduler tops at
  ~124 tok/s aggregate (B=16) and is host-bound; the registered next
  lever is a B>1 CUDA-graph decode loop (not in this release).
- Census instruments: step-budget decomposition, dispatch-mode
  elementwise attribution (`--ew-attr-out`), recompile guards.

### Measurement and research artifacts

S1 acceptance-length receipts (2.9–3.9 tokens accepted at K=16/32/64,
MARGINAL), the S2 singleton bound and its correction (the bound is a
valid conservative upper bound, not a measurement of grouped verify),
S3 grouped-verify refutation, BV2 curve receipts, and the F2 arm
receipts. All under `bench/hybrid-g9/`.

## 0.20.0 — 2026-08-23

Minor release. 0.19.1 predates the hybrid tier (Phases 1–9), the cold engine's
CPU destination, the residency and scheduling work, and the correctness round
that followed them.

### Dependency floor

**`grouped-nf4-gemm>=0.14.0` is now required** by both the `[fast]` and `[test]`
extras, raised from `>=0.12.0`. 0.14.0 carries
[grouped-nf4-gemm#205](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/205):
the MXFP4 port of the grouped kernels had dropped NF4's int64 `eid` promotion,
so `eid * stride_be` overflowed signed-int32 and an MXFP4 expert stack past
2^31 bytes raised an illegal memory access. Every arena and cold path here that
stacks MXFP4 experts crosses that boundary at DeepSeek/K3-class expert counts,
and there is no degraded mode — the launch faults.

### New capabilities

- **Hybrid three-tier engine** (`engines/hybrid.py`, `engines/hybrid_train.py`)
  — VRAM / DRAM / NVMe with a placement solver and three-bus executor (#148),
  speculative prefetch that routes layer L+1 from L (#149), and QLoRA backward
  over the whole three-tier engine (#150).
- **CPU router** (`engines/cpu_router.py`) with a native epilogue (#146, #147).
- **A second execution destination for cold experts** — NVMe → DRAM → **CPU**
  (#168), with `cold_dest="deadline"` choosing against both engines' committed
  work (#179), and setup reads given their own tier so the serving tier can
  carry a direct landing (#177).
- **Paged KV and attention** — `engines/paged_kv.py`, `engines/fp8_paged_kv.py`,
  `engines/paged_attention.py`, `engines/paged_runner.py`, `engines/fp8_kv_cache.py`:
  tiered paged KV over the generalized RowPool (#152), an FP8 KV cache with a
  quality oracle (#153), and paged FP8 attention with a continuous-batching
  engine (#155).
- **Placement and scheduling** — `engines/placement.py`, `engines/scheduler.py`,
  including the batched placement law (#155).
- **Thin-layer DRAM routing** — layers below a static population threshold serve
  their DRAM experts on the GPU (#159).
- **`protected_rows` is exposed** — R1–R10 cannot be measured without it (#172).

### Performance

Measured on the paths shipped here:

- **GPU cold stacks read the cold view (#184): −5.9% at 5% cold mass, −12.1% at
  20%.** `_TieredStack.index_select` had rebuilt every routed row with
  `segment_tensor` on each call, including for experts the cold view had already
  materialized. Gate 1 attributes ~98% of cold cost to exactly that staging.
- The hybrid tier adopts the fused expert-FFN kernel — one pool wake per DRAM
  call (#156). On the full-stack re-measure the `fused_ffn` **default flips to
  `False`** (#157): it wins only with work-stealing underneath it, so the
  default follows the measurement rather than the feature.
- Direct scatter is wired into the engine, and the engine stops building one
  cold view per module (#176).

### Correctness

- **Four unread Bugbot findings addressed (#181).** `gate1_cold_sweep`'s
  `beats_both_fixed` read an *absent* fixed arm's `exposed_ns` as `+inf`, so
  "the dynamic arm beat a measurement that does not exist" **passed** the
  clause — an inversion in a module whose stated rule is the opposite. Plus two
  fallback defects and DRAM load the deadline estimator charged to nobody.
- **The direct landing and the GPU cold path cannot share a tier (#180)**, and
  #185 then lifted the direct-landing bar for GPU and mixed destinations once
  #184 removed the `row()` call that required it — the fallback and the external
  landing are now mutually exclusive by construction rather than by timing.
- `cold_stats` forwards the reuse ratio's own denominator (#182, #183).
- Deterministic per-token combine — unique `(token, slot)` writes and one
  fixed-order sum (#151).
- `cold_dest` is a rounding path, not just a bus, with matched-reference tests
  pinning it (#173, #174).

### Measurement and research artifacts (no runtime behavior change)

`bench/hybrid-g3` … `bench/hybrid-g9` and `bench/cpu-router` hold the gate
record for the hybrid campaign. Reported as measurements, not as product claims:
G8 closed at 0.51–0.68 with the full stack, with the residual identified as
intra-call (#163); G9's program best was 45.4 tok/s (#163) and its TTFT missed
by single-digit milliseconds (#161); the call-size bandwidth ramp is the
frontier while L3-warmth and spin-window mechanisms were refuted (#164); B=8
balance closed at 0.978 on reference silicon with B=16 open at 0.618–0.698
(#166, #167); and a TTFT pool-size effect was **retracted** as a first-run
compile artifact (#158).

## 0.19.1 — 2026-08-14

### The compacted stack: measured, and closed

`engines/nvme_train.py` has always named a compacted `[R, ...]` staged stack as
what would fit a big MoE on a small card, and declined it because it splits the
kernel's expert-id space from the adapter's. Its docstring now carries the
measurement that closes the question instead of leaving it as future work.

A compacted stack only saves memory when a batch routes to far fewer than `E`
experts. Counted exactly on DeepSeek-V4-Flash's hash-routed layers — expert
selection there is a frozen `tid2eid[input_ids]` lookup, so no forward and no
149 GB download is needed — over real prose: **224 of 256** distinct experts at 128
tokens, **249** at 256, and **all 256** by 512. Compaction saves 12%, 3%, then
nothing. At decode it saves **98%**.

And where it pays, it already exists: `_HotResidency._cold_contrib` takes
`torch.unique(...)` and `index_select`s only the routed rows, and `_TieredStack`
never materializes an `[E, ...]` tensor. The change has no beneficial home.

Worth naming the quantity that misleads: this is **not** the routing skew informed
hot sets exploit. Those care which experts are hit *often*; compaction cares which
are hit *at all*, and heavy frequency-skew still leaves the tail touched.

Scope: hash-routed layers only, 3 of 43 on V4-Flash. The other 40 use the learned
top-k router and need a real forward.

The probe now ships in the **sdist** (`bench/routing/distinct_experts.py`, via a new
`MANIFEST.in`) so the table above can be re-run rather than taken on trust. Wheels
are unaffected — they carry the package only, as `tools/` and `scripts/` always have.

## 0.19.0 — 2026-08-14

### The MXFP4 arena forward: projection math, not just staging

Option B (0.18.0) made an MXFP4 arena's bytes *land* correctly; it did not make the
NF4 arithmetic interpret them, and `_e4b_mxfp4_arena` only flagged the module. This
wires the compute half.

`_dequantize_expert` is overridden per instance, so the module's **own** forward
becomes correct — whichever forward that is. Every reference lane funnels through
`_project` (`ExpertsNbit.forward`, `_DeepseekV4ForwardMixin.forward`, and
`ExpertsLoRA._base_project`), so the arch's epilogue is applied by the arch's own
code rather than re-derived. `forward` additionally routes to
`mxfp4_grouped.gemm_mxfp4_grouped` under CUDA + bf16 + **no autograd graph**: that
kernel has no `autograd.Function`, so a training step routed there would produce no
`dL/dx` and silently stop learning.

Graded against the pure-torch oracle (`dequantize_mxfp4` then matmul), never
another accelerated lane, on **two cards**. Projections 0.000e+00 on an L40S; on a
3090 the down projection's GEMV branch lands at 3.344e-06, so "the projections are
exact" was a property of that card and not of the kernel. The asserted bound is
`< 2e-2`. A control grades the same forward against an *unclamped* oracle and
requires it to be rejected at ~1.0, so the fixture demonstrably tells the epilogues
apart. Receipts in `bench/mxfp4-arena-train/`.

### Two silent-wrong-answer surfaces closed

`ExpertsLoRA._use_infer_gemv` would have routed single-row MXFP4 projections through
bitsandbytes' `gemv_4bit`. Its own probe could never have caught that:
`_gemv_4bit_matches_dequant` quantizes a synthetic weight and compares bnb against
bnb, so it never reads the module's buffers.

`enable_fast`/`enable_fast_train` would have patched the NF4 grouped kernel over
MXFP4 storage — such a base passes every check they had (`quant_type="nf4"` by
class, `_apply_gate` present) and then dies on a `view(E, n1, k1 // 64)` of a buffer
holding one scale byte per 32 elements. Both refuse explicitly now.

### The dense FP8 dequantize cost 6x what its docstring claimed

`Fp8BlockLinear` said the transient was "one weight at a time (~67 MB for V4's
largest)". That counted the bf16 result only; the fp32 route also held an expanded
fp32 scale, `weight.float()` and the fp32 product — 12-14 bytes per parameter, so
~403 MB for `wq_b [32768, 1024]`. The aligned path now decodes in `dtype` with a
broadcast scale, 2 bytes per parameter, and is **bit-exact**: e4m3 -> bf16 is
lossless and an e8m0 scale is a power of two, so the multiply cannot round. Verified
including deliberately over/underflowing exponents. Ragged shapes keep the fp32
route, which the docstring now says rather than denying.

### `util.container_free_bytes()`

`enable_nvme_residency` tells callers to size `hot_rows` from measured free RAM and
the package gave them no correct way to do it in a container. Reads cgroup **v2 and
v1** — pods are v1, so a v2-only reader measures nothing there — and counts
reclaimable page cache as free, because both versions count it as *used*: straight
after a 138 GiB arena bake the naive read said **18.3 MB**.

### `grouped-nf4-gemm` floor raised to 0.12.0 — hard, not advisory

Below it, `nvme_residency._ST_TO_TORCH` has no entry for `F8_E8M0`, the tag
DeepSeek-V4 uses for its MXFP4 expert scales, so a real V4 arena cannot be staged
for training at all. No degraded mode, nothing skips.

### The 284B claim this was built for: NOT established

`bench/mxfp4-arena-train/` registers a prereg with five OTS-stamped amendments and
reports it either way. **P1 confirmed** (the unquantized control OOMs, twice, on
memory). **P2 refuted** — no rung of the registered batch ladder fits in 24 GiB.
P3/P4 ungraded; no training step completed. The expert path itself runs end to end
against a real 284B checkpoint's own MXFP4 bytes (43/43 modules patched, gates
green); what stops it is `[E, ...]` staging at 2.12 GiB/layer, which
`engines/nvme_train.py` already documents as a deliberate trade-off it declines to
make.

## 0.18.0 — 2026-08-13

### Accepts an arena whose absmax is stored bf16

`grouped-nf4-gemm` 0.11.0 can bake absmax as bf16 — 11.1% of a Qwen3-30B row down to 5.6%,
and bitwise lossless for a bf16 checkpoint, because absmax is `|w|.amax()` over a block and
is therefore one of the source magnitudes. `check_arena_geometry` refused **any** dtype
difference between the module's home and the arena segment, which rejected such an arena
outright.

The relaxation is narrow: only casts in gnf4's exported `widening_casts()` table
(bf16/fp16 → fp32) are accepted, and this **imports that table** rather than growing a
second copy that can drift. Every other mismatch still raises — "this arena was not baked
from this model" is the far more common cause of a dtype difference.

**VRAM and the kernel contract are unchanged.** The staging destination is still the
module's fp32 home, so the kernel keeps receiving the fp32 absmax it specifies. The
geometry check returns the *module's* dtype for exactly this reason: returning the arena's
would allocate a bf16 destination, `segment_into` would take its memcpy path, and the
kernel would get bf16 absmax where its contract says fp32 — wrong scales, finite numbers,
no error.

**The `grouped-nf4-gemm` floor is raised to 0.11.0**, which is where `widening_casts` and
the CPU-scaled queue depth land.

### `hot_rows` below its floor is refused at attach, and `qd` stops being pinned

**The floor is enforced now.** A stage requests every expert one forward routed and each
protects a slot from eviction, so an undersized tier raises inside `ColdTier.ensure` — but
only when a forward is finally unlucky. Measured on Qwen3-30B-A3B at seq 384, top-8: a
forward routes a **median of 63 unique experts and a max of 97 of 128**. A tier sized to
the median survives most forwards and kills the run on one of them, minutes in, after the
checkpoint has loaded and the arena is open.

`num_experts` is the worst case that request can reach and is known at attach for free, so
`enable_nvme_train_residency` now refuses below it in the pre-flight — which opens nothing
and so has nothing to unwind. The message carries the floor, what it costs in pinned RAM at
this arena's row size, and where to size it from.

Above the floor it stays a **RAM-for-disk dial**: on Qwen3-30B, 128 rows costs ~4 GB pinned
and reads 14.4 GB/step; 3216 costs ~12 GB and reads 2.65 GB/step — 3.2× the RAM for 5.4×
fewer bytes.

**`qd` now defaults to `None`.** It was `qd: int = 4` and was forwarded on every call, so
`grouped-nf4-gemm`'s CPU-scaled queue-depth default never applied to the training path —
the exact path its measurement came from. The key is now *omitted* rather than forwarded as
`None`, because on a `grouped-nf4-gemm` older than that default (the floor is 0.10.0, which
predates it) `qd=None` would reach `ThreadPoolExecutor(max_workers=None)` and silently open
up to 32 reader threads instead of 4.

## 0.17.5 — 2026-08-13

**Docs-only. A third model corrects what two models got wrong, and the arena's cost in
TIME is measured for the first time.**

### The arena requirement is not flat

0.17.4 said the host requirement scales with expert bytes "while the arena requirement is
set by `hot_rows` and stays roughly flat". The host half holds. **The flat half is wrong.**
Gemma-4-26B-A4B — now bakeable, see below — lands between the other two and breaks it:

| | expert bytes | arena req | host req | ratio | ⇒ dense baseline |
|---|---|---|---|---|---|
| OLMoE-1B-7B | 3.62 GB | 2.28–2.42 GB | 5.91–6.17 GB | 2.56× | ~2.19 GB |
| **Gemma-4-26B-A4B** | **12.85 GB** | **5.10–5.37 GB** | **19.33–20.40 GB** | **3.80×** | **~4.94 GB** |
| Qwen3-30B-A3B | 16.31 GB | 3.89–4.03 GB | 24.70–25.77 GB | 6.40× | ~3.69 GB |

Gemma has **fewer** expert bytes than Qwen3 and a **larger** arena requirement, because its
dense side is bigger — a dense MLP in every layer plus a 262144-token vocabulary. The ratio
is expert bytes measured **against the dense baseline**, not expert bytes alone. Two points
were consistent with "flat"; three are not.

### What the arena costs in time

First timing measurement in this line, on two architectures because
[no timing claim ships on one machine](bench/host-ram-ceiling/RESULTS-timing.md):

| | RTX 3090 (Ampere) | L40S (Ada) | travels? |
|---|---|---|---|
| **load**, host/arena | 3.39× / 6.76× | 3.12× / 5.87× | **yes** |
| **step**, arena/host | 1.331 / 1.238 | **2.248 / 1.708** | **no** |

**The load saving travels; the step cost does not, and it is worse on faster hardware.**
Going 3090 → L40S the host arm's step time nearly halves while the arena arm improves only
1.2–1.4×, because part of every arena step is an NVMe read and the disk does not care which
GPU you bought. **Quote the step cost with the card attached, or not at all.**

**Size `hot_rows` to the routing floor, not to free RAM.** Sweeping 128 / 384 / 1024 on
Qwen3 produced no resolvable step-time difference, while 1024 cost resolvably more load
time and ~8× the pinned RAM. `hot_rows` also does not travel between models: it has a hard
floor at the experts one forward routes (OLMoE 89% of a layer, Qwen3 52%, Gemma-4 50%).

Both receipts carry their pre-registrations, including a mis-specified gate that was
amended before any re-run, and a `hot_rows` U-shape that one round suggested and three
rounds withdrew.

## 0.17.4 — 2026-08-13

**Docs-only. The scaling claim 0.17.3 flagged as unmeasured is now measured.**

0.17.3 said the ratio "should widen substantially on larger MoEs — but that is the
mechanism's prediction, not a measurement". On **Qwen3-30B-A3B**, which has 6144 experts
against OLMoE's 1024 and **4.50× the expert bytes**, it is **6.40×**. Receipt, raw ledger
and the pre-registration it is scored against:
[`bench/host-ram-ceiling/RESULTS-scaling.md`](bench/host-ram-ceiling/RESULTS-scaling.md).

| | expert bytes | host-RAM path | arena path | ratio |
|---|---|---|---|---|
| OLMoE-1B-7B | 3.62 GB | 5.91–6.17 GB | 2.28–2.42 GB | 2.56× |
| **Qwen3-30B-A3B** | **16.31 GB** | **24.70–25.77 GB** | **3.89–4.03 GB** | **6.40×** |

**At an 8.59 GB ceiling Qwen3-30B-A3B is OOM-killed on the host-RAM path and trains to
completion on the arena.** The host requirement grew **×4.18** against **×4.50** in expert
bytes — what "pins every expert" predicts — while the arena requirement grew only ×1.66,
and most of *that* is the larger dense side (48 layers, 151936-token vocab), not the
expert path.

**A hot row costs 1.58–1.98× the bytes it holds.** Re-running the whole ladder at
`hot_rows=512` puts the marginal cost at **4.19–5.24 MB per row** against a 2.654 MB
on-disk row. At `hot_rows=128` the expert path is therefore only ~0.5–0.7 GB of the
~3.9 GB requirement; the rest is fixed base. The cause of the per-slot overhead is not
isolated and no mechanism is offered for it.

**`hot_rows` does not travel between models.** `hot_rows=64` — correct for OLMoE — refuses
on Qwen3 with `request of 97 unique rows exceeds hot_rows=64`. That is the documented
behaviour working: the docstring already specifies a floor of `min(T*k, num_experts)`,
which is 128 here. OLMoE has exactly 64 experts per layer, so its value was silently at
the floor already and looked portable. Size from the formula, not from a previous run.

The pre-registration was committed before the checkpoint was downloaded and **all three of
its point predictions missed** — host 18–21 GB (actual 24.70–25.77), arena 2.2–3.0 GB
(actual 3.89–4.03), ratio 7–9× (actual 6.40×). Direction right, magnitudes wrong, because
both arms were sized from OLMoE's much smaller non-expert baseline. Its stop rule fired at
4.08 GB and the ratio is reported only after the investigation it demanded.

No code changed; the wheel is byte-identical to 0.17.3 apart from the version.

## 0.17.3 — 2026-08-13

**Docs-only. The case this feature exists for is now demonstrated instead of asserted.**

Every release through 0.17.2 described `enable_nvme_train_residency` as the answer to
"the experts do not fit host RAM", and every release through 0.17.2 admitted it had
never shown a model that could not be trained without it. It can now be shown, with a
number on both sides. Full receipt, raw ledger and drivers:
[`bench/host-ram-ceiling/`](bench/host-ram-ceiling/RESULTS-host-ram-ceiling.md).

**At a 5 GiB host-RAM ceiling — same model, same seed, same four steps, same box — the
host-RAM path is OOM-killed and the arena path trains to completion.** Descending the cap
until each arm stops completing brackets both requirements:

| | host-RAM offload | NVMe arena, `hot_rows=64` |
|---|---|---|
| **minimum host RAM to train** | **5.91–6.17 GB** | **2.28–2.42 GB** |
| frozen experts | all 1024 pinned (3.83 GB of homes, per 0.17.0) | ~0.2 GB pinned, 64 hot rows |
| steady RSS at `trained` | 5.88 GB | 2.34 GB |

**The saving is ~2.5×, not 2.5–3.5×.** 0.17.2's upper bound came from pairing the lowest
host sample against the lowest arena sample across *different* runs. Measured as one
quantity — the smallest ceiling in which four steps complete — it is **2.56×**, bracketed
2.44×–2.71× by the rungs either side, and steady RSS agrees independently at 2.51×. The
0.17.2 entry is left as published; this supersedes it.

**Peak RSS overstates the host arm by 2.7× — now shown causally.** That arm peaks at
16.63 GB uncapped (15.86 GB of it file-backed) and trains fine under a 6.17 GB cap, with
nothing tuned between the two runs: the kernel reclaims the mmap'd bf16 checkpoint when
RAM is scarce. The arena arm, having almost no page cache to drop, has a peak RSS that
*does* predict its threshold. Hence peak-RSS ratio **7.10×** against requirement ratio
**2.56×** — and hence the earlier 8× figure, which was this artifact.

Why it took a home box rather than a rented one: the cap has to include **swap**. A rented
container's cgroup is read-only and the kernels seen there had no `memsw` accounting, so
an over-limit process pages out and survives — a different outcome from fitting, reported
as success. `docker --memory=N --memory-swap=N` sets both limits, verified by reading them
from inside. The cap was positive-controlled in both directions before any arm ran (900 MB
under 512m → killed; the same allocation under 2g → completes).

**0.17.2 did not actually do what it says it did, and this fixes that too.** It was
titled "put the host-RAM number on the page that serves it" and put the number in
`CHANGELOG.md` — but the PyPI long description is built from **`README.md` alone**, so
no changelog text has ever appeared on the package page. The 0.17.2 page contains no
`3.83`, no `ru_maxrss`, no `steady RSS`. The release was verified by confirming it
published, not by reading the page it was for. The measured summary now sits in the
README, where the long description will carry it.

No code changed; the wheel is byte-identical to 0.17.2 apart from the version.

## 0.17.2 — 2026-08-13

**Docs-only. The published page still lacked the one number the feature is for.**

0.17.0/0.17.1 describe `enable_nvme_train_residency` as lifting the **host-RAM**
ceiling and never said by how much. The 0.17.0 entry below now carries it —
**~2.5–3.5×**, as pinned expert bytes (3.83 GB → ~0.2 GB) and steady RSS after
load (4.94–5.9 GB → 1.42–2.37 GB) — together with the reason the obvious
instruments give the wrong answer.

`ru_maxrss` is not usable here: it reports 18.6 GB for the host arm on a roomy box
and 10.8 GB for the *same arm* on a constrained one, because the 13.84 GB bf16
checkpoint is mmap'd and read in full to fuse and quantize, and those pages are
clean, file-backed and reclaimable. Peak *anonymous* is not the fix either —
~1.5 GB for **both** arms, since `smaps_rollup` counts pinned CUDA memory as
file-backed.

Established by five reproductions across two independently built stacks on one
pod whose unpinned install resolved to identical ML package versions, plus an
A/B/A memory-balloon test (18.57 → 10.80 → 18.56, bit-identical losses) that rules
out drift. No code changed; the wheel is byte-identical to 0.17.1 apart from the
version.

## 0.17.1 — 2026-08-12

**Docs-only. The published 0.17.0 page carried a timing number that was never a
measurement.**

0.17.0's release notes reported `s/step` as one measurement per arm, and from
those single samples claimed the fully-pinned arena cost **1.03×** the host-RAM
reference. Re-measured under this repo's paired protocol — 5 scored rounds, every
arm timed once per round in fixed order, warmup dropped, plus a `host_self`
control that times the *same* host-RAM model twice per round — the control came
back at **0.986 with a 0.898–1.080 spread**. That spread is the harness's
resolution limit, and `hot_rows=1024`'s 0.957 sits *inside* it.

So the honest statement is **"indistinguishable from host RAM"**, not 1.03×. The
disk-bound arms are unaffected and remain real: **1.679×** at the `hot_rows`
floor, **1.479×** at 256 — both far outside the noise, with the ladder moving as
the tier's additive law predicts.

The warmup round is the argument for the protocol: `host` and `host_self` read
3.084 vs 1.901 in round 0 — the identical model, 62% apart. No one-shot-per-arm
run can see that.

The full table, the control, and what still is **not** established (a model whose
experts genuinely exceed host RAM) are in the 0.17.0 entry below, now corrected.
No code changed; the wheel is byte-identical to 0.17.0 apart from the version.

## 0.17.0 — 2026-08-12

**Training whose frozen experts live on NVMe — plus the namespace split, and a CI
gap that had been reporting 42 tests as coverage while running none of them.**

- **`enable_nvme_train_residency(model, arena_path, hot_rows=…)`** — QLoRA on a
  MoE whose frozen experts exceed **host RAM**. The arena served inference and
  refused training in as many words (*"Load without `arena=` to train, or drop the
  adapters to serve"*); that refusal was right, since the serving engines replace
  the module's forward and would discard the adapter's delta. This moves the
  **home** instead: `_ArenaExpertOffload` is an offload handle whose homes are
  `meta` — shape and dtype, no storage — and whose rows come off the arena:
  disk row → ColdTier pinned slot → `[E, …]` device stack → kernel.

  `enable_fast_train` is untouched. `nf4_qlora`'s `weights_fn` closure already
  re-reads whatever is staged when backward runs, which is the seam that makes
  this work at all.

  **Gradient checkpointing is required, and enforced.** The evict hook fires when
  a forward returns, so the checkpoint recompute is what re-stages a layer for its
  own backward. Routed staging fills only the routed rows of a full-shaped stack,
  so a recompute that routed differently would read uninitialized memory —
  `assert_rows_staged` runs inside the weights_fn closures (i.e. **at backward**)
  and refuses instead.

  **It does not bound VRAM.** The staged stack keeps its full `[E, …]` shape so
  every consumer still indexes by global expert id; one layer is device-resident,
  as with ordinary offload. This lifts the host-RAM ceiling, not the VRAM one.

  Needs `grouped-nf4-gemm >= 0.9.0` for `nvme_residency.segment_into`.

- **Fixes a latent bug in the shared single-resident-layer policy.** The slot was
  written through `type(self)`, so a subclass bound a *second* slot and left the
  base class pointing at a handle nothing would evict — two layers resident at
  once, the exact bound that policy exists to hold. Unreachable with one handle
  class; reachable the moment there are two.

- **`enable_nvme_residency` validates before allocating now.** It built its
  `ColdTier` first, which made every refusal unreachable on a host without an
  accelerator: the tier pins its landing buffer, so a CPU-only machine raised
  "Cannot access accelerator device" and the caller got an allocator error instead
  of the message naming their mistake. On a host that does pin, a refusal leaked
  the tier. The error path also no longer closes a tier that modules are already
  serving from, and counts what *this call* attached rather than trusting a sticky
  `_e4b_hot_ref` marker that survives an earlier enable.

- **CI actually runs the arena tests now.** `.[test]` did not install
  grouped-nf4-gemm, so on the runner `tests/test_nvme_train_residency.py` went
  from 29 tests to **one skip**, and `tests/test_nvme_residency_equivalence.py`
  did the same — both reporting as coverage while executing nothing. Adding the
  dependency took the runner from **577 to 661 passing**, and immediately
  surfaced the `enable_nvme_residency` bug above, in an engine that shipped
  months earlier.

- **The README link check stopped calling throttling a dead link**, and no longer
  forwards `Authorization` across origins. It opened a fresh TLS connection per
  link and GitHub's edge dropped some of that churn; one run reported 28 of 28
  links dead on a tree where every path existed. Connection reuse fixed it
  (measured: a URL that failed four `urlopen` attempts answered 200 three times
  running under `curl`). 404 and 403 are never retried into a pass.

Also shipping here, from the preceding commits: the `arch/` `formats/` `engines/`
namespace split (old submodule paths still resolve via aliases in `__init__`), the
architecture support matrix, transformers checkpoint-key renamings, and the
gap-heuristic fix.

- **`arena_train=True` on the loader — without it the above could not be reached
  at all.** The arena branch built bare `meta` experts (its serving shape) and
  silently ignored `r`/`alpha`, so `enable_nvme_train_residency` refused every
  module with *"not ExpertsLoRA-wrapped"* and its own documented usage failed.
  29 CPU tests missed it because each constructs `ExpertsLoRA` by hand in the
  fixture — they exercised the mechanism, never the route a caller takes. It is
  gated on an explicit flag rather than on `r`, because `r` is a required
  positional and the *serving* example passes `r=8`; keying off it would have
  fixed training by breaking serving.

**Verified on a GPU** (RTX A5000, sm_86; OLMoE-1B-7B; 12 steps on Alpaca;
identical data and bit-identical starting adapters; every arm through
`enable_fast_train`, so only residency differs):

| arm | s/step | peak GB | final loss | med \|ΔL\| |
|---|---|---|---|---|
| host RAM (reference) | 1.62 | 2.26 | 0.5839 | — |
| arena, `hot_rows=64` | 2.65 | 2.03 | 0.6143 | **0.0059** |
| arena, `hot_rows=256` | 2.41 | 2.04 | 0.6426 | **0.0086** |
| arena, `hot_rows=1024` | 1.67 | 2.04 | 0.5944 | **0.0099** |

All three pass `bench/fused-train-gate`'s registered 0.05 median-|ΔL| band, 5–8×
inside it. A precondition run first established the arena's bytes are **bitwise
identical** to loader-quantized bytes, so the arms differ only in residency.

**Timing re-measured under the paired protocol** (5 scored rounds, every arm timed
once per round in fixed order so drift hits all arms equally; warmup round
dropped; optimizer not stepped so routing is identical every round). The
`s/step` column above was one measurement per arm and is superseded by this:

| arm | s/step (med) | ratio med | ratio min–max |
|---|---|---|---|
| host RAM (reference) | 1.910 | 1.000 | — |
| **`host_self` (control)** | 1.906 | **0.986** | 0.898–1.080 |
| arena, `hot_rows=64` | 3.207 | **1.679** | 1.490–1.724 |
| arena, `hot_rows=256` | 2.919 | **1.479** | 1.351–1.542 |
| arena, `hot_rows=1024` | 1.867 | **0.957** | 0.890–1.051 |

`host_self` is the *same* host-RAM model timed a second time in each round. At
0.986 the instrument is unbiased, and its 0.898–1.080 spread is the **resolution
limit** — which is the number that makes the rest readable:

- The two disk-bound arms sit far outside it, so **1.68× at the `hot_rows` floor
  and 1.48× at 256 are real**, and the ladder moves the way the tier's additive
  law predicts: the cost is disk traffic.
- **`hot_rows=1024` is indistinguishable from host RAM.** Its 0.957 sits inside
  the control's own spread, so the honest claim is *smaller than this harness can
  resolve* — not the "1.03×" a single measurement suggested.

The warmup round is why this matters: `host` and `host_self` read 3.084 vs 1.901
in round 0 — the identical model, 62% apart. A one-shot-per-arm run cannot see
that.

**How much host RAM this actually saves: ~2.5–3.5×.** The point of the feature is
the host-RAM ceiling, so here is the measurement, with the caveat that makes it
readable. Same OLMoE run, RTX A5000:

| | host-RAM offload | arena, `hot_rows=64` |
|---|---|---|
| **pinned expert bytes** | **3.83 GB** (all 1024 experts) | **~0.2 GB** (64 hot rows) |
| steady RSS after load | 4.94–5.9 GB | 1.42–2.37 GB |

**Do not measure this with `ru_maxrss`.** It reports 18.6 GB for the host arm on a
roomy box and 10.8 GB for the same arm on a constrained one — an A/B/A with a
memory balloon moved it 18.57 → 10.80 → 18.56 GB with bit-identical losses and an
unchanged unreclaimable footprint. The checkpoint is 13.84 GB of bf16
safetensors, mmap'd and read in full to fuse and quantize; those pages are clean,
file-backed and reclaimable, so the "peak" is page cache the process happened to
have mapped, not memory it needed. An earlier 8× figure came from comparing that
inflated host peak against an arena arm that never reads the expert bytes at all.

Peak *anonymous* memory is not the fix either — it is ~1.5 GB for **both** arms,
because `smaps_rollup` counts pinned CUDA memory as file-backed. Use steady-state
RSS after load, or peak of anon + `/dev/zero` mappings.

**What this still does NOT establish.** OLMoE's arena is 3.6 GB and fits
everywhere, so this shows the mechanism is correct and how the cost scales with
residency — **not** a model whose experts exceed host RAM, which is the case the
tier exists for. A demonstration of that needs a machine capped near the host
arm's true ~5–6 GB working set; attempts at 11–16 GB could not fail the host arm,
because it never needed 18 GB.

> **Superseded 2026-08-13 in 0.17.3 — demonstrated.** The prediction in the
> paragraph above was published before the measurement and held: the host arm's
> requirement is **5.91–6.17 GB**. At a 5 GiB cap it is OOM-killed while the arena
> arm trains. The ratio here, **~2.5–3.5×**, is refined to **2.56×** by measuring
> one quantity rather than pairing extremes across runs. See
> [`bench/host-ram-ceiling/`](bench/host-ram-ceiling/RESULTS-host-ram-ceiling.md). Rented-instance NVMe varies ~7× between pods, so
these ratios characterise this box and do not travel.

## 0.16.3 — 2026-08-12

**Makes 0.16.2's headline feature actually importable, and turns one opaque load
failure into an accurate refusal.**

- **`capture_decode`, `CapturedDecoder` and `probe_capture` are exported from the
  package root.** 0.16.2 shipped `capture.py` with no top-level export, no in-repo
  caller and no README mention, so the only route to it was knowing the private module
  path — the feature was published unreachable. Documented under Inference with the
  measured numbers (4.4–5.6x on 2-layer fixtures, **1.11x** on OLMoE-1B-7B, **1.04x**
  on Qwen3-30B-A3B) and both costs: `StaticCache` is allocated to `max_length` up
  front, and `step()` is greedy argmax with no logits processors, stopping criteria or
  streamer.

  Deliberately **not** used by the HTTP server: `_generate_once` needs sampling,
  repetition penalty, stop signals and streaming, and reimplementing those on a
  captured step to gain the measured ~4% at 30B is a bad trade against the risk.

- **Identity ("zero-computation") experts are refused with the counts named.**
  longcat_flash previously died with `AttributeError: ExpertsLoRA has no attribute
  '10'`, which names neither the architecture nor the limitation. LongCat-Flash
  allocates `gate_up_proj` over `n_routed_experts + zero_expert_num` (512 + 256 by
  default) but `down_proj` over the routed count only — its forward sends
  `expert_idx >= num_routed_experts` through `nn.Identity` scaled by the router weight
  and never reads those `gate_up` rows, so the surplus experts are ragged on disk by
  construction. The per-expert reader consumed `0..n_routed-1`, orphaned the rest, and
  the generic weight walk then called `get_submodule(".../experts.10")` on a path whose
  leaf was already the fused module.

  Loading only the routed experts is not a fix: the router keeps selecting over the full
  space, so it would address experts that do not exist. An identity slot belongs in the
  expert primitive, not the loader.

## 0.16.2 — 2026-08-12

**CUDA-graph decode capture, and one less allocation per decode step.**

- **`capture_decode()` / `probe_capture()`** (`experts4bit_qlora.capture`). Wraps one decode
  step in a `torch.cuda.CUDAGraph` backed by a `StaticCache`, so every step has identical
  shapes and ONE graph serves the whole generation — a growing KV cache otherwise gives a
  distinct shape, and a distinct graph, per token. The cost is explicit: the cache is
  allocated to `max_length` up front. `torch.compile` cannot be used instead; inductor dies
  on `aot_autograd() does not yet handle input mutations on views with different dtypes`,
  which is exactly the engine's one-uint8-store-viewed-as-int64-and-float32 row block.
  Capture also throws on a host sync inside the region, so a successful capture doubles as
  a check on the zero-sync decode contract.

  Measured, 16 new tokens greedy: **4.4–5.6x** on 2-layer fixtures (qwen2_moe, qwen3_moe,
  granitemoe, hunyuan_v1_moe, glm4_moe, dots1, olmoe), **1.11x** on OLMoE-1B-7B-0924-Instruct
  (3090) and **1.04x** on Qwen3-30B-A3B (A5000). The speedup is inversely proportional to
  real GPU work per step, which is what a fixed per-step launch cost predicts — so this is
  worth having for small models and for the sync contract, not as a throughput claim at
  scale. Both real-weight models replay **bit-identical** to eager decode.

  `probe_capture()` reports support rather than assuming it, and distinguishes a bf16 argmax
  tie from a real defect by measurement: it teacher-forces the same tokens down both paths
  and compares logits. The reference is eager INCREMENTAL decode against a cache — comparing
  against one full-sequence forward charges a few ulp of kernel/reduction-order difference to
  capture. `qwen3_next` is not capturable: `StaticCache` does not cover LinearAttention.

- **Persistent `row_idx` buffer in the pipelined engine** — the per-step device allocation
  and H2D copy are gone. **-16.4%** host time per decode step.

## 0.16.1 — 2026-08-12

**Correctness fix for the segmented cold source, plus per-round overhead removed.**

- **Prime each segment from its own offset.** `seg_addr` pointed every hot lane at the
  resident row START for all four segments instead of `hot_row + off[j]`. `_prime` seeds
  every slot from expert 0 with `have = -1`, so it does NOT take the hot skip — with
  expert 0 hot on the segmented (offload-homes) path it primed the absmax and down
  regions with gate_up bytes. Latent in 0.16.0: `_fetch` forces hot lanes to skip and
  they read the resident row in place, so nothing read those bytes — but that was an
  undocumented invariant holding up a wrong address table, and nothing tested it.
- **Traffic counting is now opt-in** (`E4B_PIPELINED_TRAFFIC=1`, or `count_traffic = True`
  on the engine before the first fetch). The two device reductions cost ~8.9% of the
  decode step on an A5000 to produce numbers nothing reads in production.
  `traffic()` RAISES when counting was off rather than reporting zeros, because
  `hot_d2d_bytes == 0` is a regression witness and several tests use the counters to
  prove the engine ran at all — silent zeros would let those pass while measuring nothing.

## 0.16.0 — 2026-08-12

- **Residency reads the offload homes in place** (#104, closes #86 with #87).
  Under offload the homes already hold every expert in pinned host RAM and the
  pipelined engine baked a SECOND full-size arena from them — 0.316 GiB per layer
  twice over on Qwen3-30B-A3B geometry, ~15 GiB duplicated for the one
  configuration that exists to fit a big model on a small card. The homes cannot
  be freed (prefill, grad and odd-dtype forwards still fall back to the reference
  path and need staging), so the engine stops making the copy instead.

  The homes group by tensor and the row layout groups by expert, so an expert is
  four contiguous runs and the gather issues one launch per segment. The kernel
  gained a destination offset, a length, and a separate IDENTITY vector — four
  launches read four addresses but must skip or copy together as one expert.
  Measured: RSS delta on enable 0.579 → 0.080 GiB per layer, output bit-identical
  to the copied-arena control.

  Guarded rather than assumed: offload packs homes one buffer per DTYPE with the
  offset advancing in ELEMENTS, so an odd-numel predecessor leaves the next tensor
  misaligned — undefined behaviour where the gather casts to `int64*`. Each
  segment is checked for pinned + contiguous + 8-byte-aligned base and
  8-byte-divisible length, falling back to the copied arena otherwise.
  Non-offloaded modules are unaffected.

## 0.15.0 — 2026-08-11

**Five more quantized checkpoint formats, and the DFlash drafter load path.**

- **AWQ** (`awq.py`) — the first ASYMMETRIC format, using autoawq's exact
  `[0,4,1,5,2,6,3,7]` nibble order. Packed along OUT.
- **GPTQ** (`gptq.py`) — packed along IN, sequential order, `+1` zero offset.
  Told apart from AWQ by its `g_idx` sibling; AWQ had been silently claiming all
  18624 GPTQ tensors, which is a wrong-answer bug, not a load failure. `g_idx` is
  now range- and length-validated (a negative index would wrap to the last group).
- **compressed-tensors int4** (`compressed_int.py`) — llm-compressor / vLLM.
  `num_bits`/`group_size` are DERIVED from shapes because the config often omits
  them. Vectorized unpack.
- **NVFP4** (`nvfp4.py`) — E2M1 codebook with two-level scaling; also serves
  **NVIDIA ModelOpt FP4**, verified against modelopt itself.
- **DFlash drafter** (`glimmer_draft.py`, `speculative.py`) — drafter load for both
  released spellings with coverage reconciled, plus a greedy speculative loop that
  is token-identical to plain greedy.

The dispatch matrix is pinned in BOTH directions, so a format is claimed by exactly
one decoder.

## 0.14.0 — 2026-08-10

**MoE breadth: the convention system, and every execution config measured.**

- **12 adjudicated conventions** covering 45 `model_type`s, each checked against
  transformers' own converter table so coverage DRIFT fails a test rather than
  silently going stale. Gate/up are shape-identical, so orientation can never be
  inferred — every entry is adjudicated, not guessed.
- New families: `gpt_oss` (pre-fused MXFP4 through the generic planner),
  `qwen3_vl_moe` (pre-fused + load-time transpose), `dbrx` and `jetmoe` (flat
  native stacks, bit-identical passthrough), `qwen3_5_moe` (native passthrough),
  `nemotron_h` (NON-gated: stack up/down, no gate to fuse), `minimax_m3_vl`
  (VL-prefixed mixtral), `axk1` (hybrid dense/MoE, layer-conditional keymap).
- **block-FP8 routing**, and MTP heads are never dropped silently.
- Tied heads are tied even when the checkpoint also ships the head.
- Rotary dim/theta buffers are materialized for VL vision towers.
- **Every execution config measured**, not just dtype: the decode/prefill ranking
  inverts, gains shrink as experts widen, and `dgrad=True` is the fastest training
  lane.


## 0.13.0 — 2026-08-10

**Muse Glimmer (Meta) and GLM-5 (Zhipu) checkpoint support.**

- **`glimmer.py` + `glimmer_load.py`** — serve Muse-Glimmer-30B from a released
  GGUF text tower. Glimmer is DENSE (Gemma-3 lineage), so it uses the dense lanes,
  not the expert path. Weights are decoded through grouped-nf4-gemm's k-quant lane
  (**needs gnf4 >= 0.8.0**; the import is capability-gated, so older installs get an
  actionable message rather than an AttributeError). The load streams each tensor to
  the target device as it decodes — peak host RAM is one tensor, not the ~60 GB a
  dequantized 30B would need — and ends with a coverage reconciliation: every
  text-tower parameter must be materialized or it raises.
- **`glm5.py`** — GLM-5 (`glm_moe_dsa`) checkpoint keymap and expert fusion.
  DeepSeek-V3 lineage, so it reuses the existing MLA/per-expert machinery; the new
  surface is DSA's lightning indexer.

Every mapping in both was adjudicated against the real released checkpoint AND the
instantiated transformers module tree, then reverse-armed (every model parameter must
be claimed by some checkpoint key — the direction that catches a silently dropped
weight). The traps that arithmetic alone gets wrong, now asserted:

- Glimmer's head is **untied** and must never be aliased to the embedding.
- Its four per-layer norms are centered (`x*(1+w)`) with the `+1` baked into the GGUF
  bytes, so the parameter is `gguf - 1.0` — while the FINAL norm is used as-is.
- Its `attn_q/k_norm` are uniform vectors equal to `config.qk_scale_factor` and 1.0,
  absorbed by a parameter-free norm; dropped only after asserting that identity, so a
  genuinely learned qk-norm fails loudly.
- GLM-5's checkpoint carries **one more layer than the model builds** (an MTP head);
  it is skipped explicitly, and MTP markers on a built layer raise.
- GLM-5's experts are per-expert on disk and fused in the tree, with gate/up
  concatenated as **blocks** — the interleave convention would mis-activate with every
  shape agreeing.
- Rotary `inv_freq` is computed, never shipped; it is rebuilt through the module's own
  rope initializer instead of being left on `meta`.

Validated end-to-end on an A100 80GB: the 30B loaded from Meta's released
`kquant-dynamic` GGUF (19.65 GB) in 205 s — 627 tensors assigned, 104 dropped,
0 unfilled — and generated correct text at 13.8 tok/s, 55.8 GB VRAM.

## 0.12.0 — 2026-08-06

**`serve` grows a residency dial** — the missing piece between 0.10.0's
residency-reachability fix and the deployment that needed it: `python -m
experts4bit_qlora.serve` could only stream every expert, which is why the judge
deployment decodes at 0.38 tok/s with VRAM sitting idle.

`E4B_RESIDENCY=pipelined` + `E4B_HOT_PROFILE=<jsonl>` + `E4B_HOT_PER_LAYER=<K>` attaches
`enable_pipelined_residency` after load (post-`eval()`, pre-warmup — the ordering the
wrapper's delegation preconditions require). Hot sets are **frequency-ranked from a
profile, never by index** — an index-ordered set on a 256-expert top-6 layer serves ~6%
of routed slots, so there is deliberately no by-index fallback; without a profile it
raises. `E4B_K_SLOTS` overrides the routed top-k when the config lacks
`num_experts_per_tok`. `/health` gains a `residency` block (mode, patched-module count,
profile-predicted coverage).

`E4B_EXPERT_PROFILE` now works under serve: the routing profiler was only ever attached
by `train.py`, so profiling a *serving* workload — the input the residency dial consumes —
silently wrote nothing. serve attaches it at load (no-op unless set); the JSONL lands once
at clean shutdown.

Every quiet failure mode refuses or warns instead: unknown mode, missing/most-wrong
profile (a set-count/module-count mismatch would silently shift every hot set one layer),
an engine that patches 0 modules ("residency on" in the logs, streaming in reality), and
— the subtle one — **activating a trained adapter turns residency off silently** (the
wrapper only delegates while the adapter is provably zero), so `_swap_adapter` warns when
that happens. Serving `base` (the judge/eval case) is what the dial is for.

Also: `test_reenable_with_a_different_dgrad_setting_says_so` gains a capability skip —
against a pre-0.7.0 grouped-nf4-gemm the dgrad mismatch it tests cannot be constructed
(the flag is coerced off first, with its own warning), which surfaced as a spurious
failure on any box with an old wheel installed.

## 0.11.1 — 2026-08-06

Docs-and-tests patch. The PyPI page for 0.11.0 froze training-path guidance that
same-day measurement falsified; this ships the corrected long_description plus the
receipts and one new test. No functional code changes.

- **The 24x was a toy-shape artifact, and the guidance is corrected.** 0.11.0's README
  recommended `enable_batched_train` for "VRAM to spare" on an A2000 microbench (hidden
  512) where it measures 24x. At Qwen3-30B-A3B/48-layer width it is **1.05x at the
  highest peak memory of any lane**, while `enable_fast_train(dgrad=True)` is fastest at
  **2.52x**. The README now defaults to the latter and positions `enable_batched_train`
  as the no-extras fallback it actually is. `batched.py`'s docstring — which predicted a
  bigger card "should narrow this" — carries the measured reversal.
- **The dgrad fidelity caveat is retired by measurement** (`bench/dgrad-gate/`, published
  wheels, 16 + 48 layers): dgrad adds nothing to composed gradient error; an fp32-truth
  arm shows every lane — the reference loop included — on the composed bf16 noise floor
  (vs-reference divergence is rounding *similarity*, not accuracy); a 20-step real-data
  trajectory gate passes at a third of its band, with dgrad at **2.87x** the reference's
  real-data step rate; sm_120 (RTX PRO 4500 Blackwell) runs all 95 release-tag tests
  clean with the sm_86-tuned tile default holding.
- **New test:** DeepSeek-V4's clamped SwiGLU pinned through the batched path
  (`test_batched_train.py`) — the one epilogue composition nothing covered.
- **Credit** for the `enable_batched_train` approach (@jiwoon-ahn, #38) now appears in
  the README, not only the docstring/CHANGELOG.

## 0.11.0 — 2026-08-06

**`enable_batched_train` — a kernel-free batched training path.** Training
without `grouped-nf4-gemm` fell back to `ExpertsLoRA.forward`'s per-expert Python
loop: ~10k sync-gated iterations per forward at 256 experts over 40 layers, with
the GPU idle through most of it. That extra has to build and is arch-gated, so it
is not a rare configuration.

Experts are frozen, so the decoded stack is a constant w.r.t. autograd — and it
comes out of ONE `dequantize_4bit` call, because `_quantize_stack` uses
`compress_statistics=False` and the constructor refuses straddling shapes, making
the flattened absmax an exact concatenation. Verified **bit-identical** to the
per-expert loop and pinned as a test, since a future double-quant would break it
silently. Measured 32x against the per-expert decode at E=256.

One training step, E=256, 512 tokens, top_k 8, hidden 512, RTX A2000:

| path | step | vs loop | peak |
|---|---|---|---|
| reference per-expert loop | 601.2 ms | 1.00x | 59 MB |
| `enable_fast_train` | 132.6 ms | 4.53x | 108 MB |
| `enable_batched_train` | 25.0 ms | 24.01x | 417 MB |

Faster than the kernel lane it was written to fall back *from* — and it spends
peak memory to get there, materializing a stack where the kernel lane holds one
expert. At production width that trade is ~1.6 GB per layer against a few MB, so
**the kernel lane stays the answer under offload or VRAM pressure**. The two are
mutually exclusive and each refuses to patch over the other.

The approach is [@jiwoon-ahn](https://github.com/jiwoon-ahn)'s, from #38. Two
differences from the design proposed there: the backward re-decodes rather than
letting autograd save the stack, so gradient checkpointing is an option rather
than a precondition; and the LoRA delta is a padded double-`bmm`, so expert-LoRA
trains and the package default `TRAIN_EXPERTS=1` works.

**`enable_fast_train(..., dgrad=True)`** routes the fused lane's *backward*
through `grouped-nf4-gemm >= 0.7.0`'s single-launch dgrad kernel instead of its
per-expert decode loop, which measured 78-84% of a training step. A second opt-in
rather than part of the first, because it is a second numerics change: the loop is
exact, the kernel lands near 2.9e-3. Requested against an older kernel package it
turns off with a warning rather than raising from inside a forward.

**A parity contract for training paths** (`tests/test_fused_train_parity.py`).
Gradient *values* through the `ExpertsLoRA` composition were unverified —
`enable_fast_train` was covered by a forward comparison plus
`grad is not None`. A backward wrong by a constant factor still trains and still
descends, and nothing raises. Both lanes now satisfy one contract: forward,
`dL/dx`, and `dL/d` every LoRA parameter against the reference. Tolerances are
measured, not fitted, and a control proves the contract rejects a 1% scaling
error that forward parity alone passes.

**Fixed: `fast.py`'s module header described the whole module as inference-only.**
The paragraph predated `enable_fast_train`, which lives in the same file. It was
quoted back at us in #38 as evidence the package had no training accelerator.

**Untied output heads on multimodal checkpoints were silently tied to `embed_tokens`**
(#37). `load_moe_4bit_streaming` builds the text tower by keeping only keys under the
multimodal prefix (`model.language_model.` for Gemma-4, `language_model.model.` for Kimi
K3). `lm_head.weight` sits *outside* that prefix, so the filter dropped it — and the
meta-tie fallback then assigned the embedding matrix as the output head unconditionally.
Correct for a genuinely tied checkpoint (Gemma-4 ships no head on disk); for a
`tie_word_embeddings: false` checkpoint carrying a real head it meant every logit was
computed through the wrong matrix, with nothing raised: plausibly-shaped generations,
initial train loss at `ln(vocab)`, and a LoRA that "converges" by learning to steer hidden
states into `embed_tokens` — then collapses when the adapter is served on a stack that
maps `lm_head` correctly. The symptom is quant-invariant, so it reads as a quantization
fidelity problem, and an A/B against a tied model passes because the fallback is right
there.

The loader now recovers the head from outside the prefix (logging where it found it) and
gates the tie on `tie_word_embeddings`: untied config with no head that reached the model
raises instead of tying. The multimodal test previously covered only the tied path; the
untied load and the refusal are now both tested.

## 0.10.0 — 2026-08-05

**The residency engine was unreachable for every model the streaming loader produces.**
`enable_pipelined_residency` raised `NotImplementedError` the moment every `ExpertsNbit`
under the model was an `ExpertsLoRA.base` — which is every model
`load_moe_4bit_streaming` returns, i.e. the path most callers take. The composition it
needed already existed and went unused: `ExpertsLoRA._delegate_to_base` hands the whole
forward to the base when an engine is attached and the adapter provably contributes
nothing (`B` is zero-initialised, so an untrained adapter is *identically* zero), and it
already checked for this engine's own `_e4b_pipe_ref` marker. Only the patch site was
missing.

`target_modules()` now includes `ExpertsLoRA` bases. Membership means "targetable and
index-bearing", not "reachable by every engine" — the deprecated v0 `enable_hot_residency`
is not delegated to and still skips them, consuming its `hot_sets` entry. `ExpertsLoRA(r=0)`
raises a `ValueError` naming the supported way to get a zero delta, instead of dying on
`alpha / r` with a bare `ZeroDivisionError`.

Two silent-failure modes were fixed alongside it. `enable_mxfp4_nvme_residency` refused
wrapped bases via `isinstance(m, ExpertsLoRA)` over a list that only ever held
`ExpertsNbit`, so the check could never fire. And `enable_pipelined_residency` now WARNS
when a patch installs but cannot run (train mode, or a non-zero adapter) rather than
returning a count that implies work — a residency split that never executes reproduces the
unsplit reference exactly, so a dead patch scores a perfect zero and reads as a pass.

**`dispatched_modules()`** (new, exported) closes the footgun the above created. Hook what
is CALLED, not what is patched: a wrapped base is not called until an engine is attached,
so a `register_forward_pre_hook` on one fires zero times. The usual reason to hook these
modules is to build a routing histogram for an informed hot set — and that calibration pass
runs before the engine exists, by construction. Zero counts make `topk` return `0..K-1`, so
"informed" silently becomes the by-index set it exists to beat. It fails as a plausible
null, not as an error; it did exactly that once before the helper existed.

**Hot experts are now read in place.** The hot stack and the k-slot store were separate
allocations, so a hot hit still paid a device-to-device row copy before the GEMM could read
it. One shared `[n_hot + k, row_bytes]` store lets the GEMM address a resident row directly.
`sizes` stays the host constant `[1]*k`, still one GEMM launch, and the hot/cold decision
stays device-side — the fixed-shape, zero-host-sync decode loop is untouched. OLMoE-1B-7B on
an A2000, 7 interleaved reps x 96 tokens:

| hot set | before | after | delta | p | gather MB/tok |
|---|---:|---:|---:|---:|---|
| none (pure stream) | 11.77 | 11.78 | +0.1% | 0.225 | 418.4 -> 418.4 |
| by index | 13.29 | 13.37 | +0.7% | 0.025 | 418.4 -> 356.9 |
| informed | 16.51 | **16.83** | **+1.9%** | 0.025 | 418.4 -> **263.5** |

**The byte count that motivated that change overstated it, and the correction is the more
useful result.** The re-copy was 48.5% of all gather traffic on granite, which read as a
large lever. It is not: that copy runs at HBM bandwidth (~5 us/expert), while the PCIe cold
reads are what bind — so removing 37% of gather BYTES bought 1.9% of TIME. They were the
cheap bytes. A first version also cost -0.7% (p=0.013) on pure streaming, where an empty hot
set has nothing to gain but the new per-fetch row dispatch ran anyway; guarded on `n_hot`,
after which the change is non-negative on every config measured.

**Informed hot sets are a property of the model, not just the host.**
`docs/RESIDENCY-ENGINES.md` attributed the size of the gain to the host. Holding the host
fixed at one A2000: OLMoE-1B-7B gains **+24.2% (p=0.002)** from an informed hot set over a
by-index one, and granite-3.0-1b-a400m gains **nothing** (+0.7%/+1.4%/-1.0% across three
runs, never significant) — despite coverage working exactly as designed there (49.7% of
routed slots vs 24.5%, a real 2.0x skew, and a 46% cut in cold traffic). Reads have to bind
before coverage converts, and on a 1.3B model at ~4.2 GB/s they do not.
`bench/bench_hotsets_ab.py` measures this per (model, host) instead of assuming it.

**`quantize_layers`** (loader, #63) restricts 4-bit quantization to a subset of MoE layers,
reusing the loop's existing skip semantics so no new code path appears; `None` preserves
current behaviour bit-for-bit. Motivation is measurement rather than serving: the
KL-vs-knowledge work bounded the churn-to-destruction transition to somewhere in
2.2e-02 .. 1.41e-01 KL but could not locate it, because no quantization scheme lands in that
gap.

**KL-from-reference fidelity instrument** (`bench/kl_fidelity.py`, `bench/kl_paths.py`,
`bench/kl_ikp.py`, `bench/kl_sweep.py`) with K0 control receipts gating every measurement,
a committed 200-prompt set, and a path table where every row names its reference. Its
tier-transition row — 544 GB streamed from host DRAM against an all-resident reference,
**KL exactly 0.000 over 6,813 tokens, top-1 1.000000** — is the row this release's residency
fix unblocked. That row now carries a mandatory execution witness: its expectation is 0.000,
and an engine that never ran satisfies it perfectly, so the test side must stream nonzero
cold bytes and the reference side must stream none.

## 0.9.0 — 2026-08-01

**The trainer ran at batch size 1.** Its inner loop put one variable-length row through each
forward. A fused-MoE step's cost is largely *fixed per active expert* — the reference path
dequantizes each routed expert once, the fused path launches one grouped GEMM per expert
group — so a forward carrying 100 tokens paid nearly what one carrying 2000 does. On OLMoE
(16 layers x 64 experts, top-8) a single ~100-token row was dequantizing ~128 experts.

Rows are now packed until a **token budget** is reached. Measured on an RTX A2000
(OLMoE-1B-7B, SEQ=192, alpaca, 15 steps x grad_accum 4):

| `TOKEN_BUDGET` | s/step | tok/s | peak GPU |
|---|---:|---:|---:|
| 0 (one row per forward) | 17.8 | **22** | 5.23 GB |
| 1024 | 22.2 | **144** | 5.88 GB |
| 2048 | 22.8 | **248** | 6.67 GB |

**11.3x throughput for +1.4 GB.** Steps get 28% *slower* — each carries ~15x more data — so
tok/s is the metric this moves and s/step reads like a regression. `TOKEN_BUDGET=0` restores
the one-row path the v0.2.0 convergence receipts were measured on.

**The ceiling is VRAM, and it is not knowable in advance.** 4096 OOMs on a 12 GB card — but
only sometimes, on an unlucky batch of long rows; it sustained 353 tok/s for six steps first.
A static default cannot be right for both a 12 GB card running OLMoE and a 30B model on the
offload path, so an OOM now **halves the budget and retries the step** rather than killing
the run, down to a floor of 256. Verified: a run at 4096 that previously died now backs off
to 2048 at step 7 and finishes.

Padded batching, not sequence packing: pad positions carry label `-100` and attention `0`,
so no row can see another's tokens and no padding contributes loss — correct without
touching the model. Rows are drawn length-sorted within a bucket to bound padding waste, and
the budget counts the *padded* cost (`rows * width`), which is the work the GPU actually does.

Not claimed: better optimization. The batched arms reach a lower eval loss at equal `STEPS`
(-0.351 vs -0.181) purely because they see ~15x more tokens per step. Loss-per-token parity
is unmeasured.

## 0.8.0 — 2026-08-01

**DeepSeek-V4 (Flash / Pro) loads, serves and trains.** Full V4-Flash — 43 layers
x 256 experts, 284B params — loads in ~10 s at **8.74 GiB peak VRAM** and
generates, with 147 GB of experts served from an on-disk arena. The dense side
measured 8.28 GiB against 8.40 predicted from the shard headers alone. See
`docs/DEEPSEEK-V4.md`.

V4 needed three things the package did not have. Its experts are per-expert
MXFP4 with an epilogue that is gpt-oss's *clamps* over SwiGLU's *combination*,
so neither existing class was correct. Its dense half is block-scaled FP8 rather
than bf16 — `fp8_blocks` serves it at ~1 byte/param instead of 2, which is 8.4
GiB resident against ~14, i.e. whether it fits a 12 GB card. And the published
checkpoint ships in DeepSeek's own `inference/` spelling; transformers converts
that via its central `conversion_mapping.py`, but only inside `from_pretrained`,
which the streaming loader never enters.

**Two fixes to existing code that V4 exposed.**

`mxfp4` was *value-casting* scale bytes rather than reinterpreting them. That was
right by accident for gpt-oss, which ships both blocks and scales as `U8`, and
silently catastrophic for any checkpoint labelling scales `F8_E8M0`:
`.to(torch.int32)` yields the value (`2**-5` -> 0), not the exponent byte, so
every block would be scaled by `2**-127`. torch < 2.7 fails loudly at the read;
torch >= 2.7 materializes the dtype and the error goes silent.

`ExpertsLoRA` hardcoded `act_fn(gate) * up`. Since the adapter re-implements the
expert math inline — to inject the delta before the nonlinearity — it also owns
the choice of nonlinearity, so wrapping **any** clamped-expert architecture
trained a function the frozen base does not compute, with the loss still falling.
The base now supplies its epilogue via `_apply_gate`. This is why gpt-oss and V4
were built bare; V4 is now trainable.

**Hot sets are worth choosing properly.** `expert_profile` only probed
`ExpertsLoRA`, so it found *zero* layers on gpt-oss and V4 — exactly the models
worth profiling. It now probes whichever module is dispatched, and
`hot_sets_from_profile` / `coverage_from_profile` turn a routing histogram into a
hot set. Measured on full V4-Flash: frequency-ranked hot sets are **+37.1%** over
index-ordered at identical VRAM, and index-ordered is statistically
indistinguishable from pure streaming — 4.4 GiB spent for nothing.

**Also:** `scripts/energy_probe.py` (CPU RAPL via powercap or MSR, plus GPU) for
honest J/token, which needs bare metal — containers block both interfaces.
`tools/make_v4_fixtures.py` regenerates the real-bytes test fixtures, so those
tests are coverage instead of permanent skips. README consolidated 476 -> 359
lines (30.6 -> 23.9 KB) on top of 0.7.1's rewrite, keeping every measured number and
receipt link but stating each once — the fused-train figures appeared three times. The
"which door" decision procedure is now a ten-row table on the landing page with the
reasoning in `docs/CHOOSING.md`; residency, decode and V4 long-form moved to
`docs/RESIDENCY-ENGINES.md`, `docs/INFERENCE.md` and `docs/DEEPSEEK-V4.md`. All 19 repo
links are absolute and pinned — relative links 404 on PyPI.

## 0.7.1 — 2026-07-30

**0.7.0's headline features were not importable from the top level.**
`enable_fast_train` — the differentiable fused training path that both flagship
matrices measure, twenty cells of evidence, the thing the front page leads with —
was absent from `__init__.py` entirely, as were `enable_dense_offload`,
`DenseDiskSource` / `DiskHome` / `disk_homes_for`, and `enable_nvme_residency`.
The modules shipped; the names did not. `from experts4bit_qlora import
enable_fast_train` raised `ImportError` on 0.7.0. `__all__` goes 33 → 43, and a
check now asserts every symbol the README tells you to call is actually exported.

**The README described a package two releases old.** Its "Which door? (all six,
one line each)" table listed six execution modes when there were ten, and told a
*training* reader to call nothing — which stopped being the whole answer in
0.6.5. It is now a decision procedure keyed on what ran out (VRAM, host RAM, or
disk), with every mode's entry point, when to pick it, and what it requires.

Three factual corrections in the same pass: a cost figure still quoted the *eval*
delta against a band the protocol registers on **train** loss and median
step-wise; the "measured on an RTX A2000" section header sat above numbers from
two different hosts; and a note promising `enable_hot_residency` would be removed
*in* 0.7 was falsified by 0.7.0 shipping with it still exported — withdrawn
explicitly rather than quietly edited, since someone may have planned around it.

Also in this release: the second flagship-matrix model completed all ten
registered cells under a pre-stamped protocol
(`bench/flagship-matrix-model2/`), with C1 hashing 12.85 GB per cell against the
first matrix's withdrawn gate that hashed zero — and a C4 winner that **flips
sign** when the same cell is re-run on a second host, reported beside the
registered verdict rather than in place of it.

No code behaviour changed beyond the exports.

## 0.7.0 — 2026-07-30

**If you train Gemma-4-class models with `offload=True`, 0.6.x could not do it
at all** — the first backward raised `backward re-dequantization read an
offload-evicted expert`. The evict *post*-hook fired during the
gradient-checkpoint recompute and un-staged the layer before its own backward
re-dequantized it. `offload.py` documented the opposite as an invariant
("PyTorch stops that recompute early, so the evict post-hook does **not** fire");
whether the recompute reaches the post-hook depends on where the checkpointed
region's last needed tensor is produced, which is an architecture detail. OLMoE,
Qwen3-MoE and GraniteMoe stop early, which is why the wrong premise survived.
The post-hook is now a no-op inside a backward; residency is unchanged, because
the single-resident-slot policy already evicts there.

**Two new ways to fit a model that does not fit.** `nvme_experts` serves the
cold expert tail from NVMe, and `dense_offload` + `dense_disk` serve the
*dense* side — the 114.4 GB of non-expert weights that pinned host RAM cannot
hold for a K3-class model — straight from the checkpoint's own safetensors,
byte for byte. Nothing is transformed: the alternative way to fit a 114 GB dense
side on a small card is to quantize it, which changes the model.

Also: the flagship matrix's **B1 bit-exactness gate is withdrawn** — it hashed
`getattr(module, "gate_up_proj")`, which under `offload=True` is a 0-element
placeholder, so it compared `sha256(b"")` with itself and could not fail. The
performance numbers are independent measurements and stand; the assurance that
the frozen stack was untouched during those ten cells does not, and is
separately evidenced by the fused-train gate (16.31 GB hashed, byte-flip control
fires). Its B2 table is corrected too: it reported "Δ eval" where the protocol
registers `|Δ final-**train**-loss|` *and* median step-wise `|Δ|`. Both still
pass; the worst cell is 3.4× inside the band, not the 7× the eval column implied.

The README and `METHODOLOGY.md` §11 no longer end on "a memory optimization, not
a speedup" without saying what the fused path measured: **1.75–1.81× faster per
step at 0.754–0.755× peak VRAM and 0.797–0.846× energy**, both arms offloaded.

## 0.6.7 — 2026-07-30

**`e4b serve` could not load Kimi K3 at all.** Four blockers, each hidden behind
the last: no `trust_remote_code` anywhere in the package (so `AutoConfig` raised
before the architecture gate, with an opaque message), `kimi_k3` missing from
`SUPPORTED_ARCHITECTURES`, a multimodal prefix that is per-family rather than
universal, and per-expert MXFP4. `trust_remote_code` is a new argument plus
`E4B_TRUST_REMOTE_CODE=1` and **defaults to OFF** — executing
checkpoint-supplied code is the caller's decision, never a default.

## 0.6.6 — 2026-07-29

**Per-expert MXFP4 layouts load.** `dequantize_mxfp4` ended in
`out.transpose(1, 2)`, hardcoding gpt-oss's rank-4 `[E, rows, G, B]` blocks, so a
single expert projection `[rows, G, B]` raised `IndexError: Dimension out of
range` instead of returning `[K, rows]`. That is the layout every
DeepSeek-V3-lineage checkpoint ships per expert. `transpose(-2, -1)` is
equivalent for the rank-4 case and correct for both.

## 0.6.5 — 2026-07-29

**Training could never reach the fused kernel, and `enable_fast` reported
success anyway.** Two distinct problems, both fixed here:

- `enable_fast()` patched all expert modules and returned a non-zero count while
  the kernel was invoked **zero** times in train mode. `ExpertsLoRA` hands off to
  the patched base only via `_delegate_to_base()`, which requires
  `not self.training` — and the streaming loaders return a model in `nn.Module`'s
  default train mode. Measured on an RTX 4090: 0 kernel calls / 8.34 tok/s
  against 288 calls / 33.6 tok/s. **A patch count is not a call count.**
- `enable_fast_train()` is new, and is the differentiable path: it patches the
  `ExpertsLoRA` **wrapper** — the module the model actually calls — and composes
  the frozen projection with the trainable `B(Ax)` delta at the pre-activation
  point, the only correct place, since `act(Wx + BAx) != act(Wx) + d`. Opt-in on
  purpose: it changes the expert summation order (group-sorted vs ascending
  expert id), which should be a deliberate choice in a training run.

Requires `grouped-nf4-gemm>=0.2.4`. `--help` no longer loads a model.

## 0.6.4 — 2026-07-28

**If you installed `[fast]` on 0.6.3 or earlier, the fused kernel was not
running.** `enable_fast()` patched `ExpertsNbit.forward`, but `ExpertsLoRA`
inlines the expert math and never calls `self.base(...)` — and
`load_moe_4bit_streaming` always wraps in `ExpertsLoRA`. The advertised speedup
was a silent no-op on the loader this package tells you to use. Upgrade to get
it; nothing about your code changes.

**This is a behaviour change, not only a fix.** With delegation live, the fused
path actually executes, and it is a different computation from the reference
loop — priced at **+0.023% perplexity** (see `docs/METHODOLOGY.md`). If you were
unknowingly running the reference path, your numbers will move slightly.

- **`enable_fast()` now reaches the streaming-loader path (PR #36, `c2bf990`).**
  `ExpertsLoRA` previously inlined the expert math and never called
  `self.base(...)`, so the `[fast]` fused kernel was patched onto a method that
  was never invoked — a silent no-op for every model loaded with
  `load_moe_4bit_streaming`. `ExpertsLoRA` now delegates to its base when the
  adapter provably contributes nothing (B is zero-init, so an untrained adapter
  is *identically* zero), guarded so a trained adapter is never silently dropped.
- **Docs corrections (2026-07-28).** The informed-hot-set decode gain is scoped
  to the (bandwidth-limited) hosts it was measured on — it does not replicate on
  a fat-PCIe box. The `memlock` deployment note no longer claims `cudaHostAlloc`
  is gated by `RLIMIT_MEMLOCK`; that cause is false and the observed slowdown is
  now marked unattributed. README states that both residency engines require
  standalone expert modules and refuse/skip `ExpertsLoRA`-wrapped bases.

## 0.6.3 — 2026-07-21
- **Behavior change — serve binds to `127.0.0.1` by default** (was `0.0.0.0`).
  LAN exposure is now opt-in: set `E4B_HOST=0.0.0.0` to restore the old
  default. Migration: one env var. Rationale: a localhost tool for the
  machine's owner should not be reachable from the network unless asked.
- Optional bearer auth: set `E4B_TOKEN` and the generation routes require
  `Authorization: Bearer <token>` (off by default; `/health` stays open).
- README: first-screen "It dials" bullet (informed hot sets +57-120% at
  identical VRAM); engine-tier tags on the v0 offload-path decode figures;
  the serving posture paragraph. Length pass — the storage-modes matrix,
  serving/Docker, benchmarks, and the bitsandbytes essay moved to `docs/`
  with anchor-preserving stubs.
- `[fast]` pins `grouped-nf4-gemm>=0.2.1`.

## 0.6.2 — 2026-07-21
- `enable_hot_residency` deprecated at call (superseded by
  `enable_pipelined_residency` — same capability, K is config; kept through
  0.6 so the stamped v0 receipts stay reproducible; removal in 0.7).
- README: "Which door?" decision tree covering all six execution modes with
  honest status tiers (`enable_cold_engine` labeled performance-experimental —
  the host decode is a correctness path until the AVX2 kernel lands); all
  relative links absolutized (they rendered as pypi.org 404s in the PyPI
  long_description); CPU-only bitsandbytes first-import notice documented.
- `py.typed` marker ships (the public API already carries annotations).
- Permanent built-artifact smoke in CI and the release gate: wheel installed
  into a clean venv, README-surface import battery + deprecation-warning
  check; README link check blocks publish.

## 0.6.1 — 2026-07-20
- Cold engine (`enable_cold_engine`): hot partition GPU-resident, cold tail
  computed on the host from CPU-resident NF4 (activation-sized bus traffic).
  Host decode bit-exact vs bitsandbytes' CPU `dequantize_4bit`;
  `dequant="auto"` gates bnb behind `avx512f` (on AVX2-only hosts bnb falls
  below naive torch — grouped-nf4-gemm `bench/cold-engine/` receipts).
  All-cold + `device="cpu"` is a pure-host MoE (no CUDA, no `[fast]`).
  gpt-oss epilogue supported. 0.6.0 shipped from a pre-merge tree without
  the engine; 0.6.1 is the real release.

## 0.6.0 — 2026-07-20
- Hot-expert residency (`enable_hot_residency`, #26/#27): expert-granular
  partial residency — hot experts VRAM-resident on the fused kernel, cold
  tail streamed from pinned host RAM; gpt-oss (clamped-GLU + per-expert
  biases) supported; requires `[fast]`, fails at enable time with an install
  hint.
- Routing-informed hot sets (#28): calibrate-then-pin reference driver;
  decode gain tracks routing coverage on thin-link hosts (gpt-oss +56/+120%, Gemma-4 +44%,
  OLMoE +19%); multi-socket affinity law documented (pin `taskset` before
  any cold-path number).
- Hybrid-vs-llama same-box A/B receipts + Gemma-4 gated-weights serving gate
  (`bench/RESULTS-gptoss-hybrid-ab.md`, `bench/RESULTS-informed-hotsets.md`).
- README package-family section (the `[fast]` seam with grouped-nf4-gemm).

## 0.5.0 — 2026-07-18
- `[fast]` extra: fused grouped-GEMM inference via grouped-nf4-gemm —
  `enable_fast()` routes frozen-expert inference through the single-launch
  kernel (measured 3.65× at bs=1 decode, OLMoE geometry, A2000; #25).
