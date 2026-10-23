# T-B: capturing the training step (a design note; USD 0, no box)

Map 1 (training throughput) lever T-B: graph-capture the fused training step by routing bucket, the way serving captures its
decode buckets. The maintainer ruled that T-B reshapes the training loop, so it gets this note before any box. Nothing here
was measured for this note. Every figure is a public receipt or issue, cited where it is used.

## 1. Why the launch floor is the lever

- **The step is host-bound.** TC1's profiled matched arm on an RTX 5090 (`tc1-5090-16`, Qwen3-30B-A3B, the field recipe)
  was 48 % device-busy. Over three profiled steps it made about 115,000 `cudaLaunchKernel` calls, which cost 1,929 ms of
  CPU time, and 4,436 `cudaStreamSynchronize` calls (#945).
- **The syncs are already handled.** #946 and grouped-nf4-gemm#438 took a fused MoE layer pass from 13 host syncs to 1
  (`tc1-5090-38`, TC1 amendment 10: new / legacy step time 0.866 shipped, 0.847 matched).
- **The last sync is not the lever.** Map 1's T-A, the one remaining count read per grouped forward, is parked: on a
  host-bound step that read is not exposed.
- **What is left is the launch floor.** That is roughly 38,000 launches per step, each with its Python and dispatcher cost
  on the host.
- **The ceiling.** For any lever that removes host launch cost, the ceiling is the device time: a captured step cannot run
  faster than its kernels. At 48 % busy, the ceiling is about 2x on that host. It is higher on slower CPUs and lower on
  device-bound shapes.

## 2. What capture requires, and where the training step stands

| requirement | today | what it would take |
|---|---|---|
| **No host sync inside the region** | One count read per grouped forward (`engines/fast.py::_group_by_expert`): the fused forward's launch grid, the device-tile dgrad's grid and the padded LoRA delta (`G * max(sizes)` rows) are sized from host ints (the maintainer's 2026-10-04 scoping on #945). Logging reads (`loss.item()`) sit outside the step. | Device-built tile lists for both training kernels, with an over-provisioned grid whose tiles exit early (serving's `build_group_tiles_device` is the pattern), and a LoRA delta padded to a static cap. This is T-A's parked device grid: worthless alone, a prerequisite here. |
| **Static shapes** | Packed rows are fixed (seq x mb). The field recipe pads each micro-batch to its own length, so the token count varies by step. Per-expert row counts vary every step. | One graph per padded-length bucket, as serving buckets decode batch sizes. Per-expert counts are absorbed by the over-provisioned grid, not by more buckets. |
| **Autograd inside the graph** | Eager autograd, with gradient checkpointing (non-reentrant) recomputing each layer in the backward. | Either whole-step capture (forward, backward and optimizer in one `torch.cuda.graph`, static input and gradient buffers), or per-module `make_graphed_callables`. The latter does not compose with checkpointing's saved-tensor hooks without care, so it applies most naturally to un-checkpointed pieces. |
| **The optimizer** | AdamW, eager. | `capturable=True` keeps its step count on the device, or the optimizer stays outside the graph (one extra eager region a step). |
| **Memory** | The 24 GB fit (Map 1's L3, T-C) is about 10 MiB to 1.1 GB short. | A graph's private pool holds the captured activations and workspace for the graph's life, and one graph per bucket multiplies it unless the pools are shared. This works directly against the memory frontier and must be measured, never assumed. |

## 3. Three options

- **A. Whole-step capture per padded-length bucket.**
  - Removes nearly all host launch cost: the step approaches device time plus the over-provisioned tiles' cost.
  - Needs every row of the table: the training device grid in grouped-nf4-gemm (forward and dgrad), a static-cap LoRA
    delta, a capturable optimizer, checkpointing inside the capture, and pooled memory.
  - The largest win and the largest project. Every piece also changes a default-path kernel contract.
- **B. Partial capture of the dense pieces.**
  - Capture each decoder layer's attention block, norms and router (`make_graphed_callables`, static shapes per bucket),
    and leave the MoE expert path eager.
  - Needs no grouped-kernel change.
  - Its win is bounded by the dense pieces' share of launches. That share is not known today.
- **C. No capture: cut the launch count.**
  - Fuse the glue around the grouped GEMMs (gathers, index copies, fills, casts, elementwise epilogues) into fewer
    kernels, each value-identical and opt-in, then default by its own read.
  - Incremental and low-risk, and it compounds with A or B later.

## 4. The decision needs one instrument first

Which option pays depends on where the ~38,000 launches are, and that is not known: there has been no training kernel
census since TC1's amendments 10-21. Map 1's **T-H census** answers it. It runs as its own small explore item feeding this
note (about USD 0.5 a box): one profiled step at the matched set on a 5090.

**What it records:**
- launch count and host time per kernel family, split by phase: the forward, the checkpoint recompute in the backward, and
  the backward proper;
- device time per family;
- the host/device overlap fraction.

**Where it runs:**
- at seq 512 and at the field recipe's padded lengths;
- on two hosts, because the share depends on the host: one slow-CPU host (an EPYC Zen 2-4 or a Xeon) and one fast-CPU host
  (an EPYC 9xx5 or a desktop part). Hosts are chosen by the provider's CPU name family, never by its listed GHz, which does
  not predict single-thread speed.
- Each box records its measured single-thread rate beside the census.

**Fixed in the census code before any profile:**
- **The family classifier.** Kernel-name patterns map each kernel to one of five families:
  1. the MoE expert path: the grouped NF4 forward and dgrad, the LoRA delta, dequant;
  2. attention;
  3. dense GEMM: projections, the router, the head;
  4. norms and RoPE;
  5. elementwise glue: gathers, index copies, fills, casts, epilogues.
- **An "unclassified" bucket.** If it holds more than 10 % of host launch time, the census is VOID until the patterns are
  amended, never read by eye.
- **Each option's addressable share** is the host launch time it can remove:
  - A: every in-step launch;
  - B: attention, dense GEMM, and norms and RoPE, in the FORWARD phase only. These sit inside each checkpointed layer, and
    per-module capture does not compose with the non-reentrant checkpoint's hooks (section 2), so their recompute and
    backward launches stay eager. B's share counts them only if an amendment names how B captures those phases;
  - C: the glue.

**The decision rule, fixed before data.** It applies on the slow-CPU host, where host time binds, at the field recipe.
1. **B first** if its addressable share is at least 40 % of host launch time.
2. Otherwise, **C first** if the glue's share is at least 40 %.
3. Otherwise **the MoE expert path dominates, and only A reaches it.** A's prerequisites (the training device grid) then get
   their own design and registration, and T-A reopens as part of it.
4. **When B and C both qualify, B goes first:** it needs no kernel change.

## 5. A prototype, after the census (about USD 3, explore tier)

- **The prototype:** one decoder layer of Qwen3-30B-A3B, the chosen option on that layer only and the rest eager, at the
  matched set on one 5090.
- **Arms:** eager, and the prototype. Mirrored order, a fixed batch, the profile per arm.
- **Kill criteria, fixed before the box:**
  - **Wall time.** The layer's wall time (forward + recompute + backward, synced) on the slow-CPU host falls by less than
    20 % against eager, or its device time rises by more than 5 %. Capture cuts host time almost by definition, so host
    time is never the measure. Device time is read beside it because A's over-provisioned early-exit tiles add device work.
  - **Memory.** Reserved memory (`max_memory_reserved`, or the device's used bytes) rises by more than 0.5 GB at the
    matched set. A graph's private pool appears as reserved segments, which peak allocated memory can understate, and the
    24 GB frontier is closer than 0.5 GB. `TC1_STEP_MEM` (e4b#2003) records this per step, folded across phase resets.
  - **Gradients, by option.**
    - B and C run or fuse the same math, so their gradients must be BITWISE equal to eager. A band would hide a capture
      bug such as a stale static buffer.
    - Only A, whose device grid can change the reduction order, is held to the fused path's run-to-run band.
- **Then:** a pass on all three sends the option to a registered lane at the field recipe, under the TC1 protocol.
  Otherwise T-B parks with the census as its record.

## 6. What this note does not decide

- No default changes. Capture, if it lands, starts opt-in, behind a knob, and becomes the default only by its own reading.
- No serving change: serving's decode capture is a separate engine with device-side ids from the start.
- Nothing about quality. A captured step must be value-identical to eager within the existing band; it is a speed lever
  only.
