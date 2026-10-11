# DQ11 L-only census instrument amendment

This amendment implements the reporting census in
[`DQ11-L-CENSUS-PREREG.md`](DQ11-L-CENSUS-PREREG.md). It fixes the instrument
before a 5090 quote. The preregistration's DQ11 L model, inputs, forty-update
recipe, five trained warmups, seven classes, 5% prospective BUILD bar and
`C_MLP >= 0.10*T` decision are unchanged. This amendment authorizes no rental.

The source closure is e4b `13b55990433f357d63232f75926ce19f5bda5b42`,
Loggetta `34ecb6cec6f43a6f8607ff9f192749fdc7b587e9`, and the DQ11
`science.sha256` with SHA-256
`4691e7329f12ad2f4b7518b75c4d3e351207fc4e5220aee514fdd42c429fef01`.
The worker reads those sealed source files and the canonical input/model
manifests. Its own three Python files are hashed into each run receipt. The
science path remains importable at its pinned bytes; the instrument calls its
unchanged `prepare`, `capture_pass`, `train_loop`, K8 quality check, path census
and live train-prefetch witness. No L training algorithm or admission rule is
copied into the instrument.

## Fixed capture

Use `torch.profiler`/Kineto Chrome trace schema version 1, CPU and CUDA
activities, `record_shapes=True`, `profile_memory=False`, `with_stack=False`
and `with_flops=False`. The 5090 worker requires Torch `2.12.1+cu130`; the
separate zero-dollar A2000 correctness rehearsal requires Torch
`2.11.0+cu128`, a single real A2000/sm86, the same real Mistral model and
forty-update L recipe. A2000 timings cannot establish a 5090 ceiling. Neither
mode substitutes a tiny model. The raw trace cap is 512 MiB and 2,000,000
events; exceeding either refuses certification. Trace and run receipts are
write-once. The launch amendment must separately review transport and total
size caps before any paid draw. The first real-L A2000 proof trace was 69 MiB
and 234,943 events for one update; the three-update capture includes trained
backward and optimizer activity. The cap was raised prospectively from
128 MiB/500,000 events before any deciding 5090 data. This is a refusal
limit, not a prediction of captured size or authorization to move raw traces
through the bus.

The worker first runs configuration C's same-arm observer/removal bitwise
proof on actual L under native SDPA. The CUDA proof trace must correlate
device events to CPU `External id` values. Fresh reference and capture
processes then reload the canonical initializer. The reference runs forty
unprofiled updates and sets `T` to the median of updates 6 through 40.
The capture profiles only updates 6, 7 and 8 after five trained warmups,
using temporary ranges around the original trainer's two step-clock reads.
The wrapper and its hooks are removed afterward. Captured wall times are
reported as overhead context and never replace `T`.

## Class map and refusal

The seven classes are `MLP-adapter`, `qkv-adapter`, `o-adapter`, `base NF4
GEMM`, `attention`, `optimizer`, and `other`. The reporting hooks scope all
224 seven-projection PEFT modules by canonical path and their frozen base
layers. They scope all 448 FP32 adapter parameter-gradient nodes, native
`scaled_dot_product_attention`, gradient clipping and `AdamW.step`.
Forward recomputation under non-reentrant checkpointing uses the same path
scope. The CPU operator's `External id` binds each CUDA kernel, copy and
memset; autograd sequence and forward-thread fields bind backward work.
The deepest registered scope wins; a base-layer scope overrides its outer
adapter scope. Kineto can repeat one autograd sequence on nested forward
operators and on the checkpoint worker thread. For backward work, the
reducer keys the node by its recorded forward thread and sequence, then
requires exactly one forward creator operator from the fixed
`BACKWARD_CREATOR` table. That creator's deepest registered scope supplies
the class. `AccumulateGrad` has no forward creator and must be directly
scoped by its parameter hook. The real-L A2000 correctness proof exposed
this Kineto behavior before any deciding 5090 data; the first proof refused,
and this explicit map replaces the ambiguous mapping prospectively.
The first full A2000 capture then recorded 502,233 events in a 145 MiB trace
and refused two unregistered autograd creators. Explicit flash-attention and
division creators were added before any deciding 5090 data; unknown kinds
still refuse.
Kernel names are recorded but never used to choose a class.
Unbound, ambiguous, duplicate or crossing events refuse certification.
An unscoped rank-16 adapter GEMM also refuses. Device events outside the
synchronized captured step ranges are profiler startup/shutdown evidence,
not deciding updates; a device event crossing a step boundary refuses.

The normalized schema is `dq11-l-census-events/1`: three step intervals and
one row per captured physical device event, with id, step, class, canonical
path, phase, nanosecond bounds, CPU external id, device category, kernel
name, launch operator and stream. The reducer partitions each synchronized
step timeline into seven disjoint class-only rows, a mixed-class overlap row
and a host/no-device residual. These rows must close the step interval
exactly; the preregistered tolerance is at most 1 microsecond for timestamp
rounding. It also retains raw kernel-duration sums, per-class interval unions,
counts and forward/backward detail. Every captured event is assigned once;
CPU operator durations never enter device sums.

`C_j` is the minimum across updates 6, 7 and 8 of the union of class `j`'s
device intervals. The reducer reports `C_j` and `C_j/T` for MLP, qkv and o.
If MLP clears `0.10*T`, the verdict is `BUILD_MLP`. Otherwise a uniquely
largest candidate that clears that same line is proposed for a new BUILD
review; a tie requires review; if none clears it, the BUILD lane closes.
Failed identity, gradient, attribution, K8, source or closure checks emit
`CENSUS_REFUSAL`, never a substituted candidate.

The CPU falsification suite covers duplicate operator/kernel views,
overlapping streams and class unions, unbound IDs, backward sequence
attribution, checkpoint recomputation and early stop, hook removal and
gradient neutrality, missing FP32 gradients or second projection calls,
source/input/proof mismatch, dropped steps, malformed events and interval
closure. The required next gate is a real-L A2000 CUDA/CUPTI correlation
rehearsal with an explicit `L-CENSUS-REHEARSAL.json` marker and a fresh
`gpu:a2000` claim. Its decision is always `CORRECTNESS_REHEARSAL_ONLY`.
No 5090 quote or run follows until that proof is reviewed, the instrument
amendment is merged and the maintainer gives a fresh paid launch ACK.
