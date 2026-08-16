# Pre-registration — is the arena read gap a CPU-contention artifact?

**Written 2026-08-13, before the pod was rented.** Committed ahead of the data.

## What is being tested

Profiling `ColdTier.ensure` per thread on the QNAP found the tier's own Python is not the
bottleneck — `submit` plus the pass-1/publish/lock work is **7% of ensure** — and that the
reader pool saturates its queue depth (`worker/wait = 3.90×` against qd=4). What remains is
per-worker throughput: the same pool that reads **6.94 GB/s** in isolation delivers
**3.66 GB/s** while training runs, a **1.9× gap**, with per-worker throughput falling from
1.74 to 0.94 GB/s.

The live explanation is contention for CPU and the GIL between the four reader workers and
the training process — during training there are 6+ runnable threads against a `--cpus=6`
quota. **That cannot be settled on the QNAP.** It has 12 cores, sat at load ~9.8 throughout,
and the isolated baseline and the training run saw different background load; per the
testbed policy that box is for correctness, not timing. Ratios *within* one run are robust
to drifting load, a rate compared *across* runs is not.

So: a quiet box, both measurements on it, interleaved, across a CPU sweep.

## Design

One SECURE on-demand pod, Qwen3-30B-A3B, arena baked on the pod's own disk. Both probes
run under `taskset -c 0-(C-1)` so the CPU budget applies to the reader pool and the training
process alike — the same knob the `--cpus=6` container was applying.

| arm | what runs | what it yields |
|---|---|---|
| `iso_C` | `iomode.py`, qd=4, real scattered arena offsets | isolated GB/s at budget C |
| `train_C` | `ensurephases2.py`, per-thread attribution | achieved GB/s during `wait` at C |
| `iso_C_self` | `iso_C` again, same round | **control** — self-pair |

`gap_C = iso_C / train_C`. Budgets: **C ∈ {6, 12, all vCPU}**, with 6 chosen to match the
QNAP container exactly. 3 interleaved rounds, first dropped.

Absolute rates are this pod's. The quantity that travels is `gap_C` and its trend in C.

## Gate

Per [`PREREG-timing-AMENDMENT-1.md`](PREREG-timing-AMENDMENT-1.md): an effect is resolved
when its per-round range does not overlap the control's. `iso_C_self / iso_C` is the control.

## Predictions

1. **At C=6 on a quiet box the gap is smaller than the QNAP's 1.90×** — centre ~1.4, band
   1.2–1.7 — because the QNAP figure includes competing production load that a quiet pod
   does not have.
2. **The gap falls monotonically as C rises.**
3. **At full vCPU the gap is < 1.25×.**
4. **Concurrency stays ~3.9 of 4 at every C.** The pool saturating its depth was measured
   under starvation; it should not get worse with more CPU.
5. **`submit` + `other` stays ≤ 10% of ensure at every C.** If the tier's bookkeeping were
   the real cost this is where it would show, and it did not on the QNAP.

**Secondary, and I expect to be wrong about the existing answer here:** the QNAP found qd=4
optimal with qd=8 and qd=16 *worse*. That was measured on a CPU-starved box, so it may be an
artifact of starvation rather than a property of the device. `iomode.py` sweeps qd ∈ {1,4,8,16}
at full vCPU. **Prediction: qd=8 ≥ qd=4 at full CPU**, which would mean the tier's hardcoded
queue depth is tuned to the wrong regime.

## What each outcome means

- **Gap < 1.25× at full vCPU** — the 1.9× was a loaded, CPU-starved box. No code change is
  warranted; the deliverable is a documented CPU floor for the reader, and
  [gnf4#61](https://github.com/pjordanandrsn/grouped-nf4-gemm/issues/61) closes as not a
  defect.
- **Gap > 1.5× at full vCPU** — there is a real inefficiency in the read path that ample CPU
  does not fix. #61 stays open and the next suspects are the GIL between `preadv` calls and
  `_read`'s per-read `_stats_lock` across four workers.
- **1.25–1.5×** — report the curve and call it unresolved rather than picking a side.

**Registered now: I expect outcome 1**, and I expect the qd finding to flip. Publishing both
in advance is the point — the first would mean the issue I filed describes an artifact, and
saying so ahead of the data is what keeps that from being reinterpreted afterwards.

## Cost

SECURE on-demand only, `interruptible:false`, one card. External teardown backstop armed on
the mini before the first run; `costPerHr` read off the pod after create rather than trusted
from the quote. Expected ~1.1 hr including the bake, well inside the $35/job cap.
