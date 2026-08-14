# The arena read gap was an artifact. On a quiet box there is none.

### 2026-08-13 · NVIDIA L40S (Ada sm_89), SECURE on-demand, 32 vCPU / 27.2 quota · torch 2.8.0+cu128 · e4b 0.17.5 + gnf4 0.10.0 · raw [`cpugap.jsonl`](cpugap.jsonl)

**Evidence tier: `measured`.** Pre-registered in [`PREREG-cpu-gap.md`](PREREG-cpu-gap.md)
**before the pod was rented**, including the decision rule and the direction expected.

**Outcome: no code change.** [gnf4#61](https://github.com/pjordanandrsn/grouped-nf4-gemm/issues/61)
as filed described an artifact of a loaded, CPU-starved box, not a defect in the read path.

## What was being chased

Profiling on the QNAP found the reader pool delivering **3.66 GB/s** during training where
the same pool read **6.94 GB/s** in isolation — a **1.9× gap** — with the tier's own Python
accounting for only 7% of `ensure` and the pool saturating its queue depth. The suspect was
CPU and GIL contention between the four reader workers and the training process.

That box could not answer it: 12 cores at load ~9.8 throughout, and the two rates came from
separate runs that saw different background load. So: a quiet pod, both rates measured in
one process at one CPU budget, interleaved `iso → train → iso_self` per round.

## There is no gap

Qwen3-30B-A3B, `hot_rows=128`, 3 scored rounds per budget, warm round dropped:

| CPUs | isolated | during training | **gap** | control (`iso_self/iso`) | concurrency | submit+other |
|---|---|---|---|---|---|---|
| 6 | 5.33 GB/s | 5.74 GB/s | **0.93× [0.91–0.93]** | 0.987 [0.774–1.003] | 3.89 | 8.1% |
| 12 | 5.29 | 5.74 | **0.92× [0.91–0.93]** | 0.985 [0.777–1.012] | 3.90 | 7.5% |
| 24 | 5.25 | 5.69 | **0.91× [0.90–0.94]** | 0.992 [0.823–1.024] | 3.88 | 8.7% |

The gap is **below 1.0** — the tier reads slightly *faster* under training load than the
standalone probe does, plausibly because it lands rows in pinned memory through a persistent
pool while the probe uses mmap buffers and builds a pool per call.

**And it does not resolve.** Under the pre-registered gate an effect must clear the
control's own range; the gap's range sits *inside* it. The honest reading is not "training
reads are 9% faster" but **"no gap is detectable at any budget."** It is also flat in CPU —
6, 12 and 24 give the same answer — which is what kills the contention hypothesis outright.

The tier was confirmed on `O_DIRECT` at qd=4 with no buffered fallback, so these are real
device reads and not page cache.

Two cross-checks against the QNAP run, on completely different hardware: **16.67 GB read per
step** (against 16.71 there), and the same two-thread structure — forward and the checkpoint
recompute each issuing 188 miss calls and each reading ~33 GB per 4-step round.

## Scoring the pre-registration

| registered | actual | verdict |
|---|---|---|
| at C=6 the gap is 1.2–1.7×, centre ~1.4 | **0.93×** | **miss** — right direction, wrong magnitude; it vanished rather than shrank |
| the gap falls monotonically as C rises | flat (0.93, 0.92, 0.91) | **n/a** — no gap to fall |
| at full vCPU the gap is < 1.25× | **0.91×** | **hit** |
| concurrency stays ~3.9 of 4 at every C | 3.88–3.90 | **hit** |
| `submit`+`other` ≤ 10% of ensure | 7.5–8.7% | **hit** |

The registered decision rule was *"gap < 1.25× at full vCPU → the 1.9× was a loaded,
CPU-starved box; no code change is warranted; #61 closes as not a defect."* That is what
happened and it is being followed.

Prediction 1 is a genuine miss and worth keeping visible: I expected contention to be
*part* of the story and it is none of it.

## The finding that survives: qd=4 is tuned to the wrong regime

Registered in advance as a prediction that an existing finding would flip. The QNAP measured
qd=4 optimal with qd=8 and qd=16 **worse** — on a CPU-starved box. At 24 CPUs:

| qd | O_DIRECT | buffered |
|---|---|---|
| 1 | 2.04 GB/s | 18.78 |
| 4 (**the tier's hardcoded depth**) | 5.31 | 51.18 |
| 8 | **5.95** | 64.25 |
| 16 | **6.13** | 60.84 |

**qd=8 and qd=16 beat qd=4 by 12% and 15%.** The tier hardcodes 4. That is a real, if
modest, lever — tracked separately, since the training path already achieves 5.69–5.74 at
qd=4, so the headroom above it is smaller than the isolated numbers suggest.

**The buffered column is not a 9–11× win, and should not be read as one.** This pod has
125 GB of RAM and the arena is 16.3 GB, so buffered reads are served from page cache — the
64 GB/s is memory bandwidth. Page cache spends exactly the resource the NVMe tier exists to
bound. On the machines this tier is *for*, where host RAM cannot hold the experts, that cache
cannot exist and buffered would fall back toward device speed. It is host RAM with extra
steps.

## What this does not say

One card, one model, one pod, one filesystem. The QNAP's ZFS pool tolerated unaligned
O_DIRECT buffers; this pod's overlay filesystem returns `EINVAL` for them, which is what the
probe hit first — so the two boxes do not even enforce the same I/O contract, and only the
in-run ratios are comparable across them.

Staging is a **larger** share of the step here than on the A2000 (~46% vs ~29%), because the
bytes are the same and the card is faster. That is consistent with the earlier finding that
the arena's load saving travels while its step cost does not, and it gets worse on faster
hardware.

## Cost

$0.99/hr for ~47 min ≈ **$0.78**. SECURE on-demand, `interruptible:false`. External teardown
backstop armed on a separate host before the first run and disarmed after; the pod was
verified gone from the account listing rather than from the `DELETE` status. Another
session's pod was live on the same account throughout and was terminated by neither.
