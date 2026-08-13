# The host-RAM ceiling, measured — a model that trains in 2.25 GiB and cannot be trained in 5.5

### 2026-08-13 · RTX A2000 12GB, sm_86 · torch 2.8.0+cu128 · **published wheels** e4b 0.17.2 / gnf4 0.9.0 · driver [`runarm.sh`](runarm.sh) + [`cg.py`](cg.py) · raw [`ladder.jsonl`](ladder.jsonl)

**Evidence tier: `measured`.** Both packages installed from PyPI, not a working tree.
The threshold reported here was **written down before it was measured**: 0.17.0's
changelog entry said a demonstration "needs a machine capped near the host arm's true
~5–6 GB working set", committed and tagged on 2026-08-12. The host arm's measured
requirement is **5.91–6.17 GB**.

That prediction is public in the repo at the `v0.17.0` and `v0.17.2` tags — *not* on
PyPI. The long description is built from `README.md` alone, so nothing in `CHANGELOG.md`
has ever reached the package page, which is why 0.17.2 failed to deliver what its own
notes claimed and why the summary above lives in the README this time.

## Why this exists

`enable_nvme_train_residency` exists for one case: **the frozen experts do not fit host
RAM.** Through 0.17.2 that case had never been demonstrated. Every prior attempt capped a
rented container at 8–16 GB and the host arm survived, which proves nothing — it never
needed 16 GB. Worse, two of those attempts were run against a cap that was not applied at
all, and would have reported success had the coin landed the other way.

Two things had to be fixed before the question could be asked honestly.

**The cap has to be real, and it has to include swap.** A rented container's cgroup is
read-only, and the kernels seen there had no `memsw` accounting, so a cap bounds resident
memory only: an over-limit process pages out to swap and survives. "Survived by thrashing"
is a different outcome from "fit", and it reports as success. Here `docker --memory=N
--memory-swap=N` sets **both** `memory.limit_in_bytes` and `memory.memsw.limit_in_bytes`
(confirmed by reading them from inside the container), so there is no escape.

**The instrument has to be positive-controlled in both directions.** Before any arm ran:
a 900 MB allocation under a 512m cap is killed (exit 137); the *same* allocation under a
2g cap completes (exit 0). A cap that only ever kills, or only ever passes, cannot
discriminate.

## Method

One 16-layer MoE (OLMoE-1B-7B-0924, 64 experts/layer), one seed, four training steps,
`gradient_checkpointing` on, identical data. The arms differ in exactly one thing:

- **host** — experts fused, quantized and pinned in host DRAM (`offload=True`), 3.62 GB
- **arena** — experts read from a baked arena on NVMe (`enable_nvme_train_residency`,
  `hot_rows=64`), ~0.2 GB of pinned slots

The arena is baked once, uncapped, onto a mirrored pair of Seagate FireCuda 530 NVMe
drives; the checkpoint and dataset are pre-fetched so the capped runs are fully offline.
Nothing in the setup path is part of the claim.

The cap then descends until the arm stops completing all four steps. **The verdict comes
from docker, not from the container's stdout** — a process killed by the cgroup dies on
SIGKILL and prints nothing, so "no output" has to be readable as a result.
`State.OOMKilled` is the authoritative field.

## Results

| cap | host | arena |
|---|---|---|
| none | completed | completed |
| 8192 MiB | completed | — |
| 6144 MiB | **completed** ×2 | — |
| 5888 MiB | **completed** | — |
| 5632 MiB | **OOM-killed** | — |
| 5120 MiB | **OOM-killed** ×2 | **completed** |
| 4608 MiB | OOM-killed | — |
| 4096 MiB | — | completed |
| 3072 MiB | — | completed |
| 2560 MiB | — | completed |
| 2304 MiB | — | **completed** ×2 |
| 2176 MiB | — | **OOM-killed** |
| 2048 MiB | — | OOM-killed |
| 1536 MiB | — | OOM-killed |

**Minimum host RAM in which four steps complete:**

| | host-RAM offload | NVMe arena, `hot_rows=64` |
|---|---|---|
| **minimum viable ceiling** | **(5632, 5888] MiB = 5.91–6.17 GB** | **(2176, 2304] MiB = 2.28–2.42 GB** |
| frozen expert bytes | pins all 1024 (3.83 GB of pinned homes, measured in 0.17.0) | ~0.2 GB pinned, 64 hot rows |
| steady RSS at `trained` | 5.88 GB | 2.34 GB |
| uncapped peak RSS | 16.63 GB | 2.34 GB |

(The arena file is **3.62 GB** of packed expert bytes. That is the arena's size, not
the host arm's pinned footprint — the 3.83 GB above is the separately measured
`/dev/zero` mapping total from 0.17.0, and the two are not the same quantity.)

**Ratio of requirements: 2.56×** (bracketed 2.44×–2.71× by the two rungs either side).
Steady RSS gives 2.51× independently.

### The headline cell

At **5120 MiB (5 GiB)** — same model, same seed, same four steps, same box, same cap —
the host path is **OOM-killed (exit 137, `OOMKilled=true`, reproduced twice)** and the
arena path **trains to completion**. That is the case the tier exists for.

### Peak RSS overstates the host arm by 2.7×; steady RSS does not

The host arm's uncapped peak RSS is **16.63 GB**, of which **15.86 GB is file-backed**. It
trains fine under a **6.17 GB** cap. Nothing was tuned between those two runs — the kernel
simply reclaims the mmap'd bf16 checkpoint when RAM is scarce and keeps it when RAM is
free. This is the same effect an earlier A/B/A balloon test caught (`ru_maxrss` 18.57 →
10.80 → 18.56 with bit-identical losses); here it is shown causally, as a hard OOM
boundary that sits **2.7× below** the peak the process reports.

The arena arm has almost no page cache to reclaim (0.81 GB file-backed), so **its** peak
RSS of 2.34 GB lands inside its own measured threshold of 2.28–2.42 GB.

So the two arms disagree about whether peak RSS is meaningful, and that is exactly why the
naive ratio misleads: **peak-RSS ratio 7.10×, requirement ratio 2.56×.** An 8× figure
computed this way, from an earlier run, was wrong for precisely this reason.

### Where each arm dies

The last mark reached before SIGKILL localises the failure. The host arm at 5632/5120 MiB
gets *past* `model_built` — it pins all 3.62 GB of experts successfully at ~5.0 GB — and
dies during the training steps, where RSS climbs to 5.88 GB. At 4608 MiB it dies earlier,
during the build. The arena arm's failures at 2176 MiB and below are likewise in the step
loop. In every case the cap is crossed by the workload, not by the loader.

## What this does and does not establish

**Does:** the mechanism is correct under real memory pressure, on published wheels, and
there exists a machine on which this model can be trained *only* through the arena. The
saving is **~2.5×** for a 3.62 GB expert set, and it is a saving in the quantity that
decides whether the job starts.

**Does not:** this is one model on one box. The host requirement scales with **total**
expert bytes while the arena requirement is set by `hot_rows` and stays roughly flat, so
the ratio should widen substantially on larger MoEs — but that is the mechanism's
prediction, not a measurement, and nothing here measures it. NVMe throughput varies ~7×
between rented pods; these are wall-clock-irrelevant pass/fail results, so that variance
does not affect the thresholds, but it would affect step time. **This box is a correctness
testbed and no timing claim is made from it.**

## Reproducing

```bash
docker build -t e4b-bench:2.8.0-cu128 .          # torch runtime + gcc (Triton JITs through it)
docker run --rm --runtime=nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$PWD":/work -e HF_HOME=/work/hf e4b-bench:2.8.0-cu128 bash /work/prep.sh
./ladder.sh host  8192m 6144m 5888m 5632m 5120m  # descends until two consecutive failures
./ladder.sh arena 4096m 2560m 2304m 2176m 2048m
```

`ladder.jsonl` is the raw ledger, one JSON object per run, each carrying the full
instrument set, the marks, the cgroup limits **as read from inside the container**, and
the resolved package versions. Its first row is a `host/none` run that failed with
`Failed to find C compiler` — that is the pre-`gcc` image, not a memory result, and it is
left in place rather than deleted.

These steps were then run against an **empty** mount, because the original arena was baked
into directories created by hand beforehand and the script inherited that assumption
(caught in review, fixed). The rebake from scratch produced a byte-identical arena —
`md5 91e4c5cf1751933bbff0999a655021e8`, 3,623,878,656 bytes both times — so the bake is
deterministic and the two runs above share an input.
