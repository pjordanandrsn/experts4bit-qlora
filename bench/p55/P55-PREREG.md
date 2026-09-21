# P55 — the 49.9 GiB shard: is #344's host class a memory-headroom class?

Written before any P55 data exists. Bug under test: [#344](https://github.com/pjordanandrsn/experts4bit-qlora/issues/344) — `google/gemma-4-26B-A4B-it` fails inside `load_moe_4bit_streaming` with `CUDA error: invalid argument` on 2 of 6 rented hosts, after the experts quantise, with no attention involved. Instrumentation under test: `E4B_LOAD_SYNC_DEBUG=1` (staged `torch.cuda.synchronize`) and the always-on shard-read diagnosis.

## The fingerprint, which is what makes this lane cheap

Nobody had put the six hosts' forensics side by side. Done now, from `~/.vast/{genA..genE,parity}/forensics.txt` on the mini and the receipts quoted in #344:

| host | GPU / driver | CPU | host RAM | Gemma-4 outcome |
|---|---|---|---|---|
| `parity` (49749308) | 5090 / 580.105.08 | — | **30 GiB** (31 GiB cgroup) | **`safe_open` refused**: `unable to mmap 49907246508 bytes … Cannot allocate memory (12)` |
| `genE` (P24-GEN-E) | 5090 / 580.159.03 | Ryzen 9 9950X (consumer) | not recorded | **FAILED** — `CUDA error: invalid argument` at the **first** expert read |
| box 49753736 | 5090 / 580.159.03 | — | **64 GiB** | **FAILED** — same error, in the **non-expert** pass |
| `genB` | 5090 / 580.95.05 | EPYC 7502 (server) | not recorded | **OK**, twice |
| box 49779056 | 5090 / — | — | 96 GiB | OK |
| box 49804632 | 5090 / 580.159.03 | EPYC (server) | 125 GiB | **OK under `CUDA_LAUNCH_BLOCKING=1`** |
| box 49760421 | 5090 / — | — | 188 GiB | OK |

Three facts the table settles, and one it opens:

1. **Driver is refuted, again and by construction.** 580.159.03 appears on both sides (genE fails, 49804632 passes); a third 580.159.03 host was rented deliberately for #344 and passed the whole bake under launch-blocking.
2. **The GPU is refuted.** Every host is the same RTX 5090, 32607 MiB.
3. **It is not "the box is broken".** `genE` ran granite bakes, five K8 perplexity runs and six decode arms — roughly two hours of CUDA — cleanly, and failed only on the one model in the corpus that ships a **49.9 GiB single shard**. Gemma-4's two shards are 49.9 GiB + 1.7 GiB; every other family in the lane shards at ~5 GiB.
4. **The axis that orders every outcome is host RAM against that shard.** 30 GiB refuses the map outright; 64 GiB maps it and dies opaquely; 96 / 125 / 188 GiB pass. The two failures and the one clean refusal are the same failure family at three severities, and all three are inside the `safe_open(device="cuda")` mapping — which maps the whole 49.9 GiB shard and copies each tensor to the GPU out of that mapping.

That last one is a **reading of six points, not a mechanism**, and it is what this lane tests. It also changes the experiment: #344 says to re-draw the two failing machine ids, which are unrentable. If the class is a memory-headroom class, the class is rentable — cheaply, because low-RAM boxes are the cheap ones.

## The question

**Does a 5090 host with ~64 GiB of host RAM reproduce `CUDA error: invalid argument` on this checkpoint, and does the instrumentation name where?**

## Arms

One box. `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (the snapshot sha in the failing receipts), `load_moe_4bit_streaming(device="cuda", quant_type="nf4")`, no bake, no arms, no timing. Run in order; each records its own outcome and the run continues:

| arm | env | what it answers |
|---|---|---|
| `A_baseline` | none | does this host class reproduce #344 at all, on stock settings? |
| `B_sync` | `E4B_LOAD_SYNC_DEBUG=1` + `CUDA_LAUNCH_BLOCKING=1` | where, exactly — which stage, and under launch-blocking which launch |
| `C_headroom` | as B, with `MemAvailable` sampled every 2 s through the load | the host-memory trace across the failure, which the fingerprint predicts falls |

`A` runs first and unarmed on purpose: launch-blocking changes timing, and a bug that only appears without it would be hidden by an instrumented-only lane.

## Predictions, registered before the data

- **P1 — reproduction.** On a 5090 host with ≤ 72 GiB host RAM (or cgroup limit), `A_baseline` fails with `CUDA error: invalid argument` **or** with an explicit host-memory error from `safe_open`. *Refuted by:* a clean load. A clean load on ≤ 72 GiB is the single most informative refutation available here and ends the memory-headroom reading in one draw.
- **P2 — the instrumentation names the stage.** `B_sync` reports a `[sync]` stage bound, and the diagnosis lines print the shard name, its size, and the host's `MemTotal` / `MemAvailable` / cgroup limit. *Refuted by:* a failure that still reports `<between stages>`, or a diagnosis that prints `unknown` for MemTotal on a Linux box.
- **P3 — the fault is in the mapping, not in an e4b kernel.** Under `CUDA_LAUNCH_BLOCKING=1`, the failing call is a shard read (`safe_open(...).get_tensor`) or the map itself — **not** `quantize_4bit`, not a grouped-nf4-gemm kernel, not an attention kernel. *Refuted by:* launch-blocking naming any e4b or gnf4 kernel. **That refutation would be the most valuable result this lane can produce** and it reopens #344 as a loader bug; it is registered here so that outcome cannot be reported as a surprise.
- **P4 — headroom falls.** In `C_headroom`, `MemAvailable` at the moment of failure is below the size of the largest shard still to be read. *Refuted by:* ample `MemAvailable` at failure, which would refute the mechanism while leaving the host-class correlation standing (P1 could hold with P4 refuted; say so plainly if it does).

## Decision rule, registered in advance

- **P1 ∧ P3 hold** → #344 is a **host-class refusal with a recorded fingerprint**: the loader is not at fault, the class is "host RAM marginal against the largest shard", it goes into `docs/STATUS.md` and the issue with this table, and the loader's contribution is the diagnosis that now names it. The lane blacklist gains a *criterion*, not two more machine ids.
- **P1 ∧ P3 refuted** → the faulting kernel is named; #344 becomes a live loader/kernel bug with a reproduction, and that is the next lane.
- **P1 refuted** (clean load on a low-RAM host) → the memory-headroom reading is wrong, is recorded as refuted, and #344 stays "host-specific, unreproduced" with two of its leads now dead instead of one. **No second box.**
- **P2 refuted** → the instrumentation is the defect; it is fixed before anything else is read, because a debug mode that does not bound the fault is the thing this lane exists to deliver.

Every outcome above is publishable. A refusal with this fingerprint recorded is an acceptable end state for #344; a silent pass on four of six hosts is not.

## What this lane cannot say

Nothing about any other checkpoint, nothing about training or serving throughput, and nothing about *why* a marginal mapping surfaces as `invalid argument` rather than as ENOMEM — that is a driver question this lane can only bound, not answer. It cannot exonerate the two original machine ids: they are unrentable, and a class reproduction is not proof that those two boxes failed for the class's reason.

## Budget and STOP rules

- **One box, RTX 5090 (vast verified/secure), ≤ 64 GiB host RAM, ≤ 1 h wallclock, `max_dph` $0.60 → estimate ≤ $0.60; lane ceiling $1.00, hard stop $1.50.** Under $2, so the executing seat approves in the run thread (`docs/compute-policy.json`, band `max_usd 2`). No proving run: guard ≤ 1 h.
- The checkpoint download is ~12 GiB (`snapshot_gib 11.96` in the genB receipt) and dominates the wallclock; the load itself is ~30 s. The guard is 1 h so a slow link cannot leave a box up.
- **STOP-1** — the drawn box reports **> 72 GiB** of host RAM or an absent cgroup limit above that: the arms still run and are recorded, but the lane reports **class not drawn** and P1 is neither confirmed nor refuted. A pass on a big-RAM host says nothing and must not be written up as if it did.
- **STOP-2** — `A_baseline` fails with anything other than a CUDA error or a host-memory error (a download failure, a missing dependency, an OOM on the GPU): harness fault, recorded, no verdict, no redraw inside this lane.
- **STOP-3** — no second box on any outcome, including a disappointing one.
- **STOP-4** — the driver refuses to start unless `E4B_SHA` is the commit of a clean tree (the `p54_drive.sh` rule), and unless `bench/p55/staged.sha256` matches every staged file.

## Receipts

Fetched to `receipts/experts4bit-qlora/<date>/p55/`: `logs/<arm>.log` (full loader output, which is where the diagnosis lands), `result_<arm>.json` (status, exception type, exception message, notes, the `[sync]` stage bounds), `mem_trace.csv` (arm C), `forensics.txt` (GPU, driver, CPU, `/proc/meminfo`, cgroup limit, shard sizes), `versions.txt`, `summary.txt`. `RESULTS-p55.md` is generated from those files and nothing else.
