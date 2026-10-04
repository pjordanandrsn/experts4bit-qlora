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

## Amendment 1 (2026-10-04, before any P55 data): the launcher can now draw the class, and the harness reads the class correctly

Nothing has run under P55. Preparing the launch found four defects. Each would have spent money for a wrong reading or no reading:

1. **The launcher could not draw the class.** adertha's Vast provider searched `cpu_ram >= 98 GB`, and nothing could change that. Every P55 launch would have drawn a host the lane calls "class not drawn". adertha-agents#145 adds a host-RAM band (`vast_host_ram_gb` in the manifest → `--vast-min-ram-gb` / `--vast-max-ram-gb`). It caps the search server-side and client-side, and the pre-flight refuses a box above the cap. A read-only offer search on 2026-10-04 (no rental) showed 35 verified RTX 5090 offers at ≤ 72 GB, from $0.43/h, several of them 63 GB. **P55 launches with `vast_host_ram_gb: [48, 72]`.**
   - Why 48: the opaque `invalid argument` failure this lane is about was seen at 64 GiB. The 30 GiB host gave a clean, readable `mmap` refusal, which is a different outcome, so the floor keeps the draw on the opaque side.
   - The ceiling stays the registered 72.
2. **STOP-1 read the wrong number.** On Vast an offer's `cpu_ram` is the container's allotment, but `/proc/meminfo` MemTotal inside the container is the whole host's. A real 63 GB rental on a 512 GB host would have read 512 and fired STOP-1. The class is now the memory a process there can actually have: **min(MemTotal, the cgroup memory limit)**. `bench/p55/p55_ram.py` computes it, it is staged and pinned, and the threshold is still 72 GiB. `forensics.txt` also records `ulimit -a` (RLIMIT_AS and RLIMIT_MEMLOCK are the per-process limits a marginal mapping or a pinned copy can hit).
3. **P4 measured the wrong headroom.** The C_headroom trace sampled `/proc/meminfo` MemAvailable, which inside a container is also the host's. The trace now records the cgroup's usage and limit beside it. P4 reads the headroom as **min(MemAvailable, limit − usage)** per sample. The prediction and its refutation are otherwise unchanged.
4. **The shard's size mixed units.** The largest shard is `model-00001-of-00002.safetensors`, **49,907,246,508 bytes**: 49.91 GB, which is **46.48 GiB**. Every "49.9 GiB" above means 46.48 GiB, and P4's threshold is 46.48 GiB. The checkpoint is **51.6 GB** in total (the second shard is 1.70 GB), not the ~12 GiB the budget section assumed (the Hub's blob listing at `4d7ae49`). The download, not the load, is the long pole.

**Budget, restated.**
- **Rate and estimate.** The RTX 5090 rate is now fixed by policy at $0.85/h (adertha-agents#142), and the launcher prices the download. Its estimate is $0.85 × 1.0 h + 100 GB × $0.011 = **$1.95**.
- **Actual cost.** Expected at about half that: well under an hour on a ~$0.45–0.55/h box, plus ~52 GB of download at the host's rate.
- **Ceiling.** The lane ceiling is **$3.00**, one box (STOP-3 unchanged), inside the owner's standing $15 no-ask tier.
- **Time.** The guard stays 1.0 h. At the launcher's bandwidth floor the 51.6 GB download takes at most about 20 minutes, and each load reads the shard from disk.
- **Authorization.** It is relayed on #344 before launch. The 2026-09-21 note there said a loader-debug rental needed its own word. The owner's later standing tier ($15 no-ask, 2026-09-26) and the delegation of owner decisions (2026-10-01) supply it, and the relay says so.

**Unchanged:** the question, the arms and their order, P1–P3, the decision rule, STOP-2 to STOP-4, and what the lane cannot say.

**Tests.** `tests/test_p55_staged_pin.py` covers:
- the pins, and that every pinned file is staged;
- the effective-memory arithmetic on v1, v2 and absent cgroups;
- the shard in GiB;
- the headroom term;
- the reducer on synthetic receipts: P4 HOLDS on cgroup headroom while the host's MemAvailable stays high, P4 REFUTED, and P1 withheld on STOP-1.
