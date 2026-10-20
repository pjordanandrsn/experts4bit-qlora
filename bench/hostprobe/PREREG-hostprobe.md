# Host-facts probe: registration

**What it is.** A rented box that reads the host facts other lanes' registrations name as bounds on their reads, prints
them, and is torn down. It reads files and its own environment. It installs nothing, loads no model, imports no torch,
and measures nothing. It is **never a reading**, and its result never ranks or retires a host.

**Why.** A lane's registration can say that one host fact bounds what its read can conclude. P159's PREREG says so of
the transparent-huge-page mode: under `madvise` or `never`, READ-1 cannot read IDENTIFIED. A host substituted for
another must match every such fact. As of 2026-10-20, the THP mode of the hosts that ran earlier lanes was not
recorded anywhere, and a host can be on the market for only minutes. This probe reads those facts on one named machine,
cheaply, when a decision needs them.

## Run ids and scope

- **`hostprobe-<machine>-<n>`**: one Vast machine id, and n counts from 1 per machine. The drive refuses any other id.
- **The launch pins that machine.** The pin follows the market rule (first posted owns it, posted on the bus before
  create).
- **The scope is reading host facts.** A box that does anything else is outside this registration.

## What the box reads (`hostprobe_box.py`, standard library only)

| fact | source |
|---|---|
| `thp_enabled`, `thp_defrag` | `/sys/kernel/mm/transparent_hugepage/{enabled,defrag}`, the selected mode, only if a known mode |
| `khugepaged_defrag` | `/sys/kernel/mm/transparent_hugepage/khugepaged/defrag` |
| `kernel` | `/proc/sys/kernel/osrelease` |
| `numa_balancing` | `/proc/sys/kernel/numa_balancing` |
| `mem_total_bytes` | `/proc/meminfo` MemTotal |
| `cgroup_memory_max` | cgroup v2 `memory.max`, else v1 `memory.limit_in_bytes` (`max` when unlimited) |
| `online_cpus`, `numa_nodes`, `affinity_cpus` | `/sys/devices/system/cpu/online`, `/sys/devices/system/node/node*`, `sched_getaffinity` |
| `thp_mem_alloc_enable`, `omp_num_threads` | this process's environment, i.e. the ssh session's, not any lane's |
| `measured_at` | the box's UTC clock |

A fact that cannot be read is left out, never guessed. No hostname, container id, serial, UUID or CPU model string is
printed: the receipt's machine id already names the host.

## The drive (`hostprobe_drive.sh`, controller side)

- One ssh call, bounded at 120 s, streams the box script to `python3 -`.
- It writes the one `HOSTPROBE_JSON` line to `hostprobe/facts.json` in the run's receipt directory, beside the log.
- It exits 0 when the facts line came back, and 22 when it did not. No retry: a second probe takes the next id.

## Cost and guard

- **Guard:** 0.25 h.
- **Rate cap:** $0.60/h, GPU plus disk, at Vast's quote. That covers the RTX 4090 class this probe was first wanted
  for; a host above the cap needs its own amendment.
- **Image:** `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` pinned by digest
  (`sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385`, 4.29 GB compressed as read from the
  registry on 2026-10-20). The probe needs only `python3`.
- **Download:** at most $0.011/GB, with the manifest's `download_gb` at 4.5.
- **Worst per probe:** 0.25 h x $0.60 + 4.5 GB x $0.011 = **$0.20**, inside a lane's standing per-run authority.
- **No bandwidth gate** (`preflight_bandwidth: none`): the probe pulls no checkpoint.

## Launch and record

- **Launch.** It uses the launcher current at launch, with `--validate` passing. The approval is the maintainer's
  permalinked comment on this registration's PR, posted after merge.
- **Record.** The receipt goes to the private store. A lane quotes a probe's facts as **that probe's**, with its run id
  and `measured_at`, never as the host's facts in general: a host can change its THP mode.
