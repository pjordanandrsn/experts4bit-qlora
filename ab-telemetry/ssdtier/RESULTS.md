# SSD-Tier (FileStore) — correctness + same-host relative throughput

## Quarantine rule (read first)

This session runs on containerized, shared infrastructure. **No absolute bandwidth or
absolute-latency claim may be made from it.** Overlay filesystems, shared host NVMe, page
cache we cannot drop, and neighbors we cannot see make such numbers fiction. This session
proves correctness, integration shape, and same-host relative deltas. Absolute
characterization is a separate bare-metal session.

## What this session did NOT establish

Absolute read bandwidth, absolute latency, thermal behavior, storage striping / RAID, and
GPUDirect Storage (GDS) — **all deferred to a bare-metal session.** The FileStore read mode
achieved here (`odirect`, see below) makes the *relative* deltas honest by bypassing the page
cache, but no number here is an absolute characterization of any disk.

## Pod / preflight

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (Community Cloud) |
| compute capability | (8, 6) = sm_86 — matches the published A2000 arms' architecture |
| container disk (`/`) | 120 GB overlay on local NVMe; `dd oflag=direct` write 3.8 GB/s, read 4.2 GB/s |
| store location | local overlay (`/workspace` on this pod was a small 20 GB xfs volume; the expert store lives on local disk) |
| **O_DIRECT** | **supported** — FileStore reads run `mode=odirect` (page cache bypassed; timings honest) |
| host RAM / CPU | 251 GB / 112 threads (shared) |
| bnb | `feature/experts-4bit` @ e7f4d86, built `CMAKE_CUDA_ARCHITECTURES=86` → `libbitsandbytes_cuda130.so` |
| axolotl | PR #3797 `expert-offload-integration` @ fd12f92 |

Receipts: `receipts/` (pip freeze, SHAs, driver, device, disk probes).

## Pre-registration

`ab-telemetry/ssdtier/predictions.json` was committed and pushed **before any FileStore code**
(kit `bench/ssdtier-filestore`, commit 38bbefa). Relative/directional predictions only, per the
quarantine rule.

## Phase A — Store seam + correctness gates  ✅

New `store.py` seam on the axolotl fork branch `feature/expert-store`: `RAMStore` (today's
pinned-CPU homes, byte-for-byte) and `FileStore` (packed experts written once to an aligned
read-only file; blocks streamed back through reusable pinned staging buffers; O_DIRECT where the
FS allows, buffered+`posix_fadvise(DONTNEED)` fallback; mode recorded). Eviction, the
single-resident-slot policy, the forward pre-hook, and the gradient-checkpoint recompute contract
are untouched — the seam abstracts only the *source* of a block's bytes.

| gate | result |
|---|---|
| existing expert_offload suite, **unmodified**, over RAMStore (regression) | **37 passed** (32 CPU + 5 CUDA) |
| existing expert_offload suite, **unmodified**, over FileStore (`AXOLOTL_EXPERT_OFFLOAD_STORE=file`) | **37 passed** |
| new store-parity suite (`test_expert_store.py`): SHA256 byte-parity FileStore==RAMStore (both layouts), state_dict parity, mode recording, no-alias contract, ram default | **11 passed** |
| bnb `test_experts4bit.py` on sm_86 (regression guard for the source build) | **59 passed** |

No test was edited to pass. The existing suite ran verbatim over FileStore via an env hook.

FileStore's `mode=odirect` was confirmed **in real axolotl training** (not only unit tests):
arm 3's install logs `homed 32 expert layers across 16 MoE blocks (3.22 GB) to disk (odirect; …)`.

## Phase C — Deterministic prefetch (double-buffer)  ✅

Feature-flagged `expert_offload_prefetch` (file store only): a second staging-buffer set and one
background reader load the predicted next block while the current block computes. The predictor is
direction-aware (ascending forward, descending gradient-checkpoint recompute); a mispredicted block
is discarded and that staging falls back to the synchronous read.

**A bit-exactness gate caught a real bug during development** (recorded per the handoff — this
repro outranks any timing): the reader keyed its buffers by padded byte-size, so a block's
same-size expert slots aliased one buffer — slot 1's read clobbered slot 0 before either was copied
out (the synchronous path is safe because it copies each slot to device between fetches; the reader
holds all of a block's slots as a list first). The `test_offloaded_grads_match_reference` grad-parity
test failed. Fix: key the reader's buffers per slot-index. After the fix:

| gate | result |
|---|---|
| full bit-exactness suite over FileStore **with prefetch on** | **all pass** |
| new `TestPrefetch`: grad-parity vs non-offloaded reference, residency invariant (≤1 staged block), host-cost (+1 block of buffers), misprediction fallback, ram no-op | **passed** |
| CUDA-gated correctness (RTX 3090), both stores × prefetch | ⟨reruns after the training arms free the GPU⟩ |

Peak GPU memory: unchanged by prefetch (the extra buffer is host-side; H2D still happens at stage
time). Asserted by the residency test.

## Phase B — Seed-matched arm table, one host  ✅

OLMoE-1B-7B QLoRA, the published axolotl-ab config/seed/data order, **all arms on this one 3090**
(never compared across hosts). Arms differ by store only.

| Arm | Experts | Store | BEFORE | AFTER | s/step | train mem (max_active) |
|---|---|---|:---:|:---:|:---:|:---:|
| 1 | resident | — | 5.456 | 2.375 | 2.66 | 16.96 GiB |
| 2 | offloaded | RAMStore | 5.456 | 2.376 | 6.22 | 14.15 GiB |
| 3 | offloaded | FileStore (odirect) | 5.456 | 2.371 | 9.25 | 14.15 GiB |

All four arms share **step-1 loss = 5.469 exactly** (seed + data-order matched).

**Parity — the point of Phase B — verified against an in-session control, not an arbitrary epsilon.**
Arm 1 (resident) and Arm 2 (RAM-offload) hold byte-identical weights and differ only in staging, so
`max|Arm2−Arm1| = 0.038` **is this harness's GPU non-determinism floor** (`index_add_` atomics, per
METHODOLOGY §11a). FileStore vs RAM-offload diverges by only **0.033 — at/below that floor**; FileStore
vs resident is **0.023**. Verdict: **PARITY**, corroborated by the SHA256 per-block byte-parity tests
and the CPU bit-exact (`torch.equal`) gradient tests. (An initial fixed 1e-2 epsilon flagged a false
"divergence"; it was simply below the measured noise floor — the criterion is now
`max|file−ram| ≤ max|ram−resident|`.)

### Four-cell s/step (store × prefetch) — same host, quarantined

| | no prefetch | prefetch |
|---|:---:|:---:|
| RAMStore | 6.22 | ≡ 6.22 (no-op by design; reader not created on a RAM store — asserted by test) |
| FileStore (odirect) | 9.25 | **10.06** |

**THE session number (relative, quarantined):** FileStore/RAMStore s/step ratio = **1.49×** (mode=odirect).

**Prefetch direction — a pre-registered prediction, FALSIFIED on this host:** FileStore+prefetch was
*slower* (9.25 → 10.06 s/step), not faster. Mechanism, in the code: the FileStore staging copy is
**synchronous** (`copy_required` → `src.to(device, copy=True)`, `non_blocking=False` — mandatory
because the staging buffers are reused). Prefetch hides the **disk read** but not the **H2D**, and the
reader adds contention, so it cannot close the gap here. The fix — a double-buffered *device-side*
async-H2D path — is a Session-4 (bare-metal) design item, recorded not faked. This is exactly the
handoff's "find out why before trusting any curve."

## Phase D — Routed decode probe (experimental hack)  ✅ (direction only)

Labeled experimental; does not touch the training plugin. Direction + reads-per-token only, absolute
tok/s VOID per quarantine. Resident reference decode ran; the routed-vs-whole-block reads/token
comparison is the recorded direction (`results/S3-decode-probe.json`). The clean numbers are a
bare-metal item.

## Phase E — FusedStore (commercial layer; code + data are NOT in this repo)

Per the program's IP addendum, `FusedStore`, the placement generator, and all knee data are
**commercial-layer assets**: they live in the private LLC repo `Cerin-Amroth/e4b-ssdtier`
(LLC copyright headers, `PROVENANCE.md` recording clean-room authorship). Nothing in this
public bench repo contains them. Recorded here only as the fact that the public seam was
exercised by a third backend.

The **one public change** this required is upstream-safe and lives on the axolotl fork branch
`feature/expert-store`: `install_expert_offload(store=…)` now accepts an already-constructed
store object as well as a kind string — bring-your-own-backend, no behavior change for existing
callers.

Correctness (run on this pod; the numbers are gate results, not commercial data):

| gate | result |
|---|---|
| degenerate ends — `f=1.0` reproduces RAMStore and `f=0.0` reproduces FileStore **exactly** (`torch.equal` on outputs *and* every gradient; SHA256 byte-parity), both expert layouts | **passed** |
| placement math (counts, hash stability, interleaved spreads, contiguous is a prefix) | **passed** |
| midpoint grad-parity `f ∈ {0.25, 0.5, 0.75}` × {interleaved, contiguous} × prefetch {on, off} | **passed** |
| private FusedStore suite total | **28 passed** |
| **public** expert_offload suite, **unmodified**, over FusedStore (`f=0.5` interleaved + prefetch; `f=0.25` contiguous) | **32 passed** each |

The degenerate-ends invariant gated everything downstream and ran before any sweep, as required.

## Deviations from handoff

1. **torch cu130 / CUDA-13 toolkit for the bnb source build** — axolotl's pins upgrade torch to
   2.13.0+cu130; the from-source bnb must match (`libbitsandbytes_cuda130.so`). Same finding as
   Session 1.
2. **Code pushed from the Mac**, not via a pod-side PAT — no GitHub credential placed on the rented
   pod (tighter than the handoff's env-PAT).
3. **Parity criterion is control-calibrated**, not a fixed epsilon — a 1e-2 epsilon flagged a false divergence; replaced with the byte-identical resident-vs-RAM control as the noise floor (see Phase B).
4. **FusedStore/knee are in the private LLC repo**, not here (IP boundary; see Phase E).

## Cost ledger

RTX 3090 Community @ ~$0.31/hr, single pod, ~2.7 h wall (preflight + env build + Phases A/C code +
3-arm table + arm4 + CUDA gates + knee sweep + decode probe) ≈ **$0.85**. Well inside the $3–5 budget.
