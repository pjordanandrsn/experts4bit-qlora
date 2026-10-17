# Expert tiling and streaming: where the break-even sits, from existing records

An analytic note: no runs, no GPU, no change to any path. It asks whether streaming routed experts host-to-device in
sub-expert **tiles** would beat the layouts e4b already has or has sized: the whole layer, the routed whole experts,
and layer-major. It answers from records only. `roofline.py` reproduces every derived number:

    python roofline.py --h2d 21.76 --h2d 40 --h2d 53

The worked case is Qwen3-235B-A22B under Q235's Shape B (4 micro-batches, 94 layers, checkpointed), because that is the
only training lane with staging records. Sections 1–2 were first worked from #1680's prediction. The GPU-side check is now
in: the Shape B box's per-layer read, "A7 measured" below. It confirms the ordering.

## Two different "tiles"

- **K19's split row tables (gnf4#542)** tile the *activation rows* of a grouped GEMM whose weights are already in VRAM.
  - A heavy expert (a median 398 of 4,096 routed rows per 512-token chunk on the Qwen3-30B census) gets full 32- or
    64-row tiles; the rest keep 16-row tiles.
  - That cuts weight passes per N-block from 307 to 204 (`big_m` 64).
  - It reduces **VRAM and L2 re-reads**, and moves no host-to-device bytes. Its speed is unread.
- **Weight-tile streaming**, the subject here, splits each routed expert's *weights* into pieces that cross PCIe
  separately. Each piece is computed and then evicted.
  - That changes the H2D schedule and the VRAM working set, not the arithmetic.
  - The two are independent: weight tiles could feed K19's kernel unchanged.

## Inputs (records)

| record | what it gives |
|---|---|
| `bench/q235/PREREG-q235.md` | Qwen3-235B-A22B: hidden 4096, expert intermediate 1536, 128 experts, top-8, 94 layers |
| box 1 `box/box.json` `27a7863e` (via #1680, #1681) | 254,124,490,752 B in 187 whole-layer stagings: **1,358,954,496 B per layer**, so 10,616,832 B per expert (9,437,184 packed NF4 + 1,179,648 fp32 absmax) |
| box 1 `hostrate.json` `2ce01906` (via #1680) | pinned H2D 21.76 GB/s, PCIe gen 4 x16 |
| `bench/q235/am7/PREDICTION-C.md` (#1680) | experts compute (forward, recompute, backward) **15.13 ms per micro-batch per layer** on the PRO 6000, predicted from measured ratios: **5.69 s per Shape B pass**. Staging-bound: C = 8.25 (5.0–11.0) at 21.76 GB/s |
| `bench/q235/staged-bytes/README.md` (#1681, #1682) | whole-layer stagings per Shape B pass: as-is 748, layer-major 187, routed 548.8, both 167.9. The four micro-batches' routed union is 0.898 of the experts. Layer-major needs a held forward evict (CPU toy: 749 / 470 / 187) |
| `experts4bit_qlora/engines/offload.py` docstring | Qwen3-235B-A22B inference at 22.5 GB/s, per token: bulk 5.57 s, routed staging **0.93 s**, routed + fused 0.71 s. Routed cannot combine with prefetch; prefill and training fall back to bulk |
| gnf4 `bench/phase3/flagship/RESULTS-flagship-offload.md` | Qwen3-235B per-expert copies, 4 per expert (0.39 to 6.29 MB each), ran at **101 %** of the 1 GiB pinned microbench (44.28 GB/s, H100) |
| gnf4 `bench/cold-engine/RESULTS-hybrid-phase0.md` | 8 KiB pinned H2D latency **3.3 µs** per copy (gen 5) |
| gnf4 `bench/cold-engine/RESULTS-p2-g1b.md` | **19.8 µs per row** of host dispatch: two `cudaMemcpyAsync` enqueues plus a pinned staging write |
| `bench/p66/RESULTS-p66.md` | on one gen 4 x16 host, a single synchronized 64 MB copy read 14.72 GB/s against 23.07 back-to-back, and a pipelined per-row gather ran at the single-copy rate (14.4 GB/s at h = 0) |
| `bench/dq5/RESULTS-dq5.md` | layer prefetch on a 5090 (gen 4): step 1.0050x the resident step, but 20–59 of 62 forward copies still in flight when their layer started |
| `docs/OFFLOAD-TRANSFER-NOTES.md` (A2000) | halving copies per stage (4 to 2) moved 0.08 ms of 36.8: per-copy cost negligible at 216 MiB stages |
| gnf4 `.../RESULTS-1469-locality-bytes.md` | slicing experts along the intermediate channels "does not move fewer bytes" under exact routing |

Not on record anywhere:
- an H2D rate-vs-size sweep between 1 and 25 MB;
- a measured routed staging in training;
- a GPU roofline for these kernels.

## 1. Bytes: tiling cannot cut them

Exact routing computes every tile of every routed expert, so a tiled schedule moves exactly the bytes of the layout it
tiles. RESULTS-1469 says the same for slice caching. The byte floor for a checkpointed pass is each layer's routed union
staged once forward and once in the recompute. That is (b)+(c): **167.9 whole-layer equivalents = 228.2 GB per Shape B
pass**, 22 % of as-is. Tiling (b)+(c) leaves it at 228.2 GB.

**In training, the recompute schedule decides whether tiles cost bytes.** e4b keeps a layer resident from its recompute
through its backward, so a layer crosses PCIe twice per micro-batch. Under that layer-level schedule, tiles evicted
after the recompute must cross a third time for the backward: **+50 % bytes**, unless every tile stays resident, and
then the memory saving is gone. A tile-fused op that recomputes one tile and runs its backward while the tile is
resident would keep two crossings. That is a new kernel-level schedule, not a property of tiling itself.

## 2. Time: overlap is the lever, and a forward cannot prefetch routed experts

All compute below is #1680's **experts-only** figure: a lower bound, without attention, the dense layers, the router or the
loss. Per Shape B pass, layer-major is two sweeps:
- the forward: experts compute 1.337 s;
- the recompute + backward: 4.350 s.

**A forward cannot prefetch layer L+1's routed experts during layer L.** L+1's routes come from L's output (the
`offload.py` note that keeps routed staging off prefetch). In training that holds for layer-major too. The recompute can
prefetch them, because the forward fixed every route. Whole-layer prefetch needs no routes, so it works in both sweeps.

Layer-major schedules per pass at 21.76 GB/s (`roofline.py` also prints 40 and 53). "Own bound" is the sum over sweeps
of max(staging, compute) for that schedule's bytes. "Floor" is the same with routed bytes in both sweeps, 10.49 s.

| schedule | s per pass | / own bound | / floor |
|---|---:|---:|---:|
| (b) serial (today's synchronous staging) | 17.37 | 1.487 | 1.656 |
| (b) whole-layer double buffer, both sweeps | 11.80 | 1.011 | 1.125 |
| (b)+(c): forward routed serial, recompute routed buffered | 11.88 | 1.133 | 1.133 |
| **hybrid: forward whole-layer buffered, recompute routed buffered** | **11.20** | 1.011 | **1.068** |
| (b)+(c): forward per-expert pipelined, recompute routed buffered | 10.59 | 1.010 | 1.010 |

- **Serial staging fails the 15 % stop line**, at 1.487 of its own bound.
- **Every schedule that overlaps both sweeps sits within about 1 % of its own bound.** The exception is the one that
  serves the forward's routed experts serially, at 1.133.
- **Routed staging in the forward helps only if experts stream one at a time inside the layer.** Expert e+1 copies
  while expert e computes, once that layer's router has run. Served serially, routed forward staging is slower than
  whole-layer prefetch (11.88 s against 11.80 s at 21.76 GB/s; 8.56 against 7.58 at 40 GB/s). The hybrid is the best
  schedule that needs no intra-layer pipeline.
- **Per-expert pipelining is the one place sub-layer granularity earns anything.** It takes the hybrid from 1.068 to
  1.010 of the floor, about 5 % of a pass. Its unit is the whole routed expert. Sub-expert tiles would shrink only its
  per-layer fill, from one expert copy (0.49 ms, 46 ms over the pass's 94 layers) to one tile, under 0.5 % of a pass.
- **Double buffering costs one more slot in VRAM:** a whole layer of 1.36 GB, or a routed one of 1.22 GB. Box 1 (Shape
  A) recorded a device peak of 58.9 GiB of 96 (`PREREG-q235.md` Amendment 4). That figure is an upper bound: it included
  four fp32 adapter check copies that Amendment 4 moved to the host.
- **The overlap is not free on every host.** DQ5's gen-4 5090 still had 20–59 of 62 forward prefetches in flight when
  their layer began.

### The A7 read (GPU validation)

The Shape B box `q235-pro6000-3` (RTX PRO 6000, PCIe gen 4 x16) passed its verdict. The figures come from its
`box/box.json` (`8b19c4c2`, measured-private) and `hostrate.json` (`c82685f9`).

**Staging.**
- A step staged 1,016,497,963,008 B in 748 copies: 1.359 GB per whole-layer call, absmax kept fp32 because offloaded
  stacks refuse double quantization. The calls are full layers, as modelled.
- Means over 8 sampled layers × 8 calls, without layer 0's resident forward calls: 51.40 ms forward, 50.81 ms recompute.
  That is **26.6 GB/s effective**.
- The same box's `hostrate.json` read 20.39 GB/s: 1 GiB pinned copies, back to back, CUDA-event timed. Staging ran at
  **1.30x** its own probe.
- **The probe under-measures this host's H2D**, so the staging rate of record on this machine is **26.6 GB/s** (25.8–27.9
  per layer). The receipts settle it:
  - The bytes are exact: 1,016,497,963,008 B / 748 copies = 1,358,954,496 B, 128 experts × 10,616,832 B.
  - The timer does not under-bracket. The box loads with prefetch off, and the copies run on the stream the stage
    events are recorded on. Independently of the timer, the timed pass moved 1,016.5 GB in 44.1 s: at least 23.0 GB/s
    even if staging overlapped everything, and at least 26.2 GB/s with the 5.29 s of experts compute taken out.
  - The provider's listing for the machine at launch gave pcie_bw 26.6 GB/s (the run's `instance.json`, `295f1dc7`).
  - The probe copies one 1 GiB buffer. Its D2H (28.29 GB/s) is unaffected while its H2D is low, which points at that
    one source buffer's placement. Which mechanism (NUMA or IOMMU) is not in the receipts.
  - Receipts: `q235-pro6000-3` `box/box.json` (`8b19c4c2`), `hostrate.json` (`c82685f9`), `instance.json` (`295f1dc7`).

**Experts compute** per micro-batch per layer: forward 4.76 ms, recompute 4.62, backward 4.68, so 14.07 against #1680's
15.13. The split differs from the prediction: the forward is heavier and the backward lighter.

**Re-run on the measured inputs** (`roofline.py`, "A7 measured"), per pass against the routed floor:

| schedule | @ 20.39 GB/s (the probe, an under-read) | @ 26.6 GB/s (staging, of record) |
|---|---:|---:|
| (b) serial | 1.586 | 1.730 |
| (b) whole-layer double buffer | 1.125 | 1.125 |
| (b)+(c): forward routed serial | 1.165 | 1.214 |
| **hybrid** | **1.068** | **1.068** |
| forward per-expert pipelined | 1.010 | 1.010 |

**The ordering holds on the measured box.** The box is still staging-bound per sweep: 93 or 94 stagings of about 51 ms
against 1.79 s of forward compute and 3.50 s of recompute + backward. So the levers and the stop-rule reading are
unchanged.

## 3. When tiles would matter: VRAM, not time

Tiling earns its place only where **two layer slots do not fit**: a card whose free VRAM after the resident weights,
activations, LoRA and optimizer is under about 2.7 GB (2 × 1.36) for this model, or a model with larger layers. There, a
tile slot of t / 1536 of a layer replaces the second layer slot.

The per-expert working set at tile width t is 10.6 MB × t / 1536. For the 94 experts a micro-batch routes on average,
that is 1.0 GB at t = 1536 and 0.17 GB at t = 256.

Section 8 checks every recorded offload lane against this regime: none needs a unit smaller than one whole expert.

## 4. Which dimension streams

Expert weights are stored `gate_up` [E, 2I, H] (gate rows, then up rows) and `down` [E, H, I], with one absmax per 64
consecutive values.

| tile along | intermediate it needs | contiguous on the host? |
|---|---|---|
| **I** (the expert intermediate) | [rows, t] per tile; the output accumulates into [rows, H] across tiles | `gate_up`: two contiguous row runs per tile. `down`: **no**. An I-slice of [H, I] is H = 4,096 strided segments of t / 2 bytes (128 B at t = 256), so it needs a one-time host re-layout to I-major tiles first |
| H as `gate_up`'s K | fp32 partial sums of [rows, 2I] alive until every H tile has arrived, because the activation needs the full dot product | yes (column runs of 64-value absmax blocks), but it holds the whole 2I intermediate |
| H as `down`'s output | none, but each tile needs the whole [rows, I] intermediate first | needs `gate_up` whole |

**Only I-tiling bounds the intermediate per tile**, and it needs the `down` re-layout. Its cross-tile accumulation also
reorders the down projection's K-sum, so an I-tiled expert is not bit-identical to the whole-expert kernel. Any registered
result would need a parity bar rather than `torch.equal`.

## 5. DMA size

- **No record sweeps H2D rate across 1–25 MB.**
- **Async back-to-back copies of 0.39–6.29 MB already run at link rate.** The flagship's 4-per-expert copies ran at
  101 % of the 1 GiB microbench.
- **The per-copy costs on record are microseconds:** 3.3 µs latency at 8 KiB, and 19.8 µs per row of two enqueues.
- **The serialized floor** (each copy paying latency + one enqueue in series, an inference) gives 0.86 efficiency at
  t = 256 (1.77 MB) and 0.61 at t = 64 (0.44 MB) at 21.76 GB/s. It is worse at faster links: 0.72 and 0.39 at 53 GB/s.
- **The host matters more than the size.** One gen-4 host's single copy reached only 64 % of its back-to-back rate
  (p66), and DQ5's 5090 reached 97 %.
- **A tile prototype would first need that 0.25–16 MB sweep**, back-to-back and pipelined, on the rented host class.

## 6. Routing extremes

- **Dense:** the four micro-batches' union is already 0.898 of the experts, and the largest micro-batch (662
  positions) hits 0.82. Routed staging saves little; layer-major is the lever.
- **Sparse:** decode, at T = 1 with 8 of 128 experts, is the case routed staging was built for: 0.93 s per token
  against 5.57 bulk, 5.95x. Tiling adds nothing; the bytes are already per expert.
- **Skewed:** one expert taking about 10 % of the rows (the Qwen3-30B census). Its bytes are no larger; its compute
  is. That is K19's split-table problem on the compute side, not a streaming problem.

## 7. Recompute, backward and capture

- **Recompute and backward.** Covered in section 1. With tiles evicted, the backward re-streams them (+50 % bytes),
  and the held-forward and layer-major rules (#1682) would have to hold per tile instead of per layer.
- **Graph capture.** A fixed slot with a fixed copy count captures; routed copies do not. Their sizes and sources
  depend on the router's output, which the host must read before it can enqueue copies. That is the sync that keeps
  routed staging from combining with prefetch today. Tiling routed experts multiplies those host-decided copies.
  - gnf4's `build_group_tiles_device` is capture-legal, but it builds compute tiles on the device and moves no host
    bytes.
  - A capture-legal routed stream would need worst-case fixed copies, which spend the bytes routing saves, or a
    device-driven copy path, which nothing on record has.

## 8. Close-out: no recorded lane is in section 3's regime

`closeout.py` reproduces the table. Peaks and budgets come from the records named in it, and the layer and expert sizes are
arithmetic from each family's committed shapes (NF4 packed + fp32 absmax, as offloaded stacks keep it). "Spare" is the
stated budget minus the recorded single-slot peak, so it is the room for a second slot or a smaller unit.

| lane | card | spare GiB | layer slot GiB | one expert MiB | 2nd layer slot | 1 whole expert | kind |
|---|---|---:|---:|---:|---|---|---|
| Q235 Shape B, `q235-pro6000-3` | RTX PRO 6000 96 GB | 31.90 | 1.266 | 10.1 | fits | fits | run |
| Q235 Shape A bf16 | RTX 5090 32 GB | 1.01 | 1.266 | 10.1 | **no** | fits | plan (stretch class, never run) |
| R2 Mixtral-8x22B, shape B fp32 | RTX 5090 32 GB | 3.75 | 1.266 | 162.0 | fits | fits | plan |
| TC3 fused offload, `tc3-4090-8` | RTX 4090 24 GB | 12.92 | 0.316 | 2.5 | fits | fits | run |
| TC2 Mixtral-8x7B offload, `tc1-5090-26` | RTX 5090 32 GB | 23.59 | 0.738 | 94.5 | fits | fits | run |
| METHODOLOGY 11c, Gemma-4-26B-A4B | RTX A2000 12 GB (shared) | 1.70 | 0.399 | 3.2 | fits | fits | run |
| METHODOLOGY 11c, Qwen3-30B-A3B | RTX A2000 12 GB (shared) | 2.92 | 0.316 | 2.5 | fits | fits | run |
| METHODOLOGY 11a, OLMoE-1B-7B | RTX A2000 12 GB (shared) | 7.20 | 0.211 | 3.4 | fits | fits | run |

- **One row cannot hold a second layer slot:** Q235 Shape A at bf16 on the RTX 5090, a registered stretch class that has
  never run. Its 1.0 GiB margin must also hold the unmeasured staging transient (`PREREG-q235.md`, refusal point 3). Overlap
  there would need a sub-layer second buffer, and whole experts are that unit: 10.1 MiB each, about 100 of them in 1.0 GiB.
- **Every row holds at least 24 whole experts in its spare room** (the tightest is R2's 3.75 GiB against Mixtral-8x22B's
  162 MiB experts). Routed staging and per-expert pipelining already work in whole experts, so a sub-expert tile is not
  needed for overlap anywhere on record.
- **Two recorded fits fail on the single slot itself, and tiles fix neither:**
  - TC3 on the shared A2000 at micro-batch 2 (`tc3/README.md`): the recipe peaks at 11.88 GB on the 4090, against 10.3 GB
    free, so it is 1.58 GB over. The whole Qwen3-30B layer slot is 0.34 GB, so no tile size closes it. Micro-batch 1 fits
    (10.46 GB).
  - `mxfp4-arena-train`'s 284B on a 23.5 GiB A5000 (`RESULTS-284b-ladder-fp8.md`): the binding allocation is the staged
    stack's full-`[E]` destination (2.12 GiB per layer). The record's own next step is a compacted stack of the routed
    experts, whole experts of about 8.3 MiB each, which needs its own prereg. That is a routed-staging change, not tiling.
- **The tile-fused recompute + backward kernel** (section 1) would only keep tiles at two crossings. It moves no fewer bytes
  than the whole-layer layout and saves under 0.5 % of a pass (section 2), so without a VRAM case it has nothing to win.

**Tiling is closed, not deferred.** Reopen only for a lane where the spare room beside one layer slot cannot hold one whole
routed expert, shown from that lane's records.

## Verdict

- **Sub-expert weight-tile streaming is closed** (section 8). It cannot reduce bytes. Under today's layer-level recompute it
  adds 50 %. Its time gain is at most a smaller per-layer fill, under 0.5 % of a pass. No recorded lane needs a unit smaller
  than one whole expert.
- **The levers, in order:**
  1. Layer-major, which needs a held forward evict (#1682).
  2. Whole-layer double buffering on a side stream. The async-reuse stress record (`bench/async-reuse-20261017`)
     covers its correctness.
  3. Routed staging in the recompute only: the hybrid, which needs the grad guard lifted, an unrouted-read guard and
     a parity test (#1681 (c)).
  4. Routed staging in the forward, but only with per-expert pipelining inside the layer. Served serially it loses to
     whole-layer prefetch.
- **Reopen tiles** only on section 8's trigger: a lane whose spare room beside one layer slot cannot hold one whole routed
  expert. Before any prototype there, run the copy-size sweep from section 5.
- **The A7 read confirms it.** Measured compute and staging leave the ordering and the ratios to the floor unchanged.
  Staging ran 1.30x the box's host-rate probe because the probe under-measures this host's H2D (section 2, "The A7
  read").

Nothing here changes Q235, a default or a registered result.
