# DeepSeek-V4.1-Flash-1017, gate B: memory, residency and transfer bounds (2026-10-17)

An analytic model at $0: no weights downloaded, nothing built, nothing rented. It asks where the checkpoint's bytes can live
on the cards and hosts we use, and what decode speed the transfers allow. **Every tok/s here is a transfer bound, not a
measurement**: the bytes a token must move, over the slowest link they cross. Compute, kernel and Python costs come on top.

`ds41_gate_b.py` reproduces every number. The bytes come from the checkpoint's shard headers (gate A, `bench/ds41-gate-a/`,
revision `1bae0e10`), and each rate comes from the record named below.

## Inputs

**Read from the checkpoint and its reference code:**

| quantity | value | source |
|---|---|---|
| weights a text decode keeps on the GPU, as stored (attention, shared experts and router, head, Engram projections, hyper-connections) | 8.52 GB | shard headers; the embedding (1.32 GB) is a lookup and stays on the host |
| routed experts | 288.78 GB: 18.80 MB per expert, 4.512 GB per token uncached (6 × 40) | shard headers |
| Engram tables | 202.76 GB; 48 rows of 264 B per token (12.7 KB), weights and scales in separate tensors | shard headers, `model.py:344` |
| KV cache, bf16 | 3,200 B per token: compressed KV and index keys at the four source layers (ratios 2, 2, 2, 1), plus a fixed 5.2 MB window; 0.43 GB at 128K, 3.36 GB at 1M | `model.py:437`, `:496`, `:614` |

**Rates and hosts (measured):**

| host | H2D | disk read (O_DIRECT) | host RAM | record |
|---|---|---|---|---|
| our A2000 host, 12 GB card | 5.8 GB/s (gen3 x8) | 1.63 GB/s; 0.743 GB/s on a cold load. Both were measured on the GPU container's work pool, **not** the pool that holds this checkpoint (below) | 131.8 GB total, **51.5 GB available** (read 2026-10-17T13:55Z; the rest serves the host's own services). The card: 12,282 MiB, 1,584 MiB in use with no compute app, so **11.22 GB free** (read 14:02Z) | `bench/q235/gate2/hostrate.json`, `bench/q235/gate2/README.md` |
| rented, PCIe gen 4 | 21.76 GB/s | 6.31 GB/s | 1,081 GB on that box; Q235 rents to a 239 GB floor | Q235 box 1 `hostrate.json` `2ce01906`, `host.json` `4d5e0251` (measured-private); `bench/q235/PREREG-q235.md` Amendment 5 |
| rented, PCIe gen 5 | 40–53 GB/s, **unmeasured** (an assumption) | — | — | `bench/q235/am7/PREDICTION-C.md` |

**Assumptions:**
- **Routing is uniform**, so a cache of k experts hits k / 15,360 of the time. There is no V4.1 routing record; informed hot
  sets did better on other models (`e4b.serve.informed-hot-sets`), so this assumption leans pessimistic.
- **Reserves:** 1 GB of VRAM for the CUDA context, activations and workspace at batch 1. 8 GB of host RAM is kept for the OS
  on the rented boxes, where the RAM figure is a total; the A2000 host's 51.5 GB is already what is available.
- **Card sizes are GiB, as sold:** 12, 24, 48 and 96 GiB are 12.88, 25.77, 51.54 and 103.08 GB. The A2000 uses its measured
  free memory.
- **The MTP / DSpark layers (7.93 GB) and the vision tower (0.97 GB) are left out:** neither is read by text decoding without
  speculation.

## Decode at batch 1, 128K context: where the bytes live, and the bound

Placement: the GPU caches what fits after its resident weights. The Engram tables go to host RAM or to disk, whichever gives the
higher bound (both are shown). The remaining RAM caches experts, and the rest of the experts stream from disk.

| host | GPU | Engram tables | expert hits: GPU / RAM | binding link | **decode bound** | the other placement |
|---|---|---|---|---|---|---|
| A2000 host | 12 GiB (11.22 GB free) | disk | 0.4 % / 17.4 % | disk | **0.44 tok/s, provisional** (0.20 at the cold-load rate; the rates are another pool's) | (the tables do not fit in RAM) |
| rented, 239 GB floor | 12 GiB | disk | 0.9 % / 79.5 % | H2D | **4.87 tok/s** (gen 5: 7.2, then disk-bound) | in RAM: 1.56 |
| rented, 239 GB floor | 24 GiB | disk | 5.4 % / 79.5 % | H2D | **5.10 tok/s** (gen 5: 9.3, then disk-bound) | in RAM: 1.64 |
| rented, 239 GB floor | 48 GiB | disk | 14.3 % / 79.5 % | H2D | **5.63 tok/s** (gen 5: 10.3–13.7) | in RAM: 1.83 |
| rented, 239 GB floor | 96 GiB | disk | 32.2 % / 67.8 % | H2D | **7.11 tok/s** (gen 5: 13.1–17.3) | in RAM: 2.39 |
| rented, 1,081 GB (box 1) | 12 / 24 / 48 / 96 GiB | RAM or disk | 0.9–32.2 % / the rest | H2D | **4.87 / 5.10 / 5.63 / 7.11 tok/s** (gen 5: up to 11.9–17.3) | the same |

- **The Engram tables belong on disk unless RAM is plentiful.** They cost 12.7 KB per token, against up to 4.5 GB per token of
  experts, so the RAM is worth more to the experts. On the 239 GB floor that moves the bound from about 1.6–2.4 tok/s
  (disk-bound) to 4.9–7.1 (H2D-bound).
- **The price is random reads.** The tables then need 96 random reads per token (a weight row and a scale row per lookup).
  They bind only below about 470–680 reads per second at these bounds, which any NVMe drive should exceed, but nothing here
  measures it: the placement rests on it (gate C, item 1).
- **Full host residency** needs about 298 GB of host RAM with the tables on disk, or 501 GB with them in RAM. The 239 GB floor
  holds 79.5 % of the experts; box 1 had 1,081 GB.
- **Disk:** the box needs about 492 GB of local disk for the experts and the tables. Box 1 had 697 GB free.
- **Context:** at batch 1 the A2000's free memory holds 0.46M tokens of KV, and a 12 GiB card 0.98M. The model's maximum is
  1,048,576, which needs a 24 GiB card or larger.
- **The A2000 host** keeps the tables and 82.3 % of the experts on disk, and the disk binds. Its tables bind only below 42 random
  reads per second.
- **But that row's disk rate is the wrong pool's.** The checkpoint (revision `1bae0e10`, all 48 shards) is already on the
  host, on its model pool: one 12-wide raidz2 vdev of HDDs, 128 KiB records, compression off. There, a random 264 B row read
  costs a whole 128 KiB record, so the pool's random-read rate is roughly one disk's: an estimated 100–200 reads/s, against the
  42/s the row needs. That pool's sequential rate is unmeasured too, and could be higher or lower than 1.63 GB/s. The A2000
  row is **provisional** until gate C item 1 measures that pool.

## Training: the per-step bytes (#1681's arithmetic)

With e4b's single-slot whole-layer staging, a pass stages M × (2n − 1) whole layers, and here n = 40.

| | stagings | GB per step | at 21.76 GB/s H2D (experts in RAM) | from disk at 6.31 GB/s |
|---|---|---|---|---|
| 1 micro-batch | 79 | 570.3 | 26.2 s | 90.4 s |
| 4 micro-batches (Q235's Shape B) | 316 | 2,281.3 | 104.8 s | 361.5 s |

- **A 12 GB card cannot stage a whole layer.** One layer is 7.22 GB, and with the 8.52 GB resident that makes 15.74 GB before
  activations, LoRA and the optimizer.
- **A 24 GB card might.** Whether it does depends on the activations, which nothing here measures.
- **Routed staging** (#1681's option (c)) would cut both the bytes and the slot. It needs V4.1 routing records, which don't
  exist.

## What gate C would need to test

1. **The disk's random-read rate: the key measurement.** It covers Engram's 256 B weight rows and 8 B scale rows (separate
   tensors), read at random from our NVMe and pool disks and the rented box class. The tables' placement, and so every headline
   bound above, rests on it. It needs no model: a read probe over a file of the tables' size will do. The sequential read rate
   for 18.80 MB expert reads comes with it. On the A2000 host it can run against the real tables, which are already on its HDD
   pool: read-only, at $0.
2. **A `deepseek_v41` module tree** that loads and runs one layer with real weights (gate A: none exists). Nothing below can
   run without it.
3. **The routing distribution:** the experts' hit rates on real text. They replace the uniform assumption, which drives every
   cache figure above.
4. **The VRAM reserve at batch 1,** measured on one layer, in place of the assumed 1 GB.
5. **Gen-5 H2D** on the host it would rent, if a large-RAM gen-5 box is the candidate.
