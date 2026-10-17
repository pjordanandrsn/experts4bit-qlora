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
| our A2000 host, 12 GB card | 5.8 GB/s (gen3 x8) | 1.63 GB/s; 0.743 GB/s on a cold load | 131.8 GB total, **51.5 GB available** (read 2026-10-17T13:55Z; the rest serves the host's own services) | `bench/q235/gate2/hostrate.json`, `bench/q235/gate2/README.md` |
| rented, PCIe gen 4 | 21.76 GB/s | 6.31 GB/s | 1,081 GB on that box; Q235 rents to a 239 GB floor | Q235 box 1 `hostrate.json` `2ce01906`, `host.json` `4d5e0251` (measured-private); `bench/q235/PREREG-q235.md` Amendment 5 |
| rented, PCIe gen 5 | 40–53 GB/s, **unmeasured** (an assumption) | — | — | `bench/q235/am7/PREDICTION-C.md` |

**Assumptions:**
- **Routing is uniform**, so a cache of k experts hits k / 15,360 of the time. There is no V4.1 routing record; informed hot
  sets did better on other models (`e4b.serve.informed-hot-sets`), so this assumption leans pessimistic.
- **Reserves:** 1 GB of VRAM for the CUDA context, activations and workspace at batch 1, and 8 GB of host RAM for the OS.
- **The MTP / DSpark layers (7.93 GB) and the vision tower (0.97 GB) are left out:** neither is read by text decoding without
  speculation.

## Decode at batch 1, 128K context: where the bytes live, and the bound

Placement: the GPU caches what fits after its resident weights. If the Engram tables fit in host RAM they go there, since their
reads are random. The remaining RAM caches experts, and the rest of the experts stream from disk.

| host | GPU | feasible | Engram | expert hits: GPU / RAM | binding link | **decode bound** |
|---|---|---|---|---|---|---|
| A2000 host | 12 GB | yes | disk | 0.6 % / 14.6 % | disk | **0.43 tok/s** (0.19 at the cold-load rate) |
| rented, 239 GB floor | 12 GB | yes | host RAM | 0.6 % / 9.3 % | disk | **1.55 tok/s** |
| rented, 239 GB floor | 24 GB | yes | host RAM | 4.8 % / 9.3 % | disk | **1.63 tok/s** |
| rented, 239 GB floor | 48 GB | yes | host RAM | 13.1 % / 9.3 % | disk | **1.80 tok/s** |
| rented, 239 GB floor | 96 GB | yes | host RAM | 29.7 % / 9.3 % | disk | **2.29 tok/s** |
| rented, ≥ 501 GB RAM | 12 GB | yes | host RAM | 0.6 % / 99.4 % | H2D | **4.85 tok/s** (gen 5: 8.9–11.8) |
| rented, ≥ 501 GB RAM | 24 GB | yes | host RAM | 4.8 % / 95.2 % | H2D | **5.07 tok/s** (gen 5: 9.3–12.3) |
| rented, ≥ 501 GB RAM | 48 GB | yes | host RAM | 13.1 % / 86.9 % | H2D | **5.55 tok/s** (gen 5: 10.2–13.5) |
| rented, ≥ 501 GB RAM | 96 GB | yes | host RAM | 29.7 % / 70.3 % | H2D | **6.86 tok/s** (gen 5: 12.6–16.7) |

- **Full host residency needs about 501 GB of host RAM**: the experts, the Engram tables, the embedding and the OS reserve.
  The 239 GB floor holds the Engram tables but not the experts, so those boxes are disk-bound. Q235's box 1 had 1,081 GB.
- **Context limits only the 12 GB card.** At batch 1 it holds 0.70M tokens of KV, short of the model's 1,048,576. The 24 GB
  card holds the full 1M.
- **The A2000 host keeps the Engram tables on disk.** Its 0.43 tok/s bound leaves out their 96 random reads per token. Those
  reads bind only below 41 random reads per second, which any SSD exceeds. That host's random-read rate is unmeasured.
- **Gen 5 only helps when H2D binds**, which means the large-RAM boxes. The disk-bound rows don't move.

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

1. **A `deepseek_v41` module tree** that loads and runs one layer with real weights (gate A: none exists). Nothing below can
   run without it.
2. **The routing distribution:** the experts' hit rates on real text. They replace the uniform assumption, which drives every
   cache figure above.
3. **The disk:** on the rented box class, the sequential read rate for 18.80 MB expert reads, and the random-read rate for 264 B
   Engram rows from separate weight and scale tensors.
4. **The VRAM reserve at batch 1,** measured on one layer, in place of the assumed 1 GB.
5. **Gen-5 H2D** on the host it would rent, if a large-RAM gen-5 box is the candidate.
