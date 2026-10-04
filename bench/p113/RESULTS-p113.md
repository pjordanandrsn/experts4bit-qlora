# P113 — results: **CAP_DEFAULT**. grouped-nf4-gemm's programmatic dependent launch, capped to launches of at most 8 activation rows, decodes SC1's int4 serving step's tokens exactly and 4.0 % faster with one request, at no cost with sixteen (1.000); uncapped, it costs sixteen 2.1 %

Registration: `bench/p113/PREREG-p113.md` (#1029, `234880c`). Issue: #1015. The switches: grouped-nf4-gemm#448
(`GNF4_PDL`) and grouped-nf4-gemm#453 (`GNF4_PDL_MAX_ROWS`). The lanes it follows: grouped-nf4-gemm K28 (LEVER) and
P112 (closed VOID; its first run's arms, not a reading, are what this lane replicates).

Code under test: e4b `234880c` (0.44.0 with P113's registration), grouped-nf4-gemm `bc2214c` (0.36.0 with
both switches), torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0. The subject is P112's: the default graph server
with SC1's int4_sched levers, on `Qwen/Qwen3-30B-A3B` @ `ad44e77`.

**Verdict by `p113_reduce.py`: `CAP_DEFAULT`.** The rule's steps, in order:

| step | result |
|---|---|
| VOID | no. Every arm captured every bucket with SC1's int4 configuration in force (48 int4 MoE layers, 96 int4 attention projections, every fusion), the switch read as registered (`pdl_active`, `pdl_cap` 8 in CAP and 0 elsewhere), and the accounting held: every build launched 6,927 switched kernels; OFF none with PDL; ALL 6,907 (every handled launch and variant); CAP 5,465, the cap splitting the build |
| NOISY | no. Self-pairs 0.9988–1.0018 |
| FUNCTION_FAIL | no. **Every arm emits OFF1's tokens on every row** of both workloads at both lengths, and every timed rep digests the same |
| ALL_DEFAULT | no. gALL1 = 1.0401 clears 1.02, but **gALL16 = 0.9787** is below 1.00 |
| CAP_DEFAULT | **yes**: gCAP1 = **1.0404** ≥ 1.02 and gCAP16 = **1.0000** ≥ 0.99 |

## The reading (`p113-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 580.178.04) on an AMD Ryzen 9 7900 (124 GiB RAM), Vast instance 54131692 (machine 152169).
**Cost:** $0.4710 by the provider's invoice, including the 64.8 GB checkpoint download. Teardown was proven at
09:56:34Z.

**Timeline (the box's own log):** install at 09:28:38Z; the premise (`test_decode_graph_buckets.py` 7 passed,
grouped-nf4-gemm `test_pdl.py` 25 passed) before the fetch, which ran 09:29:27–09:40:41Z; the bake until 09:41:41Z; six
arms 09:41:41–09:55:36Z; the reducer and `TP_DONE` at 09:55:36Z.

Decode-only rates (each pass's wall minus its largest ttft; p37's slope):

| arm | W1 tok/s | W1 ms/step | W16 tok/s | W16 ms/step | build launches with PDL | load s |
|---|---:|---:|---:|---:|---|---:|
| OFF1 | 228.12 | 4.3836 | 1786.77 | 8.9547 | 0 of 6,927 | 109 |
| ALL1 | 237.77 | 4.2057 | 1754.06 | 9.1217 | 6,907 of 6,927 | 102 |
| CAP1 | 237.35 | 4.2131 | 1790.52 | 8.9359 | 5,465 of 6,927 | 100 |
| CAP2 | 237.62 | 4.2084 | 1790.01 | 8.9385 | 5,465 of 6,927 | 104 |
| ALL2 | 237.55 | 4.2097 | 1751.92 | 9.1328 | 6,907 of 6,927 | 101 |
| OFF2 | 228.39 | 4.3784 | 1789.99 | 8.9386 | 0 of 6,927 | 104 |

| ratio | W1 pairs | W1 geomean | W16 pairs | W16 geomean |
|---|---|---:|---|---:|
| ALL / OFF | 1.0423 / 1.0401 | 1.0412 | 0.9817 / 0.9787 | 0.9802 |
| CAP / OFF | 1.0405 / 1.0404 | 1.0404 | 1.0021 / 1.0 | 1.0011 |

- **One request:** the step falls 4.381 → 4.211 ms with the cap (4.208 ms uncapped). Every
  switched launch at B=1 carries 1 or 8 rows, so the cap changes nothing there.
- **Sixteen requests:** uncapped PDL raises the step 8.947 → 9.127 ms, as P112's first run saw. The cap,
  which launches nothing with PDL in the 16-row bucket, leaves it at 8.937 ms.
- **Whole-pass slopes** (reported, not ruled on) agree within about 1 %: OFF1 228.01 / 1779.88,
  CAP1 236.67 / 1783.32 tok/s (W1 / W16). On this host the prefill was steady.
- **Exact:** the switch changes no token, as PDL changes no value.

## Against the predictions

Written before any P113 data but after P112 run 1's arms (stated in the pre-registration).

| prediction | result |
|---|---|
| Q1: engagement, including CAP's split | **yes** |
| Q2: every arm's tokens equal OFF1's | **yes** |
| Q3: gALL1 in [1.02, 1.06] | **yes** (1.0401) |
| Q4: gALL16 in [0.97, 1.00] | **yes** (0.9787) |
| Q5: gCAP1 in [1.02, 1.06], within 0.01 of gALL1 | **yes** (1.0404) |
| Q6: gCAP16 in [0.99, 1.01] | **yes** (1.0000) |
| Q7: self-pairs within [0.98, 1.02] | **yes** |
| Q8: CAP_DEFAULT (about 60 %) | **yes** |

## The registered consequence (CAP_DEFAULT)

- **grouped-nf4-gemm's next release turns `GNF4_PDL` on by default, with `GNF4_PDL_MAX_ROWS` defaulting to 8.** On
  NVIDIA sm_90+ it is inert elsewhere. `GNF4_PDL=0` turns it off; `GNF4_PDL_MAX_ROWS=0` removes the cap.
- **experts4bit-qlora's CI moves to that release.**
- **The register** row `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04` carries gCAP1 from the verdict file.
- **Scope.** Read on Qwen3-30B-A3B with SC1's int4 configuration on one RTX 5090. e4b's default NF4 server reaches only
  two of the switched kernels (P112's census), so its effect there is unread; the switch is value-identical by
  construction everywhere, which licenses the package default.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p113-5090-1` | **OK, CAP_DEFAULT** | $0.4710 | the reading; no proving rental |

**The lane cost $0.4710**, inside its $4.00 ceiling. With P112's two VOID runs ($1.3420) the question cost
$1.8130 in all.

**Receipts** are in `receipts/p113-5090-1/`, with `SHA256SUMS`: the six arm receipts, `verdict.json`, `summary.txt`,
`forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`, the logs (force-added) and the teardown proof. The
launcher's receipt is in adertha-receipts `6a6f632`; its ledger row landed with `35e7d00`.
