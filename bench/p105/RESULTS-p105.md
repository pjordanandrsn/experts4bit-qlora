# P105 — results: **SUPPORTED, recommend**. With flash-linear-attention and causal-conv1d installed, Qwen3.6-35B-A3B's hybrid paged path holds its dense parity premise and replays every decode graph exactly; graph decode runs 1.135× (W16) and 1.118× (W1) the torch path's on the same host

Registration: `bench/p105/PREREG-p105.md` (#939, `eeffabb`). Issue: #928. Code under test: e4b at `eeffabb`,
grouped-nf4-gemm at `34da93d`, `flash-linear-attention==0.5.2`, `causal-conv1d==1.7.0`, torch 2.8.0+cu128 held.

**Verdict by `p105_reduce.py` (P103's rule): `SUPPORTED`, `recommend: true`.**
- All three phases are SUPPORTED by P98's rule: every bucket (1–16) captured and replayed with no eager step, and the
  graph arm's tokens equal the padded-eager oracle's on all 17 requests.
- The engagement records are exactly as registered.
- Phase fc's graph arm decodes at least 1.05× phase t's on both workloads.

## The reading (`p105-5090-1`)

One RTX 5090 (sm_120, driver 595.91.07) on an Intel Xeon E5-2696 v4, Vast machine 152323 (pre-flight 112.4 MB/s).
$0.3686, 2,478 s; destroyed and proven absent at 2026-10-03T07:26:57Z.

**Premise per phase (the registered sets):**
- **t:** 11 passed.
- **f:** 9 passed.
- **fc:** 9 passed.

The dense hybrid stayed within its control on every seed, in every phase, under the stand-in and the real fp8 kernel.
The two single-seed tiny-MoE checks, reported only, failed in both kernel phases (hybrid 7.6e-2 and 8.1–8.2e-2,
against bounds of 4.05e-2), as P104 found and as predicted.

| decode, tok/s | t (torch path) | f (fla) | fc (fla + causal-conv1d) | f ÷ t | fc ÷ t |
|---|---:|---:|---:|---:|---:|
| W16, graphs (arm g) | 472.7 | 525.0 | **536.5** | 1.111× | **1.135×** |
| W1, graphs (arm g) | 90.1 | 98.9 | **100.7** | 1.097× | **1.118×** |
| W16, plain eager (arm p) | 66.1 | 71.1 | 73.6 | 1.076× | 1.113× |
| W1, plain eager (arm p) | 7.01 | 7.57 | 7.88 | 1.080× | 1.125× |
| W16, padded eager (arm e) | 77.6 | 67.0 | 67.6 | 0.863× | 0.871× |
| W1, padded eager (arm e) | 7.37 | 6.35 | 6.43 | 0.862× | 0.873× |

| graph step | t | f | fc |
|---|---:|---:|---:|
| ms at 1 row | 12.02 | 10.80 | 10.39 |
| ms at 16 rows | 26.72 | 23.88 | 23.46 |

- **Under graphs the kernels save 1.6 ms per step at 1 row and 3.3 ms at 16 rows.** That is about 13 % and 12 %.
- **fla carries 82–83 % of the gain** (W16 and W1); causal-conv1d adds the rest.
- **Peak GPU memory** is 23.76 GB (g), 23.71 GB (e) and 22.74 GB (p) in every phase.
- **The eager arms are slow on this host** (146 ms per plain-eager step at one row, against P101's 42 ms on a Ryzen 9
  7950X). Eager hybrid decode is launch-bound, as P101 found. The graph arms are nearly host-independent: 12.0 ms and
  26.7 ms per step against P101's 12.6 and 28.1.
- **The padded-eager oracle got slower with the kernels (0.87×)** while plain eager got faster. This is not explained
  here. The padded steps run only the five bucket sizes, so a per-shape cost in the kernels' Triton launch path is a
  candidate, but nothing here tests it. Arm e is the oracle for exactness, not a serving configuration.
- **The kernels change greedy trajectories.** Phase f's and fc's graph tokens agree with phase t's on 41–42 % of W16
  positions and 53–57 % of W1 positions. The kernels' arithmetic differs from the torch path's, and free-running
  greedy decode diverges for good after one flipped argmax. Quality against the torch path in nats is **not
  measured**. What the dense parity premise does bound is the paged path's error against transformers running the
  same kernels.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| SUPPORTED, `recommend` true | **yes** |
| the dense parity premise holds in every phase | **yes** |
| fc graph over t, W16, 1.15–1.60× | **no: 1.135×**, just under the band |
| fc graph over t, W1, 1.02–1.15× | **yes** (1.118×) |
| f's graph gain at least 80 % of fc's | **yes** (82 % on W16, 83 % on W1) |
| fc plain eager over t on W16 at least 1.15× | **no: 1.113×** |
| at least one single-seed MoE check above its bound in a kernel phase | **yes** (all four) |
| peak GPU memory at or below 28 GB | **yes** (23.76 GB) |

**The W16 prediction overestimated the kernels' share of the 16-row step.** It came from the A2000's per-call times,
where the torch recurrent rule's per-layer cost at 16 rows was large. On the 5090, under graphs, the whole Gated
DeltaNet saving is 3.3 ms of a 26.7 ms step.

## The registered consequence (SUPPORTED, recommend)

- `docs/SERVING.md` lists `flash-linear-attention` and `causal-conv1d` as supported for hybrid paged serving and
  recommends them, with this gain, the install line, and two caveats:
  - fla's chunked prefill depends on chunk boundaries, as transformers' own does;
  - token streams differ from the torch path's.
- Register row: `e4b.serve.p105.qwen36-gdn-kernels.5090.2026-10-03`.
- #928 closes.

## Box and cost

| run | status | machine | cost | note |
|---|---|---:|---:|---|
| `p105-prove-1` | OK, PROVED | 96642 | $0.0594 | premise t 11, f 9, fc 9 |
| `p105-5090-1` | **OK, SUPPORTED** | 152323 | $0.3686 | the reading, 2,478 s |

- **The lane: $0.4280.**
- **The kernel question across its three lanes: $0.6137** (P103 $0.1330, P104 $0.0527, P105 $0.4280).

Receipts (`receipts/p105-5090-1/`):
- the nine arm records, the three engagement records, `verdict.json`, `summary.txt`, `forensics.txt` and
  `versions.txt`;
- the premise, single-seed, kernel, fetch, bake and arm logs;
- `work/bake.json` and the teardown proof;
- `SHA256SUMS` over every file the box wrote.

The launcher's receipts and ledger rows are in the receipt store (adertha-receipts `72d2e50`, `1386fff`).
