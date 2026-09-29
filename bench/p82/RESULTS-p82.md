# P82 — results: on the licensed int4 stack with the fp32 router, the fixed graph path decodes **exactly** the eager runner's function and is 12.0–12.4× faster on this host (**CONFIRMED**); for #674, the attention pack reproduces across boxes but the K8 reading does not, and the router cast is not the whole gap

Read 2026-09-29 from `p82-5090-3`. Registration: `bench/p82/PREREG-p82.md` (#783, `06c115b`). Issues: #511, #674,
#777. Verdict by `p82_reduce.py` from the five arm receipts and their router stamps: **CONFIRMED**. P1–P4 held.

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---:|
| `p82-prove-1` | **NOT_RUN**: the launcher's pre-flight measured 21.5 MB/s of download bandwidth, below its 40 MB/s floor | $0.0358 |
| `p82-prove-2` | **PROVED**, lane rc 0; receipt **ALARM** (below) | $0.1038 |
| `p82-5090-1` | **NOT_RUN**: the instance was still `created` after 600 s | $0.0798 |
| `p82-5090-2` | **HARNESS_ERROR**: the box died a minute after the lane started (below) | $0.9734 |
| `p82-5090-3` | **the reading**: lane rc 0, five arm receipts, two K8 arms, verdict | $0.8510 |
| **lane** | | **$2.0438** |

Every teardown is proven (`vast-destroy`, HTTP 200, instance absent). The lane stayed under its $2.75 ceiling. The
ledger prices every Vast receipt as `dph_total × measured runtime`, and the receipts themselves say "not an invoice".

- **`p82-prove-2` ALARM.** The proof itself passed: rc 0 and PROVED. The tripwire passed on sm_120 (#777's routing,
  #413's append, `CAST_WEIGHTS False`), as did the 21-case reducer self-test. That host's pip install took ~9 minutes,
  so the 0.2 h wallclock guard fired at 12:48:16Z, while the controller was fetching. All 18 entries were fetched and
  nonce-checked. The registration's condition, "a proof returns rc 0 with its receipts fetched", was met.
- **`p82-5090-2`.**
  - From 13:03Z every ssh to the box was refused, and Vast listed the instance `exited` / `intended_status: stopped`.
  - The driver's lane-dead check needs an answering box, so it would have waited out the full 3 h guard.
  - The owner killed the idle driver at ~14:42Z. The launcher then tore the box down and recorded the window at the
    running rate.
  - Filed as #784.
- Neither NOT_RUN was added to the machine exclusions: the launcher accepts only "did not authenticate" pre-flight
  failures as exclusion evidence.

## The reading (`p82-5090-3`)

One RTX 5090 (sm_120, driver 595.71.05, 500 W), AMD EPYC 7R13, 818 GiB of RAM
([`forensics.txt`](receipts/p82-5090-3/forensics.txt)).
- **Software.** e4b 0.37.8 at `06c115b` and grouped-nf4-gemm 0.33.7 at `9407d49`, with torch 2.8.0+cu128, triton
  3.4.0, transformers 5.16.1 and bitsandbytes 0.50.1.
- **Timeline (UTC).** Install 14:44 and fetch 14:45–14:52. The build ran 14:53–15:57: its first calibration chunk took
  740 s, against P81's 1,121 s. The five arms ran 15:57–16:08, K32 and K16 16:08–16:17.

| arm | mode | aggregate decode tok/s (timed pass) |
|---|---|---:|
| A1 | eager `PagedModelRunner`, device grouping | 67.6 |
| B1 | bucketed graphs (1, 2, 4, 8, 16) | 811.9 |
| B2 | bucketed graphs | 814.2 |
| A2 | eager, device grouping | 65.9 |
| P | the bucket step, eager | 81.3 |

- **P2, identity (decisive, new).** **A1 ≡ P ≡ B1 ≡ B2, bitwise, in every row.** The five comparisons (B1–P, B2–P,
  B1–B2, A1–A2, A1–P) all find no divergence. On the licensed int4 stack the eager runner, its bucket step and the
  graph replay now decode one function. In P81 they did not, and the difference was #413's append and #777's bucket-1
  slot. B771b showed the same on NF4.
- **P1, the claim.** B1/A1 = **12.006** and B2/A2 = **12.358**, both above 1.03. Self-pairs: A2/A1 = 0.974 and B2/B1 =
  1.003, both inside [1/1.03, 1.03]. A2/A1 sits close to the lower edge.
- **P3, engagement.** Every bucket captured in B1 and B2, no B step ran eagerly, and P ran all five buckets eagerly.
  Every arm ran the registered trace, decoded the same tokens in its warm and timed passes, and ran the device
  grouping.
- **P4, the router.** Every arm's sidecar reads `router_epi_cast_env "0"`, `router_epi_cast_weights false`. So do the
  build's and K32's. K16 casts (`router_epi_casts_softmax_topk true`).
- **The pack gate held.** All five receipts carry expert pack `sha256:0c9955a9…`, the licensed one, and attention pack
  `sha256:d7cfa1f4…`, each installed from its artifact by fingerprint.

Mean ms per decode step, timed pass, against P81's (buggy path, EPYC 7K62):

| active rows | A1 | P | B1 | A1 / B1 | P81 A1 | P81 B1 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 91.1 | 82.0 | 13.60 | 6.7× | 129.2 | 16.85 |
| 8 | 88.1 | 73.8 | 8.06 | 10.9× | 127.8 | 9.16 |
| 4 | 87.9 | 72.1 | 6.64 | 13.2× | 126.2 | 7.59 |
| 2 | 89.3 | 72.0 | 5.49 | 16.3× | 125.6 | 6.37 |
| 1 | 102.2 | 81.2 | 4.39 | 23.3× | 139.2 | 5.38 |

- **Host cost, again.** Eager decode is flat at 88–102 ms per step from 16 rows to one, on this host as on P81's. The
  graphs remove it, and what remains falls with the rows. This host is faster at host work than P81's, so the ratio is
  lower (12.0 against 14.3), and neither ratio travels. The graph step is 0.81–0.88× P81's at every row count.
- **The one-row phase.** The fixed path attends over a context that keeps growing, where the buggy path's froze. Here
  its one-row step is 4.39 ms. That is not comparable to P81's 5.38 ms, because the hosts differ.
- **Decomposition.** P/A1 = 1.203 is the bucket step's own path: same grouping, no padding. B1/P = 9.98 is the graph.

This read supersedes P81's register row with
`e4b.serve.p82.qwen3.licensed-int4-fp32router.dynb.graph-buckets.fixed-path.5090.2026-09-29`.

## #674: the packs and K8 (reported, not decisive)

**The packs.**
- The build's expert pack is `0c9955a9…`, the licensed one. Builds on four different machines have now produced
  it: P55x's and P70's host, P64's, P81's and this one.
- The attention pack is **`d7cfa1f4…`, byte-identical to P81's**, calibrated on a different box (EPYC 7K62 / driver
  570.153.02 there, EPYC 7R13 / 595.71.05 here) and on e4b 0.37.7 there against 0.37.8 here. So the recipe's calibrated
  attention reproduces across boxes. That answers the question #674 left open, for this pair of boxes.

**K8**, wikitext, 2,048 steps, window `9ef10d760ad9`:

| reading | router | this box | the licensed builds | P81 |
|---|---|---:|---:|---:|
| build (calibrating process) | fp32 | **6.36396** | 6.36709 | — |
| K32 (both packs loaded by fingerprint) | fp32 | **6.36396** | — | — |
| K16 (both packs loaded by fingerprint) | cast (0.37.5's default) | **6.31811** | — | 6.33015 |

- **K32 equals the build exactly.** A process that loads both packs by fingerprint serves the calibrating process's
  function on this box.
- **The registered table reads "THE CAST IS NOT THE WHOLE GAP"**, because the fp32 router reads 6.36396, not
  6.36709. On one box and one set of packs, the cast moves K8 by −0.0458 ppl, which is ln(6.31811 / 6.36396) =
  **−0.0072 nats**. That is under the family's 0.0095 K8 floor, as P70's INDISTINGUISHABLE read predicts.
- **The finding that matters more: K8 does not reproduce across boxes on this stack.**
  - P81's build and this lane's K16 served **byte-identical packs with the same router setting**.
  - Their code on K8's path is the same. e4b 0.37.7 → 0.37.8 changes only graph-mode, training and dump-provenance code
    (#777, #765, #772). grouped-nf4-gemm 0.33.5 → 0.33.7 refactors `quantize_kv_fp8` into the same operations and changes only
    the graph-mode append and the MXFP4 prefill.
  - They read 6.33015 and 6.31811. The spread is 0.012 ppl, 0.0019 nats.
  - So a five-decimal match to 6.36709 is **not** an identity test across boxes. The registered table assumed it was.
- **What that means for the fp32 residual.** The fp32 reading here is 0.0031 below the licensed 6.36709. That is
  smaller than the cross-box spread just measured, so it cannot be attributed to software.
  - The licensed reading itself came from two machines: P55x's and P70's host is one Ryzen 7950X on driver 575.57.08,
    and P64's is an EPYC 7C13 on **595.71.05, this box's driver**. P64 is on older software.
  - So a driver version alone does not explain the residual. Software from before 0.37.5, or the machine, might.
- **Not isolated:** why K8 moves between these boxes (the driver, Triton's timing-based autotune, or something else).
  The registration's next step, a software bisect, assumed K8 was box-invariant. The discriminating test is a
  **same-box A/B**: P70's software (e4b 0.37.4, gnf4 0.33.0) against 0.37.8 / 0.33.7, both builds with the fp32
  router. If they agree, K8 depends on the box. If they differ, it depends on the software.

## What this establishes

- On the licensed int4 stack (calibrated int4 experts and attention, T=1 folds, fused router epilogue at fp32
  weights), `enable_decode_graphs` with #777 on grouped-nf4-gemm 0.33.7 decodes bit-identically to the eager
  `PagedModelRunner` with the device grouping, over a trace that includes a one-row phase.
  - On this host the graphs are 12.0–12.4× faster.
  - This supersedes P81's corrected row and completes #511's measurements on NF4 (B771b) and int4 (here).
- The calibrated attention pack `d7cfa1f4…` reproduces byte-for-byte on a second box and a second release.
- On one box and identical packs, the router cast moves K8 by −0.0072 nats.

**Not established.**
- Any ratio for another host.
- Other families, arrivals, or prefill.
- Whether the licensed K8 6.36709 is reproducible on current software, and why K8 moves between boxes.
- Which attention P55x licensed: its attention was never dumped.
- Any quality beyond these three K8 readings.
- `enable_decode_graphs` stays opt-in. #770 is where making it the default is decided.

## Receipts

- **In this repo:**
  - [`receipts/p82-5090-3/`](receipts/p82-5090-3/): the five arm receipts, gzipped byte-exact (`gzip -9 -n`), 1.4 MB
    each for the expert pack's method map; every `*.router.json`; `k8_K32.json`, `k8_K16.json` and
    `build_ppl_wikitext.json`; `verdict.json`, `summary.txt`, `versions.txt`, `forensics.txt` and `packs.json`;
    both pack manifests; and `SHA256SUMS` of every file as the run wrote it.
  - [`receipts/p82-prove-2/`](receipts/p82-prove-2/): the proof's summary, versions and forensics.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-09-29/p82-{prove-1,prove-2,5090-1,5090-2,5090-3}/`
  (commits `a8f6406`, `7e8d0fc`, `de6963c`, `b23533c`, `1b5c356`), with each run's `receipt.json` and
  `teardown-proof.json`, and the fetched runs with `logs/`.
