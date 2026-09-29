# P81 — results: bucketed CUDA-graph decode on the licensed int4 stack is **CONFIRMED** (B1/A1 14.29, B2/A2 13.95), and the recipe's first attention pack loads by fingerprint in every arm

Read 2026-09-29 from `p81-5090-2`. Registration: `bench/p81/PREREG-p81.md` (#767, `fa65132`). Issues: #511, #674.
Verdict by `p81_reduce.py` from the five arm receipts: **CONFIRMED**. P1, P2 and P3 held.

The ratio is large because this host's eager decode of the licensed stack is host-bound at ~125–139 ms per step
**whatever the row count**. The number measures that host cost, on an older host CPU (a Zen 2 EPYC 7K62). Read
[§ What the ratio is](#what-the-ratio-is) before quoting it.

## Runs and cost

| run | outcome | cost |
|---|---|---:|
| `p81-prove-1` | **PROVED**: refusals, install at the pins, tripwire, reducer self-test, egress probe; rc 0 | $0.0263 |
| `p81-5090-1` | **NOT_RUN**: the launcher's pre-flight measured 25.3 MB/s of download bandwidth, below its 40 MB/s floor, and refused before the command ran | $0.0445 |
| `p81-5090-2` | **the reading**: lane rc 0, five arm receipts, verdict | $1.1349 |
| **lane** | | **$1.2057** |

All three teardowns are proven (`vast-destroy`, HTTP 200, instance absent afterwards). The bandwidth NOT_RUN was **not**
added to the machine exclusions; that list is for ssh-readiness failures only (P80's prove-2 lesson).

## The reading

| arm | mode | aggregate decode tok/s (timed pass) |
|---|---|---:|
| A1 | eager `PagedModelRunner`, device grouping | 47.8 |
| B1 | bucketed graphs (1, 2, 4, 8, 16) | 683.6 |
| B2 | bucketed graphs | 679.3 |
| A2 | eager, device grouping | 48.7 |
| P | the bucket step, eager (B's oracle) | 57.7 |

- **The claim.** B1/A1 = **14.29** and B2/A2 = **13.95**, both above the registered 1.03.
- **Self-pairs.** A2/A1 = 1.0175 and B2/B1 = 0.9937, both inside [1/1.03, 1.03].
- **Correctness.** B1, B2 and P decode identical token streams, bitwise, in all 16 rows (1,008 decode tokens). A1 ≡ A2.
- **Engagement.**
  - All five buckets captured (4.8 s in B1, 4.0 s in B2), and no candidate step ran eagerly.
  - Every arm ran the registered trace in both passes, and every arm's warm pass decoded the same tokens as its timed
    pass.
  - Every arm ran the device grouping (`device_grouping: true`, `grouping_arg: device`).
  - Peak allocated VRAM was 19.6 GiB in every arm.
- **The pack gate held.** All five receipts carry expert pack `sha256:0c9955a9…` and attention pack
  `sha256:d7cfa1f4…`, and every arm installed the attention from its artifact (`attn_pack_source: artifact`).

## What the ratio is

Mean ms per decode step, timed pass:

| active rows | A1 eager | P bucket step, eager | B1 graphs | A1 / B1 |
|---:|---:|---:|---:|---:|
| 16 | 129.2 | 109.0 | 16.85 | 7.7× |
| 8 | 127.8 | 104.5 | 9.16 | 14.0× |
| 4 | 126.2 | 103.4 | 7.59 | 16.6× |
| 2 | 125.6 | 102.2 | 6.37 | 19.7× |
| 1 | 139.2 | 118.0 | 5.38 | 25.9× |

- **Eager decode of the licensed stack is host-bound, and flat.** A1 takes 125–139 ms per step from 16 rows down to one,
  so the step is paying per-launch host cost, not per-row work. The graph removes that cost, and what remains falls
  with the rows: 16.9 ms at 16, 5.4 ms at one.
- **This host is slow at host work.** It is an AMD EPYC 7K62 (Zen 2), with the card at a 450 W limit. P80's host was an
  EPYC 9755 (Zen 5) at 575 W. Host-bound eager decode is exactly the part a slower CPU inflates. The build shows it
  too: its first calibration chunk took **1,121 s here against P70's 360 s** on the same recipe. So ×14 is this host's
  number. It is not comparable to P80's ×2.3 (another stack, another host), and it is not a portable speedup.
- **What it does say.** On the licensed stack, the eager `PagedModelRunner` spends 112–134 ms per decode step more than
  the graphs on this host (A1 minus B1, by row count), and the graphs remove nearly all of the step. Serving the licensed stack eagerly (the HTTP shim
  and `infer` do not enable graphs) leaves most of the throughput on the table. #511's follow-up, graphs by default, is
  now the larger lever.
- **P splits the ratio, differently from P80.** Here A and P both run the device grouping, and the trace pads no rows
  (`pad_rows` is 0 in every bucket). So P/A1 = **1.21** is the bucket step's own path, and B1/P = **11.84** is the
  graph.

## A ≠ P, at identical rows and grouping (reported, not decisive; a correction to P80's reading)

- **The facts.** A1 and P decode the same prompts through the same weights, the same grouping and the same row counts,
  because the trace's active sizes (16, 8, 4, 2, 1) are exactly the bucket sizes and nothing is padded. Yet they agree
  on 14 of 16 rows and 917 of 1,008 decode tokens. Row 0 first differs at token 129, row 1 at token 67. B1 differs from
  A1 in exactly the same places.
- **What differs between the two steps.** The eager step appends K/V with `Fp8PagedKV.append_many` and attends from
  the scheduler's slot list. The bucket step appends with the graph-mode `append_graph_bt1` over the bucket's bound
  state (`graph_bucket_bind` / `graph_bucket_load`). This lane did not isolate which of these moves the tokens.
- **The correction to P80.** P80's read (`RESULTS-p80.md`, STATUS, the P80 row's notes) explained its A-vs-B divergence
  as "A runs other row counts … bf16 GEMMs round differently at a different row count", and attributed P/A1 = 1.33 to
  "padding plus device grouping".
  - **P80 padded no rows either.** Its receipts show `pad_rows` 0 in every bucket.
  - So in P80, A differed from P in the grouping and in the step path, not in row count.
  - Its 1.33× is device grouping plus the bucket step's path, with no padding.
  - Those documents are corrected in this PR. P80's verdict does not change, because its gate was B ≡ P, which held.

## #674: the packs (reported)

- **The expert pack is the licensed one.** `artifact1/manifest.json` records `sha256:0c9955a9…`, equal to P55x's
  licensed pack, as the builds in P55x, P64 and P70 did.
- **The attention pack exists.** `attn1/manifest.json` records `sha256:d7cfa1f496d75120ef50b72fae0a31928b6ffb04ac0c5ec49c522a673068a422`.
  - Layout `int4_b32.attn.v1`: 192 projections, 385 payloads, revision `ad44e777`.
  - Calibration token stream `sha256:e5d5eba5…`, the same stream the expert calibration names.
  - It verified with `verify_artifact`, and all five arms installed it by fingerprint.
- **It does not reproduce the licensed build's quality reading.** The build's wikitext K8, on the same window
  (`sha=9ef10d760ad9`) through the same expert bytes, reads **6.33015**. P55x, P64 and P70 all read **6.36709**. So
  this build serves a different function from the licensed build's.
  - The attention is the only half not pinned across them. P55x's calibrated attention was never dumped, so there is no
    earlier attention fingerprint to compare with.
  - The software also differs: e4b 0.37.7 here against 0.37.4 in P70. Among the changes, 0.37.5 made the router
    epilogue's bf16 cast the default for `softmax_topk` routers under `E4B_FUSE_ROUTER_EPI=1`, which the recipe sets.
  - This lane did not isolate which difference moves K8 by −0.03694.
  - So `d7cfa1f4…` is the first attention pack of the licensed **recipe**, built on 0.37.7. It is **not** a
    byte-identification of the attention P55x licensed. #674 stays open on that point, and on cross-box reproduction.
- **A provenance defect, found reading the manifests.** The attention manifest's informative `expert_pack_fingerprint`
  is `sha256:c221ab32…`, a fingerprint no artifact carries.
  - `dump_calibrated_artifact` returns the expert artifact's manifest but leaves the model's provenance at the
    live-store fingerprint, which covers the tensor payloads only.
  - The artifact's root fingerprint (`0c9955a9…`) also covers the identity and assignment payloads.
  - The field is marked informative, not identity, and the gate does not read it. It should still name the artifact.
    Filed separately.

## Box, software and time

- **Box.** One RTX 5090 (sm_120, driver 570.153.02, 450 W power limit), Vast.ai verified/secure instance 53286231, AMD
  EPYC 7K62, 251 GiB of RAM ([`receipts/p81-5090-2/forensics.txt`](receipts/p81-5090-2/forensics.txt)).
- **Software.** e4b 0.37.7 at `fa65132`; grouped-nf4-gemm 0.33.5 at `fb15cf5` (the registered pin); torch
  2.8.0+cu128, Triton 3.4.0, transformers 5.16.1, bitsandbytes 0.50.1.
- **Where the time went** (UTC, 2026-09-29):
  - Install 01:35–01:37, fetch 01:37–01:48, bake 01:48–01:50.
  - Build 01:50–03:27 (97 min; first calibration chunk 1,121 s, under the 1,500 s host-limit watchdog), then the two
    pack dumps and verification.
  - Arms 03:27–03:46, 3.5–4.3 min each. Reducer 03:46.
- **Guard use.** The run used 2.23 h of its 2.5 h guard. The build is host-sensitive (3.1× P70's on this CPU), so a
  future lane that builds on the licensed recipe should price that in.

## What this does and does not establish

- **Established.** On this box, the bucketed graph cache (#757) serves the licensed int4 stack, with every bucket
  capturing, buckets 2, 4 and 8 included. It computes exactly its padded eager
  step's function, and it removes a host cost that makes eager decode of this stack 7.7–26× slower per step. Also
  established: both packs of one build load by fingerprint in five processes.
- **Not established.**
  - Nothing about another host: the ratio is dominated by host cost and this host's CPU is slow.
  - Nothing about other families, arrivals, or prefill.
  - Nothing about the attention pack reproducing on another box, or matching P55x's licensed attention (above).
  - Nothing about quality beyond the build's one K8 reading. The arms are not scored, and A ≠ P means the eager and
    bucket steps are not the same function bit for bit.

## Receipts

- **In this repo** ([`receipts/p81-5090-2/`](receipts/p81-5090-2/), [`receipts/p81-prove-1/`](receipts/p81-prove-1/)):
  - The arm receipts, gzipped byte-exact, because the expert pack's per-expert method map makes each 1.4 MB.
  - `verdict.json`, `summary.txt`, `versions.txt`, `forensics.txt`, `packs.json`, `build_ppl_wikitext.json`, and both
    pack manifests.
  - `SHA256SUMS` of every file as the run wrote it.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-09-29/p81-{prove-1,5090-1,5090-2}/` (commits
  `f8eb071`, `e0623f1`, `7a791f8`), with the launcher's `receipt.json` and `teardown-proof.json` and the full fetched
  run, including `logs/`.
