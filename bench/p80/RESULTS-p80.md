# P80 — read: bucketed CUDA-graph decode is 2.32–2.34× the eager `PagedModelRunner` on a changing active set, NF4 Qwen3-30B-A3B, one RTX 5090

Pre-registration: [`PREREG-p80.md`](PREREG-p80.md) (merged #762, `39f6bc7`, before any run). Issue #511; mechanism #757;
the experiment is the Q2 memo's (adertha-agents#62). Run `p80-5090-1`, 2026-09-28 22:03–22:19Z.
Register row: `e4b.serve.p80.qwen3.dynb.graph-buckets.5090.2026-09-28`.

## Verdict: CONFIRMED (`verdict.json`, computed by `p80_reduce.py`)

| arm | mode | aggregate decode tok/s (timed pass, 992 tokens) |
|---|---|---|
| A1 | eager `PagedModelRunner` (unpadded, default T > 1 grouping) | 121.6 |
| B1 | bucketed graphs (1, 2, 4, 8, 16) | **284.5** |
| B2 | bucketed graphs | **284.5** |
| A2 | eager | 122.5 |
| P | the padded step, eager (B's oracle) | 162.1 |

- **The claim.** B1/A1 = **2.340** and B2/A2 = **2.323**, both above the registered 1.03.
- **Self-pairs.** A2/A1 = 1.0076 and B2/B1 = 1.0000, both inside [1/1.03, 1.03].
- **Correctness.** B1, B2 and P decode identical token streams, bitwise, in all 16 rows. A1 ≡ A2.
- **Engagement.** All five buckets captured (4.8 s), no candidate step ran eagerly, and every arm ran the registered trace
  in both passes, with the warm pass's tokens equal to the timed pass's.
- **Reported, not decisive** (the pre-registered departure from the memo). Graph vs unpadded eager first differs at row
  0, token 130, near the end of the longest row. The gate that isolates the graph is B ≡ P, and it held.
  - **Corrected 2026-09-29 (P81's read).** This line first explained the divergence as A running "other row counts"
    through bf16 GEMMs. **The trace pads no rows**: its active sizes are exactly the bucket sizes, and every bucket's
    `pad_rows` is 0 in these receipts. So A ran the same row counts as P. It differs in the grouping (default vs
    device) and in the step path (the eager step's `append_many` vs the bucket step's `append_graph_bt1` over bound
    bucket state). P81 held the grouping equal and A still differed from P, so the grouping alone does not explain it.
    Which part of the step path moves the tokens is not isolated.

## Where the gain comes from (mean ms per decode step, timed pass)

| active rows | A1 eager | P padded eager | B1 graphs | A1 / B1 |
|---:|---:|---:|---:|---:|
| 16 | 54.54 | 40.68 | 33.79 | 1.61× |
| 8 | 53.73 | 38.89 | 25.88 | 2.08× |
| 4 | 52.90 | 38.86 | 20.46 | 2.59× |
| 2 | 51.54 | 38.37 | 18.90 | 2.73× |
| 1 | 42.31 | 34.39 | 9.95 | 4.25× |

- **Eager decode is host-bound across the whole trace.** A1's step barely moves between 16 rows and 2 (54.5 → 51.5 ms);
  it is paying per-launch host cost, not per-row work. The graph step falls with the row count, and falls most where
  there is least GPU work. That is the stated expectation, and it held.
- **P splits the ratio.** P/A1 = 1.33: capture-safe device grouping plus the bucket step's own path (no rows are padded
  on this trace) is itself faster than the default T > 1 grouping, whose host sync it removes. B1/P = 1.75 is the
  graph's own effect on the same function. *(Corrected 2026-09-29: this line first said "padding plus device grouping";
  `pad_rows` is 0 in every bucket.)*

## What this does and does not establish

- **Established.** On this box, for this model and trace, the bucketed graph cache (#757) more than doubles
  the eager runner's decode throughput and computes exactly the padded step's function. This is also the first real-weight
  check of the graph cache (B511 used a tiny random model).
- **Not established.** Nothing about the licensed int4 stack (NF4 only, per the memo), or families other than Qwen3. Nothing
  about a server loop with arrivals (no admissions after the first step) or about prefill. Nothing about another host:
  the 5090 class disperses about 8.5 % and B = 1 is host-bound, so no absolute is compared to another box's.
  **Not established either: a comparison with the fixed-B graph position** (bo7's 238.1 tok/s at B = 1, p37's NF4 500
  tok/s at B = 16). Those ran other arms on other boxes; this lane compares the dynamic runner with itself.
- **Not the default.** `enable_decode_graphs` stays opt-in. The HTTP shim and `infer` do not call it.

## Box and cost

RTX 5090 (compute capability 12.0, 575 W power limit, driver 580.119.02, 32 607 MiB) on an AMD EPYC 9755 host. Vast
verified/secure, instance `53262053`, `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, 320 GB disk. Versions: e4b 0.37.6 at
`39f6bc7`, grouped-nf4-gemm 0.33.5 (`fb15cf5`), torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1, bitsandbytes
0.50.1. Qwen3-30B-A3B @ `ad44e777`; the fetch and the NF4 bake together took about 6 min (tripwire 22:08:57Z, `BAKE OK` 22:15:13Z). Prompts sha256 `f67e7e4d…`, p37's
B = 16 rows. Peak allocated VRAM 18.97 GiB for eager and 19.01 GiB with graphs.

**Cost $0.157** for the reading (954 s lifetime, estimate $1.125), teardown proven. Lane total **$0.2513**:
- `p80-prove-1`: NOT_RUN, $0.0247; the Vast box failed the launcher's 40 MB/s bandwidth pre-flight at 37.6.
- `p80-prove-2`: REFUSED at $0. **Operator error**: I listed prove-1 as a machine exclusion, which the launcher accepts
  only for an ssh-readiness failure.
- `p80-prove-3`: PROVED, $0.0696.
- `p80-5090-1`: this reading, $0.157.

The launcher receipts, teardown proofs and ledger rows are in the private store at
`receipts/experts4bit-qlora/2026-09-28/p80-*` (commit `e7f6058`).

## Files

`receipts/p80-5090-1/`: the five arm receipts (`p80_{A1,B1,B2,A2,P}.json`, with every row's tokens in both passes and
every step's wall), `verdict.json`, `summary.txt`, `versions.txt`, `forensics.txt`. `receipts/p80-prove-3/`: the proof's
`summary.txt`, `versions.txt`, `forensics.txt`.
