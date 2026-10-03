# P110 — results: **AT_PARITY**. On the default `serve_paged` server, the arithmetic decode graphs bring (device grouping and bucket padding) costs no measurable quality: teacher-forced, it sits at +0.0004 nats against the eager default, inside the eager default's own neutral perturbations, and a halved decode scale reads +1.05

Registration: `bench/p110/PREREG-p110.md` (#994, `c141276`). Issue: #770. Follows P109 (#992).

Code under test:
- e4b 0.42.0 at `c141276`;
- grouped-nf4-gemm 0.35.0 at `51a4916`;
- transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, NF4 arena baked on the box.

**Verdict by `p110_reduce.py`: `AT_PARITY`.**
- **The bias bar:** P's bias, +0.00036 nats, is ≤ B_floor + 0.01 = 0.0118.
- **The spread bar:** P's spread, 0.0114, is ≤ 2 × S_floor = 0.0254.
- **No VOID condition fired:**
  - the commits and the revision are the registered ones;
  - every arm has its 48 windows (rep: 12);
  - engagement is exact: 6,096 decode attention calls per pass (127 steps × 48 layers; `half` 12,192);
  - device grouping is off for R, rep and the floor, and on for D, P and the mutant;
  - P and the mutant ran all 127 steps as eager padded bucket-16 steps: 508 pad rows, no replays;
  - `mutant_scale` fails the bar.

## The reading (`p110-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 580.95.05) on an AMD EPYC 7C13 (256 threads, 818 GiB RAM), Vast instance
54072289. **Cost:** $0.1904. Teardown was proven at 22:46:27Z.

**Timeline (the box's own log):**
- install at 22:28:05Z;
- premise 7 passed, none skipped;
- fetch 22:28:59–22:33:53Z;
- bake until 22:35:23Z;
- the box 22:35:23–22:45:46Z: load 34 s, 576 s in all, peak 22.05 GiB;
- reduce at 22:45:46Z.

| arm | what it is | bias vs R (nats) | spread | se | max \|d\| | mean KL vs R | argmax agree |
|---|---|---:|---:|---:|---:|---:|---:|
| rep | R again | +0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 1.0000 |
| half (floor) | two half-batches of 6 | +0.00180 | 0.01119 | 0.00200 | 0.03590 | 9.6e-3 | 0.9634 |
| chunk (floor) | prompts prefilled in 256-token pieces | +0.00046 | 0.01270 | 0.00240 | 0.04437 | 1.0e-2 | 0.9606 |
| rev (floor) | reversed slots and row order | +0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 1.0000 |
| D | device grouping, unpadded | +0.00153 | 0.01218 | 0.00235 | 0.05309 | 9.0e-3 | 0.9663 |
| **P** | **device grouping + bucket padding (the graph arithmetic)** | **+0.00036** | **0.01136** | 0.00213 | 0.04251 | 8.8e-3 | 0.9645 |
| mutant_scale | P with the decode softmax scale halved | +1.05398 | 1.05398 | 0.03722 | 1.66080 | 1.19 | 0.6004 |

R's mean NLL is 2.1936 over the 48 windows, ranging from 1.239 to 3.390.

- **The graph arithmetic sits inside the floor.**
  - P's bias, +0.0004 nats, is smaller than the half-batch draw's (+0.0018) and close to the prefill-split draw's
    (+0.0005).
  - Its spread (0.0114) and KL against R (8.8e-3) lie between theirs.
  - Its standard error is 0.0021, so the bias is indistinguishable from zero.
- **Grouping alone (D) passes too** (+0.0015).
- **The floor is wider than predicted.** Changing the batch's row count or the prefill split moves single windows by up to
  0.04 nats, a mean of about 0.012 per window. Two draws were bit-identical to R: row order (`rev`) and the repeat. Neither
  changes the arithmetic.
- **The instrument can fail:** a halved decode scale reads +1.05 nats on every window.
- **Step times, for the record** (eager on this host, not the graph speed, which is P109's): R 111–139 ms per 12-row step,
  D 97–113, P 86–98.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: R repeats bit for bit | **yes** |
| Q2: the floor's \|bias\| ≤ 0.003 and its spread ≤ 0.005 nats | bias **yes** (0.0018). Spread **no: 0.0127**. Qwen3's per-window sensitivity to batch shape and prefill split is about 2.5× what the prediction allowed |
| Q3: P's bias in [−0.003, +0.003], its spread ≤ 0.006 | bias **yes** (+0.00036). Spread **no** (0.0114), for the same reason; it lies inside the floor's |
| Q4: D reads as P does, within 0.002 in bias | **yes** (0.0012 apart) |
| Q5: mutant_scale's bias above +0.3 | **yes** (+1.054) |
| Q6: engagement exact | **yes** |
| Q7: the box ≤ 30 minutes | **yes** (9.6 min) |
| Q8: AT_PARITY | **yes** |

## The registered consequence (AT_PARITY)

AT_PARITY licenses the default P109 registered. The license rests on three readings, all on this model and server:
- P109's speed: ×5.60 with 16 concurrent requests and ×9.02 with one;
- P109's function: the replay is bit-identical to its padded eager step;
- this lane's quality.

What follows:
- **The default.** `serve_paged`'s `E4B_PAGED_GRAPHS` defaults to `auto`: graphs on for a CUDA device at the `all-vram`
  placement, eager elsewhere. `0` turns them off; `1` forces them. This lands in the follow-up PR, which carries the code
  change and its tests.
- **The release notes** will say that greedy outputs change at the bf16 level from the old eager default's, at no
  measured quality cost.
- **The register.** The row `e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.2026-10-03` carries P's bias, from the
  verdict file.
- **The issue.** #770 closes with the default change.
- **Scope.** This was read on Qwen3-30B-A3B NF4. Other families ride the same capture code without a reading of their
  own; hybrid models' graphs are P101's.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p110-prove-1` | OK, PROVED | $0.0551 | the whole box on Granite in 43 s; its verdict (not a reading) AT_PARITY, P +0.0062 against a bar of 0.0180 |
| `p110-5090-1` | **OK, AT_PARITY** | $0.1904 | the reading |

**The lane cost $0.2455**, inside its $3.00 ceiling. Together with P109 ($0.4599), the decode-graph default cost
$0.7054 to license.

**Receipts** are in `receipts/p110-5090-1/`, with `SHA256SUMS`:
- `box.json` (every arm's per-window NLL, KL and agreement, and the engagement of every pass);
- `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the install, premise, fetch, bake and box logs;
- the teardown proof.

The launcher's receipts and ledger rows are in the receipt store, adertha-receipts `4dce812` (`p110-prove-1`) and
`776eed5` (`p110-5090-1`).
