# SV5 read — `sv5-4090-1` (exploratory; licenses no change)

Work item #1242 (owner-authorized, $5 cap within an owner-approved $50). Rule and consequences:
`bench/sv5/SV5-PREREG.md` (#1243, merged after review at `93d8993d`).

**Exploratory.** The box ran before #1243 was reviewed or merged. The rule and consequences were written after the
data, with the results known. Per #1243's "What this file licenses", these readings are recorded through the merged
reducer and **license no change** in this package or in the planner. The dated timeline is in `SV5-PREREG.md`.

**The run.**
- One RTX 4090 (sm_89): Vast instance 54433406 on machine 149142 (the same machine as `sv4-4090-1`). Driver 595.84,
  Intel i9-10980XE, 125 GB RAM, 320 GB disk.
- **$0.128**; teardown proven (instance gone).
- experts4bit-qlora `f49006f5` (#1243's first commit, 0.48.0; not on main), grouped-nf4-gemm 0.41.0 (`dc8f94a`),
  torch 2.8.0+cu128, transformers 5.18.0.
- The arena was baked on the box (16.3 GB, 199 s). The lane finished `TC1_SUCCESS` with two receipts.

Receipts: `receipts/sv5-4090-1/receipts/*.json` (`sv4-arm/1`), `summary.txt`, `forensics.txt`, `verdict.json`. They
are byte-identical (sha256) to the receipt store's copies.

| arm | prompts | status | estimate | allocator peak | driver peak | plan |
|---|---|---|---|---|---|---|
| `a5_short` | 8 × 1,024 | OK, 8 / 8 done | 22.150 | 21.734 (−1.9%) | 22.477 | 22.74 |
| `a5_long` | 8 × 8,000 | **OOM** | 22.150 | ≥ 22.621 (+2.1%, a floor) | ≥ 23.531 (a floor) | 22.74 |

GiB. Both floors come from the OOM message on a card that gave the process 23.52 GiB:
- `a5_long`'s allocator floor is "22.59 GiB is allocated by PyTorch" plus the 32 MiB requested.
- Its driver floor is "this process has 23.50 GiB memory in use" plus the 32 MiB.

**Readings** (`sv5_reduce.py`, verdict `READ`; exploratory):
- **Z1 (the plan fits): MISSED.** `a5_long` ran out of memory.
- **Z2 (the plan bounds the process): MISSED**, on a lower bound. 25,266,487,296 B against the plan's 24,416,889,078.
- **Z3 (the estimate, long prompts): NO_READING.** The floor is 2.1% over the estimate, inside the ±5% band, and the
  true peak is unknown.
- **Z4 (the estimate, short prompts): HELD**, −1.9%.
- **Integrity: clean.** `a5_short` finished all 8 requests. Every captured decode bucket in both arms reported `graph`.
  `a5_long`'s OOM is Z1's reading.

**Not attributed by this run.** The receipt records the OOM message, not where it was raised, so nothing here
places the failure in the code.
- The estimate listed one unmodelled item that grows with prompt length: the bulk KV flush, at 526 MiB for a full
  8,192-token slot.
- The failing arm needed at least 0.47 GiB over the estimate.
- That makes the flush the leading candidate, not a measured cause. #1247 prices it, reviewed on its code.

**Recorded, not registered.**
- The card is smaller than its listing. nvidia-smi reports 24,564 MiB, and the CUDA process was given 23.52 GiB.
- Short prompts ran under the estimate by 0.42 GiB. The staging ceiling is priced at 816 MiB for 8,704 tokens, and
  these prompts staged far fewer.
- Decode: 51.1 tok/s in `a5_short` (one draw, 8 sequences).
- Host memory: anonymous peaks of 1.84 / 2.19 GiB at load and 1.59 GiB serving; pinned (`RssShmem`) 1.45 GiB in `a5_short`.

**Reproduce** (from main, after this merges):

```bash
python bench/sv5/sv5_reduce.py --self-test && python bench/sv5/sv5_reduce.py --dir bench/sv5/receipts/sv5-4090-1
```
