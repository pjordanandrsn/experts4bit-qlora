# SV6 read — `sv6-4090-3`

Registered in `bench/sv6/SV6-PREREG.md`. #1268 was merged after review, and amendment 1 (#1272, the driver floor) was
merged after review, both before any SV6 data. Work item #1267: owner-authorized, $5 cap within the owner's $50.

**How it ran.**
- The box launched from amendment 1's merge, `af0d95df`. The launcher's prereg check (adertha-agents#182) passed,
  recording `lane_commit_sha af0d95df` against origin/main `a41c857d`. The estimate was re-verified there at
  21,985,437,184 B before launch.
- Two earlier attempts reached no arm, so they yield no data. Their receipts are in the store:
  - `sv6-4090-1`: NOT_RUN at the pre-flight, $0.023.
  - `sv6-4090-2`: driver 535, tripwire exit 9, $0.029.
  - Amendment 1 records both.
- **`sv6-4090-3`:** one RTX 4090 (sm_89) on Vast instance 54502968, machine 146201. Driver 595.84, AMD EPYC 7B13, 504 GiB
  RAM, PCIe gen 4 x16, 320 GB disk.
- **$0.320**, 15:31:58–15:50:38Z, teardown proven. Lane total: **$0.372** of $5.
- experts4bit-qlora `af0d95df` (0.48.0), grouped-nf4-gemm 0.41.0 (`dc8f94a`), torch 2.8.0+cu128, transformers 5.18.0.
- The arena was baked on the box (16.3 GB, 92 s). The lane finished `TC1_SUCCESS` with two receipts.

Receipts: `receipts/sv6-4090-3/receipts/*.json` (`sv4-arm/1`), `summary.txt`, `forensics.txt`, `verdict.json`. They are
byte-identical (sha256) to the receipt store's copies.

| arm | prompts | status | estimate | allocator peak | vs estimate | driver peak | plan | tier rows (VRAM / DRAM / NVMe) |
|---|---|---|---|---|---|---|---|---|
| `b6_short` | 8 × 1,024 | OK, 8 / 8 | 20.476 | 19.168 | −6.4% | 20.154 | 22.343 | 5,109 / 1,035 / 0 |
| `b6_long` | 8 × 8,000 | OK, 8 / 8 | 20.476 | 20.146 | −1.6% | 21.357 | 22.343 | 5,109 / 1,035 / 0 |

GiB. The card lists 24,564 MiB.

**Readings** (`sv6_reduce.py`, verdict `READ`, integrity clean in both arms):
- **Y1 (the plan fits): HELD.** The planner's tier plan served 8 × 8,000-token prompts with no out-of-memory error. The
  driver peak was 21.357 GiB, under the card's total.
- **Y2 (the plan bounds the process): HELD.** 22,932,357,120 B against the plan's 23,990,209,701: 0.986 GiB under.
- **Y3 (the estimate, long prompts): HELD**, −1.6%.
- **Y4 (the estimate, short prompts): MISSED, below**, −6.4%. The registration named it the reading most likely to fall
  below the band. At 1,024-token prompts:
  - the flush ceiling (526 MiB) is about 164 MiB used;
  - the staging ceiling (816 MiB) is about 144 MiB used;
  - together that leaves 1,034 MiB, 4.9% of the estimate, unused.
- **Y5 (the split): HELD, exactly.** Both arms' server tier rows equal the estimate's.
- **Y6 (the prompt-length items): HELD.** `b6_long` minus `b6_short` was 1,049,600,000 B (1,001.0 MiB). The bulk flush
  and staging formulas give 1,052,688,384 B (1,003.9 MiB): **−0.3%**. This is the first measurement of #1247's bulk-flush
  item. Read with staging, it prices what the server allocates when a prompt completes, to 3 MiB.

**What the registered consequences change.**
- **Y1 HELD:** the planner keeps offering this tier plan for Qwen3-30B-A3B at 8 × 8192 on this card class.
- **Y2 HELD:** the 1.367 GiB measured reserve and the 0.5 GiB context default stand for this shape. The registration
  licenses this run's receipts as same-setup evidence for the planner's reserve and context on this class. The planner
  takes them in after this read merges.
- **Y3, Y5, Y6 HELD:** the estimate's total at long prompts, its tier pricing, and the bulk flush and staging items
  stand for this setup.
- **Y4 MISSED below:** no change. The flush and staging terms are ceilings, so a margin under them at short prompts is
  a bound holding.

**Recorded, not registered.**
- **Reserved minus allocated:** 544 / 775 MiB, against the plan's 1,400 MiB reserve, which came from `t4_deep4`'s
  slack on this card.
- **Driver minus reserved:** 466 MiB in both arms, against the plan's 512 MiB context default.
- Together the plan carried 0.88 / 0.66 GiB more than this run used.
- **Decode:** 6.0 / 3.2 tok/s (one draw each, eager, 1,035 expert rows computed on the CPU). Serve wall time was 43 / 81 s.
- **Host memory:** anonymous load peak 4.13 / 4.87 GiB; serving peak 4.22 / 6.64 GiB; pinned (`RssShmem`) 1.06 GiB.

**Reproduce** (from main, after this merges):

```bash
python bench/sv6/sv6_reduce.py --self-test && python bench/sv6/sv6_reduce.py --dir bench/sv6/receipts/sv6-4090-3
```
