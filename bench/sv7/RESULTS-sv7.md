# SV7 read — `sv7-4090-1` and `sv7-4090-2`

Registered in `bench/sv7/SV7-PREREG.md`. #1297 merged after review before any SV7 box. Amendment 1 (#1308, the driver
sampler's sole-process match) merged after review between the two boxes. Work item #1294: owner-authorized, $5 cap
within the owner's $50.

**The runs.** Both launched through the launcher's prereg check (on-main) and finished `TC1_SUCCESS` with two receipts.
Both read through the merged `sv7_reduce.py`, and both had integrity clean in both arms.

| run | from | Vast instance / machine | driver | spend |
|---|---|---|---|---|
| `sv7-4090-1` | `7431a8ed` (#1297) | 54619847 / 51613 | 580.95.05 | $0.611 |
| `sv7-4090-2` | `35c8b7ba` (#1308) | 54642987 / 145317 | 595.84 | $0.408 |

- Lane total: **$1.019** of $5. Teardown was proven for both.
- Software: experts4bit-qlora 0.48.0 at the launch SHAs, grouped-nf4-gemm 0.41.0 (`dc8f94a`), torch 2.8.0+cu128,
  transformers 5.18.0.
- The estimate was 22,534,858,240 B at both SHAs, re-verified before each launch.

Receipts are in `receipts/<run>/receipts/*.json` (`sv4-arm/1`), with `summary.txt`, `forensics.txt` and `verdict.json`
for each run. All eight receipt files are byte-identical (sha256) to the receipt store's copies.

| run · arm | prompts | allocator peak | vs estimate | reserved − allocated | driver peak | plan | tier rows |
|---|---|---|---|---|---|---|---|
| `-1` · `b7_short` | 8 × 1,024 | 19.681 | −6.2% | 400 MiB | — (0 samples) | 22.344 | 5,316 / 828 / 0 |
| `-1` · `b7_long` | 8 × 8,000 | 20.657 | −1.6% | 825 MiB | — (0 samples) | 22.344 | 5,316 / 828 / 0 |
| `-2` · `b7_short` | 8 × 1,024 | 19.681 | −6.2% | 400 MiB | 20.527 | 22.344 | 5,316 / 828 / 0 |
| `-2` · `b7_long` | 8 × 8,000 | 20.657 | −1.6% | 823 MiB | 21.916 | 22.344 | 5,316 / 828 / 0 |

GiB unless marked. The estimate is 20.987 GiB, and the borrowed reserve 877 MiB (919,447,283 B). The allocator peaks
were byte-identical across the two boxes.

**Readings.**

| reading | `sv7-4090-1` | `sv7-4090-2` |
|---|---|---|
| V1 the plan fits | NO_READING (no driver samples) | **HELD**: no OOM; 21.916 GiB under the card |
| V2 the plan bounds the process | NO_READING | **HELD**: 23,532,142,592 B against the plan's 23,991,176,435 (0.43 GiB under) |
| V3 the borrowed reserve | **HELD**: 865,356,800 B ≤ 919,447,283 | **HELD**: 863,259,648 B ≤ 919,447,283 |
| V4 the estimate, long prompts | **HELD**, −1.6% | **HELD**, −1.6% |
| V5 the split | **HELD**, exactly | **HELD**, exactly |

- **`sv7-4090-1`'s V1 and V2** are amendment 1's subject. Its container listed host-namespace PIDs, so the sampler
  matched no row.
- **`sv7-4090-2`'s samples** all matched by PID (`driver_match`: 256 and 399 `pid`). Its V1 and V2 count as amendment
  1 requires.
- **V3 held at 94% of the borrowed reserve in both runs.** SV4's same-shape slack was measured at 1,024-token prompts
  and covers 8,000-token prompts at this budget, but with little room.

**What the registered consequences change.**
- **V1 HELD:** the planner keeps offering this tier plan (VRAM 13.143 GiB) for Qwen3-30B-A3B at 8 × 8192 on this card
  class.
- **V2 HELD (`sv7-4090-2`):** the plan stands.
  - `sv7-4090-2`'s receipts enter the planner's observations as same-setup evidence for the allocator reserve and the
    CUDA context on this class, and nothing more (`licensed_for: [reserve, context]`).
  - `sv7-4090-1`'s do not: its V2 is NO_READING.
- **V3 HELD:** loggetta's same-shape reserve rule (`10cbf9f`) stands for this shape on this card class.
- **V4, V5 HELD:** the estimate's long-prompt total and its tier pricing stand for this setup.

**Recorded, not registered.**
- **The prediction from SV6** (in the PREREG before the data): allocator about 20.66, reserved minus allocated about
  0.78 GiB, driver peak about 21.9. Measured: 20.657, 0.80 and 21.916.
- **Short prompts** read −6.2% against the estimate (SV6: −6.4%). Long minus short was 999 MiB (SV6: 1,001.0 MiB).
- **Driver peak minus reserved peak** was 466 MiB in both of `sv7-4090-2`'s arms (SV6: 466), against the plan's
  512 MiB context default.
- **Decode** (`sv7-4090-2`, one draw each, eager, 828 expert rows on the CPU): 6.3 / 3.2 tok/s. SV6, with 1,035 rows on
  the CPU: 6.0 / 3.2.
- **Host memory** (`sv7-4090-2`): anonymous load peak 5.97 / 4.24 GiB, serving peak 3.72 / 6.71 GiB, pinned 1.06 GiB.
- **Bake:** 1,029 s on `sv7-4090-1`'s host, 170 s on `sv7-4090-2`'s.

**Reproduce** (from main, after this merges):

```bash
python bench/sv7/sv7_reduce.py --self-test && for r in sv7-4090-1 sv7-4090-2; do python bench/sv7/sv7_reduce.py --dir bench/sv7/receipts/$r; done
```
