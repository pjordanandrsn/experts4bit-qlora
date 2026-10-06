# SV6 — the planner's revised 24 GB plan for Qwen3-30B-A3B (solver tiers, 8 × 8192) on an RTX 4090, at the longest prompts

Work item: experts4bit-qlora#1267 (the owner-authorization permalink; $5 cap within the owner's $50, of which SV2–SV5
spent $1.245). Registered before any SV6 box. The box launches only from this registration's merge SHA on main, through
`bench/tc1/tc1_drive.sh` (`TC1_RUNNER=sv6_run.sh`). The adertha controller must carry adertha-agents#182, whose
launcher check refuses a PREREG that is not on main.

## Premise

- SV5's read (#1257, exploratory, licenses no change) ran the planner's earlier all-VRAM plan for this shape out of
  memory at 8,000-token prompts.
- Two changes followed, neither licensed by that read:
  - #1247 prices the bulk KV flush, reviewed on its code;
  - the planner (loggetta, its own decision) caps a card class at the capacity a CUDA process actually got, 23.52 GiB
    on this class.
- **The plan under test** is loggetta `dd4783f`'s, planned against the 47 receipts in its `evidence/` (SV1–SV5
  included) for a stated RTX 4090 with 125 GiB of host memory. It refuses all-VRAM: 23.35 GiB + 1.18 GiB headroom
  exceeds the 23.52 GiB budget. It plans the solver's tiers instead:
  `ServeSetup(placement="solver", max_seqs=8, max_tokens_per_seq=8192, chunk_tokens=512, graphs=False, buckets=(1, 2, 4, 8), prefill_graph="0", vram_gb=12.631, dram_gb=15.187, hot_rows=1)`,
  with the bulk flush on (the server's default). The planner chooses this; this file does not license it.
- Nothing has run this plan, and no box has measured the bulk-flush item it carries.

**Question.** Does the planner's tier plan run 8 × 8,000-token prompts on the card it was made for, inside its 22.343
GiB? Are the estimate's two prompt-length items, the bulk flush and prefill staging, priced to what the server
allocates?

## Shape (fixed)

| | |
|---|---|
| model | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| arena | baked on the box by grouped-nf4-gemm's `bake_nf4` under `/root/sv6-arena`, outside the fetched tree |
| server setup | the plan's `ServeSetup` above → `to_env()`; `E4B_PAGED_TORCH_THREADS=8` |
| arms, in order | `b6_short`: 8 requests of 1,024 seeded prompt tokens (the anchor). `b6_long`: 8 requests of 8,000 seeded prompt tokens. Both 32 new tokens, served to idle, in one process each |
| measure | `bench/sv4/sv4_measure.py` (`sv4-arm/1`, on main): the estimate's items, allocator / reserved / driver peaks, host memory, the server's tier rows |
| software | experts4bit-qlora at the launch SHA (this registration's merge); grouped-nf4-gemm 0.41.0 (`dc8f94abfd868f149178623f6eb403dc8b892b02`); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2 |

## The estimate's numbers, computed before the box (experts4bit-qlora `50e24f3c`)

Device total **21,985,437,184 bytes (20.476 GiB)**:
- expert stacks, VRAM tier: 12,932.2 MiB (5,109 of 6,144 rows);
- DRAM experts run on the GPU at prefill: 386.0 MiB;
- dense weights: 2,939.4 MiB;
- FP8 paged KV pool: 3,312.8 MiB;
- **bulk KV flush (#1247): 526.0 MiB**, `fp8_paged_kv.append_prompt_peak_bytes` at the slot's 8,192 tokens;
- prefill staging at its ceiling of 8,704 tokens: 816.0 MiB;
- working set: 54.6 MiB.

Host items: the DRAM tier 2,619.8 MiB (1,035 rows), the cold-tier landing 4.0 MiB and the setup tier 20.2 MiB.
NVMe: 0 rows.

**The plan's total** is **23,990,209,701 bytes (22.343 GiB)**: the estimate, plus a 1.367 GiB allocator reserve
(measured: `sv4-4090-1/t4_deep4`'s slack on this card) and a 0.5 GiB CUDA context (default). Its budget is the
23.52 GiB the card class gave SV5's process. nvidia-smi lists 24,564 MiB.

**The prompt-length items at each arm's prompts**, from the same functions:

| prompts | bulk flush (`append_prompt_peak_bytes`) | staging (prompt + one 512-token chunk) |
|---|---|---|
| 1,024 | 163.8 MiB | 144.0 MiB (1,536 tokens) |
| 8,000 | 513.7 MiB | 798.0 MiB (8,512 tokens) |

Long minus short: **1,052,688,384 bytes (1,003.9 MiB)**. Every other estimate item is the same in both arms: the KV
pool is preallocated, and the DRAM-on-GPU transient and the working set are sized by the chunk. So the two arms'
allocator peaks should differ by this, if both peaks fall at a prompt's completion. SV5's all-VRAM pair (1,024
against an OOM at 8,000) differed by at least 908 MiB. That is consistent with this figure and does not test it.

The runner's tripwire recomputes this device total at the installed SHA before any fetch. If the total has moved, the
box exits 9.

## Readings

- **Y1 (the plan fits).** `b6_long` finishes all 8 requests with no out-of-memory error, and its driver peak is at or
  under the card's total. Expected: yes.
- **Y2 (the plan bounds the process).** `b6_long`'s driver peak against the plan's 23,990,209,701 bytes. Expected: at
  or under it.
- **Y3 (the estimate, long prompts).** `b6_long`'s allocator peak against 20.476 GiB, ±5%. Expected: within, near the
  estimate (`b6_short`'s peak plus Y6's ~1.0 GiB).
- **Y4 (the estimate, short prompts).** `b6_short`'s allocator peak against 20.476 GiB, ±5%. Expected: within, under it
  by roughly the two prompt-length items' unused ceilings (~1.0 GiB, about −5%).
  - SV4's tier arm at these short prompts read −4.3%.
  - This is the reading most likely to fall below the band.
- **Y5 (the split).** The server's tier rows in both arms equal the estimate's: 5,109 VRAM / 1,035 DRAM / 0 NVMe.
- **Y6 (the prompt-length items).** `b6_long`'s allocator peak minus `b6_short`'s against 1,003.9 MiB, ±15%.
- **Integrity**, both arms:
  - every request reaches `done`;
  - the server records its tier rows.

  An arm failing integrity is ALARM. An out-of-memory error in `b6_long` is Y1's reading, not an ALARM.
- **Recorded, not registered:**
  - tok/s (one draw each);
  - host memory (anonymous and pinned);
  - reserved minus allocated;
  - the driver peak minus the allocator peak (the context the plan defaults to 0.5 GiB).

## Consequences (registered before the data)

What each reading changes in the planner's plan, its slack, or this package's estimate items:

- **Y1.**
  - HELD: the planner keeps offering this tier plan for this model and shape on this card class.
  - MISSED: it stops offering it on this class. The OOM enters the planner's learned overheads only as a lower bound
    (allocated + requested), never as a measured peak. An estimate gap goes to Y3 / Y6.
- **Y2.**
  - HELD: the measured 1.367 GiB reserve and the 0.5 GiB context default stand for this shape. This run's own
    receipts then enter the planner's observations as same-setup evidence, as every receipt does. That is loggetta's
    design, and this file licenses using them for the reserve and the context on this class.
  - MISSED on a completed arm: if the excess is an item inside the estimate, it is Y3's or Y6's consequence, and the
    planner's slack is not widened to cover an estimate gap. If it is outside the estimate (context, workspaces,
    fragmentation), the planner's reserve for this class rises by the measured excess.
  - MISSED from an OOM's lower bound: no reserve changes; Y1 applies.
- **Y3 / Y4.**
  - HELD: the estimate's device total stands for this tier setup at that prompt length.
  - MISSED above: the unpriced item is priced in its own PR, derived from the code's allocation, reviewed on its code,
    and citing SV6. It is not fitted to this number.
  - MISSED below: no change. The staging, flush and DRAM-on-GPU items are ceilings, so a margin under them is a bound
    holding.
  - NO_READING (an OOM floor inside the band): nothing changes.
- **Y5.**
  - HELD: the estimate's tier pricing stands for this setup.
  - MISSED: the estimate does not call `solve_placement` as `build_engine` does for this setup. A PR on code fixes the
    call and cites SV6.
- **Y6.**
  - HELD: the bulk flush (#1247) and prefill staging stand as the estimate's prompt-length terms. This is the first
    measurement of the flush item.
  - MISSED above: a prompt-length-dependent allocation is unpriced. A PR on code names it and cites SV6.
  - MISSED below: the two transients do not peak together, or one is smaller than priced. No ceiling changes on this
    alone. A PR may price them as non-coexisting only from the code's own argument, citing SV6.
  - NO_READING: nothing changes.
- **Integrity ALARM:** that arm's numbers are not read, and nothing changes from them.

## The rule (`sv6_reduce.py`, self-test 30 cases)

`python bench/sv6/sv6_reduce.py --dir RUN_DIR --out verdict.json` scores the readings from `receipts/b6_short.json`,
`receipts/b6_long.json` and `forensics.txt` (nvidia-smi's total on its first line):
- **NO_READING:** no `b6_short` receipt, or a receipt that is not `sv4-arm/1`.
- **VOID:** an arm's model, revision, server setup (placement, tier budgets, `hot_rows`, sequences, graphs, buckets),
  prompt length or new-token count is not as registered, its estimate total is not 21,985,437,184 bytes, or the card
  is not an RTX 4090.
- **READ otherwise.** Each reading is HELD, MISSED or NO_READING, as above:
  - Bands are inclusive.
  - On `b6_long`'s OOM, Y1 reads MISSED.
  - Y2 compares the error's "this process has X in use" plus "Tried to allocate Y" with the plan. That is a lower
    bound: MISSED if it is over the plan, NO_READING otherwise.
  - Y3 compares the larger of the allocator peak and "X is allocated by PyTorch" plus Y with the band. It is also a
    lower bound: MISSED (above) only over +5%.
  - Y6 compares that floor minus `b6_short`'s peak with the band: MISSED (above) only over +15%.
  - Y5 reads only clean arms that record tier rows.
  - An ALARM arm's readings are NO_READING.

`python bench/sv6/sv6_reduce.py --self-test` runs 30 constructed cases (also run by `tests/test_sv6.py`). Nine mutants
each fail at least one:
- either band widened;
- Y1 ignoring the card;
- Y2 ignoring the plan, on a completed arm or on OOM;
- treating OOM as an ALARM;
- Y5 ignoring a differing split;
- Y5 reading an ALARM arm's split;
- Y3/Y6 never reading an OOM floor as MISSED.

The read PR carries a Reproduce line that runs this reducer from main on the committed receipts.

## Outcomes

- **Success:** two arm receipts with integrity clean; Y1–Y6 read. A miss is a result.
- **Exit codes.** 13 and 18 are host evidence; every other code is the workload's.
  - 13: under 120 GB of disk (host);
  - 18: under 32 GB of available or cgroup host RAM (host);
  - 9: install, tripwire, or the registered estimate total moved;
  - 10: fetch;
  - 11: fewer than two receipts (an OOM in `b6_long` still writes its receipt, status OOM);
  - 12: the bake;
  - 16: the `b6_short` anchor not finishing.
- **Receipt:** committed with its ledger row only, pushed over SSH, and `@{u}` re-synced before the read.

## Cost

Estimate: about 1 h on one RTX 4090 at the declared $0.60/h: a ~57 GB download, one ~16 GB bake and two builds. SV5
took 17 min wall on the same shape. Ceiling: wallclock 2.5 h, spend $5 (#1267), within the owner's $50 approval.
