# SV7 — the planner's re-matched 24 GB plan for Qwen3-30B-A3B (solver tiers, VRAM 13.143 GiB, 8 × 8192) on an RTX 4090, at the longest prompts

Work item: experts4bit-qlora#1294 (the owner-authorization permalink; $5 cap within the owner's $50, of which SV2–SV6
spent $1.617). Registered before any SV7 box. The box launches only from this registration's merge SHA on main, through
`bench/tc1/tc1_drive.sh` (`TC1_RUNNER=sv7_run.sh`), with the launcher's prereg check (adertha-agents#182).

## Premise

- loggetta `10cbf9f` changed how a serve plan borrows its allocator reserve (its RESULTS 6k; planning only, loggetta's
  own decision, not licensed by any read here):
  - after the candidate's whole setup, it now prefers a receipt of the same workload shape, one that differs only in
    the tier budgets and the `hot_rows` derived from them;
  - every old serve receipt now records the `bulk_kv` it ran with;
  - serving slack is no longer transferred across GPUs through an anchor ratio.
- **The plan under test** is `10cbf9f`'s, planned against the 49 receipts in its `evidence/` for a stated RTX 4090
  with 125 GiB of host memory:
  `ServeSetup(placement="solver", max_seqs=8, max_tokens_per_seq=8192, chunk_tokens=512, graphs=False, buckets=(1, 2, 4, 8), prefill_graph="0", vram_gb=13.143, dram_gb=15.187, hot_rows=1)`,
  with the bulk flush on (the server's default).
- **Its reserve:** 0.856 GiB, which is SV4 `t4_plan8`'s 4.08% slack applied to the estimate. That receipt has the same
  shape, run at VRAM 10.933 GiB with 1,024-token prompts. SV6's receipts are scoped by their licence to their own
  setup (VRAM 12.631), so they do not apply here.
- Before the change, the plan borrowed SV4's 4 × 4096 NVMe arm (6.7%) and stopped at VRAM 12.631 GiB, which SV6 ran.
  The new plan puts **5,316** expert rows on the GPU instead of 5,109. Nothing has run it.

**Question.** Does the re-matched plan serve 8 × 8,000-token prompts inside its 22.344 GiB? Does a same-shape reserve
borrowed from another tier budget cover the allocator's cached blocks at this one?

## Shape (fixed)

| | |
|---|---|
| model | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| arena | baked on the box by grouped-nf4-gemm's `bake_nf4` under `/root/sv7-arena`, outside the fetched tree |
| server setup | the plan's `ServeSetup` above → `to_env()`; `E4B_PAGED_TORCH_THREADS=8` |
| arms, in order | `b7_short`: 8 requests of 1,024 seeded prompt tokens (the anchor). `b7_long`: 8 requests of 8,000 seeded prompt tokens. Both 32 new tokens, served to idle, in one process each |
| measure | `bench/sv4/sv4_measure.py` (`sv4-arm/1`, on main): the estimate's items, allocator / reserved / driver peaks, host memory, the server's tier rows |
| software | experts4bit-qlora at the launch SHA (this registration's merge); grouped-nf4-gemm 0.41.0 (`dc8f94abfd868f149178623f6eb403dc8b892b02`); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2 |

## The estimate's numbers, computed before the box (experts4bit-qlora `c07ea7f6`)

Device total **22,534,858,240 bytes (20.987 GiB)**:
- expert stacks, VRAM tier: 13,456.1 MiB (5,316 of 6,144 rows);
- DRAM experts run on the GPU at prefill: 386.0 MiB;
- dense weights: 2,939.4 MiB;
- FP8 paged KV pool: 3,312.8 MiB;
- bulk KV flush: 526.0 MiB;
- prefill staging at its ceiling of 8,704 tokens: 816.0 MiB;
- working set: 54.6 MiB.

Host items: the DRAM tier 2,095.9 MiB (828 rows), the cold-tier landing 4.0 MiB and the setup tier 20.2 MiB.
NVMe: 0 rows.

**The plan's total** is **23,991,176,435 bytes (22.344 GiB)**: the estimate, plus the **919,447,283-byte** reserve
above and a 0.5 GiB CUDA context (default). Its budget is the 23.52 GiB the card class gave SV5's process.

**What SV6 predicts.** SV6 ran the same shape at VRAM 12.631 GiB. Its long arm measured:
- allocator 20.146 GiB (−1.6% against its estimate);
- reserved minus allocated 775 MiB;
- driver peak 21.357 GiB.

The extra 207 VRAM rows add 524 MiB to the allocator side. So the expectations are:
- long-arm allocator about 20.66 GiB (−1.6%);
- reserved minus allocated about 0.78 GiB, under the borrowed 0.856;
- driver peak about 21.9 GiB, under the 22.344 plan.

The runner's tripwire recomputes the device total at the installed SHA before any fetch. If the total has moved, the
box exits 9.

## Readings

- **V1 (the plan fits).** `b7_long` finishes all 8 requests with no out-of-memory error, and its driver peak is at or
  under the card's total. Expected: yes.
- **V2 (the plan bounds the process).** `b7_long`'s driver peak against the plan's 23,991,176,435 bytes. Expected: at
  or under it.
- **V3 (the borrowed reserve).** Over both arms (the clean ones that finished), the largest reserved peak minus
  allocator peak against the plan's 919,447,283-byte reserve. Expected: at or under it.
- **V4 (the estimate, long prompts).** `b7_long`'s allocator peak against 20.987 GiB, ±5%. Expected: within.
- **V5 (the split).** The server's tier rows in both arms equal the estimate's: 5,316 VRAM / 828 DRAM / 0 NVMe.
- **Integrity**, both arms:
  - every request reaches `done`.

  An arm failing integrity is ALARM. An out-of-memory error in `b7_long` is V1's reading, not an ALARM.
- **Recorded, not registered:**
  - `b7_short`'s allocator peak against the estimate (SV6's short arm read −6.4%: unused flush and staging ceilings);
  - the long-minus-short allocator delta (SV6: 1,001.0 MiB);
  - tok/s (one draw each), host memory;
  - the driver peak minus the reserved peak, against the 0.5 GiB context default.

## Consequences (registered before the data)

What each reading changes in the planner's plan, its slack, or this package's estimate items:

- **V1.**
  - HELD: the planner keeps offering this tier plan for Qwen3-30B-A3B at 8 × 8192 on this card class.
  - MISSED: it stops offering it on this class. The OOM enters the planner's learned overheads only as a lower bound
    (allocated + requested). An estimate gap goes to V4; a reserve gap to V3.
- **V2.**
  - HELD: the plan stands. This run's receipts enter the planner's observations as same-setup evidence for the
    allocator reserve and the CUDA context on this class, and nothing more. They are imported with
    `licensed_for: [reserve, context]`, as SV6's were.
  - MISSED on a completed arm: if the excess is inside the estimate, it is V4's consequence; if it is in the reserve,
    V3's. If it is the context, the planner's context for this class rises by the measured excess.
  - MISSED from an OOM's lower bound: no reserve or context changes; V1 applies.
- **V3.**
  - HELD: loggetta's same-shape reserve rule stands for this shape on this class.
  - MISSED: the rule did not cover a second tier budget. loggetta stops borrowing a serve slack across tier budgets:
    the backend names no budget fields, and plans fall back to the whole setup, then the same key fields. That is
    loggetta's change to make, and this reading is what licenses it, citing SV7.
- **V4.**
  - HELD: the estimate's device total stands for this tier setup at long prompts.
  - MISSED above: the unpriced item is priced in its own PR, derived from the code's allocation, reviewed on its code,
    citing SV7, and not fitted to this number.
  - MISSED below: no change. The staging, flush and DRAM-on-GPU items are ceilings, so a margin under them is a bound
    holding.
  - NO_READING (an OOM floor inside the band): nothing changes.
- **V5.**
  - HELD: the estimate's tier pricing stands for this setup.
  - MISSED: the estimate does not call `solve_placement` as `build_engine` does here. A PR on code fixes the call and
    cites SV7.
- **Integrity ALARM:** that arm's numbers are not read, and nothing changes from them.

## The rule (`sv7_reduce.py`, self-test 32 cases)

`python bench/sv7/sv7_reduce.py --dir RUN_DIR --out verdict.json` scores the readings from `receipts/b7_short.json`,
`receipts/b7_long.json` and `forensics.txt` (nvidia-smi's total on its first line):
- **NO_READING:** no `b7_short` receipt, or a receipt that is not `sv4-arm/1`.
- **VOID:** an arm's model, revision, server setup (placement, tier budgets, `hot_rows`, sequences, graphs, buckets),
  prompt length or new-token count is not as registered, its estimate total is not 22,534,858,240 bytes, or the card
  is not an RTX 4090.
- **READ otherwise.** Each reading is HELD, MISSED or NO_READING, as above:
  - Bands are inclusive.
  - On `b7_long`'s OOM, V1 reads MISSED.
  - V2 compares the error's "this process has X in use" plus "Tried to allocate Y" with the plan. That is a lower
    bound: MISSED if it is over the plan, NO_READING otherwise.
  - V3 reads only clean arms that finished; an OOM arm's reserved peak is not read.
  - V4 compares the larger of the allocator peak and "X is allocated by PyTorch" plus Y with the band. It is a lower
    bound: MISSED (above) only over +5%.
  - V5 reads only clean arms that record tier rows.
  - An ALARM arm's readings are NO_READING.

`python bench/sv7/sv7_reduce.py --self-test` runs 32 constructed cases (also run by `tests/test_sv7.py`). Eleven
mutants each fail at least one:
- the band widened;
- V1 ignoring the card;
- V2 ignoring the plan, on a completed arm or on OOM;
- treating OOM as an ALARM;
- V3 ignoring the reserve, reading only the long arm, or reading an ALARM arm;
- V5 ignoring a differing split or reading an ALARM arm's split;
- V4 never reading an OOM floor as MISSED.

The read PR carries a Reproduce line that runs this reducer from main on the committed receipts.

## Outcomes

- **Success:** two arm receipts with integrity clean; V1–V5 read. A miss is a result.
- **Exit codes**, as SV6's after its amendment 1. 13 and 18 are host evidence; every other code is the workload's.
  - 13: under 120 GB of disk (host);
  - 18: under 32 GB of available or cgroup host RAM, or an NVIDIA driver older than 570 (host);
  - 9: install, tripwire, the registered estimate total moved, or an unreadable driver;
  - 10: fetch;
  - 11: fewer than two receipts (an OOM in `b7_long` still writes its receipt, status OOM);
  - 12: the bake;
  - 16: the `b7_short` anchor not finishing.
- **Receipt:** committed with its ledger row only, pushed over SSH, and `@{u}` re-synced before the read.

## Cost

Estimate: about 1 h on one RTX 4090 at the declared $0.60/h: a ~57 GB download, one ~16 GB bake and two builds. SV6's
run took 19 min wall and $0.32. Ceiling: wallclock 2.5 h, spend $5 (#1294), within the owner's $50 approval.
