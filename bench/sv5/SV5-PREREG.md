# SV5 — the planner's all-VRAM 8 × 8192 plan for Qwen3-30B-A3B on a 24 GB RTX 4090, at the longest prompts

Work item: experts4bit-qlora#1242 (the owner-authorization permalink; $5 cap within an owner-approved $50). For one
RTX 4090 (Vast, verified-secure) through `bench/tc1/tc1_drive.sh` (`TC1_RUNNER=sv5_run.sh`).

## How this ran (dated note, 2026-10-06)

This file is **not a pre-registration for `sv5-4090-1`**. That box ran before this registration was reviewed or merged.

| time (UTC) | event |
|---|---|
| 04:53:05 | maintainer note on #1242: before any box, the registration merges after review, with registered consequences, a reducer with a self-test and host-only exit codes |
| 04:53:06 | #1243 (this registration, head `f49006f5`) opened with `ready-to-merge` and auto-merge on |
| **04:53:43** | **`sv5-4090-1` launched** from `f49006f5`, 37 s after #1243 opened, before review and not on main |
| 04:54:25 | maintainer review: changes requested before any box launches and before any merge |
| 04:55:44 | auto-merge re-enabled at the unchanged head; disabled 04:56:28 |
| 05:10:43 | box finished, $0.128; `a5_short` OK, `a5_long` OOM |

- The launch receipt's `commit_sha` (`db1abf96`) and `branch main` are the launcher's own checkout
  (adertha-agents `main`), not experts4bit-qlora's. The e4b code that ran was `f49006f5`, which is not on main.
- Its `preregistration` text says "registered before this launch". That wording was the session's, and it is wrong:
  the PR had been open for 37 s, unreviewed.
- The receipt was committed at 05:12:40 and left unpushed, which blocks every lane's launches. It was pushed over
  SSH at 05:24:53, after the maintainer flagged it.

**Written after the data.** Everything below "Consequences" and "The rule" was added after `sv5-4090-1` finished,
with its results known: `a5_short` 21.734 GiB against 22.150; `a5_long` out of memory at 22.59 GiB allocated on a
23.52 GiB card. The exit-code changes are in `sv5_run.sh` for any later run. `sv5-4090-1` ran the earlier mapping,
where a bake failure exited 13. It exited 0, so no code was misused.

## Premise

The plan under test is loggetta's (the planner's) output at `35597dd`, from receipts SV1–SV4. The planner takes slack
from receipts by design, so the plan is policy and SV4's read does not license it. SV4's read (#1240) is on main with
its dated maintainer note (#1244): it licenses no change to this package. This lane tests the planner's plan, not
SV4's read.

**Question.** With SV4's receipts on file, a planner moves Qwen3-30B-A3B at 8 × 8192 on a 24 GB card from tiers to
all-VRAM with decode graphs. It plans 22.74 GiB:
- `estimate_serve_footprint`'s 22.150 GiB;
- a 0.09 GiB reserve learned from SV4's one-sequence arm;
- a 0.5 GiB CUDA context.

Does that plan bound the process when the prompts are as long as the setup admits? The estimate lists the bulk KV
flush transient (one prompt's K/V staged for its single write) as unmodelled; at 8,000 tokens it could be ~0.75 GiB.

## Shape (fixed)

| | |
|---|---|
| model | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| arena | baked on the box by grouped-nf4-gemm's `bake_nf4` under `/root/sv5-arena`, outside the fetched tree |
| server setup | `ServeSetup(placement="all-vram", max_seqs=8, max_tokens_per_seq=8192, chunk_tokens=512, graphs=True, buckets=(1, 2, 4, 8), prefill_graph="0", hot_rows=64)` → `to_env()`; `E4B_PAGED_TORCH_THREADS=8` |
| arms, in order | `a5_short`: 8 requests of 1,024 seeded prompt tokens (the anchor). `a5_long`: 8 requests of 8,000 seeded prompt tokens. Both 32 new tokens, served to idle |
| measure | `bench/sv4/sv4_measure.py` (`sv4-arm/1`): the estimate's items, allocator / reserved / driver peaks, host memory, graph status |
| software | experts4bit-qlora at the launch head; grouped-nf4-gemm 0.41.0 (`dc8f94abfd868f149178623f6eb403dc8b892b02`); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2 |

## The estimate's numbers, computed before the box (experts4bit-qlora `fd70f75b`)

Device total **22.150 GiB** (23,783,815,168 bytes):
- frozen expert stacks 15,552 MiB;
- dense 2,939 MiB;
- FP8 paged KV pool 3,320 MiB;
- prefill staging at its ceiling (8,704 tokens) 816 MiB;
- working set 55 MiB.

The plan's total is 22.74 GiB; the card reports 23.99 GiB.

## Readings

- **Z1 (the plan fits).** `a5_long` finishes all 8 requests with no out-of-memory error, and its driver-reported peak is
  at or under the card's total. Expected: yes.
- **Z2 (the plan bounds the process).** `a5_long`'s driver peak against the plan's 22.74 GiB. Expected: at or under it.
  A miss names what the plan did not carry.
- **Z3 (the estimate, long prompts).** `a5_long`'s allocator peak against 22.150 GiB. Expected within ±5%. The bulk
  KV flush transient is unpriced and only adds.
- **Z4 (the estimate, short prompts).** `a5_short`'s allocator peak against 22.150 GiB. Expected within ±5%, below it by
  about the staging ceiling (816 MiB priced against ~144 MiB staged).
- **Integrity**, both arms:
  - every request reaches `done`;
  - every captured decode bucket reports `graph`.

  An arm failing integrity is ALARM. An out-of-memory error in `a5_long` is Z1's reading, not an ALARM.
- **Recorded, not registered:** tok/s (one draw each), host memory, slack.

## Consequences (written after the data; see the dated note)

What each reading changes in the planner's plan, its slack, or this package's estimate items:

- **Z1.**
  - HELD: the planner keeps offering all-VRAM for this model and shape on a 24 GB card. Nothing changes.
  - MISSED: the planner stops offering that plan on this card class. The OOM goes into its learned overheads as a
    lower bound (allocated + requested), never as a measured peak. The capacity the OOM message reports caps that
    class's device budget.
  - loggetta `6024451` already makes the MISSED change. It was pushed before this registration was reviewed, and it
    is reverted if this consequence is rejected here.
- **Z2.**
  - HELD: the 0.09 GiB learned reserve and the 0.5 GiB context allowance stand for this shape.
  - MISSED on a completed arm: if the excess is an item inside the estimate, it is Z3's consequence. The planner's
    slack is not widened to cover a gap in the estimate. If the excess is outside the estimate (context, workspaces,
    fragmentation), the planner's reserve for this card class rises by the measured excess.
  - MISSED from an OOM's lower bound: no reserve changes. A lower bound is not a measured overhead. Z1's consequence
    applies.
- **Z3 / Z4.**
  - HELD: the estimate's device total stands for this model and setup at that prompt length.
  - MISSED above: the unpriced item is priced in its own PR. That PR derives the price from the code's allocation,
    is reviewed on its code, and is not fitted to this number.
  - MISSED below: no change. The staging and working-set terms are ceilings, so a margin under them is a bound
    holding, not an error to fit.
  - NO_READING (an OOM floor inside the band): nothing changes.
- **Integrity ALARM:** that arm's numbers are not read, and nothing changes from them.

## The rule (`sv5_reduce.py`, self-test 25 cases)

`python bench/sv5/sv5_reduce.py --dir RUN_DIR --out verdict.json` scores the readings from `receipts/a5_short.json`,
`receipts/a5_long.json` and `forensics.txt` (nvidia-smi's total on its first line):
- **NO_READING:** no `a5_short` receipt, or a receipt that is not `sv4-arm/1`.
- **VOID:** an arm's model, revision, server setup, prompt length or new-token count is not as registered, its
  estimate total is not 23,783,815,168 bytes, or the card is not an RTX 4090.
- **READ otherwise.** Each reading is HELD, MISSED or NO_READING, as registered above:
  - The ±5% band is inclusive.
  - On `a5_long`'s OOM, Z1 reads MISSED.
  - Z2 compares the error's "this process has X in use" plus "Tried to allocate Y" with the plan. That is a lower
    bound: MISSED if it is over the plan, NO_READING otherwise.
  - Z3 compares the larger of the allocator peak and "X is allocated by PyTorch" plus Y with the band. That is also
    a lower bound: MISSED (above) only if it is over +5%, NO_READING otherwise.

`python bench/sv5/sv5_reduce.py --self-test` runs 25 constructed cases. Six mutants each fail at least one:
- a band of 50%;
- Z1 ignoring the card total;
- Z2 ignoring the plan on OOM;
- treating OOM as an ALARM;
- ignoring eager buckets;
- ignoring an arm with no bucket captured.

The read PR carries a Reproduce line that runs this reducer from main on the committed receipts.

## Outcomes

- **Success:** two arm receipts with integrity clean; Z1–Z4 read. A miss is a result.
- **Exit codes.** 13 and 18 are host evidence; every other code is the workload's.
  - 13: under 120 GB of disk (host);
  - 18: under 32 GB of available or cgroup host RAM (host);
  - 9: install or tripwire;
  - 10: fetch;
  - 11: fewer than two receipts (an OOM in `a5_long` still writes its receipt, status OOM);
  - 12: the bake;
  - 16: the `a5_short` anchor not finishing.

## Cost

Estimate: about 1 h on one RTX 4090 at the declared $0.60/h: a ~57 GB download, one ~16 GB bake and two builds.
Ceiling: wallclock 2.5 h, spend $5 (#1242), within the owner's $50 approval.
