# Pre-registration — architecture support, the part that needs a GPU

Registered 2026-09-09, before any rental. Owner authorisation for rented compute
is on record ("use pods as needed", 2026-09-09) and is the **top of the approval
band, not a waiver** of this document or of the STOP rules below.

## What is already settled, and therefore NOT in scope

Six of nine claimed families hold reference-tier **load + verify + forward** rows
on real checkpoints, produced on CPU (experts4bit-qlora#521, merged):
`olmoe`, `qwen3_moe`, `qwen3_5_moe`, `gpt_oss`, `gemma4`, `granitemoe`.

That is full evidence for what `SUPPORTED_ARCHITECTURES` actually claims — a
**loader** claim, that e4b accepts a checkpoint — and no evidence at all for
CUDA-graph capture or throughput. Nothing below re-litigates the loader claim.

`gemma4_text` is excluded permanently: no release carries that top-level
`model_type`, and it shares `GEMMA4`'s convention record with `gemma4`.

## Hypotheses, each with the observation that would refute it

**H1 — capture.** Each of the six reference-tier families can be CUDA-graph
decode-captured under the pipelined engine, 16 new tokens greedy.
*Refuted for a family by:* a throw inside the capture region, a host sync, or
captured output differing from eager beyond the router-flip noise floor.

**H2 — deepseek_v4 loads on adequate hardware.** DeepSeek-V4-Flash
(148.7 GB bf16, ~42 GB loaded by the estimate in `nas_inventory.py`) loads,
verifies strict, and forwards finite on a host with ≥128 GB usable RAM.
*Refuted by:* an OOM at ≥128 GB, or a loader refusal naming a convention gap.
*Prior:* it was OOM-killed on a 64 GB host at exit 137 after reaching RSS
16–18 GB (row `deepseek_v4.json`, which records that the run was contaminated
by a concurrent copy for ~1 min some 7 min before the kill).

**H3 — the estimate is calibrated.** `loaded_gb()` predicts resident footprint
within ±30% of peak **anonymous** memory (not RSS — RSS counts mmap'd checkpoint
pages and cannot validate it).
*Refuted by:* any family outside that band. This is the first measurement of a
number currently steering pod decisions, so it is registered as a hypothesis
rather than reported as a result.

## Explicitly deferred, with the reason

**`kimi_k3` full load — NOT attempted.** ~409 GB estimated resident. The
checkpoint (1.45 TB) and both 1.3 TB arenas are already on the LAN, but a pod
cannot reach them: uploading is the cost, not downloading. The arena path exists
precisely so K3 need not be resident, and its expert fetch is already at 78% of
device (gnf4#79) — so the useful K3 work is per-layer throughput on pod-local
NVMe, which is a different lane with a different prereg. Attempting a full load
here would be spending to confirm an arithmetic certainty.

## Pinned inputs

Checkpoints are pinned by commit at launch and recorded in the receipt; the
manifest carries the pins. Kernel and library versions come from the row's
`versions` block, never assumed. A run whose pins differ from its manifest is
VOID, not adjusted.

## Vocabulary — a partial pass is a row, not a rounding

| verdict | meaning |
|---|---|
| `pass` | every registered stage ok for that family |
| `partial` | some stages ok; the failing stage NAMED, never averaged away |
| `refused` | the loader/kernel declined, with its reason quoted |
| `host-limited` | the box could not finish; says nothing about the family |
| `void` | pins drifted, or the instrument was wrong — reported, never reused |

## STOP rules

1. **Spend.** Estimated ≤ $20 total for H1+H2. At **$35 actual** the lane stops
   and reports, whatever remains unmeasured
   ([[feedback_rented_teardown_discipline]]).
2. **Two consecutive VOIDs** on the same family stop that family. A third
   attempt after two instrument failures is not evidence-gathering.
3. **Any teardown that cannot be proven** stops the whole lane immediately.
4. **H3 refuted** stops H2 from being planned off the estimate: re-cost from
   measurement instead.
5. No result is reported before its receipt exists.

## Cost, stated before the fact

| item | box | est. |
|---|---|---|
| H1 capture, six families | one 24–32 GB GPU, ~3 h | ~$3 |
| H2 deepseek_v4 load | ≥128 GB RAM, ~3 h incl. fetch | ~$6 |
| H3 | rides H1/H2, no separate box | $0 |

**~$9 estimated, $20 registered ceiling, $35 hard stop.** Under the ≤$20
one-of band; owner already authorised.

## Launcher

Through `adertha-agents/tools/pod-launch.sh` with a reviewed manifest — the
replacement written today, whose refusal paths are tested but which has **never
launched a rental**. Its first real launch is this lane, and that is itself a
result worth recording: if it refuses wrongly, the refusal is the finding and no
money is spent.
