# P114 — what 4-bit expert projections cost in GPU energy on a rented RTX 5090: the two unchanged energy harnesses, three passes each, to supersede the A2000 energy rows (registered 2026-10-05, before any run)

Lane number claimed by `prereg/p114` (pushed 2026-10-05T19:02Z). Work item: #1133, decision 1.

**Authorization.**
- The maintainer's decision 1 on #1133 (issuecomment-5991633846), option (iii): "rerun the same one-projection harness on a
  rented card (~$0.10, no-ask tier) and supersede both rows". Owner decisions are delegated to the maintainer session.
- The standing no-ask tier for a single run under $15 (#564 issuecomment-5936673039).
- Started in the session that applied #1133's other decisions, on the owner's "continue" after they closed.

Written down by the agent, not by the owner.

## Why this lane exists

**Two register rows and the README's energy sentence rest on the NAS RTX A2000:**

| row | card | build | what it says |
|---|---|---|---|
| `e4b.train.energy-honest.scoped-a2000` | RTX A2000 (70 W cap) | bitsandbytes 0.50.0.dev0 fork | 1.2–2.3× native bf16's J/op; up to 4.4× lower J/token as freed memory buys batch |
| `e4b.train.energy-honest.a2000-bnb0502.2026-10-04` | the same card | bitsandbytes 0.50.2 | 0.91–2.15× |

**Why they are in question.** The A2000 is a correctness-only testbed, a shared production box. The policy does not name
energy, but the maintainer ruled on #1133 that J/op is NVML power integrated over a timed op loop on the shared card. So
other tenants' load reaches it the way it reaches a timing.

**What #1133 decided.** Rerun the same harness on a rented card and supersede both rows. Until then the README's
"1.2–2.3× energy penalty" sentence stays, with its scope stated.

## Subject and instrument

**The harnesses, unchanged** (sha-pinned in `bench/p114/staged.sha256`; `tests/test_p114_lane.py` asserts the pins):
- **`bench/_upstream/bench_energy.py`**, sha256 `ece6b5c8…`: the 2026-07-01 file `docs/METHODOLOGY.md` §10 and both rows
  were measured with.
  - The subject is one OLMoE-dims gate_up projection (out 2048, in 2048) in three arms:
    - native bf16 `F.linear`;
    - `dequantize_4bit` then `F.linear`;
    - `bnb.matmul_4bit` on the packed weight.
  - The workloads are decode M=1 and prefill M=512 (no grad), and train fwd+bwd M=32.
  - It records total J/op, the mean of `nvidia-smi power.draw` sampled at 50 ms over a 6 s op loop, divided by
    throughput.
- **`bench/bench_energy_excluded.py`**: Part A, the memory wall (allocation results only), and Part B, the J/token of the
  fused 4-bit MoE forward at batch 64, 256, 1024 and 4096. Part B is the basis of the scoped row's "up to 4.4×".

**Card.** One NVIDIA GeForce RTX 5090 on `vast:verified-secure`, the project's reference rented class. The policy confirms
its rate, and its launch path is proven. The card's `power.limit` is recorded in `forensics.txt` and quoted with the
reading, because the J/op of a power-capped part depends on its cap.

**Stack.**
- experts4bit-qlora at the launch commit.
- bitsandbytes **0.50.2**, the released build the second row used.
- torch 2.8.0+cu128 from `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- The box's tripwire asserts all three.

**Passes.**
- `bench_energy.py` three times, then `bench_energy_excluded.py` three times.
- 60 s of rest before each pass, so the card returns toward idle.
- Before each pass, `pre_<tag>.txt` records utilization, memory, power and temperature, plus any compute process on the
  card.
- During each pass, `nvidia-smi pmon -s u -d 1` logs every process using the card.

## Correctness first (no energy number if it fails)

`bench/p114/p114_gate.py`, on the card, against the staged harnesses:
- **Same projection.** The dequant and `matmul_4bit` arms compute the same projection over the same NF4 bytes, at M = 1, 32
  and 512: max |dequant − matmul_4bit| ≤ 5e-2 · max |dequant|. Wrong weights or a wrong layout read O(1); a different
  summation order reads about 1e-2.
- **Every arm runs.** Every arm's train backward leaves a finite, non-zero input gradient.
- **Part B runs.** The Part B layer returns a finite output at batch 64.
- On failure, rc 21 and no pass runs. The reducer's self-test also passes on the box before any pass.

**Refusals before any install:**
- the card is not exactly `NVIDIA GeForce RTX 5090` (rc 15);
- `power.draw` does not read as a number on the host (rc 17, which names the host so a retry excludes it): the
  instrument would have no signal.

## Rule (in `bench/p114/p114_reduce.py`, 10-case self-test; fixed here)

The reducer parses every number from the six pass files. It cross-checks the harnesses' printed ratios rather than
trusting them.

**VOID** if any of these holds:
- a pass file is missing;
- a pass ran on another card;
- a cell is missing, an ERROR row or NaN;
- a workload has no native arm;
- `versions.txt` does not record bitsandbytes 0.50.2 on torch 2.8.0;
- a pass's process monitor shows any process but the pass's own;
- the card held a compute process before a pass started.

**NOISY** if, over the three passes, any of these spreads ((max − min) / median) exceeds **0.10**:
- the `matmul_4bit` / native ratio of a workload;
- the dequant / native ratio of a workload;
- Part B's J/token at a batch against batch 64.

**READ** otherwise. The reading is the median of the three passes for each ratio.

**Why 0.10.** It is judgement, written before the data: a row that supersedes another should not move by more than a
tenth between back-to-back passes on one box. It is not derived from any A2000 pass.

## Consequence (registered now)

**READ.**
- **A new row** `e4b.train.energy-honest.5090.<date>` (measured). Its value is the range of the `matmul_4bit` / native
  medians across the three workloads. Its claim carries the dequant arm's range and Part B's 4096-vs-64 J/token ratio,
  with the card's power limit.
- **Both A2000 rows become `superseded`**, pointing at it. `e4b.train.energy-honest`, already superseded, is re-pointed
  directly at it, so no chain is left.
- **The README's energy sentence and every quote of the two rows are restated from the new medians, whatever their
  sign.** The quotes are in README, STATUS, `capabilities.json`, SOLUTIONS, the bitsandbytes solution page,
  BITSANDBYTES.md, STORAGE-MODES.md and METHODOLOGY §10. If 4-bit reads cheaper than bf16 somewhere, the sentence says so.
- **METHODOLOGY §10 gains the 5090 tables.** The A2000 tables stay as the record, under a testbed note.
- **Part A's memory wall stays the A2000's.** It is an allocation fact (bytes), not an energy reading. On a 32 GB card
  both allocations fit, and that is recorded, not compared.

**NOISY or VOID.** No consequence. The rows keep their stated scope. One rerun inside the ceiling, then an amendment.

## Predictions (direction only; no band)

No band is registered. The only readings of these harnesses are the A2000's, and an A2000 energy reading is not a basis
here. The direction comes from the harness's structure on a 5090:
- **Decode M=1.** Each arm is one or two small kernels plus Python dispatch in an eager loop, so throughput is bound by
  host launch rate and the card draws little above idle in all three arms.
  - `matmul_4bit` (one GEMV launch) against native (one cuBLAS launch) should read **near 1**.
  - The dequant arm (a dequantize launch, an allocation and a GEMM) should read **above 1**.
- **Prefill M=512.** bitsandbytes takes `matmul_4bit` through dequantize-then-matmul at M > 1, so it should read close to
  the dequant arm. Both should be **above 1**: the same GEMM plus a weight decode.
- **Train fwd+bwd M=32.** Both 4-bit arms should read **above 1**, since the backward decodes the weight again.
- **Part B.** J/token should **fall** as batch grows, from 64 to 4096 tokens, as the card goes from launch-bound to busy.
- **Verdict.** READ is expected. NOISY at decode is the likeliest alternative, because a launch-bound cell is the most
  host-sensitive.

These are stated expectations, not bands. The rule decides only READ, NOISY or VOID, and the rows follow the medians.

## What this lane cannot say

- **One card, one projection, an eager loop.** Not grouped MoE execution (grouped-nf4-gemm's contract), not CUDA-graphed
  serving, not other cards.
- **Total J only.** The idle-subtracted column is not meaningful, as §10 already says.
- **The host is part of the reading** wherever the loop is launch-bound. The host's CPU is recorded in `forensics.txt`.

## Rehearsal (correctness only: the A2000 is never an energy or timing instrument)

**What runs.** Before any rental, the whole box script runs end to end on the NAS RTX A2000, in a throwaway
`pytorch:2.8.0-cuda12.8-cudnn9-devel` container, with `P114_REHEARSAL=1`. That flag lifts the card check.

**What it checks:**
- install, the tripwire and the gate;
- the six passes running;
- the reducer's parse and its VOID path, which must read VOID off the target card.

No A2000 number is quoted anywhere, and its pass files are not committed as receipts.

## Budget and launch

- **Run.** `p114-5090-<n>`: one RTX 5090 on `vast:verified-secure` at the policy's $0.85/h ceiling, under a guard **1.0 h**
  (expected ~20 minutes):
  - install and gate about 4 minutes;
  - six passes of 1–2 minutes, each after 60 s of rest.
  - `preflight_bandwidth: none`: no Hugging Face checkpoint.
  - No proving rental: the guard does not exceed 1 h.
  - The guard covers the worst case: the install and gate alarms (1,500 s), six passes with their rests and the
    600 s margin come to 3,000 s. `tests/test_p114_lane.py` asserts it.
- **Estimate.** $0.85 at the ceiling (0.85 × 1.0). The expected bill is under a third of that.
- **Lane ceiling** $2.00: the run plus one rerun. **Hard stop** $2.50. Both are inside the no-ask tier.
- **Where things live.** Driven from experts4bit-qlora `bench/p114/` (`p114_drive.sh` → `p114_run.sh`) through
  `pod-launch.sh` from this session's own checkouts on the mini. Receipts go to `bench/p114/receipts/<run>/`, and results
  to `bench/p114/RESULTS-p114.md`.

## STOP rules

1. **No energy number without correctness.** A failed gate ends the run with rc 21.
2. **No pass starts that cannot finish before the launcher's deadline** with the 600 s margin. A skipped pass leaves the
   reading VOID.
3. **No result is reported before its receipt exists** in the private record.
4. **Any teardown that cannot be proven stops the lane.**

Amendments, dated, go below this line before any data is read.
