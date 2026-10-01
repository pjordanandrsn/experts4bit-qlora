# P90 — does K21 on gpt-oss-20b's native MXFP4 store make its B=16 decode faster on an RTX 5090, without moving the store's quality? (registered 2026-10-01, before any run)

Issue: experts4bit-qlora#564. This is the first end-to-end lane of the throughput push to other model families, on
the family whose quantized expert store is licensed (P44).

**Where the step goes, and what this tests:**
- **gpt-oss-20b's native MXFP4 store** (`store_r12`) is licensed by P44's KL-from-reference instrument: 0.0019 nats
  against 0.0222 for the NF4 requant control (`e4b.serve.p44.gptoss.store-r12.kl-vs-bf16.2026-09-19`).
- It has no batched kernel. Above 16 rows its experts fall back to the kept NF4 stacks, so at B=16 (64 rows) the step
  runs the NF4 requant.
- grouped-nf4-gemm's K22 and K24 censused that step on two RTX 5090s: **22.50–22.52 ms, 79 % of it one kernel**,
  `_gemm_nf4_grouped`, at 17.86–17.90 ms per step.
- On gpt-oss's recorded B=16 routing, K21 (#422, with the masked K tail of #425) read **7.62 ms/step against the served
  route's 15.18** (K24, #428). Both microbench reads were VOID by their instrument (the bench's NF4 kernel read 17 %
  under the census). The ratio is descriptive, which is why this lane measures the model's own step.
- **The route under test** is experts4bit-qlora #816, `E4B_MXFP4_GROUPED_SMALLM=1`. Every device-grouped decode row
  on the MXFP4 store goes through K21 at K24's best plan (BLOCK_N 32 / KC 128 / 4 warps / 3 stages):
  - at B=16, instead of NF4;
  - at T == 1, instead of the split-K GEMV.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first (the guard exceeds 1 h);
- receipts and ledger rows;
- proven teardown.

## The question and the instruments

**Speed.**
- bo7's `store_r12` env, the configuration K22/K24 censused and P44 licensed for single rows, with OFF
  (`E4B_MXFP4_GROUPED_SMALLM=0`) and ON (`=1`).
- B=16 and B=1, each drawn twice, the first draw censused.
- K22's harness command line: P58's `step_decomp.py`, graph-replay timing, `--no-fuse-qkv`, prompt 512, 128
  generated tokens.
- Order: B=16 OFF, B=16 ON, B=1 OFF, B=1 ON (censused), then KL OFF, KL ON, then the second draws ON/OFF.

**Quality.** P44's instrument, unchanged: `bench/p44/kl_serve.py` on arm `store_r12`.
- The reference is the dequant of the same shipped MXFP4 bytes.
- 200 committed prompts, max length 320.
- Decode-shaped on both sides: one token per forward, the shape control (i) admitted for gpt-oss.
- It runs OFF (scoring and caching the reference) and then ON (reusing the cache).

Under ON, T == 1 on the store is routed to K21 (#816's `_collapsed_grouping`), so the decode-shaped KL reads the kernel
it gates. Two checks carry it to the batched rows:
1. **The premise** (rc 25 before anything is fetched): `tests/test_k21_row_exact_gpu.py` on the card. A token's K21
   rows are bit-equal alone (T = 1) and inside a B=16 step, through gpt-oss's epilogue. Its T = 1 output matches an
   fp32 dequant oracle, and the T = 1 route captures and replays bit-equal. 3 passed, none skipped.
2. **The K0 controls** (rc 26): the instrument's own gate on this host.

**The reducer:** `p90_reduce.py`, with an 18-case self-test.
- **VOID** if:
  - the premise did not hold, did not run, or skipped a test;
  - K0 did not pass;
  - a draw is missing, or two draws of one arm are more than 3 % apart;
  - a census is missing;
  - a route is not the registered one (per decode step):
    - B=16 OFF: 48 NF4 grouped GEMMs and no K21;
    - B=16 ON: 48 K21 calls and no NF4 grouped GEMM;
    - B=1 OFF: 48 MXFP4 GEMVs and no K21;
    - B=1 ON: 48 K21 calls and no GEMV;
  - a KL row is missing or not decode-shaped, or the two rows scored different token counts;
  - OFF's KL does not reproduce P44's licensed 0.0019 within 0.001.
- **QUALITY_FAIL** if KL(ON) − KL(OFF) > 0.0005 nats (the reference's own decode-vs-prefill floor, P44), or top-1
  agreement drops by more than 0.002. K21 then stays opt-in whatever its speed.
- **LICENSED** if quality holds and B=16 ON/OFF ≤ **0.90**.
- **NOT_FASTER** otherwise.
- **B=1** is read beside the verdict: FASTER ≤ 0.97, SLOWER ≥ 1.03, else NEUTRAL. It scopes the default a LICENSED
  read proposes: all decode rows, or T > 1 only (P88's precedent for K19).

## Predictions (written before the data)

- **B=16 ON/OFF 0.55–0.70: LICENSED.**
  - The expert GEMM goes from about 17.9 ms/step to about 8–9 (K24's 7.6 descriptive, plus in-model overhead).
  - The rest of the step, about 4.6 ms, is unchanged. That puts the step at roughly 12.5–14 ms.
- **B=1 SLOWER, 1.03–1.30.** One-row tiles waste 15 of K21's 16 MMA rows. The split-K GEMV was chosen for exactly
  these rows (bo3n), and K19 read 1.10× at B=1 on Qwen3 (P88). If B=1 reads SLOWER, a LICENSED default covers T > 1
  only.
- **KL(ON) ≤ KL(OFF).** The T == 1 GEMV quantizes activations to int8 per 32-block, while K21 keeps them bf16; the
  weights are the same exact bytes either way. I expect KL(ON) about 0.0015–0.0019 and top-1 within ±0.001.
- **OFF reproduces K22/K24's census step** (22.5 ms) within host spread. Reported, not gated.

## Box and cost

1. **`p90-prove-<n>`**: one RTX 5090, any CPU vendor, 0.5 h guard at ≤ $0.75/h (≤ $0.375). It runs:
   - the refusals (class, disk);
   - the install and tripwire (K21 with its masked tail, the route and its plan, the opt-in unset, the GEMV row limit,
     `Mxfp4Config` for the reference);
   - the reducer self-test, the premise and K0;
   - K21's and K16's contracts compiled on the card.

   No model. At most three attempts.
2. **`p90-5090-<n>`**: only after a passing proof. **Guard 2.5 h at ≤ $0.75/h (≤ $1.875).**
   - Fetch about 13 GB and bake.
   - Eight timed arms of about 3 minutes.
   - The KL: the reference plus two arms, about 40 minutes (P44-b gave gpt-oss 3,000 s with fetch and bake).
- **Lane ceiling $3.00; hard stop $3.50.** Both under the $35 per-run cap.

## Rehearsal

The proving path ran on the NAS RTX A2000 (sm_86) from e4b `ecfb980`, this branch before this paragraph, and
grouped-nf4-gemm `4cc831c`. The staged bytes are pinned by `staged.sha256` and are unchanged since. It ran with
`P90_GPU_CLASS=A2000 P90_MIN_DISK_GB=20`, so it was marked REHEARSAL. It read:
- the reducer self-test, 18 cases;
- install and tripwire OK;
- the premise, 3 passed;
- K0 all passed;
- K21 + K16 contracts compiled, 23 passed;
- rc 0.

**Not rehearsed:** the speed and KL arms. gpt-oss-20b's MXFP4 store, kept NF4 stacks and the dequant reference do not
fit a 12 GB card, so the reading is their first run. A failure there reads VOID: a missing arm, census or KL row is a
VOID case. No time is quoted (the A2000 is correctness-only).

Amendments, dated, go below this line before any data is read.

### Amendment 1 (2026-10-01, after `p90-5090-1` read VOID; before the next reading's data)

**What happened.** `p90-5090-1` (RTX 5090, $0.118) read **VOID**: both KL arms exited rc 1, so there is no quality
row.
- The failure was the **reference**, not the route. gpt-oss-20b's bf16 dequant reference is about 40 GB and does not
  fit a 32 GB card.
- transformers offloaded part of it to the CPU, and its grouped-MM fallback then raised a device mismatch
  (`logs/kl_off.log`).
- P44-b scored this reference on an **H100 NVL** (94 GB). This registration put the KL arms on a 5090 without checking
  the reference's size. That was my design error.

**The amendment.** One reading is now **two runs**, selected by a new registered knob, `P90_ARMS`:

| | `speed` | `quality` |
|---|---|---|
| card | RTX 5090 (as registered) | H100 NVL, P44-b's card class for the licensed row |
| arms | the registered speed arms, unchanged | the registered KL arms, OFF then ON, unchanged |
| guard | 1.0 h at ≤ $0.75/h, so no proving rental under the rule | 1.5 h at ≤ $3.10/h (≤ $4.65), with a proving rental first (`P90_PROVE=1 P90_ARMS=quality`, 0.5 h, ≤ $1.55) |

- Both runs hold the premise and K0 on their own card. Each writes `part_<arms>.json` via `p90_reduce.py --part`.
- **The verdict** comes from `p90_reduce.py --speed-dir <speed run> --quality-dir <quality run>`. It is the registered
  rule unchanged, except that every run's premise and K0 must pass.
- **The quality claim is K21's arithmetic on the H100.** P44's licence of the store (the GEMV arithmetic) was read on
  the same card class and applied to 5090 serving, so this follows that precedent. The premise holds row-exactness on
  each card.
- **The lane ceiling rises** from $3.00 to **$8.00** (hard stop $9.00), still under the $15 single-run tier.

**`p90-5090-1` is not re-read.** Its speed numbers are recorded descriptively only:
- B=16 22.432 → 13.097 ms/step (×0.584), draws within 0.3 %;
- B=1 5.796 → 6.226 (×1.074);
- the census routes as registered: per decode step, K21 48 and NF4 0 at B=16 ON, NF4 48 at OFF; K21 48 and GEMV 0 at
  B=1 ON, GEMV 48 at OFF.

The verdict comes from fresh runs at this amendment's merge.
