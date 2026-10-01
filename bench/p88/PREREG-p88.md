# P88 — on K19's new plan, is the int4 decode faster on an RTX 5090 without moving its quality? (registered 2026-10-01, before any run)

Issue: experts4bit-qlora#564. This is P87's instrument (`bench/p87/PREREG-p87.md`), re-run on K19's new default plan,
with the two fixes P87's VOID read called for.

**What changed since P87:**
- **P87** (`bench/p87/RESULTS-p87.md`) read VOID: its calibrated K8 build ran out of its alarm on a Broadwell host.
  Its speed arms passed every check of their own and showed K19, at the plan shipped with #419 (BLOCK_N 64, KC 128),
  at 0.971× the GEMV step at B=16 and 1.236× at B=1.
- **K20** (grouped-nf4-gemm `kernel/RESULTS-k20-k19-plan-sweep-5090.md`, #421) swept K19's plans on a 5090 over
  P60's recorded routing. At BLOCK_N 32 / KC 256 the expert route reads **0.736× the served int8 GEMV route**
  (5.200 vs 7.062 ms/step), 89 % of the byte floor, with outputs bit-identical across plans. That plan is now K19's
  default (gnf4 `7b7e6b1`, #421's merge).

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first (the guard exceeds 1 h);
- receipts and ledger rows;
- proven teardown.

## What is the same as P87 (by reference, unchanged)

- **The question.** Opt-in on vs off: speed at B=16 and B=1, and K8 quality on the licensed recipe.
- **The premise test on the card** before anything is fetched (`tests/test_k19_row_exact_gpu.py`, rc 25).
- **The speed arms:** P86's harness, RTN env and command line, two draws, the first censused.
- **The quality arms:** one calibrated build dumps the pack; OFF and ON load it by fingerprint; P85's env and K8
  arguments.
- **The arm order.**
- **The reducer:** `p88_reduce.py` is `p87_reduce.py` with the lane renamed. Same 17-case self-test, same rule:
  - VOID on the premise, on engagement (96 K19 calls per step ON, the experts' 96 GEMV calls gone, none OFF), on
    missing arms or draws more than 3 % apart, or on a bit-equal K8 ON/OFF;
  - QUALITY_FAIL if |ΔK8| > 0.0095 nats;
  - LICENSED if B=16 ON/OFF ≤ 0.95;
  - NOT_FASTER otherwise.
  - B=1 is read beside the verdict and scopes the default.

## What changes

1. **The kernel pin:** grouped-nf4-gemm at #421's merge, where K19 defaults to BLOCK_N 32 / KC 256 / 4 warps /
   2 stages. The tripwire refuses an install whose K19 default is anything else.
2. **A host CPU floor.** The runner refuses a host whose CPU vendor is not `AuthenticAMD` with **rc 18**, before any
   install. That is a host-floor refusal the launcher excludes the machine on (adertha#131), so a relaunch cannot
   rebuy it.
   - Why AMD: P87's build needed 1,120 s per calibration chunk on an Intel Xeon E5-2698 v4. P85's AMD host needed
     360 s.
   - P85's licensed floats were also read on AMD hosts (P84 amendment 1). OFF's K8 is therefore comparable to them,
     as a report.
3. **The build's arm alarm is 5,400 s** (P87: 3,600), and the build is admitted only with 5,400 s plus the 10-minute
   margin left. **The guard is 3.0 h** (P87: 2.5). At P85's chunk time the whole lane should take about 110 minutes.

## Predictions (written before the data)

- **B=16:** ON/OFF **0.78–0.86**, LICENSED-faster.
  - In-model, the expert route should fall from about 7.7 ms to K20's 5.2 plus the gather/scatter glue P87 censused
    (about 0.3). That takes the step from about 12.0 ms to about 9.8.
- **B=1: SLOWER, 1.03–1.20.** K20 did not measure B=1. One-row tiles still waste 15 of 16 MMA rows, and the plan cuts
  K19's own time but not the tile build or the glue. If B=1 reads SLOWER, a LICENSED default covers T > 1 only.
- **K8:** |ΔK8| < 0.003 nats. The plan moves no output bit, and P64 read bf16 expert activations INDISTINGUISHABLE.
- **The build equals OFF to the bit**, as in P85.
- **OFF reproduces P87's steps** (12.01 and 4.28 ms) within host spread. Reported, not gated.

## Box and cost

1. **`p88-prove-<n>`**: one RTX 5090 on an AMD host, 0.5 h guard at ≤ $0.75/h (≤ $0.375). It runs:
   - the refusals (class, CPU vendor, disk);
   - the install and tripwire (including K20's default plan);
   - the reducer self-test;
   - the premise test;
   - K19's and K16's contract tests compiled on the card. At this pin that includes
     `test_plans_are_bit_identical_compiled`.

   No model. At most three attempts.
2. **`p88-5090-<n>`**: only after a passing proof. **Guard 3.0 h at ≤ $0.75/h (≤ $2.25).**
- **Lane ceiling $3.50; hard stop $4.00.** Both under the $35 per-run cap.
- **Launch only when the cheapest eligible offers are AMD hosts** (read-only offer check first, as for P85). The
  vendor refusal costs about $0.02 when the market moves anyway.

## Rehearsal

The runner differs from P87's in the CPU floor, the tripwire's plan check, the build alarm and the pin. P87's A2000
rehearsal of the proving path stands for the rest. That path ran again on two 5090s in P87 (`p87-prove-1`, and the
reading's premise).

The CI tests (`tests/test_p88_staged_pin.py`) pin:
- the staged bytes and the gnf4 pin;
- the CPU-vendor refusal before any install;
- the premise before any fetch;
- the arm order and settings;
- the driver's dry run.
