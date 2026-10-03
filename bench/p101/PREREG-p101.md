# P101 — P98's question asked again on the fixed code: Qwen3.6-35B-A3B's bucketed decode graphs through the serving stack, against its padded eager step, and the first hybrid decode speed, on one RTX 5090 (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#564; the fault #913, fixed in #918. Follows P98 (#912, VOID) and P99 (#914 / #915). (P100, #917, is the prefill lane; this is P98's question.)

**Why.**
- P98 asked whether Qwen3.6's bucketed decode graphs, through the production serving stack, replay exactly as the
  padded eager step, and how fast they decode. It read VOID: every bucket captured, then a replay hit a device-side
  assert.
- **The cause** (#913) was the expert engine's single-entry row-to-token index cache. Each bucket's eager capture
  warm-up freed the index an earlier bucket's graph still read.
- **#918 fixes it.** `tests/test_rt_cache_graph_gpu.py` reproduces the fault on its own: it fails with the fix
  reverted (silently, max abs 1.77e-2) and passes with it.
- **This lane** asks P98's question again on the fixed code, with P98's measurement, reducer and rule unchanged.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

- **P98's kit at its registered bytes,** run through `bench/p98/p98_drive.sh` (`bench/p98/staged.sha256` unchanged:
  `tests/test_p98_staged_pin.py` holds it). That covers:
  - the bake (`p98_bake.py`) and the box harness (`p98_box.py`);
  - the reducer (`p98_reduce.py`, 19 cases);
  - the runner (`p98_run.sh`) with its premise: the three hybrid GPU test files, 4 tests, none skipped.

  Everything in `bench/p98/PREREG-p98.md` holds as written: the model and revision, the three arms (g graphs, e the
  padded-eager oracle, p plain eager), the workloads (W16, W1), the gates, and the VOID, SUPPORTED and NOT_SUPPORTED
  conditions.
- **What differs is the code under test:** e4b at this lane's launch commit, which includes #918. The run ids are
  `p101-prove-<n>` and `p101-5090-<n>`; the box-side directory and logs keep P98's names (`/root/p98`, `P98_*`),
  because the kit is P98's.
- **The verdict is `p98_reduce.py`'s,** read under P98's rule.

**The registered consequence** (P98's, now live):
- **SUPPORTED:** `docs/SERVING.md`'s hybrid paragraph cites this lane as the GPU reading of hybrid decode graphs, with
  their speed, and its warning against `E4B_PAGED_GRAPHS=1` on hybrid models is lifted.
- **NOT_SUPPORTED:** the failing bucket or step is recorded on #913 before any change; the warning stays.
- **VOID:** nothing moves.

## Predictions (written before the data)

- **SUPPORTED.** Every bucket captures and replays, with no eager step, and arm g's tokens equal arm e's on all 17
  requests.
- **Graphs at least 3× plain eager on both workloads.**
  - P98's eager arms were launch-bound: about 85 ms per decode step at 1 row and at 16. W16 read 107.3 tok/s plain and
    W1 11.5.
  - A graph removes the per-op launch and Python overhead.
  - Arm g's W1 lands between 30 and 150 tok/s, and its W16 above 350 tok/s.
- **Arms e and p reproduce P98's eager readings within 10 %:** W16 128.3 / 107.3 tok/s, W1 11.98 / 11.46.
- **Peak GPU memory** stays at or below 28 GB in every arm.

## Box and cost

- **`p101-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P98_PROVE=1`: the install with its
  tripwire, the reducer's self-test, the premise and the HF CDN probe; no model.
- **`p101-5090-<n>`:** one RTX 5090 with ≥ 200 GB of disk. **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).** P98's reading
  took 859 s, including a crashed arm; with three full arms it should take about 25 minutes.
- **Lane ceiling $2.50; hard stop $3.50.**

## Rehearsal

None new. Every piece is P98's at its registered bytes, and P98 rehearsed it on the A2000 and ran it on a 5090 in
`p98-5090-2`:
- the bake;
- `build_engine`;
- arms e and p end to end;
- arm g through capture.

The new piece is #918's fix. It was verified on the A2000 by its reproduction test (fails with the fix reverted,
passes with it) and by 27 neighbouring expert-engine GPU tests.

Amendments, dated, go below this line before any data is read.
