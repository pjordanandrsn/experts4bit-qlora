# HPN (DESIGN, $0): where HP2's per-row noise comes from

**Status: DESIGN, $0.** This is the work-well task `heldout-parity-noise-source`. The maintainer accepted it as a $0 A2000 design and
ruled that no P-number registration is needed. This note does a registration's job instead: it, with N0–N3 and the stop rules,
merges before the first GPU second. It makes **no parity claim**. It authorizes no rental. Its numbers are diagnostic: they decide
whether any paired held-out-parity instrument can read a 0.005 margin, and they are never scored data.

## Why

HP2's calibration (`bench/hp2/RESULTS-HP2.md`) read the per-row gap g_f(r) = evaluator zero-adapter loss − framework f's in-loop
step-0 loss on 300 rows, for e4b and Unsloth on their own exported bases:
- sd_f is about 0.03 nats per row for both frameworks;
- the two frameworks' gaps are uncorrelated across rows (ρ = 0.009), so sd_d = 0.041;
- reading a 0.005 margin at that spread would take about 1,090 rows, and the pool holds 400.

What HP2 cannot say is **which side** of g carries the 0.03. The noise could sit in the evaluator's bf16 forward, in each
framework's in-loop forward, or in both. The two answers lead to different next steps:
- **If the evaluator carries it,** a higher-precision evaluator would shrink sd_f, and with it sd_d.
- **If the in-loop forwards carry it,** no evaluator can help. The differential δ0 then cannot reach 0.005 on 400 rows, and a
  parity read would have to drop the per-framework consistency gate or redefine it.

## The measurement

There are three readers per row r, all on the **same exported base** (the same NF4 bytes) and the same rows:
- **I(r):** the framework's in-loop step-0 held-out loss (`inloop_step0_rows`, as in HP1 and HP2);
- **E(r):** the one evaluator (`hp1_eval.py`, bf16, eager), as in HP2;
- **F(r):** an **fp32 reference**. It is the evaluator's own forward with every bf16 cast removed: weights dequantized by the same
  dequantizer but kept in fp32, fp32 projections, attention, norms and logits. The per-expert on-the-fly dequant keeps memory at one
  expert's weight at a time.

Decomposition: g(r) = E(r) − I(r) = [E(r) − F(r)] − [I(r) − F(r)]. Report, per depth, sd_r and mean of each of the three
differences, and the correlation between E − F and I − F.

**Depths.** The A2000 (12 GB) cannot hold the 48-layer base, so the read uses `hp1_trunc.py` truncations of the pinned checkpoint
at L ∈ {2, 4, 8, 16} layers (real weights). It reports how each sd grows with depth. The maintainer's three conditions on
truncated depth:
- **a. N2 is read at every depth that fits.** Its side-dominance call decides only if it is the same at the two deepest depths that
  fit. If it flips with depth, the outcome is "depth-dependent, unknown at 48 layers": stop, with the reason recorded.
- **b. Any 48-layer figure is labelled an EXTRAPOLATION**, with its fit (sd against depth) and its uncertainty. It is never quoted as
  a measurement of the 48-layer base.
- **c. The A2000 is shared.** The household SDXL sidecar can hold about 8 GB until it idle-unloads, so L16 may not fit. Free memory
  is measured and recorded at claim time, along with the deepest depth that fit. A missing L16 is a stated limit, not a failure.
  This is a numerical-correctness read, not a speed read, so it reports no timings.

**Rows.** HP2's 300 calibration rows (held-out pool rows 97–396), so the numbers line up with HP2's.

**The two-reruns control (the maintainer's test for any tolerance).** Each reader runs twice on the same base and rows: E twice,
F twice, and I from two independent one-step arms at the same seed. If a reader's per-row values repeat to five decimals, its
part of g is a fixed per-path offset, not run-to-run noise. If they scatter, the scatter is reported as that reader's own noise.
This decides how the spread is described (offset or noise). It does not change the arithmetic for δ0: either way it is spread
across rows.

**Stages.**
1. **e4b only.** One e4b K00 calibration step per depth (as HP2's phase B, on the truncated model), its base exported, then E and F.
   This needs no new install: the e4b stack is the box's (torch 2.8.0+cu128).
2. **Unsloth, only if stage 1 shows that E − F carries most of the spread.** It repeats stage 1 for Unsloth 2026.9.14 in a fresh
   venv, then recomputes δ0's sd with F in place of E. If I − F dominates, stage 2 is skipped: the in-loop side is the limit, and
   the answer is already in.
3. **Optional.** e4b round trips at L2 over 3 seeds (20 steps). This re-measures the spread of a sound round trip at 48 rows, the
   basis HP2 borrowed for `ROUNDTRIP_MATCH = 0.01` from AXO1767's 8-row control.

## Predictions (stated before data)

- **N0.** E and F repeat to five decimals per row (deterministic eager forwards). I repeats per row at a fixed seed. The per-row
  spread is therefore a per-path offset, the mirror image of tc-campaign's finding that step-0 deltas are an offset.

- **N1.** sd_r(g) grows with depth, and at L2 it is well below 0.03 (rounding accumulates through the layers).
- **N2.** E − F and I − F are of comparable size at every depth. Both forwards are bf16, so neither side dominates. If that holds,
  an fp32 evaluator cuts sd_f by about 1/√2, from 0.030 to about 0.021, so sd_d falls to about 0.030. That is still short of what
  the pool allows: at R ≤ 400 the STOP tier needs sd_d ≤ 0.00125 × √R ≈ 0.025 (sd_f ≈ 0.018 for independent frameworks). The
  outcome would be "no lower-noise reference within reach" and a stop.
- **N3.** E − F and I − F are uncorrelated across rows, as HP2 saw across frameworks.

N2 is the one that decides. It can fail either way, and each way has an action: evaluator-dominant leads to stage 2,
in-loop-dominant leads to a stop with the reason recorded.

## Cost and controls

$0: the seat A2000 under an exclusive `gpu:a2000` claim, about 2 h for stage 1. No rental. The code is a diagnostic script,
`bench/hpn/hpn_read.py`, reusing `hp1_eval.py` and `hp1_lib.py`, with an fp32 switch and tests. It comes as its own PR, reviewed
before the read. The receipt goes into this directory as the HP records do. A result is reported as a decomposition, never as a
statement about either framework's quality.
