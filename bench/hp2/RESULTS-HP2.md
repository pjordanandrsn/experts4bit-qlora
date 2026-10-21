# HP2 results: G1 STOP (no competitor reading)

HP2 asked whether e4b K00's held-out loss on Qwen3-30B-A3B is at parity with Unsloth 2026.9.14, better or worse, with the
instrument gated at the mean level (`HP2-PREREG.md`). Its one registered box **stopped at G1**, before any confirmation arm. The
instrument could not read the expected margin. No confirmation run was scored, so this record makes **no parity, BEATS or BEHIND
claim** about either framework. A G1 stop is a reading, not a strike.

**Status: CLOSED 2026-10-19 at G1 (STOP).** `HP2-PREREG.md` stays as registered (2026-10-19, before any data, on #1793); this
record carries the status, the predictions' scores and the routing.

**The null, as an interval:** no e4b vs Unsloth held-out bias difference larger than about 0.015 nats per row is resolvable with
this instrument at R = 300 (|δ0| + 2 × SE = 0.0074; the two frameworks' mean biases are 0.0009 and 0.0036). This is not a claim
that the two frameworks' biases are equal.

## The reading (registered reducer, `bench/hp2/hp2_reduce.py g1`)

`hp2-5090-1`, one RTX 5090, e4b at `68801d3e`. One e4b K00 step and one Unsloth step ran on R = 300 calibration rows (held-out
pool rows 97–396; the pool's first 48 reproduced HP1's held-out rows). Both bases were exported, and both matched their training
runs by digest. The one evaluator then made its zero-adapter pass on each base.

| Statistic | Value | Gate | Result |
|---|---|---|---|
| δ0 = mean_r [g_e4b(r) − g_uns(r)] | −0.002691 | | |
| sd_r(d), R = 300 | 0.041194 | | |
| SE_δ0 | 0.002378 | | |
| \|δ0\| + 2 × SE_δ0 | 0.007448 | CLOSE: \|δ0\| + 2 SE ≤ 0.01 | passed (0.007448) |
| m_need = 2 × (\|δ0\| + 2 × SE_δ0) | 0.014895 | STOP: m_need ≤ 0.005 | **failed: STOP** (0.014895) |

Per framework (reported, never gated):

| | b_f = mean_r g_f(r) | sd_f | mean \|gap\| | max \|gap\| |
|---|---|---|---|---|
| e4b K00 base | +0.000931 | 0.030206 | 0.021715 | 0.12066 |
| Unsloth base | +0.003622 | 0.028295 | 0.021109 | 0.09798 |

## What it shows

- **H1 FALSIFIED.** H1 predicted that the row pairing would reach ρ ≥ 0.74, so that m_need ≤ 0.005. The measured correlation
  between g_e4b(r) and g_uns(r) over the 300 rows is **ρ = 0.009**. sd_r(d) = 0.0412 is the value for independent gaps,
  0.030 × √2 = 0.042. The per-row gap between the evaluator and each framework's own training forward is therefore not a property of
  the row. It behaves as each framework's own numerical noise, and pairing the rows cancels none of it.
- **The mean biases are small; the spread is what limits.** Both frameworks' signed mean gaps are near zero (e4b +0.0009, Unsloth
  +0.0036; HP1's 48-row e4b calibration read −0.0013). The instrument fails on precision, not on bias.
- **H2 UNTESTED.** No confirmation arm ran.
- **Independently re-derived** from the store: `hp2_reduce.py g1` at `68801d3e`, rerun on the stored `hp2/eval_cal.json`,
  reproduced `hp2/g1.json` byte-identically. A separate recompute from both bases' per-row gaps gave sd_d 0.041194 (about
  √(0.0302² + 0.0283²)) and ρ 0.0094.

## Predictions, scored

- **H1** (G1 passes both tiers; the mean bias is near 0 for both frameworks, and the row pairing reaches ρ ≥ 0.74, so
  m_need ≤ 0.005): **FALSIFIED.** The mean biases were near 0 (+0.0009, +0.0036), but ρ = 0.009 and m_need = 0.014895.
- **H2** (K00 vs Unsloth reads PARITY or INCONCLUSIVE): **UNTESTED.** G1 stopped the box before any confirmation arm.

## Routing

**Arithmetic, labelled as NOT a registration.** At sd_r(d) = 0.041, the STOP tier needs SE_δ0 ≤ 0.00125 even at δ0 = 0. That
takes R ≥ about 1,090 rows, and the held-out pool has 400. As built, this differential instrument reads a margin of about 0.015,
not the 0.005 floor. No follow-up is planned unless someone needs 0.005 resolution. A pool of about 1.1k rows would be a new
registration.

**Later reads (forward pointers; this record is unchanged):**
- `bench/hpn/RESULTS-HPN.md` diagnosed the instrument's noise: MOOT at L4 and L8, because e4b's and Unsloth's bases are
  identical under one build.
- `bench/hbd/RESULTS-hbd.md` read the base term's source: at 2 layers it is consistent with a build effect (torch 2.8.0+cu128
  vs 2.12.1+cu130), not a framework effect.

## Draws and spend

| Run | Outcome | Cost | Note |
|---|---|---|---|
| `hp2-5090-1` | OK (`HP2_G1=STOP`) | $2.144 | $0.7411/h; 06:45Z–09:06Z; shakedown OK; teardown verified |
| **Total** | | **$2.144** | registered worst $6.09 |

The files that decide the reading are in [`receipts/hp2-5090-1/`](receipts/hp2-5090-1/README.md), byte-identical to the
private receipt store, with `SHA256SUMS`. The launcher's receipt and ledger row are in that store.
