# Results — P67: Gemma-4's training-parity floor, measured from the reference's own reorderings, and the two accelerated arms read against it (RTX 5090, 2026-09-24)

Pre-registration: [`P67-PREREG.md`](P67-PREREG.md) (#718; Amendment 1 in #737, rates only; both merged before any
rental). Record: [#713](https://github.com/pjordanandrsn/experts4bit-qlora/issues/713). Receipts:
[`receipts/`](receipts/).

**How the numbers were produced.** [`p67_reduce.py`](p67_reduce.py) ran on the controller over the fetched draw with
`--consistency` against the two existing same-fixture sessions, exactly as registered, and wrote
[`receipts/session.md`](receipts/session.md) and [`receipts/session.json`](receipts/session.json). Run again over the
committed receipts it reproduces both tables byte for byte ([`receipts/README.md`](receipts/README.md)). Every number
below is from those files or from the arm receipts they read.

## Runs

| run | what | outcome | cost | receipt (adertha-receipts) |
|---|---|---|---|---|
| `p67-prove-1` | the proof (`bench/p56/p56_prove.sh`) | OK: path proven, `cuda_ok`, **GO-CLASS** — 120 GiB effective host RAM against the 49.9 GiB shard | $0.0097 | `2b5a031` |
| **`p67-gemma4-floor-1`** | **the draw** | **rc 0**, `TP4_SUCCESS`; every run arm OK · VALID | **$1.4426** | `f44c594` |

- **Lane total:** $1.4523 against the $2.625 line. Both teardowns proven.
- **Box.** One RTX 5090 (32,607 MiB, driver 580.105.08, 520 W, 3,105 MHz) on an AMD EPYC 7302 (64 CPUs, 504 GiB;
  cgroup 242 GiB, so **241 GiB effective**, above the 96 GiB floor `p67_run.sh` enforces). Vast verified/secure
  instance 52449460. 2 h 22 min from launch to teardown, inside the 3 h guard.
- **Software.** e4b 0.37.3 at `cd2716d` (this lane's amendment commit), grouped-nf4-gemm 0.32.1 at `9206352f`
  (the registered pin), torch 2.8.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2, peft 0.20.0.
- **Pins and knobs.** All 8 staged files matched `staged.sha256` on the box; every registered knob matched
  `registered.knobs`; no fixture knob was set. The Unsloth venv was built (tp4_run.sh does on every box) and its arm
  skipped by registration; torchao was removed after its tripwire, the standing P38 Amendment 2 class.
- **Authorization.** The owner, relayed at
  [#713](https://github.com/pjordanandrsn/experts4bit-qlora/issues/713#issuecomment-5817951053).

## Engagement (Q1): all seven run arms VALID — HELD

| fact | value |
|---|---|
| `init_sha` | `7588b481…` on every arm |
| tokens sha | `4ea779a5…` (alpaca `5324987a…`, seq 2048, N = 20) |
| trainable | 487,280,640 parameters in 350 tensors, every arm |
| C1 frozen bytes | bit-exact: 30 quantized, 0 unquantized, every arm |
| attention projections | 115 = the structural count, every arm |
| `n_patched` | 30 on `fused_attn4` and `batched_attn4`; 0 on the six reference arms |
| batched engagement | 7,680 calls, 7,680 batched, **0 fallbacks** at `pad_waste_limit` 64 |
| `reference_order` | `perm:1` … `perm:4` on 7,680 calls each; the reference and the repeat: none, 0 calls |

`fused_attn4_nodgrad`, `ckpt_unsloth` and `hf_peft` are `not_run` stubs, as `TP4_SKIP` registered.

## The floor (Q2, Q4) and the judged arms (Q5, Q6, Q7)

D is on the TRAIN loss against the session's own `reference_attn4` (final 1.12894): `D_final = |Δ loss_last|`,
`D_med` = median over 20 steps of the per-step |Δ|. Held-out (48 rows at step 20; reference 1.19205) is beside,
gating nothing. `Δ step 0` is the first step's train loss against the reference's 5.73957.

| floor draw | D_final | D_med | Δ held-out | Δ step 0 | first differing step |
|---|---|---|---|---|---|
| `reference_attn4_repeat` (plain repeat) | 0.02236 | 0.04509 | 0.02474 | 0 | 2 |
| `reference_attn4_perm1` | **0.12100** | 0.08037 | 0.11233 | 0.21726 | 0 |
| `reference_attn4_perm2` | 0.01124 | 0.05226 | 0.02151 | 0.05744 | 0 |
| `reference_attn4_perm3` | 0.11469 | **0.12752** | 0.18907 | 0.05182 | 0 |
| `reference_attn4_perm4` | 0.02563 | 0.07790 | 0.05350 | 0.08194 | 0 |

- **All five draws admissible** (VALID, same session, same fixture key, proof of what ran, not bit-identical), so
  the floor exists (N_MIN = 3): **F_hi(final) = 0.121** (lo 0.011, an 10.8× spread), **F_hi(med) = 0.1275** (lo 0.045,
  2.8×). Bands: `max(0.05, 3 · F_hi)` = **0.363 / 0.383**.
- **Q2 HELD.** The plain repeat is not bit-identical: identical through step 1, diverging at step 2 (4.9109 against
  5.1917), to 0.022 / 0.045 by step 20. Which kernel is the source is not established, as registered.
- **Q4 REFUTED.** Both F_hi are above the registered [0.005, 0.08]: the reference's own reorderings land 1.5–1.6×
  further apart than the point prior (0.02) allowed for at its upper edge.

| judged arm | D_final | D_med | Δ held-out | Δ step 0 | constant 0.05 | floor band | carried by | detectable | D / F_hi |
|---|---|---|---|---|---|---|---|---|---|
| `fused_attn4` | **0.01331** | **0.03565** | 0.01136 | 0.03390 | PASS | **PASS** | tolerance | no | 0.11, 0.28 |
| `batched_attn4` (limit 64) | **0.03694** | **0.04803** | 0.03786 | 0.06538 | PASS | **PASS** | tolerance | no | 0.30, 0.38 |

- **Q7 (the verdicts, not predicted): fused PASS, batched PASS.** Both are carried by the tolerance — in this session
  each arm is inside tp1's constant 0.05 on both quantities on its own — and neither is detectable against the floor
  (every D under 0.4 of F_hi).
- **Q5.** Fused **REFUTED** on both quantities: 0.013 / 0.036 against the registered [0.06, 0.13] / [0.08, 0.15], the
  standing values from two earlier sessions. Batched: D_final 0.037 in [0.03, 0.08] **HELD**; D_med 0.048 against
  [0.05, 0.11] **REFUTED** by 0.002.
- **Q6 REFUTED.** D(fused) < D(batched) on both quantities; P56's ordering (batched closer than fused) did not
  reproduce.
- **Q3 REFUTED.** Every perm draw's step-0 delta (0.052–0.217) is *above* the fused arm's (0.034); two of the four
  (perm1, perm4) are above the batched arm's too (0.065). The registered prediction had every reorder below both.
  At the one deterministic point of the trajectory, the reference's own loop order is a larger perturbation of this
  model's loss than the fused kernels are.

**Consistency (registered): not MIXED.** Both existing same-fixture sessions have the new session's fixture key and
were read against its floor. Their judged pairs — tp4-c-parity-2 fused 0.083 / 0.124; p56-gemma4-ladder-3 fused
0.102 / 0.106, batched 0.054 / 0.085, fused_nodgrad 0.103 / 0.117 — all read **PASS**, carried by the floor, at
0.45–0.97 of F_hi. No disagreement, so the reading is READ.

**What the three sessions show together (reported, not registered).** The step-0 train loss is the same in all
three sessions, on three hosts, per path: reference 5.73957, fused 5.70567 (Δ 0.0339 every time). The 20-step
endpoints are not: the reference alone finishes at 1.1605, 1.1819 and 1.1289, and fused at 1.0779, 1.0796 and
1.1423. The earlier sessions' fused D of 0.08–0.10 and this one's 0.013 are that trajectory scatter, and they all
sit inside the range the reference's own reorderings span here (0.011–0.121).

## What the decision rule did

Registered table, `gemma4_text`, attention-4-bit configuration, **fused PASS and reading not MIXED**:

- **`fast_train` (attention 4-bit) → supported.** "The fused path is indistinguishable from the reference's own
  reordering noise on this model." Registered as
  `e4b.train.p67.gemma4.fused-attn4.floor-band.5090.2026-09-24`.
- **`reference_train` (attention 4-bit) → supported.** Its arm is VALID, and the floor draws are five more VALID runs
  of it.
- **The floor itself** is registered as `e4b.train.p67.gemma4.attn4-reorder-floor.5090.2026-09-24`, superseding
  P56's proxy (`e4b.parity.gemma4.train-floor`, the batched arm as the smallest perturbation): the floor is now the
  reference against itself, and it is larger than that proxy.
- **`batched_attn4` PASS** is recorded as `e4b.train.p67.gemma4.batched-attn4.floor-band.5090.2026-09-24` at
  `pad_waste_limit` 64. It does not license the shipped default (4); `batched_train` stays as tp1 left it.
- **Not moved:** `fast_train` with bf16 attention (tp1's PASS stands), every other family's `training_support`,
  tp1's tolerance. No default moves; #558 does not reopen.

## What this read says, and what it does not

- **The licence is a statement of resolution, not of equality.** On Gemma-4 at this fixture, two correct runs of the
  reference land 0.011–0.121 apart at step 20; a 20-step trajectory cannot resolve a 0.05-nat defect in an
  accelerated path against that. The registered rule makes such an arm supported; it does not make it identical.
  The instrument that *could* detect a difference on this family is per-op or matched-routing, as the registration
  named for the other branch.
- **The predictions mostly failed in the direction of noise.** The floor is bigger than predicted (Q4), the reorder
  perturbs the step-0 loss more than the kernels do (Q3), and the accelerated arms landed closer than their standing
  values (Q5, Q6). The verdicts were never predicted (Q7).
- **Not established:** which kernel makes the plain repeat non-deterministic (Q2's basis); anything about the
  bf16-attention paths; any step-time or speed statement (recorded, not quoted, as registered).
- **Scope.** One session, one box, one family, one fixture, N = 20. The other families' floors were not drawn (Q8).
