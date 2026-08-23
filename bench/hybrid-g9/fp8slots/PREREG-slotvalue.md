# PREREG — the slot-value curve: certifying the FP8 workstream's serving payoff

Registered before any measurement. Verdicts computed by the committed
[slot_verdict.py](slot_verdict.py) (self-tested: a synthetic sweep
embedding the law certifies; one with a broken conversion constant is
REFUTED); sweep driven by the committed [slot_sweep.py](slot_sweep.py)
over the extended in-repo `bench/hybrid-g9/step_decomp.py` — no
disposable box scripts this time (the g8 harness lesson).

## The claim under test, quoted

From `bench/hybrid-g9/concentration/RESULTS-router-concentration.md` §3:

> **More VRAM slots** (FP8/param-quant workstreams already on the open
> list): every slot removes its expert's P(touched) × 58 µs/step. At the
> tail's P ≈ 0.3–0.6, each additional slot is worth ~17–35 µs/step — a
> concrete slot-value curve the FP8 certification work can now cite.

Three certifiable parts: (B1) a slot removes its expert's **P(touched)**
expected uniques; (B2) each removed unique is worth **58 µs** of the CPU
decode bill; (B3) the **tail** slots price at the quoted band. A 17 µs
marginal on a ~23 ms wall is unmeasurable at rented-box noise, so the
design scores ms-scale **brackets** of an integrated sweep and derives
the marginal from the certified curve — never a single slot directly.

## Operating point (the b16close reference)

`Qwen/Qwen3-30B-A3B`, B=16, prompt 512, chunk 512, gen 48 decode steps,
NF4 arena, Fp8PagedKV, serving playbook constants (`cpu_us_fixed=55`,
`cpu_us_per_row=2`, torch intraop 8, pool = min(32, physical)). Host
class: single-socket Zen 4/5 EPYC, ≥ 32 physical cores, GPU ≥ 24 GB
(prefer 9655 + RTX 5090 to mirror the reference receipts). gnf4 pinned
at `f08a66f`; e4b at this PR's merge commit.

## Protocol

1. On-box: bake the NF4 arena (`gnf4 kernel/nvme_bake_nf4.py`, flags
   recorded in receipts), run `gnf4 bench/calibrate.py` for calib.json.
2. **Profile pass** (`slot_sweep.py --profile-pass`): one serving run at
   the top budget captures the model's own decode routing as an
   `expert_profile` JSONL. Every placement in the sweep uses it.
3. **Sweep**: ladder `vram_gb ∈ {10.0, 12.0, 13.0, 13.5, 14.0}` (finer
   near the reference capacity clamp, where the quoted tail band lives),
   `dram_gb = 60` so the NVMe tier stays EMPTY (any NVMe unique aborts
   the run — the cold path must not contaminate the DRAM bucket). Each
   (pass, budget) point is a fresh **subprocess** (no allocator/cache
   coupling); pass 1 walks the ladder descending, pass 2 ascending, so
   monotone box drift shows up as a pass split the gate scores.
4. **A/A gate** (from the lowg re-cert law): per-point noise between the
   two passes on the `dram_experts_host` bucket. ACCEPT iff median ≤ 5%
   and every point ≤ 10% — else destroy and re-hunt, **up to 3 hosts**,
   then UNRUNNABLE. The gate is computed from the same sweep that gets
   scored (the passes ARE the A/A), so a gate REJECT discards the box's
   sweep unscored.

## Bars (noise-derived; per bracket between adjacent ladder points)

Bracket experts = the set difference of the two manifests' VRAM tiers
(identical across passes by solver determinism — asserted). With
`P_e = 1 − (1 − p_e)^16` from the profile:

* **B1 (uniques accounting)**: measured Δuniques/step within
  `max(15%, 3 × |pass1 − pass2|)` of `Σ P_e` over the bracket.
* **B2 (time conversion)**: measured Δ(dram bucket) within
  `max(25%, 3 × |pass1 − pass2|)` of `58 µs × Δuniques_meas`. The 25%
  floor absorbs the known second-order terms (per-call fixed-cost
  amortization shifts, the bounded 2.4 µs/row interaction) — the claim
  certified is the dominant linear term.
* **B3 (the tail)**: the top bracket's measured value, `Δ(dram bucket) /
  Δslots`, lands in **[12, 45] µs/slot** (the quoted ~17–35 widened 30%
  for its own "~").
* **Spoiler (could it have failed)**: full-range ΔT must exceed 3× its
  pass spread, else VOID — a sweep whose effect is inside noise cannot
  certify anything.

CERTIFIED = every bracket passes B1 and B2, B3 passes, spoiler holds.
Anything less is REFUTED (constants wrong at serving shape — recorded,
no bar-shopping) or VOID/UNRUNNABLE as above.

## Reported, not scored

The FP8 KV accounting that names the payoff: freed bytes = bf16-KV −
fp8-KV at the operating shape (from `Fp8PagedKV`'s own geometry,
k_groups=4), slots = freed // bytes_per_expert (from the arena index),
**payoff = slots × the certified tail value**, stated at the measured
context and, clearly labeled, at longer contexts. G7 already closed FP8
KV quality/bandwidth/kernel 3/3; this experiment prices what those
slots BUY, it does not re-litigate FP8 correctness.

## Hard stop

Up to 3 gate hosts; ONE scored sweep total. REFUTED closes the
slot-growth pricing as stated (the placement/solver constants would
need re-derivation at B=16 before any FP8 payoff claim is made).
