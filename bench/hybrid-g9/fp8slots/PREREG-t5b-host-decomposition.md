# PREREG — T5b: decompose the host bill at the certified point (SPEC-425)

Registered 2026-08-24, before any measurement. Successor to T5
(REFUTED at a stale operating point — see RESULTS-t5's disclosure).
T5's negative result plus the kvapp receipts leave the certified-point
host bill only half-named: step ≈ 131.3 ms on the kvapp box, of which
attention_host 40.2, dram_experts_host 14.0, scheduler 0.3, and
**other_submission 76.4 ms — 58% of the step — never decomposed.**
This cycle names it, then diets the largest named site.

## Phase A — attribution (no edits)

New step_decomp instruments (identity off; the timing arms never carry
them):

- `--host-brackets`: per-step host-wall brackets around (a) each MoE
  module forward (submission side, mirroring the attention bracket),
  (b) the lm_head/logits call, (c) everything else as residual. Output:
  median ms/step per region.
- `--region-ops-out`: `torch.profiler.record_function` regions at the
  same three cut points, WITHOUT stacks (proven broken on these boxes);
  per-region descendant op counts (`copy_`, `to`, `_to_copy`,
  `index_select`, `item`, kernel-launch count) from the event tree.
  The instrument must assert every region name appears ≥ 48×/step
  (MoE) or ≥ 1×/step (lm_head) — a silent no-match must fail loudly,
  not read as zero.

## Operating point (the full command line, pinned — the T5 lesson)

```
python bench/hybrid-g9/step_decomp.py --model /root/qwen3-30b \
  --arena /root/arena/qwen3.arena --calib /root/out/calib.json \
  --vram-gb 10.0 --dram-gb 60 --batch 16 --prompt-len 512 \
  --gen-tokens 128 --chunk 512 --threads 32 \
  --cpu-us-fixed 55 --cpu-us-per-row 2 \
  --prompt-offset 0 --prompt-span 140000 --amort off
```

Batched KV append rides the flipped default (#220); nothing may pass
`--kv-per-seq`. Host: the reference EPYC + RTX 5090 vast class, NUMA
pre-gate (triad ≥ 150 GB/s) as before.

## Gates (before any attribution number is read)

- **G0 (A/A)**: two baseline runs, spread < 7.5%. FAIL ⇒ destroy,
  re-rent.
- **GS (shape-gate — NEW, the T5 lesson)**: the baseline must land in
  the certified band: step ∈ [115, 165] ms AND attention_host ≤ 55 ms
  AND dram_experts_host ≤ 25 ms. FAIL ⇒ the operating point is not the
  certified one (config drift, wrong flags, wrong tree) — ABORT before
  any arm; diagnose, fix, restart. No number from a shape-gate-failing
  run may be quoted.

## Phase A bar

- **H-A (attribution)**: the three named regions + existing brackets
  must jointly cover ≥ 80% of the step (residual < 20%), and some
  single region ≥ 25% of the step. PASS ⇒ Phase B targets the largest
  region. FAIL (nothing ≥ 25%, or residual ≥ 20%) ⇒ the bill is
  diffuse: STOP — record the decomposition, and the ladder re-points
  at T4 overlap (hide a diffuse host bill under device work) instead
  of further dieting; T3 proceeds regardless.

## Phase B — the diet (edits registered by amendment BEFORE any B arm)

After Phase A names the site, an AMENDMENT to this prereg registers
the specific edits (same PR discipline: merged before the box runs the
B arms). The BAR FORMULA is fixed now, before attribution:

- **H-B (wall)**: median step improvement ≥ max(10%, half the targeted
  region's share of the step). Example: region at 30% ⇒ bar 15%.
  PARTIAL band [6%, bar); below 6% ⇒ REFUTED, revert the edits.
- **G1 (identity)**: tokens bit-identical across all arms, as always.
- Consequences: pass ⇒ CERTIFIED, edits default-on; PARTIAL ⇒ ship
  only with zero regressions elsewhere in the brackets; REFUTED ⇒
  revert, ladder re-points at T4.

## Verdict calculator

`t5b_verdict.py`, self-tested both directions per phase, committed
with the instruments before the box is rented. Receipts in
`receipts-t5b/`.
