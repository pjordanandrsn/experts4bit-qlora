# P67 rehearsal on the NAS A2000 — NOT a reading

**This is a rehearsal. It is not a reading, it licenses nothing, and no number here is a floor for any family.**
It ran a different model (granite-3.0-1b-a400m-instruct, not any family in the register), on a different card
(RTX A2000 12GB, sm86, a shared NAS GPU, not the RTX 5090 class), at a shortened fixture (seq 512, 16 held-out
rows). It exists to answer four questions about the MECHANISM before a rental depends on it, and it is disclosed
in `../P67-PREREG.md`, §"Rehearsal".

Run 2026-09-23 21:19:57Z → 21:48:44Z under the shared `a2000.lock` (taken and released by `gpu_job.sh`;
`logs/gpu_job.lockwait` shows the 69-minute wait for other agents' jobs). Container
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (torch 2.8.0+cu128, triton 3.4.0), transformers 5.17.0,
bitsandbytes 0.50.2, grouped-nf4-gemm 0.32.1 (9206352f, the registered cut), e4b from this lane's tree before its
first commit: `versions.txt` records the sha256 of the two files that matter, `experts4bit_qlora/lora.py`
(a7438470…) and `bench/tp4/tp4_arm.py` (4a7cb27a…). The model came from the NAS store, mounted read-only.

## What ran

`rehearse.sh`: `tp4_arm.py`'s own arms, one process each (the way `tp4_run.sh` runs them), tp4's field recipe
except seq 512 and 16 held-out rows: alpaca (`DS_ALPACA_SHA` 5324987a…), micro-batch 2 × accum 4, r 16 / α 16,
lr 2e-4, AdamW-8bit, linear warm-up 5, seed 3407, N = 20, `--attn-4bit 1` (96 projections). Seven arms, all
`rc=0` (`summary.txt`): `reference_attn4`, `reference_attn4_repeat`, `reference_attn4_perm{1,2,3}`
(`E4B_REFERENCE_EXPERT_ORDER=perm:<s>`), `batched_attn4` (`E4B_BATCHED_PAD_WASTE_LIMIT=64`), `fused_attn4`.
Receipts in `receipts/`, run logs in `logs/`, reduced by the registered reducer into `session.md` /
`session.json` (`p67_reduce.py --session bench/p67/rehearsal-a2000/receipts`).

## The four questions, answered

1. **Does the switch reach the loop, and does the receipt prove it?** Yes. Each perm arm's `reference_order`
   reads `requested == order == perm:<s>` on 4,608 calls (24 layers × 192 each, across training, its
   checkpoint recomputes and the evals); the reference and the repeat read no re-ordering, 0 calls.
   All four floor candidates are admissible.
2. **Are two reference runs on one box bit-identical?** **No, on this card.** The plain repeat matches the
   reference exactly through step 4 and differs from step 5 (D_final 0.00165, D_med 0.00082). So the reference
   path is not run-to-run deterministic on sm86 with torch 2.8's defaults. That does not transfer to the 5090 as a
   fact; it is why the registered draw still runs the repeat instead of assuming either answer.
3. **Is a reorder non-zero, and how big is it at its source?** Non-zero on every seed. perm2 is bit-identical at
   steps 0–1 and diverges from step 2. **perm1 and perm3 already move the step-0 TRAIN loss, by 0.00459 each**
   (2.16933 against 2.16474) — at identical weights, before any update — while the step-0 HELD-OUT loss is
   unchanged at 5 decimals on every perm (2.19687). The two seeds give the same step-0 value, which is consistent with
   one near-tie router decision flipping the same way under either re-rounding (an interpretation: no router trace
   was taken). Either way the perturbation reaches the loss, sometimes at step 0. The prereg's Q3 was revised from "≤ 1e-4 at step 0" to a
   comparison with the accelerated arms because of this (disclosed there). For scale: the accelerated arms move
   step 0 by 0.02612 (batched) and 0.04059 (fused).
4. **Does the reducer read real receipts end to end?** Yes: 4 admissible draws, F_hi 0.00380 (final) / 0.00205
   (median), band max(0.05, 3 × F_hi) = 0.05 on both; fused 0.00136 / 0.00339 and batched 0.00141 / 0.00447 read
   PASS (carried by tolerance), `detectable: false`, D_med / F_hi = 1.65 and 2.18.

What the last line is NOT: evidence about Granite-3.1, or about Gemma-4, or that K = 3 is right. The accelerated
arms' median ratios (1.65, 2.18) sit above every floor draw's own ratio to the maximum (≤ 1), as the prereg
expects of arms that carry larger seeds; that is an observation about one small model on one card.

**Also on this card, separately:** the switch's own unit tests (`tests/test_reference_expert_order.py`) run on CUDA
— 16 passed, 2026-09-24 01:00:49Z → 01:00:59Z under the lock (`logs/switch-cuda.log`): the re-ordered loop equals
the default to 1e-5 relative on the output and every gradient, and moves at least one bit, on sm86 as on CPU.

## Files

| file | what |
|---|---|
| `receipts/*.json` | the seven arm receipts, as `tp4_arm.py` wrote them (adapters and the 2 MB tokens file not kept; the tokens sha is in each receipt) |
| `logs/run_*.log`, `logs/prepare.log` | each arm's stdout/stderr |
| `logs/gpu_job.log`, `logs/gpu_job.lockwait` | the lock wait and the container's rc |
| `logs/switch-cuda.log` | the switch's unit tests on this card's CUDA (16 passed) |
| `summary.txt`, `versions.txt`, `gpu.txt` | the rehearsal's own summary, versions (+ the two file shas), and the card |
| `session.md`, `session.json` | `p67_reduce.py --session` over `receipts/` |
| `rehearse.sh`, `gpu_job.sh`, `nas_setup.sh` | exactly what ran: the in-container script, the lock wrapper, the environment install |
