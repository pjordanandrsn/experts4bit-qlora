# P46 — THE ADAPTER PATH: does batching the per-expert LoRA delta remove the host bill P45 found? (pre-registered 2026-09-19, before any box is rented)

Work item: adertha-agents#110 (throughput + training campaign). Lineage: P45 (`bench/p45/RESULTS-p45.md`, run `p45-qwen3prof-2`): e4b's Qwen3-30B-A3B field-recipe training step is host-bound — GPU busy 10.8 %, ~1M device events and ~273k `aten::mm` per optimizer step, the per-expert LoRA adapter matmuls run as a Python loop (`nf4_qlora._lora_delta_grouped_loop`) with their autograd. The kernel package already carries a padded-`bmm` path for that delta (`lora_delta_grouped`, measured 3.0× on the end-to-end step at E=256 on an A2000 when it shipped) behind a padding-waste guard (`_PAD_WASTE_LIMIT = 4`): under the field recipe's router skew the guard sends `auto` to the loop. grouped-nf4-gemm's P46 change (#369, main `d8f737a`) made the path a recorded choice — `NF4_QLORA_LORA_PATH` = `auto` | `padded` | `loop` | `grouped_mm` (two `torch._grouped_mm` calls over the jagged groups, no padding), `NF4_QLORA_PAD_WASTE_LIMIT`, and `LORA_PATH_STATS` counters that every training step's kernel-call census now carries (`lora_path_*` keys). This lane measures the three paths at the field fixture with the parity anchor beside them. No default changes here; the decision rule says what the next release ships.

## Arms (one RTX 5090, `TP4_BOX=F`, family `qwen3lora`, 20 steps each, timed, no profiler, same tokens file)

| arm | path | what it is |
|---|---|---|
| `reference_attn4` | per-expert reference forward (no fused kernel) | tp1's parity anchor (B2/C2) |
| `fused_attn4` | `auto` (the shipped rule) | P45's arm; the census must show `lora_path_loop > 0`, `lora_path_padded = 0` per step |
| `fused_attn4_pad` | `padded` | the padded `bmm` path forced on |
| `fused_attn4_gmm` | `grouped_mm` | the jagged path; **may refuse on sm_120** (torch's grouped-GEMM kernel coverage) — a refusal is a recorded row, not a failure of the lane |

Fixture = the field recipe (alpaca, seq 2048, mb 2 × accum 4, r 16, α 16, adamw_8bit, linear warm-up 5), the same tokens sha as P43/P45. e4b at this PR's merge commit; gnf4 at the P46 merge (`d8f737a608a30112707a593d815be1df3da8f479`, unreleased — a lane pin, stated in the receipt); transformers 5.17.0, torch 2.8.0+cu128.

## Registered predictions (falsifiable)

- **P1 (mechanism):** `fused_attn4` (`auto`) takes the loop on every step at this fixture: `lora_path_loop ≥ 1` and `lora_path_padded = 0` in every step's census. Refuted → P45's attribution of the ~273k matmuls to the loop was wrong and the lane stops to re-read.
- **P2 (the padded path):** `fused_attn4_pad` trains at **≤ 0.55×** the s/step of `fused_attn4` (median over steps 11+; from 29.3 s to ≤ 16 s — the A2000 E=256 measurement was 3.0×, the field fixture has 4× more rows per expert, so the prediction is deliberately below the shipped figure). Refuted at > 0.8×.
- **P3 (the jagged path):** `fused_attn4_gmm` either refuses (recorded, P3 NOT READ) or trains at ≤ 0.45× of `fused_attn4`.
- **P4 (parity, the gate):** every fused arm is within tp1's B2/C2 band of `reference_attn4` on the same tokens: |Δ held-out final| ≤ 0.05 nats AND median per-step |Δ loss| ≤ 0.05. An arm outside the band is refused as a lever whatever its speed.
- **P5 (memory):** `fused_attn4_pad` peak VRAM ≤ 1.4 × `fused_attn4`'s (the shipped path's +36 % at E=256 is the prior).

## Decision rules

- P1 ∧ P2 ∧ P4(pad) → the padding-waste guard is the defect: the next grouped-nf4-gemm release moves `auto` to the padded path at this skew (raise `_PAD_WASTE_LIMIT` to the measured ratio or replace the rule with a rows-per-group cutoff), e4b's field-recipe H2H is re-run (tp4 box B) on the new cut, and the training position moves only from THAT receipt.
- P3(gmm) ∧ P4(gmm) faster than pad → `grouped_mm` becomes `auto` where torch has the kernel, padded elsewhere; same re-run.
- ¬P2 ∧ ¬P3 → the loop is not the lever P45 named; the lane reports what the census then says.
- ¬P4 on an arm → that arm is refused, whatever its speed (the parity gate is the gate).

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p46-qwen3lora` | RTX 5090 | 0.65 | 2 h | $1.30 |

Under the $35 cap. STOP: not a 5090 (refused); egress below the floor; a fetch alarm → `not_run`; the gnf4 cut on the box lacking `LORA_PATH_STATS` → the family refuses before any arm (tripwire). A second host-limited draw → stop and report.

## Receipts

`receipts/experts4bit-qlora/<date>/p46-qwen3lora/tp4/` — `qwen3_e4b_{reference_attn4,fused_attn4,fused_attn4_pad,fused_attn4_gmm}.json`; read by `bench/p46/p46_reduce.py` (P1–P5 against the numbers above). Amendments dated below, before the data they touch.

## Amendments

(none yet)
