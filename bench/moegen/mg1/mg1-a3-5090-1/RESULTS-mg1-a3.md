# MG1 amendment 3: the dgrad kernel served every frozen-GEMM backward of Qwen3.6's licensed fused arm, and P8 holds

*MG1 amendment 3 ([`../../MG1-PREREG.md`](../../MG1-PREREG.md#amendment-3-2026-10-04-after-amendment-2s-box-and-before-its-own-p2-for-qwen36-from-tp1s-own-fused-arm),
registered in #1085 before its box). Work item experts4bit-qlora#1049. Read 2026-10-05 from the files in this directory.*

**The answer: P8 HELD.**
- tp1's arm driver ran Qwen3.6-35B-A3B's fused arm resident for its 60 steps, with the committed file unchanged under
  `bench/moegen/p2_hook.py`.
- grouped-nf4-gemm's `DGRAD_STATS` read **kernel 4,800, loop 0** (grouped_mm 0, dense 0, no loop reasons). That is 80 frozen-GEMM
  backwards per step (40 MoE layers × 2 expert GEMMs) for 60 steps, every one on the dgrad kernel.
- 40/40 MoE layers were patched, with no recurrent-kernel fallback.
- The census's driver sha256, `d02c9e7b…5e60b129`, equals the committed
  `bench/train-parity-20260905/tp1/logs/tp1_train_smoke.py`.
- With the reading's PASS (`e4b.train.parity.mg1.qwen3_5.fused.2026-10-04`), this completes MG1's registered rule:
  `qwen3_5_moe.fast_train` enters as `supported` and the family joins `qlora-fused-moe-experts.model_families`.

## The draw

| run | box | outcome | cost |
|---|---|---|---|
| `mg1-a3-5090-1` | RTX 5090, machine 145701 (EPYC 7B13) | **the reading.** Install, tripwire, the recurrent kernels (mamba-ssm, causal-conv1d, flash-linear-attention) and the 67 GB fetch were OK. The runner printed `AMENDMENT 3 SHAPE (registered)`, the fused arm exited 0, and the `P2 census` line is in [`summary.txt`](summary.txt) ([census](receipts/qwen3_5_p2_dgrad.json), [arm receipt](receipts/qwen3_5_train_fused.json), [arm log](logs/run_qwen3_5_fused.log), [launch.json](launch.json)) | $0.803 |

Machine 145701 is the host that MG1's train anchor refused on the reading's first draw (`h2d.self_pair` 1.0321). Amendment 3
skips the anchor by registration, because P8 is an engagement count, not a timing, so the refusal does not bear on this
reading. It does mean the arm's time below is not comparable with anything.

## P8, condition by condition

| condition | reading |
|---|---|
| census present | yes: the driver reached interpreter shutdown |
| `DGRAD_STATS["loop"] == 0` with `kernel` > 0 | loop 0, kernel 4,800 |
| no recurrent-kernel fallback | `FAST_TRAIN_STATS["recurrent_fallbacks"]` = `[]` |
| driver sha256 = tp1's committed file | `d02c9e7b2357c521…` on both |
| the arm's receipt is `ok`, 40/40 patched | `status: ok`, `n_patched` 40, `FAST_TRAIN_STATS["patched"]` 40, the fused forward at least 160 calls on every step (80 required) |

## The arm's other numbers (informational, never a verdict)

This box has no anchor and no reference arm, so nothing here is a parity verdict, a timing position or a cross-box ratio.
- **Same as the reading's fused arm on machine 151350:** `init_sha` `e140b408…`, 463,093,760 trainable parameters and a peak of
  27.73 GB.
- **Different:** 1.187 s/step against 0.377 (a different host, outside the anchor band). Train loss 3.0123 → 0.2496 against
  3.0071 → 0.2538; held-out eval 3.0009 → 0.3129 against 2.9995 → 0.2751.
- Those loss differences are two runs of the same arm on two boxes. That is not a registered comparison, and this read draws
  no conclusion from it. The licence's loss parity is the reading's same-box fused-vs-reference verdict.
- The `VOID … no OK reference arm` row that `mg1_reduce.py` printed into `RESULTS-mg1.txt` is that reducer seeing a fused arm
  without its pair, as the registration expects. It is not a verdict on this family.

## Decision (amendment 3's rule, which is amendment 2's)

Loop is 0, so `qwen3_5_moe.fast_train` becomes `supported`, citing the reading's PASS and this census
(`e4b.train.parity.mg1.qwen3_5.p2.2026-10-05`). The family joins `model_families`. The fused row now carries its own
`licensed_by`.

Spend: $0.803 for amendment 3. The MG1 lane in all is $3.04, under the $15 no-ask tier.
