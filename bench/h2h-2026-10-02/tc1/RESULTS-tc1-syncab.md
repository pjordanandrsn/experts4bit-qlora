# TC1 amendment 10 (#945): e4b's grouping and index-copy syncs, A/B on one RTX 5090

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 10. Receipts:
[`receipts/tc1-5090-38/`](receipts/tc1-5090-38/). The verdicts, ratios and intervals are the reducer's
([`../../tc1/tc1_reduce.py`](../../tc1/tc1_reduce.py)), and the box's own `RESULTS-tc1.md` matches a re-reduction byte for byte.

**The box.** `tc1-5090-38`, instance 53991731, AMD EPYC 7663, driver 580.95.05. e4b ran at `4dd72be`, which carries #946, and
grouped-nf4-gemm at `b670e61`, the merge of grouped-nf4-gemm#438. It cost $0.42, and teardown is complete. The receipt reads ALARM
for one reason: the provider's instance listing answered HTTP 429 at teardown, so the launcher retried until the box was proven
gone. The workload finished, and all eight arms were fetched. An earlier draw, `tc1-5090-37`, failed pre-flight (10 MB/s
download on machine 136851) for $0.02 and ran nothing.

**What differs between the two sides.** Each arm is Qwen3-30B-A3B at the field recipe, N = 20, and the sides differ only in two
environment variables:

| side | settings | host syncs per MoE layer pass |
|---|---|---|
| legacy | `E4B_GROUPING=legacy GNF4_PINNED_RING=0` | 13 |
| new | `E4B_GROUPING=single GNF4_PINNED_RING=1` | 1 |

Each new-path arm's receipt shows the ring staging 53,680 transfers without waiting once; each legacy arm shows none.

| arm | s/step (median of steps 11..20), draws | held-out at 20 | peak VRAM | J/step (whole run) |
|---|---|---|---|---|
| shipped, legacy | 4.110 / 4.162 | 0.8131 / 0.8122 | 24.58 GB | 1,063 / 1,028 |
| shipped, new | 3.508 / 3.654 | 0.8133 / 0.8153 | 24.58 GB | 969 / 1,000 |
| matched, legacy | 5.267 / 5.213 | 0.8505 / 0.8529 | 27.82 / 27.85 GB | 1,322 / 1,314 |
| matched, new | 4.426 / 4.453 | 0.8506 / 0.8535 | 27.82 / 27.85 GB | 1,261 / 1,260 |

- **P16 (shipped) is HELD:** new / legacy = **0.866 [0.843, 0.889]**. The shipped pairs agree within 1.3 % (legacy) and 4.1 % (new).
- **P17 (matched) is HELD:** new / legacy = **0.847 [0.840, 0.854]**. The matched pairs agree within 1.0 % and 0.6 %.
- **What changes, and what doesn't.** The training step is 13-15 % faster, with the same held-out loss to within the draws' own
  noise and the same peak VRAM, and less energy per step. Both changes are value-identical by construction; the small held-out
  differences are the fused path's run-to-run nondeterminism, the same size as legacy against legacy.
- **The decision rule.** Both ratios are below 0.95, so the pinned ring becomes grouped-nf4-gemm's default outside capture.
  `GNF4_PINNED_RING=0` will turn it off.
- **What it does not change.** This is e4b against itself on one host. The cross-framework positions measured before this change
  (the TC1 matched position, the native-best rows) stand as they were measured. Any new position needs its own box, and no number
  here is divided into another box's.
- **Why it is not 0 syncs.** The one remaining sync per layer pass is the grouped kernels' contract: the launch grid is sized from
  host-side group sizes. Removing it needs a device-side grid, a device-tile dgrad and a LoRA delta that does not need the widest
  group on the host (#945).
