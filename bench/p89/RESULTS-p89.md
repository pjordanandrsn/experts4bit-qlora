# P89 — results: **LICENSED** (under amendment 1). K23's lean glue takes Qwen3-30B-A3B's B=16 int4 decode step from 10.321 to 9.805 ms on an RTX 5090 (×0.950), with every generated token identical

Registration: `bench/p89/PREREG-p89.md` (#815, `8a85200`), amendment 1 (#818, `2a623fd`). Issue: #564. The route under
test is `E4B_INT4_LEAN_GLUE=1` (#814) over grouped-nf4-gemm K23 (#427, `3990dbc`).

**Verdict by `p89_reduce.py` on `p89-5090-4`: LICENSED.**

```
P89_VERDICT LICENSED B=16 9.805 vs 10.321 ms (x0.950); tokens identical in both draw pairs; 336 fewer launches per step
```

| | OFF (draw 1 / 2) | ON (draw 1 / 2) | ratio (medians) |
|---|---:|---:|---:|
| B=16 step, ms | 10.32 / 10.32 | 9.81 / 9.81 | **0.950** (bar 0.97) |
| kernel launches per step (census) | 1,652.4 | 1,316.4 | −336, exactly 7 per layer |
| generated tokens, 16 rows × 134 | | | **identical**, both draw pairs |

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---:|
| `p89-prove-1` | HARNESS_ERROR: the guard did not arm (Vast API HTTP 429) | $0.0005 |
| `p89-prove-2` | NOT_RUN at the pre-flight: download bandwidth 25.4 MB/s < 40 on machine 147733 | $0.0165 |
| `p89-prove-3` | NOT_RUN: ssh did not authenticate on machine 147733 (its exclusion evidence from then on) | $0.0326 |
| `p89-prove-4` | **PROVED** on sm_120: premise 6 passed, K19 + K16 contracts 31 passed and K23's builder 17 passed, compiled on the card | $0.0607 |
| `p89-5090-3` | **VOID** on the registered engagement clause (below); recorded descriptively | $0.1703 |
| `p89-5090-4` | **the reading: LICENSED** under amendment 1 | $0.2029 |
| **total** | | **$0.4835** of the $2.00 ceiling |

Every teardown is proven (`vast-destroy`, HTTP 200, instance absent).

## The VOID read and amendment 1

`p89-5090-3` read VOID on the clause "at least 48 fewer `indexSelect` launches per step". That clause named the
token-row expansion's kernel **by inference**: P88's census had a 9.6 µs `indexSelect` at 48 calls per step, and I
took it to be the expansion. On torch 2.8.0 the expansion dispatches as `vectorized_gather_kernel<16, long>`. The
`indexSelect` rows are other sites, present in both arms.

The route itself had engaged fully. ON removed exactly the seven launches per layer K23 targets:

| kernel (census name, truncated) | Δ calls per step | which launch |
|---|---:|---|
| `vectorized_gather_kernel<16, long>` | −48 | the `[T × top_k, H]` expansion |
| `vectorized_elementwise_kernel<4, …>` | −144 | the tile table's three zero-fills |
| `unrolled_elementwise_kernel<…>` | −48 | the ids' int32 cast |
| `_scatter_gather_elementwise_kernel<…>`, `index_elementwise_kernel<128, 4, …>` | −48 each | the sorted-id gather and the `index_copy_` unsort |

The per-launch mapping is read from the counts. Only the total is gated.

- **Amendment 1** (#818) removed the name-based clause and raised the total-launch floor from 192 to 288 per step
  (6 per layer). Kernel names are now reported, not gated.
- Run 3 was not re-read under it. It read B=16 10.422 → 9.913 ms/step (×0.951), with tokens identical, descriptively.
- The verdict is run 4's, a fresh reading at the amendment's merge.

## The reading (`p89-5090-4`)

- **Host.** One RTX 5090 (driver 580.119.02, power.limit 475 W) on an AMD EPYC 9655 (machine 150710), image
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. Software: e4b 0.37.8 at `2a623fd`, grouped-nf4-gemm 0.33.8 at `3990dbc`,
  torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1.
- **Timeline (UTC).** Rented 13:46:53. Premise passed 13:51. Arms ran 14:03–14:07. Destroyed 14:07:25.
- **Premise on the card.** `tests/test_k19_row_exact_gpu.py` + `tests/test_k23_lean_glue_gpu.py`, 6 passed, none
  skipped:
  - K19's rows are row-count invariant;
  - the lean route is bit-equal to the default at B=16, eager and captured, from expanded rows and from token rows.
- **Engagement.** K19 ran 96 times per step and the tile builder 48 in both arms. ON launched 1,316.4 kernels per step
  against OFF's 1,652.4.
- **Token equality.** Each arm's timed JSON carries every row's prefill, warm and timed greedy tokens (134 per row).
  All 16 rows are identical OFF vs ON, in both draw pairs.
- **Draw spread.** Each arm's two draws agree to 0.01 ms.

## Against the predictions

| prediction | outcome |
|---|---|
| B=16 ON/OFF 0.91–0.95, LICENSED | **0.950**, at the edge of the band, LICENSED |
| about 288 fewer launches per step; `index_select` −48 | **336** (7 per layer, one more than predicted); `indexSelect` 0, because the expansion is a different kernel (amendment 1) |
| tokens identical in all 16 rows | **held** |
| OFF reproduces P88's ON step (10.14 ms) | 10.32 (run 3's box: 10.42). These are different boxes; run 4's card was capped at 475 W, P88's ran at 600 W. Not gated |

The saving is 0.516 ms per step. The builder kernel itself (0.43 ms per step in P88's census) remains: the lean path
removes the launches around it, not its own work.

## What follows

1. **A default.** `E4B_INT4_LEAN_GLUE` becomes `auto` on K19's rows: the lean path when the kernel package carries
   K23's options, and the separate launches when it predates them. `1` still requires K23, and `0` is the old path.
   It goes in its own PR citing this read.
2. **The next Qwen3 B=16 levers** (P86's map against vLLM 0.30.0):
   - the tile builder itself (0.43 ms per step);
   - the shared index kernels both routes pay. Their call sites are to be confirmed from a same-lane census diff or
     step_decomp's call-site table, not inferred.

## Receipts

[`receipts/p89-5090-4/`](receipts/p89-5090-4/) holds the four timed JSONs, both censuses, the premise log, verdict,
summary, forensics, versions and the teardown proof. [`receipts/p89-5090-3/`](receipts/p89-5090-3/) has the same set
for the VOID run. Each directory has a `SHA256SUMS`. The receipt store holds every run's `receipt.json`.
