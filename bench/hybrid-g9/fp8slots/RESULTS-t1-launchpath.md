# RESULTS — T1: REFUTED. Compile-with-breaks loses; the break pattern is the finding

Scored under [PREREG-t1-launchpath.md](PREREG-t1-launchpath.md);
receipts in [receipts-t1/](receipts-t1/). Box: EPYC 9755 + RTX 5090.
The reduce-overhead arm crashed as the prereg anticipated cudagraphs
might ("accessing tensor output of CUDAGraphs overwritten by a
subsequent run" — the residual stream crossing a graph break holds a
replay-reused output); the registered fallback (`--compile-mode
default`) completed the A/B/A and is what the verdict scores.
Cycle ≈ $0.75, box destroyed, zero instances.

## Verdict

| arm | step ms | attn_host | other-submission | tokens |
|---|---|---|---|---|
| A1 eager | 140.7 | 44.1 | 81.1 | — |
| B compiled (default) | **146.1** | 44.5 | **86.1** | identical |
| A2 eager | 140.5 | 43.7 | 81.1 | identical |

* **B0 PASS** — greedy continuations token-identical across all arms:
  compile changed no behavior.
* **B1 FAIL** — other-submission 86.1 vs ≤ 25: compile made the bucket
  WORSE by 5 ms.
* **B2 FAIL** — step 146.1 vs ≤ 85; the compiled arm is 5.5 ms slower
  than eager (A/A spread 0.13 ms — unambiguous).

**REFUTED.** Per the prereg, T1's compile route closes.

## Why, precisely

The dense layer body is bracketed by two dynamo-disabled dynamic
regions (paged attention, MoE tier), so every one of 48 layers carries
≥ 2 graph breaks — ~100 compiled-region entries per step. In default
mode each entry pays guard evaluation and tensor boxing; at ~50 µs a
crossing that is ~5 ms of NEW overhead, and the inductor fusion of
tiny decode-shape kernels (norms, 2048-wide projections) saves less
than that. Compile-with-many-breaks is a net loss at this shape; the
mode that eliminates per-entry cost (cudagraphs) is exactly the one
the break pattern crashed.

## Successor routes (each its own registration; neither is opened here)

1. **T1b — reduce-overhead + `torch.compiler.cudagraph_mark_step_begin()`**
   at the top of each decode step: the documented remedy for the
   observed crash, one driver line, keeps the cudagraph win on the
   table.
2. **T1c — manual op-count reduction** in the layer body (fuse norms,
   batch qkv, precompute rotary): no compiler in the loop, smaller
   ceiling, no break tax.

Host note: this 9755's eager baseline runs 140.6 ms vs the 9655's
131.3 (receipts-kvappend) — host-class drift, recorded; all bars here
scored within-box.
