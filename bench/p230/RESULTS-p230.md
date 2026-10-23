# P230 results: grouped-nf4-gemm's PDL, as shipped, speeds the default NF4 server at one request (KEEP)

**Registration:** [PREREG-p230.md](PREREG-p230.md) (e4b#2042), read once, as registered. Lever D of the decode roadmap.
**Reading:** `p230-5090-r1`, one RTX 5090, e4b `b1ba9408`, grouped-nf4-gemm v0.50.0, $0.472.
Records: [receipts/p230-5090-r1/](receipts/p230-5090-r1/README.md).

## The question

grouped-nf4-gemm's programmatic dependent launch (PDL) ships on, capped to launches of at most 8 activation rows (P113's
CAP_DEFAULT, read on SC1's int4 configuration). Since K33, the single-row NF4 decode GEMV, took the switch, the DEFAULT
NF4 server launches it too. Does it decode that server's tokens exactly, and faster, at one request?

## The answer: KEEP

| step | result |
|---|---|
| VOID | no. Engagement:<br>- every arm captured buckets 1 to 32 as graphs on the same resolved server (max_seqs 32);<br>- the switch was read as registered (`pdl_active` off in OFF and on in DEF; `pdl_cap` 8);<br>- every build launched 8,759 switched kernels: OFF none with PDL, DEF 4,513, the cap splitting the build;<br>- K33 carried PDL (478 of 480 launches in DEF);<br>- per bucket, read from the capture: DEF bucket 1 had 1,152 of 1,155 with PDL, and buckets 16 and 32 none |
| FUNCTION_FAIL | no. **Every arm emits OFF1's tokens on every row** of both workloads at both lengths, and every timed rep digests the same |
| NOISY | no. Self-pairs 0.99744 to 1.00019; the W16 control 0.99732 and 0.99959; all inside [0.995, 1.005] |
| **KEEP** | **yes**: the W1 DEF/OFF pairs read **1.00564** and **1.00526**, so g1 = **1.00526** ≥ 1.005 |

At one request the decode step went from 3.707 ms (OFF1 3.7075, OFF2 3.7068) to 3.687 ms (DEF1 3.6867, DEF2 3.6874).
At sixteen requests the cap launches nothing with PDL, and the step was 11.04 to 11.07 ms in every arm.

**KEEP at the bar's edge.** g1 clears 1.005 by 0.00026. The gain at one request is about 0.5 %, and it is only just
resolvable on this instrument: the larger W1 pair, 1.00564, is above the bar too, and both pairs agree to 0.0004. Both
W16 control pairs sit below 1 (0.99732 and 0.99959), inside the band but on its low side. That is no sign that PDL
costs sixteen requests anything beyond noise: no bucket-16 launch carries PDL.

**Consequence, as registered:**
- The shipped default stands on the NF4 server.
- The register row `e4b.serve.p230.gnf4-pdl.qwen3-nf4.5090.2026-10-23` carries g1.
- The families inventory notes the read.

The lane ends at its first answer.

## Predictions

| # | prediction | held? |
|---|---|---|
| Q1 | every arm's engagement holds, including DEF's split and at least one K33 launch with PDL | **yes** |
| Q2 | every arm's tokens equal OFF1's on every row | **yes** |
| Q3 | the self-pairs and the W16 control read inside [0.995, 1.005] | **yes** |
| Q4 | g1 lies between 1.000 and 1.015 | **yes** (1.00526) |
| Q5 | NULL (about 50 %), KEEP (about 45 %) or SLOWER (about 5 %) | KEEP |

## Two corrections to the registration's census (reported, not verdict inputs)

The registration's "Why" was written from the code. The per-bucket and per-kernel records read on the box differ from
it in two places. Neither touches the rule, which reads only W1 (bucket 1) and the W16 control (bucket 16), and both
were read as the rule requires.
- **More switched kernels reach the default server than the census counted.** The build launched seven:
  `_gemv_nf4_bw` (K33), `_swiglu_rows`, `_combine_rows`, `_rmsnorm_rows`, `_rmsnorm_resid_rows`, `_rope_norm_qk` and
  `_router_epilogue`. The census named only the first three.
- **Buckets 2 to 8 also carry PDL** (723 of 867 launches each). The registration said two or more requests are above
  the cap. That holds for K33, which keys the cap on its (token, slot) rows, but not for the kernels that key it on
  tokens (`_combine_rows`, for one). So the switch also acts at 2 to 8 requests. This lane did not read those buckets.
- **Launches outside any decode capture carry PDL too.** The build launched 3,269 switched kernels outside every bucket's
  capture (the prefill and warm-up rows), 1,192 of them with PDL in DEF. Decode-only timing excludes them: each pass's
  time is its wall less its largest ttft.

## Run ids

`p230-5090-1` was refused at create at $0: its pinned offer vanished between the screen and the create. Under bench/common
rule 3 that refusal uses an id and counts toward nothing, so the reading took the next label, `r1`. The registration's
sentence "a pre-create refusal (no instance) uses no id" contradicted rule 3, which governs; this PR corrects it.

## What this does not say

- Nothing at 2 to 15 requests, where the per-bucket record shows the switch also acts (buckets 2 to 8).
- Nothing for other families, cards or stores. This is Qwen3-30B-A3B's NF4 store on one RTX 5090, and SC1's int4
  configuration is P113's.
- Nothing about uncapped PDL on this server.
