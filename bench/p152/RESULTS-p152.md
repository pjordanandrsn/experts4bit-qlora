# P152 results: where Qwen3-30B-A3B's 16-request served step goes, kernel by kernel (e4b#1313, stream (B))

**Status:** closed. **READ** on the one census. This is an instrument: it changes no default, no gate and no speed
claim. The W16 expert byte floor reads **UNREAD**: the registration's route premise (16-row expert calls through K19
over an int4 store) was wrong for this checkpoint, whose served experts are NF4.

[The registration](PREREG-p152.md) ran P145's per-name census by substitution on Qwen3-30B-A3B at W1 and W16, on the
shipped default server. Every share divides by the **unprofiled** step from the same box's two speed arms. An eager
routing pass over the same W16 window recorded which experts each layer touched, for an expert byte floor.

| run | what | cost | outcome |
|---|---|---|---|
| [`p152-5090-1`](receipts/p152-5090-1/README.md) | the card read 475 W, under the 575 W floor; refused before any install | $0.015 | rc 17, not a strike |
| [`p152-5090-2`](receipts/p152-5090-2/README.md) | two speed arms, four captures (W1 and W16, graph and node), the routing pass | $0.506 | **READ** |

Total $0.521 on two RTX 5090 launches, inside the registered worst case of $1.98. The public receipt copies are under
[`receipts/`](receipts/), each with a README and `SHA256SUMS`. The launchers' receipts and ledger rows are in the private
receipt store. The maintainer re-derived `verdict_p152.json` byte for byte from the store: the reducer and the routing
script are byte-identical to the registered ones at `334ef5b5`, and the reducer's self-test passed (20/20).

## The read (`p152-5090-2`)

- **The box:** one RTX 5090 at a 575 W power limit read on the box, e4b `334ef5b5`, grouped-nf4-gemm v0.47.1
  `38b30888`. P136's 19 premise tests passed.
- **The denominators** (unprofiled, the mean of S1a and S1b): W1 **3.891 ms**, W16 **16.4969 ms**. Self-pairs W1
  1.0001, W16 1.0013.
- **The census:** both tables kept 61 steps and closed every replay exactly. Profiled periods (context only): W1 4.804
  ms, W16 17.528 ms.

### W16, the subject

| class | ms | share of 16.497 |
|---|---|---|
| moe_expert | 11.773 | 71.4 % |
| dense_gemm | 2.406 | 14.6 % |
| attn | 0.830 | 5.0 % |
| moe_route | 0.663 | 4.0 % |
| norm_elem | 0.463 | 2.8 % |

The five named candidates, by removable time:

| | line | launches/step | removable ms | share |
|---|---|---|---|---|
| 1 | `_gemm_nf4_grouped_smallm` 136x48 (gate_up) | 48 | 8.3951 | 50.89 % |
| 2 | `_gemm_nf4_grouped_smallm` 136x64 (down) | 48 | 3.3530 | 20.32 % |
| 3 | `Kernel2` 8x16 (dense) | 48 | 1.1111 | 6.73 % |
| 4 | `Kernel2` 8x40 (dense) | 48 | 0.7462 | 4.52 % |
| 5 | `_fp8_paged_decode_split_f8dot` 320 CTAs (attention) | 48 | 0.6586 | 3.99 % |

Next below the five is the one-program tile builder `_tile_table_r1` (moe_route, 0.4539 ms, 2.75 %). The head
(`Kernel2` 8x1187, one launch a step, 0.3774 ms) is stream (A)'s, P150's int8 head, and is not named.

### W1, the re-read

- **dense_gemm is closed** (stream (A): D1 and D2): 1.815 ms, 46.6 % of 3.891.
- **The top line not owned by stream (A)** is the attention decode, `_fp8_paged_decode_split_f8dot` (64 CTAs): 0.3952 ms,
  10.2 %.
- **The expert GEMVs sit at their isolated speed.** Ranked on removable time minus K33's isolated time, gate_up reads
  0.5896 against 0.6068 (nothing addressable) and down 0.3625 against 0.3540 (0.0085 ms). In situ, the NF4 GEMVs run as
  fast as K33 runs alone.
- **The expert byte floor at W1** (reporting only): 8 experts x 48 layers at 0.5625 B per weight is 1.019 GB a step, a
  floor of 0.680 ms against the moe_expert class's 0.983 ms removable, so 0.303 ms (7.8 % of the step). That is P142's
  held candidate (b), read again on another box.

## The routing pass and the W16 floor

**The route premise was wrong.** The registration, and the route check its review asked for, assumed that a 16-row
expert call goes through K19 over a uniform int4 store, as the replay's device-grouped route would. The served expert
store for this checkpoint is NF4: the W16 census names the grouped NF4 small-M GEMM, and the W1 census names the NF4
GEMV `_gemv_nf4_bw`. P142's W1 census had already named that GEMV, so the evidence was in hand before the
registration; neither the registration nor its review caught it.

The route check did its job. `route_b16.json` records 0 uniform-int4 expert stores on 48 MoE layers, and 16 rows
grouping as (singleton, device) = (False, False). With graphs off and no int4 store, the eager pass's 16-row expert
calls ran host-grouped, not on the replay's device-grouped GEMM. Both are recorded as faults, and the W16 floor reads
**UNREAD**, as the registration's rule says. The census stands.

The pass's routing statistics are reported as recorded, from the eager window's traffic:
- a mean of **57.02** unique experts per layer-step (40 to 81), against 82.42 under uniform routing;
- 2736.8 unique experts a step;
- a hottest expert holding 11.0 % of a layer's hits.

**Context, not a reading.** This arithmetic is outside the registered rule. It assumes the replay routed like the eager
window and that the served NF4 store holds 0.5625 B per weight; that byte count was not read on the served store. On
those assumptions, gate_up's bytes over P141's copy rate come to about 3.23 ms (the GEMM at about 38 % of copy), and
down's to about 1.62 ms (about 48 %). An NF4 W16 floor read on the served route would be its own registration.

## Predictions (graded, never ruling)

| prediction | read | grade |
|---|---|---|
| W16 unprofiled in 15.5-17.0 ms | 16.497 | HELD |
| W1 unprofiled in 3.85-3.98 ms | 3.891 | HELD |
| moe_expert >= 60 % of W16, the grouped small-M GEMM on top | 71.4 %, `_gemm_nf4_grouped_smallm` | HELD |
| 55-85 unique experts per layer-step | 57.02 (from the faulted routing pass) | HELD, reported |
| W16 expert floor 4.5-7.5 ms with >= 3 ms headroom | | UNREAD |
| W1's top non-(A) line the attention decode at 0.45-0.65 ms | the attention decode, 0.395 ms | MISSED (low) |
| both self-pairs in [0.99, 1.01] | W1 1.0001, W16 1.0013 | HELD |

## What it means for stream (B)

- **The W16 step is the expert GEMM.** Two grouped NF4 small-M GEMM launches per layer take 71.2 % of the unprofiled
  W16 step (11.75 ms). P142's held candidate (b), K33's W1 GEMV, is about 0.3 ms of a 3.9 ms step, and the W1 re-read
  shows the GEMVs already run at their isolated speed in situ. For serving at 16 requests, the grouped GEMM is the lever.
- **Next, if wanted:** an NF4 W16 floor, registered on the served route. It would read the store's bytes per expert on
  the box and the routing on the replay's own route, to price the GEMM's headroom before any kernel work.
- **Further down:** the 16-row dense projections (`Kernel2` 8x16 and 8x40, 11.3 % together) and the tile builder
  (2.75 %).

## What this lane cannot say

- Nothing about speed: it names candidates; a lever's gain is its own served read.
- Nothing about the W16 expert byte floor: it reads UNREAD, and the arithmetic above is context only.
- Nothing about other cards or families: one RTX 5090, Qwen3-30B-A3B.
- Nothing about loads other than 16 rows in steady decode (W16) and one row (W1): no prefill, no mixed steps.
- Not that the routing pass saw the replay's experts step for step: it ran eager, its expert calls took another route,
  and the census driver does not record tokens.

## Reproduce

From a checkout of e4b at `334ef5b5`, with the files of `receipts/p152-5090-2/` in a directory `DIR` (the reducer reads
the arms, the census records, the per-name tables and the routing record):

```
export PYTHONPATH=bench/p136:bench/pername:bench/sc1b   # the reducer's staged siblings, as the box stages them
python bench/p152/p152_reduce.py --self-test
python bench/p152/p152_reduce.py --dir DIR --out verdict_p152.json --e4b-sha 334ef5b50f8e0fa902f543777fd44f7ab4636769
```

The output equals `receipts/p152-5090-2/verdict_p152.json` byte for byte (on Windows, after stripping the carriage
returns its text-mode write adds).
