# The per-name served census: where the B=1 step's moe_expert time goes, kernel by kernel (e4b#1313)

**Status:** registered, not run. An instrument registration: it decides which lever comes next (P2), and it changes no
default, no gate and no speed claim. The lane number is assigned at merge.

## Why, from data already seen

P138's n=1 census of the default (`bench/vsplit/receipts/vsplit-5090-3/census_n1.json`, e4b `88ae1cfd`, gnf4 `724ccc45`)
put the B=1 step at P = 4.016 ms: dense_gemm 1.809, moe_expert 1.349, attn 0.698, moe_route 0.511, norm_elem 0.200,
idle_out 0.140, overlap_in −0.717 (same-stream overlap pairs 0.42). dense_gemm is closed for B=1 (2026-10-18). The
moe_expert class reads 1.019 GB per step (8 experts × 4,718,592 weights × 0.5625 B × 48 layers): 756 GB/s, 48 % of the
5090's STREAM triad. K33 (`gnf4.kernel.k33-nf4-decode-gemv-bw.5090.2026-10-07`) timed the same two expert GEMVs alone:
gate_up 0.6068 ms and down 0.3540 ms per 48 launches (copy floor 0.4486 / 0.2243). So 1.349 − 0.961 = **0.388 ms, 9.7 %
of the step**, is served-vs-isolated time that no receipt splits. It holds the 48 `_swiglu_rows` launches, the wait that
programmatic dependent launch bills to a kernel started early, and any in-situ cost. Class sums cannot split it. Per-kernel
times stay on the box.

## The instrument

P136's box (`bench/p136/`) at its pinned bytes, on one family: **Qwen/Qwen3-30B-A3B @ `ad44e777`**, 48 MoE layers, the
shipped default (every lever unset, 16 slots, graphs on), e4b at this registration's merge commit, grouped-nf4-gemm
v0.46.1 @ `70bc03cb` (P136's pin), transformers 5.17.0, torch 2.8.0+cu128, nsys 2025.6.1.190. Unchanged from P136: host
refusals, tripwire, self-tests, P136's 19 premise tests, two unprofiled speed arms S1a/S1b (the self-pair must sit in
[0.97, 1.03]), and a **B=1** graph-mode and node-mode capture of 64 steps after 32 skipped. SC1b's `arm` reduces them
with this lane's map, `kernel_classes_pername.json`: P136's map plus one name, `_rope_norm_qk` in norm_elem (Qwen3's
q/k norm + rope launch, as P132's v1.2 maps it). The tripwire also asserts the subject: the bandwidth GEMV `auto` at
Qwen3's two shapes, and every fusion knob resolving to `auto` (default-allowlisted) on qwen3_moe.

**The one new step.** `pername_table.py` reads the node-mode export on the box. It keeps the replays SC1b keeps, with
the same slicing and `_filter`. It walks each replay once in start order with a running frontier, and splits every
in-graph item into `gap_before` (idle before it starts), `excl` (its own advance of the frontier) and `billed` (the
rest of its duration, run under its predecessor). `gap_before + excl` summed over a replay equals its span **exactly**,
in integer nanoseconds. The table is keyed by kernel name and grid. The two expert GEMVs carry their roles (`gate_up`
just before `_swiglu_rows`, `down` just after). An item's **removable** time is `gap_before + excl`: an upper bound on
what removing it saves.

## What each line decides (`pername_reduce.py`, fixed before data)

- **VOID:** anything P136's reducer VOIDs, imported unchanged with Qwen3 registered as a W1-only family. That covers
  pins, self-pair, window, SC1b's blocking labels incl. NODE_TRACE_INFLATED, and class closure. Also VOID: the table
  missing, a replay not closing exactly, fewer than 56 kept steps, or the expert path's three lines (gate_up, swiglu,
  down) not at 48 launches per step.
- **swiglu:** P2 = the swiglu-into-down fusion read (P0's gnf4 option) **iff** `_swiglu_rows`' removable time is
  ≥ **3 %** of the node replay's period P.
- **Expert GEMVs:** each GEMV's removable time minus K33's isolated time is its in-situ excess. It is reported, and it
  is the GEMVs' addressable time in the ranking.
- **Ranking:** every line by addressable time (removable time minus its floor: K33 for the two GEMVs, 0 otherwise). It
  covers attn (`_fp8_append_bt1_side`, `_fp8_paged_decode_split_f8dot`), moe_route (`_router_epilogue`,
  `_combine_rows`), norm. dense_gemm is listed, never ranked.
- **P2 below 3 %:** the top-ranked line is *proposed* with this table. A retarget is a new registration, never automatic.

Profiled node times are never quoted as speed. Shares are over the node replay's own P. The per-name table goes to
the families lane too: their Mixtral K33 lane sits on the same moe_expert path.

**Predictions (graded in the read, never ruling):** swiglu removable 0.08–0.15 ms (2–4 % of P); the two GEMVs' in-situ
excess 0.10–0.30 ms together; most of overlap_in billed to the GEMVs and the attention append.

## The box

- **Run ids:** `pername-5090-1`, with `-2` as a pre-create spare. `-2` runs only if `-1` is refused before create or
  stops before its node capture. A READ or a VOID after a complete capture is the result. No other rerun.
- **Host:** one verified-secure RTX 5090 (sm_120, ≥ 30000 MiB, ≥ 60 GiB RAM, ≥ 150 GB disk), direct SSH, at most
  **$0.80/h** under the market pin rule.
- **Guard and downloads:** guard **0.8 h**, sized from P136 Mixtral's phases scaled to 61 GB: about 21 minutes of box
  time. The time-left checks are fetch 1200 s, bake 600, arm 420 and capture 240, each plus P136's 600 s margin.
  Download 70 GB at **≤ $0.005/GB**.
- **Worst case:** 0.8 h × $0.80 + 70 × $0.005 = **$0.99**. Under the lane's standing authority: one bus line at create,
  then the result.
- **Receipts:** the reduced records, the table, the logs and the verdict travel. The checkpoint, arena and raw Nsight
  exports stay on the box (P136's driver).
