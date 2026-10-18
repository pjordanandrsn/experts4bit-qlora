# P145 — where Mixtral-8x7B's one-request step goes after K33, kernel by kernel (an instrument)

**Status:** registered before any data. This is an instrument registration: it decides which Mixtral lever comes next,
and it changes no default, no gate and no speed claim. Lane number assigned by the maintainer (P144 is Mixtral
layer-major, expert tiling).

## Why

P141 (#1754, run `p141-5090-2`) put Mixtral's expert GEMVs on K33 at plans chosen on the card. One request now steps in
8.11 ms, down from 19.53. P136's class census of the old step (moe_expert 15.27 of 19.98 ms) no longer ranks anything.
Mixtral's attention projections and head are bf16 (about 2.7 GB a step). The K33 down plan splits K and adds a sum and
a copy per call. The attention decode is untouched. P142's per-name census (`bench/pername/`) showed on Qwen3 that
class sums double-count overlap, so the next lever needs each kernel's removable time, not its class's.

## The instrument

The per-name census (`bench/pername/PREREG-pername.md`) by counted substitutions:
- **Family:** Mixtral-8x7B-Instruct @ `eba92302` (FAM Amendment 5's pin), 32 MoE layers, at **W1 and W16**. P136's
  reducer already registers Mixtral at (1, 16).
- **Server:** the shipped default at P136's Mixtral size (16 slots × 768 tokens), graphs on, every knob unset.
- **Pins:**
  - e4b at this registration's merge commit;
  - grouped-nf4-gemm **v0.47.0** at `eea5df6baefbe38b795621286fe70eac2980aa44` (the release carrying #556; its kernel is #556's merge, `8060ed74`, byte for byte), so `GNF4_GEMV_BW=auto` serves Mixtral's expert shapes on K33
    at P141's plans with no knob set. P136's reducer refuses set knobs, so the subject is the default after #556. The
    tripwire asserts both shapes in `_BW_SHAPES` and both plans in `_BW_PLANS`;
  - transformers 5.17.0, torch 2.8.0+cu128, nsys 2025.6.1.190.
- **Unchanged from P136 and P142:** host refusals, tripwire, self-tests, P136's 19 premise tests, two unprofiled speed
  arms S1a / S1b (the self-pair must sit in [0.97, 1.03]), graph-mode and node-mode captures of 64 steps after 32
  skipped at each batch, SC1b's `arm` with P142's class map, and `pername_table.py` at its pinned bytes on each node
  export.

## What each line decides (`p145_reduce.py`, fixed before data)

- **VOID:** anything P136's reducer VOIDs for Mixtral; either batch's table missing, not closing every replay exactly,
  or under 56 kept steps; or, at W1, the expert path not at **32 launches per step** for each of gate_up, `_swiglu_rows`
  and down.
- **Expert GEMVs (W1):** each GEMV's removable time minus P141's isolated time for the same 32 launches (phase 1, the
  selected plans: gate_up **3.0862 ms**, down **1.9502 ms**) is its in-situ excess, reported.
- **Ranking:** every W1 line by addressable time = removable − its floor (P141's isolated time for the two GEMVs, 0
  otherwise). Nothing is closed for Mixtral; dense_gemm is ranked.
- **Next:** the top W1 line is **proposed** with the table; a lever is a new registration, never automatic. If dense_gemm
  is on top, the proposal is a **quality premise** for compressing those projections, not a kernel (the maintainer's
  rule of 2026-10-18; Qwen3's W8A16 attention was parked on its quality bound). W16's ranking is reported, not proposed.

Profiled node times are never quoted as speed. Shares are over each batch's node replay period P.

**Predictions (graded in the read, never ruling):**
- W1's P is 8.0–8.8 ms;
- dense_gemm removable is 1.6–2.2 ms, the top line;
- the two expert GEMVs' in-situ excess is 0.0–0.6 ms together;
- the K33 split-K tail (sum + copy) is 0.05–0.25 ms;
- attention decode is 0.3–0.7 ms.

## The box

- **Run ids:** `p145-5090-1`, with `-2` as a spare for a refusal before create or a host fault only. A READ or a VOID
  after a complete capture is the result.
- **Host:** one verified-secure RTX 5090 (sm_120, ≥ 30000 MiB, ≥ 60 GiB RAM, ≥ 220 GB disk, inet_down ≥ 800), direct
  SSH, at most **$0.85/h** including the disk, under the market pin rule.
- **Guard 1.25 h.** P136's Mixtral box, which ran these same phases, took 32 minutes. The time-left checks are fetch 3000
  s, bake 1500, arm 600 and capture 600, each plus 600 s of margin. Download 100 GB at **≤ $0.011/GB**.
- **Worst case:** 1.25 h × $0.85 + 100 × $0.011 = **$2.16**; expected about $0.80. Under the lane's standing authority:
  one bus line at create, CREATED or REFUSED when the launcher returns, then the result.
- **Receipts:** the reduced records, both tables, the logs and the verdict travel. The checkpoint, arena and raw Nsight
  exports stay on the box (P136's driver).
