# P139 — does K33's bandwidth GEMV, at plans chosen for Mixtral's expert shapes, decode the default `serve_paged` server faster at one request on Mixtral-8x7B on an RTX 5090?

**Registered 2026-10-18, before any data.** Harness: `p139_run.sh` (box), `p139_drive.sh` (controller), `p139_select.py`
(phase 1), `p139_box.py` and `p139_reduce.py`, plus P133's dry fixture `p133_tiny.py`, all at `staged.sha256`. Derived
from P134 (`bench/p134/PREREG-p134.md`), changed only where this page says so.

## Why

P136's kernel-class census of Mixtral at one request (`p136-mixtral-5090-1`, the default graph server) puts the expert
GEMVs at **15.27 of 19.98 ms** per step (76 %); at 16 requests, 44.6 of 50.8 ms. At one request that is about 80 µs per
14336 × 4096 NF4 matrix, about 0.37 TB/s on a card that streams 1.5 TB/s. Mixtral's step has headroom there that no other
class does.

grouped-nf4-gemm's K33 (`GNF4_GEMV_BW`) is the bandwidth-targeted single-row NF4 GEMV. It has plans only for Qwen3-30B's,
Granite-3.1's and OLMoE's shapes, and `auto` engages only where a served read licensed it. At those shapes it runs the
expert GEMV 2.9–4.3× faster per call than the scalar route, and served reads licensed it at 1.24× (P116), 1.40× and
1.50× (P134) per step. Mixtral's two shapes have no plan, and at them every decode runs the scalar GEMV.

## The premise, read at $0 before this registration (`premise/`)

- **The call site** (an A2000 trace of a real-width two-layer Mixtral through the GPU serve smoke's own build, e4b
  `9890a502`): e4b's served decode hands K33's gate exactly **(28672, 4096)** (gate_up, one token row read for both
  routed slots, `gather_div=2`) and **(4096, 14336)** (down, two rows), at T = 2 rows per call. `premise/trace.json`.
- **The numerics** (the A2000, sm_86, grouped-nf4-gemm 0.46.1): at both shapes, **all twelve** of K33's plans meet K33's
  own contract: `bw_prmt32` bitwise `bw_tree`, both within the fp32 reference's tolerance (error ≤ 1.05× the scalar
  route's, or ≤ 2^-9 of the output scale). The served token-row form is bitwise its expanded call. K33 is not bitwise
  the scalar route (so the read needs a quality bar). `premise/numerics.json`.

## The subject and the pins

The default `serve_paged` server at **P136's Mixtral size**, 16 slots × 768 tokens (`E4B_PAGED_MAX_SEQS=16`,
`E4B_PAGED_MAX_TOKENS_PER_SEQ=768`), in every phase. The unset 4096-token default does not fit Mixtral's NF4 weights
beside its KV on 32 GB; P136's Mixtral arms ran at this size (peak 29.97 GB allocated). Graphs on, buckets auto,
all-vram, the fusion knobs at their default (Mixtral's folds on since #1743), `GNF4_PDL` at its default. Pins, each
refused if different:
- e4b: **the launch commit**, main after this registration merges (P135's rule: the driver ships its clean checkout's
  HEAD, and the manifest's `heads.e4b` is that commit);
- grouped-nf4-gemm **0.46.1** at `70bc03cb7a61b03c4e801e7423c58c103a1b6d7d` (K33's kernel and its contract test are
  0.46.0's bytes; P136 ran this pin);
- torch 2.8.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2;
- `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61` (FAM Amendment 5's pin), fetched
  without `consolidated*`; the NF4 arena baked on the box by P39's `k8_bake.py`.

**Card:** RTX 5090 class, compute capability 12.0 and ≥ 160 SMs, checked before any install (rc 15).

## Phase 1: the plan for each shape (`p139_select.py`), on the card, before the fetch

No model is loaded. Per projection, 32 layers of seeded synthetic NF4 stores at the served shape, each layer its own
bytes, called in the served one-request form above, under B1's own environment (`GNF4_GEMV_BW=1`, decode and PDL at
their defaults). **The rule, fixed now:**
1. every plan in K33's list of twelve (`(16, 256, 4, 1)` … `(32, 256, 4, 2)`: BLOCK_N, KC, warps, split-K) is checked on
   layer 0 **before any timing**, against K33's contract above. A plan that fails to compile or misses it is not
   selectable;
2. each selectable plan's 32-layer chain is captured in one CUDA graph; after 20 untimed rounds, the graphs replay for
   **40 rounds**, the order reversed every round (CUDA events);
3. the plan with the **lowest median** wins; a tie goes to the earlier plan in the list;
4. no selectable plan at a shape: **FUNCTION_FAIL**, and the box stops before the fetch.

The plan string goes to B1 as `GNF4_GEMV_BW_PLAN` (`"28672,4096=…;4096,14336=…"`), and every arm records it. **Reported,
never ruled and never quoted as the gain:** each plan's per-call median, the incumbent's (`GNF4_GEMV_BW` unset: the
scalar GEMV) and the copy floor. Choosing the best of twelve on the same samples biases those numbers upward; the gain is
phase 2's alone.

## Phase 2: the served read (P134's)

**Four fresh processes, ABBA (B0a B1a B1b B0b), differing only in the switch:**
- **B0:** `GNF4_GEMV_BW` unset, the shipped default. At Mixtral's shapes that is the scalar GEMV on any part: neither
  shape is in `_BW_SHAPES` or dot-pad's table, and no shipped plan exists for them, all of which the tripwire asserts.
- **B1:** `GNF4_GEMV_BW=1` with phase 1's `GNF4_GEMV_BW_PLAN` exactly; no decode override (`prmt32`).

Workloads (W1 and the control, W16 at the box's own width), engagement (B1 only `bw_prmt32`, with `bw_splitk` beside it
only under a selected split-K plan; B0 only the scalar GEMV) and the fit check before the arms are P134's. A fit failure
is VOID with no rerun (rc 31).

**The quality read:** P134's (P116's instrument at one window per pass, T == 1): wikitext and c4val1, **24 windows
each**, 512 prompt tokens and 128 positions, the server built eager at 16 x 768. The bars are P110's: bias_ON ≤ B_floor
+ 0.01 and spread_ON ≤ 2 × max(S_floor, 0.005) on both texts, and |K8 Δppl| ≤ 0.05 on wikitext.

## The rule (`p139_reduce.py --family mixtral`, self-tested on 41 cases)

P134's rungs, first that applies:
1. **FUNCTION_FAIL** if phase 1 found no contract-exact plan at a shape.
2. **VOID** if:
   - the selection record is missing, its choice is not its own rule's, or its plan string is not its choice;
   - the fit check failed (no rerun);
   - a record is missing or not ok, or ran under another e4b or grouped-nf4-gemm commit, model revision or server size;
   - a B1 record ran another plan string than phase 1's;
   - P134's speed or quality integrity fails, engagement is off, or the scale mutant passes the bar.
3. **NOISY** if a self-pair (B0b/B0a, B1b/B1a) is outside **[0.985, 1.015]** at W1 or outside **[0.96, 1.04]** at the
   control.
4. **FUNCTION_FAIL** if B0b ≠ B0a or B1b ≠ B1a on any row, workload or length, or one arm's timed reps digest
   differently.
5. **QUALITY_FAIL** if either text misses P110's bars, or wikitext's K8 does.
6. **SLOWER** if **g1** = min(B1a/B0a, B1b/B0b) at W1 is **< 1.03**, or **gW** at the control is **< 0.99**. W1 is the
   ruled read; W16 is a no-regression check, and **0.99 is the regression line, fixed now**.
7. **LICENSED** otherwise.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | W1 fits at 16 x 768 (P136's arms did, at 29.97 GB); no VOID, about 85 % |
| Q2 | phase 1: every plan meets the contract at both shapes on the 5090 (12 of 12 did on the A2000); engagement exact; B1 ≠ B0 on some W1 rows; self-pairs bitwise |
| Q3 | **g1 ∈ [1.25, 2.00]**. B0's W1 step is about 18.5 ms (P136's 19.78 with the folds off, less P135's 1.2–1.5 ms). The expert bytes' floor at W1 is 6.34 GB a step, 4.19 ms at K33's measured 1.51 TB/s; K33 reached 0.58–0.74 of its floor on the other families, so the expert GEMVs take 5.7–7.2 ms under K33 against P136's 15.27. With P134's transfer of 0.55–1.0 of a kernel saving to the served step, Δ = 4.1–9.1 ms |
| Q4 | gW ∈ [0.99, 1.03]: K33 engages only on single-row calls, and at W16 most experts see several rows |
| Q5 | quality within the bar |
| Q6 | phase 1, reported only: the selected plans' per-call medians are 0.3–0.5× the incumbent's |

## Consequence, registered now

- **LICENSED:** a grouped-nf4-gemm PR adds Mixtral's two (N, K) shapes to `_BW_SHAPES` (auto, ≥ 160 SMs) and phase 1's
  plans to `_BW_PLANS`, with its own tests and release. That release may queue behind the 0.47.0 publish. It licenses
  **shapes, not families**: any model with these shapes takes them, and under auto's ≥ 160-SM gate they also engage on
  unmeasured parts above 160 SMs (P116's precedent).
- **SLOWER:** the shapes stay out of auto.
- **QUALITY_FAIL / FUNCTION_FAIL:** they stay out; the defect is examined first.
- **NOISY or VOID:** as the maintainer rules (`p139-5090-2` is a spare only for a refusal before create or a host fault).

## What this cannot say

Batches above one row beyond the no-regression check, other cards, prefill, the int4 route, and Mixtral-8x22B (its
shapes differ).

## The premise on the card

Before anything is fetched (rc 25), read from junit against `p139_box.PREMISE` (P134's, at the same bytes), on the 5090:
- the kernel contract (grouped-nf4-gemm's `test_nf4_gemv_bw.py` at the pin): **35 passed, none skipped**;
- the served route (`test_decode_graph_buckets.py`, `test_kv_step_select.py`, `test_gemv_bw_served_gpu.py`): **18 passed, none skipped**.

## The dry run

`P139_DRY=1` (run ids `p139-dry-<n>`, $0): the driver and the whole box on one tiny family (`tiny`, built by
`p133_tiny.py`), phase 1 at its shapes, on any CUDA card. The premise there reads P134's sm_86 lists. Never a reading.

**Evidence (2026-10-18, an RTX A2000, sm_86, launch commit `007e955f`):**
- `p139-dry-1` stopped at the tripwire (rc 9): the dry box's scratch venv kept another e4b commit, which pip did not
  replace. That is a property of the reused dry box; a rented box starts without e4b. The venv's e4b was removed.
- `p139-dry-2` ran the driver, staging, every self-test and both premise suites (sm_86's lists: 33 passed with the two
  PDL skips; 13 passed with the five fp8-KV skips). Phase 1 found 12 of 12 plans selectable at the fixture's shapes and
  handed B1 its plan string (`512,512=16,512,4,1;512,256=16,256,4,1`). The fixture, bake, prompts, fit check and four
  arms followed, with engagement exact: B0 only the scalar GEMV, B1 only `bw_prmt32`. The quality phase then stopped
  at the fused fp8 KV append, which needs sm_89+, as P133's and P134's dry runs did. The reducer read VOID (records
  missing), as a dry run does there.

**First executed on the paid reading:** Mixtral's checkpoint on this harness, phase 1 at Mixtral's shapes on sm_120, the
default graph server at 16 x 768 with K33's plans, decode-graph capture, device grouping, the fp8 KV append, the
quality phase on Mixtral, and the card and configuration refusal.

## Budget, host and STOP

- **Authority:** the lane's standing authority (worst ≤ $15, adertha-agents#220), inside this registration once it
  merges. One review pass.
- **Host, by the market rule:** Vast verified-secure, reliability2 ≥ 0.98, direct SSH, one RTX 5090 at ≤ $0.85/h
  including 250 GB of disk (the box's floor is 220 GB), host RAM at the launcher's floor, download ≤ $0.011/GB, and
  inet_down ≥ 800 Mbps (the 93 GB fetch). The pin is the cheapest passing offer not pinned by another lane, stable in 6
  machine-filtered reads, held through the re-read rule, and posted on the bus first.
- **Run ids:** `p139-5090-1`, the reading; `p139-5090-2`, a spare for a refusal before create or a host fault only. The
  driver refuses any other. A box-started failure is a strike.
- **Budget:** guard 2.0 h; worst = 2.0 h × $0.85 + 100 GB × $0.011 = **$2.80**. Expected about 1.3 h, about $1.15.
- **STOP-1:** refusals before any install: CUDA unusable 18, dud box 10, torch not the pin 28, card or configuration 15,
  disk 13, host RAM 16; premise rc 25.
- **STOP-2:** every time-left check, plus 600 s, fits inside the guard (`tests/test_p139_staged_pin.py`): phase 1 900 s,
  fetch 3,000, bake 1,500, fit 600, each arm 600, quality OFF 2,400 and ON 900.
- **STOP-3:** phase 1 failing to run is rc 23; a phase-1 FUNCTION_FAIL ends the box before the fetch with its verdict.
- **STOP-4:** the driver refuses an undeclared run id, a dirty checkout, or a staged file that differs from
  `staged.sha256`.
- **STOP-5:** no quality phase after a failed speed arm.
