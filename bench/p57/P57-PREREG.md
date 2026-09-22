# P57 — K17's fused reduce read in the consumer; P54's B=16 divergence split in two; the distinct-expert count measured (registered 2026-09-21, before the run)

Owner directive (Jordan, 2026-09-21): *"main thing needed is throughput work."* Record: [#652](https://github.com/pjordanandrsn/experts4bit-qlora/issues/652). Predecessors: P54 ([`bench/p54/RESULTS-p54.md`](../p54/RESULTS-p54.md)), K17 (grouped-nf4-gemm `kernel/PREREG-k17-fused-splitk-gemv.md`, #372; kernel + pilot on its `lane/k17-fused-splitk-kernel`). P55 and P56 are other lanes' numbers (#660, #662).

## The rule applied before anything else — every arm here was dry-run against what runs it

- The timed arms are P54's `speed_arm` with per-arm environment variables the runner forwards (`GNF4_GEMV_FUSED_REDUCE`, `E4B_FUSE_T1_GLUE_R2`) and the `--fuse-qkv` flag P54 already exercised. **One defect found by reading the line before registering an arm that depends on it:** P54's `speed_arm` put the caller's `"$@"` FIRST on its `env` line, so a per-arm `E4B_FUSE_T1_GLUE_R2=0` would have been silently overridden by the fixed `=1` that followed (`env` takes the last assignment). P57's runner puts `"$@"` last. Without that fix the round-2 split below would have measured two identical arms and called them different.
- The distinct-expert arm's instrument ([`distinct_experts.py`](distinct_experts.py)) and its attachment ([`hook/usercustomize.py`](hook/usercustomize.py), P42's hook byte-for-byte plus the counter) both pass a CPU dry-run: a tiny random Qwen3-MoE, a fake `enable_hybrid_tier`, decode steps, the `atexit` dump. The dry-run caught two defects before any box: transformers 5.17's router is a `Qwen3MoeTopKRouter` module returning `(logits, scores, indices)` (the first version hooked an `nn.Linear`), and a bare `except Exception: continue` around `import_module` had swallowed a `NameError` and reported "hook point missing".
- `--amort` stays `off` everywhere: the count comes from the router hook, not from the harness's amortisation counters that P54's arm asked for and the captured B>1 stage refuses.

## Arms (one RTX 5090; Qwen3-30B-A3B @ `ad44e777`; P54's protocol — int4 experts RTN + uncalibrated int4 attention, K16 route auto, 512-token prompts, 128 generated, graph loop, 70 timed steps; grouped-nf4-gemm at the K17 cut; every timed arm drawn twice, A B A B, medians read; census on the first draw)

| arm | batch | `--fuse-qkv` | `GNF4_GEMV_FUSED_REDUCE` | `E4B_FUSE_T1_GLUE_R2` | what it is |
|---|---|---|---|---|---|
| `int4_b1` | 1 | yes | 0 | 1 | control — the B=1 stack P54 licensed (fused qkv, token-identical) |
| `int4_b1_fr` | 1 | yes | **1** | 1 | K17's P4 at B=1 |
| `int4_b16` | 16 | no | 0 | 1 | control — P54's B=16 control |
| `int4_b16_fr` | 16 | no | **1** | 1 | K17's P4 at B=16 |
| `int4_b16_nor2` | 16 | no | 0 | **0** | P5 split, control leg: round-2 glue off |
| `int4_b16_fqkv_nor2` | 16 | yes | 0 | **0** | P5 split, fused leg: round-2 glue off |
| `int4_b16_distinct` | 16 | no | 0 | 1 | **untimed**, last: `E4B_FUSE_ROUTER_EPI=0`, the router hook counts distinct experts per layer per decode step (64 generated tokens) |

The K17 route check: the `_fr` arms' censuses carry **no** `_reduce_partials` row and the controls do (rc 43/44 otherwise, P54's engagement-check shape).

## Predictions, registered before the data

- **P1 (K17 P4, B=1):** `_reduce_partials` at **0 calls/step** in `int4_b1_fr`; the step falls **≥ 0.15 ms** from `int4_b1`'s median (P54 read the row at 0.300 ms on this stack); generated tokens **identical** on both draws (K17's P1 is bitwise, held under the interpreter 197/197 and compiled 27/27). *Refuted by* any token difference (K17's P1 failed in the wild — the kernel is withdrawn), or a saving under **0.08 ms** (the reduce launches were overlapped in the graph, as P54's K16 row was). **The A2000 pilot is a registered warning:** its R=1 savings were 0.0–2.8 µs per call, five of six under K17's 2 µs line, on a card whose planner collapses SK to 1 above R=8; the 5090 keeps SK 6–16 at every R, so the pilot does not transfer, but a B=1 saving near 0.1 ms would be the pilot's number and not a surprise.
- **P2 (K17 P4, B=16):** the reduce row at 0 calls; the step falls **≥ 0.10 ms** (row 0.324); tokens identical.
- **P3 (P5 split):** with the round-2 glue forced off on both legs, `int4_b16_fqkv_nor2` vs `int4_b16_nor2` — **if the tokens still diverge**, P54's divergence is the K16 GEMM's N-dependent accumulation (the kernel-shape hypothesis) and the glue path is exonerated; **if they are identical**, it came from the round-2 glue taking its `qkv_proj` path. Registered as a two-way read with no preferred outcome; the A/A (`int4_b16_nor2` draw 1 vs 2) must be identical for either reading to count.
- **P4 (distinct experts):** the mean over layers and decode steps of distinct experts per layer at B=16 on the harness's prompts is **≤ 80** (uniform-router expectation 80.9 for E=128, k=8, B=16; real routing is skewed). A soft prior on a measurement, not a claim: #564's expert-tier roofline is 5.336 ms at 64 distinct and 6.670 at 80 against a measured GEMV of 6.2–6.7 ms, so this number decides whether the largest row has ~1.3 ms of headroom or none. Reported whatever it reads.
- **P5 (bookkeeping):** `int4_b16` and `int4_b1` reproduce P54's medians within ±3 % (cross-box, informative only; STOP-1 at ±8 %).

## Decision rule

P1 ∧ P2 → `GNF4_GEMV_FUSED_REDUCE` defaults **on** in the next grouped-nf4-gemm release; the consumer preallocates `cnt`/`out` in `Int4Linear._install` (hygiene, not correctness); the B=1 position is re-quoted from THIS box's control-vs-`fr` pair. P1 ∧ ¬P2 → on at B=1 only (by R, in the planner). ¬P1 → nothing ships; K17 is re-read at the kernel level. P3 either way → the KL-from-checkpoint arm that bounds the fused-qkv B=16 divergence (bar ≤ 0.10 nats / top-1 ≥ 0.93) is written into `kl_serve` (its REFERENCE entry first — #647's lesson) and registered by amendment; the gate on any B=16 `--fuse-qkv` default stays where P54 left it. P4 → #564's table gets its measured number and the expert-GEMV lever is sized from it.

## Budget and STOP rules

One RTX 5090 (verified/secure), ≤ 2.5 h, estimate ≤ $1.65 at $0.66/h (12 timed arms ≈ 5 min each + one untimed, after a ~20-min fetch + bake); lane ceiling $3, hard stop $5. No proving run (guard ≤ 2.5 h; P54 proved this path on 2026-09-21). STOP-1 control outside ±8 % of P54's 11.42 / 4.21 ms: quote nothing across boxes. STOP-2 an arm that cannot finish 10 min before the deadline is skipped and recorded; the distinct arm is last so a slow host loses the untimed measurement first. STOP-3 a first-draw census missing, or the K17 route check failing: the run exits non-zero. STOP-4 no second box on a disappointing result.

## Receipts

`receipts/experts4bit-qlora/<date>/p57/`: `logs/census_<arm>.txt` (first draws), `e4b_b{1,16}_<arm>[_r2].json`, `distinct_experts_b16.json`, `summary.txt`, `forensics.txt`, `versions.txt`, VRAM traces. `RESULTS-p57.md` is generated by `p57_reduce.py` (P54's reducer with this arm table and the distinct-expert section) and nothing else; the receipt files are committed under `bench/p57/receipts/` with the read.

## Amendments

(none yet)


---

## Amendment 1 (2026-09-22 ~14:20Z — after the P1/P2/P3/P5 data were read, BEFORE any P4 data exists)

Lane `p57-5090-2` ran every timed arm clean (12 receipts, A/A within 0.01 ms) and the untimed distinct-expert arm **failed without measuring**: `step_decomp.py … --gen-tokens 64` at B=16 leaves the bv3 stage a **6-step window** (the harness reserves ~58 generated tokens for warm-up and capture; P54's series arm used 128 → 70 steps) and its own guard fired — `AssertionError: window too small for bv3 (6)`. The hook still dumped `distinct_experts_b16.json` with `steps: 0`, so the runner's `[ -s … ]` check passed (a check weaker than the property it guards — the file existed, the measurement did not) and the lane exited rc=0; the reducer refused the file, so **P4 is UNREAD**, not misread. The dry-run rule was applied to the hook and the counter on a tiny model, not to the harness's window arithmetic at the registered token count — the same class of gap as P54's series arm.

**Changes, before the re-run:**
1. The distinct arm passes **`--gen-tokens 128`** (P54's series arm's value; 70 decode steps counted).
2. The runner refuses a dump with `steps: 0` as **rc 45** (`DISTINCT EMPTY`), so an empty measurement can no longer exit green.
3. **`P57_ONLY_DISTINCT=1`** skips the twelve timed arms (already read from `p57-5090-2`, receipts in `bench/p57/receipts/`), so the re-run — lane **`p57b-5090`**, est ≤ $0.35, guard 45 min (install + fetch + bake ≈ 20 min, the arm ≈ 5 min) — measures P4 and nothing else. Its P4 prediction is unchanged (mean distinct experts per layer per decode step ≤ 80; uniform expectation 82.4 for E=128, k=8, B=16 as the hook computes it — the prereg's 80.9 was an arithmetic slip; the soft prior is unchanged in spirit: real routing is skewed below uniform).
4. **STOP-1 reference corrected:** this pre-registration compared `int4_b1` (fused q/k/v, per the arm table) against P54's **unfused** control 4.21 ms; P54's fused B=1 median was **3.693 ms**. On `p57-5090-2` the fused B=1 stack read **4.236 ms**, +14.7 % over P54's fused arm on P54's box — outside ±8 %, so **no B=1 number from this box is compared to P54's** (this lane has no unfused B=1 arm to say whether the fusion's saving or the whole step moved); the B=16 control reproduced within +1.4 %. Within-box ratios are the position either way (P37 rule 1).

Nothing above changes a registered prediction's band or the decision rule; the lane's P1/P2/P3/P5 read stands as taken.


## Amendment 2 (2026-09-22 ~15:05Z — after `p57b-5090-2` read 3 decode steps, before any full P4 read exists)

`p57b-5090-2` (amendment 1: `--gen-tokens 128`, `P57_ONLY_DISTINCT=1`) ran the distinct arm and the harness died **inside CUDA-graph capture**: `torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture` at `_bv3_stage`'s `torch.cuda.graph(...)`. The cause is the instrument: [`distinct_experts.py`](distinct_experts.py)'s router hook calls **`torch.unique(ids).numel()`** per layer per call — `torch.unique` needs the result size on the host, a device→host synchronisation that is illegal while a graph is being captured. The counter therefore saw only the harness's **eager warm-up decode steps — 3 of them** (`raw_calls 127 = 124 prefill chunks + 3 decode calls` per layer) — and the atexit dump carries `steps: 3`, so amendment 1's `steps: 0` refusal did not fire and the lane exited rc=0 (the arm's own rc was 1, recorded in `summary.txt`). The dry run of the counter on a tiny model used the eager path and could not have shown this.

**Partial read, stated for what it is (3 decode steps × 16 rows, warm phase, not the registered 70-step window):** mean over 48 layers of distinct experts per layer per decode step **54.8** (layer means 46.7 … 67.0; uniform-random expectation 82.4 for E=128, k=8, B=16). Consistent with the ≤ 80 prior; **not registered as a row** — three steps is a glimpse, not the measurement. Receipts: [`receipts/p57b/`](receipts/p57b/).

**Change, before the full read:** the distinct arm runs **`--b1d-loop eager`** — it is untimed by registration (its step time is never quoted), and the eager loop captures no graph, so the hook's synchronising `torch.unique` is legal on every one of the 70 timed-window steps plus warm-up. Nothing else in the arm changes (`--gen-tokens 128`, `E4B_FUSE_ROUTER_EPI=0`, `--amort off`, the same hook byte-for-byte). The runner additionally refuses a dump with `steps < 16` as rc 45 (`DISTINCT SHORT`), so a partial warm-phase count can no longer exit green. Lane **`p57c-5090`**, ≤ $0.35 (install + fetch + bake ≈ 12 min on a good host, the arm ≈ 3 min eager). P4's prediction is unchanged (≤ 80).


## Amendment 3 (2026-09-22 ~16:05Z — after `p57c-5090-2` repeated the 3-step outcome; before any full P4 read exists)

Amendment 2's fix was **inert**: `--b1d-loop` selects the loop for the B=1 stage only (`step_decomp.py` reads `a.b1d_loop` in the B=1 code paths; `_bv3_stage`, the B>1 stage, captures a CUDA graph unconditionally). `p57c-5090-2` ($0.14) therefore reproduced `p57b-5090-2` exactly — the counter's `torch.unique` raised `operation not permitted when stream is capturing`, 3 warm-up steps counted — and the new `steps < 16` refusal fired as designed (rc 45, HARNESS_ERROR, no green exit). Receipts: [`receipts/p57c/`](receipts/p57c/). The lesson is the standing one, applied to myself a third time in this lane: a flag was registered without checking what it governs ([[feedback_inert_and_untested_are_the_same_fact]]).

**Change, before the full read — the instrument, not the harness:** [`distinct_experts.py`](distinct_experts.py) v2 does **no host work in the hook**. For a decode call it scatters the routed ids into a preallocated per-layer `[E]` buffer, sums it to a device scalar and accumulates in place — sum of distinct counts, min, max, decode-call count, and per-expert touch counts — with static shapes, no allocation and no synchronisation; recorded during capture, those ops replay with the graph, so every replayed step is counted. `report()` moves the counters to the host once, after the run; the per-step `series` is gone (it needed the host per step), replaced by `touched_frac[e]` per layer. **Dry-run under a real capture, this time:** `python distinct_experts.py --capture-test` builds a fake router whose ids come from a table indexed by a device step counter, runs 3 eager steps + 1 side-stream warm + 1 captured step replayed 20×, and checks the counter against a CPU reference over exactly the executed steps — on the QNAP RTX A2000 (torch 2.8.0+cu128): **24/24 steps counted, mean/min/max exact on all three layers, `touched_frac` sums equal the means**; the CPU `--self-test` (tiny Qwen3-MoE, 3 layers × 4 steps, prefill dropped by row count) passes. The hook (`hook/usercustomize.py`) passes `num_experts` and `batch` to the new constructor; `report()` takes no arguments. The runner's distinct arm goes back to `--b1d-loop graph` (the eager value was inert and misleading); `--gen-tokens 128`, `steps < 16 → rc 45`, `P57_ONLY_DISTINCT=1` stay. Lane **`p57d-5090`**, ≤ $0.49. P4's prediction is unchanged (mean ≤ 80; the two 3-step glimpses read 54.8).
