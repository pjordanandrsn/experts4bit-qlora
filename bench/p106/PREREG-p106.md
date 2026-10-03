# P106 — what the Gated DeltaNet kernels cost in quality and buy in prefill: flash-linear-attention and causal-conv1d against transformers' torch path, switched in ONE process on ONE RTX 5090, on Qwen3.6-35B-A3B through e4b's hybrid paged path (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#944. Follows P105 (#942), which recommended the kernels for hybrid paged serving on decode
speed alone.

**Why.**
- **P105 left quality unmeasured.** The kernels change greedy trajectories: phases f and fc agreed with the torch
  path's graph tokens on 41–57 % of positions. Free-running greedy decode diverges for good after one flipped argmax,
  so that does not say whether the kernels are worse. P105's dense parity premise bounds e4b's paged path against
  transformers running the SAME kernels. It says nothing about the kernels against the torch path.
- **Tiny models cannot answer it.** A tiny random hybrid's logits are near-uniform (NLL ≈ ln 256). Its worst relative
  logit error under the kernels against the torch path reads 1.3–1.4 (this page's toggle check), while its KL reads
  about 1e-5 (the rehearsal below). Only the real checkpoint can say what the kernels cost.
- **P105 left prefill unmeasured.** P98's harness times decode only. On the A2000, fla's chunk rule is 4–10× the torch
  path's per call (#928, `bench/p103/a2000/probe6.py`), and transformers' own comment puts the torch chunk rule more
  than an order of magnitude behind on an H100. TTFT is where the kernels may matter most.

**The instrument: both paths in one process.**
- transformers 5.17 resolves each of the four Gated DeltaNet functions once, at import, into a closure (`implementation`,
  `is_new_implementation`, `applicable_params`). The models look the name up at call time.
- `gdn_toggle.GdnToggle` writes those three cells to the torch function's values (`use("torch")`), which is exactly
  what the closure holds when the package is absent. It writes the saved values back with `use("resolved")`.
- So one engine, one set of weights, one host and one moment serve both paths, with no cross-process noise.
- **The toggle is proven, not assumed.** `toggle_probe.py` saves tiny hybrids' logits in a kernel-FREE process, then
  compares them in the kernel process. The toggled-off path must equal them bit for bit, the toggled-on path must
  differ, and both must repeat.
  - On the NAS A2000, 2026-10-03: 4 seeds of a dense and a MoE hybrid, through transformers' DynamicCache and e4b's
    paged path.
  - Toggled off, under fla and under fla + causal-conv1d: **0 differing logits**, every step.
  - Toggled on: 33,507 / 36,050 differing logits under fla, and 34,844 / 36,758 under fla + causal-conv1d (HF / paged),
    with identical counts on the repeat.
  - Logs: the feasibility run 08:17:41Z–08:21:59Z (`a2000/run_toggle_feasibility.sh`, the probe's first draft, e4b
    `4bddf92`), and the bench version inside both rehearsals below.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

**One RTX 5090, one process for the measurement** (`p106_run.sh`, `p106_box.py`):
1. Install e4b and gnf4 with torch held, and run the tripwire (no kernel module present).
2. Run the reducer's self-test (26 cases).
3. **Toggle SAVE:** the tiny hybrids' logits in this kernel-free process.
4. Install `flash-linear-attention==0.5.2`, then `causal-conv1d==1.7.0`. Record the engagement (`kernels_fc.json`).
5. **Premise:** P105's kernel-phase set (the two graph files, the chunk-matched all-linear test, the dense parity file),
   **9 passed**, none skipped.
6. **Toggle CHECK** against step 3 (`toggle_check.json`).
7. Fetch Qwen3.6-35B-A3B at `995ad96`, and bake P98's NF4 arena.
8. **The box:** `serve_paged.build_engine` (P98's engine: all-vram NF4, fp8 paged KV, chunk 512, eager), then:
   - **Q (quality):** 8 wikitext windows (P98's K8 windows 0–7), each 2,048 tokens of chunked prefill plus 64
     teacher-forced single-row decode steps.
     - The torch path and the kernels run **in lockstep on two KV slots**: each prefill chunk and each decode step on
       one path, then the other.
     - Every position is compared on **fp32 log-probs** (P97's lesson): KL(torch ‖ kernels), next-token NLL under each,
       argmax agreement, bit-identity.
     - That is 16,384 prompt positions and 512 decode steps per path.
     - **Null pair:** windows 0–1 also carry a third slot on the torch path again, the instrument's own floor.
   - **M (the mutant):** window 0 again (prefill plus 16 steps), with fla's delta rules called with
     `use_qk_l2norm_in_kernel=False`, a wrong flag a caller could pass. The bar must catch it.
   - **T (TTFT):** one request at a time through the scheduler with `max_new_tokens=1` (the first token comes from the
     prefill), at prompts of 512, 2,048 and 4,096 tokens.
     - One untimed warm-up per path and length (fla's Triton autotune).
     - Then 5 rounds, torch–kernels on even rounds and kernels–torch on odd ones, CUDA-synchronised. The warm-up
       uses wikitext window 8 and each round its own window (9–13), the same for both paths.
9. Reduce.

A failed kernel install (27), premise (25) or toggle check (28) ends the run before the fetch.

**The rule** (`p106_reduce.py`, self-test 26 cases):
- **NO_READING:** a failed premise, or no box record.
- **VOID**, any of:
  - the toggle check failed;
  - the engagement is not as registered:
    - the rules resolve to `fla.ops.gated_delta_rule.{chunk,fused_recurrent}`, the conv functions to
      `causal_conv1d.causal_conv1d_interface`, with the pins installed and no `kernels` package;
    - the box's torch record is transformers' own functions;
    - the model's Gated DeltaNet module is among the toggled ones;
  - the loaded commit is not `995ad96`;
  - the kernels' logits equal the torch path's on every compared position (the switch was inert);
  - the null pair reads a mean KL above 0.005 (a tenth of the bar) on either phase;
  - the mutant passes the bar on both phases (the gate cannot fail).
- **NEUTRAL:** on BOTH phases (prompt positions, decode steps), all three hold:
  - mean KL(torch ‖ kernels) ≤ **0.05** nats;
  - argmax agreement ≥ **0.85**;
  - mean d_nll (kernels minus torch) ≤ **+0.01** nats.

  The KL and agreement bar is P97's registered faithfulness bar (G2), and the d_nll ceiling is added here.
- **COST:** otherwise.
- **TTFT is reported, never gated:** the median torch ÷ kernels ratio per length.

**The registered consequence:**
- **NEUTRAL:** `docs/SERVING.md`'s hybrid section replaces "quality … not measured" with the measured KL, agreement and
  d_nll, and adds the TTFT ratios. The recommendation stands. Register row
  `e4b.serve.p106.qwen36-gdn-kernel-quality.5090.<date>`.
- **COST:** `docs/SERVING.md` keeps the kernels as supported but **withdraws the recommendation**, stating the measured
  cost and the TTFT ratios. A register row. An issue for the cause.
- **VOID / NO_READING:** nothing moves; the reason goes on the issue.

**Not measured:**
- quality on any corpus but wikitext-2, or beyond 2,048 tokens of context;
- the kernels under decode graphs (P105 read those for speed; the box decodes eager, and P105 showed graph replays equal
  the padded eager step);
- any host but this one;
- TTFT with more than one request in flight.

## Predictions (written 2026-10-03, before the A2000 rehearsal's box output and before any 5090 data)

- **NEUTRAL.**
- **Quality:**
  - mean KL between 1e-3 and 2e-2 nats on both phases;
  - argmax agreement ≥ 0.95 on both;
  - |mean d_nll| ≤ 0.005 nats.
- **Null pair:** bit-identical on every position (KL exactly 0).
- **Mutant:** mean KL above 1 nat on both phases.
- **TTFT (torch ÷ kernels, median):** 4,096 tokens between 1.3× and 4×; 512 tokens between 1.1× and 3×; the ratio
  grows with length.
- **Peak GPU memory** at or below 28 GB.

## Box and cost

- **`p106-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P106_PROVE=1`, no model.
- **`p106-5090-<n>`:** one RTX 5090 with ≥ 200 GB of disk. **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).**
- **Lane ceiling $3.00; hard stop $4.00.**

## Rehearsal

**Rehearsed 2026-10-03 on the NAS RTX A2000 (sm_86, driver 575.64.05), at `dfcbdc1`, staged exactly as `p106_drive.sh`
stages.** The knobs were class A2000, disk 10 GB and premise skips allowed. Times are the logs' own stamps.

1. **The proving path** (`P106_PROVE=1`; `a2000/rehearse_prove_and_box.sh`, 08:28:04Z–08:32:00Z): rc 0, with `PROVED`
   and the `REHEARSAL` marker.
   - **Tripwire held:** e4b 0.41.0 @`dfcbdc1`, gnf4 `34da93d`, torch 2.8.0+cu128, transformers 5.17.0. No kernel
     module before the installs. Reducer self-test 26 cases.
   - **Kernels:** the toggle reference was saved kernel-free; fla 0.5.2 and causal-conv1d 1.7.0 installed with torch
     held; the engagement was exactly as registered.
   - **Premise:** 7 passed, 2 skipped (the fp8 tests on sm_86). Dense ratios 0.74–1.36 over the 8 seeds.
   - **Toggle check OK:** the torch path read 0 differing logits against the kernel-free process on both of its steps.
     The kernels read 34,844 / 36,758 differing (HF / paged), with the same counts on the repeat.
   - **HF CDN probe:** 46.7 MB/s.
2. **The box on Qwen3.6 does not fit this card.** The home lab's resident GPU services hold ~8 of its 12 GB. Loading
   the untied `lm_head` ran out of memory at 3.7 GB in use, so this step is not rehearsable here. The engine itself is
   P98's, run on the 5090 by P98, P101 and P105.
3. **The box end to end on a tiny random Qwen3.5-MoE hybrid** (`a2000/rehearse_prove_and_tiny_box.sh` with
   `a2000/box_rehearsal.py`; the proving path again 08:37:38Z–08:41:04Z, identical lines and CDN 55.9 MB/s; then the box
   08:41:04Z–08:41:21Z).
   - **What was replaced:** `build_engine` (by the same construction around the tiny model), the wikitext windows (by
     random tokens) and the loaded commit (by the revision).
   - **Sizes:** 2 windows × (512 + 16), chunk 128, a 4-step mutant, TTFT at 256 / 1,024 tokens × 2 rounds.
   - **Result:** rc 0. The reducer read **NEUTRAL** on the box's own record.
   - **The kernels against the torch path:** KL 9.3e-6 (prefill) and 1.3e-5 (decode); agreement 0.983 / 1.000.
   - **The null pair:** bit-identical on every position (512 / 512 and 16 / 16).
   - **The mutant:** KL only 1.2e-3 / 1.0e-3 on near-uniform logits, but agreement 0.664 / 0.25. It fails the bar on
     agreement, so the gate is live even here.
   - **TTFT ratios ~1.0:** a tiny model is launch-bound, so this is no guide to the 5090.
   - These numbers come from a random model. They show that the path works, not what the reading will find.

Amendments, dated, go below this line before any data is read.
