# P91 — where do the NF4 families' decode steps go? A kernel census of Granite-3.1-3B-A800M and OLMoE-1B-7B on their licensed configurations, at B=16 and B=1 on one RTX 5090 (registered 2026-10-01, before any run)

Issue: experts4bit-qlora#564. This is a **descriptive** lane: no change is tested and nothing is licensed. It sizes
the next family lever before a kernel is written.

**Why now.**
- Two family results landed today. P89 licensed K23's lean glue on Qwen3. P90 licensed K21 on gpt-oss's MXFP4 store,
  taking gpt-oss's B=16 step to ×0.581 by replacing the NF4 grouped GEMM its batched rows fell back to.
- That NF4 grouped GEMM (`_gemm_nf4_grouped`) was 79 % of gpt-oss's B=16 step. grouped-nf4-gemm's K22/K24 benches read
  it at about 25 % of the byte floor on gpt-oss's shapes.
- NF4 is the **licensed** expert configuration of Granite-3.1-3B-A800M (bo7's `r12epi`: NF4 experts + round-1/2 folds
  + router epilogue, K8 +0.019 ppl) and OLMoE-1B-7B (bo7's `nf4`; nothing above NF4 is licensed there).
- bo7 timed both families (Granite 3.28 / 8.71 ms at B=1 / B=16, OLMoE 3.54 / 11.87) but never censused them, so how
  much of their steps the NF4 kernels take is unknown.

## The census

- **Box.** One RTX 5090, any CPU vendor.
- **Software.** e4b at the launch commit (the current defaults: K19 / the lean glue / K21 do not touch NF4 stores) and
  grouped-nf4-gemm at `4cc831c` (P90's pin).
- **Families,** at the revisions P44 pinned:
  - Granite `ibm-granite/granite-3.1-3b-a800m-instruct @a0278068` with `r12epi`'s env;
  - OLMoE `allenai/OLMoE-1B-7B-0924-Instruct @7f1c97f4` with `nf4`'s env.

  Both envs are `bench/p44/serve_stack.py`'s `arm_env`, verbatim. The pin test checks them against it.
- **Arms.** P88's harness (`bench/p39/step_decomp.py` + the bench/p42 hook): graph-window timing, prompt 512, 128
  generated tokens, `--no-fuse-qkv`, and P42's replay census. B=16, then B=1, per family, Granite first. One draw each.

**The reducer** (`p91_reduce.py`, with an 8-case self-test):
- **READ** if all four arms are present and every census reconciles with its step: kernel ms per step within 10 % of
  `step_ms_clean`, P86's tolerance.
- **NOT_READ** otherwise. No share is quoted from a NOT_READ run.
- **Reported** per arm: the step, the census total, the NF4 expert kernels' ms and share of kernel time
  (`_gemm_nf4_grouped` at B=16, the `_gemv_nf4*` GEMVs at B=1), and the top eight kernels.

**The registered decision.**
- If the NF4 grouped GEMM is at least **40 %** of B=16 kernel time in either family, the next lane is an NF4 grouped
  small-M kernel: K19's skeleton with NF4 dequant. It changes arithmetic (bf16 MMA in place of TF32), so that lane will
  carry a per-family quality gate.
- Otherwise, the next lane is the largest non-NF4 kernel family.

## Predictions (written before the data)

- **Granite B=16:** NF4 grouped GEMM **40–60 %** of kernel time. It has small experts (H 1536, I 512, 40 experts,
  top-8) over 32 layers, so attention, norms and glue weigh more than on gpt-oss.
- **OLMoE B=16:** **50–70 %**. It has larger experts (H 2048, I 1024, 64 experts, top-8) over 16 layers.
- **B=1,** both families: the NF4 decode GEMVs at 30–50 %.
- **Both censuses reconcile** within 5 %, as P86/P88/P90's did.
- **The decision:** the NF4 small-M kernel.

## Box and cost

- **`p91-5090-<n>`:** one RTX 5090, **guard 1.0 h at ≤ $0.75/h (≤ $0.75)**, so there is no proving rental under the rule.
- The run fetches about 6.6 GB (Granite) and 14 GB (OLMoE), bakes two NF4 arenas, then runs four census arms of
  about 3 minutes each.
- **Lane ceiling $1.50; hard stop $2.00.**

## Rehearsal

The whole runner ran on the NAS RTX A2000 (sm_86) from e4b `37dfb45`, this branch before this section. It ran with
`P91_GPU_CLASS=A2000 P91_MIN_DISK_GB=20`, so it was marked REHEARSAL.
- **Up to the census, it held:**
  - the reducer self-test (8 cases);
  - install and tripwire;
  - both families fetched at their pins and baked.
- **Every census arm stopped** at the fp8 paged-KV append. A diagnostic re-run of one arm, with its log kept, read
  `fp8e4nv not supported in this architecture`. The decode stack needs sm_89+, so this is the rehearsal card's limit,
  not the runner's.
- **The reducer** then read NOT_READ with all four arms missing, as it must.
- No time is quoted (the A2000 is correctness-only).

Amendments, dated, go below this line before any data is read.
