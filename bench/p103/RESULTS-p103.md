# P103 — results: **stopped at the proving rental, no verdict**. On the 5090, flash-linear-attention and causal-conv1d install and engage and decode graphs replay exactly on them, but the hybrid parity premise fails; the cause (corrected 2026-10-03, see below) is that premise's single-seed MoE statistic, a lottery in every kernel arm

Registration: `bench/p103/PREREG-p103.md` (#929, `d1554ea`). Issue: #928. Code under test: e4b at `d1554ea`,
grouped-nf4-gemm at `34da93d`, `flash-linear-attention==0.5.2`, `causal-conv1d==1.7.0`, torch 2.8.0+cu128 held.

**No verdict is read.**
- The registration gave the proving rental this job: to answer "whether the pinned kernels install and pass the
  premise on sm_120 before any checkpoint is fetched". It answered: they install and engage, and both kernel phases
  fail the premise.
- Under the registered rule, a reading would score phases f and fc NOT_SUPPORTED on that premise and skip their arms.
  It could only re-measure phase t, which P101 (#926) already read.
- So the reading is not run, and P103 closes here. No registered consequence fires.

## Correction (2026-10-03, after P104's proving rental; #928)

**This page first attributed the premise failure to fla's chunk-split variance. That attribution is withdrawn.**

The withdrawn title read:
> P103 — results: **stopped at the proving rental, no verdict**. On the 5090, flash-linear-attention and causal-conv1d install and engage, but the hybrid GPU premise fails with them; the cause, isolated at $0, is fla's chunk kernel not being invariant to how a prompt is split, which transformers' own chunked prefill shows identically

**What changed:**
- P104 (#936) gated on a chunk-matched copy of the failing test. Its proving rental (`p104-prove-1`) still failed the
  fp8 hybrid test under fla: 7.54e-2 with fla and 8.13e-2 with fla + causal-conv1d, against a 4.05e-2 bound.
  Chunk-matching barely moved the number.
- A $0 probe on the A2000 (`bench/p104/a2000/`, probes 7 and 8) then ran the test's own models through e4b's
  `PagedModelRunner`, with SDPA standing in for the fp8 kernel's attention, over 10 seeds.
  - The test's **MoE** hybrid (4 experts, top-2) is bimodal **in every arm, including the torch path**: 5 of 10 seeds
    on the torch path land at 1–2e-1, above the test's own bound.
  - A **dense** hybrid sits at the control's error on every seed (2.1–3.9e-2), and is indistinguishable across the
    torch path, fla, and fla + causal-conv1d.
- The test's seed 0 lands in the low mode on the torch path (2.6e-2 on the 5090) and in the high mode under fla
  (7.5e-2). That is all the "failure" was.

**What stands:**
- the kernels engage on the card;
- decode graphs replay exactly on them;
- fla's chunk rule is split-variant (the isolation below), as transformers' own chunked prefill shows.

**What is withdrawn:** that this split-variance is why the hybrid test fails ("Why the premise fails" below, and its
echo in `docs/SERVING.md`, corrected in the same PR as this section).

## The proving rental (`p103-prove-2`)

One RTX 5090 (sm_120, driver 580.95.05) on an AMD EPYC 7663, Vast machine 142284. $0.0426, 271 s; destroyed and proven
absent at 2026-10-03T04:45:12Z.
- **Install and tripwire held:** e4b `d1554ea`, gnf4 `34da93d`, torch 2.8.0+cu128, transformers 5.17.0. Phase t is
  kernel-free. The reducer's self-test passed 20 cases.
- **Both kernels engaged exactly as registered:**
  - phase f: fla's `chunk` / `fused_recurrent` gated delta rules;
  - phase fc: those plus causal-conv1d's conv fn and update;
  - torch held throughout, and no `kernels` hub package.

| phase | premise | `test_the_hybrid_paged_path_on_the_real_kernel_stays_within_its_control` |
|---|---|---|
| t (torch path) | **4 passed** | control worst 2.027e-02, hybrid worst 2.632e-02, tokens equal |
| f (fla) | **1 failed, 3 passed** | control 2.027e-02, hybrid **7.577e-02**, above the bound of 4.05e-02 (2 × control); tokens equal |
| fc (fla + causal-conv1d) | **1 failed, 3 passed** | control 2.027e-02, hybrid **8.229e-02**, above the bound; tokens equal |

- **The other three premise tests pass in every phase:** hybrid bucket-graph replay against the padded eager step,
  and the linear state under real capture. **Decode graphs replay exactly on the kernels.**
- **`p103-prove-1`** (machine 152323) stayed `loading` for 600 s and was destroyed before any lane code ran: $0.0904.
- **The lane: $0.1330.**

## Isolation at $0 (NAS RTX A2000, sm_86; scripts in `a2000/`)

**Setup.** The failing test's model shape, with every layer linear, so it needs no fp8. The pool is driven as
`PagedModelRunner.run_prefill` / `run_decode` drive it. Errors are worst relative logit error against transformers'
DynamicCache with a single prefill, over 20 seeds (`probe4.py`, `probe5.py`):

| | torch path | fla 0.5.2 | fla + causal-conv1d 1.7.0 |
|---|---|---|---|
| e4b pool, 32-token chunked prefill, one sequence per decode step | **0 in 20/20** | median 6.080e-3, mean 4.948e-3, max 8.065e-3 | median 6.192e-3, mean 4.439e-3, max 8.380e-3 |
| transformers' own DynamicCache, 32-token chunked prefill | **0 in 20/20** | **median 6.080e-3, mean 4.948e-3, max 8.065e-3**; 17/20 nonzero | median 6.192e-3, mean 1.681e-1, **max 1.660**; 14/20 nonzero |
| e4b pool, single prefill, batched decode with changing row order | median 6e-8, max 6.5e-3 | median 1.3e-3, max 6.2e-3 | median 0, max 6.6e-3 |

**Kernel level** (`probe3.py`, `probe5.py`), at Qwen3.6's Gated DeltaNet shape (32 heads, K = V = 128) and the test's
(4 heads, 32):
- **The decode-step recurrent rule is exactly batch-invariant** for both the torch reference and fla's
  `fused_recurrent`: a batch of 6 or 3 equals one row at a time, and permuting the rows changes nothing, bit for bit.
  fla agrees with the torch reference to 1.9e-6 absolute (4.5e-5 relative).
- **The chunk rule, run whole versus split at 32** with the first call's final state carried in:

  | shape | torch path, relative | fla, relative |
  |---|---|---|
  | small | 6e-7 | **5.1e-3** |
  | Qwen3.6 | 1.7e-3 | **3.4e-3** |

  The four ways of passing that state (as returned, `.contiguous()`, `zeros_like().copy_()`, a fresh copy) give
  identical results, so the layout e4b's pool hands over is not a factor.

**Reading:**
- **Not an e4b defect.** Under fla, e4b's chunked prefill and transformers' own chunked prefill are
  indistinguishable: identical median, mean and maximum over the same 20 seeds. On the torch path both are exact.
- **The source** is fla's chunk kernel, whose result depends on where a prompt is split. The torch path, computing in
  fp32, is essentially split-invariant.
- **[WITHDRAWN 2026-10-03, see the correction above] Why the premise fails:** the test compares e4b's 32-token-chunked paged path with a single-call reference. Its
  bound, 2× the all-attention control, was calibrated where chunking is free.
- **Batched decode** drifts about equally in every arm: ordinary bf16 batching noise in the projections and experts.
- **Unexplained:** with causal-conv1d engaged, transformers' own chunked prefill read 1.660 relative error on one seed
  of 20, while e4b's pooled path on the same seeds peaked at 8.4e-3. The mechanism is not established, and nothing is
  filed upstream.

## What follows

- **`docs/SERVING.md`:** the hybrid section records that the kernels engage and decode graphs replay exactly on
  them, but that fla's chunk kernel makes chunked prefill depend on chunk boundaries. With them installed, e4b's hybrid
  parity premise, calibrated on the torch path, fails. **The torch path remains the read configuration.**
- **A follow-up lane (not registered):** a correctness premise that compares e4b's paged path with transformers' path
  at the **same** prefill chunking. Under fla that comparison is exact in the isolation above. Such a lane could then
  measure what the kernels buy.

Receipts (`receipts/p103-prove-2/`):
- `summary.txt`, `forensics.txt`, `versions.txt` and the three engagement records;
- the premise, probe and kernel-install logs;
- the teardown proof;
- `SHA256SUMS` over every file the box wrote.

The launcher's receipts and ledger rows are in the receipt store (adertha-receipts `5c42544`, `e2a5be9`).
