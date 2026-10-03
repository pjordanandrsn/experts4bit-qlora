# P104 — P103 re-asked with a chunk-matched premise: with flash-linear-attention and causal-conv1d installed, does e4b's hybrid paged path still match transformers and replay its decode graphs exactly, and what do the kernels buy, on one RTX 5090 and one host (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#928. Follows P103 (#929, #932), which stopped at its proving rental.

**Why.**
- P103 asked this question. On the 5090 both kernels installed and engaged, and the graph tests passed. But
  `test_linear_state_gpu.py`, which compares e4b's 32-token-chunked paged path with a **single-call** transformers
  forward, failed: hybrid 7.6e-2 against its 4.05e-2 bound, where the torch path read 2.6e-2.
- A $0 isolation on the A2000 (`bench/p103/RESULTS-p103.md`) found the cause in the kernel:
  - fla's chunk gated delta rule is not invariant to where a prompt is split;
  - transformers' own chunked prefill drifts from its single prefill exactly as e4b's does: identical median, mean and
    maximum over 20 seeds;
  - on the torch path both are exact.
  So that test charged the kernel's split-variance to e4b.
- **A new test compares like with like.** `tests/test_linear_state_chunk_matched_gpu.py`:
  - an all-linear pool against transformers' cache prefilled in the **same** 32-token chunks, required equal **bit for
    bit** (4 seeds, any CUDA card);
  - the fp8 hybrid against chunk-matched references, within 2× its all-attention control (sm_89+).
  On the A2000 the all-linear test passes on the torch path, with fla, and with fla + causal-conv1d, 4 of 4 each. A
  mutant that stops marking the state between prefill chunks fails it in every arm, 4 of 4.
- **What the kernels might buy** (indicative A2000 per-call times, #928): fla's decode-step recurrent rule is about
  5–6× faster than the torch path per layer, including under CUDA graphs (at 16 rows, 0.27 against 1.64 ms). Its
  prefill chunk rule is 4–10× faster. The conv update is a wash.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

**P103's lane, with the premise changed and nothing else:**
- P98's measurement at its registered bytes: arms **g** (bucketed graphs 1–16), **e** (the padded-eager oracle) and
  **p** (plain eager), over W16 and W1.
- Three phases on ONE box: **t** (the torch path, asserted kernel-free), **f** (`flash-linear-attention==0.5.2`) and
  **fc** (plus `causal-conv1d==1.7.0`), with torch held and an engagement record per phase.
- P103's reducer and rule, unchanged.

**The premise per phase:**
- **t:** the three pinned hybrid files plus the chunk-matched file, **9 passed**, none skipped. This keeps P97–P101's
  premise intact on the torch path.
- **f, fc:** `test_hybrid_decode_graphs_gpu.py`, `test_linear_state_graph_gpu.py` and the chunk-matched file,
  **8 passed**, none skipped.
- **Reported in every kernel phase, never gating:** P103's single-call check (`test_linear_state_gpu.py`), so its
  number stays visible.

**The rule** (`p104_reduce.py`, P103's, self-test 20 cases):
- **Per phase:** P98's registered rule, unchanged. A wrong engagement record is **VOID**. A kernel phase whose install
  fails is **UNAVAILABLE**, one that is skipped is **NOT_RUN**, and one whose premise fails with the kernels engaged is
  **NOT_SUPPORTED**.
- **The lane:** **VOID** unless phase t is SUPPORTED; otherwise phase fc's result, else phase f's, else
  **UNAVAILABLE**.
- **`recommend`** is true iff phase fc is SUPPORTED and its graph arm decodes at least 1.05× phase t's on W16 or W1.

**The registered consequence:**
- **SUPPORTED, recommend:** `docs/SERVING.md` lists the two kernels as supported for hybrid serving and recommends
  them, with the measured gain, the install line, and the caveat that fla's chunked prefill depends on chunk boundaries
  (as transformers' own does). A register row.
- **SUPPORTED, not recommend:** the same, without the recommendation.
- **NOT_SUPPORTED:** the failing phase, bucket or test is recorded on #928, and SERVING.md warns against that kernel
  for hybrid serving.
- **UNAVAILABLE / VOID:** nothing moves.

**Not measured:**
- quality in nats against the torch path;
- prefill speed (P98's harness times decode only);
- any host but this one.

## Predictions (written before the data)

- **The lane is SUPPORTED, and `recommend` is true.** Every phase's premise holds, since the chunk-matched tests
  passed under both kernels on the A2000, and graphs replay exactly.
- **fc's graph arm over t's: W16 between 1.15× and 1.60×, W1 between 1.02× and 1.15×.**
  - At 16 rows the torch recurrent rule makes several passes over a 33.5 MB fp32 state per layer: a few
    milliseconds of P101's 28.1 ms step across 30 layers. fla makes one pass.
  - At one row the state is 2.1 MB, so the gain is small.
- **f's graph gain is at least 80 % of fc's.** The conv update is a wash.
- **fc's plain-eager arm over t's on W16: at least 1.15×.**
- **P103's single-call check fails again in both kernel phases (reported):** hybrid above 4.05e-2.
- **Peak GPU memory** stays at or below 28 GB.

## Box and cost

- **`p104-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P104_PROVE=1`, no model.
- **`p104-5090-<n>`:** one RTX 5090 with ≥ 200 GB of disk. **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).** Expected about
  40 min, as P103 estimated.
- **Lane ceiling $3.00; hard stop $4.00.**

## Rehearsal

- **The new test** on the A2000, at e4b `fd19337`:
  - 4 passed and 1 skipped (the fp8 hybrid, sm_86) on the torch path, with fla, and with fla + causal-conv1d;
  - the mutant fails 4 of 4 on the torch path and under fla + causal-conv1d.
- **The runner's proving path** is rehearsed on the A2000 before the merge, as P103's was.

**Rehearsed 2026-10-03, 05:36–05:40Z, at `f190ff3`, on the NAS RTX A2000**, staged exactly as `p104_drive.sh` stages.
`P104_PROVE=1` with the knobs class A2000, disk 10 GB and premise skips allowed. Exit rc 0, with `PROVED` and the
`REHEARSAL` marker.
- **Install and tripwire held:** e4b `f190ff3`, gnf4 `34da93d`, torch 2.8.0+cu128, transformers 5.17.0. Phase t is
  kernel-free. The reducer's self-test passed 20 cases.
- **Premise t:** 6 passed, 3 skipped. On sm_86 the three fp8 tests skip and the chunk-matched all-linear tests run, so
  the expected count is exactly 6 passed.
- **Phase f:** fla 0.5.2 engaged as registered. Premise 6 passed, 2 skipped: **the chunk-matched all-linear tests pass
  with fla's chunk and recurrent rules**. The single-call check was reported (it skips on sm_86). `phase f ok`.
- **Phase fc:** causal-conv1d 1.7.0 engaged as registered. Premise 6 passed, 2 skipped. `phase fc ok`.
- **HF CDN probe:** 48.3 MB/s.

Amendments, dated, go below this line before any data is read.
