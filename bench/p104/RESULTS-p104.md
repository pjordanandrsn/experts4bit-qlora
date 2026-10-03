# P104 — results: **stopped at the proving rental, no verdict**. The chunk-matched premise also fails under fla, and the cause is the premise's statistic, not the kernels: on its tiny 4-expert MoE, one seed's worst-case logit error is a lottery in every kernel arm, the torch path included

Registration: `bench/p104/PREREG-p104.md` (#936, `a1bd83d`). Issue: #928. Code under test: e4b at `a1bd83d`,
grouped-nf4-gemm at `34da93d`, `flash-linear-attention==0.5.2`, `causal-conv1d==1.7.0`, torch 2.8.0+cu128 held.

**No verdict is read.**
- The proving rental answered the question it was there for: both kernel phases fail the registered premise.
- The registered rule would score them NOT_SUPPORTED. The reading could only re-measure phase t, which P101 already
  read.
- So the reading is not run, and P104 closes here. No registered consequence fires.

## The proving rental (`p104-prove-1`)

One RTX 5090 (sm_120, driver 595.71.05) on an Intel Xeon E5-2698 v4, Vast machine 96642 (pre-flight 53.1 MB/s). $0.0527,
346 s; destroyed and proven absent at 2026-10-03T05:54:14Z.

| phase | premise | chunk-matched fp8 hybrid test (gating) | P103's single-call check (reported) |
|---|---|---:|---:|
| t (torch path) | **9 passed** | control 2.027e-2, hybrid 2.632e-2, pass | — |
| f (fla) | **1 failed, 7 passed** | hybrid **7.535e-2**, above the bound of 4.05e-2 | 7.577e-2 |
| fc (fla + causal-conv1d) | **1 failed, 7 passed** | hybrid **8.125e-2**, above the bound | 8.229e-2 |

- **Both kernels engaged exactly as registered.**
- **The four chunk-matched all-linear tests pass in every phase,** bit for bit against transformers' chunk-matched
  cache. So do the three graph tests.
- **Greedy tokens are equal throughout.**
- **Chunk-matching the reference barely moved the hybrid number** (7.58e-2 to 7.54e-2). P103's attribution to fla's
  chunk-split variance did not hold.

## The cause, found at $0 (NAS RTX A2000; `a2000/probe7.py`, `a2000/probe8.py`)

**Setup.** The failing test's own models through e4b's real `PagedModelRunner` with the fp8 KV pool. Decode attention
is SDPA over the pool's dequantized fp8 K/V (`p98_box.py`'s stand-in; sm_86 has no fp8 kernel). The metric is the
test's own: worst relative logit error over 3 sequences × 7 steps against transformers' chunk-matched cache, with the
test's batched, row-reordering decode. 10 seeds; seed 0 is the test's.

| model | arm | control median | hybrid median (range) | hybrid seeds above 1e-1 |
|---|---|---:|---|---:|
| dense (Qwen3.5, no experts) | torch path | 2.88e-2 | 2.77e-2 (2.1–3.5e-2) | 0/10 |
| dense | fla | 2.88e-2 | 2.84e-2 (2.3–3.9e-2) | 0/10 |
| dense | fla + causal-conv1d | 2.88e-2 | 2.73e-2 (2.1–3.8e-2) | 0/10 |
| MoE (the test's: 4 experts, top-2) | torch path | 6.71e-2 | 6.49e-2 (1.9e-2–2.0e-1) | 5/10 |
| MoE | fla | 6.71e-2 | 5.03e-2 (2.2e-2–1.9e-1) | 4/10 |
| MoE | fla + causal-conv1d | 6.71e-2 | 2.76e-2 (2.0e-2–1.7e-1) | 3/10 |

Seed 0 across the arms:

| | torch path | fla | fla + causal-conv1d |
|---|---:|---:|---:|
| A2000, stand-in attention | 2.7e-2 | 7.2e-2 | 2.3e-2 |
| 5090, real fp8 kernel | 2.6e-2 | 7.5e-2 | 8.2e-2 |

Batched against one-row decode, and chunk-matched against single-call references, change the medians little
(`probe7.py`).

**Reading:**
- **Without experts the kernels are indistinguishable from the torch path,** and the hybrid sits at its control's
  error on every seed.
- **With the test's tiny MoE, the statistic is bimodal in every arm.** On the torch path, 5 of 10 seeds fail the test's
  own bound (2× the control's error). This is consistent with routing near-ties: a numerically negligible difference
  flips one token's top-2 expert choice and moves the worst-case logit, while greedy tokens stay equal.
- **The pinned premise** (`tests/test_linear_state_gpu.py`, and its chunk-matched copy) passes on the torch path
  because seed 0 lands in the low mode. fla moves seed 0 to the high mode. That is the whole of P103's and P104's
  premise failures. **Neither the kernels nor e4b are at fault.**
- **`tests/test_linear_state_gpu.py` stays byte-for-byte** (six lanes pin it). Its pass is one draw, not a
  distribution. A lane that wants a numeric parity premise for the kernels needs a dense hybrid or a seed
  distribution.

## What follows

- **`docs/SERVING.md`** records the kernels' state with the corrected cause. `bench/p103/RESULTS-p103.md` carries a
  dated correction of its withdrawn attribution.
- **Not registered:** a kernel speed lane whose premise is a distribution (a dense fp8 hybrid over many seeds, plus
  greedy-token equality on the MoE). Indicative A2000 timings (#928, `bench/p103/a2000/probe6.py`) put fla's decode
  step at about 6× the torch path per layer at 16 rows.

Receipts (`receipts/p104-prove-1/`):
- `summary.txt`, `forensics.txt`, `versions.txt` and the three engagement records;
- the premise, single-call and kernel logs;
- the teardown proof;
- `SHA256SUMS` over every file the box wrote.

The launcher's receipt and ledger row are in the receipt store (adertha-receipts `d489322`). **The lane: $0.0527.**
