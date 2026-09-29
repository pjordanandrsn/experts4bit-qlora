# B771b — results: the fixed bucketed-graph path decodes **exactly** as the eager runner, and beats it 4.79–4.81× on this host (**CONFIRMED**)

Read 2026-09-29 from `b771b-5090-1`. Registration: `bench/b771b/PREREG-b771b.md` (#779, `46d6991`). Issues and fixes: #771,
#777, grouped-nf4-gemm#413. Verdict by `b771b_reduce.py`: **CONFIRMED**. T, M, I and R all held.

## Run and cost

- `b771b-5090-1`: one RTX 5090 (sm_120, driver 570.169), Vast verified/secure, AMD EPYC 7K62 (Zen 2).
- **Cost $0.2475**, a 1,749 s lifetime against a 1.0 h guard. Lane rc 0; teardown proven (`vast-destroy` HTTP 200, instance
  absent afterwards).
- e4b at `46d6991` (a `main` with #777), installed editable from the clone under torch 2.8.0+cu128 / triton 3.4.0 constraints;
  grouped-nf4-gemm **v0.33.7** (`9407d49`); transformers 5.16.1, bitsandbytes 0.50.1.

## Arm T and arm M (T: held; M: held)

| arm | suite | result |
|---|---|---|
| T | e4b `test_decode_graph_buckets.py` + `test_bucket1_append_routing.py` | **10 passed**, none skipped. Includes `test_every_bucket_step_advances_its_rows_own_kv_length[graph]` and `[padded-eager]`, their first run on sm_89+. |
| T | gnf4 `kernel/test_fp8_kv_append.py` | **19 passed**, none skipped: every byte gate on the hardware cast, the fp32 group-math gate, and bt1 ≡ the t1 loop. |
| M | the invariant test under `mut_b771b.py` (the old single-row routing) | **2 failed**, both modes, as registered. |

#777's fix is verified on the hardware path. The invariant holds after every bucketed step, and breaks exactly when the old
routing is restored.

## The decode stage (I: held; R: held)

NF4 Qwen3-30B-A3B, P80's prompts and trace (16, 8, 4, 2, 1 active rows for 32 decode steps each).

| arm | what | tok/s (timed pass) |
|---|---|---:|
| A1 | eager, the engine's default grouping (P80's control) | 55.6 |
| B1 | bucketed graphs | 267.5 |
| B2 | bucketed graphs | 267.5 |
| A2 | eager, default grouping | 55.8 |
| P | the bucket step, eager | 67.4 |
| Ad | eager, the device grouping | 59.3 |

- **I, the correctness claim.** **Ad ≡ P ≡ B1 ≡ B2, bitwise, in all 16 rows.** With #777 and 0.33.7's append, the graph path
  decodes exactly the eager runner's function under the same grouping. B771 found this pair leaving each other at the first
  one-row step; now they don't.
- **R, P80's claim on the fixed path.** B1/A1 = **4.8148** and B2/A2 = **4.7914**, both above 1.03. The self-pairs are A2/A1 =
  1.0048 and B2/B1 = 0.9999.
- **Reported, not decisive.** A1 vs Ad first differ at row 2, token 66: the default and the device grouping are different
  kernel sets. Ad/A1 = 1.0665, P/A1 = 1.2125, B1/P = 3.9710.

Mean ms per decode step, timed pass:

| active rows | A1 | Ad | P | B1 | A1 / B1 | P80's B1 | P80's A1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 117.39 | 107.77 | 97.38 | 37.22 | 3.15× | 33.79 | 54.54 |
| 8 | 115.32 | 106.86 | 95.44 | 27.03 | 4.27× | 25.88 | 53.73 |
| 4 | 114.21 | 106.69 | 93.84 | 21.46 | 5.32× | 20.46 | 52.90 |
| 2 | 111.98 | 105.13 | 91.06 | 19.65 | 5.70× | 18.90 | 51.54 |
| 1 | 99.00 | 96.68 | 82.39 | 10.52 | 9.41× | 9.95 | 42.31 |

- **Why ×4.8 here and ×2.3 in P80.** The host. The graph step, which is GPU work, is 4–10% above P80's at every row count.
  The eager step, which is host-bound, is 2.15–2.34× slower on this Zen 2 EPYC 7K62 than on P80's Zen 5 EPYC 9755. So the
  ratio is not portable, as with P80 and P81. What is portable is the per-step graph cost and the direction of the ratio.
- **The one-row phase.** B1's one-row step is 10.52 ms against P80's 9.95. The fixed path attends over a context that keeps
  growing, where the buggy path's froze. That is the direction the registration expected, but it is not isolated from host
  differences.

## What this establishes

- The bucketed CUDA-graph decode (`enable_decode_graphs`) with #777, on grouped-nf4-gemm ≥ 0.33.7, decodes bit-identically to the
  eager `PagedModelRunner` with the device grouping, on real weights, over a trace that includes a one-row phase.
- #771 is answered: the eager and bucket steps differed for two reasons, grouped-nf4-gemm#413's append and #777's bucket-1
  append, and with both fixed they are identical.
- P80's claim holds on the fixed path. This row supersedes P80's in the register.
- **Not established.** Anything about the int4 stack (P81's row stays corrected and not superseded until that stack is
  re-measured), about other families or arrivals, or about quality.

## Receipts

- **In this repo:** [`receipts/b771b-5090-1/`](receipts/b771b-5090-1/): both junit files and their text, the six arm receipts,
  `verdict.json`, `summary.txt`, `versions.txt`, `forensics.txt` and `constraints.txt`, all byte-identical to the store's.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-09-29/b771b-5090-1/` (commit `8a96b30`), with the
  launcher's `receipt.json` and `teardown-proof.json`, and `logs/`.
