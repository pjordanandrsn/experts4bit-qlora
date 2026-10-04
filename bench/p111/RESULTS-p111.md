# P111 — results: **DEFAULT_ON**. One KV-table selection per decode step decodes the default `serve_paged` server's tokens exactly, and 3.6 % faster with 16 concurrent requests (772 → 800 tok/s; 0.72 ms of the 20.7 ms step, as SC1b's census predicted), 1.2 % with one

Registration: `bench/p111/PREREG-p111.md` (#1000, `7abf9b1`). The switch: #999 (`e50caaa`). Issue: #1001.

Code under test:
- e4b 0.43.0 plus #999 and this registration, at `7abf9b1`;
- grouped-nf4-gemm 0.35.0 at `51a4916`;
- transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, NF4 arena baked on the box.

**Verdict by `p111_reduce.py`: `DEFAULT_ON`.** The rule's steps, in order:

| step | result |
|---|---|
| VOID | no. Commits, revision, prompts and lengths agree; every arm captured every bucket (1–16); the switch was in force in S1a and S1b and off in S0a and S0b |
| NOISY | no. Self-pairs S0b/S0a 0.999 (W16) and 0.997 (W1), S1b/S1a 1.002 and 1.001 |
| FUNCTION_FAIL | no. **S1a, S1b and S0b emit S0a's tokens on every row** of both workloads at 32 and 160 tokens, and every timed rep digests the same |
| SLOWER | no. **g16 = 1.0352**, g1 = 1.0096 (bar 1.00) |

## The reading (`p111-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 610.57.04) on an AMD Ryzen 9 9950X3D (32 threads, 123 GiB RAM), Vast instance
54083342 at $0.576/h. **Cost:** $0.1586. Teardown was proven at 00:42:26Z.

**Timeline (the box's own log):**
- install at 00:26:49Z;
- premise 13 passed, none skipped (the bucket tests and the switch's GPU exactness tests, on the card);
- fetch 00:27:20–00:36:15Z;
- bake and prompts until 00:37:01Z;
- four arms 00:37:01–00:41:43Z;
- reduce at 00:41:43Z.

| arm | `E4B_KV_STEP_SELECT` | W16 decode tok/s | W1 decode tok/s | W16 ms/step | W1 ms/step | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| S0a | off | 772.40 | 109.41 | 20.715 | 9.140 | 21.560 |
| S1a | **on** | 799.61 | 110.46 | 20.010 | 9.053 | 21.561 |
| S1b | **on** | 800.87 | 110.62 | 19.978 | 9.040 | 21.561 |
| S0b | off | 771.80 | 109.11 | 20.731 | 9.165 | 21.560 |

- **16 concurrent requests:** the pair ratios are 1.0352 and 1.0377, geometric mean **1.036**. The step falls from 20.72
  to 19.99 ms. That 0.72 ms is SC1b's census estimate (about 0.8 ms of per-layer `index_select` and `index_add_`) less the
  four launches the step adds.
- **One request:** 1.0096 and 1.0138, geometric mean **1.012**. The per-layer selection at bucket 1 costs less: 0.10 ms
  of a 9.1 ms step.
- **Exact:** the switch changes no token, as the kernels see the same values.
- **Memory:** unchanged (+1 MiB, the bucket buffers).

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: every arm captures every bucket, and S1's switch is in force | **yes** |
| Q2: S1 ≡ S0 bitwise on every row | **yes** |
| Q3: g16 between 1.02 and 1.06 | **yes** (1.0352) |
| Q4: g1 between 1.02 and 1.10 | **no: 1.0096**, below the band. At bucket 1 the selection costs 0.10 ms, less than the 0.2–0.9 ms the band assumed |
| Q5: self-pairs within [0.98, 1.02] | **yes** (0.997–1.002) |
| Q6: DEFAULT_ON | **yes** |

## The registered consequence (DEFAULT_ON)

- **The default.** `E4B_KV_STEP_SELECT` defaults on in this PR; `0` keeps the per-layer selection.
- **The docs.** `docs/SERVING.md` and the CHANGELOG say so, with the gain.
- **The register.** The row `e4b.serve.p111.kv-step-select.qwen3.5090.2026-10-04` carries g16, from the verdict file.
- **The issue.** #1001 closes.
- **Scope.** Read on Qwen3-30B-A3B NF4 through the graph default. The selection is the same code for every family, and
  eager decode does not use it.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p111-prove-1` | OK, PROVED | $0.0373 | Granite; premise 13 passed on the card; its verdict (not a reading) DEFAULT_ON: identical tokens, W16 +4.8 % / +9.2 % |
| `p111-5090-1` | **OK, DEFAULT_ON** | $0.1586 | the reading |

**The lane cost $0.1959**, inside its $2.50 ceiling.

**Receipts** are in `receipts/p111-5090-1/`, with `SHA256SUMS`:
- the four arm receipts, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- the logs and the teardown proof.

The launcher's receipts and ledger rows are in the receipt store, adertha-receipts `5c83e40` (`p111-prove-1`) and
`42e459d` (`p111-5090-1`).
