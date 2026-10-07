# P115 — results (Phases A and B): **DEFAULT_AUTO**. The registered B=1 fused stack decodes the default `serve_paged` server 1.43× as fast with one request and 1.23× with 16, at no measurable quality cost (Qwen3-30B-A3B NF4, one RTX 5090)

Registration: `bench/p115/PREREG-p115.md` (#1314, `98f919de`; Amendment 1 in #1317, `598c0013`). The `auto` semantics:
#1315. Issue: #1313.

Code under test:
- e4b 0.48.0 at `598c0013`: the registration, #1315 and #1317;
- grouped-nf4-gemm 0.42.0 at `b4f93f1`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, with the NF4 arena baked on the box.

**Verdict by `p115_reduce.py`: `DEFAULT_AUTO`.** The rule's steps, in order:

| step | result |
|---|---|
| VOID | no. Commits, revision, prompts and lengths agree, and every arm captured every bucket (1–16). F1's census is the registered `48 / 193 / [48, 48] / 48`, and F0's is zero. Every quality pass engaged exactly: 127 × 48 decode attention calls, the glue kernels' per-step counts (49 / 48 / 96 / 48) on ON and none on R, and 48 `qkv_proj` calls per forward on ON. R repeated bit for bit on both texts, and the scale mutant fails the bar. |
| NOISY | no. Self-pairs F0b/F0a 1.0014 (W16) and 0.9999 (W1); F1b/F1a 1.0003 and 1.0009 |
| FUNCTION_FAIL | no. F0b emits F0a's tokens and F1b emits F1a's on every row of both workloads at 32 and 160 tokens, and every timed rep digests the same |
| QUALITY_FAIL | no. Wikitext ON bias +0.00098 nats (bar 0.0117), spread 0.0120 (bar 0.0277), K8 +0.0088 ppl (budget 0.05). c4val1 ON bias −0.00256 (bar 0.0119), spread 0.0142 (bar 0.0270); its K8 of −0.042 is reported, not gated |
| SLOWER | no. **g1 = 1.4289** (bar 1.10), **g16 = 1.2298** (bar 1.00) |

## The reading (`p115-5090-8`)

**Host:** one RTX 5090 (sm_120, driver 610.57.04) on an AMD engineering-sample CPU (100-000000897-03, 64 threads,
251 GiB RAM). It was Vast instance 54698460 on machine 13828, at $0.736/h. **Cost:** $0.496.

**Timeline (the box's own log):**
- install at 19:09:47Z;
- premise 14 passed, none skipped, by 19:10:50Z;
- fetch 19:10:43–19:14:51Z;
- bake and prompts until 19:16:04Z;
- four arms 19:16:04–19:21:31Z;
- quality OFF 19:21:31–19:29:45Z and ON until 19:31:42Z;
- reduce at 19:31:42Z.

| arm | the four knobs | W16 decode tok/s | W1 decode tok/s | W16 ms/step | W1 ms/step | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| F0a | unset (the default) | 701.10 | 94.15 | 22.821 | 10.622 | 22.17 |
| F1a | **all `1`** | 863.17 | 134.53 | 18.536 | 7.433 | 22.56 |
| F1b | **all `1`** | 863.44 | 134.65 | 18.531 | 7.427 | 22.56 |
| F0b | unset | 702.10 | 94.14 | 22.789 | 10.623 | 22.17 |

- **One request:** the pair ratios are 1.4289 and 1.4303, geometric mean **1.430**. The step falls from 10.62 to
  7.43 ms.
- **16 requests:** 1.2312 and 1.2298, geometric mean **1.231**. The step falls from 22.82 to 18.53 ms.
- **Tokens:**
  - F1 ≠ F0 at W16 on 7 of 16 rows at 32 tokens and on 15 of 16 at 160 tokens, as expected: the folds change bf16
    rounding.
  - At W1, the single row matched F0 at both lengths.
- **Memory:** +0.39 GiB at peak (22.17 → 22.56 GiB).
- **This host's default step** read 10.62 ms at W1, slower than P111's 9.05–9.14 ms on a Ryzen 9 9950X3D. The registered
  quantity is the ratio within one box.

**Quality** (P110's teacher-forced instrument, 48 windows × 128 positions per text, R = the graph server's arithmetic):

| text | B_floor (half, chunk) | S_floor | ON bias (nats) | ON spread | K8: ppl R → ON | argmax agree | mutant bias |
|---|---:|---:|---:|---:|---|---:|---:|
| wikitext | 0.0017 | 0.0139 | **+0.00098** (SE 0.0021) | 0.0120 | 8.9710 → 8.9798 (+0.0088) | 0.961 | +1.054 |
| c4val1 | 0.0019 | 0.0135 | **−0.00256** (SE 0.0031) | 0.0142 | 16.403 → 16.361 (−0.042, reported) | 0.954 | +0.775 |

ON sits inside the neutral perturbations' own band on both texts. The half-batch and prefill-chunk draws moved the NLL by
−0.0013 / +0.0017 (wikitext) and −0.0006 / −0.0019 (c4val1). The proof's c4val1 hint on Granite (+0.0151 nats on 12 × 32)
is not a Qwen3 reading. Granite gets its own full read in Phase C (Amendment 2, #1318).

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: every arm captures every bucket with the fused stack; engagement exact in every pass | **yes** |
| Q2: F0b ≡ F0a and F1b ≡ F1a bitwise; F1 ≠ F0 on some rows | **yes** (W16 differs on 7–15 of 16 rows; the W1 row matched) |
| Q3: g1 ∈ [1.25, 1.60]; F1's W1 step 5.8–7.2 ms | **g1 yes** (1.4289). **The step missed**: 7.43 ms, because this host's default step was 10.62 ms, not P111's 9.05 |
| Q4: g16 ∈ [1.03, 1.20] | **no: 1.2298**, above the band |
| Q5: self-pairs within [0.98, 1.02] | **yes** (0.9999–1.0014) |
| Q6: wikitext bias ±0.004, spread ≤ 0.015, \|ΔK8\| ≤ 0.03; c4val1 bias ±0.006, \|ΔK8\| ≤ 0.10; floors B ≤ 0.003, S ≤ 0.015; mutant > +0.3 | **yes**, every one |
| Q7: DEFAULT_AUTO | **yes** |
| Q8: each speed arm ≤ 3 min; quality ≤ 25 min; peak ≤ 23 GiB | **yes** (about 1.3 min per arm; 10.2 min; 22.56 GiB) |

## The registered consequence (DEFAULT_AUTO)

The default does **not** move here. As registered, the flip waits for Phase C's engagement read under `auto` on
gpt-oss-20b and Qwen3.6-35B-A3B, plus Granite's full quality read added in review (Amendment 2, #1318). It is a separate
pull request.

The maintainer added a requirement for that PR in review: under `auto`, a fold applies only on a family with a
registered reading. Unread families stay unfused unless set to `1`.

This read registers two rows:
- `e4b.serve.p115.fused-stack-speed.qwen3.5090.2026-10-07`: g1, the W1 ratio, with g16 in the claim text;
- `e4b.serve.p115.fused-stack-quality.qwen3.5090.2026-10-07`: wikitext ON bias, with c4val1 and the K8s in the claim text.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p115-prove-1` | REFUSED | $0 | the launcher's live provider was not armed (`E4B_RENT_LIVE` unset) |
| `p115-prove-2` | OK, PROVED | $0.172 | Granite; premise 14 passed; its verdict (not a reading) DEFAULT_AUTO |
| `p115-5090-1` | HARNESS_ERROR | $0.044 | Vast machine 34887: the image's torch could not use the GPU (rc 10; #1317 made this exit 18) |
| `p115-5090-2` | REFUSED | $0 | the launcher timed out reading `origin/main` over HTTPS |
| `p115-5090-3` | NOT_RUN | $0.035 | ssh refused for 180 s (machine 140083) |
| `p115-5090-4` | NOT_RUN | $0.014 | stuck loading for 600 s (machine 151044) |
| `p115-5090-5` | NOT_RUN | $0.032 | ssh refused for 180 s (machine 153193) |
| `p115-5090-6` | NOT_RUN | $0.036 | ssh refused for 180 s (machine 36544) |
| `p115-5090-7` | NOT_RUN | $0.123 | HF CDN 4.7 MB/s < 20 MB/s (machine 36544) |
| `p115-5090-8` | **OK, DEFAULT_AUTO** | $0.496 | the reading; 140083 and 36544 excluded, ssh-ready 300 s |

**Phases A and B cost $0.952**, inside their $5.00 ceiling. Draws 1–7 failed before any lane work, on the host or the
launcher, and none of them is a reading.

**Receipts** are in `receipts/p115-5090-8/`, with `SHA256SUMS`:
- the four arm records, `quality_off.json`, `quality_on.json`, `verdict.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- the arm, quality, premise, bake and prompt logs, and the teardown proof.

The launcher's receipts and ledger rows are in the receipt store. This reading's is adertha-receipts `fc99e88f`, and
`p115-prove-2`'s is `47eac5a4`.
