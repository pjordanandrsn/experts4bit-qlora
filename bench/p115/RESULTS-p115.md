# P115 — results: Phases A and B **DEFAULT_AUTO** (the registered B=1 fused stack decodes the default `serve_paged` server 1.43× as fast with one request and 1.23× with 16, at no measurable quality cost; Qwen3-30B-A3B NF4); Phase C **FLIP_HELD** (gpt-oss-20b fails the SANE gate on argmax agreement), with Granite **GRANITE_LICENSED**. One RTX 5090

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

## Phase C (`p115c-5090-1`): **FLIP_HELD**; Granite **GRANITE_LICENSED**

Registration: Amendment 2 (#1318, `32e1eaf6`), Phase C's scripts, the SANE implementation and Granite's full read. The
proof `p115c-prove-1` passed first.

Code under test:
- e4b 0.48.0 at `32e1eaf6` (Amendment 2 merged);
- grouped-nf4-gemm 0.42.0 at `b4f93f1`, Phase A/B's pin, which isolates the fusion knobs from #509 / #508 in 0.43.0;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `openai/gpt-oss-20b` at `6cee5e8` (SC2g's e4b path), `Qwen/Qwen3.6-35B-A3B` at `995ad96` (P98's arena) and
  `ibm-granite/granite-3.1-3b-a800m-instruct` at `a027806`, each baked on the box.

**Verdicts by `p115c_reduce.py`:**
- the flip verdict over gpt-oss-20b and Qwen3.6-35B-A3B is **`FLIP_HELD`**;
- Granite's own verdict, which does not enter the flip verdict, is **`GRANITE_LICENSED`**.

| model | census under `auto` (q/k/v / glue / r2 / router) | `=1` raised | SANE (12 windows × 128 positions) | ON ≡ OFF rows (reported) | gates |
|---|---|---|---|---|---|
| gpt-oss-20b | 0 / 49 / [24, 0] / 24 | "E4B_PAGED_FUSE_QKV=1 matched no attention module" | bias **+0.0022** nats, argmax agreement **0.924**, mean KL 0.030 | 2 / 16 | SERVED, ENGAGED, EXPLICIT_RAISE, DETERMINISM pass; **SANE fails** (argmax < 0.95) |
| Qwen3.6-35B-A3B | 0 / 0 / [0, 0] / 40 (101 RMSNorms name-matched and failed the semantic probe) | "E4B_FUSE_T1_GLUE=1 patched no RMSNorm modules" | bias +0.0004, argmax 0.995, mean KL 0.0005 | 16 / 16 | all pass |
| Granite-3.1-3b-a800m | 0 / 65 / [32, 32] / 32 | "E4B_PAGED_FUSE_QKV=1 matched no attention module" | n/a: Phase B's read instead (below) | 3 / 16 | all pass; QUALITY pass |

OFF's census was all zero on every model. Every OFF and ON serve build captured every bucket (1–16), and every `=1` build raised before serving, as registered.

**Granite's quality read** used Phase B's instrument at Phase B's size: 48 windows × 128 positions per text, with the
floor drawn from `half` and `chunk`.

| text | ON bias (bar) | ON spread (bar) | K8 perplexity | argmax agreement | floor B / S | mutant |
|---|---|---|---|---|---|---|
| wikitext-2 | −0.00268 nats (0.01204) | 0.0111 (0.0203) | 6.4027 → 6.3856 (−0.0171, gated, budget 0.05) | 0.963 | 0.0020 / 0.0102 | +2.84 nats (fails the bar, as it must) |
| c4val1 | +0.00355 (0.01400) | 0.0119 (0.0253) | 11.7045 → 11.7462 (+0.0417, reported, not gated) | 0.960 | 0.0040 / 0.0127 | +1.97 nats (fails the bar) |

### The reading (`p115c-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 595.84) on an AMD EPYC 7B13 host (256 threads, ~2 TB RAM), 320 GB free. It was
Vast instance 54770431 ("verified-secure"). **Cost:** $1.345, about 44 minutes from start to teardown.

**Timeline (the box's own log, UTC):**
- the premise passed by 03:48 (48 passed, none skipped);
- Granite: fetch 03:49, three serve builds 03:50–03:51, quality OFF 359 s, quality ON 70 s, by 03:58;
- gpt-oss-20b: fetch and bake by 04:02, serve and SANE builds 04:02–04:08;
- Qwen3.6-35B-A3B: fetch by 04:20, bake 04:22, serve and SANE builds 04:23–04:27;
- TP_DONE at 04:27:48.

### Against the predictions

| prediction | result |
|---|---|
| census gpt-oss `0 / 49 / [24, 0] / 24`, Qwen3.6 `0 / 0 / [0, 0] / 40`, Granite `0 / 65 / [32, 32] / 32` | **exact**, all three |
| EXPLICIT raises a knob's own vacuous-enable refusal (Amendment 2's correction: Qwen3.6 refuses the RMSNorm fold before the q/k/v fusion is reached) | **as corrected**: gpt-oss and Granite refuse the q/k/v fusion, Qwen3.6 the RMSNorm fold |
| FLIP_LICENSED (the launch's stated expectation) | **missed**: gpt-oss's argmax agreement was 0.924 |
| Granite: wikitext ON bias within ±0.004 nats and \|ΔK8\| ≤ 0.03; c4val1 bias in [0.000, +0.015]; GRANITE_LICENSED about 60 % | **all hold**: −0.00268, 0.0171, +0.00355 |
| reading about 1.5 h, about $2.7 | 44 minutes, $1.345 |

**Reading the gpt-oss failure (context, not a rule).**
- On gpt-oss the folds left the mean NLL almost where it was (+0.0022 nats against a 0.02 gate). They moved the
  top-1 token on 7.6 % of teacher-forced positions, and the mean KL is 0.030. On Granite the same three folds read
  0.963 / 0.008, about the same as that model's neutral perturbations: half batch 0.962, prefill chunk 0.962.
- SANE draws no neutral floor of its own, so this run cannot say whether 0.924 is gpt-oss's own sensitivity to any
  bf16 reordering or one fold's arithmetic. The rule does not ask; it held the flip.

### The registered consequence (FLIP_HELD)

- The four knobs **stay `0` by default**. As registered, any default now is "a family-scoped default only under a new
  registration".
- The families with a registered passing read under `auto` are:
  - Qwen3-30B-A3B: speed and quality, Phases A and B;
  - Qwen3.6-35B-A3B: Phase C's gates including SANE; only the router epilogue engages, and it was token-identical;
  - Granite-3.1-3b-a800m: GRANITE_LICENSED.
- **gpt-oss-20b has none.**
- This is the maintainer's allowlist ("under `auto`, a fold applies only on a family with a registered reading").
  The new registration that makes it a default also carries the combined read the maintainer asked for in #1318's
  review: one SANE pair on Qwen3-30B-A3B at the 0.43.0 kernels, fusion `0` against `auto`, the bandwidth GEMV (P116)
  at its default.
- **gpt-oss** gets a follow-up before any default reaches it: one knob per arm, with a neutral floor drawn on gpt-oss
  itself.

This read registers two rows:
- `e4b.serve.p115.fused-stack-engagement.gptoss-qwen36.5090.2026-10-08`: engagement and SANE under `auto` on the two
  families, FLIP_HELD. Its value is gpt-oss's argmax agreement, 0.924.
- `e4b.serve.p115.fused-stack-quality.granite.5090.2026-10-08`: Granite's wikitext ON bias, with c4val1 and the K8s in
  the claim text.

### What Phase C took

| run | status | cost | note |
|---|---|---:|---|
| `p115c-prove-1` | OK, PROVED | $0.158 | Granite end to end at proof sizes; premise passed; its verdict (not a reading) FLIP_LICENSED |
| `p115c-5090-1` | **OK, FLIP_HELD; GRANITE_LICENSED** | $1.345 | the reading |

**Phase C cost $1.503**, inside its $5.00 ceiling. **P115 has cost $2.455** over all phases, against its $10 hard stop.

**Receipts** are in `receipts/p115c-5090-1/`, with `SHA256SUMS`:
- the nine serve records, the four SANE records, `quality_granite_off.json`, `quality_granite_on.json`, `verdict_c.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, the three prompt files and the three bakes;
- the logs and the teardown proof.

The launcher's receipts and ledger rows are in the receipt store. The reading's is adertha-receipts `58f25d67`, and the
proof's is `07b05b78`.
