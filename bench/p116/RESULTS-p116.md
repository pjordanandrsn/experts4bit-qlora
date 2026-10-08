# P116 — results: **DEFAULT_ON**. grouped-nf4-gemm K33's NF4 decode GEMV (`GNF4_GEMV_BW=1`) decodes the default `serve_paged` server 1.24× as fast with one request, 16 requests unchanged, within P110's quality bar (Qwen3-30B-A3B NF4, one RTX 5090)

Registration: `bench/p116/PREREG-p116.md` (#1322, `6746205e`; Amendment 1 in #1329, `730c8acc`). The kernel:
grouped-nf4-gemm#500, read as LEVER by K33 (grouped-nf4-gemm#502). Issue: #1313.

Code under test:
- e4b 0.48.0 at `730c8acc`: the registration and Amendment 1;
- grouped-nf4-gemm 0.42.0 + K33 at `5a60c37`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, with the NF4 arena baked on the box.

**Verdict by `p116_reduce.py`: `DEFAULT_ON`.** The rule's steps, in order:

| step | result |
|---|---|
| VOID | no. Commits, revision, prompts, lengths and fusion census agree, and every arm captured every bucket (1–16). **Engagement exact:** B1 dispatched every single-row decode GEMV to `bw_prmt32` (288 at capture) and nothing to dot-pad, the scalar GEMV, the tree or split-K. B0 dispatched only dot-pad (288). The quality phases engaged at T == 1 the same way. Every one-window pass ran 127 eager bucket-1 steps with 127 × 48 decode attention calls and no replay. The scale mutant fails the bar. |
| NOISY | no. W1 self-pairs B0b/B0a 1.0019 and B1b/B1a 1.0007 (band [0.985, 1.015]); W16 0.9994 and 1.0025 (band [0.96, 1.04]) |
| FUNCTION_FAIL | no. B0b emits B0a's tokens and B1b emits B1a's on every row of both workloads at 32 and 160 tokens, and every timed rep digests the same |
| QUALITY_FAIL | no. Wikitext ON bias −0.0012 nats (bar 0.0118), spread 0.0127 (bar 0.0299), K8 −0.0104 ppl (budget 0.05). c4val1 ON bias −0.0053 (bar 0.0116), spread 0.0126 (bar 0.0216); its K8 of −0.090 is reported, not gated |
| SLOWER | no. **g1 = 1.2417** (bar 1.03), **g16 = 1.0003** (bar 0.99) |

## The reading (`p116-5090-2`)

**Host:** one RTX 5090 (sm_120, driver 580.126.09, 575 W) on an AMD host with 128 threads. It was Vast instance 54737263
on machine 55913, at $0.565/h. **Cost:** $1.078.

**Timeline (the box's own log):**
- install at 23:46:55Z;
- premise by 23:51Z (the kernel's contract 27 passed, the served route 18 passed, none skipped);
- fetch 23:50–23:56Z;
- bake and prompts until 23:57:34Z;
- four arms 23:57:34–00:03:14Z;
- quality OFF 00:03:14–00:29:09Z and ON until 00:38:39Z;
- reduce at 00:38:39Z.

| arm | `GNF4_GEMV_BW` | W16 decode tok/s | W1 decode tok/s | W16 ms/step | W1 ms/step | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| B0a | unset (dot-pad) | 761.42 | 97.89 | 21.013 | 10.216 | 22.17 |
| B1a | **`1`, K33's plans** | 761.66 | 121.70 | 21.007 | 8.217 | 22.17 |
| B1b | **`1`, K33's plans** | 763.58 | 121.79 | 20.954 | 8.211 | 22.17 |
| B0b | unset | 760.97 | 98.08 | 21.026 | 10.196 | 22.17 |

- **One request:** the pair ratios are 1.2432 and 1.2417, geometric mean **1.243**. The step falls from 10.21 to 8.21 ms,
  2.0 ms. K33's microbenchmark put the expert-GEMV saving at about 1.8 ms per step with PDL as served, so the whole kernel
  saving reached the served step and a little more. K6b's 0.55 transfer did not apply here.
- **16 requests:** 1.0003 and 1.0034, unchanged as registered. A T > 1 step takes device grouping and never reaches the
  switch.
- **Tokens:** B1 emitted B0's tokens on every row of both workloads at both lengths. At W16 that was predicted. At W1 the
  single row also matched, where Q2 expected a difference somewhere.
- **Memory:** unchanged.

**Quality** (P110's teacher-forced instrument at one window per pass, 24 windows × 128 positions per text, R = the
default server's arithmetic at T == 1):

| text | B_floor (chunk) | S_floor | ON bias (nats) | ON spread | K8: ppl R → ON | argmax agree | mutant bias |
|---|---:|---:|---:|---:|---|---:|---:|
| wikitext | 0.0018 | 0.0150 | **−0.0012** (SE 0.0032) | 0.0127 | 8.6774 → 8.6670 (−0.0104) | 0.964 | +1.010 |
| c4val1 | 0.0016 | 0.0108 | **−0.0053** (SE 0.0029) | 0.0126 | 16.979 → 16.889 (−0.090, reported) | 0.958 | +0.848 |

R repeated bit for bit on both texts, so the floor is `chunk` alone. ON reads slightly lower NLL than dot-pad on both
texts. The bars are one-sided upward (bias_ON ≤ B_floor + 0.01), so a lower NLL never fails them.

## The second draw (`p116-5090-1`), reported, not the reading

`p116-5090-1` ran the same box from `c3cd1c56` and VOIDed on the reducer's window table (Amendment 1). As registered,
its records were reduced by the corrected reducer (`730c8acc`) and are reported here as a labelled second draw. No rule
looks at it.

| | W1 B0 → B1 tok/s | g1 | g16 | W1 self-pairs | wikitext ON bias | c4val1 ON bias |
|---|---|---:|---:|---|---:|---:|
| `p116-5090-1` (machine with 96 threads, driver 595.91.07) | 112.86 → 143.55 | 1.2681 | 0.9980 | 1.0046 / 1.0016 | −0.0012 | −0.0053 |
| `p116-5090-2` (the reading) | 97.89 → 121.70 | 1.2417 | 1.0003 | 1.0019 / 1.0007 | −0.0012 | −0.0053 |

- **Speed** reproduces on a second host: the absolute rates differ by 15 %, and the ratio holds within 2 %.
- **Quality** reproduces exactly. Every quality number agrees between the two hosts to every printed digit. Both were
  RTX 5090s on the same software, with deterministic kernels throughout, as P115's bit-for-bit R repeat also shows.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: engagement exact; every bucket captured | **yes** |
| Q2: B0b ≡ B0a, B1b ≡ B1a bitwise; B1 ≡ B0 at W16; B1 ≠ B0 on some W1 rows | **partly.** The first three held. **The W1 row matched B0** at both lengths |
| Q3: g1 ∈ [1.06, 1.22] | **no: 1.2417**, above the band. The full kernel saving reached the step, where the band assumed K6b's 0.55 transfer |
| Q4: g16 ∈ [0.99, 1.01] | **yes** (1.0003) |
| Q5: self-pairs within [0.98, 1.02] | **yes** (0.9994–1.0025) |
| Q6: wikitext ±0.003 and \|K8 Δ\| ≤ 0.02; c4val1 ±0.005; chunk floor ≤ 0.003; mutant > +0.3 | **all but one.** c4val1's −0.0053 is 0.0003 past ±0.005, inside the bar. The rest held |
| Q7: DEFAULT_ON | **yes** |
| Q8: arms ≤ 3 min; quality OFF ≤ 25 min, ON ≤ 10 min; peak ≤ 23 GiB | **all but one.** OFF took 25.9 min; arms about 1.4 min, ON 9.5 min, peak 22.17 GiB |

## The registered consequence (DEFAULT_ON)

- **grouped-nf4-gemm** fills `_BW_SHAPES` with Qwen3-30B-A3B's two shapes, writes K33's plans in as a shape table, and
  makes `GNF4_GEMV_BW=auto` its default on ≥ 160-SM parts, with the `test_m3_defaults` trio and a release.
- **experts4bit-qlora** floors `[fast]` on that release, adds a `compatibility` record to `docs/system-manifest.json`, and
  updates `docs/SERVING.md`.
- **The register** row is `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07` (g1).
- **Scope:** speed and quality are read on Qwen3-30B-A3B NF4 at one request. Granite and OLMoE run the scalar GEMV at B=1;
  K33 read 0.22–0.27× there, and the Granite proof read W1 ×1.39 (not a reading). Their served read is its own lane.

**Not measured:** P116 and P115's fused stack together. Each was read alone against the same default: one-request saving
2.0 ms here and 3.2 ms there, on 10.2–10.6 ms steps.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p116-prove-1` | OK, PROVED | $0.211 | Granite; premise 27 + 18 passed; verdict (not a reading) DEFAULT_ON, W1 ×1.39 |
| `p116-5090-1` | OK, VOID | $0.853 | the reducer expected 48 windows per text, against the registered 24 (Amendment 1); the second draw above |
| `p116-5090-2` | **OK, DEFAULT_ON** | $1.078 | the reading |

**The lane cost $2.142**, inside its $3.00 ceiling.

**Receipts** are in `receipts/p116-5090-2/` (the reading) and `receipts/p116-5090-1/` (the second draw, with the box's VOID
`verdict.json` and the corrected `verdict_corrected.json`), each with `SHA256SUMS`:
- the four arm records, `quality_off.json`, `quality_on.json`, `verdict.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- the logs and the teardown proof.

The logs keep the box's own carriage returns. The launcher's receipts and ledger rows are in the receipt store:
adertha-receipts `831f2655` (`p116-prove-1`), `683ea367` (`p116-5090-1`) and `fa6d880b` (`p116-5090-2`).
