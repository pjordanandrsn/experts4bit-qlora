# `p230-5090-r1`: grouped-nf4-gemm's PDL, as shipped, on the default NF4 server (KEEP)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.84, a 500 W power limit, recorded and not floored) on an AMD EPYC 7402P host
(48 CPUs, 125 GiB RAM). It ran 2026-10-23 from 13:13Z to 13:31Z through the guarded launcher. Cost: $0.472.
- **Code:** e4b `b1ba9408` (P230's registration merge, e4b#2042); grouped-nf4-gemm v0.50.0 `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0 (the image's), transformers 5.17.0, bitsandbytes 0.50.2.
- **Id:** the run id is the retry label `r1`. The box before it, `p230-5090-1`, was refused at create ($0, its offer
  vanished), and under bench/common rule 3 that uses an id.

**Before the arms:**
- The refusals passed, and so did the pre-install fit: 7,481 s left against the 6,512 s needed.
- The self-tests passed: `p230_reduce.py` 38 cases, `p230_box.py` 10, `p109_box.py` 5.
- The premise passed on the card: `test_decode_graph_buckets.py` 7 passed; grouped-nf4-gemm `test_pdl.py` 26 passed and
  K33's two PDL tests 2 passed, from a clone at `41ef086e`.
- Qwen3-30B-A3B @ad44e777 was fetched and its NF4 arena baked on the box (`work/bake.json`).

**The server:** the default `serve_paged` graph server, every lever unset, resolved to max_seqs 32, buckets
1, 2, 4, 8, 16 and 32, all-vram, every bucket captured as a graph.

**The arms** (OFF1 DEF1 DEF2 OFF2, each a fresh process). OFF is `GNF4_PDL=0`; DEF is the shipped default (on, capped
at 8 rows).

| arm | W1 tok/s | W1 ms/step | W16 tok/s | W16 ms/step | build launches with PDL | load s |
|---|---:|---:|---:|---:|---|---:|
| OFF1 | 269.72 | 3.7075 | 1449.75 | 11.0364 | 0 of 8,759 | 58 |
| DEF1 | 271.24 | 3.6867 | 1445.86 | 11.0661 | 4,513 of 8,759 | 42 |
| DEF2 | 271.19 | 3.6874 | 1445.44 | 11.0693 | 4,513 of 8,759 | 39 |
| OFF2 | 269.77 | 3.7068 | 1446.04 | 11.0647 | 0 of 8,759 | 39 |

**Verdict: KEEP** (`verdict.json`).
- Tokens were identical on every row.
- The W1 DEF/OFF pairs read **1.00564** and **1.00526**, so g1 = 1.00526 ≥ 1.005.
- The W16 control pairs read 0.99732 and 0.99959; the self-pairs read 0.99744 to 1.00019. All are inside [0.995, 1.005].

**Per bucket** (DEF; each build launch tagged with the bucket being captured). Bucket 1 had 1,152 of 1,155 switched
launches with PDL; buckets 16 and 32 had none of 867 each; buckets 2, 4 and 8 had 723 of 867 each. OFF had none in any
bucket.

**Files:**
- **`SHA256SUMS`** covers every file here.
- **`STORE-ONLY.sha256`** lists the files kept only in the private receipt store: the launcher's receipt, guard record,
  instance record and teardown proof (they name the provider's instance), and six box logs whose progress bars carry
  carriage returns (the four arm logs, `fetch.log`, `prompts.log`).
- **Not copied:** the staged scripts, each byte-identical to the file `staged-p230.sha256` pins at `b1ba9408`; the
  nonce markers and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
