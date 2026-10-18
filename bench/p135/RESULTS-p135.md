# P135 — results: **FASTER**. Mixtral's ON_auto folds decode one request in 0.93–0.94 of the step on one RTX 5090 (2026-10-18)

Registration: `bench/p135/PREREG-p135.md` (#1732, merged `7a3e048f`) with Amendment 1 (#1737, `5be4c4d9`). Track B1's
read R-b (`bench/fastpath-portability/README.md` section 6). It is the speed read FAM Amendment 5 requires before
Mixtral's quality-passed folds (`e4b.serve.fam.fused-stack-t1.mixtral.5090.2026-10-09`) can default on.

Code under test:
- e4b 0.55.0 at `5be4c4d9`;
- grouped-nf4-gemm 0.45.0 at `724ccc45`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `mistralai/Mixtral-8x7B-Instruct-v0.1` at `eba92302`: the default `serve_paged` build, eager, 2 slots of 768 tokens,
  the four fusion knobs at `auto` (FAM's ON_auto), NF4 arena baked on the box.

**Verdict by `p135_reduce.py`: FASTER.** No VOID condition fired:
- the model, the revision, the e4b commit and the knobs are the registered ones; neither prefill knob is set;
- the census is `0 / 65 / [32, 32] / 32`; every setting of every block swapped all **161** fold-patched forwards
  (glue 65, round 2 64, router epilogue 32); the router epilogue's 32 routers are all on the fp32 upstream path;
- both blocks are present, in order (`a` OFF first, `b` ON first), OFF on slot 0 and ON on slot 1, with 5 warm, 256
  timed and 32 more steps;
- each runner captured bucket 1 and replayed it exactly 293 times, with no eager step;
- each setting emitted the same 293 tokens in both blocks.

## The reading (`p135-5090-2`)

**Host:** one RTX 5090 (sm_120, driver 580.95.05) on an AMD EPYC 7742 (128 CPUs, 251 GiB RAM), Vast instance 56547516.
**Cost:** $1.159. Teardown proven at 12:20:22Z.

**Timeline:** the lane started 11:59:51Z; the premise passed (35, none skipped) at 12:02:15Z; the fetch ran until
12:17:00Z and the bake until 12:18:08Z; the speed process ended and `TP_DONE` was seen at 12:20:17Z.

| block | order | OFF median (ms) | ON median (ms) | ON / OFF | saving (ms) | per-pair ratio median |
|---|---|---:|---:|---:|---:|---:|
| a | OFF, ON | 20.2442 | 19.0157 | **0.9393** | 1.229 | 0.9391 |
| b | ON, OFF | 20.5146 | 19.0305 | **0.9277** | 1.484 | 0.9293 |

- **The rule:** both ratios are at or under 0.98, and the blocks disagree by **1.26 %**, inside the 1.5 % NOISY line:
  FASTER.
- **Reported, never gated:**
  - ON and OFF tokens agree on 283 of 293 in each block, first parting at step 38 (the folds change arithmetic, which
    FAM's quality read licensed);
  - SM clock 2910–2955 MHz;
  - peak memory 27,837 / 27,853 MiB allocated, 27,906 / 27,952 MiB reserved.

## Predictions

| # | prediction | outcome |
|---|---|---|
| Y1 | integrity holds: no VOID (about 85 %) | **HIT** |
| Y2 | OFF's median within [17.0, 22.8] ms | **HIT** (20.24, 20.51) |
| Y3 | each ratio within [0.91, 0.975], a saving of 0.5–1.8 ms | **HIT** (0.9393, 0.9277; 1.23, 1.48 ms) |
| Y4 | FASTER about 70 % | **HIT** |
| Y5 | ON's tokens agree with OFF's for at least the first 20 decode steps | **HIT** (step 38) |
| Y6 | peak memory ≤ 31.0 GiB | **HIT** (27.3 GiB reserved) |

## Attempts

- `p135-5090-1` ($0.038): HARNESS_ERROR at the premise, before any fetch. A staged GPU test predated grouped-nf4-gemm's
  `rope_norm_qk` route and failed on the card; nothing was measured. Amendment 1 staged P136's route-exact copy. Strike 1
  (`receipts/p135-5090-1/`).
- `p135-5090-2` ($1.159): the reading, P135's last box-started attempt (`receipts/p135-5090-2/`).

Lane total: **$1.197**.

## Consequence

As registered, FASTER together with FAM's ON_auto quality PASS licenses the allow-list pull request that turns
Mixtral's folds on by default. The maintainer decided it: `serve_paged.FUSION_DEFAULT_FAMILIES` gains `mixtral`. Only
the three decode folds engage there (no fused q/k/v), and the prefill glue knobs are untouched.

## What this cannot say

Batches above one row, other cards, prefill, the int4 route, and Mixtral's other checkpoints.

## Evidence

- `receipts/p135-5090-2/`: `speed.json` (the box's record), `verdict.json` (the reducer's), `summary.txt`, the logs,
  `teardown-proof.json`; `SHA256SUMS` covers each directory.
- The reviewers and the maintainer each re-derived the verdict from the receipt store (`1342d2db`) byte for byte. The
  launcher's receipts and ledger rows are in that store.
- Claim: `e4b.serve.p135.fold-speed.mixtral.5090.2026-10-18` (`docs/claims.json`).
