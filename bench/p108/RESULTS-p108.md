# P108 — results: **AT_PARITY**. On Gemma-4-26B-A4B, with its 1,024-token sliding window binding, e4b's paged path is indistinguishable from transformers' own arithmetically neutral perturbations of the same model: its NLL sits between the floor draws', and a halved decode scale is caught at +2.26 nats

Registration: `bench/p108/PREREG-p108.md` (#974, `dfaf88e`; Amendments 1–3: #978 `7c59d94`, #980 `2907220`, #984
`e0ae2f3`). Issue: #359. Code under test: e4b at `e0ae2f3` (0.41.1 plus the box's memory release), grouped-nf4-gemm at
`34da93d`, transformers 5.17.0, `google/gemma-4-26B-A4B-it` at `4d7ae49`.

**Verdict by `p108_reduce.py`: `AT_PARITY`.**
- **The bias bar:** the paged bias, −0.196, is ≤ the floor's largest |bias| + 0.05 (0.258 + 0.05 = 0.308).
- **The spread bar:** the paged spread, 0.326, is ≤ 2 × the floor's largest spread (2 × 0.349 = 0.698).
- **No VOID condition fired:**
  - the commit is `4d7ae49`;
  - all 32 windows are in every arm;
  - engagement is exact: 30,600 of 30,600 decode attention calls, 25,500 of them at window 1,024 and 5,100 at 0;
  - the scale mutant fails the bar.

## The reading (`p108-5090-3`)

One RTX 5090 (sm_120, driver 610.57.04) on an AMD Ryzen 9 9950X3D, Vast machine 151972 (pre-flight 112.5 MB/s).
$0.5181. The lane started at 18:10:04Z. The box was destroyed and proven absent at 2026-10-03T18:57:42Z.
- **Premise:** 2 passed. The tiny Gemma-4's binding window stays within 2× its control, stand-in and real fp8 kernel.
- **The box:** load 10 s (the checkpoint was cached by the fetch). The groups finished at 645, 1,180, 1,722 and
  2,261 s (the box log's own lines), about 38 minutes in all.
- **Allocated memory after every group:** 16.877 GiB. Peak 20.13 GB.
- **The reference** repeated bit for bit (first group).

| arm | mean NLL (32 windows) | bias vs R | spread | se | mean KL vs R | argmax agree |
|---|---:|---:|---:|---:|---:|---:|
| R, the reference (cached, batch 1) | 5.5564 | — | — | — | — | — |
| oneshot (floor) | 5.4145 | −0.1420 | 0.2761 | 0.0613 | 0.428 | 0.790 |
| chunk (floor) | 5.2984 | −0.2580 | 0.3492 | 0.0729 | 0.483 | 0.791 |
| batch (floor) | 5.4080 | −0.1485 | 0.3137 | 0.0747 | 0.457 | 0.795 |
| **paged** | **5.3604** | **−0.1960** | **0.3262** | 0.0699 | 0.528 | 0.773 |
| mutant_scale | 7.8137 | +2.2572 | 2.2572 | 0.1272 | 5.268 | 0.203 |
| mutant_window | 4.6034 | −0.9530 | 0.9530 | 0.0756 | 1.661 | 0.584 |

**Reading:**
- **The paged path sits inside the floor.** Its mean NLL (5.360) lies between the floor draws' (5.298–5.415). Against
  each floor draw window by window, its mean difference is −0.054 (oneshot), +0.062 (chunk) and −0.048 (batch). It is
  as far from any of them as they are from each other: oneshot − chunk = +0.116.
- **The reference is the outlier, not the paged path.** Every other arm reads lower NLL than R by 0.14–0.26 nats, and
  on 19–23 of the 32 windows. R is transformers' standard order: one 1,280-token prefill, then single-token cached
  decode. The chunked floor, which differs from R only in prefilling in 256-token pieces, reads 0.258 lower.
  - This is Gemma-4's batch-shape chaos (#359), now seen in a mean over 32 windows rather than one. This lane does not
    explain why one particular arithmetic order reads higher.
  - The registered rule judges the subject against the floor's spread, so R's position moves the bar for every arm
    alike.
- **The window mutant is visible and reads LOWER NLL.** With decode attention ignoring the 1,024-token window on the
  sliding layers, so that tokens 1,280–1,535 attend over their whole past, NLL falls by 0.95 nats. The instruction-tuned
  checkpoint predicts raw wikitext much better with more context than its sliding layers were built to see. This shows
  the window binds and is enforced; a dropped window would not be mistaken for parity.
- **The NLLs are high** (about 5.4 nats, against roughly 1.6 for Qwen3 on the same corpus). That is consistent with
  P27's single windows (3.3–6.5). This is an instruction-tuned checkpoint scored on raw wikitext without its chat
  template.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| AT_PARITY | **yes** |
| each floor draw's \|bias\| ≤ 0.05 | **no**: 0.142, 0.258, 0.149. The reference itself sits apart. |
| floor spread 0.05–0.25 | **no**: 0.276–0.349 |
| paged bias between 0 and +0.05 | **no: −0.196** (paged reads lower than R, as every floor draw does) |
| paged spread ≤ 1.5 × the floor spread | **yes** (0.326 against 0.349) |
| the reference repeats bit for bit | **yes** |
| mutant_scale bias above +0.3 | **yes** (+2.257) |
| mutant_window visible, bias above +0.05 | **visible, but the sign is wrong**: −0.953 |
| peak GPU memory ≤ 28 GB | **yes** (20.13 GB) |
| the box ≤ 60 min | **met on this host** (2,261 s). **Not on runs 1–2's EPYC 7663 host**, where a group took 31–40 min (Amendments 1–2). |

**The predictions placed the reference at the centre of the floor, and it is not.** Gemma-4's chaos is larger than
P27's three windows suggested (spreads of 0.28–0.35 nats against 0.08–0.27), and transformers' own standard order is the
arm that stands apart.

## The registered consequence (AT_PARITY)

- `docs/SERVING-PARITY.md`'s Gemma-4 row moves from "no reference at this resolution" to **at parity with
  transformers' own perturbations**, with the bias, spread and floor.
- Register row: `e4b.parity.gemma4.p108.paged-vs-floor.5090.2026-10-03`.
- #359 closes.

## What it took

| run | status | machine | cost | note |
|---|---|---:|---:|---|
| `p108-prove-1` | OK, PROVED | 142284 | $0.0336 | premise 2 passed on sm_120 |
| `p108-5090-1` | HARNESS_ERROR | 142284 | $0.8047 | the box's 75-min alarm, after 2 of 4 groups (Amendment 1, then 2) |
| `p108-5090-2` | HARNESS_ERROR | 142284 | $1.0068 | CUDA out-of-memory in group 3: a reference-cycle leak (Amendment 3) |
| `p108-5090-3` | **OK, AT_PARITY** | 151972 | $0.5181 | the reading |

- **The lane: $2.3632**, within its amended $4.30 ceiling.
- **Two fixes shipped in 0.41.1 came from preparing it:** #964, `serve_paged` on Gemma-4, and #966, the unbound
  fallback's masks.
- **Two lessons:** budget a box from a measured per-step time; release CUDA memory explicitly between passes.

Receipts (`receipts/p108-5090-3/`), with `SHA256SUMS`:
- `box.json` (every arm's per-window NLL, KL and agreement), `verdict.json`, `summary.txt`, `forensics.txt`,
  `versions.txt`;
- the install, premise, fetch and box logs;
- the teardown proof.

The launcher's receipts and ledger rows are in the receipt store (adertha-receipts `5f86b74`, `bdc1b6d`, `7aad017`,
`5aebcba`).
