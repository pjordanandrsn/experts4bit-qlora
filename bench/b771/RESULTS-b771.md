# B771 — results: the fused fp8 append's non-IEEE quotient is fixed at the byte level, and it is **not** the whole cause (**REFUTED**); the rest is a bucket-1 append bug

Read 2026-09-29 from `b771-5090-1`. Registration: `bench/b771/PREREG-b771.md` (#775, `b67d471`). Issue: #771.
Verdict by `b771_reduce.py` from the two byte reports and four decode receipts: **REFUTED**. B1, C and D1 held; D2 did
not.

## Run and cost

- `b771-5090-1`: one RTX 5090 (sm_120, driver 580.173.02, 575 W), Vast verified/secure, AMD Ryzen Threadripper PRO 3955WX.
- **Cost $0.1515**, a 1,158 s lifetime against a 1.0 h guard. Lane rc 0; teardown proven (`vast-destroy` HTTP 200,
  instance absent afterwards).
- The GPU showed **7,979 MiB in use and 100% utilization before the lane touched it**, as the driver's heartbeats
  record. This is a correctness lane, so the contention does not bias the verdict. Nothing here is a timing claim, and
  the per-arm tok/s in `summary.txt` are not quoted.

## The byte stage (B1: held)

| cut | values | payload bytes differing from `quantize_kv_fp8` | scales differing |
|---|---:|---:|---:|
| OLD `fb15cf5` (v0.33.5, the append as shipped) | 536,870,912 | **21** (3.9e-8) | 0 |
| NEW `530f93b` (grouped-nf4-gemm#413) | 536,870,912 | **0** | 0 |

The hardware e4m3 cast confirms the A2000 probe, which used torch's cast as a stand-in: ~6e-8 there, 3.9e-8 here. The
shipped fused append wrote a byte different from the reference about once in 25 million values. With the quotient
IEEE-rounded it writes the reference's bytes exactly.

## The decode stage (C and D1 held; D2 did not)

NF4 Qwen3-30B-A3B, P80's prompts and trace (16, 8, 4, 2, 1 active rows for 32 decode steps each), every arm with the
device grouping.

| pair | rows that differ | first divergence |
|---|---:|---|
| A_old vs A_new (**C**, the control) | 0 / 16 | none: the swap did not touch the eager path |
| P_old vs A_old (**D1**) | 1 / 16 | row 0, token 130 |
| P_new vs A_new (**D2**) | **1 / 16** | **row 0, token 129** |
| P_old vs P_new | 1 / 16 | row 0, token 129 |

- **D2 failed, so REFUTED.** With the fixed append, the bucket step still leaves the eager step. The fused append is
  not the whole cause.
- **Where it leaves, exactly.** Only row 0 differs, the one row that survives into the one-row phase. The one-row phase
  starts at decode step 128, which produces token 129. With the append fixed, the bucket step equals the eager step in
  every row through the 16-, 8-, 4- and 2-row phases, and diverges at the **first** one-row step and nowhere else.
- **Under the old append**, row 0 happened to agree one token longer (130), and no other row differed on this trace.
  On P81's int4 stack, row 1 diverged at token 67, in the four-row phase. That fits the append's byte flips, but it is
  inferred, not measured.

## What the one-row phase exposed (found from this read; fixed separately)

- **The mechanism.** `enable_decode_graphs` calls `graph_mode_init(seq=scratch[0])`. The paged-attention shim sent
  every single-row decode to `append_graph_t1`, which writes to that init-time slot, a **scratch** slot.
  - At bucket 1, whenever exactly one row is active, each new token's K/V went to the scratch slot, and the row's own
    length stopped advancing.
  - Attention reads the row's slot through the bucket selector, so it ran without the tokens generated since the
    one-row phase began.
  - A captured bucket-1 graph bakes the scratch slot in as a Python int, so B (replay) and P (padded eager) share the
    bug. That is why B511's and P80/P81's replay ≡ padded-eager gates held while graph ≠ eager was recorded as "not
    decisive".
- **The fix** (its own PR): a bound bucket takes the batch append, which addresses the bound slot on device, whatever
  its size. It comes with a routing test through the real shim on any machine, and a GPU test that checks each stepped
  row's device length against the host count after every bucketed step.
- **Not yet shown.** That with both fixes the bucket step equals the eager step in every row. That is the next
  lane's question.

## Deviation from the registration

PREREG-b771 said each byte stage runs "before any model loads". The runner ran the OLD byte stage after the fetch and
the bake, as a separate process after the bake had exited and before the first arm; the NEW one ran after the swap as
registered. No model was resident in either byte stage's process, so the measurement is unaffected. The sentence was
wrong about the order.

## Receipts

- **In this repo:** [`receipts/b771-5090-1/`](receipts/b771-5090-1/): `bytes_{old,new}.json`,
  `b771_{A_old,P_old,P_new,A_new}.json`, `verdict.json`, `summary.txt`, `versions.txt` and `forensics.txt`,
  byte-identical to the store's.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-09-29/b771-5090-1/` (commit `41899fe`), with the
  launcher's `receipt.json` and `teardown-proof.json` and the run's `logs/`.
