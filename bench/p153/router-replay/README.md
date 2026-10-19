# P153's W16 cost: the router replay ($0; a bound on the token-divergence term)

P153's read (`p153-5090-1`) left an observed, unattributed cost at 16 requests: S1 (`E4B_INT8_HEAD_BW=1`) ran below S0
in every W16 pair, 0.9918-0.9970. Review named two candidates: the resident int8 copy, and the different tokens the
rounded head decodes at 16 rows. This replay bounds the second at $0.

**Verdict: REJECTED**, under bands stated before any data. Token divergence does not carry the W16 cost. The resident
int8 copy remains the named candidate.

## What was read

- **The idea:** the lm_head is downstream of every router, so S0's and S1's recorded W16 streams go through the SAME model,
  teacher-forced, and only the tokens differ.
  - Each row is the 512-token P109 prompt plus that arm's recorded 160-token W16 continuation, from P153's public arm
    records (`arm_S0a.json` 549b0592, `arm_S1a.json` 9f225221, `prompts.json` d0e7c022).
- **The model:** Qwen/Qwen3-30B-A3B @ad44e777 in bf16, streamed layer by layer from the lab's read-only model store.
  The config, the index and all 16 shards were checked against the store's sha256 manifest. Nothing was copied or
  downloaded, and one decoder layer was resident at a time.
- **The card:** one RTX A2000 (sm_86) under an exclusive claim. Counts only, no timings. Peak 3,212 MiB.
- **The metric, stated before the run:** per decode step that the W16 slope counts (inputs `gen[31..158]`, the steps
  that produce tokens 33-160), each layer's DISTINCT experts across the 16 rows' top-8 picks. Then r = S1/S0, summed over
  48 layers x 128 steps.
- **The bands, stated before the run:** the predicted cost is about f x (r - 1), where f is the expert-read share of a
  bucket-16 step (at most 1). The observed cost is 0.5-0.8 %.
  - r < 1.005: cannot explain it even at f = 1.
  - 1.005-1.010: only if f >= 0.5.
  - r >= 1.010: sufficient at f around 0.5.
  - r <= 1: rejected.

## The read

| | S0 | S1 | S1/S0 |
|---|---|---|---|
| distinct experts (48 layers x 128 steps) | 350,868 | 350,529 | **0.99903** |
| mean distinct per layer-step (of 128) | 57.107 | 57.052 | |

- **Per step:** median 0.9998 (5-95 %: 0.937-1.069). S1 has more on 64 steps and fewer on 64.
- **Per layer:** 0.987-1.013 (22 layers above 1, 24 below).
- **Fidelity:** the teacher-forced bf16 argmax matched the recorded next token at 0.941 (S0) and 0.936 (S1). The
  pre-stated flag was below 0.9.
- **The bound's limits:** the experts here are bf16, teacher-forced on the recorded streams. It is a BOUND on the
  token-divergence term, not the served NF4 picks.
- **Re-derived:** the maintainer recounted from `rr.json.pt` with independent code; every number matched.

## A robustness check suggested at review (outside the pre-statement)

A grouped small-M GEMM's time also depends on how each step's 128 row assignments split across experts. From the same
picks (`rr_spread.py`, `rr_spread.json`):

| metric | S1/S0 |
|---|---|
| sum of max rows per expert | 0.99735 |
| experts with M > 1 | 0.99458 |
| experts with M >= 4 | 1.00460 |
| experts with M >= 8 | 0.99596 |
| sum of M^2 | 1.00254 |

Every spread metric is within +/-0.5 %, with mixed signs. None can carry 0.5-0.8 % even if its path were the whole
step. The pre-stated metric decides the verdict; these show it was not a lucky choice of metric.

## Files

- **`rr_replay.py`:** the replay as run.
- **`rr.json`:** its summary: the model, the shards' sha256s, the counts and the fidelity.
- **`rr.log`:** its progress log.
- **`rr.json.pt`:** the router picks, a `[48, 32, 128, 8]` int16 tensor (rows 0-15 S0, 16-31 S1) plus the agreement and
  the counted positions. Load it with `torch.load(path, weights_only=True)`.
- **`rr_spread.py`, `rr_spread.json`:** the robustness check.
- **`SHA256SUMS`:** covers every file here.
